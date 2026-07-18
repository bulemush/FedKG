"""Launch isolated CWQ/WebQSP FedKSA component ablations.

The launcher intentionally reuses the paper configuration and changes only
one KG-adapter switch per ablation. Checkpoints and logs are redirected to a
dedicated ablation namespace so the paper's original runs are never reused or
overwritten.
"""

import argparse
import copy
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = Path(__file__).with_name(
    "cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml")

MODULE_FLAGS = (
    "use_hybrid_embedding",
    "use_initial_graph_token_injection",
    "use_gnn",
    "use_joint_reasoning",
)

VARIANTS = {
    "full": {
        "use_hybrid_embedding": True,
        "use_initial_graph_token_injection": True,
        "use_gnn": True,
        "use_joint_reasoning": True,
    },
    "no_hybrid_embedding": {
        "use_hybrid_embedding": False,
        "use_initial_graph_token_injection": True,
        "use_gnn": True,
        "use_joint_reasoning": True,
    },
    "no_initial_graph_token_injection": {
        "use_hybrid_embedding": True,
        "use_initial_graph_token_injection": False,
        "use_gnn": True,
        "use_joint_reasoning": True,
    },
    "no_gnn": {
        "use_hybrid_embedding": True,
        "use_initial_graph_token_injection": True,
        "use_gnn": False,
        "use_joint_reasoning": True,
    },
    "no_joint_reasoning": {
        "use_hybrid_embedding": True,
        "use_initial_graph_token_injection": True,
        "use_gnn": True,
        "use_joint_reasoning": False,
    },
}


def _nested_get(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_base_config(config):
    """Fail early if the paper configuration no longer matches the protocol."""
    errors = []
    if _nested_get(config, "data", "type") != "cwq@llm":
        errors.append("data.type must be 'cwq@llm'")
    if _nested_get(config, "data", "splitter") != "iid":
        errors.append("CWQ client data must use data.splitter='iid'")
    client_args = _nested_get(config, "data", "args") or []
    client_train = client_args[0].get("train_file", "") \
        if client_args and isinstance(client_args[0], dict) else ""
    if "CWQ/ComplexWebQuestions_train.json" not in client_train:
        errors.append("CWQ must remain the client training dataset")

    align_type = _nested_get(
        config, "llm", "offsite_tuning", "emu_align", "data", "type")
    if align_type != "webquestionssp@llm":
        errors.append("server alignment data.type must be 'webquestionssp@llm'")
    align_args = _nested_get(
        config, "llm", "offsite_tuning", "emu_align", "data", "args") or []
    align_train = align_args[0].get("train_file", "") \
        if align_args and isinstance(align_args[0], dict) else ""
    if "WebQSP/data/WebQSP.train.json" not in align_train:
        errors.append("WebQSP must remain the server alignment dataset")
    if _nested_get(config, "llm", "offsite_tuning", "emu_align", "use") \
            is not True:
        errors.append("server-side emulator alignment must stay enabled")
    if _nested_get(config, "llm", "kg_adapter", "use") is not True:
        errors.append("llm.kg_adapter.use must stay enabled")

    if errors:
        raise ValueError("Invalid ablation base config:\n- " +
                         "\n- ".join(errors))


def validate_variant(name, flags):
    """Ensure a component run is a true single-factor ablation."""
    if set(flags) != set(MODULE_FLAGS):
        raise ValueError(f"{name} does not define all module flags")
    disabled = [key for key, enabled in flags.items() if not enabled]
    expected = [] if name == "full" else [
        "use_" + name.removeprefix("no_")
    ]
    if disabled != expected:
        raise ValueError(
            f"{name} must disable exactly {expected}, got {disabled}")


def checkpoint_path(variant, run_tag):
    return (REPO_ROOT / "checkpoints" / "ablations" / "cwq_webqsp" /
            variant / f"{run_tag}.ckpt")


def resolved_config_path(variant, run_tag):
    return (REPO_ROOT / "exp" / "ablations" / "cwq_webqsp" / "configs" /
            run_tag / f"{variant}.yaml")


def build_resolved_config(base_config, variant, flags, run_tag, seed,
                          smoke_test=False):
    """Materialize one train/eval config so evaluation cannot re-enable modules."""
    config = copy.deepcopy(base_config)
    kg_config = config["llm"]["kg_adapter"]
    kg_config.update(flags)
    config["seed"] = seed
    config["federate"]["save_to"] = str(
        checkpoint_path(variant, run_tag))
    config["outdir"] = str(REPO_ROOT / "exp" / "ablations" /
                           "cwq_webqsp" / run_tag)
    config["expname"] = variant
    if smoke_test:
        config["federate"]["total_round_num"] = 1
        config["federate"]["save_freq"] = 1
        config["train"]["local_update_steps"] = 1
        align_train = config["llm"]["offsite_tuning"]["emu_align"]["train"]
        align_train["local_update_steps"] = 1
        align_train["initial_update_rounds"] = 1
        config["eval"]["freq"] = 1
    return config


def data_file_metadata(config):
    """Fingerprint the two fixed protocol inputs in the run manifest."""
    data_root = REPO_ROOT / config["data"]["root"]
    files = {
        "cwq_client_train": data_root / config["data"]["args"][0][
            "train_file"],
        "webqsp_server_alignment": data_root / config["llm"][
            "offsite_tuning"]["emu_align"]["data"]["args"][0]["train_file"],
    }
    metadata = {}
    for name, path in files.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest.hexdigest(),
        }
    return metadata


def existing_checkpoints(target):
    if not target.parent.exists():
        return []
    candidates = list(target.parent.glob(f"*_{target.name}"))
    if target.exists():
        candidates.append(target)
    return sorted(set(candidates))


def build_command(variant, flags, run_tag, seed):
    del flags, seed  # Values are persisted in the resolved variant config.
    return [
        sys.executable,
        str(REPO_ROOT / "federatedscope" / "main.py"),
        "--cfg",
        str(resolved_config_path(variant, run_tag)),
    ]


def final_checkpoint_path(variant, run_tag):
    target = checkpoint_path(variant, run_tag)
    return target.with_name("final_" + target.name)


def build_eval_command(variant, run_tag):
    eval_dir = (REPO_ROOT / "exp" / "ablations" / "cwq_webqsp" /
                run_tag / "evaluation" / variant)
    return [
        sys.executable,
        str(REPO_ROOT / "fedbiot_script" / "eval_kgqa_hit1.py"),
        "--cfg", str(resolved_config_path(variant, run_tag)),
        "--dataset", "cwq",
        "--split", "val",
        "--ckpt", str(final_checkpoint_path(variant, run_tag)),
        "--output", str(eval_dir / "cwq_val_predictions.jsonl"),
        "--summary", str(eval_dir / "cwq_val_hit1_summary.csv"),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "variants",
        nargs="+",
        choices=["all", *VARIANTS.keys()],
        help="Ablation variants to run; 'all' expands to full plus four drops.",
    )
    parser.add_argument("--run-tag", default="seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without starting training.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate CWQ validation Hit@1 after each completed variant.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one FL round and one local/alignment update for path checks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_tag):
        raise ValueError("--run-tag may contain only letters, digits, ._- ")

    with BASE_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_base_config(config)

    requested = list(VARIANTS) if "all" in args.variants else args.variants
    requested = list(dict.fromkeys(requested))
    commands = []
    for variant in requested:
        flags = VARIANTS[variant]
        validate_variant(variant, flags)
        target = checkpoint_path(variant, args.run_tag)
        existing = existing_checkpoints(target)
        if existing and not args.dry_run:
            rendered = "\n".join(str(path) for path in existing)
            raise FileExistsError(
                "Refusing to overwrite an ablation checkpoint. Choose a new "
                f"--run-tag. Existing files:\n{rendered}")
        train_command = build_command(variant, flags, args.run_tag, args.seed)
        eval_command = build_eval_command(variant, args.run_tag)
        commands.append((variant, flags, train_command, eval_command))

    for variant, flags, train_command, eval_command in commands:
        print(f"[{variant}] flags={json.dumps(flags, sort_keys=True)}")
        print(f"[{variant}:train] {shlex.join(train_command)}")
        print(f"[{variant}:eval] {shlex.join(eval_command)}")
    if args.dry_run:
        return

    manifest_dir = (REPO_ROOT / "exp" / "ablations" / "cwq_webqsp" /
                    "manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.run_tag}.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}; use a new --run-tag")
    config_dir = resolved_config_path("full", args.run_tag).parent
    if config_dir.exists():
        raise FileExistsError(
            f"Resolved config directory exists: {config_dir}; use a new "
            "--run-tag")
    config_dir.mkdir(parents=True)
    for variant, flags, _, _ in commands:
        resolved = build_resolved_config(config, variant, flags, args.run_tag,
                                         args.seed, args.smoke_test)
        with resolved_config_path(variant, args.run_tag).open(
                "x", encoding="utf-8") as stream:
            yaml.safe_dump(resolved,
                           stream,
                           sort_keys=False,
                           allow_unicode=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "client_dataset": "CWQ",
        "server_alignment_dataset": "WebQSP",
        "seed": args.seed,
        "run_tag": args.run_tag,
        "smoke_test": args.smoke_test,
        "client_num": config["federate"]["client_num"],
        "client_splitter": config["data"]["splitter"],
        "data_files": data_file_metadata(config),
        "variants": {
            variant: {
                "flags": flags,
                "checkpoint": str(
                    checkpoint_path(variant, args.run_tag).relative_to(
                        REPO_ROOT)),
                "resolved_config": str(
                    resolved_config_path(variant, args.run_tag).relative_to(
                        REPO_ROOT)),
                "train_command": train_command,
                "eval_command": eval_command,
            }
            for variant, flags, train_command, eval_command in commands
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    for variant, _, train_command, eval_command in commands:
        print(f"Starting {variant}", flush=True)
        subprocess.run(train_command, cwd=REPO_ROOT, check=True)
        if args.evaluate:
            subprocess.run(eval_command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
