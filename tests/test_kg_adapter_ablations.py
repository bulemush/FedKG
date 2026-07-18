import unittest

import torch
from torch import nn

from federatedscope.llm.kg_adapter.modules import (
    FSKGAdapterRuntime,
    KGHybridEmbedding,
    KGInjectedLayer,
)
from fedbiot_script.fedbiot.cwq.run_ablations import (
    BASE_CONFIG,
    MODULE_FLAGS,
    VARIANTS,
    build_command,
    build_eval_command,
    build_resolved_config,
    validate_base_config,
    validate_variant,
)

import yaml


class _DummySelfAttention:
    hidden_size = 4


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _DummySelfAttention()
        self.anchor = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states):
        return (hidden_states,)


class _FailModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("disabled module was executed")


class _JointRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, hidden_states, context_states, context_mask=None):
        self.called = True
        return hidden_states + 1, context_states


class _TripRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False

    def forward(self, node_states, edge_states, edge_index, node_mask=None,
                edge_mask=None):
        self.called = True
        return node_states, node_mask


class KGAdapterAblationTest(unittest.TestCase):
    def _embedding(self, **overrides):
        runtime = FSKGAdapterRuntime()
        cfg = {
            "entity_vocab_size": 16,
            "edge_vocab_size": 16,
            "entity_hidden_size": 4,
            "num_heads": 2,
            "dropout": 0.0,
        }
        cfg.update(overrides)
        return KGHybridEmbedding(nn.Embedding(32, 4), runtime, cfg, 4), runtime

    def test_original_defaults_keep_both_new_paths_enabled(self):
        module, _ = self._embedding()
        self.assertTrue(module.use_hybrid_embedding)
        self.assertTrue(module.use_initial_graph_token_injection)

    def test_no_hybrid_embedding_uses_id_states_only(self):
        module, _ = self._embedding(use_hybrid_embedding=False)
        node_ids = torch.tensor([[1, 2]])
        kg_inputs = {
            "entity_ids": node_ids,
            "nid2swid": torch.tensor([[[3, 4], [5, 6]]]),
            "edge_ids": torch.tensor([[1]]),
            "eid2swid": torch.tensor([[[7, 8]]]),
        }
        states, _ = module._build_node_states(kg_inputs, torch.device("cpu"),
                                              torch.float32)
        expected = module.entity_embedding(node_ids)
        torch.testing.assert_close(states, expected)
        edge_states, _ = module._build_edge_states(
            kg_inputs, torch.device("cpu"), torch.float32, batch_size=1)
        torch.testing.assert_close(
            edge_states, module.edge_embedding(kg_inputs["edge_ids"]))

    def test_no_initial_injection_keeps_runtime_graph_for_later_layers(self):
        module, runtime = self._embedding(
            use_initial_graph_token_injection=False)
        input_ids = torch.tensor([[3, 4, 5]])
        kg_inputs = {
            "entity_ids": torch.tensor([[1, 2]]),
            "edge_ids": torch.tensor([[1]]),
            "edge_index": torch.tensor([[[0], [1]]]),
            "token_entity_ids": torch.tensor([[[0], [-1], [1]]]),
        }
        runtime.activate(kg_inputs=kg_inputs)
        expected = module.base_embedding(input_ids)
        output = module(input_ids)
        torch.testing.assert_close(output, expected)
        self.assertIsNotNone(runtime.get()["node_states"])
        self.assertIsNotNone(runtime.get()["edge_states"])
        self.assertIsNotNone(runtime.get()["edge_index"])

    def test_no_gnn_skips_graph_reasoner_but_keeps_joint_reasoning(self):
        runtime = FSKGAdapterRuntime()
        layer = KGInjectedLayer(
            _DummyLayer(), runtime, {
                "gnn_backend": "lite",
                "entity_hidden_size": 4,
                "num_heads": 2,
                "use_gnn": False,
                "use_trips": True,
                "use_joint_reasoning": True,
            })
        layer.graph_reasoner = _FailModule()
        trip_recorder = _TripRecorder()
        layer.trip_encoder = trip_recorder
        recorder = _JointRecorder()
        layer.joint_reasoning = recorder
        runtime.activate(kg_inputs={"entity_ids": torch.tensor([[1]])})
        runtime.get()["node_states"] = torch.ones(1, 1, 4)
        runtime.get()["node_mask"] = torch.ones(1, 1, dtype=torch.bool)
        runtime.get()["edge_states"] = torch.ones(1, 1, 4)
        runtime.get()["edge_mask"] = torch.ones(1, 1, dtype=torch.bool)
        runtime.get()["edge_index"] = torch.zeros(1, 2, 1,
                                                   dtype=torch.long)
        hidden = torch.zeros(1, 2, 4)
        output = layer(hidden)[0]
        self.assertTrue(trip_recorder.called)
        self.assertTrue(recorder.called)
        torch.testing.assert_close(output, hidden + 1)

    def test_no_joint_reasoning_never_calls_joint_module(self):
        runtime = FSKGAdapterRuntime()
        layer = KGInjectedLayer(
            _DummyLayer(), runtime, {
                "gnn_backend": "lite",
                "entity_hidden_size": 4,
                "num_heads": 2,
                "use_gnn": False,
                "use_trips": False,
                "use_joint_reasoning": False,
            })
        layer.joint_reasoning = _FailModule()
        runtime.activate(kg_inputs={"entity_ids": torch.tensor([[1]])})
        runtime.get()["node_states"] = torch.ones(1, 1, 4)
        runtime.get()["node_mask"] = torch.ones(1, 1, dtype=torch.bool)
        hidden = torch.zeros(1, 2, 4)
        output = layer(hidden)[0]
        torch.testing.assert_close(output, hidden)

    def test_launcher_protocol_and_single_factor_variants(self):
        with BASE_CONFIG.open("r", encoding="utf-8") as stream:
            base_config = yaml.safe_load(stream)
        validate_base_config(base_config)
        checkpoint_args = set()
        for name, flags in VARIANTS.items():
            validate_variant(name, flags)
            resolved = build_resolved_config(base_config, name, flags,
                                              "unit-test", 7)
            self.assertEqual(resolved["seed"], 7)
            for flag in MODULE_FLAGS:
                self.assertEqual(resolved["llm"]["kg_adapter"][flag],
                                 flags[flag])
            command = build_command(name, flags, "unit-test", 7)
            eval_command = build_eval_command(name, "unit-test")
            self.assertEqual(command[command.index("--cfg") + 1],
                             eval_command[eval_command.index("--cfg") + 1])
            checkpoint_args.add(resolved["federate"]["save_to"])
        self.assertEqual(len(checkpoint_args), len(VARIANTS))

        smoke = build_resolved_config(base_config, "full", VARIANTS["full"],
                                      "smoke", 0, smoke_test=True)
        self.assertEqual(smoke["federate"]["total_round_num"], 1)
        self.assertEqual(smoke["train"]["local_update_steps"], 1)
        self.assertEqual(
            smoke["llm"]["offsite_tuning"]["emu_align"]["train"]
            ["local_update_steps"], 1)


if __name__ == "__main__":
    unittest.main()
