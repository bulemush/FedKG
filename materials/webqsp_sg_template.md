# WebQSP `sg` Template

This template shows how one WebQSP sample can be converted into the `sg`
structure consumed by the KG-Adapter path in `FedBiOT_experiment`.

Source example:
- `QuestionId`: `WebQTrn-0`
- `RawQuestion`: `what is the name of justin bieber brother?`
- topic entity: `Justin Bieber`
- answer: `Jaxon Bieber`

Readable graph:

```text
Justin Bieber
  --people.person.sibling_s-->
Sibling relation node
  --people.sibling_relationship.sibling-->
Jaxon Bieber
  --people.person.gender-->
Male
```

How the fields map:

- `sg_readable`:
  Human-readable graph extracted from `Parses`, `InferentialChain`,
  `Constraints`, and `Answers`.
- `sg.node_ids`:
  Integer ids from your entity vocabulary. The values in the template are
  placeholders and must be replaced by your own vocabulary ids.
- `sg.edge_type`:
  Integer ids from your relation vocabulary. These are also placeholders.
- `sg.edge_index`:
  Graph connectivity in PyG-style format: `[src_list, dst_list]`.
- `sg.node_type`:
  Optional node-role tags. A simple convention can be:
  `1=topic`, `2=intermediate`, `3=answer`, `4=constraint`.
- `sg.nid2swid`:
  For each node, which token positions in the question text map to that node.
  In this example, node `0` (`Justin Bieber`) is aligned to token positions
  `7` and `8`. Positions are placeholders because they depend on your
  tokenizer.
- `sg.eid2swid`:
  Optional edge-to-subword mapping. If you do not align relation text to the
  question, leave these rows as `[0]` or empty rows in your preprocessing.
- `sg.token_entity_ids`:
  For each token position, which node indices it aligns to. `-1` means no
  aligned entity. This is an alternative view of the same alignment
  information in `align_mask`.
- `sg.align_mask`:
  Shape `[seq_len, num_nodes]`. Row `t`, column `n` is `1` when token `t`
  aligns to node `n`.

Minimum fields required for the current code path:

```json
{
  "sg": {
    "node_ids": [1001, 1002],
    "edge_index": [[0], [1]],
    "edge_type": [2001]
  }
}
```

Recommended fields for the full hybrid-initialization path:

```json
{
  "sg": {
    "node_ids": [1001, 1002, 1003],
    "node_type": [1, 2, 3],
    "edge_index": [[0, 1], [1, 2]],
    "edge_type": [2001, 2002],
    "nid2swid": [[7, 8], [0], [0]],
    "token_entity_ids": [[-1], [-1], [0], [0]],
    "align_mask": [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]]
  }
}
```

Important:

- `node_ids` / `edge_type` must be integer ids, not Freebase strings.
- The readable strings such as `m.06w2sn5` and
  `people.person.sibling_s` should be converted into integer vocab ids during
  preprocessing.
- The template is designed to help you build a preprocessing script for
  `WebQSP.train.json` and `WebQSP.test.json`.
