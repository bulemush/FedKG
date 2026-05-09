Place the original `KG-Adapter-main/model/GNN.py` file in this directory.

Expected path inside the uploaded `FedBiOT_experiment` repository:

`federatedscope/llm/kg_adapter/paper_gnn/GNN.py`

The training configs already point `llm.kg_adapter.paper_gnn_path` to this
location so the paper `RGAT/SRGAT/SAGPooling` backend can be loaded without
requiring a sibling `KG-Adapter-main` repository on the server.
