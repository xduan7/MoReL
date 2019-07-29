export COMET_OPTIMIZER_ID=$(comet optimize task/graph_drug_response.config)
CUDA_VISIBLE_DEVICES=4 comet optimize graph_drug_response.py graph_drug_response.config &
CUDA_VISIBLE_DEVICES=5 comet optimize graph_drug_response.py graph_drug_response.config &
CUDA_VISIBLE_DEVICES=6 comet optimize graph_drug_response.py graph_drug_response.config &
CUDA_VISIBLE_DEVICES=7 comet optimize graph_drug_response.py graph_drug_response.config &