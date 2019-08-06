export COMET_OPTIMIZER_ID=$(comet optimize graph_drug_response.config)
CUDA_VISIBLE_DEVICES=2 python graph_drug_response.py &
CUDA_VISIBLE_DEVICES=3 python graph_drug_response.py &
CUDA_VISIBLE_DEVICES=4 python graph_drug_response.py &
CUDA_VISIBLE_DEVICES=5 python graph_drug_response.py &
CUDA_VISIBLE_DEVICES=6 python graph_drug_response.py &
CUDA_VISIBLE_DEVICES=7 python graph_drug_response.py &
