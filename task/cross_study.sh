#!/usr/bin/env bash
python ./cross_study.py \
    --train_on CTRP \
    --test_on GDSC CCLE \
    --subsample_on cell \
    --lower_percentage 1.00 \
    --higher_percentage 1.00 \
    --state_dim 1024 \
    --cuda_device 0 \
    >> ../log/cross_study_cell.txt &
echo $!

# python ./cross_study.py \
#     --train_on CTRP \
#     --test_on GDSC CCLE \
#     --subsample_on drug \
#     --lower_percentage 0.05 \
#     --higher_percentage 1.00 \
#     --state_dim 1024 \
#     --cuda_device 1 \
#     >> ../log/cross_study_drug.txt &
# echo $!