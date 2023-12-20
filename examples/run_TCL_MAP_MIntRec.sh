#!/usr/bin/bash

for seed in 0 1 2 3 4 5 6 7 8 9
do
    for method in 'tcl_map' 
    do
        for text_backbone in 'bert-base-uncased'
        do
            python run.py \
            --dataset 'MIntRec' \
            --logger_name ${method} \
            --method ${method} \
            --tune \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '2' \
            --text_backbone $text_backbone \
            --config_file_name TCL_MAP_MIntRec \
            --results_file_name "results_MIntRec.csv"
        done
    done
done