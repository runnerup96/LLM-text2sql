#!/bin/bash

# (144) Path to finetuned model in experiements folder
llama3_model_path="/home/somov/text2sql_llama_3/experiments/pauq_pauq_xsp_s1_lora_v4/final_checkpoints"
project_path="/home/somov/text2sql_llama_3"
data_dir="data"
experiments_folder="experiments"
dataset_name="pauq"
split_name="pauq_xsp"
CUDA_DEVICE_NUMBER='1'
seed=1
run_explain_name="lora_v4_inference"


if [ "$dataset_name" = "pauq" ];
then
  path2test="$project_path/$data_dir/$dataset_name/${split_name}_train.json"
  saving_path="$project_path/$experiments_folder/${dataset_name}_${split_name}_s${seed}_${run_explain_name}"
  run_name="infer_${dataset_name}_${split_name}_s${seed}_${run_explain_name}"
else
  path2test="$project_path/$data_dir/$dataset_name/test"
  saving_path="$project_path/$experiments_folder/${dataset_name}_s${seed}_${run_explain_name}"
  run_name="infer_${dataset_name}_s${seed}_${run_explain_name}"
fi
tables_path="$project_path/$data_dir/$dataset_name/tables.json"
log_dir="$saving_path/training_logs"

input_seq_length=768
output_seq_length=256
eval_batch_size=12

tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python3 infer_llm.py \
    --model_name $llama3_model_path \
    --use_lora \
    --sql_dataset_name $dataset_name \
    --path_to_testing_file $path2test \
    --tables_info_path $tables_path \
    --seed $seed \
    --max_seq_length $input_seq_length \
    --max_new_tokens $output_seq_length \
    --per_device_eval_batch_size $eval_batch_size \
    --num_beams 1 \
    --output_dir $saving_path" ENTER

tmux a -t $run_name

# the result predictions will be in saving path