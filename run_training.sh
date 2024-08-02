#!/bin/bash

#10
#llama3_model_path="/home/somov/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
#144
llama3_model_path="/home/somov/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa"

project_path="/home/somov/text2sql_llama_3"
data_dir="data"
experiments_folder="experiments"
dataset_name="pauq"
split_name="tsl_ssp"
CUDA_DEVICE_NUMBER='0'
seed=123
run_explain_name="sft_1_epoch"


if [ "$dataset_name" = "pauq" ];
then
  path2train="$project_path/$data_dir/$dataset_name/${split_name}_train.json"
  path2test="$project_path/$data_dir/$dataset_name/${split_name}_test.json"
  saving_path="$project_path/$experiments_folder/${dataset_name}_${split_name}_s${seed}_${run_explain_name}"
  run_name="${dataset_name}_${split_name}_s${seed}_${run_explain_name}"
else
  path2train="$project_path/$data_dir/$dataset_name/train"
  path2test="$project_path/$data_dir/$dataset_name/test"
  saving_path="$project_path/$experiments_folder/${dataset_name}_s${seed}_${run_explain_name}"
  run_name="${dataset_name}_s${seed}_${run_explain_name}"
fi
tables_path="$project_path/$data_dir/$dataset_name/tables.json"
log_dir="$saving_path/training_logs"


#lora
#train_batch_size=4
#eval_batch_size=8
#gradient_accumulation_steps=24
#eval_accumulation_steps=4
#lr="1.5e-4"

#sft
train_batch_size=1
eval_batch_size=1
gradient_accumulation_steps=96
eval_accumulation_steps=8
lr="1e-5"

input_seq_length=1024
output_seq_length=256

epochs_number=1

tmux new-session -d -s $run_name

tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python3 train_llm.py \
    --model_name $llama3_model_path \
    --sql_dataset_name $dataset_name \
    --path_to_training_file $path2train \
    --path_to_testing_file $path2test \
    --tables_info_path $tables_path \
    --learning_rate $lr \
    --seed $seed \
    --per_device_train_batch_size $train_batch_size \
    --per_device_eval_batch_size $eval_accumulation_steps \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_seq_length $input_seq_length \
    --report_to 'tensorboard' \
    --overwrite_output_dir \
    --output_dir $saving_path \
    --logging_dir $log_dir \
    --num_train_epochs $epochs_number \
    --run_name $run_name" ENTER

trained_model_path="$saving_path/final_checkpoints"
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python3 infer_llm.py \
    --model_name $trained_model_path \
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
