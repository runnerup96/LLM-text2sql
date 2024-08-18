#!/bin/bash
# запуск на кластере
# сначала актвиируем окружение - conda activate llm_tuning_env
# srun/sbatch -A proj_1428 train_on_cluster.sh

# run config
#SBATCH --job-name=sft_lora_tuning       # Название задачи
#SBATCH --error=/home/llm_tuning/cluster_logs/sft_lora_tuning.err        # Файл для вывода ошибок
#SBATCH --output=/home//llm_tuning/cluster_logs/sft_lora_tuning.err       # Файл для вывода результатов
#SBATCH --time=48:00:00                      # Максимальное время выполнения
#SBATCH --nodes=1                           # Требуемое кол-во узлов
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=8                            # Требуемое кол-во GPU

llama3_model_path="/home/llama3/Meta-Llama-3-8B-Instruct/hf_converted"
project_path="/home/text2sql_llama_3"
data_dir="data"
experiments_folder="experiments"
dataset_name="pauq"
split_name="pauq_xsp"


if [dataset_type="pauq"];
then
  path2train="$project_path/$data_dir/$dataset_name/${split_name}_train.json"
  path2test="project_path/$data_dir/$dataset_name/${split_name}_train.json"
  saving_path="$project_path/$experiments_folder/${dataset_name}_${split_name}_s${seed}"
else
  path2train="$project_path/$data_dir/$dataset_name/train"
  path2test="$project_path/$data_dir/$dataset_name/test"
  saving_path="$project_path/$experiments_folder/${dataset_name}_s${seed}"
fi
tables_path="$project_path/$data_dir/$dataset_name/tables.json"
log_dir="$saving_path/training_logs"

log_ratio=$(echo "0.1" | bc)
log_steps=$(echo "$epoch * $log_ratio" | bc)

seed=42
input_seq_length=512
output_seq_length=256
train_batch_size=4
eval_batch_size=16
gradient_accumulation_steps=64
epochs_number=3
lr="1e-4"


python train_llm.py \
    --model_name $llama3_model_path \
    --sql_dataset_name $dataset_name \
    --path_to_training_file $path2train \
    --tables_info_path $tables_path \
    --learning_rate $lr \
    --seed $seed \
    --per_device_train_batch_size $train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_seq_length $input_seq_length \
    --max_output_length $output_seq_length \
    --logging_steps $log_steps \
    --report_to "tensorboard" \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --output_dir $saving_path \
    --merge_and_push \
    --logging_dir $log_dir \
    --num_train_epochs $epochs_number \
    --try_one_batch

python infer_llm.py \
    --model_name $saving_path \
    --sql_dataset_name $dataset_name \
    --path_to_testing_file $path2test \
    --tables_info_path $tables_path \
    --seed $seed \
    --max_seq_length $input_seq_length \
    --max_output_length $output_seq_length \
    --per_device_eval_batch_size $eval_batch_size \
    --eval_accumulation_steps $gradient_accumulation_steps \
    --generation_max_length $output_seq_length \
    --num_beams 1 \
    --output_dir $saving_path \
    --try_one_batch
