#!/bin/bash
# запуск на кластере
# сначала актвиируем окружение - conda activate llm_tuning_env
# srun/sbatch -A proj_1428 infer_on_cluster.sh

# run config
#SBATCH --job-name=llm_test_inference       # Название задачи
#SBATCH --error=/home/text2sql_llama_3/cluster_logs/llm_test_inference.err        # Файл для вывода ошибок
#SBATCH --output=/home/text2sql_llama_3/cluster_logs/llm_test_inference.log       # Файл для вывода результатов
#SBATCH --time=24:00:00                      # Максимальное время выполнения
#SBATCH --nodes=1                           # Требуемое кол-во узлов
#SBATCH --cpus-per-task=8                   # Количество CPU на одну задачу
#SBATCH --gpus=1                            # Требуемое кол-во GPU


python infer_llm.py

