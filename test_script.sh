run_name="test_llama"
CUDA_DEVICE_NUMBER="1"
tmux new-session -d -s $run_name
tmux send-keys -t $run_name "CUDA_VISIBLE_DEVICES='$CUDA_DEVICE_NUMBER' /home/somov/.conda/envs/llm_tuning/bin/python -u test_llama_3.py" ENTER
tmux a -t $run_name