### Llama 3 SFT and LoRa training with confidence estimation for paper "Confidence Estimation for Error Detection in Text-to-SQL Systems" AAAI-2024

#### Set-up
To set-up environment for traning and inference install Python 3.10 and install requirements.txt with pip. Download Llama 3 checkpoint from here https://huggingface.co/meta-llama/Meta-Llama-3-8B 


#### Training

To run training, configure paths in run_training.sh file to SFT train and val datasets and hyper-parameters and run with bash:

```console
. run_traning.sh
```


#### Inference
To run inference, configure path in run_inference.sh to SFT test dataset and hyper-paramenters and run with bash:

```console
. run_inference.sh
```







