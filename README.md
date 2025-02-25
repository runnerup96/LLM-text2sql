# LLama8: Training and Inference Code

This repository contains the code for training and inference of the LLama8 model. It includes setup instructions, data preparation details, and steps for running training and inference.

## Table of Contents
1. [Setup](#setup)
2. [Data](#data)
3. [Training](#training)
4. [Inference](#inference)
5. [License](#license)
6. [Contact](#contact)

---

## Setup

To set up the environment, ensure you have **Python 3.10** installed. 
At first, install PyTorch with your designated CUDA version.
Then install the required libraries using the provided `requirements.txt` file. Better to set up env with **Miniconda**.

```bash
pip install -r requirements.txt
```
---
## Data

The dataset splits for training and evaluation are available from the following sources:

### Datasets
1. **Original PAUQ XSP**  
   - Repository: [PAUQ XSP](https://github.com/ai-spiderweb/pauq)  
   - Contains the database and table information for PAUQ XSP.

2. **Compositional PAUQ Template SSP and PAUQ Test Long SSP**  
   - Google Drive: [Compositional Splits](https://drive.google.com/drive/folders/12cBewVCrBObBb1qgEg1nXHoqq3hHTT7K?usp=sharing)  
   - Code for preparing compositional splits: [Splitting Strategies](https://github.com/runnerup96/splitting-strategies)  

3. **EHRSQL**  
   - Repository: [EHRSQL](https://github.com/glee4810/ehrsql-2024)  

---

## Training

To train the model, follow these steps:

1. Open the `run_training.sh` script.
2. Set up the required paths (detailed in the script comments).
3. Run the script to start training in a TMUX session.

```bash
./run_training.sh
```

### Monitoring Training Progress
To attach to the TMUX session and monitor progress:
```bash
tmux a -t RUN_NAME
```
Replace `RUN_NAME` with the name of your session. To list all active TMUX sessions, use:
```bash
tmux ls
```

---

## Inference

After training, run inference using the following steps:

1. Open the `run_inference.sh` script.
2. Set up the required paths (detailed in the script comments).
3. Run the script in a TMUX session.

```bash
./run_inference.sh
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out:

- **Telegram**: [@olg_smv](https://t.me/olg_smv)  
- **Email**: [somov.ol.dm@gmail.com](mailto:somov.ol.dm@gmail.com)