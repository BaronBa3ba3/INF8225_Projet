# INF8225 Project

Final project for **INF8225 — I.A.: techniques probabilistes et d'apprentissage**.

This repository contains a Jupyter Notebook implementation and evaluation of neural sequence models for **English-to-French machine translation**, with an a focuse on **Mamba-style sequence models**.

## Contents

- `2055734_2062203_Projet.ipynb` — main notebook containing the implementation, training, experiments, and analysis.
- `1_Documentation/` — reference documents used for the Mamba section.
- `2_Output/results/` — saved experiment results in `.json`, `.pkl`, and `.csv` formats.
- `2_Output/figures/` — generated plots used for the report.
- `2_Output/mamba_main.pt` — saved trained Mamba model checkpoint.

## Models and Experiments

The notebook includes:

- RNN, GRU, and Transformer models for translation.
- Greedy search and beam search decoding.
- BLEU score evaluation.
- Mamba model implementation for translation.
- Synthetic tasks such as selective copying and induction heads.
- Ablation studies and hyperparameter sweeps.
- Efficiency benchmarking with respect to sequence length.

## Environment

The experiments were run locally in **Jupyter Notebook** on a personal computer equipped with an **NVIDIA GeForce RTX 3060 GPU**. Note that, du to some interuptions, the code was run in segments. Thus, the Jupyter Notebook blocks do not necesserly contain the final ouput. Instead the ouput is saved in the `results/` folder.

## Setup

Install the main dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn spacy sacrebleu torchinfo einops wandb
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm