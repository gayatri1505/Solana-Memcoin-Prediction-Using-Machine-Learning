# Solana Memcoin Prediction: PumpFun to PumpSwap

This project was created as part of the [Solana Memcoin Prediction Challenge](https://www.kaggle.com/competitions/solana-skill-sprint-memcoin-graduation/overview) where the goal was to **predict if a memcoin launched on the Pump Fun platform will graduate by reaching at least 85 SOL liquidity** using data from just the first 100 blocks after minting.

## Problem Overview

Most memcoins on Pump Fun are short-lived. The challenge was to:
- Detect early behavioral signals of a token's success or failure.
- Predict the probability that a token will **graduate** using only early transaction and metadata.

## Approach

I built a **stacking ensemble model** with the following base learners:
- LightGBM
- XGBoost
- CatBoost
- Multi-layer Perceptron (MLP)

Ensembling was done using **Logistic Regression** as the meta-learner.

## Files

- `main.py`: Full end-to-end training and prediction script.
- `requirements.txt`: All dependencies to reproduce this code.
- `data/`: All data files
- `outputs/`: Stores submission files, logs, or model outputs.

## Final Score after Submission

- **Log Loss:** `0.0468`

