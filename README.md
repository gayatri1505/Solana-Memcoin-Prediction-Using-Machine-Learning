# Solana Memcoin Prediction: PumpFun to PumpSwap

This project was created as part of the [Solana Memcoin Prediction Challenge](https://www.kaggle.com/competitions/solana-skill-sprint-memcoin-graduation/overview) where the goal was to **predict if a memcoin launched on the Pump Fun platform will graduate by reaching at least 85 SOL liquidity** using data from just the first 100 blocks after minting.

## Problem Overview

Most memcoins on Pump Fun are short-lived. The challenge was to:
- Detect early behavioral signals of a token's success or failure.
- Predict the probability that a token will **graduate** using only early transaction and metadata.

## Approach

### 1. Data Preprocessing

- Integrated multiple datasets: transaction chunks, token metadata from Dune, and on-chain info.
- Filtered swap and liquidity events to only include those within the **first 100 blocks post-mint**, aligning with the competition constraints.
- Handled missing values and ensured consistent token tracking across sources.

### 2. Feature Engineering

Created features from early behavioral data, grouped into:

- **Swap and Liquidity Metrics**: total swap count, direction ratios, LP volume trends.
- **Time Dynamics**: time to first swap/liquidity, block intervals between events.
- **Wallet Behavior**: number of unique traders, early whales, deployer involvement, potential snipers.
- **Token Metadata**: initial supply, deployer reputation (cross-token deployment), token type indicators.

### 3. Model Development

I trained four diverse base learners to capture different aspects of the data:

- LightGBM: for fast training and handling of numerical features.
- XGBoost: for robust gradient boosting with regularization.
- CatBoost: for leveraging categorical token metadata natively.
- MLPClassifier: to model non-linear interactions missed by tree-based models.

Cross-validation was performed using StratifiedKFold to maintain label balance. The performance metric was **log loss**, per competition rules.

### 4. Stacking Ensemble

Predictions from each base model were stacked and fed into a **Logistic Regression** meta-learner. This helped:

- Improve calibration of the output probabilities.
- Combine complementary strengths of different model types.
- Reduce overfitting from any single learner.

Ensembling was done using **Logistic Regression** as the meta-learner.

## Files

- `main.py`: Full end-to-end training and prediction script.
- `requirements.txt`: All dependencies to reproduce this code.
- `data`: https://www.kaggle.com/datasets/dremovd/pump-fun-graduation-february-2025
- `outputs`: Stores submission files, logs, or model outputs.

## Final Score after Submission

- **Log Loss:** `0.0468`

