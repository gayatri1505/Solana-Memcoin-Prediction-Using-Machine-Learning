import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
import xgboost as xgb



# -----

DIR = '/kaggle/input/pump-fun-graduation-february-2025'

train = pd.read_csv(os.path.join(DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DIR, 'test_unlabeled.csv'))

dune_token_info = pd.read_csv(os.path.join(DIR, 'dune_token_info_v2.csv'))
token_info_onchain = pd.read_csv(os.path.join(DIR, 'token_info_onchain_divers_v2.csv'))

chunk_files = glob.glob(os.path.join(DIR, 'chunk_*.csv'))
chunks = pd.concat((pd.read_csv(f) for f in chunk_files), ignore_index=True)


# -----

chunks = chunks[chunks['base_coin'].isin(train['mint'])]

# -----

grouped = chunks.groupby('base_coin')
features = grouped.agg(
    num_trades=('direction', 'count'),
    num_buys=('direction', lambda x: (x == 'buy').sum()),
    num_sells=('direction', lambda x: (x == 'sell').sum()),
    total_base_amount=('base_coin_amount', 'sum'),
    total_quote_amount=('quote_coin_amount', 'sum'),
    final_virtual_sol_balance=('virtual_sol_balance_after', 'last'),
).reset_index()
features['buy_sell_ratio'] = features['num_buys'] / (features['num_sells'] + 1)
features.rename(columns={'base_coin': 'mint'}, inplace=True)

# -----

dune_token_info['name_length'] = dune_token_info['name'].fillna('').apply(len)
dune_token_info['symbol_length'] = dune_token_info['symbol'].fillna('').apply(len)
dune_token_info['has_metadata_url'] = dune_token_info['token_uri'].notna().astype(int)
dune_features = dune_token_info[['token_mint_address', 'decimals', 'name_length', 'symbol_length', 'has_metadata_url']]
dune_features.rename(columns={'token_mint_address': 'mint'}, inplace=True)

creator_counts = token_info_onchain['creator'].value_counts().to_dict()
token_info_onchain['creator_num_tokens'] = token_info_onchain['creator'].map(creator_counts)
onchain_features = token_info_onchain[['mint', 'bundle_size', 'gas_used', 'creator_num_tokens']]


# -----

def time_features(chunks_df):
    chunks_df['block_time'] = pd.to_datetime(chunks_df['block_time'])
    grouped = chunks_df.groupby('base_coin')
    time_feats = grouped.agg(
        first_trade_time=('block_time', 'min'),
        last_trade_time=('block_time', 'max'),
        num_trades=('direction', 'count')
    )
    time_feats['active_duration_secs'] = (time_feats['last_trade_time'] - time_feats['first_trade_time']).dt.total_seconds()

    buys = chunks_df[chunks_df['direction'] == 'buy']
    grouped_buys = buys.groupby('base_coin')['block_time']
    time_feats['first_buy_time'] = grouped_buys.min()
    time_feats['last_buy_time'] = grouped_buys.max()
    time_feats['buy_duration_secs'] = (time_feats['last_buy_time'] - time_feats['first_buy_time']).dt.total_seconds()
    time_feats['time_to_first_buy_secs'] = (time_feats['first_buy_time'] - time_feats['first_trade_time']).dt.total_seconds()

    sells = chunks_df[chunks_df['direction'] == 'sell']
    grouped_sells = sells.groupby('base_coin')['block_time']
    time_feats['first_sell_time'] = grouped_sells.min()
    time_feats['last_sell_time'] = grouped_sells.max()
    time_feats['sell_duration_secs'] = (time_feats['last_sell_time'] - time_feats['first_sell_time']).dt.total_seconds()
    time_feats['time_to_first_sell_secs'] = (time_feats['first_sell_time'] - time_feats['first_trade_time']).dt.total_seconds()

    time_feats = time_feats.fillna(0).reset_index()
    time_feats.rename(columns={'base_coin': 'mint'}, inplace=True)
    time_feats = time_feats.drop(columns=['num_trades'])
    return time_feats

time_feats = time_features(chunks)

# -----

train = train.merge(features, on='mint', how='left')
train = train.merge(dune_features, on='mint', how='left')
train = train.merge(onchain_features, on='mint', how='left')
train = train.merge(time_feats, on='mint', how='left')

test = test.merge(features, on='mint', how='left')
test = test.merge(dune_features, on='mint', how='left')
test = test.merge(onchain_features, on='mint', how='left')
test = test.merge(time_feats, on='mint', how='left')

# -----

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
test.columns = test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# -----

# identify all datetime cols
dt_cols = [
    'first_trade_time', 'last_trade_time',
    'first_buy_time',   'last_buy_time',
    'first_sell_time',  'last_sell_time'
]

# drop them from both train and test
train = train.drop(columns=[c for c in dt_cols if c in train.columns])
test  = test.drop(columns=[c for c in dt_cols if c in test.columns])

# just to be safe, re-clean names & fill NAs
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train.columns = train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
test.columns  = test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# -----

X = train.drop(columns=['mint', 'slot_min', 'slot_graduated', 'has_graduated'])
X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
y = train['has_graduated']


# -----


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
X_val.columns = X_val.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

y_pred = model.predict_proba(X_val)[:, 1]
print("Validation Log Loss:", log_loss(y_val, y_pred))


# -----

importances = model.feature_importances_
features = X_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# -----

X_test = test.drop(columns=['mint', 'slot_min']).fillna(0)
X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
# === ALIGN TEST FEATURES TO TRAIN ===

# 1) Get the list of feature names your model was trained on
train_cols = list(X_train.columns)

# 2) Subset out any extra columns in X_test
extra = [c for c in X_test.columns if c not in train_cols]
if extra:
    X_test = X_test.drop(columns=extra)

# 3) Add any missing columns (fill with zeros)
for c in train_cols:
    if c not in X_test.columns:
        X_test[c] = 0

# 4) Reorder to match exactly
X_test = X_test[train_cols]

# 5) Now predict
test_preds = model.predict_proba(X_test)[:, 1]


# -----

# after predicting `test_preds` aligned to test.index:
submission = pd.DataFrame({
    'mint': test['mint'],
    'has_graduated': test_preds
})

# average probabilities for any duplicated mint
submission = (
    submission
    .groupby('mint', as_index=False)
    .has_graduated.mean()
    .rename(columns={'has_graduated':'has_graduated'})
)
submission.to_csv('lgbm_submission.csv', index=False)

# -----

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

# Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              early_stopping_rounds=100, verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

print("XGBoost OOF Log Loss:", log_loss(y, oof_preds))


# -----

# Reshape to 2D as logistic regression expects 2D input
oof_stack = oof_preds.reshape(-1, 1)
test_stack = test_preds.reshape(-1, 1)

meta_model = LogisticRegression(C=1.0, solver='lbfgs')
meta_model.fit(oof_stack, y)

meta_oof = meta_model.predict_proba(oof_stack)[:, 1]
print("Meta-model Log Loss:", log_loss(y, meta_oof))


# -----

final_preds = meta_model.predict_proba(test_stack)[:, 1]

submission = pd.DataFrame({
    'mint': test['mint'],
    'has_graduated': final_preds
})
submission = submission.groupby('mint', as_index=False).has_graduated.mean()
submission.to_csv('submission.csv', index=False)
print("Submission saved to 'submission.csv'")
