import pandas as pd 
import numpy as np
import catboost as cbt
from sklearn.model_selection import GridSearchCV 


def generate_features(df):
    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
               'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]
    
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
    
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            if i>j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
                features.append(f'{a}_{b}_imb')    
                    
    for i,a in enumerate(prices):
        for j,b in enumerate(prices):
            for k,c in enumerate(prices):
                if i>j and j>k:
                    max_ = df[[a,b,c]].max(axis=1)
                    min_ = df[[a,b,c]].min(axis=1)
                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_

                    df[f'{a}_{b}_{c}_imb2'] = (max_-mid_)/(mid_-min_)
                    features.append(f'{a}_{b}_{c}_imb2')
    
    return df[features]

print("worked")

df_train = pd.read_csv('train.csv')
df_ = generate_features(df_train)

X = df_.values
Y = df_train['target'].values

X = X[np.isfinite(Y)]
Y = Y[np.isfinite(Y)]

index = np.arange(len(X))

print("worked 2")

cbt_param_grid = {
    'iterations': [500],
    'learning_rate': [0.1, 0.05],
    'depth': [25], #4, 8, 15
    'l2_leaf_reg': [1, 3, 5],
    'subsample': [0.5, 1.0],
    'border_count': [32, 63, 128, 255]
}

model = cbt.CatBoostRegressor(objective='MAE', random_seed=42)

print("worked 3")

# Initialize the GridSearchCV for CatBoost
grid_search = GridSearchCV(
    estimator=model,
    param_grid=cbt_param_grid,
    cv=5,
    scoring='neg_mean_absolute_error'
)

print("worked 4")
# Fit the grid search to the data
grid_search.fit(X, Y, verbose=50)

# Get the best hyperparameter values and corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print the results
print("Best parameters:", best_params)
print("Best score:", -best_score)  # Negative sign for neg_mean_squared_error