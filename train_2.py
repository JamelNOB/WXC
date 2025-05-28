import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS
HOG_PARAMS = {'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'orientations': 9}

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None
    img = cv2.resize(img, (128, 128))

    lbp = local_binary_pattern(img, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    lbp_hist = lbp_hist.astype("float32") / (lbp_hist.sum() + 1e-6)

    hog_features = hog(img, **HOG_PARAMS, feature_vector=True)

    return np.hstack([lbp_hist, hog_features])

def build_dataset(csv_file, image_dir, cache_prefix):
    X_path = os.path.join("train_2", f"{cache_prefix}_X.npy")
    y_path = os.path.join("train_2", f"{cache_prefix}_y.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        return np.load(X_path), np.load(y_path)

    df = pd.read_csv(csv_file)
    features, targets = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(image_dir, row['ID'])
        feats = extract_features(path)
        if feats is not None:
            features.append(feats)
            targets.append(row['Label'])

    X = np.array(features)
    y = np.array(targets)
    np.save(X_path, X)
    np.save(y_path, y)
    return X, y

train_csv = r"Data\train.csv"
val_csv = r"Data\val.csv"
train_img_dir = r"Data\train"
val_img_dir = r"Data\val"

X_train, y_train = build_dataset(train_csv, train_img_dir, "train_cache")
X_val, y_val = build_dataset(val_csv, val_img_dir, "val_cache")

if X_train.size == 0 or X_val.size == 0:
    print("Error: Training or validation data is empty.")
    exit()

# ---------------- Tuning HistGradientBoostingRegressor ----------------
print("Tuning HistGradientBoostingRegressor...")
param_hgb = {
    'learning_rate': [0.05, 0.1],
    'max_iter': [100, 200],
    'max_depth': [None, 5],
    'l2_regularization': [0.0, 1.0]
}
grid_search_hgb = GridSearchCV(
    HistGradientBoostingRegressor(random_state=42, early_stopping=True),
    param_grid=param_hgb,
    cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)
grid_search_hgb.fit(X_train, y_train)
best_hgb = grid_search_hgb.best_estimator_
print("Best HGB MSE:", mean_squared_error(y_val, best_hgb.predict(X_val)))

# ---------------- Tuning RandomForest ----------------
print("\nTuning RandomForestRegressor...")
param_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
}
grid_search_rf = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_rf,
    cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
print("Best RF MSE:", mean_squared_error(y_val, best_rf.predict(X_val)))

# ---------------- Tuning AdaBoost ----------------
print("\nTuning AdaBoostRegressor...")
param_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 1.0],
    'loss': ['linear', 'square']
}
grid_search_ada = GridSearchCV(
    AdaBoostRegressor(random_state=42),
    param_grid=param_ada,
    cv=2, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2
)
grid_search_ada.fit(X_train, y_train)
best_ada = grid_search_ada.best_estimator_
print("Best Ada MSE:", mean_squared_error(y_val, best_ada.predict(X_val)))

mse_ada_val = mean_squared_error(y_val, best_ada.predict(X_val))
mse_rf_val = mean_squared_error(y_val, best_rf.predict(X_val))
mse_hgb_val = mean_squared_error(y_val, best_hgb.predict(X_val))
# ---------------- Model Ensembling: Weighted VotingRegressor ----------------
# (This part remains largely the same, using the Optuna-tuned best_hgb, best_rf, best_ada)
print("\nEvaluating Weighted VotingRegressor...")
epsilon = 1e-9
weights = [
    1.0 / (mse_hgb_val + epsilon),
    1.0 / (mse_rf_val + epsilon),
    1.0 / (mse_ada_val + epsilon)
]
weighted_voting_model = VotingRegressor(
    estimators=[
        ('hgb', best_hgb),
        ('rf', best_rf),
        ('ada', best_ada)
    ],
    weights=weights
)
print("Fitting the Weighted VotingRegressor...")
weighted_voting_model.fit(X_train, y_train)
mse_weighted_voting_val = mean_squared_error(y_val, weighted_voting_model.predict(X_val))
print(f"Weighted VotingRegressor validation MSE: {mse_weighted_voting_val:.4f}")
normalized_weights = np.array(weights) / np.sum(weights)
print(f"Normalized weights used (HGB, RF, Ada): {[f'{w:.3f}' for w in normalized_weights]}")

# ---------------- Final model selection (including加权投票模型) ----------------
print(f"\nValidation MSEs for final selection:")
print(f" - HistGradientBoosting (Optuna): {mse_hgb_val:.4f}")
print(f" - RandomForest (Optuna): {mse_rf_val:.4f}")
print(f" - AdaBoost (Optuna): {mse_ada_val:.4f}")
print(f" - WeightedVotingRegressor: {mse_weighted_voting_val:.4f}")

models_performance = {
    "HistGradientBoostingRegressor": mse_hgb_val,
    "RandomForestRegressor": mse_rf_val,
    "AdaBoostRegressor": mse_ada_val,
    "WeightedVotingRegressor": mse_weighted_voting_val
}

best_model_name = min(models_performance, key=models_performance.get)
min_mse = models_performance[best_model_name]

if best_model_name == "HistGradientBoostingRegressor":
    final_model = best_hgb
elif best_model_name == "RandomForestRegressor":
    final_model = best_rf
elif best_model_name == "AdaBoostRegressor":
    final_model = best_ada
elif best_model_name == "WeightedVotingRegressor":
    final_model = weighted_voting_model
else:
    print(f"Warning: Unknown best_model_name '{best_model_name}'. Defaulting to HistGradientBoostingRegressor.")
    final_model = best_hgb

print(f"Selected final model: {best_model_name} with MSE: {min_mse:.4f}")

test_img_dir = r"Data\test"
test_ids, test_features_list = [], []

for filename in tqdm(os.listdir(test_img_dir)):
    if filename.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
        path = os.path.join(test_img_dir, filename)
        feats = extract_features(path)
        if feats is not None:
            test_ids.append(filename)
            test_features_list.append(feats)

if not test_features_list:
    print("Error: No test features extracted.")
    exit()

test_features = np.array(test_features_list)
test_pred = final_model.predict(test_features)

output_df = pd.DataFrame({'ID': test_ids, 'Label': test_pred})
output_csv_path = r"Data\test_predictions_new.csv"
output_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to: {output_csv_path}")