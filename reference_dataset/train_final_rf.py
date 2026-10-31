# -*- coding: utf-8 -*-
'Train the final random forest on all screened point samples.'

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier




base_dir = r"K:\Monthly surface water dynamics TIB\supervised classification\step1_machine_learning_output\feature matrix"
all_path = os.path.join(base_dir, "all_samples.csv")

output_dir = os.path.join(base_dir, "final_rf_unweighted_all_samples")
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "rf_unweighted_all_samples.pkl")
params_path = os.path.join(output_dir, "rf_unweighted_all_samples_params.json")
importance_path = os.path.join(output_dir, "rf_unweighted_all_samples_feature_importance.csv")
data_info_path = os.path.join(output_dir, "rf_unweighted_all_samples_data_info.json")
log_path = os.path.join(output_dir, "train_log.txt")




features = [
    "b2", "b3", "b4", "b8", "b11", "b12",
    "AWEI", "MBWI", "NDWI", "MNDWI",
    "slope", "dem", "wetness",
    "asm", "contrast", "dissimilar", "entropy"
]
label_col = "real_value"




def log(msg):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")




def load_and_clean_csv(path, features, label_col):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, usecols=features + [label_col])

    
    df = df.replace([-32768, np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    
    for col in features:
        df[col] = df[col].astype(np.float32)
    df[label_col] = df[label_col].astype(np.int8)

    return df




with open(log_path, "w", encoding="utf-8") as f:
    f.write("Training the final unweighted random forest on all samples\n")

log("Reading all_samples.csv ...")
all_df = load_and_clean_csv(all_path, features, label_col)

log(f"Total samples: {len(all_df)}")
log("Class distribution:")
log(all_df[label_col].value_counts(dropna=False))

X_all = all_df[features]
y_all = all_df[label_col]




rf_params = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,   # RF_unweighted
    "random_state": 42,
    "n_jobs": 2
}

log("\nTraining the final unweighted random forest ...")
log("Model parameters:")
log(rf_params)

model = RandomForestClassifier(**rf_params)
model.fit(X_all, y_all)

log("Model training complete.")




joblib.dump(model, model_path)
log(f"Model saved: {model_path}")




with open(params_path, "w", encoding="utf-8") as f:
    json.dump(rf_params, f, ensure_ascii=False, indent=2)
log(f"Model parameters saved: {params_path}")




importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
log(f"Feature importance saved: {importance_path}")




data_info = {
    "n_samples": int(len(all_df)),
    "n_features": int(len(features)),
    "features": features,
    "label_col": label_col,
    "class_distribution": {
        str(k): int(v) for k, v in all_df[label_col].value_counts(dropna=False).to_dict().items()
    }
}

with open(data_info_path, "w", encoding="utf-8") as f:
    json.dump(data_info, f, ensure_ascii=False, indent=2)

log(f"Training data information saved: {data_info_path}")

log("\nComplete.")
