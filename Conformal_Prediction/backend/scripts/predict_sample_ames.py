import requests
import pandas as pd
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
# nom du fichier dans le dossier models/
MODEL_FILENAME = "ames_rf_mapie.joblib"
CSV_PATH = "backend/data/ames.csv"


def build_instance_from_csv(n=1):
    df = pd.read_csv(CSV_PATH)
    if "SalePrice" in df.columns:
        X = df.drop(columns=["SalePrice"])
    else:
        X = df
    X = X.head(n).copy()

    def sanitize_value(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        return v

    records = []
    for _, row in X.iterrows():
        rec = {col: sanitize_value(row[col]) for col in X.columns}
        records.append(rec)
    return records


def main():
    instances = build_instance_from_csv(n=3)
    payload = {"model_filename": MODEL_FILENAME,
               "instances": instances, "alpha": 0.05}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        for i, item in enumerate(resp.json()):
            print(
                f"Instance {i}: pred={item['prediction']:.2f}, lower={item['lower']:.2f}, upper={item['upper']:.2f}")
    else:
        print("Erreur API:", resp.status_code, resp.text)


if __name__ == "__main__":
    main()
