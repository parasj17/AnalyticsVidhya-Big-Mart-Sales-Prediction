"""BigMart Sales Prediction - single-file solution (cleaning + features + blended boosting).

Usage:  python solution.py [train.csv] [test.csv] [output.csv]
Defaults look for train_v9rqX0R.csv / test_AbJTz2l.csv next to this file, one folder up, or in ../data.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def find(name):
    for d in (HERE, HERE.parent, HERE.parent / "data", Path.cwd()):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


train_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find("train_v9rqX0R.csv")
test_path = Path(sys.argv[2]) if len(sys.argv) > 2 else find("test_AbJTz2l.csv")
out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "submission.csv"

train = pd.read_csv(train_path).assign(src="train")
test = pd.read_csv(test_path).assign(src="test")
df = pd.concat([train, test], ignore_index=True)   # clean/engineer both identically

# ---------------- cleaning ----------------
df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
df["Item_Weight"] = df["Item_Weight"].fillna(
    df.groupby("Item_Identifier")["Item_Weight"].transform("mean"))
df["Item_Weight"] = df["Item_Weight"].fillna(
    df.groupby("Item_Type")["Item_Weight"].transform("mean"))
df["Outlet_Size"] = df["Outlet_Size"].fillna(
    df.groupby("Outlet_Type")["Outlet_Size"].transform(lambda s: s.mode()[0]))
df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)
df["Item_Visibility"] = df["Item_Visibility"].fillna(
    df.groupby("Item_Identifier")["Item_Visibility"].transform("mean"))

# ---------------- feature engineering ----------------
df["Item_Cat"] = df["Item_Identifier"].str[:2]
df.loc[df["Item_Cat"] == "NC", "Item_Fat_Content"] = "Non-Edible"
df["Item_Count"] = df.groupby("Item_Identifier")["Item_Identifier"].transform("count")
df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
df["Vis_Ratio"] = df["Item_Visibility"] / df.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")
df["MRP_per_wt"] = df["Item_MRP"] / df["Item_Weight"]
df["MRP_vs_type"] = df["Item_MRP"] / df.groupby("Item_Type")["Item_MRP"].transform("mean")

CATS = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size",
        "Outlet_Location_Type", "Outlet_Type", "Item_Cat"]
outlet_raw = df["Outlet_Identifier"].copy()
for c in CATS:
    df[c] = df[c].astype("category").cat.codes
df = df.drop(columns="Outlet_Establishment_Year")

# ---------------- modeling ----------------
DROP = ["Item_Identifier", "Item_Outlet_Sales", "src"]
feats = [c for c in df.columns if c not in DROP]
SMALL = ["Item_MRP", "Outlet_Identifier", "Outlet_Type", "Item_Visibility", "Item_Weight",
         "Item_Type", "Outlet_Age", "Outlet_Size", "Outlet_Location_Type"]
tr, te = df[df.src == "train"], df[df.src == "test"]
y = tr["Item_Outlet_Sales"]
y_t = np.sqrt(y)                                    # reduces target skew


def hgb(depth, it, lr, leaf, l2, cols):
    return lambda seed: HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=it, max_depth=depth, min_samples_leaf=leaf, l2_regularization=l2,
        categorical_features=[f in CATS for f in cols], random_state=seed)


MODELS = {
    "hgb_small": (SMALL, hgb(3, 300, .03, 60, 10, SMALL)),
    "hgb_d2": (feats, hgb(2, 600, .04, 80, 10, feats)),
    "hgb_d3": (feats, hgb(3, 400, .03, 60, 10, feats)),
    "xgb": (feats, lambda s: xgb.XGBRegressor(
        n_estimators=500, learning_rate=.02, max_depth=3, subsample=.7,
        colsample_bytree=.7, min_child_weight=10, reg_lambda=10, random_state=s)),
    "gbr": (feats, lambda s: GradientBoostingRegressor(
        n_estimators=300, learning_rate=.03, max_depth=3, subsample=.7,
        min_samples_leaf=30, random_state=s)),
}

kf = KFold(5, shuffle=True, random_state=42)
oofs = [np.square(cross_val_predict(mk(0), tr[cols], y_t, cv=kf)) for cols, mk in MODELS.values()]
print("Blend CV RMSE:", round(float(np.sqrt(np.mean((np.mean(oofs, axis=0) - y) ** 2))), 1))

pred = np.mean([np.square(mk(s).fit(tr[cols], y_t).predict(te[cols]))
                for cols, mk in MODELS.values() for s in range(3)], axis=0)

pd.DataFrame({"Item_Identifier": te["Item_Identifier"].values,
              "Outlet_Identifier": outlet_raw[te.index].values,
              "Item_Outlet_Sales": np.clip(pred, 0, None)}).to_csv(out_path, index=False)
print("saved", out_path, len(pred))
