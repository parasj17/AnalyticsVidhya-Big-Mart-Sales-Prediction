# %% [markdown]
# # 04 - Modeling & Submission
# Blend of 5 regularised boosting models on `sqrt(sales)`; 5-fold CV; 3 seeds each for the final fit.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path.cwd()
if not (ROOT / "data" / "train_v9rqX0R.csv").exists():
    ROOT = ROOT.parent
df = pd.read_csv(ROOT / "data" / "features.csv")

CATS = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size",
        "Outlet_Location_Type", "Outlet_Type", "Item_Cat"]
DROP = ["Item_Identifier", "Item_Outlet_Sales", "src", "Outlet_ID_raw"]
feats = [c for c in df.columns if c not in DROP]
tr, te = df[df.src == "train"], df[df.src == "test"]
X, y = tr[feats], tr["Item_Outlet_Sales"]
y_t = np.sqrt(y)                      # target transform (reduces skew)
cat_mask = [f in CATS for f in feats]

SMALL = ["Item_MRP", "Outlet_Identifier", "Outlet_Type", "Item_Visibility", "Item_Weight",
         "Item_Type", "Outlet_Age", "Outlet_Size", "Outlet_Location_Type"]

def hgb(depth, it, lr, leaf, l2, cols):
    return lambda seed: HistGradientBoostingRegressor(
        learning_rate=lr, max_iter=it, max_depth=depth, min_samples_leaf=leaf, l2_regularization=l2,
        categorical_features=[f in CATS for f in cols], random_state=seed)

# (columns, model factory): diverse, heavily regularised models, averaged below
MODELS = {
    "hgb_small": (SMALL, hgb(3, 300, .03, 60, 10, SMALL)),
    "hgb_d2":    (feats, hgb(2, 600, .04, 80, 10, feats)),
    "hgb_d3":    (feats, hgb(3, 400, .03, 60, 10, feats)),
    "xgb":       (feats, lambda s: xgb.XGBRegressor(
                     n_estimators=500, learning_rate=.02, max_depth=3, subsample=.7,
                     colsample_bytree=.7, min_child_weight=10, reg_lambda=10, random_state=s)),
    "gbr":       (feats, lambda s: GradientBoostingRegressor(
                     n_estimators=300, learning_rate=.03, max_depth=3, subsample=.7,
                     min_samples_leaf=30, random_state=s)),
}

# %% [markdown]
# ## Cross-validation (each model, then the blend)

# %%
kf = KFold(5, shuffle=True, random_state=42)
oofs = {}
for name, (cols, mk) in MODELS.items():
    oofs[name] = np.square(cross_val_predict(mk(0), X[cols], y_t, cv=kf))
    print(f"{name:10s} CV RMSE: {np.sqrt(np.mean((oofs[name] - y) ** 2)):.1f}")
blend = np.mean(list(oofs.values()), axis=0)
rmse = np.sqrt(np.mean((blend - y) ** 2))
print(f"BLEND      CV RMSE: {rmse:.1f}   (baseline predicting the mean: {y.std():.1f})")

# %% [markdown]
# ## Feature importance (permutation, on a holdout split)

# %%
from sklearn.inspection import permutation_importance
idx = np.random.RandomState(0).permutation(len(X)); cut = int(.8 * len(X))
m = MODELS["hgb_d3"][1](0).fit(X.iloc[idx[:cut]], y_t.iloc[idx[:cut]])
imp = permutation_importance(m, X.iloc[idx[cut:]], y_t.iloc[idx[cut:]], n_repeats=5, random_state=0)
imp = pd.Series(imp.importances_mean, index=feats).sort_values()
imp.plot.barh(figsize=(7, 6), color="#4C78A8"); plt.title("Permutation importance")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Final fit + submission

# %%
pred = np.mean([np.square(mk(s).fit(X[cols], y_t).predict(te[cols]))
                for cols, mk in MODELS.values() for s in range(3)], axis=0)
sub = pd.DataFrame({"Item_Identifier": te["Item_Identifier"].values,
                    "Outlet_Identifier": te["Outlet_ID_raw"].values,
                    "Item_Outlet_Sales": np.clip(pred, 0, None)})
sub.to_csv(ROOT / "04_modeling" / "submission.csv", index=False)
print("saved submission.csv", sub.shape)
sub.head()
