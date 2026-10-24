# %% [markdown]
# # 01 - Exploratory Data Analysis
# BigMart Sales: what does the raw data look like, and what needs fixing?

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "data" / "train_v9rqX0R.csv").exists():
    ROOT = ROOT.parent
PLOTS = ROOT / "01_eda" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})

train = pd.read_csv(ROOT / "data" / "train_v9rqX0R.csv")
test = pd.read_csv(ROOT / "data" / "test_AbJTz2l.csv")
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Missing values & dtypes

# %%
miss = pd.concat([train.isna().sum(), test.isna().sum()], axis=1, keys=["train", "test"])
print(miss[miss.sum(axis=1) > 0])
train.describe().T

# %% [markdown]
# Missing: `Item_Weight` (~17%) and `Outlet_Size` (~28%). `Item_Visibility` also has hidden
# missing values encoded as 0.

# %%
print("Zero visibility rows:", (train.Item_Visibility == 0).sum())
for c in ["Item_Fat_Content", "Outlet_Size", "Outlet_Type", "Outlet_Location_Type"]:
    print(f"\n{c}\n", train[c].value_counts(dropna=False))

# %% [markdown]
# `Item_Fat_Content` has inconsistent labels (LF / low fat / Low Fat, reg / Regular).

# %% [markdown]
# ## Target distribution

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(train.Item_Outlet_Sales, bins=50, color="#4C78A8")
ax[0].set_title("Item_Outlet_Sales (right-skewed)")
ax[1].hist(np.sqrt(train.Item_Outlet_Sales), bins=50, color="#59A14F")
ax[1].set_title("sqrt(Item_Outlet_Sales) - closer to normal")
plt.tight_layout(); plt.savefig(PLOTS / "target_dist.png"); plt.show()
print("skew raw:", round(train.Item_Outlet_Sales.skew(), 2),
      "| sqrt:", round(np.sqrt(train.Item_Outlet_Sales).skew(), 2))

# %% [markdown]
# ## Numeric drivers

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(train.Item_MRP, train.Item_Outlet_Sales, s=3, alpha=.4)
ax[0].set_title("MRP vs Sales (strongest driver; 4 price bands)")
ax[1].scatter(train.Item_Visibility, train.Item_Outlet_Sales, s=3, alpha=.4, color="#E15759")
ax[1].set_title("Visibility vs Sales")
ax[2].scatter(train.Item_Weight, train.Item_Outlet_Sales, s=3, alpha=.4, color="#59A14F")
ax[2].set_title("Weight vs Sales (no signal)")
plt.tight_layout(); plt.savefig(PLOTS / "numeric_vs_sales.png"); plt.show()

# %% [markdown]
# ## Categorical drivers

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, c in zip(axes.ravel(), ["Outlet_Identifier", "Outlet_Type", "Outlet_Size",
                                "Outlet_Location_Type", "Item_Type", "Item_Fat_Content"]):
    train.groupby(c).Item_Outlet_Sales.mean().sort_values().plot.barh(ax=ax, color="#4C78A8")
    ax.set_title(f"Mean sales by {c}"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(PLOTS / "categorical_vs_sales.png"); plt.show()

# %% [markdown]
# `OUT027` (Supermarket Type3) sells most; `OUT010`/`OUT019` (grocery stores) sell least.
# Outlet identity is a very strong signal.

# %% [markdown]
# ## Correlations

# %%
num = train.select_dtypes("number").drop(columns="Outlet_Establishment_Year")
corr = num.corr()
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar(im); plt.tight_layout(); plt.savefig(PLOTS / "corr.png"); plt.show()

# %% [markdown]
# ## Takeaways
# - Fix: fat-content labels, missing weight, missing outlet size, zero visibility.
# - Target is skewed -> model on `sqrt(sales)`.
# - Engineer: outlet age, item category (ID prefix), price ratios, relative visibility.
