# %% [markdown]
# # 02 - Data Cleaning
# Fix each issue found in EDA, with a before/after check. Output: `data/clean.csv`
# (train + test stacked, with a `src` column).

# %%
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "data" / "train_v9rqX0R.csv").exists():
    ROOT = ROOT.parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

train = pd.read_csv(ROOT / "data" / "train_v9rqX0R.csv").assign(src="train")
test = pd.read_csv(ROOT / "data" / "test_AbJTz2l.csv").assign(src="test")
df = pd.concat([train, test], ignore_index=True)   # clean both sets identically
print("combined:", df.shape)
print("missing before:\n", df.drop(columns="Item_Outlet_Sales").isna().sum()[lambda s: s > 0])

# %% [markdown]
# ## 1. Standardise `Item_Fat_Content`

# %%
print("before:", df.Item_Fat_Content.unique())
df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
    {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"})
print("after: ", df.Item_Fat_Content.unique())

# %% [markdown]
# ## 2. `Item_Weight` - same item always has the same weight
# Fill from the item's other rows; fall back to the item-type mean.

# %%
df["Item_Weight"] = df["Item_Weight"].fillna(
    df.groupby("Item_Identifier")["Item_Weight"].transform("mean"))
df["Item_Weight"] = df["Item_Weight"].fillna(
    df.groupby("Item_Type")["Item_Weight"].transform("mean"))
print("Item_Weight missing:", df.Item_Weight.isna().sum())

# %% [markdown]
# ## 3. `Outlet_Size` - impute with the mode of the outlet type

# %%
print(pd.crosstab(df.Outlet_Type, df.Outlet_Size.fillna("MISSING")))
df["Outlet_Size"] = df["Outlet_Size"].fillna(
    df.groupby("Outlet_Type")["Outlet_Size"].transform(lambda s: s.mode()[0]))
print("Outlet_Size missing:", df.Outlet_Size.isna().sum())

# %% [markdown]
# ## 4. `Item_Visibility` - 0 is impossible (item is on shelf); treat as missing

# %%
print("zeros before:", (df.Item_Visibility == 0).sum())
df["Item_Visibility"] = df["Item_Visibility"].replace(0, np.nan)
df["Item_Visibility"] = df["Item_Visibility"].fillna(
    df.groupby("Item_Identifier")["Item_Visibility"].transform("mean"))
print("zeros/missing after:", df.Item_Visibility.isna().sum())

# %% [markdown]
# ## Validate & save

# %%
assert df.drop(columns="Item_Outlet_Sales").isna().sum().sum() == 0
df.to_csv(OUT / "clean.csv", index=False)
print("saved", OUT / "clean.csv", df.shape)
df.head()
