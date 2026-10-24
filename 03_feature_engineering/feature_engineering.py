# %% [markdown]
# # 03 - Feature Engineering
# Input: `data/clean.csv`. Output: `data/features.csv`.

# %%
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "data" / "train_v9rqX0R.csv").exists():
    ROOT = ROOT.parent
df = pd.read_csv(ROOT / "data" / "clean.csv")
print(df.shape)

# %% [markdown]
# ## Item features
# - `Item_Cat`: first 2 chars of the ID (FD food, DR drink, NC non-consumable)
# - Non-consumables can't have a fat content -> `Non-Edible`
# - `Item_Count`: how many outlets stock the item (popularity proxy)

# %%
df["Item_Cat"] = df["Item_Identifier"].str[:2]
df.loc[df["Item_Cat"] == "NC", "Item_Fat_Content"] = "Non-Edible"
df["Item_Count"] = df.groupby("Item_Identifier")["Item_Identifier"].transform("count")
print(df.groupby(["Item_Cat", "Item_Fat_Content"]).size())

# %% [markdown]
# ## Outlet features
# - `Outlet_Age`: data is from 2013
# - `Vis_Ratio`: item visibility relative to the outlet's average (shelf share is only
#   meaningful within a store)

# %%
df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
df["Vis_Ratio"] = df["Item_Visibility"] / df.groupby("Outlet_Identifier")["Item_Visibility"].transform("mean")

# %% [markdown]
# ## Price features
# - `MRP_per_wt`: price per unit weight
# - `MRP_vs_type`: price relative to the average of its item type

# %%
df["MRP_per_wt"] = df["Item_MRP"] / df["Item_Weight"]
df["MRP_vs_type"] = df["Item_MRP"] / df.groupby("Item_Type")["Item_MRP"].transform("mean")

# %% [markdown]
# ## Encode categoricals (integer codes; model treats them as categorical)

# %%
CATS = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Size",
        "Outlet_Location_Type", "Outlet_Type", "Item_Cat"]
keep_ids = df[["Item_Identifier", "Outlet_Identifier"]].rename(
    columns={"Outlet_Identifier": "Outlet_ID_raw"})
for c in CATS:
    df[c] = df[c].astype("category").cat.codes
df = df.join(keep_ids["Outlet_ID_raw"])

# %% [markdown]
# ## Save

# %%
df = df.drop(columns="Outlet_Establishment_Year")   # replaced by Outlet_Age
df.to_csv(ROOT / "data" / "features.csv", index=False)
print("saved features.csv", df.shape)
df.head()
