import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import KFold
D=r"D:\AMBA\data\\"
tr=pd.read_csv(D+"train_v9rqX0R.csv"); te=pd.read_csv(D+"test_AbJTz2l.csv"); T="Item_Outlet_Sales"
f=pd.concat([tr.assign(t=1),te.assign(t=0)],ignore_index=True)
f["Item_Fat_Content"]=f["Item_Fat_Content"].replace({"LF":"Low Fat","low fat":"Low Fat","reg":"Regular"})
f["grp"]=f["Item_Identifier"].str[:2]
f.loc[f.grp=="NC","Item_Fat_Content"]="Non-Edible"
f["Item_Weight"]=f["Item_Weight"].fillna(f.groupby("Item_Identifier")["Item_Weight"].transform("mean"))
ms=f.groupby("Outlet_Type")["Outlet_Size"].agg(lambda s:s.mode().iloc[0])
f["Outlet_Size"]=f["Outlet_Size"].fillna(f["Outlet_Type"].map(ms))
v=f["Item_Visibility"].replace(0,np.nan)
f["Item_Visibility"]=v.fillna(v.groupby(f["Item_Identifier"]).transform("mean"))
f["yrs"]=2013-f["Outlet_Establishment_Year"]
f["mrp_item"]=f.groupby("Item_Identifier")["Item_MRP"].transform("mean")
f["cnt"]=f.groupby("Item_Identifier")["Item_Identifier"].transform("count")
cats=["Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Size","Outlet_Location_Type","Outlet_Type","grp"]
for c in cats: f[c]=f[c].astype("category").cat.codes
X=f[f.t==1].drop(columns=["Item_Identifier",T,"t","Outlet_Establishment_Year"]).reset_index(drop=True)
y=tr[T].values
def cv(feats,mk,tf,inv,reps=1):
    sc=[]
    for r in range(reps):
        oof=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=r).split(X):
            m=mk(); m.fit(X.loc[a,feats],tf(y[a])); oof[b]=inv(m.predict(X.loc[b,feats]))
        sc.append(np.sqrt(np.mean((oof-y)**2)))
    return np.mean(sc)
allf=list(X.columns)
mask=lambda fs:[c in cats for c in fs]
def mk(fs,**k):
    p=dict(learning_rate=.03,max_iter=300,max_depth=4,min_samples_leaf=40,l2_regularization=1.0)
    p.update(k); return lambda:H(categorical_features=mask(fs),random_state=0,**p)
ident=lambda x:x
small=["Item_MRP","Outlet_Identifier","Outlet_Type","Item_Type","Item_Weight","Item_Visibility","yrs"]
from sklearn.linear_model import Ridge
F=["Item_MRP","Outlet_Identifier"]
og=np.zeros(len(y)); ol=np.zeros(len(y)); ob=np.zeros(len(y))
for a,b in KFold(5,shuffle=True,random_state=0).split(X):
    g=mk(F,learning_rate=.02,max_iter=200,max_depth=4,min_samples_leaf=100,l2_regularization=5)(); g.fit(X.loc[a,F],y[a]); og[b]=g.predict(X.loc[b,F])
    for o in X.Outlet_Identifier.unique():
        ia=a[X.Outlet_Identifier.values[a]==o]; ib=b[X.Outlet_Identifier.values[b]==o]
        # piecewise: MRP + bands
        def fe(m): m=m.values; return np.c_[m,m**2,(m>69)*(m-69),(m>136)*(m-136),(m>203)*(m-203)]
        r=Ridge(1.0).fit(fe(X.Item_MRP.iloc[ia]),y[ia]); ol[ib]=r.predict(fe(X.Item_MRP.iloc[ib]))
rm=lambda p:round(np.sqrt(np.mean((p-y)**2)),1)
print("gbm",rm(og),"lin",rm(ol),"blend50",rm((og+ol)/2),"blend70g",rm(.7*og+.3*ol))
