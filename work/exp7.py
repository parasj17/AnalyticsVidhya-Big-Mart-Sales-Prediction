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
from sklearn.ensemble import HistGradientBoostingRegressor as H
F=["Item_MRP","Outlet_Identifier"]
def fe(m,k=3):
    m=np.asarray(m); c=[m,m**2]+[(m>t)*(m-t) for t in (69,136,203)]
    return np.c_[tuple(c)]
def lin(Xa,ya,Xb,alpha=1.0):
    l=np.zeros(len(Xb))
    for o in Xa.Outlet_Identifier.unique():
        ma=(Xa.Outlet_Identifier==o).values; mb=(Xb.Outlet_Identifier==o).values
        if mb.any(): l[mb]=Ridge(alpha).fit(fe(Xa.Item_MRP[ma]),ya[ma]).predict(fe(Xb.Item_MRP[mb]))
    return l
def gb(Xa,ya,Xb,fs=F,loss="squared_error",**k):
    p=dict(learning_rate=.03,max_iter=250,max_depth=3,min_samples_leaf=80,l2_regularization=1.0); p.update(k)
    return H(loss=loss,categorical_features=[c in cats for c in fs],random_state=0,**p).fit(Xa[fs],ya).predict(Xb[fs])
def gam(Xa,ya,Xb,fs=F):
    return H(loss="gamma",learning_rate=.03,max_iter=250,max_depth=3,min_samples_leaf=80,categorical_features=[c in cats for c in fs],random_state=0).fit(Xa[fs],ya).predict(Xb[fs])
def run(fn,reps=3):
    r=[]
    for s in range(reps):
        o=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=s).split(X): o[b]=fn(X.loc[a],y[a],X.loc[b])
        r.append(np.sqrt(np.mean((o-y)**2)))
    return round(np.mean(r),1)
bl=lambda *fs:(lambda A,ya,B:np.mean([f_(A,ya,B) for f_ in fs],axis=0))
from sklearn.ensemble import RandomForestRegressor as RF, ExtraTreesRegressor as ET
oh=lambda A:pd.get_dummies(A["Outlet_Identifier"].astype(int)).reindex(columns=range(10),fill_value=0).astype(float)
def mat(A,extra):
    M=np.c_[A.Item_MRP.values,oh(A).values]
    if extra: M=np.c_[M,A.Item_Visibility.values,A.Item_Weight.fillna(12.8).values,A.Item_Type.values,A.mrp_item.values]
    return M
def rf(A,ya,B,extra=True,leaf=60):
    return RF(300,min_samples_leaf=leaf,max_features=.6,n_jobs=-1,random_state=0).fit(mat(A,extra),ya).predict(mat(B,extra))
def et(A,ya,B,extra=False,leaf=40):
    return ET(300,min_samples_leaf=leaf,max_features=1.0,n_jobs=-1,random_state=0).fit(mat(A,extra),ya).predict(mat(B,extra))
print("rf",run(rf,2)); print("rf small",run(lambda A,y_,B:rf(A,y_,B,False),2)); print("et",run(et,2))
print("base",run(bl(gb,lin),2))
print("gb+lin+rf",run(bl(gb,lin,rf),2)); print("gb+lin+et",run(bl(gb,lin,et),2)); print("all5",run(bl(gb,lin,rf,et),2))
