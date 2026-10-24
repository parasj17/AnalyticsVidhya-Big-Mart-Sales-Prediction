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
# item target encoding of sales/MRP (nested, smoothed) 
def te_feats(A,ya,B,k=5.0):
    A=A.copy();B=B.copy()
    ratio=ya/A.Item_MRP.values
    gm=ratio.mean()
    iid=f.loc[f.t==1,"Item_Identifier"].reset_index(drop=True)
    ia=iid[A.index].values; ib=iid[B.index].values
    # OOF within A
    oa=np.zeros(len(A))
    for p,q in KFold(5,shuffle=True,random_state=7).split(A):
        d=pd.DataFrame({"i":ia[p],"r":ratio[p]}).groupby("i").r.agg(["sum","count"])
        s_=pd.Series(ia[q]).map(d["sum"]).fillna(0).values; c_=pd.Series(ia[q]).map(d["count"]).fillna(0).values
        oa[q]=(s_+k*gm)/(c_+k)
    d=pd.DataFrame({"i":ia,"r":ratio}).groupby("i").r.agg(["sum","count"])
    s_=pd.Series(ib).map(d["sum"]).fillna(0).values; c_=pd.Series(ib).map(d["count"]).fillna(0).values
    A["ite"]=oa; B["ite"]=(s_+k*gm)/(c_+k)
    return A,B
def gbte(A,ya,B,fs=None,**k):
    A,B=te_feats(A,ya,B); fs=["Item_MRP","Outlet_Identifier","ite"]
    p=dict(learning_rate=.03,max_iter=250,max_depth=3,min_samples_leaf=80,l2_regularization=1.0);p.update(k)
    return H(categorical_features=[c in cats for c in fs],random_state=0,**p).fit(A[fs],ya).predict(B[fs])
print("base",run(bl(gb,lin),2))
print("gb+te",run(gbte,2)); print("gbte+lin",run(bl(gbte,lin),2))
# scaling check
def sc(a):
    return lambda A,ya,B:a*bl(gb,lin)(A,ya,B)
for a in (0.97,1.03): print("scale",a,run(sc(a),2))
# ratio model: predict sales/MRP weighted
def ratio(A,ya,B):
    fs=["Item_MRP","Outlet_Identifier"]
    m=H(categorical_features=[False,True],random_state=0,learning_rate=.03,max_iter=250,max_depth=3,min_samples_leaf=80).fit(A[fs],ya/A.Item_MRP.values,sample_weight=A.Item_MRP.values**2)
    return m.predict(B[fs])*B.Item_MRP.values
print("ratio",run(ratio,2)); print("ratio+gb+lin",run(bl(ratio,gb,lin),2))
