"""C7: chronological market RV5 benchmark, fixed validation-only specifications.

GARCH five-day variance uses the fitted conditional variance forecast;
HAR/RF/XGB predict sum of next five squared percent returns.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from arch import arch_model
from src.features.build import load_features
from src.eval.config import WINDOWS, TRAIN_START

T=Path('reports/tables'); P=Path('results/predictions')
FEATURES=['rv1','rv5','rv22']

def qlike(y, v):
    y=np.asarray(y,dtype=float); v=np.maximum(np.asarray(v,dtype=float),1e-8)
    return float(np.mean(np.log(v)+y/v))

def make_data():
    f=load_features(); f=f[f.entity=='MARKET'].sort_index().copy()
    f['rv1']=f.r**2
    for n in (5,22):
        f[f'rv{n}']=f.groupby('seg').rv1.transform(lambda s:s.rolling(n).mean())
    return f

def evaluate(f, window, name):
    w=WINDOWS[window]; start=pd.Timestamp(w['test_start']); end=pd.Timestamp(w['test_end'])
    out=[]
    for year in range(start.year,end.year+1):
        cut=pd.Timestamp(f'{year}-01-01')
        train=f[(f.index>=TRAIN_START)&(f.index<cut)].copy()
        # Purge within each segment/entity BEFORE dropping missing targets.
        train=train.groupby('seg',group_keys=False).apply(lambda x:x.iloc[:-5])
        train=train.dropna(subset=FEATURES+['y_rv_5'])
        test=f[(f.index>=max(cut,start))&(f.index<=min(pd.Timestamp(f'{year}-12-31'),end))].dropna(subset=FEATURES+['y_rv_5'])
        if len(train)<150 or test.empty:continue
        if name=='HAR': model=LinearRegression()
        elif name=='RandomForest':model=RandomForestRegressor(n_estimators=150,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1)
        elif name=='XGBoost':model=XGBRegressor(n_estimators=150,max_depth=3,learning_rate=.05,subsample=.8,random_state=42,n_jobs=1)
        if name!='GARCH':
            model.fit(train[FEATURES],train.y_rv_5)
            pred=np.maximum(model.predict(test[FEATURES]),1e-8)
        else:
            # Fixed yearly fit; filter conditional variance with observed origin returns.
            seg=train.seg.iloc[-1]
            hist=train[train.seg==seg].r.dropna().to_numpy(float)
            if len(hist)<100:continue
            spec=dict(mean='AR',lags=1,vol='GARCH',p=1,o=1,q=1,dist='t',rescale=False)
            fitted=arch_model(hist,**spec).fit(disp='off')
            obs=list(hist);pred=[]
            # Include observations after purged fitting rows up to forecast origin.
            test_start=test.index.min()
            bridge=f[(f.index>train.index.max())&(f.index<test_start)&(f.seg==seg)].r.dropna()
            obs.extend(bridge.to_numpy(float))
            for _,row in test.iterrows():
                if row.seg!=seg:pred.append(np.nan);continue
                obs.append(float(row.r))
                state=arch_model(np.asarray(obs),**spec).fix(fitted.params)
                # Expected sum of five future squared returns = variance + mean^2.
                fc=state.forecast(horizon=5,method='analytic',reindex=False)
                pred.append(max(float((fc.variance.iloc[-1]+fc.mean.iloc[-1]**2).sum()),1e-8))
            pred=np.asarray(pred)
        out.append(pd.DataFrame({'date':test.index,'window':window,'model':name,'fit_year':year,'y_true':test.y_rv_5.to_numpy(),'y_pred':pred}))
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def main():
    T.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
    f=make_data(); pieces=[]
    for w in WINDOWS:
        for name in ('HAR','GARCH','RandomForest','XGBoost'):
            x=evaluate(f,w,name)
            if not x.empty:pieces.append(x)
    if not pieces:raise RuntimeError('No eligible volatility observations')
    predictions=pd.concat(pieces,ignore_index=True)
    predictions.to_csv(P/'chamathka_volatility_predictions.csv',index=False)
    rows=[]
    for (w,m),g in predictions.dropna(subset=['y_true','y_pred']).groupby(['window','model']):
        rows.append({'window':w,'model':m,'n':len(g),'qlike':qlike(g.y_true,g.y_pred),'rmse':float(np.sqrt(np.mean((g.y_true-g.y_pred)**2)))})
    pd.DataFrame(rows).to_csv(T/'chamathka_volatility_comparison.csv',index=False)

if __name__=='__main__':main()
