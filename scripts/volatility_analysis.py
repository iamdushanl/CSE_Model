"""C7 preliminary chronological HAR/RF/XGB variance benchmark.

GARCH variance comparison requires matching forecasts from arch; this script
refuses to mislabel a proxy as GARCH.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.features.build import load_features
from src.eval.config import WINDOWS

FEATURES = ['rv1','rv5','rv22']

def qlike(y, forecast):
    y = np.asarray(y,float); forecast = np.asarray(forecast,float)
    if not (np.isfinite(y).all() and np.isfinite(forecast).all() and (forecast>0).all() and (y>=0).all()):
        raise ValueError('QLIKE requires nonnegative realized variance and strictly positive forecasts')
    return float(np.mean(np.log(forecast) + y/forecast))

def prepare():
    df = load_features().query("entity == 'MARKET'").sort_index().copy()
    df['rv1'] = df.r**2
    for k in (5,22):
        df[f'rv{k}'] = df.groupby('seg').rv1.transform(lambda x: x.rolling(k,min_periods=k).mean())
    return df

def make_model(name):
    if name == 'HAR':return LinearRegression()
    if name == 'RandomForest':return RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1)
    if name == 'XGBoost':return XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.8,random_state=42,n_jobs=1)
    raise ValueError(name)

def run():
    df = prepare(); out = Path('reports/tables'); out.mkdir(parents=True,exist_ok=True)
    predictions=[]
    for window, bounds in WINDOWS.items():
        start,end = pd.Timestamp(bounds['test_start']),pd.Timestamp(bounds['test_end'])
        for year in range(start.year,end.year+1):
            cut=pd.Timestamp(f'{year}-01-01')
            # Purge the last five *raw* observations before target filtering.
            train=df[df.index<cut].iloc[:-5].dropna(subset=FEATURES+['y_rv_5'])
            test=df[(df.index>=max(cut,start))&(df.index<=min(end,pd.Timestamp(f'{year}-12-31')))].dropna(subset=FEATURES+['y_rv_5'])
            if len(train)<100 or test.empty:continue
            for name in ('HAR','RandomForest','XGBoost'):
                model=make_model(name);model.fit(train[FEATURES],train.y_rv_5)
                forecast=np.maximum(model.predict(test[FEATURES]),1e-8)
                predictions.extend({'window':window,'fit_year':year,'date':date,'model':name,'y_true':float(y),'y_pred':float(p)} for date,y,p in zip(test.index,test.y_rv_5,forecast))
    if not predictions:raise RuntimeError('No volatility forecasts; run python run.py data first')
    pred=pd.DataFrame(predictions)
    pred.to_csv(out/'volatility_predictions.csv',index=False)
    summary=[{'window':w,'model':m,'n':len(g),'qlike':qlike(g.y_true,g.y_pred),'rmse':float(np.sqrt(np.mean((g.y_true-g.y_pred)**2)))} for (w,m),g in pred.groupby(['window','model'])]
    pd.DataFrame(summary).to_csv(out/'volatility_comparison.csv',index=False)
    print('Preliminary HAR/RF/XGB results only: GARCH variance benchmark not yet integrated.')

if __name__=='__main__':run()
