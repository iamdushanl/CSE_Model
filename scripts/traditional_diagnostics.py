"""Yearly GJR-GARCH diagnostics; requires processed features and arch."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from src.features.build import load_features
from src.models.ts.garch import GarchForecaster


def run():
    tables, figures = Path('reports/tables'), Path('reports/figures')
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    market = load_features().query("entity == 'MARKET'").sort_index()
    params, checks = [], []
    for year in range(2018, 2026):
        train = market.loc[market.index < pd.Timestamp(f'{year}-01-01')].iloc[:-5]
        if len(train) < 100: continue
        try:
            model = GarchForecaster().fit(train, 1)
            record = dict(model.fit_record_); record['fit_year'] = year
            params.append(record)
            z = np.asarray(model.result_.std_resid, float)
            z = z[np.isfinite(z)]
            if len(z) < 30: continue
            lb = acorr_ljungbox(z, lags=[5, 10, 20], return_df=True)
            lbs = acorr_ljungbox(z*z, lags=[5, 10, 20], return_df=True)
            checks.append({'year': year, 'n':len(z), 'lb_p_10':float(lb.loc[10,'lb_pvalue']), 'lb_squared_p_10':float(lbs.loc[10,'lb_pvalue']), 'arch_lm_p':float(het_arch(z,nlags=5)[1]), 'convergence':model.convergence_})
            for suffix, arr in [('residual_acf', z), ('squared_acf', z*z)]:
                fig, ax = plt.subplots(figsize=(8,4)); plot_acf(arr, lags=min(30,len(arr)//3), ax=ax)
                fig.tight_layout(); fig.savefig(figures/f'garch_{suffix}_{year}.png',dpi=170); plt.close(fig)
            nu = float(model.params_['nu'])
            theoretical = t.ppf((np.arange(len(z))+.5)/len(z),nu)*np.sqrt((nu-2)/nu)
            fig, ax = plt.subplots(figsize=(6,5)); ax.scatter(theoretical,np.sort(z),s=6)
            ax.set(xlabel='Student-t theoretical quantile',ylabel='Standardized residual',title=f'GARCH QQ {year}')
            fig.tight_layout();fig.savefig(figures/f'garch_qq_{year}.png',dpi=170);plt.close(fig)
        except (ValueError, RuntimeError) as exc:
            checks.append({'year':year,'error':str(exc)})
    pd.DataFrame(params).to_csv(tables/'garch_parameter_drift.csv',index=False)
    pd.DataFrame(checks).to_csv(tables/'garch_diagnostics.csv',index=False)
    if params:
        p = pd.DataFrame(params).set_index('fit_year')
        cols = [c for c in ('omega','alpha[1]','gamma[1]','beta[1]','nu') if c in p]
        if cols:
            axes = p[cols].plot(subplots=True,figsize=(9,11),marker='o',legend=False)
            axes[0].figure.tight_layout();axes[0].figure.savefig(figures/'garch_parameter_drift.png',dpi=170);plt.close(axes[0].figure)

if __name__ == '__main__':run()
