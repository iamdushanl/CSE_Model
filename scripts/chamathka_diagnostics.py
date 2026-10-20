"""C2/C5: market training-only Box-Jenkins and yearly GARCH diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from src.features.build import load_features
from src.models.ts.arma import select_orders
from src.models.ts.garch import GarchForecaster

T = Path('reports/tables'); F = Path('reports/figures')

def main():
    T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True)
    df = load_features(); market = df[df.entity == 'MARKET'].sort_index()
    initial = market[(market.index < '2018-01-01') & (market.index >= '2002-01-01')]
    initial = initial[initial.seg == initial.seg.iloc[-1]]
    orders = select_orders(initial, 5, 5)
    orders.to_csv(T/'chamathka_box_jenkins_orders.csv', index=False)
    x = initial.r.dropna()
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    plot_acf(x, lags=30, ax=ax[0]); plot_pacf(x, lags=30, ax=ax[1], method='ywm')
    fig.tight_layout(); fig.savefig(F/'chamathka_market_acf_pacf.png', dpi=180); plt.close(fig)
    params, diag = [], []
    for year in [2018, 2019, 2020, 2021, 2024, 2025]:
        train = market[market.index < f'{year}-01-01'].copy()
        train = train.iloc[:-5] if len(train) > 5 else train
        try:
            model = GarchForecaster().fit(train, 1)
            params.append({'year':year, **model.fit_record_})
            z = np.asarray(model.result_.std_resid, dtype=float)
            z = z[np.isfinite(z)]
            if len(z) < 30: raise ValueError('not enough standardized residuals')
            lb = acorr_ljungbox(z, lags=[10], return_df=True)
            lbsq = acorr_ljungbox(z*z, lags=[10], return_df=True)
            arch = het_arch(z, nlags=5)
            diag.append({'year':year, 'status':'ok', 'lb_p10':float(lb.lb_pvalue.iloc[0]),
                         'lb_sq_p10':float(lbsq.lb_pvalue.iloc[0]), 'arch_lm_p5':float(arch[1])})
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            plot_acf(z, lags=25, ax=axes[0]); plot_acf(z*z, lags=25, ax=axes[1])
            nu = float(model.params_['nu']); n=len(z)
            theoretical = stats.t.ppf((np.arange(n)+0.5)/n, nu)*np.sqrt((nu-2)/nu)
            axes[2].scatter(theoretical, np.sort(z), s=5)
            axes[2].set_title('Student-t QQ')
            fig.tight_layout(); fig.savefig(F/f'chamathka_garch_diagnostics_{year}.png',dpi=180);plt.close(fig)
        except Exception as exc:
            diag.append({'year':year, 'status':'failed', 'reason':repr(exc)})
    pd.DataFrame(params).to_csv(T/'chamathka_garch_parameter_drift.csv',index=False)
    pd.DataFrame(diag).to_csv(T/'chamathka_garch_diagnostics.csv',index=False)
    if params:
        p = pd.DataFrame(params).set_index('year')
        cols = [c for c in ('omega','alpha[1]','gamma[1]','beta[1]','nu') if c in p]
        p[cols].plot(subplots=True, figsize=(9,11), title='Yearly parameter drift')
        plt.tight_layout();plt.savefig(F/'chamathka_garch_parameter_drift.png',dpi=180);plt.close()

if __name__ == '__main__': main()
