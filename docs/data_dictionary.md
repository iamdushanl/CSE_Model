# Data dictionary (Member 3 owns and extends this)
Source file: `data/raw/SP20_CSE_Dataset_2001_2025_Adjusted.xlsx`, sheet `Full_Dataset_2001_2025`
(86,950 rows, 20 companies, 2002-01-02 to 2025-12-31). Never edit the raw file.

| Column | Meaning | Trust? |
| --- | --- | --- |
| Date, CompanyCode | Trading day and company | Yes |
| MainType | Security class: N ordinary, X non-voting, W warrant, R rights, D debenture, P preference, B other | Yes. We keep N only |
| Close, High, Low | Prices | Close yes for N rows. Splits are NOT adjusted. High and Low have a few missing values |
| Open | Opening price | No: 60% missing |
| TradeVolume, ShareVolume, Turnover | Activity | Use with N rows only |
| DailyReturn_Pct, MA_30d, MA_90d, Volatility_30d_Ann, PriceRange | Pre-computed by the data provider across mixed share classes | **No. Do not use** |
| ShortName, SubType, Era, CompanyName_Official, Year, Month, Quarter, DayOfWeek | Labels | Not needed |

Processed files (built by `python run.py data`, not committed): `data/processed/returns_clean.parquet`,
`market_proxy.parquet`, `features.parquet`. Column definitions for the feature table: `docs/features.md`.

### Processed dataset columns

Processed files are built by `python run.py data` and are not committed to Git.

**`returns_clean.parquet`**
- `CompanyCode`, `Date`: company identifier and trading date.
- `Close`: closing price.
- `lr`: calculated log return.
- `lr_clean`: cleaned log return used in subsequent processing.
- `flag_gap`: indicates a return affected by a trading-date gap.
- `flag_split`: indicates a return flagged by the split-detection threshold.

**`market_proxy.parquet`**
- `MARKET`: equal-weighted market return proxy, calculated when at least eight valid stock returns are available.

**`features.parquet`**
- Contains the date, entity, segment, input features, market proxy feature, and future target columns.
- See `docs/features.md` for the feature definitions.
- Future target columns (`y_ret_1`, `y_dir_1`, `y_ret_5`, `y_dir_5`, and `y_rv_5`) must not be used as model input features.