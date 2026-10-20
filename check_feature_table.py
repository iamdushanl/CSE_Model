
import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")

print("COLUMNS:")
for col in df.columns:
    print(repr(col))

print("\nSHAPE:", df.shape)

print(
    "\nDUPLICATE (date, entity) ROWS:",
    df.duplicated(["date", "entity"]).sum()
)

target_cols = [
    "y_ret_1", "y_ret_5", "y_dir_1", "y_dir_5", "y_rv_5"
]

print("\nMISSING VALUES:")
print(df[["r", "mkt"] + target_cols].isna().sum())

print("\nTARGET SUMMARY:")
print(df[target_cols].describe().to_string())

print("\nDIRECTION TARGET VALUES:")
for col in ["y_dir_1", "y_dir_5"]:
    print(col, sorted(df[col].dropna().unique().tolist()))


import pandas as pd

df = pd.read_parquet("data/processed/features.parquet")
df = df.sort_values(["entity", "seg", "date"])

print("MISSING VALUES BY SEGMENT")
for col in ["mkt", "y_dir_1", "y_dir_5"]:
    print(f"\n{col}:")
    print(
        df.assign(missing=df[col].isna())
        .groupby(["entity", "seg"])["missing"]
        .agg(["sum", "count"])
        .query("sum > 0")
        .head(20)
        .to_string()
    )

print("\nFIRST 15 ROWS WITH MISSING MARKET RETURNS")
cols = ["date", "entity", "seg", "r", "mkt"]
print(
    df.loc[df["mkt"].isna(), cols]
    .head(15)
    .to_string(index=False)
)

print("\nFIRST 15 ROWS WITH MISSING ONE-DAY DIRECTION TARGET")
cols = ["date", "entity", "seg", "y_dir_1"]
print(
    df.loc[df["y_dir_1"].isna(), cols]
    .head(15)
    .to_string(index=False)
)
