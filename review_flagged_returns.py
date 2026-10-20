
from src.data.clean import load_raw, ordinary_shares, add_returns

df = load_raw()
ordinary = ordinary_shares(df)
checked = add_returns(ordinary, split_threshold=0.35)

flagged = checked.loc[
    checked["flag_split"] & ~checked["flag_gap"],
    ["CompanyCode", "Date"]
]

for _, row in flagged.head(15).iterrows():
    code = row["CompanyCode"]
    date = row["Date"]

    stock = checked[
        checked["CompanyCode"] == code
    ].reset_index(drop=True)

    positions = stock.index[stock["Date"] == date].tolist()

    if not positions:
        continue

    i = positions[0]
    start = max(0, i - 1)
    end = min(len(stock), i + 2)

    print(f"\n{'=' * 55}")
    print(f"Stock: {code} | Flagged date: {date.date()}")

    print(
        stock.loc[
            start:end - 1,
            ["Date", "Close", "lr", "gap_days",
             "flag_gap", "flag_split"]
        ].to_string(index=False)
    )

print("\nReview complete. No source files were changed.")
