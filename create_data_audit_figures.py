from src.data.clean import load_raw, ordinary_shares, add_returns
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

tables = Path("reports/tables")
output = Path("reports/figures")
output.mkdir(parents=True, exist_ok=True)

# 1. Data coverage by stock
coverage = pd.read_csv(tables / "data_audit_coverage.csv")
coverage = coverage.sort_values("coverage")

plt.figure(figsize=(10, 6))
plt.barh(coverage["CompanyCode"], coverage["coverage"])
plt.xlabel("Coverage")
plt.ylabel("Stock")
plt.title("Data Coverage by Stock")
plt.tight_layout()
plt.savefig(output / "data_coverage.png", dpi=200)
plt.close()

# 2. Zero-return share by stock
zero = coverage.sort_values("zero_return_share")

plt.figure(figsize=(10, 6))
plt.barh(zero["CompanyCode"], zero["zero_return_share"])
plt.xlabel("Zero-return share")
plt.ylabel("Stock")
plt.title("Zero-Return Share by Stock")
plt.tight_layout()
plt.savefig(output / "zero_return_share.png", dpi=200)
plt.close()

# 3. Number of split flags across the full dataset
thresholds = [0.25, 0.35, 0.50]
ordinary = ordinary_shares(load_raw())
counts = []

for threshold in thresholds:
    checked = add_returns(ordinary, split_threshold=threshold)
    counts.append(int(checked["flag_split"].sum()))

plt.figure(figsize=(8, 5))
plt.bar([str(t) for t in thresholds], counts)
plt.xlabel("Split threshold")
plt.ylabel("Flagged records")
plt.title("Threshold Sensitivity Analysis")
plt.tight_layout()
plt.savefig(output / "threshold_sensitivity.png", dpi=200)
plt.close()

print("Figures saved to:", output)
print("Created:")
for name in [
    "data_coverage.png",
    "zero_return_share.png",
    "threshold_sensitivity.png",
]:
    print(output / name)
