import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("IBM HR Analytics Dataset")
print("=" * 60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Attrition"].value_counts())

print("\n" + "=" * 60)
print("ASSIGNMENT REQUIREMENT CHECK")
print("=" * 60)

print(f"Total Instances : {df.shape[0]}")
print(f"Total Features  : {df.shape[1] - 1}")  # excluding target
print(f"Target Column   : Attrition")

if df.shape[0] >= 500:
    print("✓ Minimum 500 instances satisfied")
else:
    print("✗ Dataset too small")

if (df.shape[1] - 1) >= 12:
    print("✓ Minimum 12 features satisfied")
else:
    print("✗ Not enough features")