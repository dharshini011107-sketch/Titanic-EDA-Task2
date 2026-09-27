import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")

# Normalize text
for col in ["Name", "Sex", "Ticket", "Cabin", "Embarked"]:
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
df["Sex"] = df["Sex"].str.lower()
df["Embarked"] = df["Embarked"].str.upper()

# Remove exact duplicates
df = df.drop_duplicates()

# Validate numeric ranges
df.loc[(~df["Age"].between(0, 100)) & df["Age"].notna(), "Age"] = np.nan
df.loc[(df["Fare"] < 0) & df["Fare"].notna(), "Fare"] = np.nan

# Impute Age by passenger class and sex
group_medians = df.groupby(["Pclass", "Sex"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(group_medians)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Impute Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Keep Cabin and make missingness explicit
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Standardize types
for col in ["PassengerId", "Survived", "Pclass", "SibSp", "Parch"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
for col in ["Age", "Fare"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

df.to_csv("titanic_cleaned.csv", index=False)
print("Cleaning complete:", df.shape)
print("Remaining missing values:")
print(df.isna().sum())
