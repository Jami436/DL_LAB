import pandas as pd
import numpy as np

# 1. DataFrame with 3 missing values; count missing per column
df = pd.DataFrame({
    "name": ["Ali", None, "Ahmed", "Fatima", None],
    "age": [25, 34, None, 28, 32],
    "salary": [65000, None, 80000, 60000, np.nan],
    "city": ["Karachi", "Lahore", None, "Karachi", "Islamabad"]
})
print("Missing values per column:")
print(df.isnull().sum())
print()

# 2. Drop all rows with any missing value; drop rows where specific column is missing
df_drop_all = df.dropna()
df_drop_age = df.dropna(subset=["age"])
print("Rows after dropping all rows with any missing value:")
print(df_drop_all)
print()
print("Rows after dropping rows where 'age' is missing:")
print(df_drop_age)
print()

# 3. Fill missing numeric values with median; missing text with "Unknown"
numeric_cols = df.select_dtypes(include="number").columns
text_cols = df.select_dtypes(exclude="number").columns
df_filled = df.copy()
for col in numeric_cols:
    df_filled[col] = df_filled[col].fillna(df_filled[col].median())
for col in text_cols:
    df_filled[col] = df_filled[col].fillna("Unknown")
print("DataFrame after filling missing values:")
print(df_filled)
print()

# 4. DataFrame with a duplicated row; detect and remove it
df_dup = pd.DataFrame({
    "name": ["Ali", "Sara", "Ali", "Ahmed"],
    "age": [25, 34, 25, 40],
    "city": ["Karachi", "Lahore", "Karachi", "Islamabad"]
})
print("Duplicated rows:")
print(df_dup[df_dup.duplicated(keep=False)])
df_dup = df_dup.drop_duplicates()
print("DataFrame after removing duplicates:")
print(df_dup)
print()

# 5. Column with strings "10", "20", "x", "40"; convert to numbers, replace bad with mean
df5 = pd.DataFrame({"value": ["10", "20", "x", "40"]})
df5["value_num"] = pd.to_numeric(df5["value"], errors="coerce")
mean_val = df5["value_num"].mean()
df5["value_num"] = df5["value_num"].fillna(mean_val)
print("Converted column with bad value replaced by mean:")
print(df5)
