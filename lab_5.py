import pandas as pd
import numpy as np

df = pd.DataFrame({
    "price": [100.00, 250.50, 80.00, 120.75, 200.00, 150.00, 90.00, 180.00],
    "marks": [85, 65, 45, 90, 30, 75, 55, 88],
    "attended": [1, 0, 1, 1, 0, 1, 0, 1],
    "date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-08", "2023-05-21", "2023-06-10", "2023-07-15", "2023-08-20"],
    "age": [12, 25, 40, 55, 70, 18, 33, 65]
})

# 1. Add discounted column: 10% off price, rounded to 2 decimals
df["discounted"] = (df["price"] * 0.9).round(2)
print("DataFrame with discounted column:")
print(df)
print()

# 2. Use np.select to create grades A/B/C/F from marks
conditions = [
    df["marks"] >= 80,
    df["marks"] >= 60,
    df["marks"] >= 40
]
grades = ["A", "B", "C"]
df["grade"] = np.select(conditions, grades, default="F")
print("Grades using np.select:")
print(df[["marks", "grade"]])
print()

# 3. Use map to convert {0: "No", 1: "Yes"} in attended column
df["attended_label"] = df["attended"].map({0: "No", 1: "Yes"})
print("Attended column mapped:")
print(df[["attended", "attended_label"]])
print()

# 4. Convert date strings to dates; extract year, month, weekend flag
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["is_weekend"] = df["date"].dt.dayofweek >= 5
print("Date column with year, month, weekend:")
print(df[["date", "year", "month", "is_weekend"]])
print()

# 5. Use pd.cut to group 8 ages into young/middle/old; count with value_counts()
age_bins = [0, 30, 55, 100]
age_labels = ["young", "middle", "old"]
df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels)
print("Age groups counts:")
print(df["age_group"].value_counts())
