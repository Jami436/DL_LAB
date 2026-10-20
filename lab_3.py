import pandas as pd

emp = pd.DataFrame({
    "name": ["Ali", "Sara", "Ahmed", "Fatima", "Usman", "Aisha", "Bilal", "Zain"],
    "age": [25, 34, 40, 28, 32, 45, 22, 38],
    "dept": ["IT", "HR", "IT", "HR", "IT", "HR", "IT", "HR"],
    "salary": [65000, 72000, 80000, 60000, 75000, 68000, 55000, 90000],
    "city": ["Karachi", "Lahore", "Islamabad", "Karachi", "Lahore", "Islamabad", "Karachi", "Islamabad"]
})

# 1. Name and age of employees older than 30
older_than_30 = emp.loc[emp["age"] > 30, ["name", "age"]]
print("Employees older than 30:")
print(older_than_30)
print()

# 2. IT employees earning more than 70,000
it_high_earners = emp.loc[(emp["dept"] == "IT") & (emp["salary"] > 70000)]
print("IT employees earning more than 70,000:")
print(it_high_earners)
print()

# 3. Using iloc, last 3 rows and first 2 columns
last_3_first_2 = emp.iloc[-3:, :2]
print("Last 3 rows and first 2 columns (iloc):")
print(last_3_first_2)
print()

# 4. Using isin, employees living in Karachi or Islamabad
karachi_islamabad = emp[emp["city"].isin(["Karachi", "Islamabad"])]
print("Employees living in Karachi or Islamabad:")
print(karachi_islamabad)
print()

# 5. Give 10% raise to HR employees using loc, then print HR rows
emp.loc[emp["dept"] == "HR", "salary"] *= 1.1
hr_rows = emp[emp["dept"] == "HR"]
print("HR rows after 10% raise:")
print(hr_rows)
