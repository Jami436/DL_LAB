import pandas as pd

# 1. Create two tables (employees and departments) that share dept_id and merge them with an inner join.
employees = pd.DataFrame({
    "emp_id": [101, 102, 103, 104],
    "name": ["Ali", "Sara", "Ahmed", "Zoya"],
    "dept_id": [1, 2, 1, 3]
})

departments = pd.DataFrame({
    "dept_id": [1, 2, 4],
    "dept_name": ["AI", "Civil", "IT"]
})

result = pd.merge(employees, departments, on="dept_id", how="left")
print(result)

# 2. Repeat with a left join and find which rows have no match (rows containing NaN).
result = pd.merge(employees, departments, on="dept_id", how="left")
print(result)

unmatched = result[result["dept_name"].isna()]
print("\nEmployees without a matching department:")
print(unmatched)

# 3. Use indicator=True with an outer join and count how many rows are left_only.
result = pd.merge(employees, departments, on="dept_id", how="outer", indicator=True)
print(result)

left_count = (result["_merge"] == "left_only").sum()
print(f"\nNumber of rows that are left_only: {left_count}")

# 4. Concatenate three small monthly DataFrames into one and find the total sold per item.
jan = pd.DataFrame({
    "item": ["pen", "book"],
    "sold": [10, 5]
})

feb = pd.DataFrame({
    "item": ["pen", "bag"],
    "sold": [8, 7]
})

mar = pd.DataFrame({
    "item": ["book", "bag"],
    "sold": [5, 6]
})

# combine the three DataFrames
all_sales = pd.concat([jan, feb, mar], ignore_index=True)

print("\nCombined sales data:")
print(all_sales)

# calculate total sold per item
totals = all_sales.groupby("item")["sold"].sum()

print("\nTotal sold per item:")
print(totals)

# 5. Merge two tables whose key columns have different names (id and student_id). (Hint: left_on="id", right_on="student_id".)
students = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Ali", "Taha", "Zoya"]
})

marks = pd.DataFrame({
    "Student_id": [1, 2, 4],
    "marks": [85, 90, 75]
})

result = pd.merge(students, marks, left_on="id", right_on="Student_id", how="left")
print("\nMerged students and marks:")
print(result)