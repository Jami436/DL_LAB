import pandas as pd

sales = pd.DataFrame({
    "region": ["North", "South", "North", "South",
               "North", "South", "North", "South"],
    "product": ["pen", "pen", "book", "book",
                "pen", "book", "book", "pen"],
    "units": [10, 8, 5, 7, 12, 4, 6, 9],
    "revenue": [200, 160, 750, 1050, 240, 600, 900, 180]
})

print(sales)

# 1. Find the average units per product.
print(sales.groupby("product")["units"].mean())

# 2. Find the region with the highest total revenue using groupby and idxmax().
revenue_by_region = sales.groupby("region")["revenue"].sum()

print(revenue_by_region)
print("Highest revenue region:", revenue_by_region.idxmax())

# 3. Create a summary table with total revenue, average units and order count per region using named aggregation.
summary = sales.groupby("region").agg(
    total_revenue=("revenue", "sum"),
    average_units=("units", "mean"),
    order_count=("units", "count")
)

print(summary)

# 4. Add a column showing each row’s revenue as a percentage of its region’s total using transform("sum").
sales["region_total"] = (
    sales.groupby("region")["revenue"].transform("sum")
)

sales["revenue_percentage"] = (
    sales["revenue"] / sales["region_total"] * 100
)

print(sales[["region", "revenue", "region_total",
             "revenue_percentage"]].round(2))

# 5. Build a pivot table with products as rows, regions as columns and average units as values, filling missing cells with 0 (fill_value=0).

pt = pd.pivot_table(
    sales,
    index="product",
    columns="region",
    values="units",
    aggfunc="mean",
    fill_value=0
)

print(pt)