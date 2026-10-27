import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n" + "=" * 55)
print("QUESTION 1: ONE-HOT ENCODING")
print("=" * 55)

colours = pd.DataFrame({
    "colour": ["red", "green", "blue", "red", "green", "blue"]
})

encoded_colours = pd.get_dummies(
    colours, columns=["colour"], dtype=int
)

print("\nOriginal Data:")
print(colours)

print("\nOne-Hot Encoded Data:")
print(encoded_colours)

# Visualization 1
encoded_colours.plot(kind="bar", figsize=(8, 4))
plt.title("Q1: One-Hot Encoding of Colours")
plt.xlabel("Sample")
plt.ylabel("Encoded Value")
plt.xticks(rotation=0)
plt.legend(title="Colour")
plt.tight_layout()
plt.show()


# ============================================================
# PRACTICE QUESTION 2: MIN-MAX SCALING
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 2: MIN-MAX SCALING")
print("=" * 55)

marks = pd.DataFrame({
    "marks": [20, 40, 60, 80, 100]
})

marks["scaled"] = (
    (marks["marks"] - marks["marks"].min())
    / (marks["marks"].max() - marks["marks"].min())
)

print("\nOriginal and Scaled Marks:")
print(marks)

print("\nMinimum:", marks["scaled"].min())
print("Maximum:", marks["scaled"].max())

# Visualization 2
plt.figure(figsize=(8, 4))
plt.plot(
    marks["marks"], marks["scaled"],
    marker="o", label="Scaled marks"
)
plt.title("Q2: Min-Max Scaling")
plt.xlabel("Original Marks")
plt.ylabel("Scaled Value (0 to 1)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PRACTICE QUESTION 3: TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 3: DATASET SPLITTING")
print("=" * 55)

df = pd.DataFrame({
    "student_id": range(1, 21),
    "marks": range(50, 70)
})

# Shuffle the rows reproducibly
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

n = len(df)
n_train = int(0.70 * n)
n_val = int(0.15 * n)

train = df.iloc[:n_train]
validation = df.iloc[n_train:n_train + n_val]
test = df.iloc[n_train + n_val:]

print("\nOriginal shape:", (20, 2))
print("Training shape:", train.shape)
print("Validation shape:", validation.shape)
print("Testing shape:", test.shape)

# Visualization 3
split_sizes = [len(train), len(validation), len(test)]
split_labels = ["Training", "Validation", "Testing"]

plt.figure(figsize=(7, 5))
plt.pie(
    split_sizes,
    labels=split_labels,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Q3: Train / Validation / Test Split")
plt.tight_layout()
plt.show()


# ============================================================
# PRACTICE QUESTION 4: WHY USE TRAINING STATISTICS?
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 4: DATA LEAKAGE")
print("=" * 55)

print("""
1. Calculate the mean and standard deviation using only
   training data to prevent data leakage.

2. Use those same training statistics to scale validation
   and test data.

3. This ensures fair evaluation on unseen data, similar
   to predictions in the real world.
""")

# Small demonstration
training_values = pd.Series([80, 100, 120, 140, 160])
test_value = 180

train_mean = training_values.mean()
train_std = training_values.std()

scaled_test_value = (test_value - train_mean) / train_std

print("Training mean:", train_mean)
print("Training standard deviation:", round(train_std, 2))
print("Test value:", test_value)
print("Scaled test value:", round(scaled_test_value, 2))

# Visualization 4
scaled_training = (training_values - train_mean) / train_std

plt.figure(figsize=(8, 4))
plt.hist(scaled_training, bins=5, edgecolor="black")
plt.axvline(0, linestyle="--", label="Mean = 0")
plt.title("Q4: Training Data After Standardization")
plt.xlabel("Standardized Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# PRACTICE QUESTION 5: COMPLETE DEEP LEARNING PIPELINE
# Clean -> Encode -> Split -> Scale -> NumPy
# ============================================================

print("\n" + "=" * 55)
print("QUESTION 5: COMPLETE DATA PREPARATION PIPELINE")
print("=" * 55)

# STEP 1: Create our own dataset
houses = pd.DataFrame({
    "area": [
        850, 1200, 1500, np.nan, 2000,
        1100, 1750, 900, 1350, 1600
    ],
    "bedrooms": [2, 3, 3, 2, 4, 3, 4, 2, 3, 4],
    "city": [
        "Karachi", "Lahore", "Karachi", "Islamabad",
        "Lahore", "Karachi", "Islamabad", "Lahore",
        "Karachi", "Lahore"
    ],
    "price": [45, 75, 90, 60, 130, 68, 120, 48, 82, 105]
})

print("\nOriginal House Dataset:")
print(houses)

# STEP 2: Clean missing values
houses["area"] = houses["area"].fillna(houses["area"].median())

print("\nMissing values after cleaning:")
print(houses.isnull().sum())

# STEP 3: Separate features and target
X = houses.drop(columns="price")
y = houses["price"]

# Encode city into numeric columns
X = pd.get_dummies(X, columns=["city"], dtype=int)

print("\nEncoded Features:")
print(X.head())

# STEP 4: Shuffle and split into 80% train, 20% test
idx = X.sample(frac=1, random_state=42).index
n_train = int(0.8 * len(X))

X_train = X.loc[idx[:n_train]].copy()
X_test = X.loc[idx[n_train:]].copy()

y_train = y.loc[idx[:n_train]].copy()
y_test = y.loc[idx[n_train:]].copy()

# STEP 5: Standardize using training statistics only
numeric_columns = ["area", "bedrooms"]

mean = X_train[numeric_columns].mean()
std = X_train[numeric_columns].std()

# Avoid division by zero if a column has no variation
std = std.replace(0, 1)

X_train.loc[:, numeric_columns] = (
    X_train[numeric_columns] - mean
) / std

X_test.loc[:, numeric_columns] = (
    X_test[numeric_columns] - mean
) / std

# STEP 6: Convert data to NumPy float32 arrays
X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)

y_train_np = y_train.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)

# STEP 7: Print final shapes and data types
print("\nFinal NumPy Results:")
print("X_train:", X_train_np.shape, X_train_np.dtype)
print("y_train:", y_train_np.shape, y_train_np.dtype)
print("X_test: ", X_test_np.shape, X_test_np.dtype)
print("y_test: ", y_test_np.shape, y_test_np.dtype)

print("\nMissing values in X_train:", np.isnan(X_train_np).any())
print("Missing values in X_test:", np.isnan(X_test_np).any())

# Visualization 5A: Area versus house price
plt.figure(figsize=(8, 5))

train_indices = X_train.index
test_indices = X_test.index

plt.scatter(
    houses.loc[train_indices, "area"],
    y_train,
    label="Training data",
    marker="o"
)

plt.scatter(
    houses.loc[test_indices, "area"],
    y_test,
    label="Test data",
    marker="x"
)

plt.title("Q5: House Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price (dataset units)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualization 5B: Training feature distributions after scaling
X_train_df = pd.DataFrame(
    X_train_np,
    columns=X_train.columns
)

X_train_df[["area", "bedrooms"]].plot(
    kind="box",
    figsize=(8, 5)
)

plt.title("Q5: Standardized Training Features")
plt.ylabel("Standardized Value")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 55)
print("LAB 8 COMPLETED SUCCESSFULLY!")
print("All 5 questions and visualizations are complete.")
print("=" * 55)
