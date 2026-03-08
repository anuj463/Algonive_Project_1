import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Retail_Sales_Dataset.csv")

print("First 5 rows of dataset:")
print(df.columns)

# -----------------------------
# 2. Data Cleaning
# -----------------------------

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Create Month column
df["Month"] = df["Date"].dt.month

print("\nDataset after preprocessing:")
print(df.head())

# -----------------------------
# 3. Exploratory Data Analysis
# -----------------------------

# Total revenue
total_revenue = df["Total Amount"].sum()
print("\nTotal Revenue:", total_revenue)

# Revenue by product category
category_sales = df.groupby("Product Category")["Total Amount"].sum()
print("\nSales by Category:")
print(category_sales)

# Sales by gender
gender_sales = df.groupby("Gender")["Total Amount"].sum()
print("\nSales by Gender:")
print(gender_sales)

# Monthly sales
monthly_sales = df.groupby("Month")["Total Amount"].sum()
print("\nMonthly Sales:")
print(monthly_sales)

# -----------------------------
# 4. Data Visualization
# -----------------------------

# Sales by Product Category
plt.figure(figsize=(8,5))
sns.barplot(x=category_sales.index, y=category_sales.values)

plt.title("Sales by Product Category")
plt.xlabel("Category")
plt.ylabel("Revenue")

plt.savefig("category_sales.png")
plt.show()

# Monthly Sales Trend
plt.figure(figsize=(8,5))
monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.savefig("monthly_sales.png")
plt.show()

# Sales by Gender
plt.figure(figsize=(6,4))
sns.barplot(x=gender_sales.index, y=gender_sales.values)

plt.title("Sales by Gender")
plt.xlabel("Gender")
plt.ylabel("Revenue")

plt.savefig("gender_sales.png")
plt.show()

# -----------------------------
# 5. Sales Prediction Model
# -----------------------------

X = df["Month"].values.reshape(-1,1)
y = df["Total Amount"].values

model = LinearRegression()
model.fit(X,y)

# Predict next month revenue
future_month = np.array([[13]])
prediction = model.predict(future_month)

print("\nPredicted revenue for next month:", prediction[0])