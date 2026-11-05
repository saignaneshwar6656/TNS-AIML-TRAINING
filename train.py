
# Step 1: Import Necessary Packages
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
df = pd.read_csv("dataset/amazon_ecommerce_1M.csv")
print(df.head())
print("Dataset Shape:", df.shape)
print("Column Names:", df.columns.tolist())

# Step 3: Data Preprocessing
df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
df = df.dropna(subset=["purchase_date", "final_price"])
df = df[df["is_returned"] == False].copy()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

# Step 4: Calculate Monthly Sales
df["month"] = df["purchase_date"].apply(lambda x: str(x)[:7])
monthly_sales = df.groupby("month")["final_price"].sum().reset_index()
monthly_sales["month"] = pd.to_datetime(monthly_sales["month"])
print("\nMonthly Sales:")
print(monthly_sales)

# Step 5: Feature Selection
monthly_sales["year"] = monthly_sales["month"].apply(lambda x: x.year)
monthly_sales["month_number"] = monthly_sales["month"].apply(lambda x: x.month)
X = monthly_sales[["year", "month_number"]]
y = monthly_sales["final_price"]
print("\nFeatures:")
print(X.head())
print("\nTarget:")
print(y.head())

# Step 6: Split the Dataset
split_index = int(len(X) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

# Step 7: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# Step 8: Prediction, Evaluation and Model Saving
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation Results")
print("Actual Sales:", y_test.values)
print("Predicted Sales:", y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

model_data = {
    "model": model,
    "features": X_train.columns.tolist()
}
joblib.dump(model_data, "sales_model.pkl")
print("\nModel and feature names saved successfully!")
print("Feature Names:", model_data["features"])
print("Model training completed successfully!")
