import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt




# Load dataset
data = pd.read_csv("Advertising.csv")

# Display first 5 rows
print(data.head())

# Dataset information
print("\nDataset Information:")
print(data.info())

# Statistical summary
print("\nSummary Statistics:")
print(data.describe())

# Features (TV advertising)
X = data[['TV']]

# Target (Sales)
y = data['Sales']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")

y_pred = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(y_pred[:5])

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

print("\nLinear Regression Equation")
print(f"Sales = {model.intercept_:.4f} + ({model.coef_[0]:.4f} × TV)")


#scatter plot
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test)

plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("Scatter Plot")

plt.savefig("scatter_plot.png")
plt.show()

#Regression Line Graph
plt.figure(figsize=(8,5))

plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, label="Regression Line")

plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")

plt.legend()

plt.savefig("regression_line.png")
plt.show()

#Actual vs Predicted Comparison
comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

comparison = comparison.reset_index(drop=True)

plt.figure(figsize=(10,5))
plt.plot(comparison['Actual'], label='Actual')
plt.plot(comparison['Predicted'], label='Predicted')

plt.title("Actual vs Predicted Sales")
plt.xlabel("Data Points")
plt.ylabel("Sales")

plt.legend()

plt.savefig("prediction_comparison.png")
plt.show()

