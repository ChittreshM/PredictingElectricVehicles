import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
import os

# Define dynamic file paths
base_path = os.path.dirname(os.path.abspath(__file__))
ev_charging_path = os.path.join(base_path, "Dataset 1_EV charging reports.csv")
traffic_path = os.path.join(base_path, "Dataset 6_Local traffic distribution.csv")

# Load datasets with correct delimiter
ev_charging_reports = pd.read_csv(ev_charging_path, sep=";")
traffic_reports = pd.read_csv(traffic_path, sep=";")

# Convert datetime columns and ensure consistency
ev_charging_reports["Start_plugin_hour"] = ev_charging_reports["Start_plugin_hour"].astype(str).str.replace(",", ".")
ev_charging_reports["Start_plugin_hour"] = pd.to_datetime(ev_charging_reports["Start_plugin_hour"], format="%d.%m.%Y %H:%M", errors="coerce")
traffic_reports["Date_from"] = pd.to_datetime(traffic_reports["Date_from"], errors="coerce")

# Debugging: Print fixed datetime values before merging
print("Fixed Start_plugin_hour values:", ev_charging_reports["Start_plugin_hour"].unique()[:5])
print("Fixed Date_from values:", traffic_reports["Date_from"].unique()[:5])
print("Unique Start_plugin_hour values:", ev_charging_reports["Start_plugin_hour"].unique()[:5])
print("Unique Date_from values:", traffic_reports["Date_from"].unique()[:5])

# Round timestamps to nearest hour to improve matching
ev_charging_reports["Start_plugin_hour"] = ev_charging_reports["Start_plugin_hour"].dt.floor("H")
traffic_reports["Date_from"] = traffic_reports["Date_from"].dt.floor("H")
ev_charging_reports["Start_plugin_hour"] = pd.to_datetime(ev_charging_reports["Start_plugin_hour"], errors="coerce")
traffic_reports["Date_from"] = pd.to_datetime(traffic_reports["Date_from"], errors="coerce")

# Debugging: Print initial dataset info
print("EV Charging Reports Shape:", ev_charging_reports.shape)
print("Traffic Reports Shape:", traffic_reports.shape)
print(ev_charging_reports.head())
print(traffic_reports.head())

# Merge datasets and check the shape after merging
ev_charging_traffic = ev_charging_reports.merge(traffic_reports, left_on="Start_plugin_hour", right_on="Date_from", how="inner")
print("Merged Data Shape:", ev_charging_traffic.shape)
print(ev_charging_traffic.head())
if "Date_from" in traffic_reports.columns:
    ev_charging_traffic = ev_charging_reports.merge(traffic_reports, left_on="Start_plugin_hour", right_on="Date_from")
else:
    print("Error: 'Date_from' column not found in traffic_reports")
    exit()

# Drop unnecessary columns
columns_to_drop = ["session_ID", "Garage_ID", "User_ID", "Shared_ID", "Plugin_category", "Duration_category", "Start_plugin", "Start_plugin_hour", "End_plugout", "End_plugout_hour", "Date_from", "Date_to"]
ev_charging_traffic = ev_charging_traffic.drop(columns=columns_to_drop, axis=1, errors="ignore")

# Convert all columns to float
ev_charging_traffic = ev_charging_traffic.apply(pd.to_numeric, errors="coerce")

# Debugging: Check for missing values before dropping
print("Missing values before drop:")
print(ev_charging_traffic.isna().sum())

# Drop NaN values
ev_charging_traffic = ev_charging_traffic.dropna()

# Define features and target
if "El_kWh" in ev_charging_traffic.columns:
    X = ev_charging_traffic.drop(["El_kWh"], axis=1)
    y = ev_charging_traffic["El_kWh"]
else:
    print("Error: 'El_kWh' column not found in dataset")
    exit()

# Debugging: Ensure dataset is not empty before training
print("Final Dataset Shape:", ev_charging_traffic.shape)
print("Columns in Dataset:", ev_charging_traffic.columns)

# Ensure there is data before splitting
if len(X) == 0 or len(y) == 0:
    print("Error: No data available for training.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# PyTorch model setup
input_size = X_train.shape[1]
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
)

# Convert data for PyTorch
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# Define loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Convert test data to tensors
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Load the model
model4500 = torch.load('models/model4500.pth')

# Using the loaded neural network
model4500.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculations
    predictions = model4500(X_test_tensor)  # Generate apartment rent predictions
    test_loss = criterion(predictions, y_test_tensor)  # Calculate testing set MSE loss

print('Neural Network - Test Set MSE:', test_loss.item())  # Print testing set MSE
