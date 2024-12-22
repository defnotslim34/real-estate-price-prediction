import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

file_path = 'C:/Users/azama/aliinfotheory/not me/krisha.csv'
real_estate_data = pd.read_csv(file_path)

# Preprocess data: Drop rows with missing target or feature values
real_estate_data = real_estate_data.dropna(subset=['price'])

# Features and target
features = ['construction_year', 'area', 'room_count', 'district', 'floor', 'floor_count', 'condition']
real_estate_data = real_estate_data.dropna(subset=features)
X = real_estate_data[features]
y = real_estate_data['price']

# Preprocessing pipeline
categorical_features = ['district', 'condition']
numerical_features = ['construction_year', 'area', 'room_count', 'floor', 'floor_count']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline with preprocessing and a linear regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Split data into 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict prices
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mse ** 0.5
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='black')
plt.title('Actual vs Predicted Prices', fontsize=16)
plt.xlabel('Actual Prices', fontsize=14)
plt.ylabel('Predicted Prices', fontsize=14)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# Future price prediction function
def predict_future_prices(year, area, room_count, district, floor, floor_count, condition):
    """Predict future prices given property details."""
    input_data = pd.DataFrame({
        'construction_year': [year],
        'area': [area],
        'room_count': [room_count],
        'district': [district],
        'floor': [floor],
        'floor_count': [floor_count],
        'condition': [condition]
    })
    predicted_price = model_pipeline.predict(input_data)[0]
    return predicted_price

# Predict price for a 3-room house (120 sqm, floor 5/10) built in 2025
future_price = predict_future_prices(2025, 120, 3, 'Есильский р-н', 5, 10, 'new')
print(f"Predicted price for a 3-room house (120 sqm, floor 5/10) built in 2025 in Есильский р-н: {future_price:.2f}")