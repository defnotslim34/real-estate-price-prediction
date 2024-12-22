import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'C:/Users/azama/aliinfotheory/not me/krisha.csv'
real_estate_data = pd.read_csv(file_path)

# Filling missed data
real_estate_data['price'] = pd.to_numeric(real_estate_data['price'], errors='coerce')
real_estate_data['construction_year'] = pd.to_numeric(real_estate_data['construction_year'], errors='coerce')
real_estate_data = real_estate_data.dropna(subset=['price', 'construction_year'])
real_estate_data = real_estate_data[real_estate_data['price'] > 0]

# Apply a maximum price limit
price_limit = 200_000_000  # 200 million KZT
filtered_data = real_estate_data[real_estate_data['price'] <= price_limit]

# Price Distribution with filtering
plt.figure(figsize=(10, 6))
plt.hist(filtered_data['price'], bins=50, color='skyblue', edgecolor='black', alpha=0.8)
plt.title('Price Distribution (Filtered)', fontsize=16)
plt.xlabel('Price (Millions KZT)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1e6:.1f}M'))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Average price and listing counts
neighborhood_analysis = filtered_data.groupby('district')['price'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

# Top 10 neighborhoods by average price
top_neighborhoods = neighborhood_analysis.head(10)

# Plot top neighborhoods by average price
plt.figure(figsize=(12, 6))
top_neighborhoods['mean'].plot(kind='bar', color='orange', edgecolor='black', alpha=0.9)
plt.title('Top 10 Neighborhoods by Average Price', fontsize=16)
plt.xlabel('Neighborhood', fontsize=14)
plt.ylabel('Average Price (Millions KZT)', fontsize=14)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}M'))
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Price trends over construction years
price_trend = filtered_data.groupby('construction_year')['price'].mean()

# Plot price trend over construction years
plt.figure(figsize=(12, 6))
price_trend.plot(kind='line', color='green', marker='o', linewidth=2, markersize=6)
plt.title('Price Trends Over Construction Years', fontsize=16)
plt.xlabel('Construction Year', fontsize=14)
plt.ylabel('Average Price (Millions KZT)', fontsize=14)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y / 1e6:.1f}M'))
plt.grid(linestyle='--', alpha=0.7)

# Highlight max and min points
max_year = price_trend.idxmax()
max_price = price_trend.max()
min_year = price_trend.idxmin()
min_price = price_trend.min()

# Annotate maximum price
plt.annotate(f'Max: {max_price / 1e6:.1f}M KZT',
             xy=(max_year, max_price),
             xytext=(max_year + 2, max_price + 1e7),
             arrowprops=dict(facecolor='red', arrowstyle='->'), fontsize=12)

# Annotate minimum price
plt.annotate(f'Min: {min_price / 1e6:.1f}M KZT',
             xy=(min_year, min_price),
             xytext=(min_year + 2, min_price + 1e7),
             arrowprops=dict(facecolor='blue', arrowstyle='->'), fontsize=12)

plt.tight_layout()
plt.show()
