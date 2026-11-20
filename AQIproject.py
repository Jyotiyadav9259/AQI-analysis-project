import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

file_path = "enriched_aqi_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print("Shape (rows, cols):", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()
print("Shape after removing duplicates:", df.shape)

numeric_cols = [
    "latitude",
    "longitude",
    "AQI",
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Day_of_Week",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Pollutant columns (agar multiple hain) ko bhi numeric mein convert karein
pollutant_cols = [c for c in df.columns if c.lower().startswith("pollutant")]
for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Exploratory Data Analysis (EDA)
valid_aqi = df.dropna(subset=["AQI"]).copy()

print("\n===== OVERALL AQI SUMMARY =====")
print("Average AQI:", round(valid_aqi["AQI"].mean(), 2))
print("Highest AQI:", valid_aqi["AQI"].max())
print("Lowest AQI:", valid_aqi["AQI"].min())
print("Median AQI:", round(valid_aqi["AQI"].median(), 2))

if not valid_aqi.empty:
    max_row = valid_aqi.loc[valid_aqi["AQI"].idxmax()]
    min_row = valid_aqi.loc[valid_aqi["AQI"].idxmin()]
    print(f"Highest AQI Reading: {max_row.get('station', '')} ({max_row.get('city', '')}, {max_row.get('state', '')}) -> {max_row['AQI']}")
    print(f"Lowest AQI Reading: {min_row.get('station', '')} ({min_row.get('city', '')}, {min_row.get('state', '')}) -> {min_row['AQI']}")

# State-wise average AQI
state_aqi = valid_aqi.groupby("state")["AQI"].mean().sort_values(ascending=False)
print("\n===== TOP 10 STATES BY AVERAGE AQI =====")
print(state_aqi.head(10).round(2))

# City-wise average AQI
city_aqi = valid_aqi.groupby("city")["AQI"].mean().sort_values(ascending=False)
print("\n===== TOP 10 CITIES BY AVERAGE AQI =====")
print(city_aqi.head(10).round(2))

# AQI_Bucket distribution
if "AQI_Bucket" in valid_aqi.columns:
    print("\n===== AQI BUCKET COUNTS =====")
    print(valid_aqi["AQI_Bucket"].value_counts())

# Weather & AQI correlation
weather_cols = [c for c in ["AQI", "Temperature", "Humidity", "Wind_Speed"] if c in valid_aqi.columns]
if len(weather_cols) > 1:
    print("\n===== CORRELATION (AQI & WEATHER) =====")
    print(valid_aqi[weather_cols].corr().round(2))

# 5. Visualizations
# Graph 1: AQI Distribution Histogram
plt.figure(figsize=(10, 5))
plt.hist(valid_aqi["AQI"], bins=20, edgecolor="black")
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("1_aqi_histogram.png")
plt.show()

# Graph 2: Top 10 States by Average AQI
plt.figure(figsize=(12, 6))
sns.barplot(x=state_aqi.head(10).values, y=state_aqi.head(10).index)
plt.title("Top 10 States with Highest Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("2_top10_states_aqi.png")
plt.show()

# Graph 3: Top 10 Cities by Average AQI
plt.figure(figsize=(12, 6))
sns.barplot(x=city_aqi.head(10).values, y=city_aqi.head(10).index)
plt.title("Top 10 Cities with Highest Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("3_top10_cities_aqi.png")
plt.show()

# Graph 4: AQI_Bucket Countplot
if "AQI_Bucket" in valid_aqi.columns:
    plt.figure(figsize=(10, 5))
    order = valid_aqi["AQI_Bucket"].value_counts().index
    sns.countplot(data=valid_aqi, x="AQI_Bucket", order=order)
    plt.title("Count of Readings by AQI Bucket")
    plt.xlabel("AQI Bucket")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("4_aqi_bucket_count.png")
    plt.show()

# Graph 5: Season-wise Average AQI (agar Season column hai)
if "Season" in valid_aqi.columns:
    season_aqi = valid_aqi.groupby("Season")["AQI"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=season_aqi.index, y=season_aqi.values)
    plt.title("Average AQI by Season")
    plt.xlabel("Season")
    plt.ylabel("Average AQI")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("5_season_aqi.png")
    plt.show()

    # LINE GRAPH: Average AQI Trend
x_col: str | None = None
for col in ("Hour", "Month", "Day"):
    if col in valid_aqi.columns and valid_aqi[col].nunique() > 1:
        x_col = col
        break

if x_col is not None and "AQI" in valid_aqi.columns:
    line_data = valid_aqi.groupby(x_col)["AQI"].mean().dropna().sort_index()
    plt.figure(figsize=(10, 5))
    plt.plot(line_data.index, line_data.values, marker="o")
    plt.title(f"Average AQI by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel("Average AQI")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("aqi_line_graph.png")
    plt.show()

print("\nCode successfully executed!")