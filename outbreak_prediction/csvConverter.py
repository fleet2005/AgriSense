import pandas as pd

# Load dataset and skip metadata rows
file_path = "C:/Users/nothi/OneDrive/Desktop/Prediction/Backend/FutureWeatherPrediction.csv"
df = pd.read_csv(file_path)

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Extract only the first occurrence per day
df_daily = df.groupby(df['time'].dt.date).first()

# Rename index to 'date' (so it doesn't conflict with the 'time' column)
df_daily.index.name = 'date'

# Reset index properly
df_daily = df_daily.reset_index()

# Save processed data
output_path = "processed_FutureWeatherPrediction.csv"
df_daily.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
