import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# API URL
url = "https://api.open-meteo.com/v1/forecast"

# Latitude and Longitude for Manipur
latitude = 24.639717
longitude = 93.95075

# Request parameters for 7-day forecast with required weather variables
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "rain", 
        "wind_speed_100m", "soil_temperature_7_to_28cm", "soil_moisture_7_to_28cm"
    ],
    "forecast_days": 7,
    "timezone": "Asia/Kolkata"
}

# Fetch weather data
responses = openmeteo.weather_api(url, params=params)
response = responses[0]  # Process the first location response

# Extract hourly data
hourly = response.Hourly()

time_series = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)

hourly_data = {
    "time": time_series,
    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
    "rain": hourly.Variables(2).ValuesAsNumpy(),
    "wind_speed_100m": hourly.Variables(3).ValuesAsNumpy(),
    "soil_temperature_7_to_28cm": hourly.Variables(4).ValuesAsNumpy(),
    "soil_moisture_7_to_28cm": hourly.Variables(5).ValuesAsNumpy()
}

# Create DataFrame
future_weather_df = pd.DataFrame(data=hourly_data)

# Save to CSV
future_weather_df.to_csv("FutureWeatherPrediction.csv", index=False)

print("Future weather data saved as FutureWeatherPrediction.csv")
