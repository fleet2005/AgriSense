import pandas as pd
from datetime import datetime

class DiseaseOutbreakPredictor:
    def __init__(self, future_weather_path, past_weather_path, crop_susceptibility_path,
                 disease_virality_path, city_weightage_path, month_weightage_path):
        # Load all datasets
        self.future_weather = pd.read_csv(future_weather_path)
        self.past_weather = pd.read_csv(past_weather_path)
        self.crop_susceptibility = pd.read_csv(crop_susceptibility_path)
        self.disease_virality = pd.read_csv(disease_virality_path)
        
        # Load city_weightage and strip leading/trailing spaces from column names
        self.city_weightage = pd.read_csv(city_weightage_path)
        self.city_weightage.columns = self.city_weightage.columns.str.strip()  # Remove spaces
        
        # Load month_weightage and strip leading/trailing spaces from column names
        self.month_weightage = pd.read_csv(month_weightage_path)
        self.month_weightage.columns = self.month_weightage.columns.str.strip()  # Remove spaces
        
        # Debugging: Print column names
        print("City Weightage Columns:", self.city_weightage.columns)
        print("Month Weightage Columns:", self.month_weightage.columns)
        
        # Create lookup dictionaries
        self.crop_dict = dict(zip(self.crop_susceptibility['Crop'], 
                               self.crop_susceptibility['DiseaseSusceptibilityScore']))
        self.disease_dict = dict(zip(self.disease_virality['Disease'],
                                   self.disease_virality['ViralityFactor']))
        
        # Handle city weightage
        if 'Weightage' in self.city_weightage.columns:
            self.city_dict = dict(zip(self.city_weightage['City'], self.city_weightage['Weightage']))
        else:
            print("Warning: 'Weightage' column not found in city_weightage.csv. Using default weightage of 0.4.")
            self.city_dict = {city: 0.3 for city in self.city_weightage['City']}
        
        # Handle month weightage
        if 'Weightage' in self.month_weightage.columns:
            self.month_dict = dict(zip(self.month_weightage['Month'], self.month_weightage['Weightage']))
        else:
            print("Warning: 'Weightage' column not found in month_weightage.csv. Using default weightage of 0.5.")
            self.month_dict = {month: 0.5 for month in self.month_weightage['Month']}
        
        # Normalize month weights
        max_month_weight = max(self.month_dict.values()) if self.month_dict else 1
        self.month_dict = {k: v/max_month_weight for k, v in self.month_dict.items()}

    def _preprocess_data(self):
        # Process future weather data
        self.future_weather['date'] = pd.to_datetime(self.future_weather['date'])
        
        # Process past weather data
        self.past_weather['date'] = pd.to_datetime(self.past_weather['date'])
        self.past_weather['month'] = self.past_weather['date'].dt.strftime('%m')
        self.past_weather['year'] = self.past_weather['date'].dt.year

    def _calculate_future_weather_score(self):
        # Weather parameter weights (sum to 1)
        WEATHER_WEIGHTS = {
            'temperature': 0.2,
            'humidity': 0.25,
            'rain': 0.2,
            'wind': 0.15,
            'soil_temp': 0.1,
            'soil_moisture': 0.1
        }
        
        total_score = 0
        valid_days = 0
        
        for _, day in self.future_weather.iterrows():
            try:
                # Get values with defaults for missing data
                temp = day['temperature_2m'] if not pd.isna(day['temperature_2m']) else 22.0
                humidity = day['relative_humidity_2m'] if not pd.isna(day['relative_humidity_2m']) else 70.0
                rain = day['rain'] if not pd.isna(day['rain']) else 0.0
                wind = day['wind_speed_100m'] if not pd.isna(day['wind_speed_100m']) else 10.0
                soil_temp = day['soil_temperature_7_to_28cm'] if not pd.isna(day['soil_temperature_7_to_28cm']) else 20.0
                soil_moisture = day['soil_moisture_7_to_28cm'] if not pd.isna(day['soil_moisture_7_to_28cm']) else 0.4

                # Normalize parameters (0-1 scale)
                temp_score = max(0, min(1, (temp - 10) / 20))  # 10-30°C range
                humidity_score = humidity / 100
                rain_score = min(1, rain / 20)  # 0-20mm scale
                wind_score = max(0, min(1, wind / 30))  # 0-30 km/h
                soil_temp_score = max(0, min(1, (soil_temp - 10) / 20))
                soil_moisture_score = soil_moisture

                # Calculate daily score
                daily_score = (temp_score * WEATHER_WEIGHTS['temperature'] +
                              humidity_score * WEATHER_WEIGHTS['humidity'] +
                              rain_score * WEATHER_WEIGHTS['rain'] +
                              wind_score * WEATHER_WEIGHTS['wind'] +
                              soil_temp_score * WEATHER_WEIGHTS['soil_temp'] +
                              soil_moisture_score * WEATHER_WEIGHTS['soil_moisture'])
                
                total_score += daily_score
                valid_days += 1
            except Exception as e:
                continue
        
        return total_score / valid_days if valid_days > 0 else 0.5

    def _calculate_past_weather_score(self, target_month):
        current_year = datetime.now().year
        past_years = [current_year - 1, current_year - 2, current_year - 3]
        unfavorable_months = 0
        
        for year in past_years:
            try:
                monthly_data = self.past_weather[(self.past_weather['month'] == target_month) &
                                               (self.past_weather['year'] == year)]
                
                if monthly_data.empty:
                    unfavorable_months += 1
                    continue
                
                # Calculate average conditions
                avg_humidity = monthly_data['relative_humidity_2m (%)'].mean()
                avg_rain = monthly_data['rain (mm)'].mean()
                
                # Unfavorable conditions (humidity > 80% OR rain > 10mm)
                if avg_humidity > 80 or avg_rain > 10:
                    unfavorable_months += 1
            except:
                unfavorable_months += 1
        
        return 1 - (unfavorable_months / 3)

    def calculate_risk_score(self, month, crop, disease, city):
        # Get individual component scores
        future_score = self._calculate_future_weather_score()
        past_score = self._calculate_past_weather_score(month)
        crop_score = self.crop_dict.get(crop, 50) / 100
        disease_score = self.disease_dict.get(disease, 70) / 100
        city_score = self.city_dict.get(city, 0.4)
        month_score = self.month_dict.get(month, 0.5)

        # Weighted sum of components
        FINAL_WEIGHTS = {
            'future_weather': 0.35,
            'past_weather': 0.2,
            'crop': 0.15,
            'disease': 0.15,
            'city': 0.10,
            'month': 0.15
        }

        total_score = (
            future_score * FINAL_WEIGHTS['future_weather'] +
            past_score * FINAL_WEIGHTS['past_weather'] +
            crop_score * FINAL_WEIGHTS['crop'] +
            disease_score * FINAL_WEIGHTS['disease'] +
            city_score * FINAL_WEIGHTS['city'] +
            month_score * FINAL_WEIGHTS['month']
        ) * 100

        rescaled_score = ((total_score - 25) / (65 - 35)) * 100
        return max(0, min(100, rescaled_score))

# Example usage
if __name__ == "__main__":
    predictor = DiseaseOutbreakPredictor(
        future_weather_path="processed_FutureWeatherPrediction.csv",
        past_weather_path="processed_PastWeatherData.csv",
        crop_susceptibility_path="DiseaseSusceptibilityOfCrops.csv",
        disease_virality_path="ViralityScoreOfDiseases.csv",
        city_weightage_path="CityWeightage.csv",
        month_weightage_path="MonthWeightage.csv"
    )

    # Example prediction
    risk_score = predictor.calculate_risk_score(
        month="01",
        crop="95",
        disease="Potato Late Blight",
        city="Imphal West"
    )

    print(f"Disease Outbreak Risk Score: {risk_score:.1f}/100")