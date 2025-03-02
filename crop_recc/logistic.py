import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Starting script...")

df = pd.read_csv('crop_recc/data.csv')  

 
X = df[['N', 'P', 'K', 'pH', 'soil_moisture']]
y = df['Crop']

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

 
y_pred = model.predict(X_test_scaled)

 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

 
def predict_closest_crop(N, P, K, pH, soil_moisture):
    features = [[N, P, K, pH, soil_moisture]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]
 
N_input = 40
P_input = 30 
K_input = 30 
pH_input = 6
soil_moisture_input = 70

predicted_crop = predict_closest_crop(N_input, P_input, K_input, pH_input, soil_moisture_input)
print(f'The closest crop variety is: {predicted_crop}')
