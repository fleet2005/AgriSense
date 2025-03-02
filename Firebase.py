import firebase_admin
from firebase_admin import credentials, db

def call_firebase():
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'sample.com'
        })

    # Reference Firebase database
    ref = db.reference('/data2')

    # Retrieve the latest data
    latest_data = ref.order_by_key().limit_to_last(1).get()

    if latest_data:
        first_key = list(latest_data.keys())[0]
        data = latest_data[first_key]  # Extract the dictionary with sensor values

        # Extract relevant values
        nitrogen = data.get('nitrogen')
        phosphorus = data.get('phosphorus')
        potassium = data.get('potassium')
        soilMoisture = data.get('soilMoisture')
        soilPH = data.get('soilPH')

        # Return data as a list
        return [nitrogen, phosphorus, potassium, soilMoisture, soilPH]
    
    return None  # Return None if no data is found
