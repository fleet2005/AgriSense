# AgriSense: Manipur Precision Farming Assistant 🌱

AgriSense is an integrated precision agriculture platform designed to assist farmers and researchers in Manipur with advanced crop disease detection, outbreak prediction, and crop recommendation. Leveraging deep learning, weather analytics, and real-time data, AgriSense aims to boost crop yield, reduce losses, and promote sustainable farming.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model and Dataset](#model-and-dataset)
- [API & Integrations](#api--integrations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **Plant Disease Detection**: Upload a leaf image to detect plant diseases using a custom-trained CNN model (39 classes, including healthy and diseased states for major crops).
- **Disease Outbreak Prediction**: Predict the risk of disease outbreaks based on real-time and historical weather data, crop susceptibility, and regional factors.
- **Crop Recommendation**: Get crop recommendations based on soil nutrients (N, P, K), pH, and moisture using a logistic regression model.
- **Supplement Suggestions**: Receive actionable treatment steps and recommended supplements for detected diseases.
- **Weather Forecast Integration**: Real-time weather data fetching for location-based predictions.
- **Email Notifications**: OTP-based user verification and notification system.
- **Firebase Integration**: Fetch real-time sensor data for soil and environmental parameters.

---

## Demo

- **[Model Download](https://drive.google.com/file/d/1ieQZquso2Quik18msmVVE7xtwhr6-Adz/view?usp=sharing)**
- **[Dataset Download](https://data.mendeley.com/datasets/tywbtsjrjv/1)**

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AgriSense
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the model weights:**
   - Place `plant_disease_model_1_latest.pt` in the project root.

4. **Set up environment variables:**
   - Create a `.env` file with your email credentials for notifications.

5. **(Optional) Firebase Integration:**
   - Add your `firebase.json` credentials file to the root directory.

---

## Usage

- **Run the Streamlit app:**
  ```bash
  streamlit run app.py
  ```
- **Access the web interface** at `https://agrisense-kffgzvvhbsq5lbmmewpzks.streamlit.app/`.

- **Main functionalities:**
  - Upload plant leaf images for disease detection.
  - Enter soil and weather parameters for crop recommendations.
  - View disease outbreak risk predictions for your region.

---

## Project Structure

```
.
├── app.py                      # Main Streamlit web app
├── CNN.py                      # CNN model architecture for disease detection
├── Firebase.py                 # Firebase integration for sensor data
├── requirements.txt            # Python dependencies
├── outbreak_prediction/        # Disease outbreak prediction logic and data
├── crop_recc/                  # Crop recommendation logic and data
├── Model/                      # Model training notebooks and documentation
├── images_frontend/            # Frontend images/assets
├── test_images/                # Sample/test images
├── disease_info.csv            # Disease class info and descriptions
├── supplement_info.csv         # Supplement recommendations
└── ...
```

---

## Model and Dataset

- **Model**: Custom CNN with 4 convolutional blocks and dense layers, trained on 39 plant disease classes.
- **Dataset**: PlantVillage and other open datasets ([link](https://data.mendeley.com/datasets/tywbtsjrjv/1)).
- **Outbreak Prediction**: Uses weather, crop susceptibility, and virality data for risk scoring.
- **Crop Recommendation**: Logistic regression model trained on soil and crop data.

---

## API & Integrations

- **Weather API**: OpenWeatherMap for real-time forecasts.
- **Email**: OTP and notifications via SMTP (Zoho).
- **Firebase**: Real-time sensor data ingestion (optional).

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

[MIT License](LICENSE) (or specify your license here)

---

## Acknowledgements

- My friend Harsh Vardhan (https://github.com/harsh-vardhan3) for his contributions
- Mendley Dataset
- OpenWeatherMap API
- Streamlit, PyTorch, scikit-learn, and other open-source libraries

 
