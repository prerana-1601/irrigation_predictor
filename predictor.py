"""
Daily Irrigation Predictor - Simplified Version
Uses Visual Crossing API for weather data
2 Clusters: No Irrigation vs Irrigation Needed
Location: Melbourne, Victoria, Australia
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import pickle
import warnings
from typing import Dict, List, Tuple
import logging
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the irrigation predictor"""
    # Visual Crossing API Configuration
    VISUAL_CROSSING_API_KEY = "MXXWP8CNWRS8J3WZMBFKTVWNX"
    VISUAL_CROSSING_BASE_URL = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    LOCATION = "melbourne"
    
    # Database
    DB_NAME = "irrigation_system.db"
    
    # Irrigation Thresholds
    SOIL_MOISTURE_CRITICAL = 30
    SOIL_MOISTURE_OPTIMAL = 70
    
    # Model file paths
    MODEL_DIR = "models/"
    #MODEL_DIR = "/var/task/models/"
    SCALER_FILE = "models/scaler.pkl"
    PCA_FILE = "models/pca.pkl"
    KMEANS_FILE = "models/kmeans.pkl"
    MODEL_INFO_FILE = "models/model_info.json"

# ============================================================================
# WEATHER DATA COLLECTOR - VISUAL CROSSING API
# ============================================================================

class VisualCrossingWeatherCollector:
    """Collects weather data from Visual Crossing API"""
    
    def __init__(self):
        self.api_key = Config.VISUAL_CROSSING_API_KEY
        self.base_url = Config.VISUAL_CROSSING_BASE_URL
        self.location = Config.LOCATION
    
    def fahrenheit_to_celsius(self, f: float) -> float:
        """Convert Fahrenheit to Celsius"""
        return (f - 32) * 5/9
    
    def mph_to_ms(self, mph: float) -> float:
        """Convert miles per hour to meters per second"""
        return mph * 0.44704
    
    def fetch_today_and_forecast(self) -> Dict:
        """Fetch today's weather and tomorrow's forecast"""
        try:
            # Get today's weather
            today_url = f"{self.base_url}/{self.location}/today"
            today_params = {
                'unitGroup': 'us',
                'include': 'days',
                'key': self.api_key,
                'contentType': 'json'
            }
            
            response = requests.get(today_url, params=today_params)
            response.raise_for_status()
            data = response.json()
            
            if 'days' not in data or len(data['days']) == 0:
                logger.error("No day data in API response")
                return self.get_mock_data()
            
            today = data['days'][0]
            
            # Parse today's data
            weather_data = {
                'today': {
                    'date': today['datetime'],
                    'temperature': self.fahrenheit_to_celsius(today['temp']),
                    'tempmax': self.fahrenheit_to_celsius(today['tempmax']),
                    'tempmin': self.fahrenheit_to_celsius(today['tempmin']),
                    'humidity': today['humidity'],
                    'pressure': today['pressure'],
                    'wind_speed': self.mph_to_ms(today['windspeed']),
                    'wind_dir': today['winddir'],
                    'clouds': today['cloudcover'],
                    'precipitation': today['precip'] * 25.4,  # inches to mm
                    'uvindex': today.get('uvindex', 0),
                    'solarradiation': today.get('solarradiation', 0),
                    'conditions': today.get('conditions', 'Unknown')
                },
                'forecast': []
            }
            
            # Get tomorrow's forecast
            tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            forecast_url = f"{self.base_url}/{self.location}/{tomorrow_date}"
            forecast_params = {
                'unitGroup': 'us',
                'include': 'days',
                'key': self.api_key,
                'contentType': 'json'
            }
            
            try:
                forecast_response = requests.get(forecast_url, params=forecast_params)
                forecast_response.raise_for_status()
                forecast_data = forecast_response.json()
                
                if 'days' in forecast_data and len(forecast_data['days']) > 0:
                    tomorrow = forecast_data['days'][0]
                    weather_data['forecast'].append({
                        'date': tomorrow['datetime'],
                        'temperature': self.fahrenheit_to_celsius(tomorrow['temp']),
                        'tempmax': self.fahrenheit_to_celsius(tomorrow['tempmax']),
                        'tempmin': self.fahrenheit_to_celsius(tomorrow['tempmin']),
                        'humidity': tomorrow['humidity'],
                        'wind_speed': self.mph_to_ms(tomorrow['windspeed']),
                        'clouds': tomorrow['cloudcover'],
                        'precipitation': tomorrow['precip'] * 25.4,
                        'precipprob': tomorrow.get('precipprob', 0),
                        'conditions': tomorrow.get('conditions', 'Unknown')
                    })
            except Exception as e:
                logger.warning(f"Could not fetch tomorrow's forecast: {e}")
            
            logger.info("Weather data fetched successfully from Visual Crossing")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self.get_mock_data()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return self.get_mock_data()
    
    def get_mock_data(self) -> Dict:
        """Generate mock data for demonstration"""
        logger.warning("Using mock weather data")
        return {
            'today': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'temperature': 22.5,
                'tempmax': 28.0,
                'tempmin': 17.0,
                'humidity': 65.0,
                'pressure': 1015.0,
                'wind_speed': 5.2,
                'wind_dir': 180,
                'clouds': 45,
                'precipitation': 0,
                'uvindex': 6,
                'solarradiation': 250,
                'conditions': 'Mock Data'
            },
            'forecast': [{
                'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'temperature': 23.0,
                'tempmax': 29.0,
                'tempmin': 18.0,
                'humidity': 60.0,
                'wind_speed': 4.5,
                'clouds': 40,
                'precipitation': 2.0,
                'precipprob': 30,
                'conditions': 'Mock Data'
            }]
        }

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class PredictionFeatureEngineer:
    """Handles feature engineering for daily predictions"""
    
    def calculate_et0_with_solar(self, temp: float, humidity: float, 
                                  wind_speed: float, solar_radiation: float) -> float:
        """Calculate ET0 using solar radiation"""
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        ea = es * (humidity / 100)
        vpd = es - ea
        
        # Convert solar radiation from W/m² to MJ/m²/day
        solar_radiation_mj = (solar_radiation * 86400) / 1000000
        
        # Simplified Penman-Monteith
        et0_rad = 0.408 * 0.77 * solar_radiation_mj
        et0_aero = (900 / (temp + 273)) * wind_speed * vpd
        et0 = et0_rad + et0_aero
        
        return max(0, et0)
    
    def calculate_vpd(self, temp: float, humidity: float) -> float:
        """Calculate Vapor Pressure Deficit (kPa)"""
        svp = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (humidity / 100)
        return svp - avp
    
    def calculate_degree_days(self, tempmax: float, tempmin: float, base_temp: float = 10) -> float:
        """Calculate degree days"""
        temp_avg = (tempmax + tempmin) / 2
        return max(0, temp_avg - base_temp)
    
    def get_recent_precipitation(self, days: int) -> float:
        """Get cumulative precipitation for the last n days"""
        try:
            conn = sqlite3.connect(Config.DB_NAME)
            query = f"""
                SELECT COALESCE(SUM(precipitation), 0) as total_precip
                FROM historical_weather
                WHERE date >= date('now', '-{days} days')
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df['total_precip'].iloc[0] if not df.empty else 0
        except:
            return 0
    
    def estimate_soil_moisture(self, precip_3d: float, et0_3d: float, base_moisture: float = 70) -> float:
        """Estimate current soil moisture"""
        moisture_loss = et0_3d * 1.5
        moisture_gain = precip_3d * 0.8
        current_moisture = base_moisture - moisture_loss + moisture_gain
        return max(0, min(100, current_moisture))
    
    def engineer_features(self, weather_data: Dict) -> Dict:
        """Engineer features from weather data"""
        today = weather_data['today']
        forecast = weather_data['forecast'][0] if weather_data['forecast'] else today
        
        # Calculate ET0
        et0_current = self.calculate_et0_with_solar(
            today['temperature'],
            today['humidity'],
            today['wind_speed'],
            today['solarradiation']
        )
        
        # Calculate other features
        vpd = self.calculate_vpd(today['temperature'], today['humidity'])
        degree_days = self.calculate_degree_days(today['tempmax'], today['tempmin'])
        
        # Get recent precipitation
        precip_3d = self.get_recent_precipitation(3)
        precip_7d = self.get_recent_precipitation(7)
        
        # Add today's precipitation
        precip_3d += today['precipitation']
        precip_7d += today['precipitation']
        
        # Estimate soil moisture
        et0_3d = et0_current * 3
        soil_moisture = self.estimate_soil_moisture(precip_3d, et0_3d)
        
        # Calculate moisture deficit
        moisture_deficit = max(0, Config.SOIL_MOISTURE_OPTIMAL - soil_moisture)
        
        features = {
            'soil_moisture_estimate': soil_moisture,
            'et0': et0_current,
            'cumulative_precipitation_3d': precip_3d,
            'cumulative_precipitation_7d': precip_7d,
            'vpd': vpd,
            'degree_days': degree_days,
            'moisture_deficit': moisture_deficit,
            'forecast_rain_24h': forecast.get('precipitation', 0),
            'current_conditions': {
                'temperature': today['temperature'],
                'humidity': today['humidity'],
                'wind_speed': today['wind_speed'],
                'clouds': today['clouds'],
                'conditions': today['conditions']
            }
        }
        
        return features

# ============================================================================
# IRRIGATION PREDICTOR
# ============================================================================

class IrrigationPredictor:
    """Makes irrigation predictions using trained model"""
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.kmeans = None
        self.model_info = None
        self.feature_columns = [
            'soil_moisture_estimate', 'et0', 'moisture_deficit',
            'cumulative_precipitation_3d', 'cumulative_precipitation_7d',
            'vpd', 'degree_days'
        ]
        self.load_models()
    
    def load_models(self) -> bool:
        """Load all trained model components"""
        try:
            model_files = [Config.SCALER_FILE, Config.PCA_FILE, Config.KMEANS_FILE, Config.MODEL_INFO_FILE]
            for file_path in model_files:
                if not os.path.exists(file_path):
                    logger.error(f"Model file not found: {file_path}")
                    return False
            
            with open(Config.SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(Config.PCA_FILE, 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(Config.KMEANS_FILE, 'rb') as f:
                self.kmeans = pickle.load(f)
            
            with open(Config.MODEL_INFO_FILE, 'r') as f:
                self.model_info = json.load(f)
            
            logger.info("All model components loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def predict_irrigation_need(self, features: Dict) -> Tuple[int, str, float]:
        """Predict irrigation need - 2 clusters only"""
        if not all([self.scaler, self.pca, self.kmeans]):
            raise ValueError("Models not loaded. Please train the model first.")
        
        # Prepare feature vector
        feature_vector = pd.DataFrame([{
            col: features.get(col, 0) for col in self.feature_columns
        }])
        
        # Scale and transform
        X_scaled = self.scaler.transform(feature_vector)
        X_pca = self.pca.transform(X_scaled)
        
        # Predict cluster
        cluster = self.kmeans.predict(X_pca)[0]
        
        # Calculate confidence
        distances = self.kmeans.transform(X_pca)[0]
        min_distance = distances[cluster]
        confidence = 1 / (1 + min_distance)
        
        # Map cluster to decision (2 clusters only)
        irrigation_decisions = {
            0: "No irrigation needed",
            1: "Irrigation recommended"
        }
        
        decision = irrigation_decisions.get(cluster, "Unknown")
        
        return cluster, decision, confidence

# ============================================================================
# MAIN PREDICTION SYSTEM
# ============================================================================

class DailyIrrigationSystem:
    """Main system for daily irrigation predictions"""
    
    def __init__(self):
        self.weather_collector = VisualCrossingWeatherCollector()
        self.feature_engineer = PredictionFeatureEngineer()
        self.predictor = IrrigationPredictor()
    
    def run_daily_prediction(self) -> Dict:
        """Run complete daily prediction workflow"""
        logger.info("="*60)
        logger.info("STARTING DAILY IRRIGATION PREDICTION")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        try:
            # Check if models are loaded
            if not all([self.predictor.scaler, self.predictor.pca, self.predictor.kmeans]):
                raise ValueError("Trained models not found. Please run training first.")
            
            # Collect weather data
            logger.info("Fetching weather data from Visual Crossing...")
            weather_data = self.weather_collector.fetch_today_and_forecast()
            
            if not weather_data:
                raise ValueError("Failed to collect weather data")
            
            # Engineer features
            logger.info("Engineering prediction features...")
            features = self.feature_engineer.engineer_features(weather_data)
            
            # Make prediction
            logger.info("Running irrigation prediction...")
            cluster, decision, confidence = self.predictor.predict_irrigation_need(features)
            
            # Create prediction result
            result = {
                'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'cluster': int(cluster),
                'decision': decision,
                'confidence': round(confidence * 100, 1),
                'weather': weather_data,
                'field_conditions': {
                    'soil_moisture_percent': round(features['soil_moisture_estimate'], 1),
                    'moisture_deficit_percent': round(features['moisture_deficit'], 1),
                    'evapotranspiration_mm': round(features['et0'], 2),
                    'recent_rainfall_3d_mm': round(features['cumulative_precipitation_3d'], 1),
                    'vpd_kpa': round(features['vpd'], 2)
                }
            }
            
            # Display results
            self.display_results(result)
            
            # Save results
            self.save_results(result)
            
            logger.info("="*60)
            logger.info("PREDICTION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def display_results(self, result: Dict):
        """Display prediction results"""
        if not result:
            return
        
        print("\n" + "="*70)
        print("IRRIGATION PREDICTION RESULTS")
        print("="*70)
        
        print(f"\nPrediction Time: {result['prediction_date']}")
        print(f"Location: Melbourne, Victoria, Australia")
        
        # Main prediction
        print(f"\n{'PREDICTION':^70}")
        print("-"*70)
        status = "✓ NO IRRIGATION" if result['cluster'] == 0 else "⚠ IRRIGATION NEEDED"
        print(f"Status: {status}")
        print(f"Decision: {result['decision']}")
        print(f"Confidence: {result['confidence']}%")
        
        # Current weather
        today = result['weather']['today']
        header1 = "TODAY'S WEATHER"
        print(f"\n{header1:^70}")
        print("-"*70)
        print(f"Conditions: {today['conditions']}")
        print(f"Temperature: {today['temperature']:.1f}°C (High: {today['tempmax']:.1f}°C, Low: {today['tempmin']:.1f}°C)")
        print(f"Humidity: {today['humidity']:.0f}%")
        print(f"Wind Speed: {today['wind_speed']:.1f} m/s")
        print(f"Cloud Cover: {today['clouds']:.0f}%")
        print(f"Precipitation: {today['precipitation']:.1f} mm")
        
        # Forecast
        if result['weather']['forecast']:
            forecast = result['weather']['forecast'][0]
            header2 = "TOMORROW'S FORECAST"
            print(f"\n{header2:^70}")
            print("-"*70)
            print(f"Conditions: {forecast['conditions']}")
            print(f"Temperature: {forecast['temperature']:.1f}°C")
            print(f"Expected Rain: {forecast['precipitation']:.1f} mm")
            print(f"Rain Probability: {forecast['precipprob']:.0f}%")
        
        # Field conditions
        field = result['field_conditions']
        print(f"\n{'FIELD CONDITIONS':^70}")
        print("-"*70)
        moisture_status = "GOOD" if field['soil_moisture_percent'] > 60 else "LOW" if field['soil_moisture_percent'] > 30 else "CRITICAL"
        print(f"Soil Moisture: {field['soil_moisture_percent']:.1f}% ({moisture_status})")
        print(f"Moisture Deficit: {field['moisture_deficit_percent']:.1f}%")
        print(f"Evapotranspiration: {field['evapotranspiration_mm']:.2f} mm/day")
        print(f"Recent Rain (3d): {field['recent_rainfall_3d_mm']:.1f} mm")
        
        print("\n" + "="*70)
    
    def save_results(self, result: Dict):
        """Save prediction results to file"""
        if not result:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"irrigation_prediction_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(result, f, indent=4, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DAILY IRRIGATION PREDICTOR")
    print("Location: Melbourne, Victoria, Australia")
    print("Data Source: Visual Crossing Weather API")
    print("="*60)
    
    try:
        irrigation_system = DailyIrrigationSystem()
        
        # Check model status
        if not irrigation_system.predictor.model_info:
            print("\nERROR: No trained model found!")
            print("Please run the training script first.")
            return
        
        print(f"\nModel Status: Loaded")
        print(f"Training Date: {irrigation_system.predictor.model_info['training_date']}")
        print(f"Training Samples: {irrigation_system.predictor.model_info['training_samples']:,}")
        
        # Run prediction
        print("\nRunning daily irrigation prediction...")
        irrigation_system.run_daily_prediction()
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()

import json
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def lambda_handler(event, context):
    """
    AWS Lambda entry point
    """
    logger.info("Lambda invoked with event: %s", json.dumps(event))
    try:
        irrigation_system = DailyIrrigationSystem()
        result = irrigation_system.run_daily_prediction()
        logger.info("Prediction result: %s", result)

        return {
            'statusCode': 200,
            'body': json.dumps(result, default=str)
        }

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error("Error occurred: %s", e)
        logger.error("Traceback: %s", traceback_str)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e), 'traceback': traceback_str})
        }

if __name__ == "__main__":
    main()
