"""
Daily Irrigation Predictor
Uses the trained model to make irrigation predictions for the next day
Fetches current weather data and displays recommendations
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
from typing import Dict, List, Tuple, Optional
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
    # API Configuration
    API_KEY = "9745d148e613ab1f00e32bb8e10a65fc"
    BASE_URL = "https://pro.openweathermap.org/data/2.5"
    LOCATION = "Melbourne,Victoria,AU"
    LAT = -37.8136  # Melbourne coordinates
    LON = 144.9631

    # Database
    DB_NAME = "irrigation_system.db"

    # Irrigation Thresholds
    SOIL_MOISTURE_CRITICAL = 30  # Critical soil moisture level (%)
    SOIL_MOISTURE_OPTIMAL = 70   # Optimal soil moisture level (%)

    # Field Information
    FIELD_AREA = 10000  # m²
    CROP_TYPE = "wheat"
    IRRIGATION_RATE = 50  # liters per minute

    # Model file paths
    MODEL_DIR = "models/"
    SCALER_FILE = "models/scaler.pkl"
    PCA_FILE = "models/pca.pkl"
    KMEANS_FILE = "models/kmeans.pkl"
    MODEL_INFO_FILE = "models/model_info.json"

# ============================================================================
# CURRENT WEATHER DATA COLLECTOR
# ============================================================================

class WeatherDataCollector:
    """Collects current weather data from OpenWeatherMap API"""

    def __init__(self):
        self.api_key = Config.API_KEY
        self.base_url = Config.BASE_URL
        self.location = Config.LOCATION
        self.lat = Config.LAT
        self.lon = Config.LON

    def fetch_current_weather(self) -> Dict:
        """Fetch current weather data"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': self.location,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Parse weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'precipitation': data.get('rain', {}).get('1h', 0),
                'timestamp': datetime.now()
            }

            logger.info(f"Current weather data collected successfully")
            return weather_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            # Return mock data for demonstration
            return self.get_mock_weather_data()

    def fetch_forecast(self) -> List[Dict]:
        """Fetch weather forecast for the next 24 hours"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': self.location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': 8  # Next 24 hours (3-hour intervals)
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            forecast_data = []
            for item in data['list']:
                forecast_data.append({
                    'timestamp': datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'clouds': item['clouds']['all'],
                    'precipitation': item.get('rain', {}).get('3h', 0),
                    'pop': item.get('pop', 0)  # Probability of precipitation
                })

            logger.info(f"Forecast data collected successfully")
            return forecast_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching forecast data: {e}")
            return self.get_mock_forecast_data()

    def get_mock_weather_data(self) -> Dict:
        """Generate mock current weather data for demonstration"""
        return {
            'temperature': 22.5,
            'humidity': 65.0,
            'pressure': 1015.2,
            'wind_speed': 5.2,
            'wind_deg': 180,
            'clouds': 45,
            'precipitation': 0,
            'timestamp': datetime.now()
        }

    def get_mock_forecast_data(self) -> List[Dict]:
        """Generate mock forecast data for demonstration"""
        forecast_data = []
        for i in range(8):
            forecast_data.append({
                'timestamp': datetime.now() + timedelta(hours=3*i),
                'temperature': 20 + np.random.normal(0, 3),
                'humidity': 60 + np.random.normal(0, 10),
                'pressure': 1015 + np.random.normal(0, 5),
                'wind_speed': 5 + np.random.normal(0, 2),
                'clouds': 50 + np.random.normal(0, 20),
                'precipitation': max(0, np.random.normal(0, 2)),
                'pop': np.random.uniform(0, 0.5)
            })
        return forecast_data

# ============================================================================
# FEATURE ENGINEERING FOR PREDICTION
# ============================================================================

class PredictionFeatureEngineer:
    """Handles feature engineering for daily predictions"""

    def __init__(self):
        pass

    def estimate_solar_radiation(self, cloud_cover: float) -> float:
        """Estimate incoming solar radiation (MJ/m²/day) based on cloud cover percentage"""
        clear_sky_radiation = 25.0
        reduction_factor = (100 - cloud_cover) / 100.0
        return clear_sky_radiation * reduction_factor

    def calculate_et0(self, temp: float, humidity: float, wind_speed: float, cloud_cover: float = 50) -> float:
        """Calculate reference evapotranspiration (ET0) using simplified Penman-Monteith"""
        solar_radiation = self.estimate_solar_radiation(cloud_cover)
        
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        ea = es * (humidity / 100)
        vpd = es - ea
        
        et0 = 0.0023 * (temp + 17.8) * np.sqrt(abs(temp - humidity)) * (solar_radiation / 25)
        et0 += 0.1 * wind_speed * vpd
        
        return max(0, et0)

    def calculate_vpd(self, temp: float, humidity: float) -> float:
        """Calculate Vapor Pressure Deficit (kPa)"""
        svp = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (humidity / 100)
        return svp - avp

    def calculate_degree_days(self, temp: float, base_temp: float = 10) -> float:
        """Calculate degree days for crop development"""
        return max(0, temp - base_temp)

    def get_recent_precipitation(self, days: int) -> float:
        """Get cumulative precipitation for the last n days from historical data"""
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
            # If no historical data, estimate based on forecast
            return 0

    def estimate_soil_moisture(self, precip_3d: float, et0_3d: float, base_moisture: float = 70) -> float:
        """Estimate current soil moisture based on water balance"""
        moisture_loss = et0_3d * 2  # ET factor
        moisture_gain = precip_3d * 10  # Precipitation absorption factor
        current_moisture = base_moisture - moisture_loss + moisture_gain
        return max(0, min(100, current_moisture))

    def engineer_current_features(self, weather_data: Dict, forecast_data: List[Dict]) -> Dict:
        """Engineer features from current weather data for prediction"""
        
        # Calculate current ET0
        et0_current = self.calculate_et0(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data.get('clouds', 0)
        )

        # Calculate VPD and degree days
        vpd = self.calculate_vpd(weather_data['temperature'], weather_data['humidity'])
        degree_days = self.calculate_degree_days(weather_data['temperature'])

        # Get recent precipitation
        precip_3d = self.get_recent_precipitation(3)
        precip_7d = self.get_recent_precipitation(7)
        
        # Add expected precipitation from forecast
        forecast_precip_24h = sum(f.get('precipitation', 0) for f in forecast_data[:8])
        
        # Estimate current soil moisture
        # Estimate ET0 for last 3 days (approximation)
        et0_3d = et0_current * 3  # Rough approximation
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
            'forecast_rain_24h': forecast_precip_24h,
            'current_conditions': {
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'wind_speed': weather_data['wind_speed'],
                'clouds': weather_data.get('clouds', 0),
                'pressure': weather_data.get('pressure', 1013)
            }
        }

        return features

# ============================================================================
# MODEL LOADER AND PREDICTOR
# ============================================================================

class IrrigationPredictor:
    """Loads trained model and makes irrigation predictions"""

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
            # Check if model files exist
            model_files = [Config.SCALER_FILE, Config.PCA_FILE, Config.KMEANS_FILE, Config.MODEL_INFO_FILE]
            for file_path in model_files:
                if not os.path.exists(file_path):
                    logger.error(f"Model file not found: {file_path}")
                    return False

            # Load scaler
            with open(Config.SCALER_FILE, 'rb') as f:
                self.scaler = pickle.load(f)

            # Load PCA
            with open(Config.PCA_FILE, 'rb') as f:
                self.pca = pickle.load(f)

            # Load K-Means
            with open(Config.KMEANS_FILE, 'rb') as f:
                self.kmeans = pickle.load(f)

            # Load model info
            with open(Config.MODEL_INFO_FILE, 'r') as f:
                self.model_info = json.load(f)

            logger.info("All model components loaded successfully")
            logger.info(f"Model trained on: {self.model_info['training_date']}")
            logger.info(f"Training samples: {self.model_info['training_samples']:,}")
            logger.info(f"Model performance: {self.model_info['silhouette_score']:.3f}")
            
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def predict_irrigation_need(self, features: Dict) -> Tuple[int, str, float, Dict]:
        """
        Predict irrigation need based on current features
        Returns: (cluster_label, irrigation_decision, confidence, detailed_analysis)
        """
        if not all([self.scaler, self.pca, self.kmeans]):
            raise ValueError("Models not loaded. Please train the model first.")

        # Prepare feature vector
        feature_vector = pd.DataFrame([{
            col: features.get(col, 0) for col in self.feature_columns
        }])

        # Scale and transform features
        X_scaled = self.scaler.transform(feature_vector)
        X_pca = self.pca.transform(X_scaled)

        # Predict cluster
        cluster = self.kmeans.predict(X_pca)[0]

        # Calculate distances to all cluster centers for confidence
        distances = self.kmeans.transform(X_pca)[0]
        min_distance = distances[cluster]
        confidence = 1 / (1 + min_distance)  # Convert distance to confidence

        # Map cluster to irrigation decision
        irrigation_decisions = {
            0: "No irrigation needed",
            1: "Light irrigation recommended", 
            2: "Heavy irrigation required"
        }

        decision = irrigation_decisions.get(cluster, "Unknown")

        # Create detailed analysis
        detailed_analysis = {
            'cluster_distances': {f'cluster_{i}': float(dist) for i, dist in enumerate(distances)},
            'feature_contributions': self.analyze_feature_contributions(features),
            'risk_factors': self.identify_risk_factors(features),
            'cluster_characteristics': self.get_cluster_characteristics(cluster)
        }

        return cluster, decision, confidence, detailed_analysis

    def analyze_feature_contributions(self, features: Dict) -> Dict:
        """Analyze which features are contributing most to the prediction"""
        contributions = {}
        
        # Analyze key irrigation drivers
        soil_moisture = features.get('soil_moisture_estimate', 70)
        moisture_deficit = features.get('moisture_deficit', 0)
        et0 = features.get('et0', 0)
        recent_rain = features.get('cumulative_precipitation_3d', 0)
        
        # Categorize feature impacts
        if soil_moisture < Config.SOIL_MOISTURE_CRITICAL:
            contributions['critical_moisture'] = f"Soil moisture critically low at {soil_moisture:.1f}%"
        elif soil_moisture < Config.SOIL_MOISTURE_OPTIMAL:
            contributions['low_moisture'] = f"Soil moisture below optimal at {soil_moisture:.1f}%"
        
        if moisture_deficit > 20:
            contributions['high_deficit'] = f"High moisture deficit of {moisture_deficit:.1f}%"
        
        if et0 > 6:
            contributions['high_evapotranspiration'] = f"High ET0 of {et0:.2f} mm/day"
        
        if recent_rain < 5:
            contributions['low_recent_rainfall'] = f"Low recent rainfall: {recent_rain:.1f} mm in 3 days"
        
        return contributions

    def identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify risk factors that could affect irrigation decisions"""
        risks = []
        
        # Weather-based risks
        vpd = features.get('vpd', 0)
        if vpd > 1.5:
            risks.append(f"High vapor pressure deficit ({vpd:.2f} kPa) increases water stress")
        
        # Forecast-based risks
        forecast_rain = features.get('forecast_rain_24h', 0)
        if forecast_rain > 10:
            risks.append(f"Heavy rain expected ({forecast_rain:.1f} mm) - consider delaying irrigation")
        
        # Temperature risks
        temp = features.get('current_conditions', {}).get('temperature', 20)
        if temp > 30:
            risks.append(f"High temperature ({temp:.1f}°C) increases evaporation - prefer early morning irrigation")
        
        # Wind risks
        wind_speed = features.get('current_conditions', {}).get('wind_speed', 0)
        if wind_speed > 15:
            risks.append(f"High wind speed ({wind_speed:.1f} m/s) may affect irrigation efficiency")
        
        return risks

    def get_cluster_characteristics(self, cluster: int) -> Dict:
        """Get characteristics of the predicted cluster"""
        characteristics = {
            0: {
                'description': 'No Irrigation Needed',
                'typical_conditions': 'Good soil moisture, recent rainfall, low evapotranspiration',
                'action': 'Monitor conditions, no immediate action required',
                'water_savings': 'Maximum water conservation'
            },
            1: {
                'description': 'Light Irrigation Recommended',
                'typical_conditions': 'Moderate moisture deficit, some recent rainfall, moderate ET',
                'action': 'Apply light irrigation to maintain optimal moisture',
                'water_savings': 'Efficient water use with targeted application'
            },
            2: {
                'description': 'Heavy Irrigation Required',
                'typical_conditions': 'Low soil moisture, little recent rainfall, high evapotranspiration',
                'action': 'Immediate irrigation required to prevent crop stress',
                'water_savings': 'Essential irrigation to prevent yield loss'
            }
        }
        
        return characteristics.get(cluster, {})

# ============================================================================
# IRRIGATION PLAN GENERATOR
# ============================================================================

class IrrigationPlanGenerator:
    """Generates specific irrigation plans based on predictions"""

    def __init__(self):
        pass

    def calculate_water_volume(self, cluster: int, features: Dict) -> float:
        """Calculate required water volume in liters based on cluster and conditions"""
        # Base volume per m² based on cluster
        base_volumes = {0: 0, 1: 2, 2: 5}  # liters per m²
        
        volume_per_m2 = base_volumes.get(cluster, 0)
        
        if volume_per_m2 == 0:
            return 0
        
        # Adjust based on conditions
        moisture_deficit = features.get('moisture_deficit', 0)
        et0 = features.get('et0', 0)
        forecast_rain = features.get('forecast_rain_24h', 0)
        
        # Moisture deficit adjustment
        if moisture_deficit > 0:
            deficit_factor = 1 + (moisture_deficit / 100)
            volume_per_m2 *= deficit_factor
        
        # ET0 adjustment
        if et0 > 5:
            et_factor = 1 + ((et0 - 5) * 0.1)
            volume_per_m2 *= et_factor
        
        # Forecast rain adjustment
        if forecast_rain > 5:
            rain_reduction = min(0.5, forecast_rain / 20)
            volume_per_m2 *= (1 - rain_reduction)
        
        # Calculate total volume
        total_volume = volume_per_m2 * Config.FIELD_AREA
        
        return max(0, total_volume)

    def calculate_irrigation_duration(self, volume: float) -> float:
        """Calculate irrigation duration in minutes"""
        if volume <= 0:
            return 0
        return volume / Config.IRRIGATION_RATE

    def determine_optimal_timing(self, forecast_data: List[Dict]) -> Tuple[str, str]:
        """Determine optimal irrigation timing"""
        if not forecast_data:
            return "05:00", "Early morning default"
        
        # Analyze next 24 hours
        tomorrow_hours = []
        tomorrow = (datetime.now() + timedelta(days=1)).date()
        
        for forecast in forecast_data:
            if forecast['timestamp'].date() == tomorrow:
                tomorrow_hours.append(forecast)
        
        if not tomorrow_hours:
            return "05:00", "No forecast data for tomorrow"
        
        # Find optimal window (4 AM - 8 AM preferred)
        morning_hours = [f for f in tomorrow_hours if 4 <= f['timestamp'].hour <= 8]
        
        if morning_hours:
            # Select time with lowest temperature and wind
            best_time = min(morning_hours, 
                          key=lambda x: x['temperature'] + x['wind_speed'] + x.get('pop', 0))
            return best_time['timestamp'].strftime("%H:00"), "Optimal morning window"
        else:
            # If no morning data, find best available time
            best_time = min(tomorrow_hours,
                          key=lambda x: x['temperature'] + x['wind_speed'] + x.get('pop', 0))
            return best_time['timestamp'].strftime("%H:00"), "Best available time"

    def generate_irrigation_plan(self, prediction: Tuple, features: Dict, forecast_data: List[Dict]) -> Dict:
        """Generate comprehensive irrigation plan"""
        cluster, decision, confidence, analysis = prediction
        
        # Calculate irrigation requirements
        water_volume = self.calculate_water_volume(cluster, features)
        duration = self.calculate_irrigation_duration(water_volume)
        optimal_time, timing_reason = self.determine_optimal_timing(forecast_data)
        
        # Analyze forecast
        forecast_rain = features.get('forecast_rain_24h', 0)
        avg_temp_tomorrow = np.mean([f['temperature'] for f in forecast_data[:8]]) if forecast_data else 20
        avg_wind_tomorrow = np.mean([f['wind_speed'] for f in forecast_data[:8]]) if forecast_data else 5
        rain_probability = np.mean([f.get('pop', 0) for f in forecast_data[:8]]) if forecast_data else 0
        
        # Generate recommendations
        recommendations = self.generate_recommendations(cluster, features, forecast_data, analysis)
        
        # Create comprehensive plan
        plan = {
            'prediction_date': datetime.now().strftime("%Y-%m-%d"),
            'irrigation_date': (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            'cluster': cluster,
            'decision': decision,
            'confidence': round(confidence * 100, 1),
            'irrigation_details': {
                'needed': cluster > 0,
                'water_volume_liters': round(water_volume, 0),
                'duration_minutes': round(duration, 0),
                'optimal_start_time': optimal_time,
                'timing_reason': timing_reason
            },
            'weather_forecast': {
                'expected_rain_mm': round(forecast_rain, 1),
                'rain_probability_percent': round(rain_probability * 100, 1),
                'avg_temperature_c': round(avg_temp_tomorrow, 1),
                'avg_wind_speed_ms': round(avg_wind_tomorrow, 1)
            },
            'field_conditions': {
                'soil_moisture_percent': round(features.get('soil_moisture_estimate', 0), 1),
                'moisture_deficit_percent': round(features.get('moisture_deficit', 0), 1),
                'evapotranspiration_mm': round(features.get('et0', 0), 2),
                'recent_rainfall_3d_mm': round(features.get('cumulative_precipitation_3d', 0), 1),
                'recent_rainfall_7d_mm': round(features.get('cumulative_precipitation_7d', 0), 1),
                'vapor_pressure_deficit_kpa': round(features.get('vpd', 0), 2)
            },
            'analysis': analysis,
            'recommendations': recommendations,
            'model_info': {
                'model_confidence': f"{confidence:.2%}",
                'cluster_characteristics': analysis['cluster_characteristics']
            }
        }
        
        return plan

    def generate_recommendations(self, cluster: int, features: Dict, forecast_data: List[Dict], analysis: Dict) -> List[str]:
        """Generate specific actionable recommendations"""
        recommendations = []
        
        # Cluster-based recommendations
        if cluster == 0:
            recommendations.append("Field conditions are good - no irrigation required")
            recommendations.append("Continue monitoring soil moisture levels")
            if features.get('forecast_rain_24h', 0) > 5:
                recommendations.append("Expected rainfall will maintain good moisture levels")
        
        elif cluster == 1:
            recommendations.append("Apply light irrigation to maintain optimal moisture")
            recommendations.append("Early morning application (4-8 AM) recommended")
            recommendations.append("Monitor application rate to avoid overwatering")
        
        else:  # cluster == 2
            recommendations.append("Immediate irrigation required to prevent crop stress")
            recommendations.append("Apply heavy irrigation as soon as possible")
            recommendations.append("Consider split applications if temperature is high")
        
        # Weather-specific recommendations
        temp = features.get('current_conditions', {}).get('temperature', 20)
        if temp > 30:
            recommendations.append(f"High temperature ({temp:.1f}°C) - irrigate early morning or evening")
        
        wind_speed = features.get('current_conditions', {}).get('wind_speed', 0)
        if wind_speed > 15:
            recommendations.append(f"High winds ({wind_speed:.1f} m/s) - use drip irrigation if available")
        
        forecast_rain = features.get('forecast_rain_24h', 0)
        if forecast_rain > 10:
            recommendations.append(f"Heavy rain expected ({forecast_rain:.1f} mm) - consider delaying irrigation")
        elif forecast_rain > 5:
            recommendations.append(f"Rain expected ({forecast_rain:.1f} mm) - reduce irrigation volume")
        
        # Risk-based recommendations
        for risk in analysis.get('risk_factors', []):
            recommendations.append(f"Warning: {risk}")
        
        # Efficiency recommendations
        if cluster > 0:
            recommendations.append("Consider using soil moisture sensors for precise monitoring")
            recommendations.append("Track water usage for cost optimization")
        
        return recommendations

# ============================================================================
# MAIN PREDICTION SYSTEM
# ============================================================================

class DailyIrrigationSystem:
    """Main system for daily irrigation predictions"""

    def __init__(self):
        self.weather_collector = WeatherDataCollector()
        self.feature_engineer = PredictionFeatureEngineer()
        self.predictor = IrrigationPredictor()
        self.plan_generator = IrrigationPlanGenerator()

    def run_daily_prediction(self) -> Dict:
        """Run complete daily prediction workflow"""
        logger.info("="*60)
        logger.info("STARTING DAILY IRRIGATION PREDICTION")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        try:
            # Step 1: Check if models are loaded
            if not all([self.predictor.scaler, self.predictor.pca, self.predictor.kmeans]):
                raise ValueError("Trained models not found. Please run irrigation_model_trainer.py first.")

            # Step 2: Collect current weather data
            logger.info("Collecting current weather data...")
            current_weather = self.weather_collector.fetch_current_weather()
            
            if not current_weather:
                raise ValueError("Failed to collect weather data")

            # Step 3: Collect forecast data
            logger.info("Collecting weather forecast...")
            forecast_data = self.weather_collector.fetch_forecast()

            # Step 4: Engineer features
            logger.info("Engineering prediction features...")
            features = self.feature_engineer.engineer_current_features(current_weather, forecast_data)

            # Step 5: Make prediction
            logger.info("Running irrigation prediction model...")
            prediction = self.predictor.predict_irrigation_need(features)

            cluster, decision, confidence, analysis = prediction
            logger.info(f"Prediction: {decision} (Cluster {cluster})")
            logger.info(f"Confidence: {confidence:.1%}")

            # Step 6: Generate irrigation plan
            logger.info("Generating irrigation plan...")
            irrigation_plan = self.plan_generator.generate_irrigation_plan(
                prediction, features, forecast_data
            )

            # Step 7: Display results
            self.display_prediction_results(irrigation_plan)

            # Step 8: Save results
            self.save_prediction_results(irrigation_plan)

            logger.info("="*60)
            logger.info("DAILY PREDICTION COMPLETED SUCCESSFULLY")
            logger.info("="*60)

            return irrigation_plan

        except Exception as e:
            logger.error(f"Prediction workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def display_prediction_results(self, plan: Dict):
        """Display prediction results in a formatted way"""
        if not plan:
            print("No prediction results to display")
            return

        print("\n" + "="*80)
        print("IRRIGATION PREDICTION RESULTS")
        print("="*80)
        
        # Header information
        print(f"Prediction Date: {plan['prediction_date']}")
        print(f"Irrigation Date: {plan['irrigation_date']}")
        print(f"Location: Melbourne, Victoria, Australia")
        
        # Main prediction
        print(f"\nPREDICTION")
        print("-"*40)
        decision_status = "GOOD" if plan['cluster'] == 0 else "CAUTION" if plan['cluster'] == 1 else "URGENT"
        print(f"Status: {decision_status}")
        print(f"Decision: {plan['decision']}")
        print(f"Confidence: {plan['confidence']}%")
        print(f"Cluster: {plan['cluster']}")
        
        # Irrigation details
        irrigation = plan['irrigation_details']
        if irrigation['needed']:
            print(f"\nIRRIGATION DETAILS")
            print("-"*40)
            print(f"Water Volume: {irrigation['water_volume_liters']:,.0f} liters")
            print(f"Duration: {irrigation['duration_minutes']:.0f} minutes")
            print(f"Optimal Start Time: {irrigation['optimal_start_time']}")
            print(f"Timing Reason: {irrigation['timing_reason']}")
        
        # Weather forecast
        weather = plan['weather_forecast']
        print(f"\nWEATHER FORECAST (Next 24h)")
        print("-"*40)
        print(f"Expected Rain: {weather['expected_rain_mm']:.1f} mm")
        print(f"Rain Probability: {weather['rain_probability_percent']:.0f}%")
        print(f"Avg Temperature: {weather['avg_temperature_c']:.1f}°C")
        print(f"Avg Wind Speed: {weather['avg_wind_speed_ms']:.1f} m/s")
        
        # Field conditions
        field = plan['field_conditions']
        print(f"\nCURRENT FIELD CONDITIONS")
        print("-"*40)
        moisture_status = "GOOD" if field['soil_moisture_percent'] > 60 else "LOW" if field['soil_moisture_percent'] > 30 else "CRITICAL"
        print(f"Soil Moisture: {field['soil_moisture_percent']:.1f}% ({moisture_status})")
        print(f"Moisture Deficit: {field['moisture_deficit_percent']:.1f}%")
        print(f"Evapotranspiration: {field['evapotranspiration_mm']:.2f} mm/day")
        print(f"Recent Rain (3d): {field['recent_rainfall_3d_mm']:.1f} mm")
        print(f"Recent Rain (7d): {field['recent_rainfall_7d_mm']:.1f} mm")
        
        # Recommendations
        recommendations = plan['recommendations']
        if recommendations:
            print(f"\nRECOMMENDATIONS")
            print("-"*40)
            for i, rec in enumerate(recommendations[:8], 1):  # Show first 8 recommendations
                print(f"{i:2d}. {rec}")
            
            if len(recommendations) > 8:
                print(f"    ... and {len(recommendations) - 8} more recommendations")
        
        # Feature contributions (if any critical factors)
        contributions = plan['analysis'].get('feature_contributions', {})
        if contributions:
            print(f"\nKEY FACTORS")
            print("-"*40)
            for factor, description in contributions.items():
                print(f"- {description}")
        
        print("\n" + "="*80)

    def save_prediction_results(self, plan: Dict):
        """Save prediction results to file"""
        if not plan:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"irrigation_prediction_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(plan, f, indent=4, default=str)
            
            logger.info(f"Prediction results saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")

    def get_model_status(self) -> Dict:
        """Get current model status and information"""
        try:
            if self.predictor.model_info:
                return {
                    'model_loaded': True,
                    'training_date': self.predictor.model_info['training_date'],
                    'training_samples': self.predictor.model_info['training_samples'],
                    'model_performance': self.predictor.model_info['silhouette_score'],
                    'cluster_distribution': self.predictor.model_info['cluster_distribution']
                }
            else:
                return {'model_loaded': False}
        except:
            return {'model_loaded': False}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("DAILY IRRIGATION PREDICTOR")
    print("Location: Melbourne, Victoria, Australia")
    print("="*60)

    try:
        # Initialize prediction system
        irrigation_system = DailyIrrigationSystem()
        
        # Check model status
        model_status = irrigation_system.get_model_status()
        
        if not model_status.get('model_loaded', False):
            print("\nERROR: No trained model found!")
            print("Please run 'python irrigation_model_trainer.py' first to train the model.")
            print("This will collect historical data and train the prediction model.")
            return
        
        # Display model info
        print(f"\nModel Status: Loaded")
        print(f"Training Date: {model_status['training_date']}")
        print(f"Training Samples: {model_status['training_samples']:,}")
        print(f"Model Performance: {model_status['model_performance']:.3f}")
        
        # User options
        print("\n" + "="*60)
        print("Select operation:")
        print("1. Run daily irrigation prediction")
        print("2. View model information")
        print("3. Test prediction with custom values")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Run daily prediction
            print("\nRunning daily irrigation prediction...")
            irrigation_system.run_daily_prediction()
            
        elif choice == "2":
            # Display detailed model information
            print("\nMODEL INFORMATION")
            print("-" * 40)
            for key, value in model_status.items():
                if key == 'cluster_distribution':
                    print(f"Cluster Distribution:")
                    for cluster, count in value.items():
                        print(f"  {cluster}: {count:,} samples")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
                    
        elif choice == "3":
            # Test with custom values
            print("\nCUSTOM PREDICTION TEST")
            print("Enter field conditions (press Enter for defaults):")
            
            try:
                soil_moisture = float(input("Soil moisture % (default 50): ") or "50")
                et0 = float(input("ET0 mm/day (default 4): ") or "4")
                recent_rain = float(input("Recent rainfall 3d mm (default 5): ") or "5")
                
                # Create test features
                test_features = {
                    'soil_moisture_estimate': soil_moisture,
                    'et0': et0,
                    'cumulative_precipitation_3d': recent_rain,
                    'cumulative_precipitation_7d': recent_rain * 1.5,
                    'moisture_deficit': max(0, 70 - soil_moisture),
                    'vpd': 1.0,
                    'degree_days': 15.0,
                    'forecast_rain_24h': 0,
                    'current_conditions': {'temperature': 22, 'humidity': 65, 'wind_speed': 5, 'clouds': 50}
                }
                
                # Make prediction
                prediction = irrigation_system.predictor.predict_irrigation_need(test_features)
                cluster, decision, confidence, analysis = prediction
                
                print(f"\nTEST RESULT:")
                print(f"Decision: {decision}")
                print(f"Confidence: {confidence:.1%}")
                print(f"Cluster: {cluster}")
                
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()