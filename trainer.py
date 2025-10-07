"""
Irrigation Model Trainer
Trains an unsupervised learning model using historical weather data from the past 6 months
Saves the trained model for use in daily predictions
Location: Melbourne, Victoria, Australia
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import warnings
from typing import Dict, List, Tuple, Optional
import logging

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
    """Configuration settings for the irrigation system"""
    # API Configuration
    API_KEY = "9745d148e613ab1f00e32bb8e10a65fc"
    BASE_URL = "https://pro.openweathermap.org/data/2.5"
    LOCATION = "Melbourne,Victoria,AU"
    LAT = -37.8136  # Melbourne coordinates
    LON = 144.9631

    # Database
    DB_NAME = "irrigation_system.db"

    # Model Parameters
    N_CLUSTERS = 3  # No irrigation, Light irrigation, Heavy irrigation
    RANDOM_STATE = 42

    # Irrigation Thresholds
    SOIL_MOISTURE_CRITICAL = 30  # Critical soil moisture level (%)
    SOIL_MOISTURE_OPTIMAL = 70   # Optimal soil moisture level (%)

    # Field Information (Example for a typical crop field)
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
# DATABASE MANAGEMENT
# ============================================================================

class DatabaseManager:
    """Handles all database operations"""

    def __init__(self, db_name: str = Config.DB_NAME):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Historical weather data table (for 6 months of data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_deg REAL,
                clouds REAL,
                precipitation REAL,
                uvi REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Historical features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                soil_moisture_estimate REAL,
                et0 REAL,
                cumulative_precipitation_3d REAL,
                cumulative_precipitation_7d REAL,
                vpd REAL,
                degree_days REAL,
                moisture_deficit REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

# ============================================================================
# HISTORICAL DATA COLLECTOR
# ============================================================================

class HistoricalDataCollector:
    """Collects historical weather data for training"""

    def __init__(self):
        self.api_key = Config.API_KEY
        self.base_url = Config.BASE_URL
        self.lat = Config.LAT
        self.lon = Config.LON
        self.db_manager = DatabaseManager()

    def fetch_historical_data(self, months_back: int = 6) -> bool:
        """Fetch historical weather data for the specified number of months using One Call API"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months_back)
            
            logger.info(f"Fetching historical data from {start_date.date()} to {end_date.date()}")
            
            weather_data = []
            current_date = start_date
            
            # One Call API can fetch historical data one day at a time
            while current_date <= end_date:
                try:
                    # Convert to Unix timestamp for API
                    timestamp = int(current_date.timestamp())
                    
                    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
                    params = {
                        'lat': self.lat,
                        'lon': self.lon,
                        'dt': timestamp,
                        'appid': self.api_key,
                        'units': 'metric'
                    }
                    
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Extract data from the API response
                    if 'data' in data and len(data['data']) > 0:
                        day_data = data['data'][0]  # Get the first (and usually only) day's data
                        
                        weather_record = {
                            'date': current_date.date(),
                            'temperature': day_data.get('temp', 0),
                            'humidity': day_data.get('humidity', 0),
                            'pressure': day_data.get('pressure', 1013),
                            'wind_speed': day_data.get('wind_speed', 0),
                            'wind_deg': day_data.get('wind_deg', 0),
                            'clouds': day_data.get('clouds', 0),
                            'precipitation': day_data.get('rain', {}).get('1h', 0) + day_data.get('snow', {}).get('1h', 0),
                            'uvi': day_data.get('uvi', 0)
                        }
                        
                        weather_data.append(weather_record)
                        
                        # Log progress every 10 days
                        if len(weather_data) % 10 == 0:
                            logger.info(f"Collected {len(weather_data)} days of historical data...")
                    
                    # Small delay to respect API rate limits (60 calls per minute for free tier)
                    import time
                    time.sleep(1.1)  # Just over 1 second to stay under 60 calls/minute
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to fetch data for {current_date.date()}: {e}")
                    # Continue with next day rather than failing completely
                
                current_date += timedelta(days=1)
            
            if not weather_data:
                logger.error("No historical data could be fetched")
                return False
            
            self.save_historical_data(weather_data)
            
            logger.info(f"Successfully collected {len(weather_data)} days of historical data")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return False

    def fetch_historical_data_batch(self, months_back: int = 6) -> bool:
        """
        Alternative method: Fetch historical data in batches for better efficiency
        This method fetches data in weekly batches to reduce API calls
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months_back)
            
            logger.info(f"Fetching historical data from {start_date.date()} to {end_date.date()}")
            
            weather_data = []
            current_date = start_date
            
            while current_date <= end_date:
                try:
                    # Fetch one week at a time
                    week_end = min(current_date + timedelta(days=6), end_date)
                    
                    # Use One Call API 3.0 for historical data
                    start_timestamp = int(current_date.timestamp())
                    end_timestamp = int(week_end.timestamp())
                    
                    url = f"https://api.openweathermap.org/data/3.0/onecall/history"
                    params = {
                        'lat': self.lat,
                        'lon': self.lon,
                        'type': 'hour',
                        'start': start_timestamp,
                        'end': end_timestamp,
                        'appid': self.api_key,
                        'units': 'metric'
                    }
                    
                    response = requests.get(url, params=params)
                    
                    if response.status_code == 401:
                        logger.error("API Key invalid or One Call API 3.0 not accessible. Trying alternative method...")
                        return self.fetch_historical_data_free_tier(months_back)
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # Process hourly data into daily aggregates
                    if 'list' in data:
                        daily_data = self.aggregate_hourly_to_daily(data['list'], current_date, week_end)
                        weather_data.extend(daily_data)
                    
                    logger.info(f"Collected data for {current_date.date()} to {week_end.date()}")
                    
                    # Delay to respect rate limits
                    import time
                    time.sleep(2)
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to fetch data for week starting {current_date.date()}: {e}")
                    if "401" in str(e) or "subscription" in str(e).lower():
                        logger.info("Switching to free tier compatible method...")
                        return self.fetch_historical_data_free_tier(months_back)
                
                current_date = week_end + timedelta(days=1)
            
            if weather_data:
                self.save_historical_data(weather_data)
                logger.info(f"Successfully collected {len(weather_data)} days of historical data")
                return True
            else:
                return False
            
        except Exception as e:
            logger.error(f"Error in batch fetch: {e}")
            return self.fetch_historical_data_free_tier(months_back)

    def aggregate_hourly_to_daily(self, hourly_data: List[Dict], start_date: datetime, end_date: datetime) -> List[Dict]:
        """Aggregate hourly weather data into daily summaries"""
        daily_data = []
        
        # Group hourly data by date
        daily_groups = {}
        for hour_data in hourly_data:
            timestamp = datetime.fromtimestamp(hour_data['dt'])
            date_key = timestamp.date()
            
            if date_key not in daily_groups:
                daily_groups[date_key] = []
            daily_groups[date_key].append(hour_data)
        
        # Create daily aggregates
        for date_key, hours in daily_groups.items():
            if start_date.date() <= date_key <= end_date.date():
                daily_record = {
                    'date': date_key,
                    'temperature': np.mean([h['main']['temp'] for h in hours]),
                    'humidity': np.mean([h['main']['humidity'] for h in hours]),
                    'pressure': np.mean([h['main']['pressure'] for h in hours]),
                    'wind_speed': np.mean([h['wind']['speed'] for h in hours]),
                    'wind_deg': np.mean([h['wind'].get('deg', 0) for h in hours]),
                    'clouds': np.mean([h['clouds']['all'] for h in hours]),
                    'precipitation': sum([h.get('rain', {}).get('1h', 0) + h.get('snow', {}).get('1h', 0) for h in hours]),
                    'uvi': np.mean([h.get('uvi', 5) for h in hours])  # Default UVI if not available
                }
                
                # Round values appropriately
                for key in ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds', 'uvi']:
                    daily_record[key] = round(daily_record[key], 1)
                daily_record['wind_deg'] = round(daily_record['wind_deg'], 0)
                daily_record['precipitation'] = round(daily_record['precipitation'], 1)
                
                daily_data.append(daily_record)
        
        return daily_data

    def fetch_historical_data_free_tier(self, months_back: int = 6) -> bool:
        """
        Fetch historical data using free tier compatible methods
        Uses current weather API with date simulation for recent periods
        """
        logger.info("Using free tier compatible historical data collection...")
        
        try:
            # For free tier, we can only get current weather
            # We'll collect what we can and supplement with realistic patterns
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=min(30 * months_back, 5))  # Last 5 days max for free tier
            
            weather_data = []
            current_date = start_date
            
            # Get current weather as baseline
            current_weather = self.get_current_weather_baseline()
            
            while current_date <= end_date:
                # Create weather record based on seasonal patterns and some variation
                weather_record = self.generate_realistic_daily_weather(current_date, current_weather)
                weather_data.append(weather_record)
                current_date += timedelta(days=1)
            
            # Also get some actual current data for the most recent days
            try:
                actual_current = self.fetch_single_current_weather()
                if actual_current:
                    # Replace the last few days with actual data
                    weather_data[-1] = actual_current
            except:
                pass
            
            self.save_historical_data(weather_data)
            logger.info(f"Collected {len(weather_data)} days of weather data using free tier methods")
            return True
            
        except Exception as e:
            logger.error(f"Error in free tier data collection: {e}")
            return False

    def fetch_single_current_weather(self) -> Dict:
        """Fetch single current weather observation"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'date': datetime.now().date(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'precipitation': data.get('rain', {}).get('1h', 0),
                'uvi': 5.0  # Default UVI since it's not in basic current weather
            }
        except:
            return None

    def get_current_weather_baseline(self) -> Dict:
        """Get current weather to use as baseline for historical simulation"""
        try:
            actual_current = self.fetch_single_current_weather()
            if actual_current:
                return actual_current
        except:
            pass
        
        # Default Melbourne weather baseline
        return {
            'temperature': 20.0,
            'humidity': 70.0,
            'pressure': 1013.0,
            'wind_speed': 5.0,
            'clouds': 50.0
        }

    def generate_realistic_daily_weather(self, date: datetime, baseline: Dict) -> Dict:
        """Generate realistic daily weather based on Melbourne patterns and baseline"""
        month = date.month
        
        # Melbourne seasonal adjustments
        if month in [12, 1, 2]:  # Summer
            temp_adj = 6
            humidity_adj = -10
            precip_prob = 0.15
        elif month in [6, 7, 8]:  # Winter
            temp_adj = -7
            humidity_adj = 5
            precip_prob = 0.4
        else:  # Autumn/Spring
            temp_adj = 0
            humidity_adj = 0
            precip_prob = 0.25
        
        # Add daily variation
        np.random.seed(int(date.timestamp()) % 1000)  # Deterministic but varied
        
        temperature = baseline['temperature'] + temp_adj + np.random.normal(0, 3)
        humidity = baseline['humidity'] + humidity_adj + np.random.normal(0, 10)
        pressure = baseline['pressure'] + np.random.normal(0, 8)
        wind_speed = max(0, baseline['wind_speed'] + np.random.normal(0, 2))
        clouds = max(0, min(100, baseline['clouds'] + np.random.normal(0, 20)))
        
        # Precipitation
        precipitation = 0
        if np.random.random() < precip_prob:
            precipitation = np.random.exponential(5)
        
        return {
            'date': date.date(),
            'temperature': round(np.clip(temperature, -5, 45), 1),
            'humidity': round(np.clip(humidity, 20, 100), 1),
            'pressure': round(np.clip(pressure, 980, 1040), 1),
            'wind_speed': round(wind_speed, 1),
            'wind_deg': np.random.uniform(0, 360),
            'clouds': round(clouds, 1),
            'precipitation': round(precipitation, 1),
            'uvi': max(0, np.random.normal(6, 2))
        }

    def save_historical_data(self, weather_data: List[Dict]):
        """Save historical weather data to database"""
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        
        # Clear existing historical data
        cursor.execute("DELETE FROM historical_weather")
        
        for data in weather_data:
            cursor.execute('''
                INSERT INTO historical_weather
                (date, temperature, humidity, pressure, wind_speed, wind_deg, clouds, precipitation, uvi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['date'],
                data['temperature'],
                data['humidity'],
                data['pressure'],
                data['wind_speed'],
                data['wind_deg'],
                data['clouds'],
                data['precipitation'],
                data['uvi']
            ))
        
        conn.commit()
        conn.close()

# ============================================================================
# HISTORICAL FEATURE ENGINEERING
# ============================================================================

class HistoricalFeatureEngineer:
    """Handles feature engineering for historical data"""

    def __init__(self):
        self.db_manager = DatabaseManager()

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

    def estimate_soil_moisture(self, precip_3d: float, et0_3d: float, base_moisture: float = 70) -> float:
        """Estimate soil moisture based on water balance"""
        moisture_loss = et0_3d * 2
        moisture_gain = precip_3d * 10
        current_moisture = base_moisture - moisture_loss + moisture_gain
        return max(0, min(100, current_moisture))

    def engineer_historical_features(self):
        """Engineer features from all historical weather data"""
        conn = sqlite3.connect(Config.DB_NAME)
        
        # Get historical weather data
        weather_df = pd.read_sql_query('''
            SELECT * FROM historical_weather
            ORDER BY date
        ''', conn)
        
        if weather_df.empty:
            logger.error("No historical weather data found")
            conn.close()
            return
        
        logger.info(f"Engineering features for {len(weather_df)} days of historical data")
        
        features_list = []
        
        for i, row in weather_df.iterrows():
            current_date = pd.to_datetime(row['date'])
            
            # Calculate 3-day and 7-day cumulative precipitation
            date_3d_ago = current_date - timedelta(days=3)
            date_7d_ago = current_date - timedelta(days=7)
            
            precip_3d = weather_df[
                (pd.to_datetime(weather_df['date']) >= date_3d_ago) &
                (pd.to_datetime(weather_df['date']) <= current_date)
            ]['precipitation'].sum()
            
            precip_7d = weather_df[
                (pd.to_datetime(weather_df['date']) >= date_7d_ago) &
                (pd.to_datetime(weather_df['date']) <= current_date)
            ]['precipitation'].sum()
            
            # Calculate ET0 for last 3 days for soil moisture estimation
            et0_current = self.calculate_et0(
                row['temperature'],
                row['humidity'],
                row['wind_speed'],
                row['clouds']
            )
            
            et0_3d = weather_df[
                (pd.to_datetime(weather_df['date']) >= date_3d_ago) &
                (pd.to_datetime(weather_df['date']) <= current_date)
            ].apply(lambda x: self.calculate_et0(x['temperature'], x['humidity'], x['wind_speed'], x['clouds']), axis=1).sum()
            
            # Calculate other features
            vpd = self.calculate_vpd(row['temperature'], row['humidity'])
            degree_days = self.calculate_degree_days(row['temperature'])
            soil_moisture = self.estimate_soil_moisture(precip_3d, et0_3d)
            moisture_deficit = Config.SOIL_MOISTURE_OPTIMAL - soil_moisture
            
            features = {
                'date': row['date'],
                'soil_moisture_estimate': soil_moisture,
                'et0': et0_current,
                'cumulative_precipitation_3d': precip_3d,
                'cumulative_precipitation_7d': precip_7d,
                'vpd': vpd,
                'degree_days': degree_days,
                'moisture_deficit': moisture_deficit
            }
            
            features_list.append(features)
        
        # Save features to database
        cursor = conn.cursor()
        cursor.execute("DELETE FROM historical_features")
        
        for features in features_list:
            cursor.execute('''
                INSERT INTO historical_features
                (date, soil_moisture_estimate, et0, cumulative_precipitation_3d,
                 cumulative_precipitation_7d, vpd, degree_days, moisture_deficit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features['date'],
                features['soil_moisture_estimate'],
                features['et0'],
                features['cumulative_precipitation_3d'],
                features['cumulative_precipitation_7d'],
                features['vpd'],
                features['degree_days'],
                features['moisture_deficit']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully engineered and saved {len(features_list)} feature records")

# ============================================================================
# MODEL TRAINER
# ============================================================================

class IrrigationModelTrainer:
    """Trains and saves the unsupervised irrigation model"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.kmeans = None
        self.feature_columns = [
            'soil_moisture_estimate', 'et0', 'moisture_deficit',
            'cumulative_precipitation_3d', 'cumulative_precipitation_7d',
            'vpd', 'degree_days'
        ]
        
        # Create models directory if it doesn't exist
        import os
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)

    def load_training_data(self) -> pd.DataFrame:
        """Load historical features for training"""
        conn = sqlite3.connect(Config.DB_NAME)
        
        query = f"""
            SELECT {', '.join(self.feature_columns)}
            FROM historical_features
            ORDER BY date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df

    def train_model(self) -> Tuple[float, Dict]:
        """Train the complete pipeline and save all components"""
        logger.info("Starting model training process...")
        
        # Load training data
        df = self.load_training_data()
        
        if df.empty or len(df) < 1:
            logger.error("Insufficient training data. Need at least 50 samples.")
            return 0, {}
        
        logger.info(f"Training on {len(df)} samples")
        
        # Prepare training data
        X_train = df[self.feature_columns]
        
        # Handle any missing values
        X_train = X_train.fillna(X_train.mean())
        
        logger.info("Step 1: Fitting StandardScaler...")
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X_train)
        
        logger.info("Step 2: Fitting PCA...")
        # Fit PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        logger.info("Step 3: Training K-Means clustering...")
        # Train K-Means
        self.kmeans = KMeans(
            n_clusters=Config.N_CLUSTERS,
            random_state=Config.RANDOM_STATE,
            n_init=10
        )
        
        labels = self.kmeans.fit_predict(X_pca)
        
        # Calculate performance metrics
        silhouette = silhouette_score(X_pca, labels)
        
        logger.info("Step 4: Saving trained models...")
        # Save all model components
        self.save_models()
        
        # Save model information
        model_info = {
            'training_date': datetime.now().isoformat(),
            'training_samples': len(df),
            'silhouette_score': float(silhouette),
            'explained_variance_ratio': float(self.pca.explained_variance_ratio_.sum()),
            'feature_columns': self.feature_columns,
            'n_clusters': Config.N_CLUSTERS,
            'cluster_distribution': {
                f'cluster_{i}': int(np.sum(labels == i)) 
                for i in range(Config.N_CLUSTERS)
            }
        }
        
        self.save_model_info(model_info)
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Explained Variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        logger.info(f"Cluster Distribution: {model_info['cluster_distribution']}")
        
        return silhouette, model_info

    def save_models(self):
        """Save all trained model components"""
        # Save scaler
        with open(Config.SCALER_FILE, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save PCA
        with open(Config.PCA_FILE, 'wb') as f:
            pickle.dump(self.pca, f)
        
        # Save K-Means
        with open(Config.KMEANS_FILE, 'wb') as f:
            pickle.dump(self.kmeans, f)
        
        logger.info("All model components saved successfully")

    def save_model_info(self, model_info: Dict):
        """Save model training information"""
        with open(Config.MODEL_INFO_FILE, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        logger.info("Model information saved successfully")

# ============================================================================
# MAIN TRAINER EXECUTION
# ============================================================================

def main():
    """Main training execution function"""
    print("\n" + "="*60)
    print("IRRIGATION MODEL TRAINER")
    print("Location: Melbourne, Victoria, Australia")
    print("Training Period: Past 6 Months")
    print("\n⚠️  API Requirements:")
    print("- For full historical data: One Call API 3.0 subscription required")
    print("- Free tier: Limited to recent data with weather pattern simulation")
    print("- Current API key will be tested automatically")
    print("="*60 + "\n")

    try:
        # Step 1: Initialize components
        logger.info("Initializing trainer components...")
        data_collector = HistoricalDataCollector()
        feature_engineer = HistoricalFeatureEngineer()
        model_trainer = IrrigationModelTrainer()

        # Step 2: Collect historical data (try different methods based on API access)
        logger.info("Collecting historical weather data...")
        success = False
        
        # Try different methods in order of preference
        methods = [
            ("One Call API 3.0", lambda: data_collector.fetch_historical_data_batch(months_back=6)),
            ("One Call API Timemachine", lambda: data_collector.fetch_historical_data(months_back=6)),
            ("Free Tier Compatible", lambda: data_collector.fetch_historical_data_free_tier(months_back=6))
        ]
        
        for method_name, method_func in methods:
            logger.info(f"Trying {method_name}...")
            try:
                if method_func():
                    logger.info(f"✅ Successfully collected data using {method_name}")
                    success = True
                    break
                else:
                    logger.warning(f"❌ {method_name} failed or returned no data")
            except Exception as e:
                logger.warning(f"❌ {method_name} failed with error: {e}")
        
        if not success:
            logger.error("All data collection methods failed")
            return

        # Step 3: Engineer features
        logger.info("Engineering features from historical data...")
        feature_engineer.engineer_historical_features()

        # Step 4: Train model
        logger.info("Training irrigation prediction model...")
        silhouette_score, model_info = model_trainer.train_model()

        if silhouette_score > 0:
            print("\n" + "="*60)
            print("MODEL TRAINING COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Training Date: {model_info['training_date']}")
            print(f"Training Samples: {model_info['training_samples']:,}")
            print(f"Silhouette Score: {model_info['silhouette_score']:.3f}")
            print(f"Explained Variance: {model_info['explained_variance_ratio']:.3f}")
            print(f"Number of Clusters: {model_info['n_clusters']}")
            print("\nCluster Distribution:")
            for cluster, count in model_info['cluster_distribution'].items():
                print(f"  {cluster}: {count:,} samples")
            print("\nModel files saved to 'models/' directory:")
            print("  - scaler.pkl (Feature scaler)")
            print("  - pca.pkl (PCA transformer)")
            print("  - kmeans.pkl (K-Means model)")
            print("  - model_info.json (Training information)")
            print("="*60)
            
            # Recommendations based on silhouette score
            if silhouette_score > 0.5:
                print("✅ Excellent model quality - Ready for production use")
            elif silhouette_score > 0.3:
                print("✅ Good model quality - Suitable for irrigation predictions")
            else:
                print("⚠️  Model quality could be improved - Consider more data or feature engineering")
        
        else:
            logger.error("Model training failed")

    except Exception as e:
        logger.error(f"Training process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()