"""
Irrigation Model Trainer - CSV Version
Trains an unsupervised learning model using historical weather data from CSV file
CSV Format: Visual Crossing Weather Services format
Location: Melbourne, Victoria, Australia
"""

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
    """Configuration settings for the irrigation system"""
    
    # CSV Configuration
    CSV_FILE = "melbourne_weather_history.csv"  # Path to your weather CSV file
    
    # Database
    DB_NAME = "irrigation_system.db"

    # Model Parameters
    N_CLUSTERS = 2  # No irrigation, Light irrigation, Heavy irrigation
    RANDOM_STATE = 42

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
# DATABASE MANAGEMENT
# ============================================================================

class DatabaseManager:
    """Handles all database operations"""

    def __init__(self, db_name: str = Config.DB_NAME):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize database tables with migration support"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Check if historical_weather table exists and get its columns
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='historical_weather'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(historical_weather)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # If old schema detected, drop and recreate
            if 'tempmax' not in columns or 'solarradiation' not in columns:
                logger.info("Detected old database schema. Migrating to new schema...")
                cursor.execute("DROP TABLE IF EXISTS historical_weather")
                cursor.execute("DROP TABLE IF EXISTS historical_features")
                logger.info("Old tables dropped. Creating new schema...")

        # Historical weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                temperature REAL,
                tempmax REAL,
                tempmin REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_dir REAL,
                clouds REAL,
                precipitation REAL,
                uvindex REAL,
                solarradiation REAL,
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
        logger.info("Database initialized successfully")

# ============================================================================
# CSV DATA LOADER
# ============================================================================

class CSVWeatherLoader:
    """Loads and processes weather data from CSV file"""

    def __init__(self, csv_file: str = Config.CSV_FILE, db_manager: DatabaseManager = None):
        self.csv_file = csv_file
        self.db_manager = db_manager

    def fahrenheit_to_celsius(self, f: float) -> float:
        """Convert Fahrenheit to Celsius"""
        return (f - 32) * 5/9

    def mph_to_ms(self, mph: float) -> float:
        """Convert miles per hour to meters per second"""
        return mph * 0.44704

    def inhg_to_hpa(self, inhg: float) -> float:
        """Convert inches of mercury to hectopascals"""
        return inhg * 33.8639

    def load_and_process_csv(self) -> bool:
        """Load weather data from CSV and convert units"""
        try:
            if not os.path.exists(self.csv_file):
                logger.error(f"CSV file not found: {self.csv_file}")
                return False

            logger.info(f"Loading weather data from {self.csv_file}")
            
            # Read CSV file
            df = pd.read_csv(self.csv_file)
            
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Process and convert data
            weather_data = []
            
            for _, row in df.iterrows():
                try:
                    # Parse date
                    date = pd.to_datetime(row['datetime']).date()
                    
                    # Convert temperature from Fahrenheit to Celsius
                    temp = self.fahrenheit_to_celsius(row['temp'])
                    tempmax = self.fahrenheit_to_celsius(row['tempmax'])
                    tempmin = self.fahrenheit_to_celsius(row['tempmin'])
                    
                    # Convert wind speed from mph to m/s
                    wind_speed = self.mph_to_ms(row['windspeed'])
                    
                    # Convert pressure from mbar to hPa (same units, just rename)
                    pressure = row['sealevelpressure']
                    
                    # Precipitation (already in inches, convert to mm)
                    precipitation = row['precip'] * 25.4  # inches to mm
                    
                    # UV index and solar radiation (use as-is)
                    uvindex = row['uvindex']
                    solarradiation = row['solarradiation']
                    
                    # Cloud cover percentage
                    clouds = row['cloudcover']
                    
                    # Humidity percentage
                    humidity = row['humidity']
                    
                    # Wind direction (degrees)
                    wind_dir = row['winddir']
                    
                    weather_record = {
                        'date': date,
                        'temperature': round(temp, 1),
                        'tempmax': round(tempmax, 1),
                        'tempmin': round(tempmin, 1),
                        'humidity': round(humidity, 1),
                        'pressure': round(pressure, 1),
                        'wind_speed': round(wind_speed, 1),
                        'wind_dir': round(wind_dir, 1),
                        'clouds': round(clouds, 1),
                        'precipitation': round(precipitation, 1),
                        'uvindex': round(uvindex, 1),
                        'solarradiation': round(solarradiation, 1)
                    }
                    
                    weather_data.append(weather_record)
                    
                except Exception as e:
                    logger.warning(f"Error processing row for date {row.get('datetime', 'unknown')}: {e}")
                    continue
            
            if not weather_data:
                logger.error("No valid weather data could be processed from CSV")
                return False
            
            # Save to database
            self.save_weather_data(weather_data)
            
            logger.info(f"Successfully processed and saved {len(weather_data)} days of weather data")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_weather_data(self, weather_data: List[Dict]):
        """Save weather data to database"""
        conn = sqlite3.connect(Config.DB_NAME)
        cursor = conn.cursor()
        
        # Clear existing historical data
        cursor.execute("DELETE FROM historical_weather")
        
        for data in weather_data:
            cursor.execute('''
                INSERT INTO historical_weather
                (date, temperature, tempmax, tempmin, humidity, pressure, 
                 wind_speed, wind_dir, clouds, precipitation, uvindex, solarradiation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['date'],
                data['temperature'],
                data['tempmax'],
                data['tempmin'],
                data['humidity'],
                data['pressure'],
                data['wind_speed'],
                data['wind_dir'],
                data['clouds'],
                data['precipitation'],
                data['uvindex'],
                data['solarradiation']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(weather_data)} weather records to database")

# ============================================================================
# HISTORICAL FEATURE ENGINEERING
# ============================================================================

class HistoricalFeatureEngineer:
    """Handles feature engineering for historical data"""

    def __init__(self):
        self.db_manager = DatabaseManager()

    def calculate_et0_with_solar(self, temp: float, humidity: float, 
                                  wind_speed: float, solar_radiation: float) -> float:
        """
        Calculate reference evapotranspiration (ET0) using FAO Penman-Monteith
        with actual solar radiation data
        """
        # Saturation vapor pressure (kPa)
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        
        # Actual vapor pressure (kPa)
        ea = es * (humidity / 100)
        
        # Vapor pressure deficit (kPa)
        vpd = es - ea
        
        # Convert solar radiation from W/m² to MJ/m²/day
        solar_radiation_mj = (solar_radiation * 86400) / 1000000
        
        # Simplified Penman-Monteith equation
        # Radiation component
        et0_rad = 0.408 * 0.77 * solar_radiation_mj
        
        # Aerodynamic component
        et0_aero = (900 / (temp + 273)) * wind_speed * vpd
        
        # Total ET0 (mm/day)
        et0 = et0_rad + et0_aero
        
        return max(0, et0)

    def calculate_vpd(self, temp: float, humidity: float) -> float:
        """Calculate Vapor Pressure Deficit (kPa)"""
        svp = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        avp = svp * (humidity / 100)
        return svp - avp

    def calculate_degree_days(self, tempmax: float, tempmin: float, base_temp: float = 10) -> float:
        """Calculate degree days for crop development using max/min temps"""
        temp_avg = (tempmax + tempmin) / 2
        return max(0, temp_avg - base_temp)

    def estimate_soil_moisture(self, precip_3d: float, et0_3d: float, 
                               base_moisture: float = 70) -> float:
        """Estimate soil moisture based on water balance"""
        # Water loss from ET (mm converted to moisture %)
        moisture_loss = et0_3d * 1.5
        
        # Water gain from precipitation (mm converted to moisture %)
        # Assuming 1mm rain = ~1% soil moisture increase for top 30cm
        moisture_gain = precip_3d * 0.8
        
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
            
            # Calculate ET0 using actual solar radiation
            et0_current = self.calculate_et0_with_solar(
                row['temperature'],
                row['humidity'],
                row['wind_speed'],
                row['solarradiation']
            )
            
            # Calculate 3-day cumulative ET0
            et0_3d = weather_df[
                (pd.to_datetime(weather_df['date']) >= date_3d_ago) &
                (pd.to_datetime(weather_df['date']) <= current_date)
            ].apply(lambda x: self.calculate_et0_with_solar(
                x['temperature'], x['humidity'], x['wind_speed'], x['solarradiation']
            ), axis=1).sum()
            
            # Calculate other features
            vpd = self.calculate_vpd(row['temperature'], row['humidity'])
            degree_days = self.calculate_degree_days(row['tempmax'], row['tempmin'])
            
            # Estimate soil moisture
            soil_moisture = self.estimate_soil_moisture(precip_3d, et0_3d)
            
            # Calculate moisture deficit
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
        
        if df.empty or len(df) < 30:
            logger.error("Insufficient training data. Need at least 30 samples.")
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
        print("HIiiiiiii",self.pca.n_components_)

        
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
        
        # Calculate cluster characteristics
        cluster_stats = self.analyze_clusters(df, labels)
        
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
            },
            'cluster_characteristics': cluster_stats
        }
        
        self.save_model_info(model_info)
        
        logger.info(f"Model training completed successfully!")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Explained Variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return silhouette, model_info

    def analyze_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Analyze cluster characteristics for interpretation"""
        cluster_stats = {}
        
        for i in range(Config.N_CLUSTERS):
            cluster_data = df[labels == i]
            
            cluster_stats[f'cluster_{i}'] = {
                'avg_soil_moisture': float(cluster_data['soil_moisture_estimate'].mean()),
                'avg_et0': float(cluster_data['et0'].mean()),
                'avg_moisture_deficit': float(cluster_data['moisture_deficit'].mean()),
                'avg_precip_3d': float(cluster_data['cumulative_precipitation_3d'].mean())
            }
        
        return cluster_stats

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
    def visualize_clusters(self, df: pd.DataFrame = None):
        """
        Visualize K-Means clusters in 2D using PCA-transformed features.
        If df is not provided, it loads the training data.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load data if not provided
        if df is None:
            df = self.load_training_data()

        X = df[self.feature_columns].fillna(df.mean())

        # Transform using saved scaler and PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Get cluster labels
        if self.kmeans is None:
            # Load K-Means if not in memory
            with open(Config.KMEANS_FILE, 'rb') as f:
                self.kmeans = pickle.load(f)
        
        labels = self.kmeans.predict(X_pca)

        # Plot clusters
        plt.figure(figsize=(8,6))
        sns.scatterplot(
            x=X_pca[:,0],
            y=X_pca[:,1],
            hue=labels,
            palette='tab10',
            s=60,
            alpha=0.7
        )

        # Plot centroids
        centers = self.kmeans.cluster_centers_
        plt.scatter(
            centers[:,0], centers[:,1],
            c='black', s=100, marker='X', label='Centroids'
        )

        plt.title("K-Means Clusters (2D PCA Projection)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()

# ============================================================================
# MAIN TRAINER EXECUTION
# ============================================================================

def main():
    """Main training execution function"""
    print("\n" + "="*60)
    print("IRRIGATION MODEL TRAINER - CSV VERSION")
    print("Location: Melbourne, Victoria, Australia")
    print("Data Source: Visual Crossing Weather Services CSV")
    print("="*60 + "\n")

    try:
        # Step 1: Initialize database first (this will handle migration)
        logger.info("Initializing database...")
        db_manager = DatabaseManager()
        
        # Step 2: Initialize other components
        logger.info("Initializing trainer components...")
        csv_loader = CSVWeatherLoader(db_manager=db_manager)
        feature_engineer = HistoricalFeatureEngineer()
        model_trainer = IrrigationModelTrainer()

        # Step 3: Load and process CSV data
        logger.info(f"Loading weather data from CSV: {Config.CSV_FILE}")
        if not csv_loader.load_and_process_csv():
            logger.error("Failed to load CSV data. Please check the file path and format.")
            return

        # Step 4: Engineer features
        logger.info("Engineering features from historical data...")
        feature_engineer.engineer_historical_features()

        # Step 5: Train model
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
            
            print("\nCluster Characteristics:")
            for cluster, stats in model_info['cluster_characteristics'].items():
                print(f"\n  {cluster}:")
                print(f"    Avg Soil Moisture: {stats['avg_soil_moisture']:.1f}%")
                print(f"    Avg ET0: {stats['avg_et0']:.2f} mm/day")
                print(f"    Avg Moisture Deficit: {stats['avg_moisture_deficit']:.1f}%")
                print(f"    Avg 3-day Precip: {stats['avg_precip_3d']:.1f} mm")
            
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
                print("⚠️  Model quality could be improved - Consider more data")
        
        else:
            logger.error("Model training failed")

        #model_trainer.visualize_clusters() 

    except Exception as e:
        logger.error(f"Training process failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()