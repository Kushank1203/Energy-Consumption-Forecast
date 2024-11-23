import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(file_path):
    """
    Load and preprocess power consumption data from a text file.
    
    Args:
        file_path (str): Path to the input text file
        
    Returns:
        pd.DataFrame: Processed DataFrame or None if error occurs
    """
    try:
        # Define data types for each column
        dtype_dict = {
            'Date': str,
            'Time': str,
            'Global_active_power': float,
            'Global_reactive_power': float,
            'Voltage': float,
            'Global_intensity': float,
            'Sub_metering_1': float,
            'Sub_metering_2': float,
            'Sub_metering_3': float
        }
        
        # Read the txt file with specified data types
        df = pd.read_csv(
            file_path,
            sep=';',
            encoding='utf-8',
            dtype=dtype_dict,
            na_values=['?']  # Handle missing values marked as '?'
        )
        
        # Convert datetime after reading
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                      format='%d/%m/%Y %H:%M:%S')
        
        # Drop original Date and Time columns
        df = df.drop(['Date', 'Time'], axis=1)
        
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Set datetime as index
        df = df.set_index('datetime')
        
        print("Data loaded successfully!")
        print(f"Shape of data: {df.shape}")
        
        # Display data info
        print("\nData Types:")
        print(df.dtypes)
        print("\nFirst few rows of processed data:")
        print(df.head())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing values:")
        print(missing_values)
        
        # Handle missing values if any exist
        if missing_values.sum() > 0:
            print("\nHandling missing values...")
            # First try linear interpolation
            df = df.interpolate(method='linear', limit_direction='both')
            print("Missing values after handling:")
            print(df.isnull().sum())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def prepare_features(df):
    """
    Prepare time-based features from the datetime index.
    
    Args:
        df (pd.DataFrame): Input DataFrame with datetime index
        
    Returns:
        pd.DataFrame: DataFrame with additional time-based features
    """
    df_features = df.copy()
    
    # Extract time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['year'] = df_features.index.year
    
    # Add lag features for Global_active_power
    df_features['power_lag_1h'] = df_features['Global_active_power'].shift(1)
    df_features['power_lag_24h'] = df_features['Global_active_power'].shift(24)
    df_features['power_lag_7d'] = df_features['Global_active_power'].shift(24*7)
    
    # Drop rows with NaN values created by shifting
    df_features = df_features.dropna()
    
    return df_features

def train_model(df):
    """
    Train an XGBoost model for power consumption prediction.
    
    Args:
        df (pd.DataFrame): Processed DataFrame with features
        
    Returns:
        tuple: (trained model, scaler, feature names)
    """
    # Prepare features
    df_features = prepare_features(df)
    
    # Define target and features
    target = 'Global_active_power'
    features = [col for col in df_features.columns if col != target]
    
    X = df_features[features]
    y = df_features[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, scaler, features

def plot_results(model, df, scaler, features):
    """
    Plot feature importance and actual vs predicted values.
    
    Args:
        model: Trained XGBoost model
        df (pd.DataFrame): Original DataFrame
        scaler: Fitted StandardScaler
        features (list): Feature names
    """
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

# def save_model(model, scaler, features, model_path='power_model.joblib'):
    """
    Save the trained model and associated objects.
    
    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        features (list): Feature names
        model_path (str): Path to save the model
    """
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features
    }
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")

def main():
    # File path
    file_path = 'household_power_consumption.txt'
    
    # Step 1: Load data
    df = load_data(file_path)
    if df is None:
        return
    
    # Step 2: Train model
    model, scaler, features = train_model(df)
    
    # Step 3: Plot results
    plot_results(model, df, scaler, features)
    
    # # Step 4: Save model
    # save_model(model, scaler, features)

if __name__ == "__main__":
    main()