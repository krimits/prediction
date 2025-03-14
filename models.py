import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def calculate_metrics(y_true, y_pred):
    """
    Calculate prediction performance metrics
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        
    Returns:
        dict: Dictionary containing RMSE and MAE metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }

def prepare_data(df, prediction_days):
    """
    Prepare data for model training
    
    Args:
        df (pandas.DataFrame): DataFrame with features
        prediction_days (int): Number of days to predict
        
    Returns:
        tuple: X_train, y_train, X_test, y_test, scaler
    """
    # Features to use for prediction
    features = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band']
    
    # Use available features
    available_features = [f for f in features if f in df.columns]
    data = df[available_features].values
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Split data into training and testing sets (80% train, 20% test)
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Create sequences for training data
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i])
        y_train.append(train_data[i, 0])  # 0 index is the Close price
    
    # Create sequences for testing data
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i])
        y_test.append(test_data[i, 0])
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    return X_train, y_train, X_test, y_test, scaler, available_features

def train_linear_model(df, prediction_days):
    """
    Train a linear regression model and make predictions
    
    Args:
        df (pandas.DataFrame): DataFrame with features
        prediction_days (int): Number of days to predict
        
    Returns:
        tuple: predictions, accuracy, metrics
    """
    try:
        # Prepare data
        X_train, y_train, X_test, y_test, scaler, features = prepare_data(df, prediction_days)
        
        # Reshape data for linear regression
        X_train_lr = X_train.reshape(X_train.shape[0], -1)
        X_test_lr = X_test.reshape(X_test.shape[0], -1)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train_lr, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_lr)
        accuracy = 100 - (mean_absolute_percentage_error(y_test, y_pred) * 100)
        
        # Calculate additional metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Make future predictions
        last_sequence = df[features].values[-60:]
        scaled_last_sequence = scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = scaled_last_sequence.copy()
        
        for _ in range(prediction_days):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, -1)
            
            # Predict next value
            next_pred = model.predict(X_pred)[0]
            
            # Add prediction to future predictions
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            next_data_point = np.zeros((1, len(features)))
            next_data_point[0, 0] = next_pred  # Set Close price
            
            # Shift sequence and add new prediction
            current_sequence = np.vstack((current_sequence[1:], next_data_point))
        
        # Inverse transform to get actual prices
        future_data_points = np.zeros((len(future_predictions), len(features)))
        future_data_points[:, 0] = future_predictions  # Set Close price column
        
        # Use the last known values for other features
        for i in range(1, len(features)):
            future_data_points[:, i] = scaled_last_sequence[-1, i]
        
        predicted_prices = scaler.inverse_transform(future_data_points)[:, 0]
        
        return predicted_prices.tolist(), accuracy, metrics
    
    except Exception as e:
        print(f"Error in linear model: {str(e)}")
        return [], 0, {'RMSE': 0, 'MAE': 0}

def train_lstm_model(df, prediction_days):
    """
    Train an LSTM model and make predictions
    
    Args:
        df (pandas.DataFrame): DataFrame with features
        prediction_days (int): Number of days to predict
        
    Returns:
        tuple: predictions, accuracy, metrics
    """
    try:
        # Prepare data
        X_train, y_train, X_test, y_test, scaler, features = prepare_data(df, prediction_days)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = 100 - (mean_absolute_percentage_error(y_test, y_pred) * 100)
        
        # Calculate additional metrics
        metrics = calculate_metrics(y_test, y_pred)
        
        # Make future predictions
        last_sequence = df[features].values[-60:]
        scaled_last_sequence = scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = scaled_last_sequence.reshape(1, scaled_last_sequence.shape[0], scaled_last_sequence.shape[1])
        
        for _ in range(prediction_days):
            # Predict next value
            next_pred = model.predict(current_sequence)[0, 0]
            
            # Add prediction to future predictions
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            next_data_point = np.zeros((1, 1, len(features)))
            next_data_point[0, 0, 0] = next_pred  # Set Close price
            
            # Copy other features from the last known values
            for i in range(1, len(features)):
                next_data_point[0, 0, i] = current_sequence[0, -1, i]
            
            # Shift sequence and add new prediction
            current_sequence = np.concatenate((current_sequence[:, 1:, :], next_data_point), axis=1)
        
        # Inverse transform to get actual prices
        future_data_points = np.zeros((len(future_predictions), len(features)))
        future_data_points[:, 0] = future_predictions  # Set Close price column
        
        # Use the last known values for other features
        for i in range(1, len(features)):
            future_data_points[:, i] = scaled_last_sequence[-1, i]
        
        predicted_prices = scaler.inverse_transform(future_data_points)[:, 0]
        
        return predicted_prices.tolist(), accuracy, metrics
    
    except Exception as e:
        print(f"Error in LSTM model: {str(e)}")
        return [], 0, {'RMSE': 0, 'MAE': 0}
