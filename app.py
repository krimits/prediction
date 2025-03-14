import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

from data_loader import load_crypto_data
from models import train_linear_model, train_lstm_model
from technical_indicators import add_technical_indicators

# Page configuration and styling
def setup_page():
    st.set_page_config(
        page_title="Crypto Price Predictor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stAlert > div {
            padding: 0.5rem 1rem;
            margin-bottom: 1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

# Sidebar controls
def create_sidebar():
    with st.sidebar:
        st.title("Settings")
        
        crypto_symbol = st.selectbox(
            "Select Cryptocurrency",
            ["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "SOL-USD"]
        )
        
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1mo", "3mo", "6mo", "1y"]
        )
        
        prediction_days = st.slider(
            "Prediction Days",
            min_value=7,
            max_value=30,
            value=14,
            help="Number of days to predict into the future"
        )
        
        model_type = st.selectbox(
            "Select Model",
            ["Linear Regression", "Neural Network", "Both"]
        )
        
        return crypto_symbol, timeframe, prediction_days, model_type

# Create price charts
def create_price_chart(df, title="Price Chart"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=600
    )
    
    return fig

# Create prediction chart
def create_prediction_chart(df, predictions, dates, model_name):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index[-30:],
        y=df['Close'].tail(30),
        name='Historical Price',
        line=dict(color='blue')
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        name=f'{model_name} Prediction',
        line=dict(color='green', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{model_name} Price Predictions',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=500
    )
    
    return fig

# Create technical indicators chart
def create_technical_chart(df):
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Price and Moving Averages', 'RSI'))
    
    # Price and MAs
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name='Close Price',
        line=dict(color='black')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='20-day MA',
        line=dict(color='orange')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'],
        name='50-day MA',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)
    
    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True
    )
    
    return fig

# Display metrics
def display_metrics(accuracy, metrics, model_name):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{model_name} Accuracy", f"{accuracy:.2f}%")
    with col2:
        st.metric("RMSE", f"${metrics['RMSE']:.2f}")
    with col3:
        st.metric("MAE", f"${metrics['MAE']:.2f}")

def main():
    setup_page()
    
    st.title("🚀 Cryptocurrency Price Prediction")
    st.markdown("""
    This app predicts cryptocurrency prices using machine learning models.
    Select your parameters in the sidebar to get started.
    """)
    
    # Get sidebar inputs
    crypto_symbol, timeframe, prediction_days, model_type = create_sidebar()
    
    try:
        # Load and process data
        with st.spinner('Loading data...'):
            df = load_crypto_data(crypto_symbol, timeframe)
            df = add_technical_indicators(df)
            
            # Display current price and 24h change
            current_price = df['Close'].iloc[-1]
            price_change = ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            
            st.sidebar.metric(
                "Current Price",
                f"${current_price:,.2f}",
                f"{price_change:+.2f}%"
            )
        
        # Display main price chart
        st.subheader("Historical Price Chart")
        price_chart = create_price_chart(df)
        st.plotly_chart(price_chart, use_container_width=True)
        
        # Model predictions
        if model_type in ["Linear Regression", "Both"]:
            with st.spinner('Training Linear Regression model...'):
                lr_predictions, lr_accuracy, lr_metrics = train_linear_model(df, prediction_days)
                
                if lr_predictions:
                    prediction_dates = pd.date_range(df.index[-1], periods=prediction_days+1)[1:]
                    lr_chart = create_prediction_chart(df, lr_predictions, prediction_dates, "Linear Regression")
                    st.plotly_chart(lr_chart, use_container_width=True)
                    display_metrics(lr_accuracy, lr_metrics, "Linear Regression")
                else:
                    st.error("Linear Regression model failed to generate predictions.")
        
        if model_type in ["Neural Network", "Both"]:
            with st.spinner('Training Neural Network model... This may take a few moments.'):
                nn_predictions, nn_accuracy, nn_metrics = train_lstm_model(df, prediction_days)
                
                if nn_predictions:
                    prediction_dates = pd.date_range(df.index[-1], periods=prediction_days+1)[1:]
                    nn_chart = create_prediction_chart(df, nn_predictions, prediction_dates, "Neural Network")
                    st.plotly_chart(nn_chart, use_container_width=True)
                    display_metrics(nn_accuracy, nn_metrics, "Neural Network")
                else:
                    st.error("Neural Network model failed to generate predictions.")
        
        # Technical Analysis
        st.subheader("Technical Analysis")
        technical_chart = create_technical_chart(df)
        st.plotly_chart(technical_chart, use_container_width=True)
        
        # Additional Information
        with st.expander("About the Models"):
            st.markdown("""
            ### Model Information
            - **Linear Regression**: Uses multiple technical indicators to make predictions
            - **Neural Network**: LSTM-based model that captures temporal dependencies
            
            ### Metrics Explanation
            - **Accuracy**: Percentage accuracy of predictions
            - **RMSE**: Root Mean Square Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try different parameters or check your internet connection.")

if __name__ == "__main__":
    main()
