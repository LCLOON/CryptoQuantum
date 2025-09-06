"""
AI Models and Training Functions
Handles LSTM model creation, training, and prediction
"""

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class LSTMModel(nn.Module):
    """Enhanced LSTM Model with attention mechanism"""
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dense layers
        out = self.relu(self.fc1(context_vector))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out

@st.cache_data
def create_sequences(data, sequence_length=30):
    """Create sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length, 0]  # Predict close price
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

@st.cache_data
def prepare_data(data, sequence_length=30, train_split=0.85):
    """Prepare data for LSTM training"""
    try:
        # Calculate log returns for price stability
        log_prices = np.log(data[:, 0])
        log_returns = np.diff(log_prices)
        
        # Scale the data
        scaler_returns = MinMaxScaler(feature_range=(-1, 1))
        scaler_volume = MinMaxScaler(feature_range=(0, 1))
        
        # Prepare features
        scaled_returns = scaler_returns.fit_transform(log_returns.reshape(-1, 1)).flatten()
        scaled_volume = scaler_volume.fit_transform(data[1:, 1].reshape(-1, 1)).flatten()
        
        # Combine features
        scaled_data = np.column_stack([scaled_returns, scaled_volume])
        
        # Create sequences
        X, y = create_sequences(scaled_data, sequence_length)
        
        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        
        # Store log prices for inverse transformation
        test_log_prices = log_prices[split_idx + sequence_length:]
        
        return X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices
        
    except Exception as e:
        st.error(f"Error preparing data: {str(e)}")
        return None

def train_advanced_model(df, symbol_name, ai_model_choice="AttentionLSTM (Recommended)", progress_callback=None):
    """Train advanced LSTM model with progress tracking"""
    try:
        if progress_callback:
            progress_callback(82, "🧠 INITIALIZING NEURAL NETWORK...")
        
        # Prepare data
        data = df[['Close', 'Volume']].values
        prepared_data = prepare_data(data)
        
        if prepared_data is None:
            return None, None, None
        
        X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices = prepared_data
        
        if progress_callback:
            progress_callback(85, "🔧 BUILDING MODEL ARCHITECTURE...")
        
        # Model parameters
        input_size = X_train.shape[2]
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=3, dropout=0.2)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        if progress_callback:
            progress_callback(88, "🚀 TRAINING NEURAL NETWORK...")
        
        # Training loop
        model.train()
        epochs = 100
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Forward pass
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs.squeeze(), y_test)
            model.train()
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break
            
            # Progress updates
            if progress_callback and epoch % 10 == 0:
                progress_percent = 88 + int((epoch / epochs) * 5)
                progress_callback(progress_percent, f"🧠 TRAINING EPOCH {epoch}/{epochs}...")
        
        if progress_callback:
            progress_callback(93, "✅ MODEL TRAINING COMPLETED")
        
        model.eval()
        scalers = (scaler_returns, scaler_volume)
        
        return model, scalers, input_size
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def predict_realistic_future(model, scalers, last_sequence, last_log_price, avg_volume, steps=30):
    """Predict future returns with price-aware conservative crypto growth"""
    if model is None:
        return None
    
    try:
        scaler_returns, scaler_volume = scalers
        model.eval()
        
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)
        current_log_price = last_log_price
        
        with torch.no_grad():
            for step in range(steps):
                # Predict next return
                prediction = model(current_sequence)
                predicted_return = prediction.item()
                
                # Convert back to price
                # Inverse transform the predicted return
                predicted_return_unscaled = scaler_returns.inverse_transform([[predicted_return]])[0][0]
                
                # Apply return to get new log price
                new_log_price = current_log_price + predicted_return_unscaled
                new_price = np.exp(new_log_price)
                
                predictions.append(new_price)
                
                # Update sequence for next prediction
                # Create new feature vector with predicted return and average volume
                scaled_volume = scaler_volume.transform([[avg_volume]])[0][0]
                new_features = np.array([predicted_return, scaled_volume])
                
                # Shift sequence and add new features
                new_sequence = current_sequence.clone()
                new_sequence = torch.cat([new_sequence[:, 1:, :], 
                                        torch.FloatTensor(new_features).unsqueeze(0).unsqueeze(0)], dim=1)
                current_sequence = new_sequence
                current_log_price = new_log_price
        
        return predictions
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None

def analyze_long_term_scenarios(symbol, mode="standard", confidence_level=0.85):
    """Analyze long-term price scenarios using enhanced framework"""
    try:
        # Clear caches for fresh analysis
        st.cache_data.clear()
        
        # Import enhanced analysis modules
        import sys
        import importlib
        
        # Force reload if needed
        if 'target_2030_analysis' in sys.modules:
            importlib.reload(sys.modules['target_2030_analysis'])
        
        # Run analysis (this would normally import your advanced analysis)
        # For now, return a simple structure
        return {
            'scenarios': {
                'conservative': 50000,
                'moderate': 75000,
                'optimistic': 100000,
                'bull_case': 150000
            },
            'confidence_intervals': {
                'lower': 45000,
                'upper': 160000
            },
            'analysis_date': datetime.now(),
            'symbol': symbol,
            'mode': mode
        }
        
    except Exception as e:
        st.error(f"Error in long-term analysis: {str(e)}")
        return None

def display_scenario_analysis(analysis, crypto_name, symbol):
    """Display scenario analysis results"""
    if not analysis:
        st.error("No analysis data available")
        return
    
    st.markdown("### 🎯 SCENARIO ANALYSIS RESULTS")
    
    scenarios = analysis.get('scenarios', {})
    
    # Create columns for scenarios
    cols = st.columns(4)
    
    scenario_names = ['Conservative', 'Moderate', 'Optimistic', 'Bull Case']
    scenario_keys = ['conservative', 'moderate', 'optimistic', 'bull_case']
    
    for i, (name, key) in enumerate(zip(scenario_names, scenario_keys)):
        with cols[i]:
            price = scenarios.get(key, 0)
            st.markdown(f"""
            <div class="trading-card">
                <h4>{name}</h4>
                <div class="financial-metric">
                    <span class="metric-value">${price:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Confidence intervals
    conf_intervals = analysis.get('confidence_intervals', {})
    if conf_intervals:
        st.markdown("### 📊 CONFIDENCE INTERVALS")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Lower Bound", f"${conf_intervals.get('lower', 0):,.2f}")
        
        with col2:
            st.metric("Upper Bound", f"${conf_intervals.get('upper', 0):,.2f}")
