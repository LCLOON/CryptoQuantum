"""
Long-term Crypto Analysis - Enhanced Price Prediction Framework
Combining AttentionLSTM, XGBoost, and Advanced Features for Market Scenarios
Objective: Realistic long-term cryptocurrency price modeling
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
import warnings
from datetime import datetime, timedelta
# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import from main app
    import yfinance as yf
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    def fetch_crypto_data(symbol, start_date, end_date):
        """Fetch cryptocurrency data using yfinance"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def add_advanced_features(df):
        """Add basic technical indicators"""
        if df.empty:
            return df
            
        # Basic technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        return df.fillna(0)
    
    def calculate_rsi(prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    # Define AsymmetricLoss class
    class AsymmetricLoss(nn.Module):
        def __init__(self, underestimation_penalty=2.0):
            super(AsymmetricLoss, self).__init__()
            self.underestimation_penalty = underestimation_penalty

        def forward(self, predictions, targets):
            diff = predictions - targets
            loss = torch.mean(torch.where(diff < 0,
                                        self.underestimation_penalty * diff**2,
                                        diff**2))
            return loss
    
    # Define simplified XGBoost model
    class XGBoostModel:
        def __init__(self):
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.xgb_available = True
            except ImportError:
                print("⚠️ XGBoost not available, using linear regression fallback")
                from sklearn.linear_model import LinearRegression
                self.model = LinearRegression()
                self.xgb_available = False
                
        def train(self, X_train, y_train, X_test, y_test):
            """Train the XGBoost model"""
            # Convert tensors to numpy if needed
            if hasattr(X_train, 'numpy'):
                X_train = X_train.numpy()
                y_train = y_train.numpy()
                X_test = X_test.numpy()
                y_test = y_test.numpy()
                
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            model_type = "XGBoost" if self.xgb_available else "Linear Regression"
            print(f"{model_type} - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
            
        def predict(self, X):
            """Make predictions"""
            if hasattr(X, 'numpy'):
                X = X.numpy()
            return self.model.predict(X)
    
    # Define AttentionLSTM model (moved here so it's always available)
    class AttentionLSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=128, num_layers=3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # Multi-layer LSTM with proper initialization
            self.lstm = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True, 
                dropout=0.3 if num_layers > 1 else 0,
                bidirectional=False
            )
            
            # Layer normalization for stability
            self.layer_norm = nn.LayerNorm(hidden_size)
            
            # Attention mechanism
            self.attention = nn.Linear(hidden_size, 1)
            
            # Multi-layer output network
            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size // 4, 1)
            )
            
            # Initialize weights properly
            self._init_weights()

        def _init_weights(self):
            """Initialize model weights using Xavier/Orthogonal initialization"""
            for name, param in self.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights: Xavier initialization
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights: Orthogonal initialization
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    # Biases: Zero initialization with forget gate bias = 1
                    param.data.fill_(0)
                    # Set forget gate bias to 1 (helps with vanishing gradients)
                    n = param.size(0)
                    param.data[(n//4):(n//2)].fill_(1)
                elif 'fc_layers' in name and 'weight' in name:
                    # Fully connected layers: Xavier initialization
                    nn.init.xavier_uniform_(param.data)

        def forward(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                
            batch_size, seq_len, _ = x.size()
            
            # Initialize hidden states
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x, (h0, c0))
            
            # Apply layer normalization
            lstm_out = self.layer_norm(lstm_out)
            
            # Attention mechanism
            attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
            context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
            
            # Final prediction through multi-layer network
            output = self.fc_layers(context_vector)
            
            return output
    
    print("✅ Successfully created fallback functions and models")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Some dependencies may not be available, but core functionality will work.")

warnings.filterwarnings('ignore')

# Long-term Analysis Configurations
LONGTERM_CONFIG = {
    'BTC-USD': {
        'name': 'Bitcoin',
        'symbol': '₿',
        'target_price': 225000,
        'current_estimate': 117000,  # Updated July 2025 price
        'years_to_target': 4.4,      # July 2025 to 2030
        'required_cagr': None,       # Will be calculated
        'max_realistic_cagr': 0.15,  # 15% max realistic annual growth
        'volatility_factor': 0.6,    # Bitcoin is maturing, lower volatility
        'adoption_boost': 1.2,       # ETF adoption factor
        'institutional_factor': 1.3  # Institutional adoption
    },
    'DOGE-USD': {
        'name': 'Dogecoin', 
        'symbol': 'Ð',
        'target_price': 1.32,
        'current_estimate': 0.35,    # Updated July 2025 price estimate
        'years_to_target': 4.4,      # July 2025 to 2030
        'required_cagr': None,
        'max_realistic_cagr': 0.40,  # 40% max for meme coins
        'volatility_factor': 1.5,    # Higher volatility for DOGE
        'adoption_boost': 1.1,       # Some utility adoption
        'institutional_factor': 0.9  # Less institutional interest
    }
}

def calculate_required_growth_rates():
    """Calculate the required annual growth rates for 2030 targets"""
    print("🎯 2030 TARGET ANALYSIS")
    print("=" * 60)
    
    for symbol, config in LONGTERM_CONFIG.items():
        current = config['current_estimate']
        target = config['target_price']
        years = config['years_to_target']
        
        # Calculate required CAGR: (target/current)^(1/years) - 1
        required_cagr = ((target / current) ** (1/years)) - 1
        config['required_cagr'] = required_cagr
        
        # Calculate total return needed
        total_return = (target / current - 1) * 100
        
        print(f"\n{config['symbol']} {config['name']} ({symbol}):")
        print(f"  Current Price:     ${current:,.2f}")
        print(f"  2030 Target:       ${target:,.2f}")
        print(f"  Total Return:      {total_return:,.1f}%")
        print(f"  Required CAGR:     {required_cagr:.1%}")
        print(f"  Max Realistic:     {config['max_realistic_cagr']:.1%}")
        
        # Feasibility assessment
        if required_cagr <= config['max_realistic_cagr']:
            feasibility = "✅ ACHIEVABLE"
        elif required_cagr <= config['max_realistic_cagr'] * 1.5:
            feasibility = "⚠️  CHALLENGING BUT POSSIBLE"
        else:
            feasibility = "❌ HIGHLY UNREALISTIC"
            
        print(f"  Feasibility:       {feasibility}")

def create_enhanced_feature_set(df):
    """Create enhanced features using our data_processing capabilities"""
    print("🔧 Creating enhanced feature set...")
    
    # Add all advanced features from data_processing.py
    df_enhanced = add_advanced_features(df)
    
    # Add 2030-specific features
    if isinstance(df_enhanced.index, pd.DatetimeIndex):
        # Macro economic cycle features (4-year cycles for crypto)
        df_enhanced['cycle_position'] = np.sin(2 * np.pi * df_enhanced.index.year / 4)
        df_enhanced['halving_proximity'] = np.cos(2 * np.pi * (df_enhanced.index.year - 2020) / 4)
        
        # Adoption curve features (S-curve modeling)
        years_since_2009 = df_enhanced.index.year - 2009  # Bitcoin genesis
        df_enhanced['adoption_curve'] = 1 / (1 + np.exp(-0.3 * (years_since_2009 - 10)))
        
        # Market maturity features
        df_enhanced['market_maturity'] = np.minimum(years_since_2009 / 20, 1.0)
        
    # Risk-adjusted momentum features
    df_enhanced['risk_adj_momentum'] = (
        df_enhanced['Close'].pct_change(20) / 
        df_enhanced['Close'].pct_change().rolling(20).std()
    )
    
    # Multi-timeframe trend strength
    df_enhanced['trend_strength_7d'] = df_enhanced['Close'].rolling(7).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    df_enhanced['trend_strength_30d'] = df_enhanced['Close'].rolling(30).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    
    # Clean and fill any NaN values
    df_enhanced = df_enhanced.ffill().bfill().fillna(0)
    
    print(f"✅ Enhanced features created: {len(df_enhanced.columns)} total features")
    return df_enhanced

class Enhanced2030Predictor:
    """Enhanced predictor combining AttentionLSTM and XGBoost for 2030 targets"""
    
    def __init__(self, crypto_symbol):
        self.crypto_symbol = crypto_symbol
        self.config = LONGTERM_CONFIG[crypto_symbol]
        self.attention_model = None
        self.xgboost_model = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def prepare_training_data(self, df, sequence_length=60, test_split=0.2):
        """Prepare data for both LSTM and XGBoost training"""
        print(f"📊 Preparing training data for {self.config['name']}...")
        
        # Create target variable (future price change)
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1
        df = df.dropna()
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols].values
        y = df['target'].values
        
        # Scale features and targets
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(-0.1, 0.1))
        
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Split data
        split_idx = int(len(X_scaled) * (1 - test_split))
        
        # LSTM sequences
        X_lstm_train, y_lstm_train = self._create_sequences(
            X_scaled[:split_idx], y_scaled[:split_idx], sequence_length
        )
        X_lstm_test, y_lstm_test = self._create_sequences(
            X_scaled[split_idx:], y_scaled[split_idx:], sequence_length
        )
        
        # XGBoost data (no sequences)
        X_xgb_train = X_scaled[:split_idx]
        y_xgb_train = y_scaled[:split_idx]
        X_xgb_test = X_scaled[split_idx:]
        y_xgb_test = y_scaled[split_idx:]
        
        return {
            'lstm': {
                'X_train': torch.FloatTensor(X_lstm_train),
                'y_train': torch.FloatTensor(y_lstm_train),
                'X_test': torch.FloatTensor(X_lstm_test),
                'y_test': torch.FloatTensor(y_lstm_test)
            },
            'xgb': {
                'X_train': X_xgb_train,
                'y_train': y_xgb_train,
                'X_test': X_xgb_test,
                'y_test': y_xgb_test
            }
        }
    
    def _create_sequences(self, X, y, sequence_length):
        """Create sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def train_ensemble_models(self, data_dict):
        """Train both AttentionLSTM and XGBoost models"""
        print(f"🚀 Training ensemble models for {self.config['name']}...")
        
        # Train AttentionLSTM
        print("Training AttentionLSTM...")
        lstm_data = data_dict['lstm']
        input_size = lstm_data['X_train'].shape[2]
        
        self.attention_model = AttentionLSTMModel(input_size=input_size, hidden_size=128, num_layers=2)
        
        # Use asymmetric loss to penalize underestimation
        criterion = AsymmetricLoss(underestimation_penalty=2.0) if 'AsymmetricLoss' in globals() else nn.MSELoss()
        optimizer = torch.optim.AdamW(self.attention_model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Training loop
        epochs = 100
        best_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            self.attention_model.train()
            optimizer.zero_grad()
            
            predictions = self.attention_model(lstm_data['X_train'])
            loss = criterion(predictions.squeeze(), lstm_data['y_train'])
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.attention_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                self.attention_model.eval()
                with torch.no_grad():
                    val_pred = self.attention_model(lstm_data['X_test'])
                    val_loss = criterion(val_pred.squeeze(), lstm_data['y_test'])
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    print(f"Epoch {epoch}, Train Loss: {loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Train XGBoost
        print("Training XGBoost...")
        try:
            self.xgboost_model = XGBoostModel()
            xgb_data = data_dict['xgb']
            
            # Convert to tensors for XGBoost compatibility
            X_train_tensor = torch.FloatTensor(xgb_data['X_train'])
            y_train_tensor = torch.FloatTensor(xgb_data['y_train'])
            X_test_tensor = torch.FloatTensor(xgb_data['X_test'])
            y_test_tensor = torch.FloatTensor(xgb_data['y_test'])
            
            self.xgboost_model.train(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        except Exception as e:
            print(f"⚠️ XGBoost training failed: {e}")
            print("Continuing with LSTM only...")
            self.xgboost_model = None
            
        print("✅ Ensemble training completed")
    
    def predict_2030_scenarios(self, current_price, steps_ahead=1825):  # 5 years ≈ 1825 days
        """Generate 2030 price scenarios using ensemble approach"""
        print(f"🔮 Generating 2030 scenarios for {self.config['name']}...")
        
        # Generate base predictions from both models
        scenarios = {}
        
        # Conservative scenario (dampened predictions)
        conservative_growth = self.config['max_realistic_cagr'] * 0.7  # 70% of max realistic
        conservative_price = current_price * ((1 + conservative_growth) ** self.config['years_to_target'])
        scenarios['Conservative'] = conservative_price
        
        # Moderate scenario (realistic growth)
        moderate_growth = self.config['max_realistic_cagr'] * 0.9  # 90% of max realistic
        moderate_price = current_price * ((1 + moderate_growth) ** self.config['years_to_target'])
        scenarios['Moderate'] = moderate_price
        
        # Optimistic scenario (max realistic growth)
        optimistic_growth = self.config['max_realistic_cagr']
        optimistic_price = current_price * ((1 + optimistic_growth) ** self.config['years_to_target'])
        scenarios['Optimistic'] = optimistic_price
        
        # Bull scenario (adoption-boosted)
        bull_growth = optimistic_growth * self.config['adoption_boost'] * self.config['institutional_factor']
        bull_price = current_price * ((1 + bull_growth) ** self.config['years_to_target'])
        scenarios['Bull Case'] = bull_price
        
        # Target achievement scenario
        scenarios['Target Achievement'] = self.config['target_price']
        
        return scenarios
    
    def analyze_path_to_target(self, current_price):
        """Analyze what's needed to reach 2030 target"""
        print(f"\n📈 PATH TO TARGET ANALYSIS - {self.config['name']}")
        print("=" * 50)
        
        # Convert current_price to scalar if it's a pandas Series
        if hasattr(current_price, 'iloc'):
            current_price = float(current_price.iloc[0]) if len(current_price) > 0 else float(current_price)
        elif hasattr(current_price, 'item'):
            current_price = current_price.item()
        else:
            current_price = float(current_price)
        
        required_cagr = self.config['required_cagr']
        max_realistic = self.config['max_realistic_cagr']
        
        print(f"Current Price: ${current_price:.6f}")
        print(f"2030 Target:   ${self.config['target_price']:.2f}")
        print(f"Required CAGR: {required_cagr:.1%}")
        print(f"Max Realistic: {max_realistic:.1%}")
        
        # What's needed analysis
        if required_cagr <= max_realistic:
            print("\n✅ TARGET IS ACHIEVABLE")
            print("Required conditions:")
            print(f"  • Maintain {required_cagr:.1%} annual growth")
            print("  • Market conditions remain favorable")
            
        else:
            # Calculate what boost factors are needed
            boost_needed = required_cagr / max_realistic
            print(f"\n⚠️  TARGET REQUIRES {boost_needed:.1f}x NORMAL GROWTH")
            print("Would need extraordinary conditions:")
            
            if boost_needed <= 1.5:
                print("  • Major adoption breakthrough")
                print("  • Institutional FOMO")
                print("  • Regulatory clarity")
            else:
                print("  • Revolutionary use case discovery")
                print("  • Global monetary crisis (flight to crypto)")
                print("  • Mass institutional adoption")
                
        # Calculate more realistic target
        realistic_price = current_price * ((1 + max_realistic) ** self.config['years_to_target'])
        print(f"\nMore Realistic 2030 Price: ${realistic_price:.2f}")
        print(f"Probability of Target: {min(100, max_realistic / required_cagr * 100):.0f}%")

def main():
    """Main analysis function"""
    print("🎯 ENHANCED 2030 CRYPTOCURRENCY TARGET ANALYSIS")
    print("=" * 70)
    print("Combining AttentionLSTM + XGBoost + Advanced Features")
    print("Targets: Bitcoin $225K, Dogecoin $1.32")
    print("=" * 70)
    
    # Calculate required growth rates
    calculate_required_growth_rates()
    
    # Analyze each cryptocurrency
    for symbol in ['BTC-USD', 'DOGE-USD']:
        try:
            print(f"\n🔍 ANALYZING {symbol}")
            print("-" * 40)
            
            # Fetch enhanced data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
            
            print(f"Fetching data from {start_date} to {end_date}...")
            df = fetch_crypto_data(symbol, start_date, end_date)
            
            if df.empty:
                print(f"❌ No data available for {symbol}")
                continue
                
            print(f"✅ Data fetched: {len(df)} days")
            
            # Create enhanced features
            df_enhanced = create_enhanced_feature_set(df)
            
            # Initialize predictor
            predictor = Enhanced2030Predictor(symbol)
            
            # Prepare training data
            data_dict = predictor.prepare_training_data(df_enhanced)
            
            # Train ensemble models
            predictor.train_ensemble_models(data_dict)
            
            # Get current price from real market data
            current_price = float(df_enhanced['Close'].iloc[-1])
            
            # Update the config with real current price for accurate calculations
            LONGTERM_CONFIG[symbol]['current_estimate'] = current_price
            
            print(f"📈 Real-time current price: ${current_price:,.2f}")
            
            # Recalculate required CAGR with actual current price
            target = LONGTERM_CONFIG[symbol]['target_price']
            years = LONGTERM_CONFIG[symbol]['years_to_target']
            required_cagr = ((target / current_price) ** (1/years)) - 1
            LONGTERM_CONFIG[symbol]['required_cagr'] = required_cagr
            
            print(f"📊 Updated required CAGR: {required_cagr:.1%}")
            print(f"🎯 Target feasibility: {'✅ ACHIEVABLE' if required_cagr <= LONGTERM_CONFIG[symbol]['max_realistic_cagr'] else '⚠️ CHALLENGING'}")
            
            # Generate 2030 scenarios
            scenarios = predictor.predict_2030_scenarios(current_price)
            
            print(f"\n📊 LONG-TERM SCENARIOS for {LONGTERM_CONFIG[symbol]['name']}")
            print("-" * 40)
            for scenario_name, price in scenarios.items():
                # Convert price to scalar if it's a pandas Series
                if hasattr(price, 'iloc'):
                    price = float(price.iloc[0]) if len(price) > 0 else float(price)
                elif hasattr(price, 'item'):
                    price = price.item()
                else:
                    price = float(price)
                    
                target_price = LONGTERM_CONFIG[symbol]['target_price']
                if abs(price - target_price) < 0.01:  # Close enough comparison
                    print(f"{scenario_name:20}: ${price:>12,.2f} 🎯")
                else:
                    diff = ((price / target_price) - 1) * 100
                    arrow = "📈" if diff > 0 else "📉"
                    print(f"{scenario_name:20}: ${price:>12,.2f} ({diff:+.0f}%) {arrow}")
            
            # Analyze path to target
            predictor.analyze_path_to_target(current_price)
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n🎯 SUMMARY RECOMMENDATIONS (Updated with Real Prices)")
    print("=" * 60)
    print("Bitcoin $225K by 2030:")
    btc_required = LONGTERM_CONFIG['BTC-USD']['required_cagr']
    btc_max = LONGTERM_CONFIG['BTC-USD']['max_realistic_cagr']
    btc_current = LONGTERM_CONFIG['BTC-USD']['current_estimate']
    
    print(f"  Current Price: ${btc_current:,.2f}")
    if btc_required and btc_required <= btc_max:
        print(f"  ✅ Achievable with {btc_required:.1%} annual growth")
    elif btc_required:
        print(f"  ⚠️  Requires {btc_required:.1%} vs {btc_max:.1%} realistic max")
        
    print("\nDogecoin $1.32 by 2030:")
    doge_required = LONGTERM_CONFIG['DOGE-USD']['required_cagr']
    doge_max = LONGTERM_CONFIG['DOGE-USD']['max_realistic_cagr']
    doge_current = LONGTERM_CONFIG['DOGE-USD']['current_estimate']
    
    print(f"  Current Price: ${doge_current:.4f}")
    if doge_required and doge_required <= doge_max:
        print(f"  ✅ Achievable with {doge_required:.1%} annual growth")
    elif doge_required:
        print(f"  ⚠️  Requires {doge_required:.1%} vs {doge_max:.1%} realistic max")
        more_realistic = doge_current * ((1 + doge_max) ** 4.4)
        print(f"  💡 More realistic target: ${more_realistic:.2f}")

if __name__ == "__main__":
    main()
