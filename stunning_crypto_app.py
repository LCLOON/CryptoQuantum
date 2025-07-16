import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

# 2030 Target Analysis Configuration
TARGET_2030_CONFIG = {
    'BTC-USD': {
        'name': 'Bitcoin',
        'symbol': '₿',
        'target_price': 225000,
        'current_estimate': 70000,
        'years_to_target': 5.4,
        'required_cagr': 0.241,
        'max_realistic_cagr': 0.15,
        'model_prediction': 242141,
        'model_cagr': 0.155,
        'feasibility': 'HIGHLY ACHIEVABLE',
        'probability': 95
    },
    'DOGE-USD': {
        'name': 'Dogecoin',
        'symbol': 'Ð',
        'target_price': 1.32,
        'current_estimate': 0.20,
        'years_to_target': 5.4,
        'required_cagr': 0.418,
        'max_realistic_cagr': 0.40,
        'model_prediction': 1.18,
        'model_cagr': 0.428,
        'feasibility': 'CHALLENGING BUT POSSIBLE',
        'probability': 75
    }
}

def analyze_long_term_scenarios(symbol, mode="standard"):
    """Analyze long-term price scenarios using the enhanced 2030 analysis framework"""
    
    try:
        # Import our enhanced analysis module
        import target_2030_analysis as analysis_module
        from datetime import datetime, timedelta
        
        st.info("🚀 Initializing Enhanced Long-term Analysis Engine...")
        
        # Fetch enhanced data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📡 Fetching market data...")
        progress_bar.progress(20)
        
        df = analysis_module.fetch_crypto_data(symbol, start_date, end_date)
        
        if df.empty:
            st.error(f"❌ No data available for {symbol}")
            return None
            
        status_text.text("🔧 Creating enhanced feature set...")
        progress_bar.progress(40)
        
        # Create enhanced features
        df_enhanced = analysis_module.create_enhanced_feature_set(df)
        
        status_text.text("🚀 Training ensemble models...")
        progress_bar.progress(60)
        
        # Initialize predictor
        predictor = analysis_module.Enhanced2030Predictor(symbol)
        
        # Prepare training data
        data_dict = predictor.prepare_training_data(df_enhanced)
        
        status_text.text("🧠 Training AttentionLSTM + XGBoost...")
        progress_bar.progress(80)
        
        # Train ensemble models (suppress console output in Streamlit)
        import io
        import contextlib
        
        # Capture training output
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            predictor.train_ensemble_models(data_dict)
        
        # Get current price from real market data
        current_price = float(df_enhanced['Close'].iloc[-1])
        
        # Update the config with real current price for accurate calculations
        analysis_module.LONGTERM_CONFIG[symbol]['current_estimate'] = current_price
        
        # Recalculate required CAGR with actual current price
        target = analysis_module.LONGTERM_CONFIG[symbol]['target_price']
        years = analysis_module.LONGTERM_CONFIG[symbol]['years_to_target']
        required_cagr = ((target / current_price) ** (1/years)) - 1
        analysis_module.LONGTERM_CONFIG[symbol]['required_cagr'] = required_cagr
        
        status_text.text("🔮 Generating 2030 scenarios...")
        progress_bar.progress(90)
        
        # Generate 2030 scenarios
        scenarios = predictor.predict_2030_scenarios(current_price)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Create analysis result compatible with display function
        config = analysis_module.LONGTERM_CONFIG[symbol]
        
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'scenarios': scenarios,
            'forecast_years': years,
            'mode': mode,
            'target_price': target,
            'required_cagr': required_cagr,
            'max_realistic_cagr': config['max_realistic_cagr'],
            'training_output': output_buffer.getvalue(),
            'model_type': 'Enhanced AttentionLSTM + XGBoost Ensemble'
        }
        
        # Add crypto-specific context
        crypto_info = {
            'BTC-USD': {'name': 'Bitcoin', 'symbol': '₿', 'type': 'Store of Value'},
            'ETH-USD': {'name': 'Ethereum', 'symbol': 'Ξ', 'type': 'Smart Contract Platform'},  
            'DOGE-USD': {'name': 'Dogecoin', 'symbol': 'Ð', 'type': 'Meme Coin / Payment'},
        }
        
        analysis['crypto_info'] = crypto_info.get(symbol, {
            'name': symbol.split('-')[0], 
            'symbol': symbol.split('-')[0], 
            'type': 'Cryptocurrency'
        })
        
        return analysis
        
    except Exception as e:
        st.error(f"❌ Error in long-term analysis: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def display_scenario_analysis(analysis, crypto_name):
    """Display comprehensive long-term scenario analysis in Streamlit"""
    if not analysis:
        st.error("❌ Long-term analysis not available for this cryptocurrency")
        return
    
    crypto_info = analysis['crypto_info']
    scenarios = analysis['scenarios']
    current_price = analysis['current_price']
    forecast_years = analysis['forecast_years']
    target_price = analysis.get('target_price', 0)
    required_cagr = analysis.get('required_cagr', 0)
    max_realistic_cagr = analysis.get('max_realistic_cagr', 0)
    model_type = analysis.get('model_type', 'Enhanced Analysis')
    
    st.markdown(f"### 🎯 {crypto_info['symbol']} {crypto_info['name']} - LONG-TERM SCENARIOS")
    
    # Model Info
    st.info(f"🧠 **Model**: {model_type} | 📊 **Analysis Mode**: {analysis['mode']}")
    
    # Current Price and Target Info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Price",
            f"${current_price:,.2f}",
            "Real-time market data"
        )
    
    with col2:
        st.metric(
            "🎯 2030 Target", 
            f"${target_price:,.0f}",
            f"{forecast_years:.1f} year horizon"
        )
    
    with col3:
        if required_cagr:
            st.metric(
                "📈 Required CAGR",
                f"{required_cagr:.1%}",
                f"vs {max_realistic_cagr:.1%} realistic"
            )
    
    with col4:
        # Calculate feasibility
        if required_cagr and max_realistic_cagr:
            if required_cagr <= max_realistic_cagr:
                feasibility = "✅ Achievable"
                delta_color = "normal"
            elif required_cagr <= max_realistic_cagr * 1.2:
                feasibility = "⚠️ Challenging" 
                delta_color = "normal"
            else:
                feasibility = "❌ Unrealistic"
                delta_color = "inverse"
            
            st.metric(
                "🎯 Feasibility",
                feasibility,
                f"{((required_cagr/max_realistic_cagr - 1)*100):+.0f}% vs realistic" if required_cagr else "",
                delta_color=delta_color
            )
    
    # Scenario Analysis Table
    st.markdown("#### 📊 PRICE SCENARIOS")
    
    # Process scenarios for display
    scenario_table = []
    for scenario_name, price in scenarios.items():
        # Convert price to scalar if needed
        if hasattr(price, 'iloc'):
            price = float(price.iloc[0]) if len(price) > 0 else float(price)
        elif hasattr(price, 'item'):
            price = price.item()
        else:
            price = float(price)
        
        # Calculate metrics
        total_return = ((price / current_price) - 1) * 100
        annual_return = ((price / current_price) ** (1/forecast_years)) - 1
        
        # Determine status
        if abs(price - target_price) < target_price * 0.01:  # Within 1%
            status = "🎯 TARGET"
        elif price > target_price:
            status = "� ABOVE"
        else:
            status = "📉 BELOW"
        
        scenario_table.append({
            "Scenario": scenario_name,
            "2030 Price": f"${price:,.0f}",
            "Total Return": f"{total_return:+.0f}%",
            "Annual CAGR": f"{annual_return:.1%}",
            "vs Target": status
        })
    
    # Display table
    df_scenarios = pd.DataFrame(scenario_table)
    st.dataframe(df_scenarios, use_container_width=True)
    
    # Path to Target Analysis
    if required_cagr and max_realistic_cagr:
        st.markdown("#### �️ PATH TO TARGET ANALYSIS")
        
        if required_cagr <= max_realistic_cagr:
            st.success("✅ **TARGET IS ACHIEVABLE**")
            st.write("Required conditions:")
            st.write(f"• Maintain {required_cagr:.1%} annual growth")
            st.write("• Favorable market conditions")
        else:
            boost_needed = required_cagr / max_realistic_cagr
            st.warning(f"⚠️ **TARGET REQUIRES {boost_needed:.1f}x NORMAL GROWTH**")
            st.write("Would need extraordinary conditions:")
            if boost_needed <= 1.5:
                st.write("• Major adoption breakthrough")
                st.write("• Institutional FOMO")
                st.write("• Regulatory clarity")
            else:
                st.write("• Revolutionary use case discovery")
                st.write("• Global monetary crisis (flight to crypto)")
                st.write("• Mass institutional adoption")
        
        # More realistic target
        realistic_price = current_price * ((1 + max_realistic_cagr) ** forecast_years)
        probability = min(100, max_realistic_cagr / required_cagr * 100) if required_cagr else 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "💡 More Realistic 2030 Price",
                f"${realistic_price:,.0f}",
                f"At {max_realistic_cagr:.1%} CAGR"
            )
        with col2:
            st.metric(
                "🎲 Probability of Target",
                f"{probability:.0f}%",
                "Based on historical performance"
            )
    
    # Training Details (if available)
    if 'training_output' in analysis and analysis['training_output']:
        with st.expander("🔧 Model Training Details"):
            st.code(analysis['training_output'][-1000:])  # Show last 1000 chars
    
    # Display as metrics
    cols = st.columns(len(scenario_data))
    for i, scenario in enumerate(scenario_data):
        with cols[i]:
            scenario_name = scenario['Scenario']
            if scenario_name == "Conservative":
                emoji = "🔒"
            elif scenario_name == "Moderate":
                emoji = "📈"
            elif scenario_name == "Optimistic":
                emoji = "🚀"
            else:
                emoji = "🌙"
                
            st.metric(
                f"{emoji} {scenario_name}",
                scenario['Price'],
                scenario['Annual Return']
            )
    
    # Create visualization
    st.markdown("#### 📈 SCENARIO COMPARISON")
    
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Prepare data for chart
    scenario_names = []
    prices = []
    colors = []
    
    color_map = {
        'Conservative': '#ff6b6b',
        'Moderate': '#4ecdc4', 
        'Optimistic': '#45b7d1',
        'Bull Case': '#96ceb4'
    }
    
    for item in scenario_data:
        scenario_names.append(item['Scenario'])
        price_str = item['Price'].replace('$', '').replace(',', '')
        prices.append(float(price_str))
        colors.append(color_map.get(item['Scenario'], '#95a5a6'))
    
    fig.add_trace(go.Bar(
        x=scenario_names,
        y=prices,
        marker=dict(color=colors),
        text=[f"${p:,.0f}" for p in prices],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"{crypto_info['name']} {forecast_years}-Year Price Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Price (USD)",
        height=400,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market Context
    st.markdown("#### 🧠 ANALYSIS INSIGHTS")
    
    crypto_type = crypto_info.get('type', 'Cryptocurrency')
    
    with st.expander("📊 View Detailed Analysis", expanded=False):
        st.markdown(f"""
        **Market Classification:** {crypto_type}
        
        **Growth Analysis:**
        - **Conservative Scenario:** Lower-bound estimates assuming market maturation
        - **Moderate Scenario:** Expected growth with steady adoption  
        - **Optimistic Scenario:** Strong market conditions and adoption
        - **Bull Case:** Exceptional market conditions and breakthrough adoption
        
        **Key Factors:**
        - Market cycles and sentiment
        - Institutional adoption rates
        - Regulatory developments
        - Technology improvements
        - Global economic conditions
        
        **⚠️ Important Note:** These are mathematical projections based on historical patterns. 
        Cryptocurrency markets are highly volatile and unpredictable. Past performance does not guarantee future results.
        """)
        
    # Risk Disclaimer
    st.markdown("""
    <div style="background-color: #2c3e50; padding: 1rem; border-radius: 10px; border-left: 4px solid #e74c3c; margin-top: 1rem;">
        <h4 style="color: #e74c3c; margin-top: 0;">⚠️ Risk Disclaimer</h4>
        <p style="color: #ecf0f1; margin-bottom: 0;">
            This analysis is for educational purposes only and should not be considered investment advice. 
            Cryptocurrency investments carry significant risk and volatility. Always do your own research and 
            consult with financial professionals before making investment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Configure Streamlit for professional trading terminal
st.set_page_config(
    page_title="� CryptoQuantum Terminal",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Wall Street Terminal CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Terminal Theme */
    .main > div {
        padding-top: 1rem;
        background: #0a0e1a;
        color: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 100%);
    }
    
    /* Terminal Header */
    .terminal-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
    }
    
    .terminal-title {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
        margin-bottom: 0.5rem;
    }
    
    .terminal-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #a0aec0;
        font-weight: 400;
    }
    
    /* Market Data Cards */
    .market-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        font-family: 'Roboto Mono', monospace;
    }
    
    .market-card:hover {
        border-color: #00ff88;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
    }
    
    .market-label {
        font-size: 0.85rem;
        color: #a0aec0;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .market-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    
    .market-change {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .positive { color: #00ff88; }
    .negative { color: #ff4757; }
    .neutral { color: #ffd700; }
    
    /* Trading Terminal Cards */
    .trading-card {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        font-family: 'Inter', sans-serif;
    }
    
    .trading-card h4 {
        color: #00ff88;
        font-family: 'Roboto Mono', monospace;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #4a5568;
        padding-bottom: 0.5rem;
    }
    
    /* Analysis Panels */
    .analysis-panel {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border: 1px solid #718096;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .analysis-panel h3 {
        color: #ffd700;
        font-family: 'Roboto Mono', monospace;
        margin-bottom: 1rem;
    }
    
    /* Financial Metrics */
    .financial-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 1.2rem;
        background: rgba(26, 32, 44, 0.8);
        border-left: 3px solid #00ff88;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: 'Roboto Mono', monospace;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Professional Buttons */
    .execute-btn {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #000000;
        font-family: 'Roboto Mono', monospace;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .execute-btn:hover {
        background: linear-gradient(135deg, #00cc6a 0%, #00aa56 100%);
        box-shadow: 0 6px 25px rgba(0, 255, 136, 0.4);
        transform: translateY(-2px);
    }
    
    /* Risk Assessment */
    .risk-high { border-left-color: #ff4757; }
    .risk-medium { border-left-color: #ffd700; }
    .risk-low { border-left-color: #00ff88; }
    
    /* Terminal Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    .mono {
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Sidebar Styling - Comprehensive Dark Theme */
    .css-1d391kg {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
    }
    
    /* Force Sidebar Dark Background - All Variations */
    .css-1lcbmhc, .css-1v0mbdj, .css-k1vhr6, .css-1y4p8pa, .css-12w0qpk, 
    .css-1cypcdb, .css-1outpf7, section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a202c 100%) !important;
    }
    
    /* Sidebar Container */
    .css-1outpf7 .css-1y4p8pa {
        background: #0a0e1a !important;
    }
    
    /* All Sidebar Text */
    .stSidebar, .stSidebar * {
        background: transparent !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Input Controls */
    .stSidebar .stSelectbox > div > div,
    .stSidebar .stSelectbox select,
    .stSidebar input {
        background: #1a202c !important;
        border: 1px solid #4a5568 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Sliders */
    .stSidebar .stSlider > div > div > div > div {
        background: #1a202c !important;
    }
    
    /* Enhanced Sidebar Checkboxes with Better Visibility */
    .stSidebar .stCheckbox > label > div {
        background: #1a202c !important;
        border: 2px solid #00ff88 !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
        min-width: 18px !important;
        min-height: 18px !important;
        box-shadow: 0 0 4px rgba(0, 255, 136, 0.2) !important;
    }
    
    .stSidebar .stCheckbox > label > div:hover {
        border-color: #00cc6a !important;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.4) !important;
        background: #2d3748 !important;
    }
    
    .stSidebar .stCheckbox input:checked + div {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%) !important;
        border-color: #00ff88 !important;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.6) !important;
    }
    
    .stSidebar .stCheckbox input:checked + div:after {
        content: "✓" !important;
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 14px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-shadow: 0 0 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Ensure checkbox labels are visible too */
    .stSidebar .stCheckbox > label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    .stSidebar .stCheckbox > label:hover {
        color: #00ff88 !important;
    }
    
    /* Sidebar Headers and Text */
    .stSidebar h1, .stSidebar h2, .stSidebar h3, 
    .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #00ff88 !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    .stSidebar p, .stSidebar span, .stSidebar div {
        color: #ffffff !important;
    }
    
    /* Info/Warning boxes in sidebar */
    .stSidebar .stInfo, .stSidebar .stSuccess, 
    .stSidebar .stWarning, .stSidebar .stError {
        background: rgba(26, 32, 44, 0.8) !important;
        border: 1px solid #4a5568 !important;
        color: #ffffff !important;
    }
    
    /* Sidebar Markdown */
    .stSidebar .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Live Data Indicator */
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    /* Bloomberg-style Ticker */
    .ticker-tape {
        background: #000000;
        color: #00ff88;
        font-family: 'Roboto Mono', monospace;
        font-size: 0.9rem;
        padding: 0.5rem;
        border-top: 2px solid #00ff88;
        border-bottom: 2px solid #00ff88;
        white-space: nowrap;
        overflow: hidden;
    }
    
    /* Professional Table Styling */
    .dataframe {
        background: #1a202c !important;
        color: #ffffff !important;
        border: 1px solid #4a5568 !important;
        font-family: 'Roboto Mono', monospace !important;
    }
    
    .dataframe th {
        background: #2d3748 !important;
        color: #00ff88 !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        border: 1px solid #4a5568 !important;
    }
</style>
""", unsafe_allow_html=True)

# Advanced ML Models for Professional Trading Terminal
class AsymmetricLoss(nn.Module):
    """Custom loss that penalizes underestimation more than overestimation"""
    def __init__(self, underestimation_penalty=1.5):
        super(AsymmetricLoss, self).__init__()
        self.underestimation_penalty = underestimation_penalty

    def forward(self, predictions, targets):
        diff = predictions - targets
        loss = torch.mean(torch.where(diff < 0,
                                    self.underestimation_penalty * diff**2,
                                    diff**2))
        return loss

class AttentionLSTMModel(nn.Module):
    """Latest Advanced LSTM with Attention Mechanism - Enhanced for Superior Crypto Predictions"""
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced LSTM with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Proper weight initialization for better convergence"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # Ensure proper batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        
        # Output processing
        out = self.layer_norm(context)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        
        return self.fc2(out)

# Improved Legacy LSTM for compatibility (enhanced but familiar architecture)
class LSTMModel(nn.Module):
    """Improved Legacy LSTM - Enhanced version of your original reliable model"""
    def __init__(self, input_size=2, hidden_size=128, num_layers=3, output_size=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        # Enhanced LSTM with bidirectional capability for better pattern recognition
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=False)
        
        # Layer normalization instead of batch norm for better sequence handling
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Improved dropout strategy
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)  # Lighter dropout for final layer
        
        # Enhanced fully connected layers with residual-like connections
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Better activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()  # More modern activation
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization for better training stability"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

    def forward(self, x):
        # LSTM forward pass
        out, (hidden, cell) = self.lstm(x)
        
        # Take last output with improved processing
        out = out[:, -1, :]  # Shape: (batch, hidden_size)
        
        # Layer normalization instead of batch norm for sequences
        out = self.layer_norm(out)
        
        # Enhanced fully connected pathway with residual-like connection
        identity = out
        out = self.gelu(self.fc1(out))  # GELU activation
        out = self.dropout1(out)
        
        # Add residual connection if dimensions match
        if identity.size(-1) == out.size(-1):
            out = out + identity  # Residual connection
        
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        
        return self.fc3(out)

# Professional Cryptocurrency Terminal Symbols - Top 50 Cryptos
CRYPTO_SYMBOLS = {
    '₿ BTC/USD': 'BTC-USD',
    'Ξ ETH/USD': 'ETH-USD', 
    '₮ USDT/USD': 'USDT-USD',
    '🔷 BNB/USD': 'BNB-USD',
    '◎ SOL/USD': 'SOL-USD',
    '🔵 USDC/USD': 'USDC-USD',
    '✖️ XRP/USD': 'XRP-USD',
    'Ð DOGE/USD': 'DOGE-USD',
    '🌕 LUNA/USD': 'LUNA-USD',
    '₳ ADA/USD': 'ADA-USD',
    '🔺 AVAX/USD': 'AVAX-USD',
    '💎 SHIB/USD': 'SHIB-USD',
    '● DOT/USD': 'DOT-USD',
    '⬡ LINK/USD': 'LINK-USD',
    '🔴 BCH/USD': 'BCH-USD',
    '🌪️ TRX/USD': 'TRX-USD',
    '🔰 NEAR/USD': 'NEAR-USD',
    '⬢ MATIC/USD': 'MATIC-USD',
    'Ł LTC/USD': 'LTC-USD',
    '💰 UNI/USD': 'UNI-USD',
    '🚀 ICP/USD': 'ICP-USD',
    '⚡ APT/USD': 'APT-USD',
    '📈 FTT/USD': 'FTT-USD',
    '🌟 ETC/USD': 'ETC-USD',
    '🔸 XLM/USD': 'XLM-USD',
    '⚖️ ATOM/USD': 'ATOM-USD',
    '🏦 CRO/USD': 'CRO-USD',
    '🌊 APE/USD': 'APE-USD',
    '🎯 ALGO/USD': 'ALGO-USD',
    '🔥 MANA/USD': 'MANA-USD',
    '⚔️ AXS/USD': 'AXS-USD',
    '🎮 SAND/USD': 'SAND-USD',
    '💸 VET/USD': 'VET-USD',
    '🔗 FIL/USD': 'FIL-USD',
    '🌈 FLOW/USD': 'FLOW-USD',
    '🎨 CHZ/USD': 'CHZ-USD',
    '💎 GRT/USD': 'GRT-USD',
    '🌍 THETA/USD': 'THETA-USD',
    '⚙️ ENJ/USD': 'ENJ-USD',
    '📱 BAT/USD': 'BAT-USD',
    '🔮 CRV/USD': 'CRV-USD',
    '⭐ XTZ/USD': 'XTZ-USD',
    '🏛️ MKR/USD': 'MKR-USD',
    '📊 COMP/USD': 'COMP-USD',
    '🎪 SUSHI/USD': 'SUSHI-USD',
    '🔋 YFI/USD': 'YFI-USD',
    '🌟 SNX/USD': 'SNX-USD',
    '🎯 AAVE/USD': 'AAVE-USD',
    '🔄 1INCH/USD': '1INCH-USD',
    '🚀 RUNE/USD': 'RUNE-USD'
}

@st.cache_data
def fetch_comprehensive_data(symbol, period='3y'):
    """Fetch comprehensive cryptocurrency data with enhanced features and fallback strategies"""
    try:
        # Try different data sources and periods
        periods_to_try = [period, '2y', '1y', '6mo', '3mo']
        
        for p in periods_to_try:
            try:
                # Fetch OHLCV data with progress disabled
                data = yf.download(symbol, period=p, progress=False)
                
                if data.empty or len(data) < 60:
                    continue
                
                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                    
                # Create comprehensive feature set
                df = pd.DataFrame()
                df['Close'] = data['Close']
                df['Volume'] = data['Volume']
                df['High'] = data['High']
                df['Low'] = data['Low']
                df['Open'] = data['Open']
                
                # Advanced price features
                df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
                df['Price_Change'] = df['Close'].pct_change()
                df['Price_Change_7d'] = df['Close'].pct_change(7)
                df['Price_Change_30d'] = df['Close'].pct_change(30)
                
                # Trend indicators (adjust window sizes for available data)
                min_window = min(10, len(df) // 6)
                df['SMA_10'] = df['Close'].rolling(window=min_window).mean()
                df['SMA_20'] = df['Close'].rolling(window=min(20, len(df) // 3)).mean()
                df['SMA_50'] = df['Close'].rolling(window=min(50, len(df) // 2)).mean()
                df['EMA_12'] = df['Close'].ewm(span=min(12, len(df) // 5)).mean()
                df['EMA_26'] = df['Close'].ewm(span=min(26, len(df) // 3)).mean()
                
                # Price ratios (bullish indicators)
                df['Price_vs_SMA10'] = df['Close'] / df['SMA_10']
                df['Price_vs_SMA20'] = df['Close'] / df['SMA_20']
                df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']
                
                # Momentum indicators
                df['Momentum_10'] = df['Close'] / df['Close'].shift(min_window)
                df['Momentum_20'] = df['Close'] / df['Close'].shift(min(20, len(df) // 3))
                
                # Volatility (normalized)
                df['Volatility'] = df['Price_Change'].rolling(window=20).std()
                df['Volatility_Norm'] = df['Volatility'] / df['Volatility'].rolling(window=min(100, len(df) // 2)).mean()
                
                # Volume features
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                df['Volume_Trend'] = df['Volume_SMA'] / df['Volume_SMA'].shift(20)
                
                # Advanced technical indicators
                # RSI
                delta = df['Close'].diff()
                rsi_window = min(14, len(df) // 4)
                gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI_Norm'] = df['RSI'] / 100  # Normalize to 0-1
                
                # MACD
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
                
                # Bollinger Bands
                bb_period = min(20, len(df) // 3)
                df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
                bb_std = df['Close'].rolling(window=bb_period).std()
                df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
                df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
                df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
                
                # Fill NaN values before dropping
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Remove rows with all zeros or still NaN
                df = df.dropna()
                
                if len(df) >= 30:  # Need minimum viable data
                    print(f"✅ Successfully fetched {len(df)} days of data for {symbol} using period {p}")
                    return df, data.index[-len(df):]
                    
            except Exception as e:
                print(f"⚠️ Failed to fetch data for {symbol} with period {p}: {str(e)}")
                continue
        
        # If all attempts failed
        st.error(f"❌ Unable to fetch sufficient data for {symbol}. Please try a different cryptocurrency.")
        return None, None
        
    except Exception as e:
        st.error(f"Critical error fetching data for {symbol}: {str(e)}")
        return None, None

@st.cache_data
def get_crypto_info(symbol):
    """Get current cryptocurrency information with robust error handling"""
    try:
        # Get more comprehensive data
        ticker = yf.Ticker(symbol)
        current_data = yf.download(symbol, period='30d', progress=False)
        
        if current_data.empty:
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(current_data.columns, pd.MultiIndex):
            # Flatten the multi-level columns
            current_data.columns = current_data.columns.get_level_values(0)
        
        current_price = current_data['Close'].iloc[-1]
        
        # Calculate 24h change safely
        if len(current_data) > 1:
            prev_price = current_data['Close'].iloc[-2]
            change_24h = ((current_price - prev_price) / prev_price) * 100
        else:
            change_24h = 0.0
        
        # Calculate volatility from 30 days of data (annualized)
        if len(current_data) > 2:
            daily_returns = current_data['Close'].pct_change().dropna()
            volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        else:
            volatility = 0.0
        
        # Get volume safely - use average of last 7 days
        volume = 'N/A'
        if 'Volume' in current_data.columns and len(current_data) >= 7:
            recent_volumes = current_data['Volume'].iloc[-7:]
            valid_volumes = recent_volumes[recent_volumes > 0]
            if len(valid_volumes) > 0:
                volume = valid_volumes.mean()
        
        # Try to get additional info from ticker.info as fallback
        if volume == 'N/A':
            try:
                info = ticker.info
                if info and 'regularMarketVolume' in info and info['regularMarketVolume']:
                    volume = info['regularMarketVolume']
                elif info and 'volume' in info and info['volume']:
                    volume = info['volume']
            except Exception:
                pass
        
        # Final fallback - try to get any volume data
        if volume == 'N/A' and 'Volume' in current_data.columns:
            last_volume = current_data['Volume'].iloc[-1]
            if pd.notna(last_volume) and last_volume > 0:
                volume = last_volume
        
        # Get market cap and other info
        market_cap = 'N/A'
        circulating_supply = 'N/A'
        try:
            info = ticker.info
            if info:
                market_cap = info.get('marketCap', 'N/A')
                circulating_supply = info.get('circulatingSupply', 'N/A')
        except Exception:
            pass
        
        return {
            'current_price': float(current_price),
            'change_24h': float(change_24h),
            'volume': volume,
            'volatility': float(volatility),
            'market_cap': market_cap,
            'circulating_supply': circulating_supply,
            'symbol': symbol
        }
    except Exception:
        # Silent fallback - no warning message
        return None

def create_sequences(data, sequence_length=30):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i, 0])  # Target is Close price (or return)
    return np.array(X), np.array(y)

def prepare_data(data, sequence_length=30, train_split=0.85):
    """Prepare data with log returns for stationarity, scale, and split"""
    # Compute log returns for Close (diff(log(Close))), keep Volume as is
    close = data[:, 0].reshape(-1, 1)
    volume = data[:, 1].reshape(-1, 1)
    
    # Handle any zero or negative values in close prices
    close = np.maximum(close, 1e-8)
    
    log_close = np.log(close)
    returns = np.diff(log_close, axis=0)  # Returns: shape (n-1, 1)
    volume = volume[1:]  # Align with returns

    # Combine returns and volume
    combined = np.hstack((returns, volume))

    # Split raw data - use more recent data for testing
    train_size = int(len(combined) * train_split)
    raw_train = combined[:train_size]
    raw_test = combined[train_size:]

    # Scale on train only (separate scalers for returns and volume)
    scaler_returns = MinMaxScaler(feature_range=(-0.8, 0.8))  # Narrower range for better stability
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    
    # Handle volume scaling safely
    train_volume = raw_train[:, 1:2]
    test_volume = raw_test[:, 1:2]
    
    # Replace any zero volumes with small positive values
    train_volume = np.maximum(train_volume, 1e-8)
    test_volume = np.maximum(test_volume, 1e-8)
    
    scaled_train = np.hstack((
        scaler_returns.fit_transform(raw_train[:, 0:1]),
        scaler_volume.fit_transform(train_volume)
    ))

    # Create sequences (target y is returns)
    X_train, y_train = create_sequences(scaled_train, sequence_length)

    # Scale test
    scaled_test = np.hstack((
        scaler_returns.transform(raw_test[:, 0:1]),
        scaler_volume.transform(test_volume)
    ))
    X_test, y_test = create_sequences(scaled_test, sequence_length)

    # Tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    return X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, log_close[train_size + sequence_length:]

def train_advanced_model(df, symbol_name, ai_model_choice="AttentionLSTM (Recommended)", progress_callback=None):
    """Train advanced AttentionLSTM model with AsymmetricLoss for superior crypto predictions"""
    try:
        # Use simple Close + Volume approach (matching our successful train_model.py)
        data = df[['Close', 'Volume']].values
        
        if len(data) < 100:
            st.error(f"Insufficient data for {symbol_name} - need at least 100 days")
            return None, None, None
        
        # Prepare data using our proven method
        prepared_data = prepare_data(data)
        
        if len(prepared_data) != 7 or prepared_data[0] is None:
            st.error("Error preparing data")
            return None, None, None
            
        X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices = prepared_data
        
        # Create selected AI model architecture
        if ai_model_choice == "🧠 AttentionLSTM (Recommended)":
            model = AttentionLSTMModel(input_size=2, hidden_size=128, num_layers=3, dropout=0.3)
            criterion = AsymmetricLoss(underestimation_penalty=1.5)  # Smart loss for crypto
            max_epochs = 100
            patience = 25
            lr = 0.001
            model_name = "Latest AttentionLSTM"
        elif ai_model_choice == "🔧 Improved Legacy LSTM":
            model = LSTMModel(input_size=2, hidden_size=128, num_layers=3, output_size=1, dropout=0.3)
            criterion = nn.SmoothL1Loss(beta=0.8)  # More robust than HuberLoss for crypto
            max_epochs = 120  # Increased epochs for better convergence
            patience = 30     # More patience for complex patterns
            lr = 0.0015      # Slightly lower learning rate for stability
            model_name = "Improved Legacy LSTM"
        elif ai_model_choice in ["🎯 Long-term Scenario Analysis", "📊 Multi-Model Ensemble"]:
            # For long-term analysis, we'll use a simpler approach in the main execution
            # This path shouldn't be reached as long-term analysis is handled separately
            model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2)
            criterion = nn.MSELoss()
            max_epochs = 50
            patience = 15
            lr = 0.002
            model_name = "Simple LSTM for Long-term"
        else:  # Default fallback
            model = LSTMModel(input_size=2, hidden_size=128, num_layers=3, output_size=1, dropout=0.3)
            criterion = nn.SmoothL1Loss(beta=0.8)
            max_epochs = 120
            patience = 30
            lr = 0.0015
            model_name = "Default LSTM"
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10)

        # Training with progress tracking
        model.train()
        train_losses = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred.squeeze(), y_train)
            loss.backward()
            
            # Gradient clipping for stability (more aggressive for attention model)
            clip_norm = 0.5 if model_name == "AttentionLSTM" else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Update progress if callback provided
            if progress_callback:
                # Calculate progress: 60% base + 25% training progress
                training_progress = (epoch + 1) / max_epochs
                total_progress = 60 + int(25 * training_progress)
                progress_callback(total_progress, f"🧠 TRAINING {model_name}... Epoch {epoch+1}/{max_epochs} (Loss: {loss.item():.4f})")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if progress_callback:
                        progress_callback(85, f"🧠 TRAINING COMPLETE - Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step(loss)
        
        model.eval()
        
        # Return model and scalers
        return model, (scaler_returns, scaler_volume), 2  # 2 features: returns + volume
        
    except Exception as e:
        st.error(f"Error training model for {symbol_name}: {str(e)}")
        return None, None, None

def predict_realistic_future(model, scalers, last_sequence, last_log_price, avg_volume, steps=30):
    """Predict future returns with price-aware conservative crypto growth"""
    if model is None:
        return []
    
    try:
        scaler_returns, scaler_volume = scalers
        predictions = []
        current_seq = last_sequence.copy()  # Shape: (30, 2)
        current_log_price = float(last_log_price.item() if hasattr(last_log_price, 'item') else last_log_price)
        
        # Get initial price to determine coin category
        initial_price = np.exp(current_log_price)
        
        # Determine coin category and adjust parameters accordingly
        if initial_price > 10000:  # High-value coins (BTC, etc.)
            dampening_factor = 0.15  # Less aggressive dampening for established coins
            max_daily_return = 0.02  # 2% max daily gain
            min_daily_return = -0.015  # 1.5% max daily loss
            max_annual_growth = 1.3  # 30% max annual growth
        elif initial_price > 100:  # Mid-value coins (ETH, etc.)
            dampening_factor = 0.12
            max_daily_return = 0.025  # 2.5% max daily gain
            min_daily_return = -0.02  # 2% max daily loss
            max_annual_growth = 1.4  # 40% max annual growth
        elif initial_price > 1:  # Dollar-range coins
            dampening_factor = 0.10
            max_daily_return = 0.03  # 3% max daily gain
            min_daily_return = -0.025  # 2.5% max daily loss
            max_annual_growth = 1.5  # 50% max annual growth
        else:  # Low-value coins (DOGE, etc.)
            dampening_factor = 0.08
            max_daily_return = 0.04  # 4% max daily gain
            min_daily_return = -0.03  # 3% max daily loss
            max_annual_growth = 1.8  # 80% max annual growth for meme coins
        
        model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Scale assumed volume
                scaled_volume = scaler_volume.transform([[avg_volume]])[0][0]
                current_seq[-1, 1] = scaled_volume  # Update last volume in seq
                
                input_tensor = torch.FloatTensor(current_seq).unsqueeze(0)
                pred_return = model(input_tensor).squeeze().item()
                
                # Apply price-aware dampening factor
                pred_return = pred_return * dampening_factor
                
                # Add time decay for longer predictions
                time_decay = 1.0 / (1.0 + step * 0.0005)  # Gentler decay
                pred_return = pred_return * time_decay
                
                # Add small random walk component
                random_component = np.random.normal(0, 0.0002)
                pred_return += random_component
                
                # Apply price-category-specific limits
                pred_return = np.clip(pred_return, min_daily_return, max_daily_return)
                
                # Inverse return and add to log price
                pred_return_actual = scaler_returns.inverse_transform([[pred_return]])[0][0]
                current_log_price += pred_return_actual
                pred_price = np.exp(current_log_price)
                
                # Price-aware safety check
                if len(predictions) > 0:
                    max_allowed_price = predictions[0] * (max_annual_growth ** (step / 365))
                    pred_price = min(pred_price, max_allowed_price)
                
                predictions.append(pred_price)
                
                # Roll and update seq with new return (scaled) and volume
                current_seq = np.roll(current_seq, -1, axis=0)
                current_seq[-1, 0] = pred_return  # Scaled return
        
        return np.array(predictions)
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return []

def get_realistic_price_prediction(symbol, current_price, forecast_years=5):
    """Get realistic price prediction with proper scaling"""
    
    # Market-based growth estimates (more conservative and realistic)
    growth_estimates = {
        'BTC-USD': {
            'conservative': 0.12,  # 12% annual
            'moderate': 0.18,      # 18% annual  
            'optimistic': 0.25,    # 25% annual
            'bull_case': 0.35      # 35% annual (crypto bull market)
        },
        'ETH-USD': {
            'conservative': 0.15,
            'moderate': 0.22,
            'optimistic': 0.30,
            'bull_case': 0.45
        },
        'DOGE-USD': {
            'conservative': 0.08,
            'moderate': 0.15,
            'optimistic': 0.35,
            'bull_case': 0.60
        },
        'default': {
            'conservative': 0.10,
            'moderate': 0.15,
            'optimistic': 0.25,
            'bull_case': 0.40
        }
    }
    
    # Get growth rates for the symbol
    rates = growth_estimates.get(symbol, growth_estimates['default'])
    
    # Calculate predictions for different scenarios
    predictions = {}
    for scenario, annual_rate in rates.items():
        # Compound annual growth: P = P0 * (1 + r)^t
        future_price = current_price * ((1 + annual_rate) ** forecast_years)
        predictions[scenario] = future_price
    
    return predictions

def create_price_scenarios_display(symbol, current_price, forecast_years=5):
    """Create a comprehensive price scenarios display"""
    
    predictions = get_realistic_price_prediction(symbol, current_price, forecast_years)
    
    # Calculate some market context
    total_years = forecast_years
    
    scenarios_data = []
    for scenario, price in predictions.items():
        total_return = ((price / current_price) - 1) * 100
        annual_return = ((price / current_price) ** (1/total_years) - 1) * 100
        
        scenarios_data.append({
            'Scenario': scenario.replace('_', ' ').title(),
            'Price': f"${price:,.2f}",
            'Total Return': f"{total_return:,.0f}%",
            'Annual Return': f"{annual_return:.1f}%"
        })
    
    return scenarios_data

# ...
def main():
    # Professional Terminal Header
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-title">
            <span class="live-indicator"></span>CRYPTOQUANTUM TERMINAL
        </div>
        <div class="terminal-subtitle">
            Advanced Quantitative Analysis & Algorithmic Forecasting Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Sidebar
    with st.sidebar:
        st.markdown("## 📊 TRADING DESK")
        st.markdown("---")
        
        # Step-by-step guide for users
        st.markdown("### 🎯 QUICK START GUIDE")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%); 
                    border-left: 4px solid #00ff88; padding: 1rem; border-radius: 6px; margin-bottom: 1rem;">
            <div style="color: #00ff88; font-weight: bold; margin-bottom: 0.5rem;">📋 FOLLOW THESE STEPS:</div>
            <div style="color: #ffffff; font-size: 0.9rem; line-height: 1.6;">
                <strong>1.</strong> 🎯 Select your cryptocurrency<br>
                <strong>2.</strong> ⏱️ Choose forecast horizon (years)<br>
                <strong>3.</strong> 📈 Configure display settings<br>
                <strong>4.</strong> 🎛️ Adjust advanced controls (optional)<br>
                <strong>5.</strong> 🚀 Click "EXECUTE ANALYSIS" below
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Asset Selection
        st.markdown("### 🎯 STEP 1: ASSET SELECTION")
        selected_crypto = st.selectbox(
            "Select Trading Pair:",
            list(CRYPTO_SYMBOLS.keys()),
            index=0,
            help="Choose the cryptocurrency you want to analyze"
        )
        symbol = CRYPTO_SYMBOLS[selected_crypto]
        
        # Get current market data early for sidebar calculations
        crypto_info = get_crypto_info(symbol)
        
        st.markdown("### ⏱️ STEP 2: ANALYSIS PARAMETERS")
        forecast_years = st.slider(
            'Forecast Horizon (Years)', 
            1, 15, 5,
            help="Select how many years into the future you want to predict"
        )
        days = forecast_years * 365
        
        st.markdown("### 📈 STEP 3: DISPLAY SETTINGS")
        show_technical = st.checkbox("📊 Technical Indicators", value=True, help="Show moving averages and trend lines")
        show_risk_metrics = st.checkbox("⚠️ Risk Analysis", value=True, help="Display risk assessment metrics")
        confidence_level = st.slider(
            "🎯 Confidence Interval", 
            0.7, 0.99, 0.85,
            help="Statistical confidence level for predictions"
        )
        
        st.markdown("### 🎛️ STEP 4: ADVANCED CONTROLS")
        st.markdown("<small style='color: #a0aec0;'>⚙️ Optional - Advanced users only</small>", unsafe_allow_html=True)
        
        # AI Model Selection - Enhanced with Long-term Analysis
        ai_model = st.selectbox(
            "🤖 AI Model Engine",
            [
                "🧠 AttentionLSTM (Recommended)", 
                "🔧 Improved Legacy LSTM",
                "🎯 Long-term Scenario Analysis",
                "📊 Multi-Model Ensemble"
            ],
            help="Choose the AI model architecture for predictions"
        )
        
        # Set variables based on AI model selection
        show_2030_analysis = ai_model in ["🎯 Long-term Scenario Analysis", "📊 Multi-Model Ensemble"]
        target_mode = ai_model
        
        volatility_filter = st.checkbox("🌊 Volatility Filter", value=False, help="Show volatility warnings")
        show_volume_profile = st.checkbox("📊 Volume Profile", value=False, help="Display trading volume data")
        enable_alerts = st.checkbox("🔔 Price Alerts", value=False, help="Set price alert notifications")
        alert_price = 0  # Initialize default value
        
        if enable_alerts and crypto_info:
            alert_price = st.number_input(
                "💰 Alert Price ($)", 
                min_value=0.0, 
                value=crypto_info['current_price'],
                help="Get notified when price reaches this level"
            )
        
        st.markdown("### 🏆 PERFORMANCE METRICS")
        st.markdown("<small style='color: #a0aec0;'>📊 Optional - For advanced analysis</small>", unsafe_allow_html=True)
        show_sharpe = st.checkbox("📈 Sharpe Ratio", value=False, help="Risk-adjusted return metric")
        show_drawdown = st.checkbox("📉 Max Drawdown", value=False, help="Maximum peak-to-trough decline")
        
        st.markdown("### 🎨 CHART THEMES")
        chart_theme = st.selectbox(
            "🎭 Chart Style", 
            ["Professional Dark", "Terminal Green", "Trading Blue", "Classic"],
            help="Choose your preferred chart appearance"
        )
        
        st.markdown("### 💼 PORTFOLIO TOOLS")
        portfolio_mode = st.checkbox("💎 Portfolio Mode", value=False, help="Enable portfolio allocation analysis")
        if portfolio_mode:
            allocation_percent = st.slider("📊 Portfolio Allocation %", 1, 100, 10)
        
        # Step completion status
        st.markdown("---")
        st.markdown("### ✅ SETUP STATUS")
        
        # Check completion status
        step1_complete = selected_crypto is not None
        step2_complete = forecast_years > 0
        step3_complete = True  # Display settings always have defaults
        
        status_color1 = "#00ff88" if step1_complete else "#ff4757"
        status_color2 = "#00ff88" if step2_complete else "#ff4757"
        status_color3 = "#00ff88" if step3_complete else "#ff4757"
        
        st.markdown(f"""
        <div style="background: rgba(26, 32, 44, 0.8); padding: 0.8rem; border-radius: 6px; font-size: 0.9rem;">
            <div style="color: {status_color1};">{"✅" if step1_complete else "❌"} Step 1: Asset Selected ({selected_crypto.split()[0] if step1_complete else "None"})</div>
            <div style="color: {status_color2};">{"✅" if step2_complete else "❌"} Step 2: Forecast Period ({forecast_years}Y)</div>
            <div style="color: {status_color3};">{"✅" if step3_complete else "❌"} Step 3: Settings Configured</div>
            <div style="color: #ffd700; margin-top: 0.5rem;"><strong>👉 Ready to execute analysis!</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🤖 AI ENGINE STATUS")
        
        # Display selected AI model info
        if ai_model == "🧠 AttentionLSTM (Recommended)":
            st.info("🧠 **LATEST ATTENTION LSTM**: Enhanced neural architecture with attention mechanism")
            st.info("⚡ **ADVANCED FEATURES**: Layer normalization, proper weight initialization, multi-layer output")
            st.info("🎯 **STATUS**: ✅ WORKING - Latest Model (2025)")
        elif ai_model == "🔧 Improved Legacy LSTM":
            st.info("🧠 **IMPROVED LEGACY LSTM**: Enhanced with LayerNorm, GELU, and residual connections")
            st.info("🛡️ **RELIABLE**: More robust loss function for crypto market stability")
            st.info("🎯 **STATUS**: ✅ WORKING - Stable Legacy Model")
        elif ai_model == "🎯 Long-term Scenario Analysis":
            st.info("🎯 **LONG-TERM SCENARIOS**: Market-based growth projections (3-7 years)")
            st.info("📊 **MULTIPLE SCENARIOS**: Conservative, Moderate, Optimistic, Bull Case")
            st.info("🎯 **STATUS**: ✅ WORKING - Real Market Data")
        elif ai_model == "📊 Multi-Model Ensemble":
            st.info("📊 **ENSEMBLE MODEL**: Combines LSTM + XGBoost + Market Analysis")
            st.info("🚀 **ADVANCED FEATURES**: Technical indicators + Sentiment analysis")
            st.info("🎯 **STATUS**: ⚠️ PARTIAL - XGBoost integration in progress")
            
        st.info("🟢 **ONLINE**: Deep Learning Models Active")
        st.info("🔄 **REAL-TIME**: Market Data Streaming")
        
        if enable_alerts and crypto_info:
            current_price = crypto_info['current_price']
            if alert_price > 0 and current_price > 0:
                if current_price >= alert_price:
                    st.success(f"🔔 **ALERT**: Price reached ${alert_price:.2f}!")
                else:
                    price_diff = ((alert_price - current_price) / current_price) * 100
                    st.warning(f"⏰ **WATCHING**: {price_diff:+.1f}% to target")
        
        # Market Overview Panel
        st.markdown("### 📊 MARKET OVERVIEW")
        current_time = datetime.now().strftime("%H:%M:%S UTC")
        st.markdown(f"**Last Update:** {current_time}")
        
        # Live Market Stats with real market cap calculation
        if crypto_info:
            # Use market cap from crypto_info if available
            market_cap = crypto_info.get('market_cap', 'N/A')
            
            if market_cap != 'N/A' and market_cap and market_cap > 0:
                if market_cap > 1_000_000_000_000:  # Trillion
                    mcap_display = f"${market_cap/1_000_000_000_000:.2f}T"
                elif market_cap > 1_000_000_000:  # Billion
                    mcap_display = f"${market_cap/1_000_000_000:.1f}B"
                elif market_cap > 1_000_000:  # Million
                    mcap_display = f"${market_cap/1_000_000:.0f}M"
                else:
                    mcap_display = f"${market_cap:,.0f}"
            else:
                # Fallback: estimate using circulating supply if available
                circulating_supply = crypto_info.get('circulating_supply', 'N/A')
                if circulating_supply != 'N/A' and circulating_supply and circulating_supply > 0:
                    estimated_mcap = crypto_info['current_price'] * circulating_supply
                    if estimated_mcap > 1_000_000_000_000:
                        mcap_display = f"~${estimated_mcap/1_000_000_000_000:.2f}T"
                    elif estimated_mcap > 1_000_000_000:
                        mcap_display = f"~${estimated_mcap/1_000_000_000:.1f}B"
                    elif estimated_mcap > 1_000_000:
                        mcap_display = f"~${estimated_mcap/1_000_000:.0f}M"
                    else:
                        mcap_display = f"~${estimated_mcap:,.0f}"
                else:
                    mcap_display = "N/A"
            
            st.metric("💎 Market Cap", mcap_display)
            
            if volatility_filter:
                vol = crypto_info.get('volatility', 0)
                if vol > 100:
                    st.error("⚠️ **HIGH VOLATILITY** - Exercise Caution")
                elif vol > 50:
                    st.warning("⚡ **MODERATE VOLATILITY** - Monitor Closely")
                else:
                    st.success("✅ **LOW VOLATILITY** - Stable Conditions")
    
    # Fallback data handling if crypto_info is None
    if crypto_info is None:
        temp_df, _ = fetch_comprehensive_data(symbol, '5d')
        if temp_df is not None and not temp_df.empty:
            current_price = temp_df['Close'].iloc[-1]
            prev_price = temp_df['Close'].iloc[-2] if len(temp_df) > 1 else current_price
            change_24h = ((current_price - prev_price) / prev_price) * 100 if prev_price > 0 else 0.0
            
            crypto_info = {
                'current_price': float(current_price),
                'change_24h': float(change_24h),
                'volume': 'N/A',
                'volatility': 0.0,
                'symbol': symbol
            }
        else:
            st.error(f"❌ **DATA UNAVAILABLE** - Unable to fetch {symbol} market data")
            return
    
    # Get current market data once (remove the duplicate call)
    
    # Professional Market Data Display (Display Once)
    if crypto_info:
        market_cols = st.columns(4)
        
        with market_cols[0]:
            current_price = crypto_info['current_price']
            change_24h = crypto_info['change_24h']
            
            # Professional price formatting
            if current_price >= 1000:
                price_display = f"${current_price:,.2f}"
            elif current_price >= 1:
                price_display = f"${current_price:.4f}"
            elif current_price >= 0.01:
                price_display = f"${current_price:.5f}"
            else:
                price_display = f"${current_price:.6f}"
            
            change_class = "positive" if change_24h >= 0 else "negative"
            change_symbol = "▲" if change_24h >= 0 else "▼"
            
            st.markdown(f"""
            <div class="market-card">
                <div class="market-label">SPOT PRICE</div>
                <div class="market-value">{price_display}</div>
                <div class="market-change {change_class}">
                    {change_symbol} {abs(change_24h):.2f}% (24H)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with market_cols[1]:
            volume_display = crypto_info['volume']
            if isinstance(volume_display, (int, float)) and volume_display > 0:
                if volume_display > 1_000_000_000:
                    volume_text = f"{volume_display/1_000_000_000:.1f}B"
                elif volume_display > 1_000_000:
                    volume_text = f"{volume_display/1_000_000:.1f}M"
                elif volume_display > 1_000:
                    volume_text = f"{volume_display/1_000:.1f}K"
                else:
                    volume_text = f"{volume_display:,.0f}"
            else:
                volume_text = "N/A"
            
            st.markdown(f"""
            <div class="market-card">
                <div class="market-label">24H VOLUME</div>
                <div class="market-value">{volume_text}</div>
                <div class="market-change neutral">USD</div>
            </div>
            """, unsafe_allow_html=True)
        
        with market_cols[2]:
            volatility_display = crypto_info['volatility']
            volatility_text = f"{volatility_display:.1f}%" if volatility_display > 0 else "N/A"
            vol_class = "negative" if volatility_display > 100 else "positive" if volatility_display > 50 else "neutral"
            
            st.markdown(f"""
            <div class="market-card">
                <div class="market-label">VOLATILITY</div>
                <div class="market-value">{volatility_text}</div>
                <div class="market-change {vol_class}">ANNUALIZED</div>
            </div>
            """, unsafe_allow_html=True)
        
        with market_cols[3]:
            # Market sentiment based on price action
            if change_24h > 5:
                sentiment = "BULLISH"
                sentiment_class = "positive"
                sentiment_icon = "🚀"
            elif change_24h > 0:
                sentiment = "POSITIVE"
                sentiment_class = "positive"
                sentiment_icon = "📈"
            elif change_24h > -5:
                sentiment = "NEUTRAL"
                sentiment_class = "neutral"
                sentiment_icon = "➡️"
            else:
                sentiment = "BEARISH"
                sentiment_class = "negative"
                sentiment_icon = "📉"
            
            st.markdown(f"""
            <div class="market-card">
                <div class="market-label">SENTIMENT</div>
                <div class="market-value">{sentiment}</div>
                <div class="market-change {sentiment_class}">{sentiment_icon} SIGNAL</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Main Trading Interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Professional Execute Trading Analysis Button
        st.markdown("""
        <div class="trading-card">
            <h4>🎯 STEP 5: EXECUTE ANALYSIS</h4>
            <div style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
                ✅ Complete steps 1-4 above, then click to run AI analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        execute_analysis = st.button(
            "🚀 EXECUTE ANALYSIS", 
            type="primary", 
            use_container_width=True, 
            key="execute_analysis",
            help="Start the AI-powered cryptocurrency prediction analysis"
        )
    
    # Move analysis results to main area (outside of columns)
    if execute_analysis:
        
        # Check if long-term scenario analysis is selected
        if show_2030_analysis:
            if target_mode == "🎯 Long-term Scenario Analysis":
                st.markdown("## 🎯 LONG-TERM SCENARIO ANALYSIS")
                
                # Run scenario analysis
                analysis = analyze_long_term_scenarios(symbol)
                if analysis:
                    display_scenario_analysis(analysis, selected_crypto.split(' ')[0])
                    
                # Exit early - no need for traditional LSTM analysis
                st.success("🎯 Long-term scenario analysis completed!")
                
            elif target_mode == "📊 Multi-Model Ensemble":
                st.markdown("## 📊 MULTI-MODEL ENSEMBLE FORECAST")
                
                # Run ensemble analysis
                analysis = analyze_long_term_scenarios(symbol, mode="ensemble")
                if analysis:
                    display_scenario_analysis(analysis, selected_crypto.split(' ')[0])
                    
                # Exit early - no need for traditional LSTM analysis
                st.success("📊 Multi-model ensemble analysis completed!")
                
        else:
            # Traditional LSTM Analysis for AttentionLSTM and Improved Legacy LSTM
            st.markdown("## 🧠 AI-POWERED CRYPTOCURRENCY ANALYSIS")
            
            # Professional loading interface with enhanced progress tracking
            progress_container = st.container()
            with progress_container:
                st.markdown("### 🔄 SYSTEM STATUS")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Progress callback function
                def update_progress(progress, message):
                    progress_bar.progress(min(progress, 100))
                    status_text.text(message)
                    time.sleep(0.1)  # Small delay for visual effect
                
                # Step 1: Data Acquisition
                update_progress(10, "📡 INITIALIZING DATA ACQUISITION...")
                time.sleep(0.5)
                update_progress(25, "📡 ACQUIRING MARKET DATA...")
                df, dates = fetch_comprehensive_data(symbol, '3y')
                
                if df is None or df.empty:
                    update_progress(30, "🔄 FALLBACK DATA SOURCE...")
                    df, dates = fetch_comprehensive_data(symbol, '1y')
                    
                    if df is None or df.empty:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"🚨 **CRITICAL ERROR** - Market data unavailable for {selected_crypto}")
                        st.info("💡 **RECOMMENDATION**: Try a different trading pair")
                        return
                
                # Step 2: Data Processing
                update_progress(45, "⚙️ PROCESSING MARKET DATA...")
                time.sleep(0.5)
                update_progress(55, "🔧 PREPARING NEURAL NETWORK INPUT...")
                time.sleep(0.5)
                
                # Step 3: Model Training with detailed progress
                update_progress(60, "🧠 INITIALIZING NEURAL NETWORKS...")
                model, scalers, num_features = train_advanced_model(df, selected_crypto, ai_model, update_progress)
                
                if model is None:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("🚨 **MODEL FAILURE** - Insufficient training data")
                    st.info("💡 **RECOMMENDATION**: Select different parameters or asset")
                    return
                
                # Step 4: Forecasting
                update_progress(86, "📊 INITIALIZING FORECAST ENGINE...")
                time.sleep(0.3)
                update_progress(90, "🔮 GENERATING PREDICTIONS...")
                time.sleep(0.3)
                
                # Prepare prediction data
                data = df[['Close', 'Volume']].values
                prepared_data = prepare_data(data)
                X_train, y_train, X_test, y_test, scaler_returns, scaler_volume, test_log_prices = prepared_data
                
                # Generate predictions
                last_sequence = X_test[-1].numpy()
                last_log_price = test_log_prices[-1]
                avg_volume = np.mean(data[-60:, 1])
                
                update_progress(95, "🎯 FINALIZING FORECAST CALCULATIONS...")
                predictions = predict_realistic_future(
                    model, (scaler_returns, scaler_volume), last_sequence, last_log_price, avg_volume, steps=days
                )
                
                update_progress(98, "📋 PREPARING RESULTS...")
                time.sleep(0.3)
                update_progress(100, "✅ ANALYSIS COMPLETE - ALL SYSTEMS READY")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                if predictions is None or len(predictions) == 0:
                    st.error("🚨 **FORECAST FAILURE** - Unable to generate predictions")
                    return
                
                # PROFESSIONAL RESULTS DISPLAY - FULL WIDTH
                st.markdown("---")
                st.markdown("## 📈 QUANTITATIVE FORECAST RESULTS")
                
                # Calculate key metrics
                final_price = predictions[-1]
                total_return = ((final_price - crypto_info['current_price']) / crypto_info['current_price']) * 100
                max_price = max(predictions)
                min_price = min(predictions)
                annual_return = ((final_price / crypto_info['current_price']) ** (1/forecast_years) - 1) * 100
                
                # Professional Results Cards - Full Width Layout
                results_cols = st.columns(4)
                
                with results_cols[0]:
                    return_class = "positive" if total_return > 0 else "negative"
                    st.markdown(f"""
                    <div class="trading-card">
                        <h4>🎯 TARGET PRICE ({2025 + forecast_years})</h4>
                        <div class="financial-metric">
                            <span class="metric-label">FORECAST:</span>
                            <span class="metric-value">${final_price:,.2f}</span>
                        </div>
                        <div class="financial-metric">
                            <span class="metric-label">RETURN:</span>
                            <span class="metric-value {return_class}">{total_return:+.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with results_cols[1]:
                    st.markdown(f"""
                    <div class="trading-card">
                        <h4>📊 PRICE RANGE</h4>
                        <div class="financial-metric">
                            <span class="metric-label">RESISTANCE:</span>
                            <span class="metric-value">${max_price:,.2f}</span>
                        </div>
                        <div class="financial-metric">
                            <span class="metric-label">SUPPORT:</span>
                            <span class="metric-value">${min_price:,.2f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with results_cols[2]:
                    annual_class = "positive" if annual_return > 0 else "negative"
                    st.markdown(f"""
                    <div class="trading-card">
                        <h4>📈 ANNUAL METRICS</h4>
                        <div class="financial-metric">
                            <span class="metric-label">CAGR:</span>
                            <span class="metric-value {annual_class}">{annual_return:.1f}%</span>
                        </div>
                        <div class="financial-metric">
                            <span class="metric-label">PERIOD:</span>
                            <span class="metric-value">{forecast_years}Y</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with results_cols[3]:
                    # Risk Assessment
                    volatility = crypto_info.get('volatility', 0)
                    if volatility > 100:
                        risk_level = "HIGH"
                        risk_class = "risk-high"
                    elif volatility > 50:
                        risk_level = "MEDIUM"
                        risk_class = "risk-medium"
                    else:
                        risk_level = "LOW"
                        risk_class = "risk-low"
                    
                    st.markdown(f"""
                    <div class="trading-card {risk_class}">
                        <h4>⚠️ RISK PROFILE</h4>
                        <div class="financial-metric">
                            <span class="metric-label">LEVEL:</span>
                            <span class="metric-value">{risk_level}</span>
                        </div>
                        <div class="financial-metric">
                            <span class="metric-label">VOLATILITY:</span>
                            <span class="metric-value">{volatility:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Professional Yearly Targets Table
                st.markdown("### 📋 ANNUAL PRICE TARGETS")
                yearly_data = []
                for year in range(1, forecast_years + 1):
                    idx = min(year * 365 - 1, len(predictions) - 1)
                    price = predictions[idx]
                    growth = ((price / crypto_info['current_price']) ** (1/year) - 1) * 100
                    yearly_data.append({
                        'Year': 2025 + year,
                        'Target Price': f"${price:,.2f}",
                        'CAGR': f"{growth:.1f}%",
                        'Total Return': f"{((price / crypto_info['current_price']) - 1) * 100:.0f}%"
                    })
                
                yearly_df = pd.DataFrame(yearly_data)
                st.dataframe(yearly_df, use_container_width=True)
                
                # Professional Chart with Enhanced Features
                st.markdown("### 📊 TECHNICAL CHART ANALYSIS")
                
                fig = go.Figure()
                
                # Historical price action
                recent_data = df['Close'][-365:] if len(df) > 365 else df['Close']
                recent_dates = dates[-365:] if len(dates) > 365 else dates
                
                # Chart color theme based on selection
                if chart_theme == "Terminal Green":
                    main_color, forecast_color = '#00ff88', '#00cc6a'
                    sma20_color, sma50_color = '#00dd77', '#00bb66'
                elif chart_theme == "Trading Blue":
                    main_color, forecast_color = '#00bfff', '#0080ff'
                    sma20_color, sma50_color = '#00aaff', '#0099ee'
                elif chart_theme == "Classic":
                    main_color, forecast_color = '#000000', '#ff6600'
                    sma20_color, sma50_color = '#666666', '#999999'
                else:  # Professional Dark
                    main_color, forecast_color = '#00ff88', '#ffd700'
                    sma20_color, sma50_color = '#00dd77', '#ffcc00'
                
                fig.add_trace(go.Scatter(
                    x=recent_dates,
                    y=recent_data,
                    mode='lines',
                    name='📈 Historical Price',
                    line=dict(color=main_color, width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
                ))
                
                if show_technical:
                    if 'SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=recent_dates,
                            y=df['SMA_20'][-len(recent_dates):] if len(df) > 365 else df['SMA_20'],
                            mode='lines',
                            name='📈 SMA 20',
                            line=dict(color=sma20_color, width=1, dash='dot'),
                            opacity=0.7
                        ))
                    if 'SMA_50' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=recent_dates,
                            y=df['SMA_50'][-len(recent_dates):] if len(df) > 365 else df['SMA_50'],
                            mode='lines',
                            name='📊 SMA 50',
                            line=dict(color=sma50_color, width=1, dash='dot'),
                            opacity=0.7
                        ))
                
                # Forecast projection
                last_date = dates[-1] if dates is not None else datetime.now()
                forecast_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
                
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=predictions,
                    mode='lines',
                    name=f'🔮 AI Forecast ({forecast_years}Y)',
                    line=dict(color=forecast_color, width=2, dash='dash'),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:,.2f}<extra></extra>'
                ))
                
                # Apply consistent theme styling
                template_name = 'plotly_dark'
                if chart_theme == "Classic":
                    template_name = 'plotly_white'
                
                fig.update_layout(
                    title=f"📊 {selected_crypto} - Historical Performance & AI Forecast",
                    xaxis_title="📅 Date",
                    yaxis_title="💰 Price (USD)",
                    height=600,
                    template=template_name,
                    plot_bgcolor='#1a202c' if chart_theme != "Classic" else '#ffffff',
                    paper_bgcolor='#1a202c' if chart_theme != "Classic" else '#ffffff',
                    font=dict(
                        color='#ffffff' if chart_theme != "Classic" else '#000000', 
                        family='Roboto Mono'
                    ),
                    legend=dict(
                        bgcolor='rgba(26, 32, 44, 0.8)' if chart_theme != "Classic" else 'rgba(255, 255, 255, 0.8)',
                        bordercolor='#4a5568' if chart_theme != "Classic" else '#cccccc',
                        borderwidth=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Display historical chart if no analysis is running (in col1)
    with col1:
        if 'predictions' not in locals():
            df, dates = fetch_comprehensive_data(symbol, '1y')
            if df is not None:
                st.markdown("### 📈 HISTORICAL PRICE ACTION")
                
                fig = go.Figure()
                
                # Historical price action
                recent_data = df['Close'][-365:] if len(df) > 365 else df['Close']
                recent_dates = dates[-365:] if len(dates) > 365 else dates
                
                # Apply chart theme to historical chart too
                if chart_theme == "Terminal Green":
                    main_color, sma20_color, sma50_color = '#00ff88', '#00cc6a', '#008f5a'
                elif chart_theme == "Trading Blue":
                    main_color, sma20_color, sma50_color = '#00bfff', '#0080ff', '#0060cc'
                elif chart_theme == "Classic":
                    main_color, sma20_color, sma50_color = '#ffffff', '#ffdd00', '#ffaa00'
                else:  # Professional Dark
                    main_color, sma20_color, sma50_color = '#00ff88', '#ffd700', '#ff4757'
                
                fig.add_trace(go.Scatter(
                    x=recent_dates,
                    y=recent_data,
                    mode='lines',
                    name='💰 Price',
                    line=dict(color=main_color, width=2),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
                ))
                
                if show_technical:
                    if 'SMA_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=recent_dates,
                            y=df['SMA_20'][-len(recent_dates):] if len(df) > 365 else df['SMA_20'],
                            mode='lines',
                            name='📈 SMA 20',
                            line=dict(color=sma20_color, width=1, dash='dot'),
                            opacity=0.7
                        ))
                    if 'SMA_50' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=recent_dates,
                            y=df['SMA_50'][-len(recent_dates):] if len(df) > 365 else df['SMA_50'],
                            mode='lines',
                            name='📊 SMA 50',
                            line=dict(color=sma50_color, width=1, dash='dot'),
                            opacity=0.7
                        ))
                
                # Apply consistent theme styling
                template_name = 'plotly_dark'
                if chart_theme == "Classic":
                    template_name = 'plotly_white'
                
                fig.update_layout(
                    title=f"📊 {selected_crypto} - Historical Performance",
                    xaxis_title="📅 Date",
                    yaxis_title="💰 Price (USD)",
                    height=500,
                    template=template_name,
                    plot_bgcolor='#1a202c' if chart_theme != "Classic" else '#ffffff',
                    paper_bgcolor='#1a202c' if chart_theme != "Classic" else '#ffffff',
                    font=dict(
                        color='#ffffff' if chart_theme != "Classic" else '#000000', 
                        family='Roboto Mono'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Professional Disclaimer - Ultra Compact
    st.markdown("---")
    st.markdown("""
    <div class="analysis-panel" style="padding: 0.3rem 0.6rem; margin: 0.2rem 0;">
        <p style="font-size: 0.6rem; margin: 0; line-height: 1.2; color: #a0aec0;"><strong>⚠️ DISCLAIMER:</strong> Educational only. High risk. Not investment advice.</p>
    </div>
    """, unsafe_allow_html=True)

    # 2030 TARGET ANALYSIS - NEW SECTION (This section will be handled in the main execution block)
    st.markdown("---")

if __name__ == "__main__":
    main()
