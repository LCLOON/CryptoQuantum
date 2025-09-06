"""
UI Components for CryptoQuantum Terminal
Handles all user interface elements and styling
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from config import CHART_COLORS, CHART_THEMES

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="CryptoQuantum Terminal",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_terminal_header():
    """Render the professional terminal header"""
    st.markdown("""
    <div class="terminal-header">
        <div class="terminal-title">
            <span class="live-indicator"></span>CRYPTOQUANTUM TERMINAL
        </div>
        <div class="terminal-subtitle">
            Advanced Quantitative Analysis & Algorithmic Forecasting Platform
        </div>
        <div style="text-align: right; margin-top: 0.5rem; font-family: 'Roboto Mono', monospace; font-size: 0.8rem; color: #00ff88; opacity: 0.8;">
            Developed by <strong>Lewis Loon</strong> | Professional Trading Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_cache_status(cache_loader):
    """Render cache status in sidebar"""
    st.markdown("### ⚡ PERFORMANCE CACHE")
    cache_stats = cache_loader.get_cache_stats()
    if cache_stats:
        st.success(f"✅ **{cache_stats['models_count']} cryptos** cached")
        st.info(f"📊 **{cache_stats['total_size_mb']:.1f} MB** cache size")
        available_symbols = cache_loader.get_available_symbols()
        st.caption(f"🚀 Lightning-fast predictions for {len(available_symbols)} cryptocurrencies")
    else:
        st.warning("⚠️ **No cache available**")
        st.caption("Run `python pretrain_models.py` for faster predictions")

def render_quick_start_guide():
    """Render the quick start guide"""
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

def render_asset_selection(crypto_symbols, cache_loader):
    """Render asset selection with cache indicators"""
    st.markdown("### 🎯 STEP 1: ASSET SELECTION")
    
    # Create options with cache indicators
    available_symbols = cache_loader.get_available_symbols()
    crypto_options = []
    for crypto_key in crypto_symbols.keys():
        symbol = crypto_symbols[crypto_key]
        if symbol in available_symbols:
            crypto_options.append(f"⚡ {crypto_key}")  # Lightning bolt for cached
        else:
            crypto_options.append(f"🔄 {crypto_key}")  # Loading symbol for non-cached
    
    selected_crypto_display = st.selectbox(
        "Select Trading Pair:",
        crypto_options,
        index=0,
        help="⚡ = Ultra-fast cached predictions | 🔄 = Live training required"
    )
    
    # Extract the actual crypto key
    selected_crypto = selected_crypto_display.split(' ', 1)[1]  # Remove emoji prefix
    symbol = crypto_symbols[selected_crypto]
    
    return selected_crypto, symbol

def render_analysis_parameters():
    """Render analysis parameters section"""
    st.markdown("### ⏱️ STEP 2: ANALYSIS PARAMETERS")
    forecast_years = st.slider(
        'Forecast Horizon (Years)', 
        1, 5, 3,
        help="Select how many years into the future you want to predict"
    )
    days = forecast_years * 365
    return forecast_years, days

def render_display_settings():
    """Render display settings section"""
    st.markdown("### 📈 STEP 3: DISPLAY SETTINGS")
    
    # Technical indicators toggle
    show_technical = st.button(
        "📊 Technical Indicators: ON" if 'show_technical' not in st.session_state or st.session_state['show_technical'] else "📊 Technical Indicators: OFF",
        key="show_technical_btn",
        use_container_width=True,
        help="Show moving averages and trend lines"
    )
    if show_technical:
        st.session_state['show_technical'] = not st.session_state.get('show_technical', True)
    show_technical = st.session_state.get('show_technical', True)

    # Risk metrics toggle
    show_risk_metrics = st.button(
        "⚠️ Risk Analysis: ON" if 'show_risk_metrics' not in st.session_state or st.session_state['show_risk_metrics'] else "⚠️ Risk Analysis: OFF",
        key="show_risk_metrics_btn",
        use_container_width=True,
        help="Display risk assessment metrics"
    )
    if show_risk_metrics:
        st.session_state['show_risk_metrics'] = not st.session_state.get('show_risk_metrics', True)
    show_risk_metrics = st.session_state.get('show_risk_metrics', True)

    # Confidence level
    confidence_level = st.select_slider(
        "🎯 Confidence Interval", 
        options=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
        value=0.85,
        help="Statistical confidence level for predictions"
    )
    
    return show_technical, show_risk_metrics, confidence_level

def render_advanced_controls():
    """Render advanced controls section"""
    st.markdown("### 🎛️ STEP 4: ADVANCED CONTROLS")
    st.markdown("<small style='color: #a0aec0;'>⚙️ Optional - Advanced users only</small>", unsafe_allow_html=True)

    # AI Model Selection
    ai_model = st.selectbox(
        "🤖 AI Model Engine",
        [
            "⚡ Ultra-Fast Cache Mode (Recommended)",
            "🎯 Advanced AttentionLSTM + Market Analysis",
            "📊 Multi-Model Ensemble (AttentionLSTM + XGBoost)"
        ],
        help="Choose the AI model architecture for predictions"
    )

    # Chart theme selection
    chart_theme = st.selectbox(
        "🎭 Chart Style", 
        CHART_THEMES,
        help="Choose your preferred chart appearance"
    )
    
    return ai_model, chart_theme

def render_results_cards(predictions, crypto_info, forecast_years):
    """Render professional results cards"""
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

def render_yearly_targets_table(predictions, crypto_info, forecast_years):
    """Render yearly price targets table"""
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

def render_technical_chart(df, dates, predictions, selected_crypto, forecast_years, chart_theme, show_technical):
    """Render technical analysis chart"""
    st.markdown("### 📊 TECHNICAL CHART ANALYSIS")
    
    fig = go.Figure()
    
    # Get chart colors
    colors = CHART_COLORS.get(chart_theme, CHART_COLORS["Professional Dark"])
    
    # Historical price action
    recent_data = df['Close'][-365:] if len(df) > 365 else df['Close']
    recent_dates = dates[-365:] if len(dates) > 365 else dates
    
    fig.add_trace(go.Scatter(
        x=recent_dates,
        y=recent_data,
        mode='lines',
        name='📈 Historical Price',
        line=dict(color=colors['main_color'], width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Technical indicators
    if show_technical:
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=recent_dates,
                y=df['SMA_20'][-len(recent_dates):] if len(df) > 365 else df['SMA_20'],
                mode='lines',
                name='📈 SMA 20',
                line=dict(color=colors['sma20_color'], width=1, dash='dot'),
                opacity=0.7
            ))
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(
                x=recent_dates,
                y=df['SMA_50'][-len(recent_dates):] if len(df) > 365 else df['SMA_50'],
                mode='lines',
                name='📊 SMA 50',
                line=dict(color=colors['sma50_color'], width=1, dash='dot'),
                opacity=0.7
            ))
    
    # Forecast projection
    last_date = dates[-1] if dates is not None else datetime.now()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(predictions)+1)]
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines',
        name=f'🔮 AI Forecast ({forecast_years}Y)',
        line=dict(color=colors['forecast_color'], width=2, dash='dash'),
        hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Apply theme styling
    template_name = 'plotly_dark' if chart_theme != "Classic" else 'plotly_white'
    
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

def render_market_data_cards(crypto_info):
    """Render current market data cards"""
    st.markdown("### 💹 CURRENT MARKET DATA")
    
    # Professional Market Data Cards
    price_change_24h = crypto_info.get('price_change_24h', 0)
    price_change_class = "positive" if price_change_24h > 0 else "negative"
    
    st.markdown(f"""
    <div class="market-card">
        <div class="market-label">💰 CURRENT PRICE</div>
        <div class="market-value">${crypto_info['current_price']:,.2f}</div>
        <div class="market-change {price_change_class}">
            {price_change_24h:+.2f}% (24H)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'market_cap' in crypto_info and crypto_info['market_cap']:
        st.markdown(f"""
        <div class="market-card">
            <div class="market-label">📊 MARKET CAP</div>
            <div class="market-value">${crypto_info['market_cap']/1e9:.2f}B</div>
        </div>
        """, unsafe_allow_html=True)
    
    if 'volume_24h' in crypto_info and crypto_info['volume_24h']:
        st.markdown(f"""
        <div class="market-card">
            <div class="market-label">📈 24H VOLUME</div>
            <div class="market-value">${crypto_info['volume_24h']/1e6:.0f}M</div>
        </div>
        """, unsafe_allow_html=True)

def render_disclaimer():
    """Render professional disclaimer"""
    st.markdown("---")
    st.markdown("""
    <div class="analysis-panel" style="padding: 0.3rem 0.6rem; margin: 0.2rem 0;">
        <p style="font-size: 0.6rem; margin: 0; line-height: 1.2; color: #a0aec0;"><strong>⚠️ DISCLAIMER:</strong> Educational only. High risk. Not investment advice.</p>
    </div>
    """, unsafe_allow_html=True)

def render_progress_interface():
    """Render professional loading interface"""
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 SYSTEM STATUS")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(min(progress, 100))
            status_text.text(message)
        
        return progress_bar, status_text, update_progress
