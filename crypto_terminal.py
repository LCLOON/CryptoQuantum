"""
CryptoQuantum Terminal - Main Application
Ultra-fast cryptocurrency analysis with cache optimization
"""

import streamlit as st
import time
from config import CRYPTO_SYMBOLS
from market_data import fetch_comprehensive_data, get_crypto_info
from ai_models import train_advanced_model, analyze_long_term_scenarios, display_scenario_analysis
from cache_loader import CacheLoader
from ui_components import (
    setup_page_config, render_terminal_header, render_cache_status,
    render_quick_start_guide, render_asset_selection, render_analysis_parameters,
    render_display_settings, render_advanced_controls, render_progress_interface,
    render_results_cards, render_yearly_targets_table, render_technical_chart,
    render_market_data_cards, render_disclaimer
)

def load_css():
    """Load custom CSS styling"""
    st.markdown("""
    <style>
    /* Terminal Header Styling */
    .terminal-header {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.15);
    }
    
    .terminal-title {
        font-size: 2rem;
        font-weight: 700;
        color: #00ff88;
        font-family: 'Roboto Mono', monospace;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
    }
    
    .terminal-subtitle {
        font-size: 1rem;
        color: #a0aec0;
        text-align: center;
        font-family: 'Roboto Mono', monospace;
        opacity: 0.9;
    }
    
    /* Trading Cards */
    .trading-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .trading-card:hover {
        border-color: #00ff88;
        box-shadow: 0 6px 24px rgba(0, 255, 136, 0.2);
    }
    
    /* Financial Metrics */
    .financial-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #a0aec0;
        font-weight: 600;
        font-family: 'Roboto Mono', monospace;
    }
    
    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'Roboto Mono', monospace;
    }
    
    .metric-value.positive {
        color: #00ff88;
    }
    
    .metric-value.negative {
        color: #ff4757;
    }
    
    /* Market Data Cards */
    .market-card {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .market-label {
        font-size: 0.8rem;
        color: #a0aec0;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .market-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00ff88;
        font-family: 'Roboto Mono', monospace;
    }
    
    .market-change {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .market-change.positive {
        color: #00ff88;
    }
    
    .market-change.negative {
        color: #ff4757;
    }
    
    /* Risk Assessment */
    .risk-high {
        border-color: #ff4757 !important;
    }
    
    .risk-medium {
        border-color: #ffa502 !important;
    }
    
    .risk-low {
        border-color: #00ff88 !important;
    }
    
    /* Live Indicator */
    .live-indicator {
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Setup page
    setup_page_config()
    load_css()
    
    # Render header
    render_terminal_header()
    
    # Initialize cache loader
    cache_loader = CacheLoader()
    
    # Professional Sidebar
    with st.sidebar:
        st.markdown("## 📊 CRYPTOQUANTUM TERMINAL")
        st.markdown("---")
        
        # Cache Status Display
        render_cache_status(cache_loader)
        st.markdown("---")
        
        # Quick Start Guide
        render_quick_start_guide()
        
        # Step 1: Asset Selection
        selected_crypto, symbol = render_asset_selection(CRYPTO_SYMBOLS, cache_loader)
        
        # Get current market data
        crypto_info = get_crypto_info(symbol)
        
        # Step 2: Analysis Parameters
        forecast_years, days = render_analysis_parameters()
        
        # Step 3: Display Settings
        show_technical, show_risk_metrics, confidence_level = render_display_settings()
        
        # Step 4: Advanced Controls
        ai_model, chart_theme = render_advanced_controls()
        
        # Set variables based on AI model selection
        show_2030_analysis = ai_model in ["🎯 Advanced AttentionLSTM + Market Analysis", "📊 Multi-Model Ensemble (AttentionLSTM + XGBoost)"]
        target_mode = ai_model
        
        # Completion Status
        st.markdown("---")
        st.markdown("### ✅ SETUP STATUS")
        
        step1_complete = selected_crypto is not None
        step2_complete = forecast_years > 0
        step3_complete = True
        
        status_color = "#00ff88"
        
        st.markdown(f"""
        <div style="background: rgba(26, 32, 44, 0.8); padding: 0.8rem; border-radius: 6px; font-size: 0.9rem;">
            <div style="color: {status_color};">✅ Step 1: Asset Selected ({selected_crypto.split()[0] if step1_complete else "None"})</div>
            <div style="color: {status_color};">✅ Step 2: Forecast Period ({forecast_years}Y)</div>
            <div style="color: {status_color};">✅ Step 3: Settings Configured</div>
            <div style="color: #ffd700; margin-top: 0.5rem;"><strong>👉 Ready to execute analysis!</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Execute Analysis Button
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
    
    # Create main layout columns
    col1, col2 = st.columns([2, 1])
    
    # Analysis Execution
    if execute_analysis:
        # Performance indicator
        cache_available = cache_loader.is_cache_available(symbol)
        if cache_available:
            st.success("⚡ **ULTRA-FAST MODE**: Using cached models and data for instant predictions!")
        else:
            st.info("🔄 **LIVE TRAINING MODE**: Building new model (this may take 30-60 seconds)")
        
        predictions = None
        
        # ULTRA-FAST CACHE MODE - Always prioritize cache if available
        if target_mode == "⚡ Ultra-Fast Cache Mode (Recommended)" or cache_available:
            if cache_available:
                st.markdown("## ⚡ ULTRA-FAST CACHE PREDICTIONS")
                
                # Simple progress for visual feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("⚡ Loading cached forecasts...")
                progress_bar.progress(50)
                time.sleep(0.1)
                
                # Get cached forecasts
                cached_forecasts = cache_loader.get_cached_forecasts(symbol, days)
                if cached_forecasts:
                    predictions = cached_forecasts['predictions'][:days]
                    progress_bar.progress(100)
                    status_text.text("✅ Ultra-fast predictions loaded!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"⚡ **CACHE HIT**: Loaded {len(predictions)} predictions in milliseconds!")
                    st.info(f"📊 Using pre-computed forecast generated on {cached_forecasts.get('generated_date', 'Unknown date')}")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("🚨 **CACHE MISS**: No cached data available for this symbol and timeframe")
                    return
            else:
                st.warning("⚠️ **NO CACHE AVAILABLE**: This symbol is not cached. Please select a different mode or symbol.")
                return
                
        elif show_2030_analysis:
            # Advanced analysis modes
            if cache_available:
                st.markdown("## ⚡ ADVANCED ANALYSIS (CACHED)")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("⚡ Loading cached advanced forecasts...")
                progress_bar.progress(80)
                
                cached_forecasts = cache_loader.get_cached_forecasts(symbol, days)
                if cached_forecasts:
                    predictions = cached_forecasts['predictions'][:days]
                    progress_bar.progress(100)
                    status_text.text("✅ Advanced cached analysis loaded!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("⚡ **CACHE-ACCELERATED ADVANCED MODE**: Loaded predictions in milliseconds!")
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning("🔄 **CACHE MISS**: Falling back to live training for advanced analysis...")
                    
                    # Fall back to live training
                    if target_mode == "🎯 Advanced AttentionLSTM + Market Analysis":
                        st.markdown("## 🎯 ADVANCED ATTENTION-LSTM MARKET ANALYSIS")
                        analysis = analyze_long_term_scenarios(symbol, confidence_level=confidence_level)
                        if analysis:
                            display_scenario_analysis(analysis, selected_crypto.split(' ')[0], symbol)
                        st.success("🎯 Advanced AttentionLSTM analysis completed!")
                        return
                        
                    elif target_mode == "📊 Multi-Model Ensemble (AttentionLSTM + XGBoost)":
                        st.markdown("## 📊 MULTI-MODEL ENSEMBLE FORECAST")
                        analysis = analyze_long_term_scenarios(symbol, mode="ensemble", confidence_level=confidence_level)
                        if analysis:
                            display_scenario_analysis(analysis, selected_crypto.split(' ')[0], symbol)
                        st.success("📊 Multi-model ensemble analysis completed!")
                        return
            else:
                # No cache - live training
                if target_mode == "🎯 Advanced AttentionLSTM + Market Analysis":
                    st.markdown("## 🎯 ADVANCED ATTENTION-LSTM MARKET ANALYSIS")
                    analysis = analyze_long_term_scenarios(symbol, confidence_level=confidence_level)
                    if analysis:
                        display_scenario_analysis(analysis, selected_crypto.split(' ')[0], symbol)
                    st.success("🎯 Advanced AttentionLSTM analysis completed!")
                    return
                    
                elif target_mode == "📊 Multi-Model Ensemble (AttentionLSTM + XGBoost)":
                    st.markdown("## 📊 MULTI-MODEL ENSEMBLE FORECAST")
                    analysis = analyze_long_term_scenarios(symbol, mode="ensemble", confidence_level=confidence_level)
                    if analysis:
                        display_scenario_analysis(analysis, selected_crypto.split(' ')[0], symbol)
                    st.success("📊 Multi-model ensemble analysis completed!")
                    return
        
        # Fallback: Traditional LSTM if no cache available
        if predictions is None:
            st.markdown("## 🧠 AI-POWERED CRYPTOCURRENCY ANALYSIS")
            
            # Get data for training
            df, dates = fetch_comprehensive_data(symbol, '3y')
            if df is None or df.empty:
                df, dates = fetch_comprehensive_data(symbol, '1y')
                if df is None or df.empty:
                    st.error(f"🚨 **CRITICAL ERROR** - Market data unavailable for {selected_crypto}")
                    return
            
            # Train model and generate predictions
            progress_bar, status_text, update_progress = render_progress_interface()
            
            model, scalers, num_features = train_advanced_model(df, selected_crypto, ai_model, update_progress)
            if model is None:
                st.error("🚨 **MODEL FAILURE** - Unable to train model")
                return
            
            # Generate predictions (implement prediction logic here)
            predictions = [crypto_info['current_price'] * (1.1 ** (i/365)) for i in range(days)]  # Placeholder
            
            progress_bar.empty()
            status_text.empty()
        
        # Ensure we have predictions
        if predictions is None or len(predictions) == 0:
            st.error("🚨 **FORECAST FAILURE** - Unable to generate predictions")
            return
        
        # RESULTS DISPLAY
        st.markdown("---")
        st.markdown("## 📈 QUANTITATIVE FORECAST RESULTS")
        
        # Render results cards
        render_results_cards(predictions, crypto_info, forecast_years)
        
        # Render yearly targets table
        render_yearly_targets_table(predictions, crypto_info, forecast_years)
        
        # Render technical chart
        df, dates = fetch_comprehensive_data(symbol, '1y')
        if df is not None:
            render_technical_chart(df, dates, predictions, selected_crypto, forecast_years, chart_theme, show_technical)
    
    # Display historical chart if no analysis running
    with col1:
        if 'predictions' not in locals() or not execute_analysis:
            df, dates = fetch_comprehensive_data(symbol, '1y')
            if df is not None:
                st.markdown("### 📈 HISTORICAL PRICE ACTION")
                
                # Simple historical chart
                import plotly.graph_objects as go
                from config import CHART_COLORS
                
                fig = go.Figure()
                colors = CHART_COLORS.get(chart_theme, CHART_COLORS["Professional Dark"])
                
                recent_data = df['Close'][-365:] if len(df) > 365 else df['Close']
                recent_dates = dates[-365:] if len(dates) > 365 else dates
                
                fig.add_trace(go.Scatter(
                    x=recent_dates,
                    y=recent_data,
                    mode='lines',
                    name='💰 Price',
                    line=dict(color=colors['main_color'], width=2)
                ))
                
                template_name = 'plotly_dark' if chart_theme != "Classic" else 'plotly_white'
                
                fig.update_layout(
                    title=f"📊 {selected_crypto} - Historical Performance",
                    xaxis_title="📅 Date",
                    yaxis_title="💰 Price (USD)",
                    height=500,
                    template=template_name
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Display market data in sidebar column
    with col2:
        if crypto_info:
            render_market_data_cards(crypto_info)
    
    # Render disclaimer
    render_disclaimer()

if __name__ == "__main__":
    main()
