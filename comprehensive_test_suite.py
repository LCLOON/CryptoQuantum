"""
Comprehensive CryptoQuantum App Test Suite
Tests every function and analyzes all 30 cached models' 5-year predictions
"""
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

class CryptoQuantumTester:
    def __init__(self):
        self.test_results = {}
        self.prediction_analysis = {}
        self.issues_found = []
        self.recommendations = []
        
    def log_test(self, test_name, success, details="", error=None):
        """Log test results"""
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now()
        }
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        if error:
            print(f"   Error: {error}")
    
    def test_imports(self):
        """Test all module imports"""
        print("\n🔧 Testing Module Imports")
        print("-" * 40)
        
        modules_to_test = [
            ('config', 'CRYPTO_SYMBOLS, AI_MODEL_OPTIONS'),
            ('market_data', 'fetch_comprehensive_data, get_crypto_info'),
            ('ai_models', 'train_advanced_model, analyze_long_term_scenarios'),
            ('cache_loader', 'CacheLoader'),
            ('ui_components', 'setup_page_config, render_terminal_header')
        ]
        
        for module_name, components in modules_to_test:
            try:
                if module_name == 'config':
                    from config import CRYPTO_SYMBOLS, AI_MODEL_OPTIONS
                    self.log_test(f"Import {module_name}", True, f"Symbols: {len(CRYPTO_SYMBOLS)}, Models: {len(AI_MODEL_OPTIONS)}")
                elif module_name == 'market_data':
                    from market_data import fetch_comprehensive_data, get_crypto_info
                    self.log_test(f"Import {module_name}", True, "Market data functions available")
                elif module_name == 'ai_models':
                    from ai_models import train_advanced_model, analyze_long_term_scenarios, display_scenario_analysis
                    self.log_test(f"Import {module_name}", True, "AI model functions available")
                elif module_name == 'cache_loader':
                    from cache_loader import CacheLoader
                    self.log_test(f"Import {module_name}", True, "Cache loader class available")
                elif module_name == 'ui_components':
                    from ui_components import setup_page_config, render_terminal_header
                    self.log_test(f"Import {module_name}", True, "UI components available")
            except Exception as e:
                self.log_test(f"Import {module_name}", False, f"Failed to import {components}", e)
    
    def test_cache_loader_functions(self):
        """Test all CacheLoader functions"""
        print("\n⚡ Testing Cache Loader Functions")
        print("-" * 40)
        
        try:
            from cache_loader import CacheLoader
            cache_loader = CacheLoader()
            
            # Test initialization
            self.log_test("CacheLoader init", True, "Cache loader initialized successfully")
            
            # Test cache availability check
            available_symbols = []
            from config import CRYPTO_SYMBOLS
            for crypto_name, symbol in CRYPTO_SYMBOLS.items():
                if cache_loader.is_cache_available(symbol):
                    available_symbols.append(symbol)
            
            self.log_test("Cache availability check", True, f"{len(available_symbols)} symbols cached")
            
            # Test cached forecasts loading
            if available_symbols:
                test_symbol = available_symbols[0]
                forecasts = cache_loader.get_cached_forecasts(test_symbol, 1825)  # 5 years
                if forecasts:
                    self.log_test("Cache forecast loading", True, f"Loaded {len(forecasts['predictions'])} predictions")
                else:
                    self.log_test("Cache forecast loading", False, "No forecasts returned")
            
        except Exception as e:
            self.log_test("CacheLoader functions", False, "Cache loader testing failed", e)
    
    def test_market_data_functions(self):
        """Test market data functions"""
        print("\n📊 Testing Market Data Functions")
        print("-" * 40)
        
        try:
            from market_data import fetch_comprehensive_data, get_crypto_info
            
            # Test data fetching
            test_symbol = "BTC-USD"
            df, dates = fetch_comprehensive_data(test_symbol, '1y')
            
            if df is not None and not df.empty:
                self.log_test("Market data fetch", True, f"Fetched {len(df)} records with {len(df.columns)} columns")
                
                # Test required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if not missing_cols:
                    self.log_test("Market data columns", True, "All required columns present")
                else:
                    self.log_test("Market data columns", False, f"Missing columns: {missing_cols}")
            else:
                self.log_test("Market data fetch", False, "No data returned")
            
            # Test crypto info
            crypto_info = get_crypto_info(test_symbol)
            if crypto_info and 'current_price' in crypto_info:
                self.log_test("Crypto info fetch", True, f"Current price: ${crypto_info['current_price']:.2f}")
            else:
                self.log_test("Crypto info fetch", False, "No crypto info returned")
                
        except Exception as e:
            self.log_test("Market data functions", False, "Market data testing failed", e)
    
    def test_ai_model_functions(self):
        """Test AI model functions"""
        print("\n🧠 Testing AI Model Functions")
        print("-" * 40)
        
        try:
            from ai_models import analyze_long_term_scenarios, display_scenario_analysis
            
            # Test long-term scenarios
            test_symbol = "BTC-USD"
            analysis = analyze_long_term_scenarios(test_symbol, mode="standard")
            
            if analysis:
                self.log_test("Long-term analysis", True, f"Analysis contains {len(analysis)} keys")
                
                # Check required keys
                required_keys = ['scenarios', 'confidence_intervals', 'symbol']
                missing_keys = [key for key in required_keys if key not in analysis]
                if not missing_keys:
                    self.log_test("Analysis structure", True, "All required keys present")
                else:
                    self.log_test("Analysis structure", False, f"Missing keys: {missing_keys}")
            else:
                self.log_test("Long-term analysis", False, "No analysis returned")
            
            # Test ensemble mode
            ensemble_analysis = analyze_long_term_scenarios(test_symbol, mode="ensemble")
            if ensemble_analysis:
                self.log_test("Ensemble analysis", True, f"Mode: {ensemble_analysis.get('mode', 'unknown')}")
            else:
                self.log_test("Ensemble analysis", False, "No ensemble analysis returned")
                
        except Exception as e:
            self.log_test("AI model functions", False, "AI model testing failed", e)
    
    def analyze_all_cached_predictions(self):
        """Analyze all 30 cached models' 5-year predictions"""
        print("\n🔍 Analyzing All Cached 5-Year Predictions")
        print("=" * 50)
        
        try:
            from cache_loader import CacheLoader
            from config import CRYPTO_SYMBOLS
            
            cache_loader = CacheLoader()
            analysis_results = {}
            
            for crypto_name, symbol in CRYPTO_SYMBOLS.items():
                if cache_loader.is_cache_available(symbol):
                    print(f"\n📊 Analyzing {crypto_name} ({symbol})")
                    
                    # Get 5-year predictions (1825 days)
                    forecasts = cache_loader.get_cached_forecasts(symbol, 1825)
                    
                    if forecasts and forecasts['predictions']:
                        predictions = forecasts['predictions']
                        
                        # Calculate analysis metrics
                        first_price = predictions[0]
                        final_price = predictions[-1]
                        max_price = max(predictions)
                        min_price = min(predictions)
                        
                        # Calculate growth metrics
                        total_growth = (final_price / first_price - 1) * 100
                        max_growth = (max_price / first_price - 1) * 100
                        min_growth = (min_price / first_price - 1) * 100
                        
                        # Calculate CAGR (Compound Annual Growth Rate)
                        years = 5
                        cagr = ((final_price / first_price) ** (1/years) - 1) * 100
                        
                        # Volatility analysis
                        daily_returns = [predictions[i+1]/predictions[i] - 1 for i in range(len(predictions)-1)]
                        volatility = np.std(daily_returns) * np.sqrt(365) * 100  # Annualized volatility %
                        
                        # Detect issues
                        issues = []
                        if cagr > 500:  # More than 500% annual growth
                            issues.append(f"Extreme CAGR: {cagr:.1f}%")
                        if cagr < -90:  # More than 90% annual decline
                            issues.append(f"Extreme decline: {cagr:.1f}%")
                        if volatility > 200:  # More than 200% volatility
                            issues.append(f"Extreme volatility: {volatility:.1f}%")
                        if any(np.isnan(p) or np.isinf(p) for p in predictions):
                            issues.append("Contains NaN/Inf values")
                        if any(p <= 0 for p in predictions):
                            issues.append("Contains negative/zero prices")
                        
                        # Check for flat predictions (same value repeated)
                        unique_predictions = len(set(predictions))
                        if unique_predictions < len(predictions) * 0.1:  # Less than 10% unique values
                            issues.append("Predictions too flat/repetitive")
                        
                        analysis_results[symbol] = {
                            'crypto_name': crypto_name,
                            'first_price': first_price,
                            'final_price': final_price,
                            'total_growth': total_growth,
                            'cagr': cagr,
                            'max_growth': max_growth,
                            'min_growth': min_growth,
                            'volatility': volatility,
                            'issues': issues,
                            'prediction_count': len(predictions)
                        }
                        
                        # Print summary
                        status = "🚨 ISSUES" if issues else "✅ HEALTHY"
                        print(f"  {status} Price: ${first_price:.6f} → ${final_price:.6f}")
                        print(f"  📈 Total Growth: {total_growth:+.1f}% | CAGR: {cagr:+.1f}%")
                        print(f"  📊 Volatility: {volatility:.1f}% | Range: {min_growth:+.1f}% to {max_growth:+.1f}%")
                        
                        if issues:
                            print(f"  ⚠️  Issues: {', '.join(issues)}")
                    else:
                        print(f"  ❌ No predictions available")
                        analysis_results[symbol] = {'error': 'No predictions available'}
            
            self.prediction_analysis = analysis_results
            self.log_test("5-year predictions analysis", True, f"Analyzed {len(analysis_results)} symbols")
            
        except Exception as e:
            self.log_test("5-year predictions analysis", False, "Analysis failed", e)
    
    def generate_fix_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n🔧 Generating Fix Recommendations")
        print("=" * 50)
        
        extreme_cagr_symbols = []
        extreme_volatility_symbols = []
        data_quality_issues = []
        flat_prediction_symbols = []
        
        for symbol, analysis in self.prediction_analysis.items():
            if 'error' in analysis:
                continue
                
            issues = analysis.get('issues', [])
            
            for issue in issues:
                if "Extreme CAGR" in issue:
                    extreme_cagr_symbols.append((symbol, analysis['cagr']))
                elif "Extreme volatility" in issue:
                    extreme_volatility_symbols.append((symbol, analysis['volatility']))
                elif "NaN/Inf" in issue or "negative" in issue:
                    data_quality_issues.append(symbol)
                elif "flat" in issue:
                    flat_prediction_symbols.append(symbol)
        
        recommendations = []
        
        # CAGR Issues
        if extreme_cagr_symbols:
            recommendations.append({
                'issue': 'Extreme CAGR Values',
                'affected_symbols': len(extreme_cagr_symbols),
                'examples': extreme_cagr_symbols[:3],
                'severity': 'HIGH',
                'fix': 'Implement progressive CAGR caps: 30d: 800%, 90d: 600%, 1y: 300%, 5y: 150%',
                'code_location': 'stunning_crypto_app.py lines 280-330 (validation logic)'
            })
        
        # Volatility Issues
        if extreme_volatility_symbols:
            recommendations.append({
                'issue': 'Extreme Volatility',
                'affected_symbols': len(extreme_volatility_symbols),
                'examples': extreme_volatility_symbols[:3],
                'severity': 'MEDIUM',
                'fix': 'Add volatility smoothing in prediction generation and validation',
                'code_location': 'ai_models.py predict_realistic_future function'
            })
        
        # Data Quality Issues
        if data_quality_issues:
            recommendations.append({
                'issue': 'Data Quality Problems',
                'affected_symbols': len(data_quality_issues),
                'examples': data_quality_issues[:3],
                'severity': 'CRITICAL',
                'fix': 'Add NaN/Inf detection and replacement in validation pipeline',
                'code_location': 'All prediction validation sections'
            })
        
        # Flat Predictions
        if flat_prediction_symbols:
            recommendations.append({
                'issue': 'Flat/Repetitive Predictions',
                'affected_symbols': len(flat_prediction_symbols),
                'examples': flat_prediction_symbols[:3],
                'severity': 'MEDIUM',
                'fix': 'Add noise injection and ensure model diversity in predictions',
                'code_location': 'cache_loader.py and ai_models.py prediction generation'
            })
        
        # Print recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"\n🔧 RECOMMENDATION #{i}")
            print(f"Issue: {rec['issue']}")
            print(f"Severity: {rec['severity']}")
            print(f"Affected: {rec['affected_symbols']} symbols")
            print(f"Examples: {rec['examples']}")
            print(f"Fix: {rec['fix']}")
            print(f"Location: {rec['code_location']}")
        
        self.recommendations = recommendations
        
        # Overall health assessment
        total_symbols = len(self.prediction_analysis)
        healthy_symbols = sum(1 for analysis in self.prediction_analysis.values() 
                            if 'issues' in analysis and not analysis['issues'])
        
        health_percentage = (healthy_symbols / total_symbols) * 100 if total_symbols > 0 else 0
        
        print(f"\n📊 OVERALL HEALTH ASSESSMENT")
        print(f"Total Symbols Analyzed: {total_symbols}")
        print(f"Healthy Predictions: {healthy_symbols}")
        print(f"Health Percentage: {health_percentage:.1f}%")
        
        if health_percentage >= 80:
            print("✅ EXCELLENT: Most predictions are healthy")
        elif health_percentage >= 60:
            print("⚠️  GOOD: Some issues need attention")
        elif health_percentage >= 40:
            print("🔧 FAIR: Multiple issues require fixes")
        else:
            print("🚨 POOR: Significant problems need immediate attention")
    
    def test_ui_components(self):
        """Test UI component functions"""
        print("\n🎨 Testing UI Components")
        print("-" * 40)
        
        try:
            from ui_components import (
                setup_page_config, render_terminal_header, render_cache_status,
                render_asset_selection, render_analysis_parameters
            )
            
            # Test function availability
            ui_functions = [
                'setup_page_config', 'render_terminal_header', 'render_cache_status',
                'render_asset_selection', 'render_analysis_parameters', 'render_display_settings',
                'render_advanced_controls', 'render_results_cards'
            ]
            
            available_functions = 0
            for func_name in ui_functions:
                try:
                    func = getattr(__import__('ui_components'), func_name)
                    available_functions += 1
                except AttributeError:
                    pass
            
            self.log_test("UI components availability", True, f"{available_functions}/{len(ui_functions)} functions available")
            
        except Exception as e:
            self.log_test("UI components", False, "UI components testing failed", e)
    
    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("🧪 CryptoQuantum Comprehensive Test Suite")
        print("=" * 60)
        print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all tests
        self.test_imports()
        self.test_cache_loader_functions()
        self.test_market_data_functions()
        self.test_ai_model_functions()
        self.test_ui_components()
        self.analyze_all_cached_predictions()
        self.generate_fix_recommendations()
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate final summary report"""
        print("\n" + "=" * 60)
        print("📋 FINAL TEST SUMMARY REPORT")
        print("=" * 60)
        
        # Test results summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Test Results: {passed_tests}/{total_tests} PASSED ({(passed_tests/total_tests)*100:.1f}%)")
        
        if failed_tests > 0:
            print(f"\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result['success']:
                    print(f"  • {test_name}: {result['error']}")
        
        # Prediction analysis summary
        if self.prediction_analysis:
            symbols_with_issues = sum(1 for analysis in self.prediction_analysis.values() 
                                    if 'issues' in analysis and analysis['issues'])
            total_analyzed = len(self.prediction_analysis)
            
            print(f"\n📈 Prediction Analysis: {symbols_with_issues}/{total_analyzed} symbols have issues")
        
        # Priority recommendations
        if self.recommendations:
            critical_issues = [r for r in self.recommendations if r['severity'] == 'CRITICAL']
            high_issues = [r for r in self.recommendations if r['severity'] == 'HIGH']
            
            print(f"\n🚨 Priority Fixes Needed:")
            print(f"  • Critical Issues: {len(critical_issues)}")
            print(f"  • High Priority Issues: {len(high_issues)}")
        
        print(f"\n🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

def main():
    """Run the comprehensive test suite"""
    tester = CryptoQuantumTester()
    tester.run_complete_test_suite()

if __name__ == "__main__":
    main()
