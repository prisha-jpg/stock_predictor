"""
AI Stock Predictor - FastAPI REST API
Comprehensive API covering all features from stock_predictor.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import warnings
import os
import json
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()
warnings.filterwarnings('ignore')

# Import from stock_predictor module
try:
    from stock_predictor import (
        TransparentStockAnalyzer,
        StockPredictor,
        NewsAnalyzer,
        get_ist_time,
        format_ist_time,
        TOP_INDIAN_STOCKS,
        scan_stocks,
        calculate_risk_management,
        analyze_chart_bullish_bearish_signals,
        get_premarket_data
    )
    STOCK_PREDICTOR_AVAILABLE = True
except ImportError:
    STOCK_PREDICTOR_AVAILABLE = False
    print("Warning: stock_predictor.py not found. Some features will be limited.")

# Initialize FastAPI app
app = FastAPI(
    title="AI Stock Predictor API",
    description="Comprehensive Stock Analysis API with AI-Powered Predictions",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache
cache_store = {}
cache_ttl = {}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_cached_data(key: str, ttl_minutes: int = 60):
    """Get cached data if not expired"""
    if key in cache_store:
        if key in cache_ttl and time.time() - cache_ttl[key] < ttl_minutes * 60:
            return cache_store[key]
        else:
            del cache_store[key]
            if key in cache_ttl:
                del cache_ttl[key]
    return None

def set_cached_data(key: str, data, ttl_minutes: int = 60):
    """Set data in cache with TTL"""
    cache_store[key] = data
    cache_ttl[key] = time.time()

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, RELIANCE.NS)")
    prediction_days: int = Field(7, description="Number of days to predict", ge=1, le=30)
    analysis_mode: str = Field("deep", description="Analysis mode: 'quick' or 'deep'")
    user_position: str = Field("dont_own", description="User position: 'own' or 'dont_own'")

class DailyRecommendationsRequest(BaseModel):
    max_stocks: int = Field(15, description="Maximum number of recommendations", ge=5, le=50)
    min_return: float = Field(0.0, description="Minimum predicted return percentage")

class RiskAnalysisRequest(BaseModel):
    current_price: float = Field(..., description="Current stock price")
    predicted_price: float = Field(..., description="Predicted stock price")
    confidence: float = Field(..., description="Prediction confidence (0-100)")
    portfolio_value: float = Field(100000, description="Total portfolio value")

class NewsAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    company_name: Optional[str] = Field(None, description="Company name")

class ChartAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    period: str = Field("1y", description="Data period (e.g., 1mo, 3mo, 6mo, 1y, 2y)")

class StockComparisonRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of stock symbols to compare")
    metrics: List[str] = Field(["price", "volume", "rsi"], description="Metrics to compare")

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Stock Predictor API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 15px;
                color: #333;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 { 
                color: #667eea; 
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .endpoint { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .method { 
                display: inline-block;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
                margin-right: 10px;
                font-size: 12px;
            }
            .get { background: #28a745; color: white; }
            .post { background: #007bff; color: white; }
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .feature-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .feature-card h3 {
                margin-top: 0;
                font-size: 18px;
            }
            .docs-link {
                text-align: center;
                margin: 30px 0;
            }
            .docs-link a {
                display: inline-block;
                padding: 15px 30px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                margin: 0 10px;
                transition: all 0.3s;
            }
            .docs-link a:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 AI Stock Predictor API</h1>
            <p class="subtitle">Comprehensive Stock Analysis API with AI-Powered Predictions</p>
            
            <div class="docs-link">
                <a href="/docs">📚 Interactive API Docs (Swagger)</a>
                <a href="/redoc">📖 API Documentation (ReDoc)</a>
            </div>

            <h2>🌟 Key Features</h2>
            <div class="features">
                <div class="feature-card">
                    <h3>📊 Stock Analysis</h3>
                    <p>Complete technical & fundamental analysis with 20+ indicators</p>
                </div>
                <div class="feature-card">
                    <h3>🤖 AI Predictions</h3>
                    <p>6 ML models with multi-timeframe predictions (1-30 days)</p>
                </div>
                <div class="feature-card">
                    <h3>🎯 Daily Picks</h3>
                    <p>Automated daily stock recommendations from 100+ stocks</p>
                </div>
                <div class="feature-card">
                    <h3>⚠️ Risk Management</h3>
                    <p>Stop-loss, position sizing, and portfolio recommendations</p>
                </div>
                <div class="feature-card">
                    <h3>📰 News Sentiment</h3>
                    <p>Real-time news analysis and sentiment scoring</p>
                </div>
                <div class="feature-card">
                    <h3>🌍 Global Markets</h3>
                    <p>Track 9+ major indices worldwide</p>
                </div>
            </div>

            <h2>🚀 Quick Start Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/health</strong>
                <p>Check API health status</p>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/analyze</strong>
                <p>Analyze individual stock with AI predictions</p>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/daily-recommendations</strong>
                <p>Get daily stock recommendations</p>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/stock/{symbol}</strong>
                <p>Get quick stock overview</p>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/technical-analysis</strong>
                <p>Get technical indicators for a stock</p>
            </div>

            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/news-sentiment</strong>
                <p>Analyze news sentiment for a stock</p>
            </div>

            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/global-markets</strong>
                <p>Get global market indices data</p>
            </div>

            <h2>📖 Example Usage</h2>
            <div class="endpoint">
                <strong>cURL Example:</strong>
                <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST "http://localhost:8000/api/analyze" \\
  -H "Content-Type: application/json" \\
  -d '{
    "symbol": "AAPL",
    "prediction_days": 7,
    "analysis_mode": "deep"
  }'</pre>
            </div>

            <div class="endpoint">
                <strong>Python Example:</strong>
                <pre style="background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto;">
import requests

response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "symbol": "RELIANCE.NS",
        "prediction_days": 7,
        "analysis_mode": "deep"
    }
)
print(response.json())</pre>
            </div>

            <h2>📊 Supported Markets</h2>
            <ul>
                <li><strong>US Stocks:</strong> AAPL, MSFT, GOOGL, TSLA, etc.</li>
                <li><strong>Indian Stocks:</strong> RELIANCE.NS, TCS.NS, INFY.NS (100+ stocks)</li>
                <li><strong>Global Markets:</strong> European, Asian stocks</li>
            </ul>

            <h2>🔑 API Features</h2>
            <ul>
                <li>✅ No authentication required (for now)</li>
                <li>✅ RESTful JSON API</li>
                <li>✅ Comprehensive error handling</li>
                <li>✅ Caching for performance</li>
                <li>✅ CORS enabled</li>
                <li>✅ OpenAPI/Swagger documentation</li>
            </ul>

            <h2>⚠️ Disclaimer</h2>
            <p style="background: #fff3cd; padding: 15px; border-radius: 8px; color: #856404; border-left: 4px solid #ffc107;">
                <strong>Important:</strong> This API is for educational and informational purposes only. 
                It should not be considered as financial advice. Always conduct your own research 
                and consult with financial advisors before making investment decisions.
            </p>

            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee;">
                <p><strong>AI Stock Predictor API v2.0</strong></p>
                <p>Built with FastAPI • Powered by Machine Learning</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": format_ist_time() if STOCK_PREDICTOR_AVAILABLE else datetime.now().isoformat(),
        "version": "2.0",
        "features": {
            "stock_predictor": STOCK_PREDICTOR_AVAILABLE,
            "cache_size": len(cache_store),
            "uptime": "running"
        }
    }

# ============================================================================
# STOCK ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/analyze")
async def analyze_stock(request: StockAnalysisRequest):
    """
    Comprehensive stock analysis with AI predictions
    
    - **symbol**: Stock symbol (e.g., AAPL, RELIANCE.NS)
    - **prediction_days**: Number of days to predict (1-30)
    - **analysis_mode**: 'quick' or 'deep'
    - **user_position**: 'own' or 'dont_own'
    """
    try:
        if not STOCK_PREDICTOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Stock predictor module not available")
        
        # Check cache
        cache_key = f"analysis_{request.symbol}_{request.prediction_days}_{request.analysis_mode}"
        cached_result = get_cached_data(cache_key, ttl_minutes=15)
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Initialize analyzer
        analyzer = TransparentStockAnalyzer()
        
        # Fetch stock data
        data, full_symbol = analyzer.get_global_stock_data(request.symbol, period="1y")
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"Stock symbol '{request.symbol}' not found")
        
        current_price = float(data['Close'].iloc[-1])
        
        # Get fundamentals
        fundamentals = analyzer.get_comprehensive_fundamentals(full_symbol)
        company_name = fundamentals.get('company_name', request.symbol)
        
        # News analysis
        news_data = analyzer.get_real_time_news(request.symbol, company_name)
        
        # Technical analysis
        patterns = analyzer.get_historical_patterns(data, request.symbol)
        df_with_indicators = analyzer.calculate_advanced_indicators(data)
        
        # Model training
        model_results = analyzer.train_prediction_models(df_with_indicators)
        
        # Make prediction
        predictions, confidence, reasoning = analyzer.make_transparent_prediction(
            df_with_indicators, model_results, fundamentals, news_data, 
            patterns, request.prediction_days
        )
        
        # Get world market data
        world_data = analyzer.get_world_market_data()
        
        # Multi-timeframe predictions
        multi_predictions = analyzer.make_multi_timeframe_prediction(
            df_with_indicators, model_results, fundamentals, news_data, patterns, world_data
        )
        
        # Chart analysis
        print("[DEBUG API] Starting chart analysis")
        chart_analysis = analyze_chart_bullish_bearish_signals(df_with_indicators)
        print("[DEBUG API] Chart analysis complete")
        
        # Risk management
        print("[DEBUG API] Starting risk management")
        risk_analysis = calculate_risk_management(current_price, multi_predictions, chart_analysis, fundamentals)
        print("[DEBUG API] Risk management complete")
        
        # Extract confidence value (it's a dict like {'Transparent AI Model': 65.0})
        confidence_value = list(confidence.values())[0] if isinstance(confidence, dict) and confidence else 60.0
        
        # Build response
        print("[DEBUG API] Building response")
        result = {
            "symbol": request.symbol,
            "full_symbol": full_symbol,
            "company_name": company_name,
            "current_price": current_price,
            "analysis_timestamp": format_ist_time(),
            "predictions": convert_numpy_types(multi_predictions),
            "confidence": float(confidence_value),
            "reasoning": convert_numpy_types(reasoning),
            "fundamentals": convert_numpy_types(fundamentals),
            "technical_analysis": {
                "chart_sentiment": chart_analysis.get('overall_sentiment'),
                "confidence_level": chart_analysis.get('confidence_level'),
                "bullish_signals": chart_analysis.get('bullish_signals', []),
                "bearish_signals": chart_analysis.get('bearish_signals', []),
                "key_levels": chart_analysis.get('key_levels', [])
            },
            "risk_management": convert_numpy_types(risk_analysis),
            "news_sentiment": {
                "score": news_data.get('sentiment_score', 0),
                "headlines": news_data.get('headlines', [])[:5]
            },
            "market_data": {
                "daily_change": float(current_price - data['Close'].iloc[-2]) if len(data) > 1 else 0,
                "daily_change_pct": float((current_price / data['Close'].iloc[-2] - 1) * 100) if len(data) > 1 else 0,
                "volume": int(data['Volume'].iloc[-1]),
                "week_high": float(data['High'].tail(5).max()),
                "week_low": float(data['Low'].tail(5).min())
            },
            "cached": False
        }
        
        print("[DEBUG API] Response built, caching...")
        # Cache the result
        set_cached_data(cache_key, result, ttl_minutes=15)
        
        print("[DEBUG API] Returning result")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/api/stock/{symbol}")
async def get_stock_overview(symbol: str):
    """Get quick stock overview"""
    try:
        # Check cache
        cache_key = f"overview_{symbol}"
        cached_result = get_cached_data(cache_key, ttl_minutes=5)
        if cached_result:
            return cached_result
        
        # Fetch data
        stock = yf.Ticker(symbol)
        data = stock.history(period="1mo")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"Stock '{symbol}' not found")
        
        info = stock.info
        current_price = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        
        result = {
            "symbol": symbol,
            "company_name": info.get('longName', symbol),
            "current_price": current_price,
            "previous_close": prev_close,
            "change": current_price - prev_close,
            "change_percent": ((current_price / prev_close - 1) * 100) if prev_close > 0 else 0,
            "volume": int(data['Volume'].iloc[-1]),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "sector": info.get('sector'),
            "industry": info.get('industry'),
            "week_high": float(data['High'].tail(5).max()),
            "week_low": float(data['Low'].tail(5).min()),
            "timestamp": format_ist_time() if STOCK_PREDICTOR_AVAILABLE else datetime.now().isoformat()
        }
        
        set_cached_data(cache_key, result, ttl_minutes=5)
        return convert_numpy_types(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

# ============================================================================
# DAILY RECOMMENDATIONS
# ============================================================================

@app.get("/api/daily-recommendations")
async def get_daily_recommendations(
    max_stocks: int = Query(15, ge=5, le=50, description="Maximum number of recommendations"),
    min_return: float = Query(0.0, description="Minimum predicted return percentage")
):
    """
    Get daily stock recommendations
    
    Scans 100+ Indian stocks and returns top picks
    """
    try:
        if not STOCK_PREDICTOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Stock predictor module not available")
        
        # Check cache
        cache_key = f"daily_recommendations_{max_stocks}_{min_return}"
        cached_result = get_cached_data(cache_key, ttl_minutes=60)
        if cached_result:
            cached_result['cached'] = True
            return cached_result
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Scan stocks
        recommendations = []
        processed = 0
        total = len(TOP_INDIAN_STOCKS)
        
        def process_stock(symbol, company_name):
            return predictor.quick_predict(symbol, company_name)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock = {
                executor.submit(process_stock, symbol, company_name): (symbol, company_name)
                for symbol, company_name in TOP_INDIAN_STOCKS.items()
            }
            
            for future in as_completed(future_to_stock):
                try:
                    result = future.result(timeout=20)
                    if result and result.get('predicted_return', 0) >= min_return:
                        recommendations.append(result)
                except:
                    pass
        
        # Sort by predicted return
        recommendations.sort(key=lambda x: x.get('predicted_return', 0), reverse=True)
        top_recommendations = recommendations[:max_stocks]
        
        result = {
            "timestamp": format_ist_time(),
            "total_scanned": total,
            "recommendations_found": len(recommendations),
            "top_picks": convert_numpy_types(top_recommendations),
            "market_status": "open" if 555 <= (datetime.now(pytz.timezone('Asia/Kolkata')).hour * 60 + datetime.now(pytz.timezone('Asia/Kolkata')).minute) <= 930 else "closed",
            "cached": False
        }
        
        set_cached_data(cache_key, result, ttl_minutes=60)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================

@app.post("/api/technical-analysis")
async def get_technical_analysis(request: ChartAnalysisRequest):
    """
    Get technical analysis for a stock
    
    Returns 20+ technical indicators
    """
    try:
        if not STOCK_PREDICTOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Stock predictor module not available")
        
        cache_key = f"technical_{request.symbol}_{request.period}"
        cached_result = get_cached_data(cache_key, ttl_minutes=10)
        if cached_result:
            return cached_result
        
        analyzer = TransparentStockAnalyzer()
        data, full_symbol = analyzer.get_global_stock_data(request.symbol, period=request.period)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"Stock '{request.symbol}' not found")
        
        df_with_indicators = analyzer.calculate_advanced_indicators(data)
        chart_analysis = analyze_chart_bullish_bearish_signals(df_with_indicators)
        
        # Get latest values
        latest = df_with_indicators.iloc[-1]
        
        result = {
            "symbol": request.symbol,
            "timestamp": format_ist_time(),
            "current_price": float(latest['Close']),
            "indicators": {
                "rsi": float(latest.get('RSI', 0)),
                "macd": float(latest.get('MACD', 0)),
                "macd_signal": float(latest.get('MACD_signal', 0)),
                "bb_upper": float(latest.get('BB_upper', 0)),
                "bb_middle": float(latest.get('BB_middle', 0)),
                "bb_lower": float(latest.get('BB_lower', 0)),
                "sma_20": float(latest.get('SMA_20', 0)),
                "sma_50": float(latest.get('SMA_50', 0)),
                "sma_200": float(latest.get('SMA_200', 0)),
                "ema_12": float(latest.get('EMA_12', 0)),
                "ema_26": float(latest.get('EMA_26', 0)),
                "stoch_k": float(latest.get('Stoch_K', 0)),
                "stoch_d": float(latest.get('Stoch_D', 0)),
                "williams_r": float(latest.get('Williams_R', 0)),
                "atr": float(latest.get('ATR', 0)),
                "adx": float(latest.get('ADX', 0)),
                "cci": float(latest.get('CCI', 0)),
                "obv": float(latest.get('OBV', 0))
            },
            "chart_analysis": {
                "overall_sentiment": chart_analysis.get('overall_sentiment'),
                "confidence_level": chart_analysis.get('confidence_level'),
                "bullish_signals": chart_analysis.get('bullish_signals', []),
                "bearish_signals": chart_analysis.get('bearish_signals', []),
                "trend_analysis": chart_analysis.get('trend_analysis', []),
                "momentum_analysis": chart_analysis.get('momentum_analysis', []),
                "volume_analysis": chart_analysis.get('volume_analysis', [])
            }
        }
        
        set_cached_data(cache_key, result, ttl_minutes=10)
        return convert_numpy_types(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Technical analysis error: {str(e)}")

# ============================================================================
# NEWS & SENTIMENT
# ============================================================================

@app.post("/api/news-sentiment")
async def get_news_sentiment(request: NewsAnalysisRequest):
    """Get news sentiment analysis for a stock"""
    try:
        if not STOCK_PREDICTOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Stock predictor module not available")
        
        cache_key = f"news_{request.symbol}"
        cached_result = get_cached_data(cache_key, ttl_minutes=30)
        if cached_result:
            return cached_result
        
        analyzer = TransparentStockAnalyzer()
        company_name = request.company_name or request.symbol
        news_data = analyzer.get_real_time_news(request.symbol, company_name)
        
        result = {
            "symbol": request.symbol,
            "timestamp": format_ist_time(),
            "sentiment_score": news_data.get('sentiment_score', 0),
            "sentiment_label": "positive" if news_data.get('sentiment_score', 0) > 0.3 
                              else "negative" if news_data.get('sentiment_score', 0) < -0.3 
                              else "neutral",
            "headlines": news_data.get('headlines', [])[:10],
            "total_articles": len(news_data.get('headlines', []))
        }
        
        set_cached_data(cache_key, result, ttl_minutes=30)
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News analysis error: {str(e)}")

# ============================================================================
# GLOBAL MARKETS
# ============================================================================

@app.get("/api/global-markets")
async def get_global_markets():
    """Get global market indices data"""
    try:
        if not STOCK_PREDICTOR_AVAILABLE:
            raise HTTPException(status_code=503, detail="Stock predictor module not available")
        
        cache_key = "global_markets"
        cached_result = get_cached_data(cache_key, ttl_minutes=15)
        if cached_result:
            return cached_result
        
        analyzer = TransparentStockAnalyzer()
        world_data = analyzer.get_world_market_data()
        
        result = {
            "timestamp": format_ist_time(),
            "indices": world_data.get('major_indices', []),
            "market_summary": world_data.get('market_summary', ''),
            "total_indices": len(world_data.get('major_indices', []))
        }
        
        set_cached_data(cache_key, result, ttl_minutes=15)
        return convert_numpy_types(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Global markets error: {str(e)}")

# ============================================================================
# RISK ANALYSIS
# ============================================================================

@app.post("/api/risk-analysis")
async def analyze_risk(request: RiskAnalysisRequest):
    """Calculate risk management metrics"""
    try:
        # Calculate basic risk metrics
        predicted_change_pct = ((request.predicted_price / request.current_price) - 1) * 100
        
        # Stop loss (2% below current)
        stop_loss = request.current_price * 0.98
        
        # Take profit (based on prediction)
        take_profit = request.predicted_price
        
        # Risk-reward ratio
        potential_gain = take_profit - request.current_price
        potential_loss = request.current_price - stop_loss
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Position sizing (2% portfolio risk)
        risk_amount = request.portfolio_value * 0.02
        position_size = int(risk_amount / potential_loss) if potential_loss > 0 else 0
        
        # Risk level
        if abs(predicted_change_pct) < 5:
            risk_level = "low"
        elif abs(predicted_change_pct) < 10:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        result = {
            "current_price": request.current_price,
            "predicted_price": request.predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "confidence": request.confidence,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": risk_reward,
            "position_size": position_size,
            "max_loss_amount": potential_loss * position_size,
            "potential_gain_amount": potential_gain * position_size,
            "risk_level": risk_level,
            "recommended_allocation": 0.02 if risk_level == "low" else 0.015 if risk_level == "medium" else 0.01,
            "warnings": [
                f"High volatility detected" if abs(predicted_change_pct) > 10 else None,
                f"Low confidence prediction" if request.confidence < 60 else None
            ],
            "recommendations": [
                f"Use stop-loss at ₹{stop_loss:.2f}",
                f"Target price: ₹{take_profit:.2f}",
                f"Position size: {position_size} shares"
            ]
        }
        
        # Remove None values
        result['warnings'] = [w for w in result['warnings'] if w]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis error: {str(e)}")

# ============================================================================
# COMPARISON
# ============================================================================

@app.post("/api/compare-stocks")
async def compare_stocks(request: StockComparisonRequest):
    """Compare multiple stocks"""
    try:
        if len(request.symbols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for comparison")
        
        comparisons = []
        
        for symbol in request.symbols:
            try:
                stock = yf.Ticker(symbol)
                data = stock.history(period="1mo")
                
                if not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    prev_close = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                    
                    comparison = {
                        "symbol": symbol,
                        "current_price": current_price,
                        "change_percent": ((current_price / prev_close - 1) * 100) if prev_close > 0 else 0,
                        "volume": int(data['Volume'].iloc[-1]),
                        "avg_volume": int(data['Volume'].mean())
                    }
                    
                    # Add RSI if requested
                    if "rsi" in request.metrics and STOCK_PREDICTOR_AVAILABLE:
                        analyzer = TransparentStockAnalyzer()
                        df_indicators = analyzer.calculate_advanced_indicators(data)
                        comparison["rsi"] = float(df_indicators['RSI'].iloc[-1]) if 'RSI' in df_indicators else None
                    
                    comparisons.append(comparison)
            except:
                continue
        
        return {
            "timestamp": format_ist_time() if STOCK_PREDICTOR_AVAILABLE else datetime.now().isoformat(),
            "symbols": request.symbols,
            "comparisons": convert_numpy_types(comparisons)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

@app.get("/api/cache/status")
async def get_cache_status():
    """Get cache status"""
    return {
        "total_items": len(cache_store),
        "items": list(cache_store.keys()),
        "oldest_item": min(cache_ttl.values()) if cache_ttl else None,
        "newest_item": max(cache_ttl.values()) if cache_ttl else None
    }

@app.delete("/api/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    cache_store.clear()
    cache_ttl.clear()
    return {"message": "Cache cleared successfully", "items_cleared": 0}

# ============================================================================
# INDIAN STOCKS LIST
# ============================================================================

@app.get("/api/indian-stocks")
async def get_indian_stocks():
    """Get list of supported Indian stocks"""
    if STOCK_PREDICTOR_AVAILABLE:
        return {
            "total_stocks": len(TOP_INDIAN_STOCKS),
            "stocks": [
                {"symbol": symbol, "name": name}
                for symbol, name in TOP_INDIAN_STOCKS.items()
            ]
        }
    else:
        raise HTTPException(status_code=503, detail="Stock list not available")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
