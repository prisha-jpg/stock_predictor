import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import requests
import time
import re
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')
import os

# Check and import required packages
missing_packages = []

# Additional imports for new features
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    missing_packages.append("beautifulsoup4")
    BEAUTIFULSOUP_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    missing_packages.append("groq")
    GROQ_AVAILABLE = False

try:
    import yfinance as yf
except ImportError:
    missing_packages.append("yfinance")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    missing_packages.append("plotly")
    PLOTLY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    missing_packages.append("scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    missing_packages.append("ta")
    TA_AVAILABLE = False

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    missing_packages.append("tavily-python")
    TAVILY_AVAILABLE = False

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    missing_packages.append("newsapi-python")
    NEWSAPI_AVAILABLE = False

try:
    import feedparser
    RSS_AVAILABLE = True
except ImportError:
    missing_packages.append("feedparser")
    RSS_AVAILABLE = False

# Show installation instructions if packages are missing
if missing_packages:
    st.error("Missing Required Packages!")
    st.write("Please install the following packages:")
    for package in missing_packages:
        st.code(f"pip install {package}")
    
    st.write("**Complete installation command:**")
    st.code("pip install " + " ".join(missing_packages))
    st.stop()

class TransparentStockAnalyzer:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=8),
            'Linear Regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.analysis_log = []
        self.data_sources = []
        
    def log_analysis(self, step: str, data: str, reasoning: str):
        """Log every analysis step with full transparency"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.analysis_log.append({
            'timestamp': timestamp,
            'step': step,
            'data': data,
            'reasoning': reasoning
        })
        
    def get_real_time_news(self, stock_symbol: str, company_name: str) -> Dict:
        """Get real-time news from multiple sources with full transparency"""
        news_data = {
            'headlines': [],
            'sentiment_score': 0,
            'key_events': [],
            'sources': [],
            'analysis_notes': []
        }
        
        try:
            # Method 1: Tavily Search (Most Comprehensive)
            if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
                self.log_analysis("NEWS_SEARCH", "Using Tavily API", "Tavily provides comprehensive web search with real-time data")
                tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                
                # Search for recent news
                queries = [
                    f"{company_name} latest news earnings results financial performance 2024 2025",
                    f"{stock_symbol} NSE stock price movement analyst rating upgrade downgrade",
                    f"{company_name} quarterly results revenue profit growth expansion plans",
                    f"Indian stock market {stock_symbol} sector outlook government policy impact"
                ]
                
                for query in queries:
                    try:
                        results = tavily.search(query=query, max_results=5, topic="news", days=7)
                        for result in results.get('results', []):
                            news_data['headlines'].append({
                                'title': result.get('title', ''),
                                'content': result.get('content', ''),
                                'url': result.get('url', ''),
                                'published_date': result.get('published_date', ''),
                                'score': result.get('score', 0)
                            })
                            news_data['sources'].append(f"Tavily: {result.get('url', '')}")
                    except Exception as e:
                        self.log_analysis("NEWS_ERROR", f"Tavily query failed: {str(e)}", "Continuing with alternative sources")
            
            # Method 2: RSS Feeds from Financial Sources
            if RSS_AVAILABLE:
                self.log_analysis("RSS_FEEDS", "Fetching RSS feeds", "Using RSS feeds from major financial news sources")
                rss_sources = [
                    f"https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
                    f"https://www.moneycontrol.com/rss/results.xml",
                    f"https://www.business-standard.com/rss/markets-106.rss"
                ]
                
                for rss_url in rss_sources:
                    try:
                        feed = feedparser.parse(rss_url)
                        for entry in feed.entries[:5]:  # Get latest 5 entries
                            if stock_symbol.lower() in entry.title.lower() or company_name.lower() in entry.title.lower():
                                news_data['headlines'].append({
                                    'title': entry.title,
                                    'content': getattr(entry, 'summary', ''),
                                    'url': entry.link,
                                    'published_date': getattr(entry, 'published', ''),
                                    'source': 'RSS'
                                })
                                news_data['sources'].append(f"RSS: {entry.link}")
                    except Exception as e:
                        self.log_analysis("RSS_ERROR", f"RSS feed failed: {str(e)}", "RSS source temporarily unavailable")
            
            # Method 3: Free Financial APIs
            self.log_analysis("API_SEARCH", "Using financial APIs", "Fetching data from free financial APIs")
            
            # Alpha Vantage News (Free tier)
            try:
                alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
                if alpha_vantage_key:
                    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}.NSE&apikey={alpha_vantage_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'feed' in data:
                            for item in data['feed'][:5]:
                                news_data['headlines'].append({
                                    'title': item.get('title', ''),
                                    'content': item.get('summary', ''),
                                    'url': item.get('url', ''),
                                    'published_date': item.get('time_published', ''),
                                    'sentiment': item.get('overall_sentiment_label', 'Neutral')
                                })
                                news_data['sources'].append(f"Alpha Vantage: {item.get('url', '')}")
            except Exception as e:
                self.log_analysis("ALPHA_VANTAGE_ERROR", f"Alpha Vantage failed: {str(e)}", "Alpha Vantage API temporarily unavailable")
            
            # Enhanced sentiment analysis from collected news
            if news_data['headlines']:
                sentiment_scores = []
                positive_keywords = [
                    'growth', 'profit', 'expansion', 'bullish', 'upgrade', 'positive', 'strong', 'beat', 'outperform',
                    'surge', 'rally', 'gains', 'earnings beat', 'revenue growth', 'margin expansion', 'guidance raise',
                    'buy rating', 'target price increase', 'upgrade to buy', 'strong quarter', 'record revenue',
                    'market share gain', 'new product launch', 'strategic acquisition', 'partnership', 'contract win',
                    'government approval', 'regulatory clearance', 'expansion plans', 'capacity addition', 'innovation'
                ]
                negative_keywords = [
                    'loss', 'decline', 'bearish', 'downgrade', 'negative', 'weak', 'miss', 'underperform', 'concern',
                    'crash', 'plunge', 'sell-off', 'earnings miss', 'revenue decline', 'margin compression', 'guidance cut',
                    'sell rating', 'target price cut', 'downgrade to sell', 'weak quarter', 'disappointing results',
                    'market share loss', 'product recall', 'regulatory issues', 'legal problems', 'competition threat',
                    'supply chain issues', 'cost inflation', 'demand weakness', 'economic headwinds', 'recession risk'
                ]
                
                # Market-specific keywords for Indian stocks
                indian_market_keywords = {
                    'positive': ['gst', 'demonetization recovery', 'make in india', 'atmanirbhar', 'infrastructure push', 
                               'digital india', 'startup ecosystem', 'fdi inflow', 'government support', 'policy reforms'],
                    'negative': ['gst impact', 'demonetization', 'policy uncertainty', 'regulatory overhang', 
                               'government intervention', 'tax issues', 'compliance burden', 'bureaucratic delays']
                }
                
                for headline in news_data['headlines']:
                    title_content = (headline.get('title', '') + ' ' + headline.get('content', '')).lower()
                    
                    # Basic sentiment scoring
                    positive_score = sum(1 for word in positive_keywords if word in title_content)
                    negative_score = sum(1 for word in negative_keywords if word in title_content)
                    
                    # Indian market specific scoring
                    indian_positive = sum(1 for word in indian_market_keywords['positive'] if word in title_content)
                    indian_negative = sum(1 for word in indian_market_keywords['negative'] if word in title_content)
                    
                    # Weighted scoring
                    total_score = (positive_score - negative_score) + (indian_positive - indian_negative) * 0.5
                    
                    # Add sentiment from API if available
                    if 'sentiment' in headline:
                        sentiment_label = headline['sentiment'].lower()
                        if 'bullish' in sentiment_label or 'positive' in sentiment_label:
                            total_score += 2
                        elif 'bearish' in sentiment_label or 'negative' in sentiment_label:
                            total_score -= 2
                    
                    sentiment_scores.append(total_score)
                
                news_data['sentiment_score'] = np.mean(sentiment_scores) if sentiment_scores else 0
                
                self.log_analysis("SENTIMENT_ANALYSIS", 
                    f"Analyzed {len(news_data['headlines'])} headlines, sentiment score: {news_data['sentiment_score']:.2f}",
                    "Sentiment calculated using keyword analysis of headlines and content")
            
        except Exception as e:
            self.log_analysis("NEWS_GENERAL_ERROR", f"General news fetch error: {str(e)}", "Using fallback analysis")
            news_data['analysis_notes'].append("Real-time news temporarily unavailable, using technical analysis only")
        
        return news_data
    
    def get_world_market_data(self) -> Dict:
        """Scrape world market data from investing.com"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            url = "https://www.investing.com/indices/major-indices"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                market_data = {
                    'global_sentiment': 'neutral',
                    'major_indices': [],
                    'market_summary': ''
                }
                
                # Extract major indices data
                indices_table = soup.find('table', {'id': 'cr1'})
                if indices_table:
                    rows = indices_table.find_all('tr')[1:6]  # Get first 5 major indices
                    
                    positive_count = 0
                    negative_count = 0
                    
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            name = cols[1].text.strip()
                            change = cols[3].text.strip()
                            
                            if '+' in change:
                                positive_count += 1
                            elif '-' in change:
                                negative_count += 1
                                
                            market_data['major_indices'].append({
                                'name': name,
                                'change': change
                            })
                    
                    # Determine global sentiment
                    if positive_count > negative_count:
                        market_data['global_sentiment'] = 'positive'
                    elif negative_count > positive_count:
                        market_data['global_sentiment'] = 'negative'
                    
                    market_data['market_summary'] = f"Global markets: {positive_count} positive, {negative_count} negative out of 5 major indices"
            
            self.log_analysis("WORLD_MARKET_DATA", f"Global sentiment: {market_data['global_sentiment']}", 
                             "World market data scraped from investing.com")
            return market_data
            
        except Exception as e:
            self.log_analysis("WORLD_MARKET_ERROR", f"Failed to get world market data: {str(e)}", 
                             "Using fallback market analysis")
            return {'global_sentiment': 'neutral', 'major_indices': [], 'market_summary': 'Data unavailable'}
    
    def get_comprehensive_fundamentals(self, symbol: str) -> Dict:
        """Get comprehensive fundamental data with transparency"""
        fundamentals = {}
        
        try:
            self.log_analysis("FUNDAMENTALS", f"Fetching fundamental data for {symbol}", "Using yfinance for comprehensive fundamental analysis")
            
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Financial metrics
            fundamentals.update({
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'ev_revenue': info.get('enterpriseToRevenue', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'gross_margins': info.get('grossMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'profit_margins': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'operating_cash_flow': info.get('operatingCashflow', 0),
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
                'beta': info.get('beta', 1),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'float_shares': info.get('floatShares', 0),
                'held_percent_institutions': info.get('heldPercentInstitutions', 0),
                'held_percent_insiders': info.get('heldPercentInsiders', 0),
                'book_value': info.get('bookValue', 0),
                'price_to_book': info.get('priceToBook', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'target_high_price': info.get('targetHighPrice', 0),
                'target_low_price': info.get('targetLowPrice', 0),
                'target_mean_price': info.get('targetMeanPrice', 0),
                'recommendation_mean': info.get('recommendationMean', 0),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'company_name': info.get('longName', symbol),
                'business_summary': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0),
                'city': info.get('city', ''),
                'country': info.get('country', ''),
                'exchange': info.get('exchange', ''),
                'currency': info.get('currency', 'INR')
            })
            
            # Get financial statements
            try:
                # Income statement
                income_stmt = stock.financials
                if not income_stmt.empty:
                    fundamentals['total_revenue'] = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
                    fundamentals['gross_profit'] = income_stmt.loc['Gross Profit'].iloc[0] if 'Gross Profit' in income_stmt.index else 0
                    fundamentals['net_income'] = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
                
                # Balance sheet
                balance_sheet = stock.balance_sheet
                if not balance_sheet.empty:
                    fundamentals['total_assets'] = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                    fundamentals['total_equity'] = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0] if 'Total Equity Gross Minority Interest' in balance_sheet.index else 0
                
                # Cash flow
                cash_flow = stock.cashflow
                if not cash_flow.empty:
                    fundamentals['free_cash_flow_stmt'] = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0
                
            except Exception as e:
                self.log_analysis("FINANCIAL_STATEMENTS", f"Error getting financial statements: {str(e)}", "Using available info data instead")
            
            self.log_analysis("FUNDAMENTALS_SUCCESS", f"Retrieved {len(fundamentals)} fundamental metrics", "Comprehensive fundamental analysis completed successfully")
            
        except Exception as e:
            self.log_analysis("FUNDAMENTALS_ERROR", f"Error fetching fundamentals: {str(e)}", "Using fallback fundamental analysis")
            fundamentals = {'error': str(e)}
        
        return fundamentals
    
    def get_historical_patterns(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze historical patterns and seasonality"""
        patterns = {}
        
        try:
            self.log_analysis("HISTORICAL_ANALYSIS", f"Analyzing historical patterns for {symbol}", "Identifying seasonal trends, support/resistance levels, and historical performance")
            
            # Monthly seasonality
            data_with_month = data.copy()
            data_with_month['month'] = data_with_month.index.month
            data_with_month['returns'] = data_with_month['Close'].pct_change()
            
            monthly_returns = data_with_month.groupby('month')['returns'].mean()
            patterns['best_months'] = monthly_returns.nlargest(3).index.tolist()
            patterns['worst_months'] = monthly_returns.nsmallest(3).index.tolist()
            patterns['monthly_avg_returns'] = monthly_returns.to_dict()
            
            # Weekly patterns
            data_with_dow = data.copy()
            data_with_dow['day_of_week'] = data_with_dow.index.dayofweek
            data_with_dow['returns'] = data_with_dow['Close'].pct_change()
            
            weekly_returns = data_with_dow.groupby('day_of_week')['returns'].mean()
            patterns['weekly_patterns'] = weekly_returns.to_dict()
            
            # Support and resistance levels
            highs = data['High'].rolling(window=20).max()
            lows = data['Low'].rolling(window=20).min()
            
            # Find significant levels (price levels that were tested multiple times)
            price_levels = []
            current_price = data['Close'].iloc[-1]
            
            for i in range(len(data) - 50, len(data)):
                if i > 0:
                    high_level = data['High'].iloc[i]
                    low_level = data['Low'].iloc[i]
                    
                    # Count how many times these levels were tested
                    high_tests = sum(abs(data['High'][max(0, i-20):i+20] - high_level) < (high_level * 0.02))
                    low_tests = sum(abs(data['Low'][max(0, i-20):i+20] - low_level) < (low_level * 0.02))
                    
                    if high_tests >= 3:
                        price_levels.append({'level': high_level, 'type': 'resistance', 'strength': high_tests})
                    if low_tests >= 3:
                        price_levels.append({'level': low_level, 'type': 'support', 'strength': low_tests})
            
            # Remove duplicates and sort by proximity to current price
            unique_levels = []
            for level in price_levels:
                if not any(abs(level['level'] - existing['level']) < (current_price * 0.01) for existing in unique_levels):
                    unique_levels.append(level)
            
            patterns['key_levels'] = sorted(unique_levels, key=lambda x: abs(x['level'] - current_price))[:5]
            
            # Volatility analysis
            returns = data['Close'].pct_change().dropna()
            patterns['avg_volatility'] = returns.std() * np.sqrt(252)  # Annualized
            patterns['volatility_trend'] = 'increasing' if returns.tail(30).std() > returns.head(30).std() else 'decreasing'
            
            # Trend analysis
            short_ma = data['Close'].rolling(window=20).mean()
            long_ma = data['Close'].rolling(window=50).mean()
            
            patterns['trend_strength'] = (short_ma.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            patterns['trend_direction'] = 'bullish' if patterns['trend_strength'] > 0.05 else 'bearish' if patterns['trend_strength'] < -0.05 else 'sideways'
            
            # Performance analysis
            returns_1m = (data['Close'].iloc[-1] / data['Close'].iloc[-22] - 1) * 100 if len(data) >= 22 else 0
            returns_3m = (data['Close'].iloc[-1] / data['Close'].iloc[-66] - 1) * 100 if len(data) >= 66 else 0
            returns_6m = (data['Close'].iloc[-1] / data['Close'].iloc[-132] - 1) * 100 if len(data) >= 132 else 0
            returns_1y = (data['Close'].iloc[-1] / data['Close'].iloc[-252] - 1) * 100 if len(data) >= 252 else 0
            
            patterns['performance'] = {
                '1_month': returns_1m,
                '3_months': returns_3m,
                '6_months': returns_6m,
                '1_year': returns_1y
            }
            
            self.log_analysis("PATTERNS_ANALYSIS", 
                f"Identified {len(patterns['key_levels'])} key levels, trend: {patterns['trend_direction']}", 
                "Historical pattern analysis provides insights into price behavior and potential future movements")
            
        except Exception as e:
            self.log_analysis("PATTERNS_ERROR", f"Error in historical analysis: {str(e)}", "Using basic pattern analysis")
            patterns = {'error': str(e)}
        
        return patterns
    
    def analyze_chart_patterns_with_llm(self, data: pd.DataFrame, patterns: Dict) -> str:
        """Use Groq LLM to analyze chart patterns"""
        try:
            if not os.getenv("GROQ_API_KEY"):
                return "Chart pattern analysis requires Groq API key"
            
            client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            # Prepare chart data summary
            current_price = data['Close'].iloc[-1]
            price_change_5d = ((current_price / data['Close'].iloc[-5]) - 1) * 100
            volume_trend = "increasing" if data['Volume'].tail(5).mean() > data['Volume'].head(-5).tail(5).mean() else "decreasing"
            
            # Get candlestick patterns for last 10 days
            recent_data = data.tail(10)
            candle_info = []
            
            for i in range(len(recent_data)):
                open_price = recent_data['Open'].iloc[i]
                close_price = recent_data['Close'].iloc[i] 
                high_price = recent_data['High'].iloc[i]
                low_price = recent_data['Low'].iloc[i]
                
                body_size = abs(close_price - open_price)
                total_range = high_price - low_price
                
                if body_size / total_range > 0.7:
                    candle_type = "strong_bullish" if close_price > open_price else "strong_bearish"
                elif body_size / total_range < 0.3:
                    candle_type = "doji_indecision"
                else:
                    candle_type = "bullish" if close_price > open_price else "bearish"
                    
                candle_info.append(f"Day {i+1}: {candle_type}")
            
            prompt = f"""
            Analyze this stock chart pattern and provide a clear prediction:

            Stock Data:
            - Current Price: {current_price:.2f}
            - 5-day change: {price_change_5d:+.1f}%
            - Volume trend: {volume_trend}
            - Trend direction: {patterns.get('trend_direction', 'unknown')}
            
            Recent Candlestick Patterns (last 10 days):
            {chr(10).join(candle_info)}
            
            Key Levels:
            {str(patterns.get('key_levels', [])[:3])}
            
            Based on these chart patterns, provide:
            1. What the candlesticks are indicating (2-3 sentences)
            2. Short-term direction prediction (bullish/bearish/neutral)
            3. Key levels to watch
            4. Risk assessment
            
            Keep response under 150 words and be specific about what the charts suggest.
            """
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=200,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content
            self.log_analysis("CHART_LLM_ANALYSIS", "Generated LLM chart analysis", 
                             "AI-powered interpretation of candlestick patterns and chart signals")
            
            return analysis
            
        except Exception as e:
            self.log_analysis("CHART_LLM_ERROR", f"LLM analysis failed: {str(e)}", "Using basic pattern analysis")
            return f"Chart shows {patterns.get('trend_direction', 'sideways')} trend with current support/resistance levels"
    
    def calculate_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        try:
            self.log_analysis("TECHNICAL_INDICATORS", "Calculating advanced technical indicators", "Using comprehensive technical analysis for accurate signals")
            
            if TA_AVAILABLE:
                # Trend Indicators
                df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
                df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
                df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
                df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
                df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
                
                df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
                df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
                df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
                
                # MACD
                df['MACD'] = ta.trend.macd(df['Close'])
                df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
                df['MACD_histogram'] = ta.trend.macd_diff(df['Close'])
                
                # Momentum Indicators
                df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                df['RSI_30'] = ta.momentum.rsi(df['Close'], window=30)
                df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
                df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
                df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
                df['ROC'] = ta.momentum.roc(df['Close'], window=12)
                
                # Volatility Indicators
                bollinger = ta.volatility.BollingerBands(df['Close'])
                df['BB_upper'] = bollinger.bollinger_hband()
                df['BB_middle'] = bollinger.bollinger_mavg()
                df['BB_lower'] = bollinger.bollinger_lband()
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
                df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
                
                df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
                df['Keltner_upper'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
                df['Keltner_lower'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
                
                # Volume Indicators
                df['Volume_SMA'] = ta.volume.VolumeSMAIndicator(df['Close'], df['Volume']).volume_sma()
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
                
                # Additional Custom Indicators
                df['Price_Position'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
                df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
                df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
                
                # Trend Strength
                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
                df['Plus_DI'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'])
                df['Minus_DI'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'])
                
                # Advanced Momentum Indicators
                df['CCI'] = ta.momentum.cci(df['High'], df['Low'], df['Close'])
                df['TSI'] = ta.momentum.tsi(df['Close'])
                df['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
                df['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])
                
                # Advanced Volatility Indicators
                df['DC_upper'] = ta.volatility.donchian_channel_hband(df['High'], df['Low'], df['Close'])
                df['DC_lower'] = ta.volatility.donchian_channel_lband(df['High'], df['Low'], df['Close'])
                df['DC_middle'] = (df['DC_upper'] + df['DC_lower']) / 2
                df['DC_position'] = (df['Close'] - df['DC_lower']) / (df['DC_upper'] - df['DC_lower'])
                
                # Advanced Volume Indicators
                df['EOM'] = ta.volume.ease_of_movement(df['High'], df['Low'], df['Volume'])
                df['FI'] = ta.volume.force_index(df['Close'], df['Volume'])
                df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
                df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
                
                # Support and Resistance Levels
                df['Resistance_Level'] = df['High'].rolling(window=20).max()
                df['Support_Level'] = df['Low'].rolling(window=20).min()
                df['Price_vs_Resistance'] = (df['Close'] - df['Resistance_Level']) / df['Resistance_Level']
                df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Support_Level']
                
                # Fibonacci Levels (simplified)
                recent_high = df['High'].rolling(window=50).max().iloc[-1]
                recent_low = df['Low'].rolling(window=50).min().iloc[-1]
                fib_range = recent_high - recent_low
                df['Fib_23.6'] = recent_low + (fib_range * 0.236)
                df['Fib_38.2'] = recent_low + (fib_range * 0.382)
                df['Fib_50.0'] = recent_low + (fib_range * 0.5)
                df['Fib_61.8'] = recent_low + (fib_range * 0.618)
                df['Fib_78.6'] = recent_low + (fib_range * 0.786)
                
                # Market Structure Analysis
                df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
                df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
                df['Higher_Low'] = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
                df['Lower_High'] = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
                
                # Gap Analysis
                df['Gap_Up'] = df['Open'] > df['High'].shift(1)
                df['Gap_Down'] = df['Open'] < df['Low'].shift(1)
                df['Gap_Size'] = np.where(df['Gap_Up'], (df['Open'] - df['High'].shift(1)) / df['High'].shift(1),
                                        np.where(df['Gap_Down'], (df['Low'].shift(1) - df['Open']) / df['Low'].shift(1), 0))
                
                # Price Action Patterns
                df['Doji'] = abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1
                df['Hammer'] = (df['Close'] > df['Open']) & ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open'])) & ((df['High'] - df['Close']) < 0.3 * (df['Close'] - df['Open']))
                df['Shooting_Star'] = (df['Open'] > df['Close']) & ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & ((df['Close'] - df['Low']) < 0.3 * (df['Open'] - df['Close']))
                
                # Trend Continuity
                df['Trend_Continuity'] = np.where(df['Close'] > df['SMA_20'], 1, -1)
                df['Momentum_Continuity'] = np.where(df['RSI'] > 50, 1, -1)
                df['Volume_Confirmation'] = np.where(df['Volume'] > df['Volume_SMA'], 1, -1)
                
            else:
                # Fallback calculations
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['RSI'] = self.calculate_rsi_manual(df['Close'])
                df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            # Custom composite indicators
            df['Momentum_Score'] = (
                (df['RSI'] - 50) / 50 * 0.2 +
                np.where(df['MACD'] > df['MACD_signal'], 1, -1) * 0.2 +
                np.where(df['Close'] > df['SMA_20'], 1, -1) * 0.2 +
                (df['CCI'] / 100) * 0.1 +
                (df['TSI'] / 100) * 0.1 +
                (df['UO'] - 50) / 50 * 0.1 +
                (df['AO'] / 100) * 0.1
            ) if 'MACD' in df.columns else 0
            
            df['Volatility_Score'] = (df['ATR'] / df['Close'] * 100) if 'ATR' in df.columns else df['Volatility']
            
            # Advanced Composite Scores
            df['Trend_Strength_Score'] = (
                (df['ADX'] / 100) * 0.4 +
                np.where(df['Plus_DI'] > df['Minus_DI'], 1, -1) * 0.3 +
                np.where(df['Close'] > df['SMA_50'], 1, -1) * 0.3
            ) if 'ADX' in df.columns else 0
            
            df['Volume_Strength_Score'] = (
                (df['CMF'] + 1) / 2 * 0.3 +
                np.where(df['Volume'] > df['Volume_SMA'], 1, -1) * 0.3 +
                (df['MFI'] - 50) / 50 * 0.2 +
                np.where(df['OBV'].diff() > 0, 1, -1) * 0.2
            ) if 'CMF' in df.columns else 0
            
            df['Support_Resistance_Score'] = (
                np.where(df['Price_vs_Resistance'] > -0.02, -0.5, 0) +
                np.where(df['Price_vs_Support'] < 0.02, 0.5, 0) +
                np.where(df['Close'] > df['Fib_50.0'], 0.3, -0.3) +
                np.where(df['Close'] > df['Fib_61.8'], 0.2, 0)
            ) if 'Price_vs_Resistance' in df.columns else 0
            
            df['Pattern_Score'] = (
                np.where(df['Hammer'], 0.3, 0) +
                np.where(df['Shooting_Star'], -0.3, 0) +
                np.where(df['Doji'], 0, 0) +
                np.where(df['Gap_Up'], 0.2, 0) +
                np.where(df['Gap_Down'], -0.2, 0) +
                np.where(df['Higher_High'], 0.2, 0) +
                np.where(df['Lower_Low'], -0.2, 0)
            )
            
            # Overall Technical Score
            df['Overall_Technical_Score'] = (
                df['Momentum_Score'] * 0.3 +
                df['Trend_Strength_Score'] * 0.25 +
                df['Volume_Strength_Score'] * 0.2 +
                df['Support_Resistance_Score'] * 0.15 +
                df['Pattern_Score'] * 0.1
            )
            
            self.log_analysis("INDICATORS_SUCCESS", f"Calculated {len([col for col in df.columns if col not in data.columns])} technical indicators", "Advanced technical analysis completed successfully")
            
        except Exception as e:
            self.log_analysis("INDICATORS_ERROR", f"Error calculating indicators: {str(e)}", "Using basic technical indicators")
            
        return df
    
    def calculate_rsi_manual(self, prices, window=14):
        """Manual RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_global_stock_data(self, symbol: str, period: str = "2y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Fetch global stock data with automatic market detection"""
        try:
            self.log_analysis("DATA_FETCH", f"Fetching data for {symbol}", f"Retrieving {period} of historical data for comprehensive analysis")
            
            # List of common stock exchanges and their suffixes
            exchange_suffixes = {
                'US': '',  # No suffix for US stocks
                'India_NSE': '.NS',
                'India_BSE': '.BO', 
                'UK': '.L',
                'Canada': '.TO',
                'Germany': '.DE',
                'Australia': '.AX',
                'Japan': '.T',
                'Hong Kong': '.HK'
            }
            
            # Try different exchanges
            for exchange, suffix in exchange_suffixes.items():
                try_symbol = f"{symbol}{suffix}" if suffix else symbol
                
                stock = yf.Ticker(try_symbol)
                data = stock.history(period=period)
                
                if not data.empty:
                    self.log_analysis("DATA_SUCCESS", f"Retrieved {len(data)} days of data from {exchange}", 
                                    f"Found data for {try_symbol}")
                    return data, try_symbol
            
            self.log_analysis("DATA_ERROR", f"No data found for {symbol} in any exchange", "Symbol may be incorrect or delisted")
            return None, None
            
        except Exception as e:
            self.log_analysis("DATA_FETCH_ERROR", f"Error fetching data: {str(e)}", "Data retrieval failed")
            return None, None
    
    def train_prediction_models(self, df: pd.DataFrame) -> Dict:
        """Train multiple models with enhanced ensemble methods and feature engineering"""
        try:
            self.log_analysis("MODEL_TRAINING", "Training enhanced prediction models", "Using ensemble methods with advanced feature engineering")
            
            # Enhanced feature set with all available indicators
            feature_columns = [
                # Basic Technical Indicators
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26', 'EMA_50',
                'RSI', 'RSI_30', 'MACD', 'MACD_signal', 'MACD_histogram',
                'BB_position', 'BB_width', 'ATR', 'Volatility',
                'Volume_Ratio', 'Volume_SMA',
                
                # Advanced Momentum Indicators
                'CCI', 'TSI', 'UO', 'AO', 'Stoch_K', 'Stoch_D', 'Williams_R', 'ROC',
                
                # Volume Indicators
                'OBV', 'CMF', 'VWAP', 'EOM', 'FI', 'MFI', 'VPT',
                
                # Trend Indicators
                'ADX', 'Plus_DI', 'Minus_DI',
                
                # Support/Resistance
                'Price_vs_Resistance', 'Price_vs_Support',
                'DC_position', 'DC_middle',
                
                # Fibonacci Levels
                'Fib_23.6', 'Fib_38.2', 'Fib_50.0', 'Fib_61.8', 'Fib_78.6',
                
                # Pattern Recognition
                'Doji', 'Hammer', 'Shooting_Star', 'Gap_Up', 'Gap_Down', 'Gap_Size',
                'Higher_High', 'Lower_Low', 'Higher_Low', 'Lower_High',
                
                # Composite Scores
                'Momentum_Score', 'Trend_Strength_Score', 'Volume_Strength_Score',
                'Support_Resistance_Score', 'Pattern_Score', 'Overall_Technical_Score',
                
                # Price Action
                'Price_Position', 'Trend_Continuity', 'Momentum_Continuity', 'Volume_Confirmation'
            ]
            
            # Filter available features
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 5:
                self.log_analysis("MODEL_WARNING", f"Only {len(available_features)} features available", "Limited features may reduce prediction accuracy")
                return {}
            
            # Create multiple target variables for different timeframes
            df['target_1d'] = df['Close'].shift(-1) / df['Close'] - 1
            df['target_3d'] = df['Close'].shift(-3) / df['Close'] - 1
            df['target_7d'] = df['Close'].shift(-7) / df['Close'] - 1
            
            # Feature Engineering
            df_enhanced = df.copy()
            
            # Add lagged features
            for col in ['Close', 'Volume', 'RSI', 'MACD']:
                if col in df_enhanced.columns:
                    for lag in [1, 2, 3, 5]:
                        df_enhanced[f'{col}_lag_{lag}'] = df_enhanced[col].shift(lag)
            
            # Add rolling statistics
            for col in ['Close', 'Volume']:
                if col in df_enhanced.columns:
                    for window in [5, 10, 20]:
                        df_enhanced[f'{col}_mean_{window}'] = df_enhanced[col].rolling(window).mean()
                        df_enhanced[f'{col}_std_{window}'] = df_enhanced[col].rolling(window).std()
                        df_enhanced[f'{col}_min_{window}'] = df_enhanced[col].rolling(window).min()
                        df_enhanced[f'{col}_max_{window}'] = df_enhanced[col].rolling(window).max()
            
            # Add price ratios
            if 'SMA_20' in df_enhanced.columns and 'SMA_50' in df_enhanced.columns:
                df_enhanced['SMA_ratio_20_50'] = df_enhanced['SMA_20'] / df_enhanced['SMA_50']
            if 'High' in df_enhanced.columns and 'Low' in df_enhanced.columns:
                df_enhanced['Price_range'] = (df_enhanced['High'] - df_enhanced['Low']) / df_enhanced['Close']
            
            # Update feature list with new engineered features
            engineered_features = [col for col in df_enhanced.columns if col not in df.columns and col not in ['target_1d', 'target_3d', 'target_7d']]
            all_features = available_features + engineered_features
            all_features = [col for col in all_features if col in df_enhanced.columns]
            
            # Prepare data for different timeframes
            model_results = {}
            
            for target_col in ['target_1d', 'target_3d', 'target_7d']:
                if target_col not in df_enhanced.columns:
                    continue
                    
                # Prepare data
                X = df_enhanced[all_features].dropna()
                y = df_enhanced[target_col].dropna()
                
                # Align X and y
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y.iloc[:min_len]
                
                if len(X) < 100:  # Increased minimum samples
                    self.log_analysis("MODEL_WARNING", f"Insufficient data for {target_col}: {len(X)} samples", "Need at least 100 samples for reliable training")
                    continue
                
                # Time series split (more appropriate for financial data)
                train_size = int(len(X) * 0.7)
                val_size = int(len(X) * 0.15)
                
                X_train = X.iloc[:train_size]
                X_val = X.iloc[train_size:train_size + val_size]
                X_test = X.iloc[train_size + val_size:]
                
                y_train = y.iloc[:train_size]
                y_val = y.iloc[train_size:train_size + val_size]
                y_test = y.iloc[train_size + val_size:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
                
                # Enhanced model ensemble
                models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
                    'Ridge Regression': Ridge(alpha=1.0),
                    'Lasso Regression': Lasso(alpha=0.1),
                    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
                }
                
                # Add more models if available
                try:
                    from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
                    from sklearn.svm import SVR
                    from sklearn.neural_network import MLPRegressor
                    
                    models.update({
                        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
                        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
                        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
                        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                    })
                except ImportError:
                    pass
                
                timeframe_results = {}
                
                # Train and evaluate models
                for name, model in models.items():
                    try:
                        # Use scaled data for linear models, original for tree-based
                        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'SVR', 'Neural Network']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate comprehensive metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        
                        # Calculate directional accuracy
                        y_test_direction = np.sign(y_test)
                        y_pred_direction = np.sign(y_pred)
                        directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
                        
                        # Calculate R-squared
                        r2 = model.score(X_test_scaled if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'SVR', 'Neural Network'] else X_test, y_test)
                        
                        # Calculate Sharpe ratio for predictions
                        prediction_returns = y_pred
                        actual_returns = y_test
                        sharpe_ratio = np.mean(prediction_returns) / np.std(prediction_returns) if np.std(prediction_returns) > 0 else 0
                        
                        timeframe_results[name] = {
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'directional_accuracy': directional_accuracy,
                            'sharpe_ratio': sharpe_ratio,
                            'model': model,
                            'scaler': scaler if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'SVR', 'Neural Network'] else None
                        }
                        
                        self.log_analysis("MODEL_TRAINED", f"{name} ({target_col}): DA {directional_accuracy:.1f}%, R² {r2:.3f}", 
                            f"Trained on {len(X_train)} samples, tested on {len(X_test)} samples")
                        
                    except Exception as e:
                        self.log_analysis("MODEL_ERROR", f"{name} ({target_col}) training failed: {str(e)}", "Skipping this model")
                        continue
                
                # Create ensemble prediction
                if timeframe_results:
                    # Weight models by their performance
                    weights = {}
                    for name, results in timeframe_results.items():
                        # Composite score based on directional accuracy and R²
                        composite_score = (results['directional_accuracy'] / 100) * 0.6 + (max(0, results['r2']) * 0.4)
                        weights[name] = composite_score
                    
                    # Normalize weights
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        weights = {k: v/total_weight for k, v in weights.items()}
                    
                    model_results[target_col] = {
                        'individual_models': timeframe_results,
                        'ensemble_weights': weights,
                        'best_model': max(timeframe_results.items(), key=lambda x: x[1]['directional_accuracy'])[0]
                    }
            
            return model_results
            
        except Exception as e:
            self.log_analysis("TRAINING_ERROR", f"Enhanced model training failed: {str(e)}", "Unable to train prediction models")
            return {}
    
    def make_transparent_prediction(self, df: pd.DataFrame, model_results: Dict, fundamentals: Dict, 
                                  news_data: Dict, patterns: Dict, days: int = 7) -> Tuple[List[float], Dict, str]:
        """Make prediction with full transparency and reasoning"""
        try:
            current_price = df['Close'].iloc[-1]
            predictions = []
            confidence_scores = {}
            reasoning_steps = []
            
            self.log_analysis("PREDICTION_START", f"Making {days}-day prediction from price ₹{current_price:.2f}", 
                "Starting comprehensive prediction analysis")
            
            # Technical Analysis Score (40% weight)
            technical_score = 0
            technical_reasons = []
            
            # Moving Average Analysis
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma_20 = df['SMA_20'].iloc[-1]
                sma_50 = df['SMA_50'].iloc[-1]
                
                if current_price > sma_20:
                    technical_score += 0.15
                    technical_reasons.append(f"✅ Price (₹{current_price:.2f}) above 20-day SMA (₹{sma_20:.2f})")
                else:
                    technical_score -= 0.15
                    technical_reasons.append(f"❌ Price (₹{current_price:.2f}) below 20-day SMA (₹{sma_20:.2f})")
                
                if current_price > sma_50:
                    technical_score += 0.15
                    technical_reasons.append(f"✅ Price above 50-day SMA (₹{sma_50:.2f}) - Long-term bullish")
                else:
                    technical_score -= 0.15
                    technical_reasons.append(f"❌ Price below 50-day SMA (₹{sma_50:.2f}) - Long-term bearish")
            
            # RSI Analysis
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[-1]
                if rsi < 30:
                    technical_score += 0.2
                    technical_reasons.append(f"✅ RSI {rsi:.1f} - Oversold, potential bounce")
                elif rsi > 70:
                    technical_score -= 0.2
                    technical_reasons.append(f"❌ RSI {rsi:.1f} - Overbought, correction risk")
                elif 40 <= rsi <= 60:
                    technical_score += 0.05
                    technical_reasons.append(f"⚪ RSI {rsi:.1f} - Neutral momentum")
                else:
                    technical_reasons.append(f"⚪ RSI {rsi:.1f} - Moderate momentum")
            
            # MACD Analysis
            if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                macd = df['MACD'].iloc[-1]
                macd_signal = df['MACD_signal'].iloc[-1]
                
                if macd > macd_signal:
                    technical_score += 0.1
                    technical_reasons.append(f"✅ MACD ({macd:.4f}) above signal ({macd_signal:.4f}) - Bullish momentum")
                else:
                    technical_score -= 0.1
                    technical_reasons.append(f"❌ MACD ({macd:.4f}) below signal ({macd_signal:.4f}) - Bearish momentum")
            
            # Volume Analysis
            if 'Volume_Ratio' in df.columns:
                vol_ratio = df['Volume_Ratio'].iloc[-1]
                if vol_ratio > 1.5:
                    technical_score += 0.1
                    technical_reasons.append(f"✅ High volume activity ({vol_ratio:.1f}x average) - Strong interest")
                elif vol_ratio < 0.7:
                    technical_score -= 0.05
                    technical_reasons.append(f"⚠️ Low volume ({vol_ratio:.1f}x average) - Weak participation")
            
            self.log_analysis("TECHNICAL_ANALYSIS", f"Technical Score: {technical_score:.3f}", 
                f"Based on {len(technical_reasons)} technical factors")
            
            # Fundamental Analysis Score (35% weight)
            fundamental_score = 0
            fundamental_reasons = []
            
            if fundamentals and 'error' not in fundamentals:
                # Valuation Analysis
                pe_ratio = fundamentals.get('pe_ratio', 0)
                if pe_ratio > 0:
                    if pe_ratio < 15:
                        fundamental_score += 0.15
                        fundamental_reasons.append(f"✅ P/E ratio {pe_ratio:.1f} - Undervalued")
                    elif pe_ratio > 30:
                        fundamental_score -= 0.15
                        fundamental_reasons.append(f"❌ P/E ratio {pe_ratio:.1f} - Overvalued")
                    elif 15 <= pe_ratio <= 25:
                        fundamental_score += 0.05
                        fundamental_reasons.append(f"⚪ P/E ratio {pe_ratio:.1f} - Fairly valued")
                
                # Growth Analysis
                revenue_growth = fundamentals.get('revenue_growth', 0)
                if revenue_growth > 0.15:
                    fundamental_score += 0.15
                    fundamental_reasons.append(f"✅ Revenue growth {revenue_growth*100:.1f}% - Strong growth")
                elif revenue_growth < -0.05:
                    fundamental_score -= 0.15
                    fundamental_reasons.append(f"❌ Revenue declining {revenue_growth*100:.1f}% - Concerning")
                elif revenue_growth > 0.05:
                    fundamental_score += 0.08
                    fundamental_reasons.append(f"⚪ Revenue growth {revenue_growth*100:.1f}% - Moderate growth")
                
                # Profitability
                roe = fundamentals.get('roe', 0)
                if roe > 0.15:
                    fundamental_score += 0.1
                    fundamental_reasons.append(f"✅ ROE {roe*100:.1f}% - Excellent returns")
                elif roe > 0.1:
                    fundamental_score += 0.05
                    fundamental_reasons.append(f"⚪ ROE {roe*100:.1f}% - Good returns")
                elif roe < 0.05:
                    fundamental_score -= 0.1
                    fundamental_reasons.append(f"❌ ROE {roe*100:.1f}% - Poor returns")
                
                # Debt Analysis
                debt_equity = fundamentals.get('debt_to_equity', 0)
                if debt_equity < 0.3:
                    fundamental_score += 0.05
                    fundamental_reasons.append(f"✅ Low debt-to-equity {debt_equity:.2f} - Strong balance sheet")
                elif debt_equity > 1.0:
                    fundamental_score -= 0.1
                    fundamental_reasons.append(f"❌ High debt-to-equity {debt_equity:.2f} - Financial risk")
                
                # Analyst Targets
                target_mean = fundamentals.get('target_mean_price', 0)
                if target_mean > 0:
                    target_upside = (target_mean - current_price) / current_price
                    if target_upside > 0.1:
                        fundamental_score += 0.1
                        fundamental_reasons.append(f"✅ Analyst target ₹{target_mean:.2f} implies {target_upside*100:.1f}% upside")
                    elif target_upside < -0.1:
                        fundamental_score -= 0.1
                        fundamental_reasons.append(f"❌ Analyst target ₹{target_mean:.2f} implies {target_upside*100:.1f}% downside")
            
            self.log_analysis("FUNDAMENTAL_ANALYSIS", f"Fundamental Score: {fundamental_score:.3f}", 
                f"Based on {len(fundamental_reasons)} fundamental factors")
            
            # News Sentiment Analysis (15% weight)
            news_score = 0
            news_reasons = []
            
            if news_data and 'sentiment_score' in news_data:
                sentiment = news_data['sentiment_score']
                if sentiment > 0.5:
                    news_score += 0.1
                    news_reasons.append(f"✅ Positive news sentiment ({sentiment:.2f}) - Market optimism")
                elif sentiment < -0.5:
                    news_score -= 0.1
                    news_reasons.append(f"❌ Negative news sentiment ({sentiment:.2f}) - Market pessimism")
                else:
                    news_reasons.append(f"⚪ Neutral news sentiment ({sentiment:.2f})")
                
                # Count of recent headlines
                if 'headlines' in news_data and len(news_data['headlines']) > 0:
                    news_reasons.append(f"📰 Found {len(news_data['headlines'])} recent news articles")
            
            self.log_analysis("NEWS_ANALYSIS", f"News Score: {news_score:.3f}", 
                f"Based on sentiment analysis of recent news")
            
            # Historical Pattern Analysis (10% weight)
            pattern_score = 0
            pattern_reasons = []
            
            if patterns and 'error' not in patterns:
                # Trend analysis
                trend_direction = patterns.get('trend_direction', 'sideways')
                trend_strength = patterns.get('trend_strength', 0)
                
                if trend_direction == 'bullish' and abs(trend_strength) > 0.05:
                    pattern_score += 0.05
                    pattern_reasons.append(f"✅ Strong bullish trend (strength: {trend_strength:.3f})")
                elif trend_direction == 'bearish' and abs(trend_strength) > 0.05:
                    pattern_score -= 0.05
                    pattern_reasons.append(f"❌ Strong bearish trend (strength: {trend_strength:.3f})")
                
                # Seasonal patterns
                current_month = datetime.now().month
                best_months = patterns.get('best_months', [])
                worst_months = patterns.get('worst_months', [])
                
                if current_month in best_months:
                    pattern_score += 0.03
                    pattern_reasons.append(f"✅ Current month historically bullish")
                elif current_month in worst_months:
                    pattern_score -= 0.03
                    pattern_reasons.append(f"❌ Current month historically bearish")
                
                # Key levels
                key_levels = patterns.get('key_levels', [])
                if key_levels:
                    nearest_support = None
                    nearest_resistance = None
                    
                    for level in key_levels:
                        if level['type'] == 'support' and level['level'] < current_price:
                            if not nearest_support or level['level'] > nearest_support['level']:
                                nearest_support = level
                        elif level['type'] == 'resistance' and level['level'] > current_price:
                            if not nearest_resistance or level['level'] < nearest_resistance['level']:
                                nearest_resistance = level
                    
                    if nearest_support:
                        support_distance = (current_price - nearest_support['level']) / current_price
                        pattern_reasons.append(f"📊 Key support at ₹{nearest_support['level']:.2f} ({support_distance*100:.1f}% below)")
                    
                    if nearest_resistance:
                        resistance_distance = (nearest_resistance['level'] - current_price) / current_price
                        pattern_reasons.append(f"📊 Key resistance at ₹{nearest_resistance['level']:.2f} ({resistance_distance*100:.1f}% above)")
                        if resistance_distance < 0.05:  # Less than 5% away
                            pattern_score -= 0.02
                            pattern_reasons.append(f"⚠️ Near resistance level - potential reversal")
            
            self.log_analysis("PATTERN_ANALYSIS", f"Pattern Score: {pattern_score:.3f}", 
                f"Based on historical patterns and key levels")
            
            # Model Predictions (bonus factor)
            model_score = 0
            model_reasons = []
            
            if model_results:
                best_model = max(model_results.items(), key=lambda x: x[1]['directional_accuracy'])
                model_name = best_model[0]
                model_accuracy = best_model[1]['directional_accuracy']
                
                # Make prediction with best model
                try:
                    feature_columns = [col for col in ['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_position',
                                                     'Volume_Ratio', 'Volatility', 'Momentum_Score'] if col in df.columns]
                    
                    if len(feature_columns) >= 3:
                        latest_features = df[feature_columns].iloc[-1:].fillna(0)
                        
                        if model_name == 'Linear Regression':
                            latest_features_scaled = self.scaler.transform(latest_features)
                            model_pred = best_model[1]['model'].predict(latest_features_scaled)[0]
                        else:
                            model_pred = best_model[1]['model'].predict(latest_features)[0]
                        
                        if model_pred > 0.02:  # More than 2% predicted gain
                            model_score += 0.05
                            model_reasons.append(f"✅ {model_name} predicts {model_pred*100:.1f}% gain (accuracy: {model_accuracy:.1f}%)")
                        elif model_pred < -0.02:  # More than 2% predicted loss
                            model_score -= 0.05
                            model_reasons.append(f"❌ {model_name} predicts {model_pred*100:.1f}% loss (accuracy: {model_accuracy:.1f}%)")
                        else:
                            model_reasons.append(f"⚪ {model_name} predicts minimal change {model_pred*100:.1f}% (accuracy: {model_accuracy:.1f}%)")
                
                except Exception as e:
                    model_reasons.append(f"⚠️ Model prediction unavailable: {str(e)}")
            
            # Combine all scores with weights
            total_score = (
                technical_score * 0.40 +      # 40% weight
                fundamental_score * 0.35 +    # 35% weight  
                news_score * 0.15 +           # 15% weight
                pattern_score * 0.10 +        # 10% weight
                model_score                   # Bonus factor
            )
            
            self.log_analysis("SCORE_COMBINATION", 
                f"Total Score: {total_score:.3f} (Tech: {technical_score:.3f}, Fund: {fundamental_score:.3f}, News: {news_score:.3f}, Pattern: {pattern_score:.3f})",
                "Combined weighted analysis of all factors")
            
            # Calculate realistic prediction
            # More conservative approach - limit extreme predictions
            base_change_pct = total_score * 0.8  # Reduce impact
            base_change_pct = np.clip(base_change_pct, -0.20, 0.20)  # Cap at ±20%
            
            # Add volatility consideration
            volatility = df['Close'].pct_change().tail(30).std() if len(df) >= 30 else 0.02
            volatility_factor = min(volatility * 5, 0.05)  # Cap volatility impact at 5%
            
            # Generate daily predictions with realistic progression
            daily_predictions = []
            cumulative_change = 0
            
            for day in range(1, days + 1):
                # Progressive change with some randomness
                daily_change = (base_change_pct / days) + np.random.normal(0, volatility_factor / 3)
                cumulative_change += daily_change
                
                # Add mean reversion for longer periods
                if day > 3:
                    mean_reversion = -cumulative_change * 0.1  # 10% mean reversion
                    cumulative_change += mean_reversion
                
                predicted_price = current_price * (1 + cumulative_change)
                daily_predictions.append(predicted_price)
            
            # Calculate confidence based on multiple factors
            base_confidence = 65  # Conservative base
            
            # Adjust confidence based on data quality
            if len(technical_reasons) >= 4:
                base_confidence += 10
            if len(fundamental_reasons) >= 3:
                base_confidence += 10
            if news_data and len(news_data.get('headlines', [])) >= 3:
                base_confidence += 5
            if model_results:
                best_accuracy = max([m['directional_accuracy'] for m in model_results.values()])
                base_confidence += (best_accuracy - 50) * 0.3  # Add based on model performance
            
            # Reduce confidence for extreme predictions
            if abs(base_change_pct) > 0.15:
                base_confidence -= 15
            elif abs(base_change_pct) > 0.10:
                base_confidence -= 10
            
            final_confidence = np.clip(base_confidence, 50, 90)
            
            # Compile reasoning
            all_reasoning = {
                'technical_factors': technical_reasons,
                'fundamental_factors': fundamental_reasons,
                'news_factors': news_reasons,
                'pattern_factors': pattern_reasons,
                'model_factors': model_reasons,
                'final_score': total_score,
                'predicted_change_pct': base_change_pct * 100,
                'confidence': final_confidence
            }
            
            self.log_analysis("PREDICTION_COMPLETE", 
                f"Final prediction: ₹{daily_predictions[-1]:.2f} ({base_change_pct*100:+.1f}%) with {final_confidence:.0f}% confidence",
                f"Based on comprehensive analysis of {len(technical_reasons + fundamental_reasons + news_reasons + pattern_reasons + model_reasons)} factors")
            
            return daily_predictions, {'Transparent AI Model': final_confidence}, all_reasoning
            
        except Exception as e:
            self.log_analysis("PREDICTION_ERROR", f"Prediction failed: {str(e)}", "Using fallback prediction method")
            # Fallback simple prediction
            fallback_predictions = [current_price * (1 + np.random.normal(0, 0.02)) for _ in range(days)]
            return fallback_predictions, {'Fallback Model': 60}, {'error': str(e)}
    
    def generate_position_based_recommendations(self, predicted_change_pct: float, final_confidence: float, 
                                              user_position: str, current_price: float) -> Dict:
        """Generate recommendations based on user's current position"""
        
        recommendations = {
            'primary_action': '',
            'action_color': '',
            'specific_actions': [],
            'reasoning': ''
        }
        
        owns_stock = user_position == "Own Stock"
        
        # Strong upward movement expected
        if predicted_change_pct > 5 and final_confidence > 75:
            if owns_stock:
                recommendations['primary_action'] = "HOLD/ADD MORE"
                recommendations['action_color'] = "green"
                recommendations['specific_actions'] = [
                    "Continue holding your position - strong upward momentum",
                    f"Consider adding more shares if you have available capital",
                    f"Set trailing stop loss at 8-10% below current price",
                    "Take partial profits if position becomes too large in portfolio"
                ]
            else:
                recommendations['primary_action'] = "BUY"
                recommendations['action_color'] = "green"
                recommendations['specific_actions'] = [
                    f"Initiate position around current price {current_price:.2f}",
                    "Strong buy signal identified by AI analysis",
                    "Consider position sizing of 2-5% of portfolio",
                    f"Set stop loss at {current_price * 0.92:.2f} (-8%)"
                ]
        
        # Moderate upward movement
        elif predicted_change_pct > 2 and final_confidence > 70:
            if owns_stock:
                recommendations['primary_action'] = "HOLD"
                recommendations['action_color'] = "green" 
                recommendations['specific_actions'] = [
                    "Hold your current position",
                    "Good opportunity to average up if you want more exposure",
                    "Monitor for stronger signals before adding significantly",
                    "Set stop loss below recent support levels"
                ]
            else:
                recommendations['primary_action'] = "BUY (Small Position)"
                recommendations['action_color'] = "green"
                recommendations['specific_actions'] = [
                    "Consider small initial position (1-2% of portfolio)",
                    "Good entry point but not urgent",
                    "Can wait for better entry if price dips",
                    "Build position gradually on any weakness"
                ]
        
        # Sideways/neutral movement
        elif abs(predicted_change_pct) <= 2:
            if owns_stock:
                recommendations['primary_action'] = "HOLD/MONITOR"
                recommendations['action_color'] = "orange"
                recommendations['specific_actions'] = [
                    "Hold current position - no urgent action needed",
                    "Monitor for clearer directional signals",
                    "Good time to review other opportunities",
                    "Consider trimming if you need cash for better opportunities"
                ]
            else:
                recommendations['primary_action'] = "WAIT"
                recommendations['action_color'] = "orange"
                recommendations['specific_actions'] = [
                    "No compelling reason to buy at current levels",
                    "Wait for clearer directional signals",
                    "Focus on stocks with stronger trends",
                    "Add to watchlist for future opportunities"
                ]
        
        # Downward movement expected
        elif predicted_change_pct < -2 and final_confidence > 70:
            if owns_stock:
                recommendations['primary_action'] = "CONSIDER SELLING"
                recommendations['action_color'] = "red"
                recommendations['specific_actions'] = [
                    "Consider reducing or exiting position",
                    "Especially if holding at a profit",
                    "Set tight stop losses if you decide to hold",
                    "Look for better opportunities in stronger stocks"
                ]
            else:
                recommendations['primary_action'] = "AVOID/WAIT"
                recommendations['action_color'] = "red"
                recommendations['specific_actions'] = [
                    "Avoid initiating new positions",
                    "Wait for price to stabilize or show strength",
                    "Consider for future buying after correction",
                    "Focus capital on stocks with better outlook"
                ]
        
        # Uncertain signals
        else:
            if owns_stock:
                recommendations['primary_action'] = "HOLD WITH CAUTION"
                recommendations['action_color'] = "orange"
                recommendations['specific_actions'] = [
                    "Mixed signals - maintain current position cautiously",
                    "Set protective stops to limit downside",
                    "Avoid adding to position until clarity emerges",
                    "Consider reducing position size if nervous"
                ]
            else:
                recommendations['primary_action'] = "WAIT FOR CLARITY"
                recommendations['action_color'] = "orange"
                recommendations['specific_actions'] = [
                    "Too much uncertainty for new positions",
                    "Wait for clearer technical or fundamental signals",
                    "Keep on watchlist for future opportunities",
                    "Focus on higher-conviction ideas"
                ]
        
        return recommendations
    
    def make_multi_timeframe_prediction(self, df: pd.DataFrame, model_results: Dict, fundamentals: Dict, 
                                      news_data: Dict, patterns: Dict, world_data: Dict) -> Dict:
        """Generate predictions for 1, 7, and 30 days with reasoning"""
        
        try:
            current_price = df['Close'].iloc[-1]
            current_date = datetime.now()
            
            # Get base prediction logic (same as before but return scores)
            scores = self.calculate_prediction_scores(df, fundamentals, news_data, patterns, world_data)
            
            # Adjust predictions for different timeframes
            timeframe_predictions = {}
            
            for days in [1, 7, 30]:
                # Base change percentage 
                base_change = scores['total_score'] * 0.8
                
                # Timeframe adjustments
                if days == 1:
                    # More weight on technical and news
                    adjusted_change = base_change * 0.5  # Conservative for 1 day
                    confidence_modifier = 1.1  # Higher confidence for short term
                elif days == 7:
                    # Balanced approach
                    adjusted_change = base_change * 0.8
                    confidence_modifier = 1.0
                else:  # 30 days
                    # More weight on fundamentals
                    adjusted_change = base_change * 1.2
                    confidence_modifier = 0.9  # Lower confidence for longer term
                
                # Cap the changes realistically
                adjusted_change = np.clip(adjusted_change, -0.25, 0.25)  # ±25% max
                
                predicted_price = current_price * (1 + adjusted_change)
                confidence = np.clip(scores['base_confidence'] * confidence_modifier, 50, 90)
                
                # Calculate target date
                target_date = current_date + timedelta(days=days)
                
                timeframe_predictions[f"{days}_day"] = {
                    'target_date': target_date.strftime("%d/%m/%y"),
                    'predicted_price': predicted_price,
                    'change_amount': predicted_price - current_price,
                    'change_percent': ((predicted_price / current_price) - 1) * 100,
                    'confidence': confidence,
                    'key_factors': scores['key_factors'][:3]  # Top 3 factors
                }
            
            return timeframe_predictions
            
        except Exception as e:
            # Fallback predictions
            fallback = {}
            for days in [1, 7, 30]:
                target_date = (current_date + timedelta(days=days)).strftime("%d/%m/%y")
                fallback[f"{days}_day"] = {
                    'target_date': target_date,
                    'predicted_price': current_price * (1 + np.random.normal(0, 0.02)),
                    'change_amount': 0,
                    'change_percent': 0,
                    'confidence': 60,
                    'key_factors': ['Technical Analysis', 'Market Sentiment', 'Historical Patterns']
                }
            return fallback
    
    def calculate_prediction_scores(self, df: pd.DataFrame, fundamentals: Dict, news_data: Dict, 
                                   patterns: Dict, world_data: Dict) -> Dict:
        """Calculate prediction scores for multi-timeframe analysis"""
        try:
            current_price = df['Close'].iloc[-1]
            
            # Technical Analysis Score (40% weight) - Enhanced
            technical_score = 0
            technical_reasons = []
            
            # Use Overall Technical Score if available
            if 'Overall_Technical_Score' in df.columns:
                overall_score = df['Overall_Technical_Score'].iloc[-1]
                technical_score = overall_score * 0.4
                
                # Detailed analysis based on individual components
                if 'Momentum_Score' in df.columns:
                    momentum = df['Momentum_Score'].iloc[-1]
                    if momentum > 0.2:
                        technical_reasons.append(f"Strong momentum (Score: {momentum:.2f})")
                    elif momentum < -0.2:
                        technical_reasons.append(f"Weak momentum (Score: {momentum:.2f})")
                    else:
                        technical_reasons.append(f"Neutral momentum (Score: {momentum:.2f})")
                
                if 'Trend_Strength_Score' in df.columns:
                    trend_strength = df['Trend_Strength_Score'].iloc[-1]
                    if trend_strength > 0.3:
                        technical_reasons.append(f"Strong uptrend (ADX: {df['ADX'].iloc[-1]:.1f})")
                    elif trend_strength < -0.3:
                        technical_reasons.append(f"Strong downtrend (ADX: {df['ADX'].iloc[-1]:.1f})")
                    else:
                        technical_reasons.append(f"Weak trend (ADX: {df['ADX'].iloc[-1]:.1f})")
                
                if 'Volume_Strength_Score' in df.columns:
                    volume_strength = df['Volume_Strength_Score'].iloc[-1]
                    if volume_strength > 0.2:
                        technical_reasons.append(f"Strong volume confirmation")
                    elif volume_strength < -0.2:
                        technical_reasons.append(f"Weak volume confirmation")
                
                if 'Support_Resistance_Score' in df.columns:
                    sr_score = df['Support_Resistance_Score'].iloc[-1]
                    if sr_score > 0.2:
                        technical_reasons.append(f"Near support level - potential bounce")
                    elif sr_score < -0.2:
                        technical_reasons.append(f"Near resistance level - potential pullback")
                
                if 'Pattern_Score' in df.columns:
                    pattern_score = df['Pattern_Score'].iloc[-1]
                    if pattern_score > 0.1:
                        technical_reasons.append(f"Bullish pattern detected")
                    elif pattern_score < -0.1:
                        technical_reasons.append(f"Bearish pattern detected")
            else:
                # Fallback to basic analysis
                # Moving Average Analysis
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    sma_20 = df['SMA_20'].iloc[-1]
                    sma_50 = df['SMA_50'].iloc[-1]
                    
                    if current_price > sma_20:
                        technical_score += 0.15
                        technical_reasons.append(f"Price above 20-day SMA")
                    else:
                        technical_score -= 0.15
                        technical_reasons.append(f"Price below 20-day SMA")
                    
                    if current_price > sma_50:
                        technical_score += 0.15
                        technical_reasons.append(f"Price above 50-day SMA")
                    else:
                        technical_score -= 0.15
                        technical_reasons.append(f"Price below 50-day SMA")
                
                # RSI Analysis
                if 'RSI' in df.columns:
                    rsi = df['RSI'].iloc[-1]
                    if rsi < 30:
                        technical_score += 0.2
                        technical_reasons.append(f"RSI oversold")
                    elif rsi > 70:
                        technical_score -= 0.2
                        technical_reasons.append(f"RSI overbought")
                    elif 40 <= rsi <= 60:
                        technical_score += 0.05
                        technical_reasons.append(f"RSI neutral")
                
                # MACD Analysis
                if 'MACD' in df.columns and 'MACD_signal' in df.columns:
                    macd = df['MACD'].iloc[-1]
                    macd_signal = df['MACD_signal'].iloc[-1]
                    
                    if macd > macd_signal:
                        technical_score += 0.1
                        technical_reasons.append(f"MACD bullish")
                    else:
                        technical_score -= 0.1
                        technical_reasons.append(f"MACD bearish")
            
            # Volume Analysis
            if 'Volume_Ratio' in df.columns:
                vol_ratio = df['Volume_Ratio'].iloc[-1]
                if vol_ratio > 1.5:
                    technical_score += 0.1
                    technical_reasons.append(f"High volume")
                elif vol_ratio < 0.7:
                    technical_score -= 0.05
                    technical_reasons.append(f"Low volume")
            
            # Fundamental Analysis Score (35% weight)
            fundamental_score = 0
            fundamental_reasons = []
            
            if fundamentals and 'error' not in fundamentals:
                # Valuation Analysis
                pe_ratio = fundamentals.get('pe_ratio', 0)
                if pe_ratio > 0:
                    if pe_ratio < 15:
                        fundamental_score += 0.15
                        fundamental_reasons.append(f"Undervalued P/E")
                    elif pe_ratio > 30:
                        fundamental_score -= 0.15
                        fundamental_reasons.append(f"Overvalued P/E")
                    elif 15 <= pe_ratio <= 25:
                        fundamental_score += 0.05
                        fundamental_reasons.append(f"Fair P/E")
                
                # Growth Analysis
                revenue_growth = fundamentals.get('revenue_growth', 0)
                if revenue_growth > 0.15:
                    fundamental_score += 0.15
                    fundamental_reasons.append(f"Strong growth")
                elif revenue_growth < -0.05:
                    fundamental_score -= 0.15
                    fundamental_reasons.append(f"Declining revenue")
                elif revenue_growth > 0.05:
                    fundamental_score += 0.08
                    fundamental_reasons.append(f"Moderate growth")
                
                # Profitability
                roe = fundamentals.get('roe', 0)
                if roe > 0.15:
                    fundamental_score += 0.1
                    fundamental_reasons.append(f"Excellent ROE")
                elif roe > 0.1:
                    fundamental_score += 0.05
                    fundamental_reasons.append(f"Good ROE")
                elif roe < 0.05:
                    fundamental_score -= 0.1
                    fundamental_reasons.append(f"Poor ROE")
            
            # News Sentiment Analysis (15% weight)
            news_score = 0
            news_reasons = []
            
            if news_data and 'sentiment_score' in news_data:
                sentiment = news_data['sentiment_score']
                if sentiment > 0.5:
                    news_score += 0.1
                    news_reasons.append(f"Positive news")
                elif sentiment < -0.5:
                    news_score -= 0.1
                    news_reasons.append(f"Negative news")
                else:
                    news_reasons.append(f"Neutral news")
            
            # Historical Pattern Analysis (10% weight)
            pattern_score = 0
            pattern_reasons = []
            
            if patterns and 'error' not in patterns:
                # Trend analysis
                trend_direction = patterns.get('trend_direction', 'sideways')
                trend_strength = patterns.get('trend_strength', 0)
                
                if trend_direction == 'bullish' and abs(trend_strength) > 0.05:
                    pattern_score += 0.05
                    pattern_reasons.append(f"Bullish trend")
                elif trend_direction == 'bearish' and abs(trend_strength) > 0.05:
                    pattern_score -= 0.05
                    pattern_reasons.append(f"Bearish trend")
            
            # World Market Analysis (bonus factor)
            world_score = 0
            world_reasons = []
            
            if world_data and 'global_sentiment' in world_data:
                sentiment = world_data['global_sentiment']
                if sentiment == 'positive':
                    world_score += 0.05
                    world_reasons.append(f"Global markets positive")
                elif sentiment == 'negative':
                    world_score -= 0.05
                    world_reasons.append(f"Global markets negative")
            
            # Combine all scores with weights
            total_score = (
                technical_score * 0.40 +      # 40% weight
                fundamental_score * 0.35 +    # 35% weight  
                news_score * 0.15 +           # 15% weight
                pattern_score * 0.10 +        # 10% weight
                world_score                   # Bonus factor
            )
            
            # Calculate confidence
            base_confidence = 65
            if len(technical_reasons) >= 4:
                base_confidence += 10
            if len(fundamental_reasons) >= 3:
                base_confidence += 10
            if news_data and len(news_data.get('headlines', [])) >= 3:
                base_confidence += 5
            
            final_confidence = np.clip(base_confidence, 50, 90)
            
            # Compile key factors
            key_factors = technical_reasons + fundamental_reasons + news_reasons + pattern_reasons + world_reasons
            
            return {
                'total_score': total_score,
                'base_confidence': final_confidence,
                'key_factors': key_factors,
                'technical_score': technical_score,
                'fundamental_score': fundamental_score,
                'news_score': news_score,
                'pattern_score': pattern_score,
                'world_score': world_score
            }
            
        except Exception as e:
            return {
                'total_score': 0,
                'base_confidence': 60,
                'key_factors': ['Analysis error'],
                'technical_score': 0,
                'fundamental_score': 0,
                'news_score': 0,
                'pattern_score': 0,
                'world_score': 0
            }

def create_comprehensive_charts(data: pd.DataFrame, fundamentals: Dict, patterns: Dict, title: str):
    """Create detailed transparent charts with improved spacing"""
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=(
            'Price Action & Key Levels', 'Volume Analysis',
            'RSI & MACD', 'Moving Averages & Trends', 
            'Bollinger Bands & Volatility', 'Fundamental Metrics',
            'Historical Performance', 'Prediction Confidence'
        ),
        vertical_spacing=0.08,  # Increased from 0.03 to 0.08
        horizontal_spacing=0.12,  # Increased from 0.05 to 0.12
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"colspan": 2}, None]],
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
    )
    
    # Price action with candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        showlegend=False
    ), row=1, col=1)
    
    # Add key levels if available
    if patterns and 'key_levels' in patterns:
        for level in patterns['key_levels'][:3]:  # Show top 3 levels
            color = 'red' if level['type'] == 'resistance' else 'green'
            fig.add_hline(y=level['level'], line_dash="dash", line_color=color, 
                         row=1, col=1, annotation_text=f"{level['type'].title()}: ₹{level['level']:.1f}")
    
    # Volume
    fig.add_trace(go.Bar(
        x=data.index, y=data['Volume'],
        name='Volume', opacity=0.6
    ), row=1, col=2)
    
    # Volume moving average
    if 'Volume_SMA' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Volume_SMA'],
            name='Volume MA', line=dict(color='orange')
        ), row=1, col=2)
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            name='MACD', line=dict(color='blue')
        ), row=2, col=2)
        if 'MACD_signal' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data['MACD_signal'],
                name='MACD Signal', line=dict(color='red')
            ), row=2, col=2)
    
    # Moving averages
    ma_colors = ['orange', 'red', 'blue']
    ma_columns = ['SMA_20', 'SMA_50', 'SMA_200']
    for i, ma_col in enumerate(ma_columns):
        if ma_col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index, y=data[ma_col],
                name=ma_col, line=dict(color=ma_colors[i], width=2)
            ), row=3, col=1)
    
    # Price line for comparison
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        name='Close Price', line=dict(color='black', width=1)
    ), row=3, col=1)
    
    # Trend direction indicator
    if patterns and 'trend_direction' in patterns:
        trend_color = 'green' if patterns['trend_direction'] == 'bullish' else 'red' if patterns['trend_direction'] == 'bearish' else 'gray'
        fig.add_annotation(
            text=f"Trend: {patterns['trend_direction'].upper()}", 
            xref="x4", yref="y4",
            x=data.index[-1], y=data['Close'].iloc[-1],
            showarrow=True, arrowcolor=trend_color,
            row=3, col=2
        )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_upper'],
            name='BB Upper', line=dict(color='red', dash='dash')
        ), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_middle'],
            name='BB Middle', line=dict(color='blue')
        ), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_lower'],
            name='BB Lower', line=dict(color='green', dash='dash')
        ), row=3, col=2)
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            name='Price', line=dict(color='black', width=2)
        ), row=3, col=2)
    
    # Volatility
    if 'Volatility' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index, y=data['Volatility'] * 100,
            name='Volatility %', line=dict(color='orange')
        ), row=4, col=1)
    
    # Performance comparison
    if patterns and 'performance' in patterns:
        performance_data = patterns['performance']
        periods = list(performance_data.keys())
        returns = list(performance_data.values())
        
        colors = ['green' if r > 0 else 'red' for r in returns]
        fig.add_trace(go.Bar(
            x=periods, y=returns,
            name='Returns %', marker_color=colors
        ), row=4, col=2)
    
    # Overall summary in bottom panel
    summary_text = f"""
    <b>Analysis Summary:</b><br>
    Current Price: ₹{data['Close'].iloc[-1]:.2f}<br>
    """
    
    if patterns:
        summary_text += f"Trend: {patterns.get('trend_direction', 'Unknown').title()}<br>"
        summary_text += f"Volatility: {patterns.get('avg_volatility', 0)*100:.1f}% (annualized)<br>"
    
    if fundamentals and 'pe_ratio' in fundamentals:
        summary_text += f"P/E Ratio: {fundamentals['pe_ratio']:.1f}<br>"
        summary_text += f"Sector: {fundamentals.get('sector', 'Unknown')}<br>"
    
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=0.02, y=0.05,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1
    )
    
    fig.update_layout(
        title=title,
        height=1400,  # Increased from 1200 to 1400 for better spacing
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def analyze_chart_bullish_bearish_signals(data: pd.DataFrame) -> Dict:
    """Comprehensive chart analysis explaining bullish/bearish signals"""
    analysis = {
        'overall_sentiment': 'neutral',
        'bullish_signals': [],
        'bearish_signals': [],
        'key_levels': [],
        'pattern_analysis': [],
        'momentum_analysis': [],
        'volume_analysis': [],
        'trend_analysis': [],
        'risk_factors': [],
        'confidence_level': 0
    }
    
    try:
        current_price = data['Close'].iloc[-1]
        recent_data = data.tail(20)  # Last 20 days for recent analysis
        
        # 1. Trend Analysis
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            if current_price > sma_20 > sma_50:
                analysis['bullish_signals'].append(f"Strong uptrend: Price (₹{current_price:.1f}) > 20-SMA (₹{sma_20:.1f}) > 50-SMA (₹{sma_50:.1f})")
                analysis['trend_analysis'].append("All moving averages aligned upward - strong bullish trend")
            elif current_price < sma_20 < sma_50:
                analysis['bearish_signals'].append(f"Strong downtrend: Price (₹{current_price:.1f}) < 20-SMA (₹{sma_20:.1f}) < 50-SMA (₹{sma_50:.1f})")
                analysis['trend_analysis'].append("All moving averages aligned downward - strong bearish trend")
            elif current_price > sma_20:
                analysis['bullish_signals'].append(f"Short-term bullish: Price above 20-SMA")
                analysis['trend_analysis'].append("Short-term uptrend but mixed longer-term signals")
            else:
                analysis['bearish_signals'].append(f"Short-term bearish: Price below 20-SMA")
                analysis['trend_analysis'].append("Short-term downtrend but mixed longer-term signals")
        
        # 2. Momentum Analysis
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi < 30:
                analysis['bullish_signals'].append(f"RSI oversold ({rsi:.1f}) - potential bounce opportunity")
                analysis['momentum_analysis'].append("Oversold conditions suggest potential reversal")
            elif rsi > 70:
                analysis['bearish_signals'].append(f"RSI overbought ({rsi:.1f}) - correction risk")
                analysis['momentum_analysis'].append("Overbought conditions suggest potential pullback")
            elif 40 <= rsi <= 60:
                analysis['bullish_signals'].append(f"RSI neutral ({rsi:.1f}) - healthy momentum")
                analysis['momentum_analysis'].append("Neutral momentum - no extreme conditions")
            else:
                analysis['momentum_analysis'].append(f"RSI at {rsi:.1f} - moderate momentum")
        
        # 3. MACD Analysis
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_signal'].iloc[-1]
            macd_hist = macd - macd_signal
            
            if macd > macd_signal and macd_hist > 0:
                analysis['bullish_signals'].append(f"MACD bullish crossover - momentum building")
                analysis['momentum_analysis'].append("MACD above signal line with increasing histogram")
            elif macd < macd_signal and macd_hist < 0:
                analysis['bearish_signals'].append(f"MACD bearish crossover - momentum weakening")
                analysis['momentum_analysis'].append("MACD below signal line with decreasing histogram")
            else:
                analysis['momentum_analysis'].append("MACD showing mixed signals")
        
        # 4. Volume Analysis
        if 'Volume_Ratio' in data.columns:
            vol_ratio = data['Volume_Ratio'].iloc[-1]
            if vol_ratio > 1.5:
                analysis['bullish_signals'].append(f"High volume ({vol_ratio:.1f}x average) - strong interest")
                analysis['volume_analysis'].append("Above-average volume confirms price movement")
            elif vol_ratio < 0.7:
                analysis['bearish_signals'].append(f"Low volume ({vol_ratio:.1f}x average) - weak conviction")
                analysis['volume_analysis'].append("Below-average volume suggests weak conviction")
            else:
                analysis['volume_analysis'].append("Normal volume levels")
        
        # 5. Support and Resistance Analysis
        if 'Resistance_Level' in data.columns and 'Support_Level' in data.columns:
            resistance = data['Resistance_Level'].iloc[-1]
            support = data['Support_Level'].iloc[-1]
            
            resistance_distance = (resistance - current_price) / current_price
            support_distance = (current_price - support) / current_price
            
            if resistance_distance < 0.02:  # Within 2% of resistance
                analysis['bearish_signals'].append(f"Near resistance at ₹{resistance:.1f} - potential pullback")
                analysis['key_levels'].append(f"Resistance: ₹{resistance:.1f} (within 2%)")
            elif support_distance < 0.02:  # Within 2% of support
                analysis['bullish_signals'].append(f"Near support at ₹{support:.1f} - potential bounce")
                analysis['key_levels'].append(f"Support: ₹{support:.1f} (within 2%)")
            else:
                analysis['key_levels'].append(f"Support: ₹{support:.1f}, Resistance: ₹{resistance:.1f}")
        
        # 6. Bollinger Bands Analysis
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            bb_position = data['BB_position'].iloc[-1] if 'BB_position' in data.columns else 0.5
            
            if current_price > bb_upper:
                analysis['bearish_signals'].append(f"Price above upper Bollinger Band - overbought")
                analysis['pattern_analysis'].append("Price extended above normal range - mean reversion likely")
            elif current_price < bb_lower:
                analysis['bullish_signals'].append(f"Price below lower Bollinger Band - oversold")
                analysis['pattern_analysis'].append("Price extended below normal range - bounce likely")
            elif bb_position > 0.8:
                analysis['bearish_signals'].append(f"Near upper Bollinger Band - resistance zone")
                analysis['pattern_analysis'].append("Approaching upper band - potential resistance")
            elif bb_position < 0.2:
                analysis['bullish_signals'].append(f"Near lower Bollinger Band - support zone")
                analysis['pattern_analysis'].append("Approaching lower band - potential support")
            else:
                analysis['pattern_analysis'].append("Price within normal Bollinger Band range")
        
        # 7. Pattern Recognition
        if 'Hammer' in data.columns and data['Hammer'].iloc[-1]:
            analysis['bullish_signals'].append("Hammer pattern detected - potential reversal")
            analysis['pattern_analysis'].append("Hammer candlestick suggests selling exhaustion")
        elif 'Shooting_Star' in data.columns and data['Shooting_Star'].iloc[-1]:
            analysis['bearish_signals'].append("Shooting Star pattern detected - potential reversal")
            analysis['pattern_analysis'].append("Shooting Star candlestick suggests buying exhaustion")
        elif 'Doji' in data.columns and data['Doji'].iloc[-1]:
            analysis['pattern_analysis'].append("Doji pattern - indecision in market")
        
        # 8. Gap Analysis
        if 'Gap_Up' in data.columns and data['Gap_Up'].iloc[-1]:
            gap_size = data['Gap_Size'].iloc[-1] if 'Gap_Size' in data.columns else 0
            analysis['bullish_signals'].append(f"Gap up of {gap_size:.1%} - bullish momentum")
            analysis['pattern_analysis'].append("Gap up indicates strong buying interest")
        elif 'Gap_Down' in data.columns and data['Gap_Down'].iloc[-1]:
            gap_size = data['Gap_Size'].iloc[-1] if 'Gap_Size' in data.columns else 0
            analysis['bearish_signals'].append(f"Gap down of {gap_size:.1%} - bearish momentum")
            analysis['pattern_analysis'].append("Gap down indicates strong selling pressure")
        
        # 9. Market Structure Analysis
        if 'Higher_High' in data.columns and data['Higher_High'].iloc[-1]:
            analysis['bullish_signals'].append("Higher High pattern - uptrend continuation")
            analysis['pattern_analysis'].append("Market structure showing higher highs - bullish")
        elif 'Lower_Low' in data.columns and data['Lower_Low'].iloc[-1]:
            analysis['bearish_signals'].append("Lower Low pattern - downtrend continuation")
            analysis['pattern_analysis'].append("Market structure showing lower lows - bearish")
        
        # 10. Overall Sentiment Calculation
        bullish_count = len(analysis['bullish_signals'])
        bearish_count = len(analysis['bearish_signals'])
        
        if bullish_count > bearish_count + 2:
            analysis['overall_sentiment'] = 'bullish'
            analysis['confidence_level'] = min(90, 60 + (bullish_count - bearish_count) * 5)
        elif bearish_count > bullish_count + 2:
            analysis['overall_sentiment'] = 'bearish'
            analysis['confidence_level'] = min(90, 60 + (bearish_count - bullish_count) * 5)
        else:
            analysis['overall_sentiment'] = 'neutral'
            analysis['confidence_level'] = 50
        
        # 11. Risk Factors
        if 'Volatility' in data.columns:
            volatility = data['Volatility'].iloc[-1]
            if volatility > 0.3:  # High volatility
                analysis['risk_factors'].append(f"High volatility ({volatility:.1%}) - increased risk")
            elif volatility < 0.1:  # Low volatility
                analysis['risk_factors'].append(f"Low volatility ({volatility:.1%}) - potential breakout")
        
        if 'ATR' in data.columns:
            atr = data['ATR'].iloc[-1]
            atr_percent = (atr / current_price) * 100
            if atr_percent > 3:
                analysis['risk_factors'].append(f"High ATR ({atr_percent:.1f}%) - large price swings expected")
        
    except Exception as e:
        analysis['risk_factors'].append(f"Analysis error: {str(e)}")
        analysis['confidence_level'] = 30
    
    return analysis

def analyze_individual_charts(data: pd.DataFrame) -> Dict:
    """Analyze individual chart components with detailed pointwise analysis"""
    analysis = {
        'price_action': [],
        'volume_analysis': [],
        'rsi_macd_analysis': [],
        'moving_averages_analysis': [],
        'bollinger_bands_analysis': [],
        'fundamental_metrics_analysis': [],
        'historical_performance_analysis': []
    }
    
    try:
        current_price = data['Close'].iloc[-1]
        recent_data = data.tail(20)
        
        # 1. Price Action & Key Levels Analysis
        if 'High' in data.columns and 'Low' in data.columns:
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            price_range = recent_high - recent_low
            current_position = (current_price - recent_low) / price_range if price_range > 0 else 0.5
            
            analysis['price_action'].extend([
                f"Current price: ₹{current_price:.2f}",
                f"20-day high: ₹{recent_high:.2f}",
                f"20-day low: ₹{recent_low:.2f}",
                f"Price position in range: {current_position:.1%}",
                f"Price range: ₹{price_range:.2f} ({price_range/current_price:.1%} of current price)"
            ])
            
            # Candlestick pattern analysis
            if len(data) >= 3:
                last_3 = data.tail(3)
                if len(last_3) >= 3:
                    # Check for patterns
                    if (last_3['Close'].iloc[-1] > last_3['Open'].iloc[-1] and 
                        last_3['Close'].iloc[-2] > last_3['Open'].iloc[-2] and 
                        last_3['Close'].iloc[-3] > last_3['Open'].iloc[-3]):
                        analysis['price_action'].append("Three consecutive bullish candles - strong upward momentum")
                    elif (last_3['Close'].iloc[-1] < last_3['Open'].iloc[-1] and 
                          last_3['Close'].iloc[-2] < last_3['Open'].iloc[-2] and 
                          last_3['Close'].iloc[-3] < last_3['Open'].iloc[-3]):
                        analysis['price_action'].append("Three consecutive bearish candles - strong downward momentum")
        
        # 2. Volume Analysis
        if 'Volume' in data.columns:
            avg_volume = data['Volume'].tail(20).mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            analysis['volume_analysis'].extend([
                f"Current volume: {current_volume:,.0f}",
                f"20-day average volume: {avg_volume:,.0f}",
                f"Volume ratio: {volume_ratio:.2f}x average",
                f"Volume trend: {'Increasing' if volume_ratio > 1.2 else 'Decreasing' if volume_ratio < 0.8 else 'Normal'}"
            ])
            
            if 'Volume_SMA' in data.columns:
                volume_sma = data['Volume_SMA'].iloc[-1]
                analysis['volume_analysis'].append(f"Volume SMA: {volume_sma:,.0f}")
        
        # 3. RSI & MACD Analysis
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            analysis['rsi_macd_analysis'].extend([
                f"RSI: {rsi:.1f}",
                f"RSI Status: {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}",
                f"RSI Trend: {'Bullish' if rsi > 50 else 'Bearish'}"
            ])
        
        if 'MACD' in data.columns and 'MACD_signal' in data.columns:
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_signal'].iloc[-1]
            macd_hist = macd - macd_signal
            
            analysis['rsi_macd_analysis'].extend([
                f"MACD: {macd:.4f}",
                f"MACD Signal: {macd_signal:.4f}",
                f"MACD Histogram: {macd_hist:.4f}",
                f"MACD Status: {'Bullish' if macd > macd_signal else 'Bearish'}",
                f"MACD Momentum: {'Increasing' if macd_hist > 0 else 'Decreasing'}"
            ])
        
        # 4. Moving Averages Analysis
        ma_analysis = []
        if 'SMA_20' in data.columns:
            sma_20 = data['SMA_20'].iloc[-1]
            ma_analysis.append(f"20-SMA: ₹{sma_20:.2f}")
            ma_analysis.append(f"Price vs 20-SMA: {'Above' if current_price > sma_20 else 'Below'} ({((current_price/sma_20-1)*100):+.1f}%)")
        
        if 'SMA_50' in data.columns:
            sma_50 = data['SMA_50'].iloc[-1]
            ma_analysis.append(f"50-SMA: ₹{sma_50:.2f}")
            ma_analysis.append(f"Price vs 50-SMA: {'Above' if current_price > sma_50 else 'Below'} ({((current_price/sma_50-1)*100):+.1f}%)")
        
        if 'SMA_200' in data.columns:
            sma_200 = data['SMA_200'].iloc[-1]
            ma_analysis.append(f"200-SMA: ₹{sma_200:.2f}")
            ma_analysis.append(f"Price vs 200-SMA: {'Above' if current_price > sma_200 else 'Below'} ({((current_price/sma_200-1)*100):+.1f}%)")
        
        # Check for golden/death cross
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            if sma_20 > sma_50:
                ma_analysis.append("Golden Cross: 20-SMA above 50-SMA - Bullish signal")
            else:
                ma_analysis.append("Death Cross: 20-SMA below 50-SMA - Bearish signal")
        
        analysis['moving_averages_analysis'] = ma_analysis
        
        # 5. Bollinger Bands Analysis
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns and 'BB_middle' in data.columns:
            bb_upper = data['BB_upper'].iloc[-1]
            bb_lower = data['BB_lower'].iloc[-1]
            bb_middle = data['BB_middle'].iloc[-1]
            bb_width = bb_upper - bb_lower
            bb_position = (current_price - bb_lower) / bb_width if bb_width > 0 else 0.5
            
            analysis['bollinger_bands_analysis'].extend([
                f"Upper Band: ₹{bb_upper:.2f}",
                f"Middle Band: ₹{bb_middle:.2f}",
                f"Lower Band: ₹{bb_lower:.2f}",
                f"Band Width: ₹{bb_width:.2f} ({bb_width/current_price:.1%} of price)",
                f"Price Position: {bb_position:.1%} of band width",
                f"Volatility: {'High' if bb_width/current_price > 0.1 else 'Low' if bb_width/current_price < 0.05 else 'Normal'}"
            ])
            
            if current_price > bb_upper:
                analysis['bollinger_bands_analysis'].append("Price above upper band - Overbought condition")
            elif current_price < bb_lower:
                analysis['bollinger_bands_analysis'].append("Price below lower band - Oversold condition")
            else:
                analysis['bollinger_bands_analysis'].append("Price within normal range")
        
        # 6. Historical Performance Analysis
        if len(data) >= 30:
            month_ago = data['Close'].iloc[-30] if len(data) >= 30 else data['Close'].iloc[0]
            month_change = (current_price / month_ago - 1) * 100
            
            if len(data) >= 90:
                quarter_ago = data['Close'].iloc[-90]
                quarter_change = (current_price / quarter_ago - 1) * 100
                analysis['historical_performance_analysis'].append(f"3-month performance: {quarter_change:+.1f}%")
            
            analysis['historical_performance_analysis'].extend([
                f"1-month performance: {month_change:+.1f}%",
                f"Performance trend: {'Improving' if month_change > 0 else 'Declining'}"
            ])
        
        # 7. Volatility Analysis
        if 'Volatility' in data.columns:
            volatility = data['Volatility'].iloc[-1]
            analysis['bollinger_bands_analysis'].extend([
                f"Annualized Volatility: {volatility:.1%}",
                f"Risk Level: {'High' if volatility > 0.3 else 'Low' if volatility < 0.1 else 'Medium'}"
            ])
        
    except Exception as e:
        analysis['price_action'].append(f"Analysis error: {str(e)}")
    
    return analysis

def calculate_risk_management(current_price: float, predictions: Dict, chart_analysis: Dict, fundamentals: Dict) -> Dict:
    """Calculate risk management recommendations"""
    risk_analysis = {
        'position_size': 0,
        'stop_loss': 0,
        'take_profit': 0,
        'risk_reward_ratio': 0,
        'max_loss_amount': 0,
        'recommended_allocation': 0,
        'risk_level': 'medium',
        'warnings': [],
        'recommendations': []
    }
    
    try:
        # Calculate position size based on risk tolerance (2% of portfolio)
        portfolio_value = 100000  # Assume 1 lakh portfolio
        risk_per_trade = portfolio_value * 0.02  # 2% risk per trade
        
        # Calculate stop loss based on technical levels
        if 'Support_Level' in chart_analysis.get('key_levels', []):
            support_level = float(chart_analysis['key_levels'][0].split('₹')[1].split(' ')[0])
            stop_loss = support_level * 0.95  # 5% below support
        else:
            # Use ATR-based stop loss
            atr_multiplier = 2.0
            stop_loss = current_price * (1 - (atr_multiplier * 0.02))  # 2% ATR-based stop
        
        # Calculate take profit based on resistance or target
        if 'Resistance_Level' in chart_analysis.get('key_levels', []):
            resistance_level = float(chart_analysis['key_levels'][0].split('₹')[1].split(' ')[0])
            take_profit = resistance_level * 0.95  # 5% below resistance
        else:
            # Use prediction-based target
            if predictions:
                avg_prediction = np.mean([pred['predicted_price'] for pred in predictions.values()])
                take_profit = avg_prediction * 0.95  # 5% below prediction
            else:
                take_profit = current_price * 1.15  # 15% upside target
        
        # Calculate position size
        risk_per_share = current_price - stop_loss
        if risk_per_share > 0:
            position_size = int(risk_per_trade / risk_per_share)
            position_value = position_size * current_price
            risk_analysis['position_size'] = position_size
            risk_analysis['max_loss_amount'] = position_size * risk_per_share
        else:
            risk_analysis['warnings'].append("Stop loss calculation error - avoid trade")
            return risk_analysis
        
        # Calculate risk-reward ratio
        potential_profit = take_profit - current_price
        potential_loss = current_price - stop_loss
        if potential_loss > 0:
            risk_analysis['risk_reward_ratio'] = potential_profit / potential_loss
        
        # Set stop loss and take profit
        risk_analysis['stop_loss'] = stop_loss
        risk_analysis['take_profit'] = take_profit
        
        # Calculate recommended allocation
        risk_analysis['recommended_allocation'] = min(0.1, position_value / portfolio_value)  # Max 10% allocation
        
        # Determine risk level
        volatility = fundamentals.get('volatility', 0.2) if fundamentals else 0.2
        if volatility > 0.3:
            risk_analysis['risk_level'] = 'high'
            risk_analysis['warnings'].append("High volatility detected - reduce position size")
        elif volatility < 0.1:
            risk_analysis['risk_level'] = 'low'
        else:
            risk_analysis['risk_level'] = 'medium'
        
        # Add recommendations based on analysis
        if chart_analysis['overall_sentiment'] == 'bullish':
            risk_analysis['recommendations'].append("Bullish sentiment - consider long position")
        elif chart_analysis['overall_sentiment'] == 'bearish':
            risk_analysis['recommendations'].append("Bearish sentiment - consider short position or avoid")
        else:
            risk_analysis['recommendations'].append("Neutral sentiment - wait for clearer signals")
        
        if risk_analysis['risk_reward_ratio'] < 1:
            risk_analysis['warnings'].append("Poor risk-reward ratio - consider avoiding trade")
        elif risk_analysis['risk_reward_ratio'] > 2:
            risk_analysis['recommendations'].append("Excellent risk-reward ratio - good trade setup")
        
        # Add fundamental risk factors
        if fundamentals and 'pe_ratio' in fundamentals:
            pe_ratio = fundamentals['pe_ratio']
            if pe_ratio > 30:
                risk_analysis['warnings'].append("High P/E ratio - overvalued risk")
            elif pe_ratio < 10:
                risk_analysis['recommendations'].append("Low P/E ratio - potential value opportunity")
        
    except Exception as e:
        risk_analysis['warnings'].append(f"Risk calculation error: {str(e)}")
    
    return risk_analysis

def display_prediction_table(predictions_dict: Dict, current_price: float):
    """Display predictions in tabular format"""
    
    st.subheader("Multi-Timeframe Price Predictions")
    
    # Create prediction table
    table_data = []
    today = datetime.now().strftime("%d/%m/%y")
    
    # Add current data
    table_data.append({
        'Timeframe': 'Current',
        'Date': today,
        'Predicted Price': f"{current_price:.2f}",
        'Change': "0.00",
        'Change %': "0.0%",
        'Confidence': "-"
    })
    
    # Add predictions
    timeframes = ['1_day', '7_day', '30_day']
    labels = ['1 Day', '7 Days', '30 Days']
    
    for timeframe, label in zip(timeframes, labels):
        pred = predictions_dict[timeframe]
        table_data.append({
            'Timeframe': label,
            'Date': pred['target_date'],
            'Predicted Price': f"{pred['predicted_price']:.2f}",
            'Change': f"{pred['change_amount']:+.2f}",
            'Change %': f"{pred['change_percent']:+.1f}%",
            'Confidence': f"{pred['confidence']:.0f}%"
        })
    
    # Display as dataframe
    df_predictions = pd.DataFrame(table_data)
    
    # Color code the dataframe
    def color_changes(val):
        if '+' in str(val) and val != "0.0%":
            return 'background-color: #d4edda; color: #155724'  # Green
        elif '-' in str(val):
            return 'background-color: #f8d7da; color: #721c24'  # Red
        return ''
    
    # Apply styling to specific columns
    styled_df = df_predictions.style.applymap(
        color_changes, 
        subset=['Change %']
    )
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Add key insights below table
    st.markdown("**Key Prediction Insights:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_1d = predictions_dict['1_day']
        direction = "UP" if pred_1d['change_percent'] > 0 else "DOWN" if pred_1d['change_percent'] < 0 else "FLAT"
        st.metric(
            "1-Day Outlook", 
            direction,
            f"{pred_1d['change_percent']:+.1f}%"
        )
    
    with col2:
        pred_7d = predictions_dict['7_day'] 
        direction = "UP" if pred_7d['change_percent'] > 0 else "DOWN" if pred_7d['change_percent'] < 0 else "FLAT"
        st.metric(
            "7-Day Outlook",
            direction, 
            f"{pred_7d['change_percent']:+.1f}%"
        )
    
    with col3:
        pred_30d = predictions_dict['30_day']
        direction = "UP" if pred_30d['change_percent'] > 0 else "DOWN" if pred_30d['change_percent'] < 0 else "FLAT"
        st.metric(
            "30-Day Outlook",
            direction,
            f"{pred_30d['change_percent']:+.1f}%"
        )

def main():
    st.set_page_config(
        page_title="AI Stock Predictor", 
        layout="wide",
        page_icon="📈"
    )
    
    # Header with clean design
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; background: linear-gradient(90deg, #1f4e79, #2e7d32); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>📈 AI Stock Predictor</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>Smart, Fast, Accurate Stock Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = TransparentStockAnalyzer()
    
    # Main input section - prominently displayed
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        stock_symbol = st.text_input(
            "Stock Symbol",
            placeholder="Enter Stock Symbol (e.g., AAPL, RELIANCE, TSLA)",
            help="Enter any global stock symbol",
            key="stock_input",
            label_visibility="collapsed"
        ).upper()
    
    with col2:
        prediction_days = st.selectbox("Prediction Period", [1, 3, 5, 7, 14, 30], index=3)
    
    with col3:
        user_position = st.selectbox(
            "Current Position", 
            ["Don't Own", "Own Stock"],
            help="Do you currently hold this stock?"
        )
    
    with col4:
        analyze_button = st.button("🔍 Analyze", type="primary", width='stretch')
    
    # User preferences - collapsible
    with st.expander("⚙️ Customize Your Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 What to Show:**")
            show_prediction = st.checkbox("Price Prediction", value=True)
            show_current_overview = st.checkbox("Current Stock Overview", value=True)
            show_company_info = st.checkbox("Company Information", value=True)
            show_charts = st.checkbox("Technical Charts", value=True)
            show_news = st.checkbox("Market News", value=True)
        
        with col2:
            # Fixed to Deep Analysis mode
            analysis_depth = "Deep Analysis"
            st.markdown("**🔍 Analysis Mode:**")
            st.info("🚀 Deep Analysis Mode - Complete AI Analysis with all features enabled")
            
            st.markdown("**📈 Additional Options:**")
            show_reasoning = st.checkbox("Show Analysis Reasoning", value=True)
            show_methodology = st.checkbox("Show Methodology", value=True)
    
    # API Configuration - only if user wants advanced features
    if show_news or analysis_depth == "Deep Analysis":
        with st.expander("🔑 API Keys for Enhanced Analysis (Optional)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                tavily_key = st.text_input("Tavily API Key", type="password", help="For real-time news")
                if tavily_key:
                    os.environ["TAVILY_API_KEY"] = tavily_key
            
            with col2:
                alpha_key = st.text_input("Alpha Vantage API Key", type="password", help="For enhanced data")
                if alpha_key:
                    os.environ["ALPHA_VANTAGE_API_KEY"] = alpha_key
            
            with col3:
                groq_key = st.text_input("Groq API Key", type="password", help="For AI chart analysis")
                if groq_key:
                    os.environ["GROQ_API_KEY"] = groq_key
    
    # Main analysis section
    if analyze_button and stock_symbol:
        
        # Clear previous analysis log
        analyzer.analysis_log = []
        analyzer.data_sources = []
        
        # Progress tracking
        progress_placeholder = st.empty()
        
        with st.spinner("🔄 Analyzing..."):
            
            # Always use deep analysis
            progress_placeholder.info("📊 Fetching stock data...")
            
            data, full_symbol = analyzer.get_global_stock_data(stock_symbol, period="2y")
            
            if data is None or data.empty:
                st.error(f"❌ Could not fetch data for {stock_symbol}. Please verify the symbol.")
                st.stop()
            
            current_price = data['Close'].iloc[-1]
            
            # Step 2: Get fundamentals - always for deep analysis
            progress_placeholder.info("🏢 Analyzing fundamentals...")
            fundamentals = analyzer.get_comprehensive_fundamentals(full_symbol)
            
            company_name = fundamentals.get('company_name', stock_symbol)
            
            # Step 3: News analysis - always for deep analysis
            progress_placeholder.info("📰 Gathering market news...")
            news_data = analyzer.get_real_time_news(stock_symbol, company_name)
            
            # Step 4: Technical analysis - comprehensive
            progress_placeholder.info("📈 Computing technical indicators...")
            patterns = analyzer.get_historical_patterns(data, stock_symbol)
            df_with_indicators = analyzer.calculate_advanced_indicators(data)
            
            # Step 5: Model training - always for deep analysis
            progress_placeholder.info("🤖 Training prediction models...")
            model_results = analyzer.train_prediction_models(df_with_indicators)
            
            # Step 6: Make prediction
            progress_placeholder.info("🔮 Generating prediction...")
            predictions, confidence, reasoning = analyzer.make_transparent_prediction(
                df_with_indicators, model_results, fundamentals, news_data, patterns, prediction_days
            )
            
        # Clear progress indicator
        progress_placeholder.empty()
        
        # SUCCESS MESSAGE - More prominent
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #2e7d32, #1565c0); color: white; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='margin: 0;'>✅ Analysis Complete!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # === PREDICTION RESULTS - MAIN FOCUS ===
        if show_prediction and predictions and reasoning:
            # Get world market data
            world_data = analyzer.get_world_market_data()
            
            # Generate multi-timeframe predictions
            multi_predictions = analyzer.make_multi_timeframe_prediction(
                df_with_indicators, model_results, fundamentals, news_data, patterns, world_data
            )
            
            # Display prediction table
            display_prediction_table(multi_predictions, current_price)
            
            # Enhanced Chart Analysis
            st.markdown("### 📊 Advanced Chart Analysis")
            chart_analysis = analyze_chart_bullish_bearish_signals(df_with_indicators)
            
            # Display overall sentiment
            sentiment_color = "green" if chart_analysis['overall_sentiment'] == 'bullish' else "red" if chart_analysis['overall_sentiment'] == 'bearish' else "orange"
            st.markdown(f"""
            <div style='padding: 1rem; background: {sentiment_color}20; border-left: 4px solid {sentiment_color}; border-radius: 5px; margin: 1rem 0;'>
                <h4 style='margin: 0; color: {sentiment_color};'>Overall Chart Sentiment: {chart_analysis['overall_sentiment'].upper()}</h4>
                <p style='margin: 0.5rem 0 0 0;'>Confidence Level: {chart_analysis['confidence_level']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display bullish and bearish signals
            col1, col2 = st.columns(2)
            
            with col1:
                if chart_analysis['bullish_signals']:
                    st.markdown("#### 🟢 Bullish Signals")
                    for signal in chart_analysis['bullish_signals']:
                        st.write(f"✅ {signal}")
                else:
                    st.markdown("#### 🟢 Bullish Signals")
                    st.write("No strong bullish signals detected")
            
            with col2:
                if chart_analysis['bearish_signals']:
                    st.markdown("#### 🔴 Bearish Signals")
                    for signal in chart_analysis['bearish_signals']:
                        st.write(f"❌ {signal}")
                else:
                    st.markdown("#### 🔴 Bearish Signals")
                    st.write("No strong bearish signals detected")
            
            # Risk Management Analysis
            st.markdown("### ⚠️ Risk Management Analysis")
            risk_analysis = calculate_risk_management(current_price, multi_predictions, chart_analysis, fundamentals)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Position Size", f"{risk_analysis['position_size']} shares")
                st.metric("Risk Level", risk_analysis['risk_level'].upper())
            
            with col2:
                st.metric("Stop Loss", f"₹{risk_analysis['stop_loss']:.2f}")
                st.metric("Take Profit", f"₹{risk_analysis['take_profit']:.2f}")
            
            with col3:
                st.metric("Risk-Reward Ratio", f"{risk_analysis['risk_reward_ratio']:.2f}")
                st.metric("Max Loss", f"₹{risk_analysis['max_loss_amount']:.0f}")
            
            with col4:
                st.metric("Allocation", f"{risk_analysis['recommended_allocation']:.1%}")
                st.metric("Portfolio Risk", "2%")
            
            # Risk warnings and recommendations
            if risk_analysis['warnings']:
                st.markdown("#### ⚠️ Risk Warnings")
                for warning in risk_analysis['warnings']:
                    st.warning(f"⚠️ {warning}")
            
            if risk_analysis['recommendations']:
                st.markdown("#### 💡 Recommendations")
                for rec in risk_analysis['recommendations']:
                    st.info(f"💡 {rec}")
            
            # Show world market influence
            if world_data['market_summary']:
                st.markdown("#### Global Market Influence")
                st.info(f"**World Market Status**: {world_data['market_summary']}")
                
                if world_data['major_indices']:
                    st.write("**Major Global Indices:**")
                    for idx in world_data['major_indices']:
                        st.write(f"• {idx['name']}: {idx['change']}")
        
        # === CURRENT STOCK OVERVIEW ===
        if show_current_overview:
            st.markdown("### 📊 Current Stock Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                daily_change = current_price - data['Close'].iloc[-2] if len(data) > 1 else 0
                daily_change_pct = (daily_change / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
                st.metric("Current Price", f"₹{current_price:.2f}", f"{daily_change_pct:+.1f}%")
            
            with col2:
                week_high = data['High'].tail(5).max()
                week_low = data['Low'].tail(5).min()
                st.metric("5-Day Range", f"₹{week_low:.2f} - ₹{week_high:.2f}")
            
            with col3:
                if fundamentals.get('market_cap', 0) > 0:
                    market_cap_cr = fundamentals['market_cap'] / 10000000
                    st.metric("Market Cap", f"₹{market_cap_cr:,.0f} Cr")
                else:
                    avg_volume = data['Volume'].tail(20).mean()
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            with col4:
                if fundamentals.get('pe_ratio', 0) > 0:
                    st.metric("P/E Ratio", f"{fundamentals['pe_ratio']:.1f}")
                else:
                    month_change = (current_price / data['Close'].iloc[-22] - 1) * 100 if len(data) >= 22 else 0
                    st.metric("1M Change", f"{month_change:+.1f}%")
        
        # === COMPANY INFORMATION ===
        if show_company_info and fundamentals and 'error' not in fundamentals:
            st.markdown("### 🏢 Company Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Company**: {company_name}")
                st.write(f"**Sector**: {fundamentals.get('sector', 'N/A')}")
                st.write(f"**Industry**: {fundamentals.get('industry', 'N/A')}")
                if fundamentals.get('employees', 0) > 0:
                    st.write(f"**Employees**: {fundamentals['employees']:,}")
            
            with col2:
                if fundamentals.get('business_summary'):
                    st.write("**Business Overview**:")
                    summary = fundamentals['business_summary']
                    if len(summary) > 250:
                        summary = summary[:250] + "..."
                    st.write(summary)
            
            with col2:
                # Calculate daily change for direction
                daily_change = current_price - data['Close'].iloc[-2] if len(data) > 1 else 0
                direction = "📈 BULLISH" if daily_change > 0 else "📉 BEARISH"
                confidence_level = "HIGH" if reasoning.get('confidence', 70) > 80 else "MODERATE"
                st.metric("Market Outlook", f"{direction}")
                st.write(f"Confidence: {confidence_level}")
            
            with col3:
                final_confidence = reasoning.get('confidence', 70)
                st.metric("AI Confidence", f"{final_confidence:.0f}%")
                
                # Risk indicator
                daily_change_pct = (daily_change / data['Close'].iloc[-2]) * 100 if len(data) > 1 else 0
                risk_level = "LOW" if abs(daily_change_pct) < 5 else "MODERATE" if abs(daily_change_pct) < 10 else "HIGH"
                st.write(f"Risk Level: {risk_level}")
        
        # Detailed reasoning display
        if show_reasoning and reasoning and 'error' not in reasoning:
            st.subheader("🧠 Detailed Analysis Reasoning")
            
            # Technical Analysis
            if reasoning.get('technical_factors'):
                with st.expander("⚙️ Technical Analysis Factors", expanded=True):
                    for factor in reasoning['technical_factors']:
                        st.markdown(f"• {factor}")
            
            # Fundamental Analysis
            if reasoning.get('fundamental_factors'):
                with st.expander("🏢 Fundamental Analysis Factors", expanded=True):
                    for factor in reasoning['fundamental_factors']:
                        st.markdown(f"• {factor}")
            
            # News Analysis
            if reasoning.get('news_factors'):
                with st.expander("📰 News & Sentiment Analysis", expanded=True):
                    for factor in reasoning['news_factors']:
                        st.markdown(f"• {factor}")
            
            # Pattern Analysis
            if reasoning.get('pattern_factors'):
                with st.expander("📊 Historical Pattern Analysis", expanded=True):
                    for factor in reasoning['pattern_factors']:
                        st.markdown(f"• {factor}")
            
            # Model Analysis
            if reasoning.get('model_factors'):
                with st.expander("🤖 AI Model Analysis", expanded=True):
                    for factor in reasoning['model_factors']:
                        st.markdown(f"• {factor}")
        
        # Action recommendations
        st.subheader("💡 Actionable Recommendations")
        
        if predictions and reasoning:
            predicted_change_pct = reasoning.get('predicted_change_pct', 0)
            final_confidence = reasoning.get('confidence', 70)
            
            # Generate position-based recommendations
            recommendations = analyzer.generate_position_based_recommendations(
                predicted_change_pct, final_confidence, user_position, current_price
            )
            
            # Display recommendation
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background-color: {recommendations['action_color']}; color: white; border-radius: 10px; margin: 1rem 0;'>
                <h3 style='margin: 0;'>{recommendations['primary_action']}</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>For investors who {user_position.lower()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Specific Actions:**")
            for action in recommendations['specific_actions']:
                st.write(f"• {action}")
            
            # Risk factors
            st.write("**⚠️ Key Risk Factors to Monitor:**")
            risk_factors = [
                "📊 Overall market sentiment and major indices direction",
                "📰 Company-specific news and quarterly results",
                "🌍 Global economic factors and institutional flows",
                "📈 Technical level breaks (support/resistance)",
                "💹 Sector rotation and peer performance"
            ]
            
            for risk in risk_factors:
                st.write(f"• {risk}")
        
        # === TECHNICAL CHARTS ===
        if show_charts:
            st.markdown("### 📈 Technical Analysis Charts & Pattern Interpretation")
            
            # Display the chart first
            enhanced_chart = create_comprehensive_charts(
                df_with_indicators, fundamentals, patterns, 
                f"{stock_symbol} - Deep Technical Analysis"
            )
            st.plotly_chart(enhanced_chart, width='stretch')
            
            # Add comprehensive chart analysis
            st.markdown("#### 📊 Detailed Chart Analysis")
            
            # Get comprehensive chart analysis
            chart_analysis = analyze_chart_bullish_bearish_signals(df_with_indicators)
            
            # Display detailed analysis in pointwise manner
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🟢 **Bullish Signals**")
                if chart_analysis['bullish_signals']:
                    for i, signal in enumerate(chart_analysis['bullish_signals'], 1):
                        st.write(f"**{i}.** {signal}")
                else:
                    st.write("• No strong bullish signals detected")
                
                st.markdown("##### 📈 **Trend Analysis**")
                for trend in chart_analysis['trend_analysis']:
                    st.write(f"• {trend}")
                
                st.markdown("##### 📊 **Momentum Analysis**")
                for momentum in chart_analysis['momentum_analysis']:
                    st.write(f"• {momentum}")
            
            with col2:
                st.markdown("##### 🔴 **Bearish Signals**")
                if chart_analysis['bearish_signals']:
                    for i, signal in enumerate(chart_analysis['bearish_signals'], 1):
                        st.write(f"**{i}.** {signal}")
                else:
                    st.write("• No strong bearish signals detected")
                
                st.markdown("##### 📉 **Volume Analysis**")
                for volume in chart_analysis['volume_analysis']:
                    st.write(f"• {volume}")
                
                st.markdown("##### 🎯 **Pattern Analysis**")
                for pattern in chart_analysis['pattern_analysis']:
                    st.write(f"• {pattern}")
            
            # Key Levels and Risk Factors
            st.markdown("##### 🎯 **Key Levels**")
            if chart_analysis['key_levels']:
                for level in chart_analysis['key_levels']:
                    st.write(f"• {level}")
            else:
                st.write("• No significant key levels identified")
            
            st.markdown("##### ⚠️ **Risk Factors**")
            if chart_analysis['risk_factors']:
                for risk in chart_analysis['risk_factors']:
                    st.write(f"• {risk}")
            else:
                st.write("• No significant risk factors identified")
            
            # Overall sentiment summary
            sentiment_color = "green" if chart_analysis['overall_sentiment'] == 'bullish' else "red" if chart_analysis['overall_sentiment'] == 'bearish' else "orange"
            st.markdown(f"""
            <div style='padding: 1rem; background: {sentiment_color}20; border-left: 4px solid {sentiment_color}; border-radius: 5px; margin: 1rem 0;'>
                <h5 style='color: {sentiment_color}; margin-top: 0;'>Overall Chart Sentiment: {chart_analysis['overall_sentiment'].upper()}</h5>
                <p style='margin-bottom: 0;'>Confidence Level: {chart_analysis['confidence_level']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual Chart Component Analysis
            st.markdown("#### 📊 Individual Chart Component Analysis")
            
            # Get detailed analysis for each chart component
            individual_analysis = analyze_individual_charts(df_with_indicators)
            
            # Display analysis for each chart component
            chart_components = [
                ("Price Action & Key Levels", individual_analysis['price_action'], "📈"),
                ("Volume Analysis", individual_analysis['volume_analysis'], "📊"),
                ("RSI & MACD", individual_analysis['rsi_macd_analysis'], "📉"),
                ("Moving Averages & Trends", individual_analysis['moving_averages_analysis'], "📈"),
                ("Bollinger Bands & Volatility", individual_analysis['bollinger_bands_analysis'], "📊"),
                ("Historical Performance", individual_analysis['historical_performance_analysis'], "📈")
            ]
            
            for component_name, analysis_points, emoji in chart_components:
                if analysis_points:
                    st.markdown(f"##### {emoji} **{component_name}**")
                    for point in analysis_points:
                        st.write(f"• {point}")
                    st.write("")  # Add spacing
            
            # Add LLM-powered chart analysis
            st.markdown("#### 🤖 AI Chart Pattern Analysis")
            
            with st.spinner("Analyzing chart patterns with AI..."):
                llm_chart_analysis = analyzer.analyze_chart_patterns_with_llm(data, patterns)
            
            st.markdown(f"""
            <div style='padding: 1rem; background-color: #f8f9fa; border-left: 4px solid #007bff; border-radius: 5px;'>
                <h5 style='color: #007bff; margin-top: 0;'>What the Charts Are Telling Us:</h5>
                <p style='margin-bottom: 0;'>{llm_chart_analysis}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # === NEWS AND MARKET INTELLIGENCE ===
        if show_news and news_data and 'headlines' in news_data and news_data['headlines']:
            st.markdown("### 📰 Market News & Sentiment")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Recent Headlines:**")
                for i, headline in enumerate(news_data['headlines'][:5]):  # Show only top 5
                    if isinstance(headline, dict):
                        title = headline.get('title', 'No title')
                        source = headline.get('source', 'Unknown')
                        st.write(f"{i+1}. **{title}** _{source}_")
                    else:
                        st.write(f"{i+1}. {headline}")
            
            with col2:
                sentiment_score = news_data.get('sentiment_score', 0)
                if sentiment_score > 0.3:
                    sentiment_emoji = "😊"
                    sentiment_text = "Positive"
                    sentiment_color = "green"
                elif sentiment_score < -0.3:
                    sentiment_emoji = "😟"
                    sentiment_text = "Negative"
                    sentiment_color = "red"
                else:
                    sentiment_emoji = "😐"
                    sentiment_text = "Neutral"
                    sentiment_color = "orange"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; border: 2px solid {sentiment_color}; border-radius: 10px;'>
                    <h3 style='margin: 0; color: {sentiment_color};'>{sentiment_emoji}</h3>
                    <h4 style='margin: 0; color: {sentiment_color};'>{sentiment_text}</h4>
                    <p style='margin: 0; color: #666;'>Score: {sentiment_score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # === DETAILED REASONING (OPTIONAL) ===
        if show_reasoning and reasoning and 'error' not in reasoning:
            with st.expander("🧠 Detailed Analysis Reasoning", expanded=False):
                
                # Technical Analysis
                if reasoning.get('technical_factors'):
                    st.markdown("**🔧 Technical Analysis:**")
                    for factor in reasoning['technical_factors']:
                        st.markdown(f"• {factor}")
                    st.markdown("---")
                
                # Fundamental Analysis
                if reasoning.get('fundamental_factors'):
                    st.markdown("**🏢 Fundamental Analysis:**")
                    for factor in reasoning['fundamental_factors']:
                        st.markdown(f"• {factor}")
                    st.markdown("---")
                
                # News Analysis
                if reasoning.get('news_factors'):
                    st.markdown("**📰 News Sentiment:**")
                    for factor in reasoning['news_factors']:
                        st.markdown(f"• {factor}")
                    st.markdown("---")
                
                # Score breakdown
                final_score = reasoning.get('final_score', 0)
                predicted_change = reasoning.get('predicted_change_pct', 0)
                final_confidence = reasoning.get('confidence', 70)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📊 Analysis Weights:**")
                    st.write("• Technical: 40% | Fundamental: 35%")
                    st.write("• News: 15% | Patterns: 10%")
                
                with col2:
                    st.markdown("**📈 Final Results:**")
                    st.write(f"• Combined Score: {final_score:.3f}")
                    st.write(f"• Confidence: {final_confidence:.0f}%")
        
        # === METHODOLOGY (OPTIONAL) ===
        if show_methodology:
            with st.expander("🔬 Analysis Methodology", expanded=False):
                st.markdown("""
                **Our AI Analysis Process:**
                
                1. **Data Collection**: Real-time stock prices, volume, and market data
                2. **Technical Analysis**: 20+ indicators including RSI, MACD, Moving Averages
                3. **Fundamental Analysis**: P/E ratio, growth metrics, financial health
                4. **Sentiment Analysis**: News headlines and market sentiment
                5. **Pattern Recognition**: Historical trends and seasonal patterns
                6. **AI Prediction**: Multiple machine learning models for price forecasting
                7. **Risk Assessment**: Confidence scoring and risk factor identification
                
                **Key Features:**
                - ✅ Real-time data integration
                - ✅ Multi-factor analysis approach
                - ✅ Conservative prediction methodology
                - ✅ Transparent reasoning process
                - ✅ Risk-adjusted recommendations
                """)
        
        # === MODEL PERFORMANCE (DEEP ANALYSIS ONLY) ===
        if analysis_depth == "Deep Analysis" and model_results:
            with st.expander("🤖 AI Model Performance", expanded=False):
                model_df = pd.DataFrame({
                    'Model': list(model_results.keys()),
                    'Accuracy (%)': [results['directional_accuracy'] for results in model_results.values()],
                    'Error': [f"{results['mae']:.3f}" for results in model_results.values()]
                })
                
                st.dataframe(model_df, width='stretch')
                
                best_model = max(model_results.items(), key=lambda x: x[1]['directional_accuracy'])
                st.write(f"**Best Model**: {best_model[0]} ({best_model[1]['directional_accuracy']:.1f}% accuracy)")
    
    # === FOOTER ===
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚠️ Investment Disclaimer**")
        st.write("This analysis is for educational purposes only. Always do your own research and consult financial advisors.")
    
    with col2:
        st.markdown("**📊 Data Sources**")
        st.write("Yahoo Finance, Technical Analysis Libraries, Real-time News APIs")
    
    with col3:
        st.markdown("**🔄 Last Updated**")
        st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()