# Stock Predictor - Streamlit Application

A comprehensive stock analysis and prediction application built with Streamlit, featuring machine learning models, technical analysis, and real-time data visualization.

## Features

- **Real-time Stock Data**: Fetches live stock data using Yahoo Finance API
- **Machine Learning Predictions**: Multiple ML models for price prediction
- **Technical Analysis**: Advanced technical indicators and charts
- **News Integration**: Real-time news analysis and sentiment
- **Interactive Visualizations**: Dynamic charts with Plotly
- **Multi-timeframe Analysis**: 1-day to 30-day predictions

## Requirements

- **Python**: 3.10.x (specified in `runtime.txt`)
- **Dependencies**: See `requirements.txt` for a complete list

## Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd stock_predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (copy from .env.template):
```bash
cp .env.template .env
# Edit .env with your actual API keys
```

4. Run the application:
```bash
streamlit run stock_predictor.py
```

## Deployment on Render

This application is configured for deployment on Render. See the deployment section below for step-by-step instructions.

### Required Environment Variables for Deployment:

- `GROQ_API_KEY`: For AI-powered analysis
- `NEWSAPI_KEY`: For news integration
- `TAVILY_API_KEY`: For web scraping capabilities
- `OPENAI_API_KEY`: (Optional) For enhanced AI features

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## Disclaimer

This application is for educational and informational purposes only. Stock predictions are based on historical data and technical analysis. Always consult with financial advisors and do your own research before making investment decisions.

## License

MIT License