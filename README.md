
# Stock Portfolio Management and Trading Algorithm

This project implements a stock portfolio management system with a trading algorithm that utilizes various technical indicators such as Moving Averages, RSI, MACD, and ARIMA-based predictions to guide buy and sell decisions. The application integrates with both Yahoo Finance and a MySQL database for storing and managing stock data, simulating a mock trading environment based on historical stock data.

## Features

### 1. Portfolio Management
- Create, display, and manage stock portfolios.
- Add or remove stocks to/from portfolios.
- Store real-time stock data and historical data for analysis.
- View portfolio details including stock symbols and their respective data.

### 2. Technical Analysis & Trading Strategy
The trading strategy is based on several well-known technical indicators:
- **SMA (Simple Moving Average)**: Compares short-term and long-term price trends.
- **RSI (Relative Strength Index)**: Signals overbought or oversold conditions in the market.
- **OBV (On-Balance Volume)**: Measures buying and selling pressure based on volume.
- **MACD (Moving Average Convergence Divergence)**: Helps identify trends and potential reversals.
- **ARIMA**: Forecasting future prices and trends based on time-series analysis.

The strategy combines these indicators using a **weighted voting system**, which triggers buy and sell signals based on the overall market sentiment.

### 3. ARIMA Forecasting
ARIMA (AutoRegressive Integrated Moving Average) is used to predict stock prices for the next 15 days. The slope of the ARIMA forecast is used as a signal:
- Positive slope → **Buy Signal**
- Negative slope → **Sell Signal**

### 4. Sharpe Ratio Calculation
The program ranks stocks based on their **Sharpe Ratio**, which is used to measure risk-adjusted returns. Stocks with a higher Sharpe ratio are preferred in the portfolio.

## Project Setup

### Requirements

- Python 3.x
- MySQL Server
- Libraries: `yfinance`, `mysql.connector`, `pandas`, `finta`, `matplotlib`, `pmdarima`, `numpy`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-portfolio-management.git
   cd stock-portfolio-management
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup MySQL database:
   - Create a MySQL database named `DSCI560`.
   - Run the SQL commands from the code to create the necessary tables for storing stock and portfolio data.

4. Update the database connection settings in the script to match your MySQL credentials:
   ```python
   conn = mysql.connector.connect(
       host="localhost",
       user="your-username",
       password="your-password",
       database="DSCI560"
   )
   ```

### Running the Application

#### Portfolio Management
1. To create and manage portfolios:
   ```bash
   python stock_portfolio_management.py
   ```

   You will be prompted with a menu to create portfolios, add or remove stocks, display portfolio details, and fetch stock data for a specified date range.

#### Trading Simulation
2. To run the mock trading environment:
   ```bash
   python trading_algorithm.py
   ```

   - Enter your initial budget.
   - Provide start and end dates for the trading simulation.
   - The program will fetch stock data, analyze using the technical indicators, and provide buy/sell signals based on the strategy.

### Database Tables
The following tables are used for storing data:
- **past_stocks**: Historical stock data (open, close, high, low, volume).
- **portfolios**: Stores the portfolio names.
- **portfolio_stocks**: Keeps track of which stocks belong to which portfolio.
- **real_time_stocks**: Real-time stock data with calculated indicators (SMA, RSI, MACD, OBV, etc.).

### Example Workflow

1. **Create Portfolio**: Create a new portfolio and add stock symbols (e.g., AAPL, MSFT).
2. **Fetch Stock Data**: Download and store stock data in the database.
3. **Simulate Trading**: The system generates buy/sell signals using the defined strategies and tracks portfolio value and cash balance.
4. **View Results**: See the performance of your mock portfolio including annualized return and stock performance.

## Visualization

During trading simulations, the stock prices and buy/sell signals are plotted using `matplotlib` to visualize the trading activity.

## Future Improvements
- Adding more sophisticated strategies for better decision making.
- Incorporating additional features such as stop-loss, risk management, and transaction fees.
- Expanding the database to store more detailed stock information.
