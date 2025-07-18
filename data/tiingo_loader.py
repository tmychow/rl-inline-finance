import os
import sys
import asyncio
from typing import List, Dict
import httpx
from dotenv import load_dotenv

# Add parent directory to path to import backend modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from backend.database import (
    init_database, insert_ticker, insert_metric, 
    insert_financial_data
)

load_dotenv()

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo"

class TiingoDataLoader:
    def __init__(self):
        self.api_key = TIINGO_API_KEY
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        # Track missing data
        self.missing_metrics = {}  # {ticker: {period: [missing_metrics]}}
        self.failed_tickers = []   # List of tickers that failed to load
        self.all_expected_metrics = set()  # Will be populated from database
        self.inserted_metrics = {}  # {ticker: {period: set(metrics)}} - track what actually got inserted
        self.statement_periods = {}  # {ticker: {period: date}} - track periods from statements
    
    async def fetch_fundamentals(self, ticker: str) -> Dict:
        url = f"{TIINGO_BASE_URL}/fundamentals/{ticker}/statements"
        params = {'token': self.api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                print(f"Successfully fetched data for {ticker}")
                return data
            else:
                print(f"Error fetching data for {ticker}: {response.status_code}")
                if response.status_code == 400:
                    error_detail = response.json().get('detail', '')
                    if 'Fundamental Data API' in error_detail:
                        print(f"Fundamental Data API add-on required for {ticker}")
                        print("Contact support@tiingo.com to add this to your account")
                    else:
                        print(f"Bad request for {ticker}: {error_detail}")
                elif response.status_code == 404:
                    print(f"Ticker {ticker} not found or no fundamental data available")
                return None
    
    async def fetch_daily_data(self, ticker: str) -> List[Dict]:
        """Fetch daily data from Tiingo for market cap, PE ratio, etc."""
        url = f"{TIINGO_BASE_URL}/fundamentals/{ticker}/daily"
        params = {'token': self.api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                print(f"Successfully fetched daily data for {ticker}")
                return data
            else:
                print(f"Error fetching daily data for {ticker}: {response.status_code}")
                return []
    
    async def process_daily_data(self, ticker: str, daily_data: List[Dict], statement_data: List[Dict] = None):
        """Process daily data, aligning with statement periods"""
        if not daily_data or not isinstance(daily_data, list):
            print(f"No daily data to process for {ticker}")
            return
        
        print(f"Processing {len(daily_data)} daily entries for {ticker}")
        print(f"  First entry date: {daily_data[0].get('date', '')[:10] if daily_data else 'N/A'}")
        print(f"  Last entry date: {daily_data[-1].get('date', '')[:10] if daily_data else 'N/A'}")
        
        # Create a date index for fast lookup
        daily_by_date = {}
        for entry in daily_data:
            date_str = entry.get('date', '')[:10]
            if date_str:
                daily_by_date[date_str] = entry
        
        # Get periods from statement data if available
        statement_periods = {}
        if statement_data:
            for stmt in statement_data:
                year = stmt.get('year')
                quarter = stmt.get('quarter')
                date_str = stmt.get('date', '')[:10]
                
                if year and quarter is not None and date_str:
                    if quarter == 0:
                        period = f"{year}FY"
                    else:
                        period = f"{year}Q{quarter}"
                    statement_periods[period] = date_str
        
        # Process daily metrics for each statement period
        daily_metrics = ['marketCap', 'peRatio', 'pbRatio', 'trailingPEG1Y']
        
        for period, statement_date in statement_periods.items():
            # Look for daily data starting from statement date and going back up to 7 days
            metrics_found = {}
            
            # Try to find data on or near the statement date
            from datetime import datetime, timedelta
            try:
                base_date = datetime.strptime(statement_date, '%Y-%m-%d')
                
                # Look back up to 7 days for each metric
                for days_back in range(8):  # 0 to 7 days
                    check_date = (base_date - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    
                    if check_date in daily_by_date:
                        entry = daily_by_date[check_date]
                        
                        # Collect any metrics we haven't found yet
                        for metric in daily_metrics:
                            if metric not in metrics_found and entry.get(metric) is not None:
                                metrics_found[metric] = entry.get(metric)
                        
                        # If we found all metrics, stop looking
                        if len(metrics_found) == len(daily_metrics):
                            break
            except ValueError:
                print(f"Invalid date format: {statement_date}")
                continue
            
            # Debug: check what we found vs what we expected
            if len(metrics_found) < len(daily_metrics):
                missing = [m for m in daily_metrics if m not in metrics_found]
                print(f"  Period {period}: Could not find daily data for {missing} within 7 days of {statement_date}")
            
            # Insert the metrics we found
            for metric_name, value in metrics_found.items():
                try:
                    unit = self._determine_unit(metric_name)
                    
                    # Convert large USD values to billions (unless already in billions)
                    if unit == 'USD' and abs(value) > 1e6:
                        value = value / 1e9
                        unit = 'USD_billions'
                    elif unit == 'USD_billions' and abs(value) > 1e6:
                        # Already should be in billions, but if raw value is too large, convert
                        value = value / 1e9
                    
                    await insert_financial_data(
                        ticker=ticker,
                        metric_name=metric_name,
                        period=period,
                        value=float(value),
                        unit=unit
                    )
                    print(f"Inserted daily {metric_name}={value:.2f} {unit} for {period}")
                    
                    # Track successful insertion
                    if ticker not in self.inserted_metrics:
                        self.inserted_metrics[ticker] = {}
                    if period not in self.inserted_metrics[ticker]:
                        self.inserted_metrics[ticker][period] = set()
                    self.inserted_metrics[ticker][period].add(metric_name)
                except Exception as e:
                    print(f"Error inserting daily {metric_name}: {e}")
    
    async def process_financial_data(self, ticker: str, data: List[Dict]):
        if not data or not isinstance(data, list):
            print(f"No data to process for {ticker}")
            return
        
        print(f"Processing {len(data)} statement objects for {ticker}")
        
        # Tiingo returns a list of statement objects
        for i, statement_obj in enumerate(data):
            if not isinstance(statement_obj, dict):
                continue
                
            print(f"Processing statement {i+1} for {ticker}")
            
            # Debug: print keys of statement object
            if i == 0:  # Only for first statement
                print(f"Statement keys: {list(statement_obj.keys())[:5]}...")
            
            # Tiingo structure: each object has date/quarter/year and statementData
            date_str = statement_obj.get('date', '')[:10]
            year = statement_obj.get('year')
            quarter = statement_obj.get('quarter')
            
            # Convert to financial period notation
            if quarter == 0:
                # Annual report (FY)
                period = f"{year}FY"
            else:
                # Quarterly report
                period = f"{year}Q{quarter}"
            
            print(f"Date: {date_str}, Period: {period}")
            
            # Store period mapping for daily data alignment
            if ticker not in self.statement_periods:
                self.statement_periods[ticker] = {}
            self.statement_periods[ticker][period] = date_str
            
            # Get the nested statement data
            statements = statement_obj.get('statementData', {})
            if not statements:
                print(f"No statementData found in object {i+1}")
                continue
            
            # Extract the sub-statements (they are lists)
            income_statements = statements.get('incomeStatement', [])
            balance_sheets = statements.get('balanceSheet', [])
            cash_flows = statements.get('cashFlow', [])
            overview_data = statements.get('overview', [])
            
            # Debug first statement structure
            if i == 0 and income_statements:
                print(f"Found {len(income_statements)} income statement entries")
                if income_statements and isinstance(income_statements[0], dict):
                    print(f"Income statement keys (sample): {list(income_statements[0].keys())[:5]}")
            
            inserted_count = 0
            
            # Convert dataCode/value pairs to a dictionary
            def convert_to_dict(statement_list):
                result = {}
                for item in statement_list:
                    if isinstance(item, dict) and 'dataCode' in item and 'value' in item:
                        result[item['dataCode']] = item['value']
                return result
            
            # Process income statement metrics
            if income_statements:
                income_data = convert_to_dict(income_statements)
                
                # Direct metric mappings - 1:1 with Tiingo codes
                metrics_to_insert = []
                
                # Check all income statement metrics (removed calculated metrics that are in overview)
                income_metrics = ['revenue', 'epsDil', 'netinc', 'eps', 
                                  'ebitda', 'rnd', 'sga', 'opinc', 'grossProfit', 'opex', 'ebt', 'costRev', 'ebit', 
                                  'netIncComStock', 'intexp', 'consolidatedIncome', 'shareswa', 'nonControllingInterests']
                
                found_metrics = set()
                for metric in income_metrics:
                    if metric in income_data and income_data[metric] is not None:
                        metrics_to_insert.append((metric, income_data[metric]))
                        found_metrics.add(metric)
            
            # Process balance sheet metrics
            if balance_sheets:
                balance_data = convert_to_dict(balance_sheets)
                
                # Check all balance sheet metrics (removed calculated metrics that are in overview)
                balance_metrics = ['ppeq', 'investmentsCurrent', 'liabilitiesCurrent', 'acctRec', 
                                   'cashAndEq', 'assetsCurrent', 'inventory', 'acctPay', 
                                   'totalLiabilities', 'sharesBasic', 'equity', 'taxLiabilities', 
                                   'deferredRev', 'debt']
                
                for metric in balance_metrics:
                    if metric in balance_data and balance_data[metric] is not None:
                        metrics_to_insert.append((metric, balance_data[metric]))
                        found_metrics.add(metric)
            
            # Process cash flow metrics
            if cash_flows:
                cashflow_data = convert_to_dict(cash_flows)
                
                # Check all cash flow metrics
                cashflow_metrics = ['freeCashFlow', 'payDiv', 'depamor', 'issrepayDebt', 
                                    'investmentsAcqDisposals', 'ncff', 'issrepayEquity', 
                                    'ncfo', 'ncfi', 'capex', 'sbcomp'] 
                
                for metric in cashflow_metrics:
                    if metric in cashflow_data and cashflow_data[metric] is not None:
                        metrics_to_insert.append((metric, cashflow_data[metric]))
                        found_metrics.add(metric)
            
            # Process overview metrics (calculated ratios and margins)
            if overview_data:
                overview_dict = convert_to_dict(overview_data)
                
                # Check all overview metrics
                overview_metrics = ['revenueQoQ', 'grossMargin', 'debtEquity', 'profitMargin',
                                    'rps', 'roa', 'currentRatio', 'longTermDebtEquity', 
                                    'roe', 'epsQoQ', 'bvps', 'bookVal']
                
                for metric in overview_metrics:
                    if metric in overview_dict and overview_dict[metric] is not None:
                        metrics_to_insert.append((metric, overview_dict[metric]))
                        found_metrics.add(metric)
            
            # Insert all collected metrics
            for metric_name, value in metrics_to_insert:
                try:
                    unit = self._determine_unit(metric_name)
                    
                    # Convert large USD values to billions for better readability
                    if unit == 'USD' and abs(value) > 1e6:
                        value = value / 1e9
                        unit = 'USD_billions'
                    
                    await insert_financial_data(
                        ticker=ticker,
                        metric_name=metric_name,
                        period=period,
                        value=float(value),
                        unit=unit
                    )
                    inserted_count += 1
                    print(f"Inserted {metric_name}={value:.2f} {unit}")
                    
                    # Track successful insertion
                    if ticker not in self.inserted_metrics:
                        self.inserted_metrics[ticker] = {}
                    if period not in self.inserted_metrics[ticker]:
                        self.inserted_metrics[ticker][period] = set()
                    self.inserted_metrics[ticker][period].add(metric_name)
                except Exception as e:
                    print(f"Error inserting {metric_name}: {e}")
            
            # Don't track missing metrics here anymore - we'll do it after both endpoints
            
            if inserted_count > 0:
                print(f"Statement {i+1}: Inserted {inserted_count} metrics")
    
    
    def _determine_unit(self, metric_name: str) -> str:
        """Determine the unit for a given metric"""
        # Percentage metrics
        percentage_metrics = ['grossMargin', 'roe', 'roa', 
                              'revenueQoQ', 'epsQoQ']
        
        # Ratio metrics
        ratio_metrics = ['peRatio', 'debtEquity', 'currentRatio', 'trailingPEG1Y', 'pbRatio']
        
        # Large USD metrics that should be in billions
        billion_dollar_metrics = ['marketCap']
        
        # Per share metrics
        per_share_metrics = ['eps', 'epsDil', 'bvps']
        
        # Count metrics
        count_metrics = ['sharesBasic', 'shareswa']
        
        if metric_name in percentage_metrics:
            return 'percentage'
        elif metric_name in ratio_metrics:
            return 'ratio'
        elif metric_name in per_share_metrics:
            return 'USD'
        elif metric_name in count_metrics:
            return 'count'
        elif metric_name in billion_dollar_metrics:
            return 'USD_billions'
        else:
            # Default to USD for financial metrics
            return 'USD'
    
    async def calculate_ratios(self, ticker: str):
        pass
    
    def reconcile_missing_metrics(self, ticker: str):
        """After processing both endpoints, determine what's truly missing"""
        if ticker not in self.inserted_metrics:
            # No data was inserted for this ticker at all
            return
        
        ticker_inserted = self.inserted_metrics[ticker]
        
        # Check each period that we have data for
        for period in ticker_inserted:
            inserted = ticker_inserted[period]
            missing = self.all_expected_metrics - inserted
            
            if missing:
                if ticker not in self.missing_metrics:
                    self.missing_metrics[ticker] = {}
                self.missing_metrics[ticker][period] = sorted(list(missing))
                print(f"After both endpoints, missing {len(missing)} metrics for {ticker} in {period}")
    
    def save_missing_data_report(self, filename: str = "missing_data_report.json"):
        """Save a report of all missing data to a JSON file"""
        import json
        
        # Define which metrics come from which endpoint
        statement_metrics = {
            'revenue', 'epsDil', 'netinc', 'sharesBasic', 'eps', 'ebitda', 'rnd', 'sga', 
            'opinc', 'grossProfit', 'sbcomp', 'opex', 'ebt', 'costRev', 'ebit', 
            'netIncComStock', 'intexp', 'consolidatedIncome', 'shareswa', 'ppeq', 
            'investmentsCurrent', 'liabilitiesCurrent', 'acctRec', 'cashAndEq', 'assetsCurrent', 
            'inventory', 'acctPay', 'totalLiabilities', 'equity', 'taxLiabilities', 
            'nonControllingInterests', 'deferredRev', 'debt', 'freeCashFlow', 'payDiv', 
            'depamor', 'issrepayDebt', 'investmentsAcqDisposals', 'ncff', 'issrepayEquity', 
            'ncfo', 'ncfi', 'capex', 'revenueQoQ', 'grossMargin', 'debtEquity', 'profitMargin',
            'rps', 'roa', 'currentRatio', 'longTermDebtEquity', 'roe', 'epsQoQ', 'bvps', 'bookVal'
        }
        
        daily_metrics = {'marketCap', 'peRatio', 'pbRatio', 'trailingPEG1Y'}
        
        # Calculate missing metrics breakdown by period type
        fiscal_year_missing = 0
        quarterly_missing = 0
        fiscal_year_missing_details = {}
        quarterly_missing_details = {}
        
        for ticker, periods in self.missing_metrics.items():
            for period, metrics in periods.items():
                if period.endswith('FY'):
                    fiscal_year_missing += len(metrics)
                    if ticker not in fiscal_year_missing_details:
                        fiscal_year_missing_details[ticker] = 0
                    fiscal_year_missing_details[ticker] += len(metrics)
                else:  # Quarterly (Q1, Q2, Q3, Q4)
                    quarterly_missing += len(metrics)
                    if ticker not in quarterly_missing_details:
                        quarterly_missing_details[ticker] = 0
                    quarterly_missing_details[ticker] += len(metrics)
        
        report = {
            "failed_tickers": self.failed_tickers,
            "missing_metrics_by_ticker": self.missing_metrics,
            "expected_sources": {
                "statement_endpoint": sorted(list(statement_metrics)),
                "daily_endpoint": sorted(list(daily_metrics))
            },
            "summary": {
                "total_failed_tickers": len(self.failed_tickers),
                "tickers_with_missing_metrics": len(self.missing_metrics),
                "total_missing_datapoints": sum(
                    len(metrics) for periods in self.missing_metrics.values() 
                    for metrics in periods.values()
                ),
                "missing_by_period_type": {
                    "fiscal_year": fiscal_year_missing,
                    "quarterly": quarterly_missing,
                    "fiscal_year_percentage": round(100 * fiscal_year_missing / (fiscal_year_missing + quarterly_missing), 1) if (fiscal_year_missing + quarterly_missing) > 0 else 0,
                    "quarterly_percentage": round(100 * quarterly_missing / (fiscal_year_missing + quarterly_missing), 1) if (fiscal_year_missing + quarterly_missing) > 0 else 0
                },
                "missing_by_ticker_and_type": {
                    "fiscal_year": fiscal_year_missing_details,
                    "quarterly": quarterly_missing_details
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nMissing data report saved to {filename}")
        print(f"Summary:")
        print(f"  - Failed tickers: {report['summary']['total_failed_tickers']}")
        print(f"  - Tickers with missing metrics: {report['summary']['tickers_with_missing_metrics']}")
        print(f"  - Total missing datapoints: {report['summary']['total_missing_datapoints']}")
        print(f"  - Missing by period type:")
        print(f"    * Fiscal year: {fiscal_year_missing} ({report['summary']['missing_by_period_type']['fiscal_year_percentage']}%)")
        print(f"    * Quarterly: {quarterly_missing} ({report['summary']['missing_by_period_type']['quarterly_percentage']}%)")
    
    async def load_ticker_data(self, ticker: str, company_name: str, sector: str = None):
        print(f"Loading data for {ticker}...")
        
        await insert_ticker(ticker, company_name, sector)
        
        if self.api_key:
            # Try to fetch fundamentals for any ticker
            statement_data = await self.fetch_fundamentals(ticker)
            daily_data = await self.fetch_daily_data(ticker)
            
            if statement_data:
                await self.process_financial_data(ticker, statement_data)
            else:
                print(f"No fundamental statement data available for {ticker}")
                
            if daily_data:
                # Pass statement data to daily processing for period alignment
                await self.process_daily_data(ticker, daily_data, statement_data)
            else:
                print(f"No daily data available for {ticker}")
                
            # Only mark as failed if both endpoints failed
            if not statement_data and not daily_data:
                self.failed_tickers.append(ticker)
            else:
                # Reconcile missing metrics after processing both endpoints
                self.reconcile_missing_metrics(ticker)
                
            await self.calculate_ratios(ticker)
        else:
            print(f"No API key found")
            self.failed_tickers.append(ticker)
    
    async def _load_sample_data(self, ticker: str):
        # NO FAKE DATA - only use real data from APIs
        print(f"No financial data available for {ticker} - Tiingo Fundamental Data API add-on required")

async def main():
    await init_database()
    
    # Insert all metrics with their descriptions
    metrics = [
        ("roa", "Return on Assets ROA", "percentage"),
        ("revenue", "Revenue", "USD_billions"),
        ("epsDil", "Earnings Per Share Diluted", "USD"),
        ("netinc", "Net Income", "USD_billions"),
        ("revenueQoQ", "Revenue QoQ Growth", "percentage"),
        ("grossMargin", "Gross Margin", "percentage"),
        ("roe", "Return on Equity ROE", "percentage"),
        ("sharesBasic", "Shares Outstanding", "count"),
        ("epsQoQ", "Earnings Per Share QoQ Growth", "percentage"),
        ("peRatio", "Price to Earnings Ratio", "ratio"),
        ("eps", "Earnings Per Share", "USD"),
        ("ppeq", "Property, Plant & Equipment", "USD_billions"),
        ("ebitda", "EBITDA", "USD_billions"),
        ("freeCashFlow", "Free Cash Flow", "USD_billions"),
        ("capex", "Capital Expenditure", "USD_billions"),
        ("rnd", "Research & Development", "USD_billions"),
        ("sga", "Selling, General & Administrative", "USD_billions"),
        ("investmentsCurrent", "Current Investments", "USD_billions"),
        ("payDiv", "Payment of Dividends & Other Cash Distributions", "USD_billions"),
        ("opinc", "Operating Income", "USD_billions"),
        ("grossProfit", "Gross Profit", "USD_billions"),
        ("sbcomp", "Shared-based Compensation", "USD_billions"),
        ("liabilitiesCurrent", "Current Liabilities", "USD_billions"),
        ("acctRec", "Accounts Receivable", "USD_billions"),
        ("cashAndEq", "Cash and Equivalents", "USD_billions"),
        ("depamor", "Depreciation, Amortization & Accretion", "USD_billions"),
        ("assetsCurrent", "Current Assets", "USD_billions"),
        ("opex", "Operating Expenses", "USD_billions"),
        ("inventory", "Inventory", "USD_billions"),
        ("ebt", "Earnings before tax", "USD_billions"),
        ("costRev", "Cost of Revenue", "USD_billions"),
        ("acctPay", "Accounts Payable", "USD_billions"),
        ("totalLiabilities", "Total Liabilities", "USD_billions"),
        ("ebit", "Earning Before Interest & Taxes EBIT", "USD_billions"),
        ("netIncComStock", "Net Income Common Stock", "USD_billions"),
        ("intexp", "Interest Expense", "USD_billions"),
        ("consolidatedIncome", "Consolidated Income", "USD_billions"),
        ("equity", "Shareholders Equity", "USD_billions"),
        ("marketCap", "Market Capitalization", "USD_billions"),
        ("bookVal", "Book Value", "USD_billions"),
        ("bvps", "Book Value Per Share", "USD"),
        ("debtEquity", "Debt to Equity Ratio", "ratio"),
        ("currentRatio", "Current Ratio", "ratio"),
        ("issrepayDebt", "Issuance or Repayment of Debt Securities", "USD_billions"),
        ("investmentsAcqDisposals", "Investment Acquisitions & Disposals", "USD_billions"),
        ("taxLiabilities", "Tax Liabilities", "USD_billions"),
        ("ncff", "Net Cash Flow from Financing", "USD_billions"),
        ("nonControllingInterests", "Net Income to Non-Controlling Interests", "USD_billions"),
        ("issrepayEquity", "Issuance or Repayment of Equity", "USD_billions"),
        ("ncfo", "Net Cash Flow from Operations", "USD_billions"),
        ("shareswa", "Weighted Average Shares", "count"),
        ("deferredRev", "Deferred Revenue", "USD_billions"),
        ("debt", "Total Debt", "USD_billions"),
        ("ncfi", "Net Cash Flow from Investing", "USD_billions"),
        ("trailingPEG1Y", "PEG Ratio", "ratio"),
        ("pbRatio", "Price to Book Ratio", "ratio")
    ]
    
    for metric_name, description, unit in metrics:
        await insert_metric(metric_name, description, unit)
    
    loader = TiingoDataLoader()
    
    # Populate expected metrics - now we expect all of them since we have both endpoints
    loader.all_expected_metrics = {m[0] for m in metrics}
    
    tickers_to_load = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corporation", "Technology"),
        ("GOOGL", "Alphabet Inc.", "Technology"),
        ("AMZN", "Amazon.com Inc.", "Technology"),
        ("META", "Meta Platforms Inc.", "Technology"),
        ("TSLA", "Tesla Inc.", "Consumer Cyclical"),
        ("NVDA", "NVIDIA Corporation", "Technology"),
        ("INTC", "Intel Corporation", "Technology"),
        ("VST", "Vistra Corp.", "Utilities"),
        ("GEV", "GE Vernova Inc.", "Industrials"),
        ("MU", "Micron Technology Inc.", "Technology"),
        ("CEG", "Constellation Energy Corporation", "Utilities"),
        ("VRT", "Vertiv Holdings Co.", "Technology"),
        ("NRG", "NRG Energy Inc.", "Utilities"),
        ("TLN", "Talen Energy Corporation", "Utilities"),
        ("NBIS", "Nebius Group N.V.", "Technology"),
        ("COHR", "Coherent Corp.", "Technology"),
        ("ONTO", "Onto Innovation Inc.", "Technology"),
        ("BE", "Bloom Energy Corporation", "Industrials"),
        ("CRDO", "Credo Technology Group Holding Ltd", "Technology"),
        ("FN", "Fabrinet", "Technology"),
        ("TSM", "Taiwan Semiconductor Manufacturing", "Technology"),
        ("AVGO", "Broadcom Inc.", "Technology"),
        ("MRVL", "Marvell Technology Inc.", "Technology"),
        ("AMAT", "Applied Materials Inc.", "Technology"),
        ("ORCL", "Oracle Corporation", "Technology"),
        ("TSSI", "TSS Inc.", "Technology"),
        ("LRCX", "Lam Research Corporation", "Technology"),
        ("ASML", "ASML Holding N.V.", "Technology"),
        ("ANET", "Arista Networks Inc.", "Technology"),
    ]
    
    for ticker, company_name, sector in tickers_to_load:
        await loader.load_ticker_data(ticker, company_name, sector)
    
    # Save missing data report
    loader.save_missing_data_report()
    
    print("\nData loading completed!")

if __name__ == "__main__":
    asyncio.run(main())