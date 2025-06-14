from typing import Dict, List, Optional
import logging
import pandas as pd
import numpy as np
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_financial_data(symbol: str) -> Dict:
    """Load financial data for a given symbol from JSON files or scrape from web if not available"""
    try:
        financial_data = {}
        json_path = f"data/financials/{symbol}.json"
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                financial_data = json.load(f)
                logger.info(f"Loaded financial data for {symbol} from local storage")
                return financial_data
        else:
            logger.warning(f"No financial data found for {symbol} at {json_path}, attempting to scrape data from web")
            financial_data = scrape_financial_data(symbol)
            if financial_data and financial_data.get('income_statement'):
                # Save scraped data for future use
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w') as f:
                    json.dump(financial_data, f, indent=2)
                logger.info(f"Saved scraped financial data for {symbol} to {json_path}")
                return financial_data
            else:
                logger.warning(f"Failed to scrape financial data for {symbol}")
                return {
                    'income_statement': {},
                    'balance_sheet': {},
                    'cash_flow': {},
                    'ratios': {},
                    'last_updated': None
                }
            
    except Exception as e:
        logger.error(f"Error loading financial data for {symbol}: {e}")
        return {
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'ratios': {},
            'last_updated': None
        }

def scrape_financial_data(symbol: str) -> Dict:
    """Scrape financial data from dps.psx.com.pk and simplywall.st for all symbols using Selenium"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from datetime import datetime
        import time
        
        # Initialize financial data structure
        financial_data = {
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'ratios': {},
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Set up Selenium WebDriver with headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(180)  # Increase timeout to 180 seconds
        driver.set_script_timeout(180)     # Increase script timeout to 180 seconds
        
        # Function to extract data from PSX with enhanced retry mechanism
        def scrape_psx_data(url):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "span[class*='quote__close']")))
                    price_elem = driver.find_element(By.CSS_SELECTOR, "span[class*='quote__close']")
                    data = {}
                    if price_elem:
                        price_text = price_elem.text.strip()
                        data['price'] = float(price_text) if price_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0
                    
                    try:
                        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.ID, "financialTab")))
                        financial_tab = driver.find_element(By.ID, "financialTab")
                        eps_rows = financial_tab.find_elements(By.CSS_SELECTOR, "tbody tr")
                        for row in eps_rows:
                            if 'EPS' in row.text:
                                cells = row.find_elements(By.CSS_SELECTOR, "td")
                                if len(cells) > 1:
                                    eps_text = cells[1].text.strip()
                                    data['eps'] = float(eps_text) if eps_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0
                                    break
                    except Exception as inner_e:
                        logger.warning(f"Could not find EPS element for {symbol} in financial tab: {inner_e}")
                        try:
                            eps_elements = driver.find_elements(By.CSS_SELECTOR, "td")
                            for elem in eps_elements:
                                if 'EPS' in elem.text.upper():
                                    eps_text = elem.text.strip().replace('EPS', '').strip()
                                    data['eps'] = float(eps_text) if eps_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0
                                    break
                        except Exception as inner_e2:
                            logger.error(f"Alternative EPS search failed for {symbol}: {inner_e2}")
                    
                    return data
                except Exception as e:
                    logger.error(f"Error scraping PSX data for {symbol}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 15 * (2 ** attempt)
                        logger.info(f"Retrying PSX scrape for {symbol} after {wait_time} seconds due to potential rate limiting or network issues")
                        time.sleep(wait_time)
            return {}
        
        # Function to extract data from Simply Wall St with enhanced retry mechanism
        def scrape_sws_data(url):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "span[class*='eps-ttm']")))
                    eps_elem = driver.find_element(By.CSS_SELECTOR, "span[class*='eps-ttm']")
                    if eps_elem:
                        eps_text = eps_elem.text.strip()
                        return {'eps': float(eps_text) if eps_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0}
                except Exception as e:
                    logger.error(f"Error scraping Simply Wall St data for {symbol}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 15 * (2 ** attempt)
                        logger.info(f"Retrying Simply Wall St scrape for {symbol} after {wait_time} seconds due to potential rate limiting or network issues")
                        time.sleep(wait_time)
            return {}
        
        # Function to extract data from Investing.com with enhanced retry mechanism
        def scrape_investing_data(url):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-test='key-stats']")))
                    stats_div = driver.find_element(By.CSS_SELECTOR, "div[data-test='key-stats']")
                    eps_elements = stats_div.find_elements(By.CSS_SELECTOR, "div")
                    data = {}
                    for elem in eps_elements:
                        text = elem.text.strip()
                        if 'EPS' in text:
                            eps_text = text.split('EPS')[1].strip().split()[0]
                            data['eps'] = float(eps_text) if eps_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0
                            break
                    price_elements = driver.find_elements(By.CSS_SELECTOR, "span[data-test='instrument-price-last']")
                    if price_elements:
                        price_text = price_elements[0].text.strip()
                        data['price'] = float(price_text) if price_text.replace('.', '', 1).lstrip('-').isdigit() else 0.0
                    return data
                except Exception as e:
                    logger.error(f"Error scraping Investing.com data for {symbol}, attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 15 * (2 ** attempt)
                        logger.info(f"Retrying Investing.com scrape for {symbol} after {wait_time} seconds due to potential rate limiting or network issues")
                        time.sleep(wait_time)
            return {}
        
        try:
            # Attempt scraping from PSX for all symbols
            psx_url = f"https://dps.psx.com.pk/company/{symbol.upper()}"
            psx_data = scrape_psx_data(psx_url)
            if psx_data:
                if 'price' in psx_data:
                    financial_data['income_statement']['current_price'] = psx_data.get('price', 0.0)
                if 'eps' in psx_data:
                    financial_data['income_statement']['eps'] = psx_data.get('eps', 0.0)
            
            # Attempt scraping from Simply Wall St for all symbols
            # Construct Simply Wall St URL based on symbol with a more specific pattern
            sws_url = f"https://simplywall.st/stocks/pk/energy/kase-{symbol.lower()}/{symbol.lower()}-shares"
            sws_data = scrape_sws_data(sws_url)
            if sws_data and 'eps' in sws_data:
                financial_data['income_statement']['eps'] = sws_data.get('eps', financial_data['income_statement'].get('eps', 0.0))
            
            # Attempt scraping from Investing.com as an additional source
            # Construct Investing.com URL based on symbol (assuming a pattern; may need adjustment for different symbols)
            investing_url = f"https://www.investing.com/equities/{symbol.lower()}"
            investing_data = scrape_investing_data(investing_url)
            if investing_data:
                if 'price' in investing_data and (not financial_data['income_statement'].get('current_price') or financial_data['income_statement']['current_price'] == 0.0):
                    financial_data['income_statement']['current_price'] = investing_data.get('price', 0.0)
                if 'eps' in investing_data and (not financial_data['income_statement'].get('eps') or financial_data['income_statement']['eps'] == 0.0):
                    financial_data['income_statement']['eps'] = investing_data.get('eps', 0.0)
            
            # If data is found, return it
            if financial_data['income_statement'] or financial_data['balance_sheet']:
                logger.info(f"Successfully scraped financial data for {symbol} from PSX/Simply Wall St/Investing.com using Selenium")
                return financial_data
            
            logger.error(f"Failed to fetch data from all sources for {symbol}")
            return {
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow': {},
                'ratios': {},
                'last_updated': None
            }
        finally:
            driver.quit()
        
    except Exception as e:
        logger.error(f"Error scraping financial data for {symbol} using Selenium: {e}")
        return {
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'ratios': {},
            'last_updated': None
        }

def analyze_financials(financial_data: Dict) -> Dict:
    """Analyze financial statements and ratios with advanced metrics"""
    try:
        analysis = {
            'profitability': analyze_profitability(financial_data.get('income_statement', {}), 
                                                 financial_data.get('ratios', {})),
            'liquidity': analyze_liquidity(financial_data.get('balance_sheet', {}), 
                                         financial_data.get('ratios', {})),
            'solvency': analyze_solvency(financial_data.get('balance_sheet', {}), 
                                       financial_data.get('ratios', {})),
            'efficiency': analyze_efficiency(financial_data.get('income_statement', {}), 
                                           financial_data.get('balance_sheet', {}), 
                                           financial_data.get('ratios', {})),
            'growth': analyze_growth(financial_data.get('income_statement', {}), 
                                   financial_data.get('cash_flow', {})),
            'valuation': analyze_valuation(financial_data.get('ratios', {}))
        }
        
        # Calculate overall financial score
        analysis['financial_score'] = calculate_financial_score(analysis)
        
        # Placeholder for advanced features such as machine learning-based financial predictions
        # Future implementation could include predictive models for financial health or anomaly detection
        analysis['advanced_metrics'] = {
            'note': 'Advanced financial metrics and predictive analytics to be implemented in future updates.'
        }
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing financials: {e}")
        return {
            'profitability': {},
            'liquidity': {},
            'solvency': {},
            'efficiency': {},
            'growth': {},
            'valuation': {},
            'financial_score': 0.0,
            'advanced_metrics': {}
        }

def analyze_profitability(income_statement: Dict, ratios: Dict) -> Dict:
    """Analyze profitability metrics"""
    try:
        gross_margin = ratios.get('grossProfitMargin', 0.0)
        operating_margin = ratios.get('operatingProfitMargin', 0.0)
        net_margin = ratios.get('netProfitMargin', 0.0)
        roe = ratios.get('returnOnEquity', 0.0)
        roa = ratios.get('returnOnAssets', 0.0)
        
        score = 0.0
        if gross_margin > 0.3:
            score += 0.2
        elif gross_margin < 0.1:
            score -= 0.2
            
        if operating_margin > 0.15:
            score += 0.2
        elif operating_margin < 0.05:
            score -= 0.2
            
        if net_margin > 0.1:
            score += 0.2
        elif net_margin < 0.03:
            score -= 0.2
            
        if roe > 0.15:
            score += 0.2
        elif roe < 0.05:
            score -= 0.2
            
        if roa > 0.05:
            score += 0.2
        elif roa < 0.02:
            score -= 0.2
            
        status = 'strong' if score >= 0.6 else 'weak' if score <= -0.6 else 'moderate'
        
        return {
            'status': status,
            'strength': score,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'roe': roe,
            'roa': roa
        }
        
    except Exception as e:
        logger.error(f"Error analyzing profitability: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_liquidity(balance_sheet: Dict, ratios: Dict) -> Dict:
    """Analyze liquidity metrics"""
    try:
        current_ratio = ratios.get('currentRatio', 0.0)
        quick_ratio = ratios.get('quickRatio', 0.0)
        cash_ratio = ratios.get('cashRatio', 0.0)
        
        score = 0.0
        if current_ratio > 2.0:
            score += 0.3
        elif current_ratio < 1.0:
            score -= 0.3
            
        if quick_ratio > 1.0:
            score += 0.2
        elif quick_ratio < 0.5:
            score -= 0.2
            
        if cash_ratio > 0.5:
            score += 0.2
        elif cash_ratio < 0.2:
            score -= 0.2
            
        status = 'strong' if score >= 0.5 else 'weak' if score <= -0.5 else 'moderate'
        
        return {
            'status': status,
            'strength': score,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': cash_ratio
        }
        
    except Exception as e:
        logger.error(f"Error analyzing liquidity: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_solvency(balance_sheet: Dict, ratios: Dict) -> Dict:
    """Analyze solvency metrics"""
    try:
        debt_to_equity = ratios.get('debtToEquity', 0.0)
        debt_to_assets = ratios.get('debtToAssets', 0.0)
        interest_coverage = ratios.get('interestCoverage', 0.0)
        
        score = 0.0
        if debt_to_equity < 0.5:
            score += 0.3
        elif debt_to_equity > 1.5:
            score -= 0.3
            
        if debt_to_assets < 0.3:
            score += 0.2
        elif debt_to_assets > 0.6:
            score -= 0.2
            
        if interest_coverage > 3.0:
            score += 0.2
        elif interest_coverage < 1.5:
            score -= 0.2
            
        status = 'strong' if score >= 0.5 else 'weak' if score <= -0.5 else 'moderate  moderate'
        
        return {
            'status': status,
            'strength': score,
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'interest_coverage': interest_coverage
        }
        
    except Exception as e:
        logger.error(f"Error analyzing solvency: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_efficiency(income_statement: Dict, balance_sheet: Dict, ratios: Dict) -> Dict:
    """Analyze efficiency metrics"""
    try:
        asset_turnover = ratios.get('assetTurnover', 0.0)
        inventory_turnover = ratios.get('inventoryTurnover', 0.0)
        receivables_turnover = ratios.get('receivablesTurnover', 0.0)
        
        score = 0.0
        if asset_turnover > 0.5:
            score += 0.2
        elif asset_turnover < 0.2:
            score -= 0.2
            
        if inventory_turnover > 5.0:
            score += 0.2
        elif inventory_turnover < 2.0:
            score -= 0.2
            
        if receivables_turnover > 6.0:
            score += 0.2
        elif receivables_turnover < 3.0:
            score -= 0.2
            
        status = 'efficient' if score >= 0.4 else 'inefficient' if score <= -0.4 else 'average'
        
        return {
            'status': status,
            'strength': score,
            'asset_turnover': asset_turnover,
            'inventory_turnover': inventory_turnover,
            'receivables_turnover': receivables_turnover
        }
        
    except Exception as e:
        logger.error(f"Error analyzing efficiency: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_growth(income_statement: Dict, cash_flow: Dict) -> Dict:
    """Analyze growth metrics"""
    try:
        revenue_growth = income_statement.get('revenueGrowth', 0.0)
        earnings_growth = income_statement.get('earningsGrowth', 0.0)
        cash_flow_growth = cash_flow.get('operatingCashFlowGrowth', 0.0)
        
        score = 0.0
        if revenue_growth > 0.1:
            score += 0.3
        elif revenue_growth < -0.05:
            score -= 0.3
            
        if earnings_growth > 0.1:
            score += 0.3
        elif earnings_growth < -0.05:
            score -= 0.3
            
        if cash_flow_growth > 0.1:
            score += 0.2
        elif cash_flow_growth < -0.05:
            score -= 0.2
            
        status = 'high_growth' if score >= 0.6 else 'declining' if score <= -0.6 else 'stable'
        
        return {
            'status': status,
            'strength': score,
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'cash_flow_growth': cash_flow_growth
        }
        
    except Exception as e:
        logger.error(f"Error analyzing growth: {e}")
        return {'status': 'error', 'strength': 0.0}

def analyze_valuation(ratios: Dict) -> Dict:
    """Analyze valuation metrics"""
    try:
        pe_ratio = ratios.get('trailingPE', 0.0)
        pb_ratio = ratios.get('priceToBook', 0.0)
        dividend_yield = ratios.get('dividendYield', 0.0)
        
        score = 0.0
        if pe_ratio < 15.0 and pe_ratio > 0:
            score += 0.3
        elif pe_ratio > 25.0:
            score -= 0.3
            
        if pb_ratio < 2.0 and pb_ratio > 0:
            score += 0.2
        elif pb_ratio > 4.0:
            score -= 0.2
            
        if dividend_yield > 0.03:
            score += 0.2
        elif dividend_yield < 0.01:
            score -= 0.1
            
        status = 'undervalued' if score >= 0.5 else 'overvalued' if score <= -0.5 else 'fair_valued'
        
        return {
            'status': status,
            'strength': score,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield
        }
        
    except Exception as e:
        logger.error(f"Error analyzing valuation: {e}")
        return {'status': 'error', 'strength': 0.0}

def calculate_financial_score(analysis: Dict) -> float:
    """Calculate overall financial score"""
    try:
        score = 0.0
        
        # Sum up strengths from all financial aspects
        aspects = [
            analysis.get('profitability', {}).get('strength', 0.0),
            analysis.get('liquidity', {}).get('strength', 0.0),
            analysis.get('solvency', {}).get('strength', 0.0),
            analysis.get('efficiency', {}).get('strength', 0.0),
            analysis.get('growth', {}).get('strength', 0.0),
            analysis.get('valuation', {}).get('strength', 0.0)
        ]
        
        score = sum(aspects)
        
        # Normalize score to be between -1 and 1
        score = max(min(score, 1.0), -1.0)
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating financial score: {e}")
        return 0.0

def calculate_intrinsic_value(financial_data: Dict, current_price: float) -> Dict:
    """Calculate intrinsic value using multiple valuation methods"""
    try:
        dcf_value = calculate_dcf_value(financial_data)
        relative_value = calculate_relative_value(financial_data)
        graham_value = calculate_graham_number(financial_data)
        
        # Weight different valuation methods
        weights = {
            'dcf': 0.4,
            'relative': 0.3,
            'graham': 0.3
        }
        
        intrinsic_value = (
            dcf_value * weights['dcf'] +
            relative_value * weights['relative'] +
            graham_value * weights['graham']
        )
        
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value != 0 else 0
        
        return {
            'intrinsic_value': round(intrinsic_value, 2),
            'dcf_value': round(dcf_value, 2),
            'relative_value': round(relative_value, 2),
            'graham_value': round(graham_value, 2),
            'margin_of_safety': round(margin_of_safety, 2),
            'current_price': round(current_price, 2)
        }
        
    except Exception as e:
        logger.error(f"Error calculating intrinsic value: {e}")
        return {
            'intrinsic_value': 0.0,
            'dcf_value': 0.0,
            'relative_value': 0.0,
            'graham_value': 0.0,
            'margin_of_safety': 0.0,
            'current_price': current_price
        }

def calculate_dcf_value(financial_data: Dict) -> float:
    """Calculate intrinsic value using Discounted Cash Flow method"""
    try:
        cash_flow = financial_data.get('cash_flow', {}).get('freeCashFlow', 0.0)
        growth_rate = financial_data.get('income_statement', {}).get('revenueGrowth', 0.0)
        discount_rate = 0.1  # 10% discount rate
        terminal_growth_rate = 0.025  # 2.5% terminal growth rate
        
        if cash_flow == 0 or growth_rate == 0:
            return 0.0
            
        # Project cash flows for 5 years
        cash_flows = []
        for year in range(1, 6):
            future_cash_flow = cash_flow * (1 + growth_rate) ** year
            discounted_cash_flow = future_cash_flow / (1 + discount_rate) ** year
            cash_flows.append(discounted_cash_flow)
        
        # Calculate terminal value
        final_cash_flow = cash_flow * (1 + growth_rate) ** 5
        terminal_value = final_cash_flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
        
        # Sum up discounted cash flows and terminal value
        dcf_value = sum(cash_flows) + discounted_terminal_value
        
        return dcf_value if dcf_value > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating DCF value: {e}")
        return 0.0

def calculate_relative_value(financial_data: Dict) -> float:
    """Calculate intrinsic value using relative valuation"""
    try:
        eps = financial_data.get('income_statement', {}).get('eps', 0.0)
        industry_pe = 15.0  # Default=15.0
        
        if eps == 0:
            return 0.0
            
        relative_value = eps * industry_pe
        
        return relative_value if relative_value > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating relative value: {e}")
        return 0.0

def calculate_graham_number(financial_data: Dict) -> float:
    """Calculate intrinsic value using Benjamin Graham's formula"""
    try:
        eps = financial_data.get('income_statement', {}).get('eps', 0.0)
        book_value = financial_data.get('balance_sheet', {}).get('bookValuePerShare', 0.0)
        
        if eps == 0 or book_value == 0:
            return 0.0
            
        graham_number = (22.5 * eps * book_value) ** 0.5
        
        return graham_number if graham_number > 0 else 0.0
        
    except Exception as e:
        logger.error(f"Error calculating Graham number: {e}")
        return 0.0
