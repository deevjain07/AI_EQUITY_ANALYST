import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from io import StringIO


# =====================================================================
# 1. FINANCIAL DATAFRAME FORMATTING (Clean Numbers & Crores)
# =====================================================================

def format_financial_dataframe(df: pd.DataFrame, currency_symbol: str = "₹") -> pd.DataFrame:
    """
    Cleans raw yfinance multi-year financials:
    - Eliminates scientific notation (e.g., 1.45e+10).
    - Converts large numbers into readable Crores (Cr) or Lakhs (L).
    - Preserves EPS, tax rates, and ratios as standard decimals.
    - Formats timestamp column headers into readable Fiscal Years (e.g., FY 2026).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    clean_df = df.copy()

    # 1. Format date column headers
    formatted_cols = []
    for col in clean_df.columns:
        if hasattr(col, "strftime"):
            formatted_cols.append(col.strftime("FY %Y (%b %d)"))
        else:
            formatted_cols.append(str(col)[:10])
    clean_df.columns = formatted_cols

    # 2. Value formatter
    def format_val(row_name: str, val):
        if pd.isna(val) or val is None:
            return "—"
        if not isinstance(val, (int, float, np.number)):
            return str(val)

        row_str = str(row_name).lower()

        # Ratio and per-share metric formatting
        if any(k in row_str for k in ["eps", "rate", "ratio", "per share", "factor"]):
            return f"{val:,.2f}"

        # Large currency formatting (Indian Crores & Lakhs)
        abs_val = abs(val)
        if abs_val >= 1e7:
            return f"{currency_symbol}{val / 1e7:,.2f} Cr"
        elif abs_val >= 1e5:
            return f"{currency_symbol}{val / 1e5:,.2f} L"
        elif abs_val >= 1e3:
            return f"{currency_symbol}{val:,.2f}"
        else:
            return f"{val:,.2f}"

    # Apply row-aware formatting across all cells
    formatted_df = pd.DataFrame(index=clean_df.index, columns=clean_df.columns)
    for row in clean_df.index:
        for col in clean_df.columns:
            formatted_df.loc[row, col] = format_val(row, clean_df.loc[row, col])

    return formatted_df


# =====================================================================
# 2. CORE STOCK DATA & FINANCIAL STATEMENTS
# =====================================================================

def get_stock_data(ticker: str) -> dict:
    """
    Fetches comprehensive stock overview, price history, ratios,
    and multi-year financial statements for a given ticker.
    """
    # Ensure standard NSE ticker format if no exchange is provided
    formatted_ticker = ticker.strip().upper()
    if not ("." in formatted_ticker or "^" in formatted_ticker):
        formatted_ticker = f"{formatted_ticker}.NS"

    stock = yf.Ticker(formatted_ticker)
    info = stock.info or {}

    # 1. Price History (1 Year)
    try:
        hist = stock.history(period="1y")
        if hist.empty and formatted_ticker.endswith(".NS"):
            # Fallback to BSE if NSE fails
            formatted_ticker = formatted_ticker.replace(".NS", ".BO")
            stock = yf.Ticker(formatted_ticker)
            info = stock.info or {}
            hist = stock.history(period="1y")
    except Exception:
        hist = pd.DataFrame()

    # 2. Multi-Year Financial Statements
    financials = stock.financials if hasattr(stock, "financials") else pd.DataFrame()
    balance_sheet = stock.balance_sheet if hasattr(stock, "balance_sheet") else pd.DataFrame()
    cashflow = stock.cashflow if hasattr(stock, "cashflow") else pd.DataFrame()

    # 3. Key Financial Ratios & Metrics
    current_price = info.get("currentPrice") or (hist["Close"].iloc[-1] if not hist.empty else 0.0)
    market_cap = info.get("marketCap", 0)
    pe_ratio = info.get("trailingPE") or info.get("forwardPE") or "N/A"
    pb_ratio = info.get("priceToBook", "N/A")
    roe = info.get("returnOnEquity")
    roe_formatted = f"{round(roe * 100, 2)}%" if roe is not None else "N/A"
    debt_to_equity = info.get("debtToEquity", "N/A")
    free_cashflow = info.get("freeCashflow", "N/A")
    revenue = info.get("totalRevenue", "N/A")
    profit_margin = info.get("profitMargins")
    margin_formatted = f"{round(profit_margin * 100, 2)}%" if profit_margin is not None else "N/A"

    metrics = {
        "Ticker": formatted_ticker,
        "Company Name": info.get("longName", formatted_ticker),
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "Current Price": current_price,
        "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52W Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Market Cap": market_cap,
        "Market Cap (Cr)": f"₹{market_cap / 1e7:,.2f} Cr" if market_cap else "N/A",
        "P/E Ratio": pe_ratio if pe_ratio == "N/A" else round(pe_ratio, 2),
        "P/B Ratio": pb_ratio if pb_ratio == "N/A" else round(pb_ratio, 2),
        "ROE": roe_formatted,
        "Debt to Equity": debt_to_equity if debt_to_equity == "N/A" else round(debt_to_equity, 2),
        "Profit Margin": margin_formatted,
        "Total Revenue": revenue,
        "Free Cash Flow": free_cashflow,
    }

    # 4. Formatted Display Tables
    formatted_financials = format_financial_dataframe(financials)
    formatted_balance_sheet = format_financial_dataframe(balance_sheet)
    formatted_cashflow = format_financial_dataframe(cashflow)

    # 5. Build AI Context String for Agents
    context_str = f"""
    --- COMPANY OVERVIEW ---
    Company: {metrics['Company Name']} ({metrics['Ticker']})
    Sector: {metrics['Sector']} | Industry: {metrics['Industry']}
    Current Price: ₹{metrics['Current Price']}
    Market Cap: {metrics['Market Cap (Cr)']}
    P/E Ratio: {metrics['P/E Ratio']} | P/B Ratio: {metrics['P/B Ratio']}
    ROE: {metrics['ROE']} | Profit Margin: {metrics['Profit Margin']}
    Debt to Equity: {metrics['Debt to Equity']}

    --- RECENT FINANCIAL PERFORMANCE SUMMARY ---
    """
    if not financials.empty:
        try:
            rev = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 'N/A'
            ni = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 'N/A'
            rev_str = f"₹{rev/1e7:,.2f} Cr" if isinstance(rev, (int, float)) else str(rev)
            ni_str = f"₹{ni/1e7:,.2f} Cr" if isinstance(ni, (int, float)) else str(ni)
            context_str += f"Latest Annual Revenue: {rev_str}\nLatest Annual Net Income: {ni_str}\n"
        except Exception:
            pass

    return {
        "metrics": metrics,
        "history": hist,
        "raw_financials": financials,
        "raw_balance_sheet": balance_sheet,
        "raw_cashflow": cashflow,
        "display_financials": formatted_financials,
        "display_balance_sheet": formatted_balance_sheet,
        "display_cashflow": formatted_cashflow,
        "ai_context_string": context_str,
    }


# =====================================================================
# 3. STOCK NEWS & MARKET CATALYSTS
# =====================================================================

def get_stock_news(ticker: str, max_items: int = 5) -> list:
    """
    Fetches recent news items and headlines for a given stock ticker.
    """
    formatted_ticker = ticker.strip().upper()
    if not ("." in formatted_ticker or "^" in formatted_ticker):
        formatted_ticker = f"{formatted_ticker}.NS"

    stock = yf.Ticker(formatted_ticker)
    news_items = []

    try:
        raw_news = stock.news or []
        for item in raw_news[:max_items]:
            content = item.get("content", item)
            title = content.get("title", "No Title")
            publisher = content.get("provider", {}).get("displayName") or content.get("publisher", "Market News")
            link = content.get("canonicalUrl", {}).get("url") or content.get("link", "#")
            pub_date = content.get("pubDate") or item.get("providerPublishTime")

            # Parse publication time
            if isinstance(pub_date, int):
                formatted_time = datetime.fromtimestamp(pub_date).strftime("%b %d, %Y")
            elif isinstance(pub_date, str):
                formatted_time = pub_date[:10]
            else:
                formatted_time = "Recent"

            news_items.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": formatted_time
            })
    except Exception as e:
        print(f"Warning: Failed to fetch news for {ticker}: {e}")

    return news_items


# =====================================================================
# 4. COMPETITOR & PEER BENCHMARKING
# =====================================================================

def get_competitor_data(ticker: str) -> pd.DataFrame:
    """
    Identifies major sector peers/competitors and returns a comparative table.
    """
    clean_sym = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Industry peer mapping for prominent Indian equity sectors
    peer_groups = {
        "HDFCBANK": ["ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
        "ICICIBANK": ["HDFCBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
        "SBIN": ["HDFCBANK.NS", "ICICIBANK.NS", "PNB.NS", "BANKBARODA.NS"],
        "TCS": ["INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
        "INFY": ["TCS.NS", "WIPRO.NS", "HCLTECH.NS", "LTIM.NS"],
        "RELIANCE": ["ONGC.NS", "BPCL.NS", "IOC.NS", "ADANIENT.NS"],
        "TATAMOTORS": ["M&M.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
        "NESTLEIND": ["HINDUNILVR.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS"],
        "HINDUNILVR": ["NESTLEIND.NS", "BRITANNIA.NS", "ITC.NS", "GODREJCP.NS"],
        "ITC": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "VBL.NS"],
    }

    target_ticker = f"{clean_sym}.NS"
    peer_list = peer_groups.get(clean_sym, ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"])
    
    all_tickers = [target_ticker] + [p for p in peer_list if p != target_ticker][:3]
    comparison_rows = []

    for sym in all_tickers:
        try:
            p_stock = yf.Ticker(sym)
            p_info = p_stock.info or {}
            
            mkt_cap = p_info.get("marketCap", 0)
            mkt_cap_str = f"₹{mkt_cap / 1e7:,.0f} Cr" if mkt_cap else "—"
            
            pe = p_info.get("trailingPE") or p_info.get("forwardPE")
            pe_str = f"{pe:.2f}" if isinstance(pe, (int, float)) else "—"
            
            pb = p_info.get("priceToBook")
            pb_str = f"{pb:.2f}" if isinstance(pb, (int, float)) else "—"
            
            roe = p_info.get("returnOnEquity")
            roe_str = f"{roe * 100:.2f}%" if isinstance(roe, (int, float)) else "—"

            comparison_rows.append({
                "Company": p_info.get("shortName", sym),
                "Ticker": sym,
                "Price": f"₹{p_info.get('currentPrice', '—')}",
                "Market Cap": mkt_cap_str,
                "P/E": pe_str,
                "P/B": pb_str,
                "ROE": roe_str
            })
        except Exception:
            continue

    if comparison_rows:
        return pd.DataFrame(comparison_rows)
    return pd.DataFrame()


# =====================================================================
# 5. SUPPLEMENTARY DATA (Financials & Shareholding)
# =====================================================================

def get_financial_statements(ticker: str, frequency: str = "Annual"):
    """
    Fetches historical financial statements (Income Statement, Balance Sheet, Cash Flow).
    Returns them cleanly formatted for Streamlit.
    Accepts frequency='Annual' or 'Quarterly'.
    """
    formatted_ticker = ticker.strip().upper()
    if not ("." in formatted_ticker or "^" in formatted_ticker):
        formatted_ticker = f"{formatted_ticker}.NS"
        
    stock = yf.Ticker(formatted_ticker)
    
    # Grab the raw data based on the requested frequency
    if frequency.lower() == 'quarterly':
        raw_fin = stock.quarterly_financials if hasattr(stock, 'quarterly_financials') else pd.DataFrame()
        raw_bs = stock.quarterly_balance_sheet if hasattr(stock, 'quarterly_balance_sheet') else pd.DataFrame()
        raw_cf = stock.quarterly_cashflow if hasattr(stock, 'quarterly_cashflow') else pd.DataFrame()
    else:
        raw_fin = stock.financials if hasattr(stock, 'financials') else pd.DataFrame()
        raw_bs = stock.balance_sheet if hasattr(stock, 'balance_sheet') else pd.DataFrame()
        raw_cf = stock.cashflow if hasattr(stock, 'cashflow') else pd.DataFrame()
    
    # Format them cleanly using our helper function
    fin = format_financial_dataframe(raw_fin)
    bs = format_financial_dataframe(raw_bs)
    cf = format_financial_dataframe(raw_cf)
    
    return fin, bs, cf

    
def get_shareholding_data(ticker: str):
    """
    Scrapes accurate SEBI-compliant shareholding data (Promoter, FII, DII, Public) 
    directly from Screener.in.
    """
    clean_ticker = ticker.replace('.NS', '').replace('.BO', '').strip().upper()
    breakdown = {}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = f"https://www.screener.in/company/{clean_ticker}/"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Target the correct section ID: 'shareholders'
        shareholding_section = soup.find('section', id='shareholders')
        
        if shareholding_section:
            table = shareholding_section.find('table')
            if table:
                df = pd.read_html(StringIO(str(table)))[0]
                latest_quarter_col = df.columns[-1]
                
                fii_dii_total = 0.0
                for index, row in df.iterrows():
                    investor_type = str(row[df.columns[0]]).strip()
                    try:
                        val_str = str(row[latest_quarter_col]).replace('%', '').strip()
                        val = float(val_str)
                    except (ValueError, TypeError):
                        val = 0.0
                    
                    if "Promoter" in investor_type:
                        breakdown["Promoters (Insiders)"] = val
                    elif "FII" in investor_type:
                        fii_dii_total += val
                    elif "DII" in investor_type:
                        fii_dii_total += val
                    elif "Public" in investor_type:
                        breakdown["Public (Retail Float)"] = val
                
                if fii_dii_total > 0:
                    breakdown["Institutions (FII & DII)"] = round(fii_dii_total, 2)
                    
    except Exception as e:
        print(f"\n[Scraper Error] Failed to fetch shareholding for {clean_ticker}: {e}\n")
    
    return breakdown, pd.DataFrame(), pd.DataFrame()