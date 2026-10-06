import os
import yfinance as yf
import pandas as pd
import numpy as np
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# =====================================================================
# 1. DEFINE FINANCIAL & TECHNICAL TOOLS (Pure Python & Pandas)
# =====================================================================

@tool("Altman_Z_Score_Calculator")
def calculate_altman_z_score(ticker: str) -> str:
    """Calculates the Altman Z-Score to estimate bankruptcy risk for a given stock ticker."""
    print(f"\n[TOOL EXECUTED] The Bear Agent is calculating Altman Z-Score for {ticker}...\n")
    
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        financials = stock.financials
        
        total_assets = bs.loc['Total Assets'].iloc[0]
        
        try:
            total_liabilities = bs.loc['Total Liabilities Net Minority Interest'].iloc[0]
        except KeyError:
            total_liabilities = bs.loc['Total Liabilities'].iloc[0]

        current_assets = bs.loc['Current Assets'].iloc[0]
        current_liabilities = bs.loc['Current Liabilities'].iloc[0]
        working_capital = current_assets - current_liabilities
        
        retained_earnings = bs.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in bs.index else 0
        
        ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else financials.loc['Operating Income'].iloc[0]
        total_revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else financials.loc['Operating Revenue'].iloc[0]
        
        market_cap = stock.info.get('marketCap')
        
        if not all([total_assets, total_liabilities, market_cap]):
            return "Z-Score Error: Missing critical balance sheet data."

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = total_revenue / total_assets
        
        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
        
        if z_score > 2.99:
            zone = "Safe Zone (Negligible Bankruptcy Risk)"
        elif 1.81 <= z_score <= 2.99:
            zone = "Grey Zone (Moderate Risk)"
        else:
            zone = "Distress Zone (High Bankruptcy Risk - Red Flag!)"
            
        return (
            f"Altman Z-Score Analysis for {ticker}:\n"
            f"- Z-Score: {round(z_score, 2)}\n"
            f"- Risk Assessment: {zone}\n"
            f"(Breakdown: X1={round(x1,2)}, X2={round(x2,2)}, X3={round(x3,2)}, X4={round(x4,2)}, X5={round(x5,2)})"
        )

    except Exception as e:
        return f"Altman Z-Score Calculation failed due to a missing metric in the filing: {str(e)}"


@tool("Technical_Indicator_Calculator")
def calculate_technical_indicators(ticker: str) -> str:
    """Calculates RSI, 50-day SMA, 200-day SMA, trend crossovers, and volume momentum."""
    print(f"\n[TOOL EXECUTED] The Technical Analyst is computing moving averages, RSI, and volume for {ticker}...\n")
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty or len(hist) < 50:
            return "Technical Analysis Error: Insufficient price history (requires at least 50 trading days)."

        close = hist['Close']
        volume = hist['Volume']
        
        # 1. Moving Averages
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1] if len(close) >= 200 else None
        current_price = close.iloc[-1]
        
        trend_status = "N/A"
        if sma_200:
            if current_price > sma_50 > sma_200:
                trend_status = "Bullish Uptrend (Price > 50 SMA > 200 SMA)"
            elif current_price < sma_50 < sma_200:
                trend_status = "Bearish Downtrend (Price < 50 SMA < 200 SMA)"
            else:
                trend_status = "Consolidation / Mixed Trend"

        # 2. 14-Day Relative Strength Index (RSI)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]
        
        rsi_zone = "Neutral"
        if rsi_14 >= 70:
            rsi_zone = "Overbought (High pullback risk)"
        elif rsi_14 <= 30:
            rsi_zone = "Oversold (Potential bounce candidate)"

        # 3. Volume Spikes (Current vs. 20-Day Average)
        avg_vol_20 = volume.rolling(window=20).mean().iloc[-1]
        curr_vol = volume.iloc[-1]
        vol_ratio = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0
        vol_status = "Elevated/Spike" if vol_ratio >= 1.5 else "Subdued/Normal"

        result = (
            f"Technical & Momentum Indicators for {ticker}:\n"
            f"- Current Price: ₹{round(current_price, 2)}\n"
            f"- 50-Day SMA: ₹{round(sma_50, 2)}\n"
            f"- 200-Day SMA: ₹{round(sma_200, 2) if sma_200 else 'Insufficient 200d data'}\n"
            f"- Primary Trend: {trend_status}\n"
            f"- 14-Day RSI: {round(rsi_14, 2)} ({rsi_zone})\n"
            f"- Volume Trend: {round(vol_ratio, 2)}x 20-day average ({vol_status})"
        )
        return result

    except Exception as e:
        return f"Technical calculation failed: {str(e)}"


# =====================================================================
# 2. ASSEMBLE THE 4-AGENT AI CREW
# =====================================================================

def create_investment_crew(financial_context: str, custom_wacc: float = 0.10, custom_growth: float = 0.05):
    """Creates the 4-agent committee (Bull, Bear, Technical, Arbiter) with custom DCF slider inputs."""
    
    # Dynamic DCF Tool bound to UI slider inputs
    @tool("DCF_Calculator")
    def calculate_dcf(ticker: str) -> str:
        """Calculates the Discounted Cash Flow (DCF) intrinsic value using customizable slider settings."""
        print(f"\n[TOOL EXECUTED] Calculating DCF for {ticker} with WACC={custom_wacc}, Growth={custom_growth}...\n")
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fcf = info.get('freeCashflow') 
            total_debt = info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0)
            shares_out = info.get('sharesOutstanding')
            current_price = info.get('currentPrice')

            if not fcf or not shares_out:
                return "DCF Error: Missing critical Free Cash Flow or Shares Outstanding data for this ticker."

            wacc = custom_wacc            
            short_term_growth = custom_growth 
            terminal_growth = 0.025  
            
            present_value_fcf = 0
            projected_fcf = fcf
            for year in range(1, 6):
                projected_fcf *= (1 + short_term_growth)
                discounted_fcf = projected_fcf / ((1 + wacc) ** year)
                present_value_fcf += discounted_fcf
                
            terminal_value = (projected_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
            present_value_tv = terminal_value / ((1 + wacc) ** 5)
            
            enterprise_value = present_value_fcf + present_value_tv
            equity_value = enterprise_value + total_cash - total_debt
            intrinsic_value = equity_value / shares_out
            
            valuation_gap = ((intrinsic_value - current_price) / current_price) * 100 if current_price else 0
            status = "UNDERVALUED" if intrinsic_value > current_price else "OVERVALUED"

            return (
                f"DCF Analysis for {ticker}:\n"
                f"- Current Market Price: ₹{current_price}\n"
                f"- Calculated Intrinsic Value: ₹{round(intrinsic_value, 2)}\n"
                f"- Status: {status} by {abs(round(valuation_gap, 2))}%\n"
                f"(Custom Assumptions: WACC={wacc*100}%, Short-Term Growth={short_term_growth*100}%)"
            )
        except Exception as e:
            return f"DCF Calculation failed due to an API error: {str(e)}"

    # Initialize Google AI Studio Flash LLM
    # Initialize Google AI Studio Flash LLM
    gemini_llm = LLM(
       model="gemini/gemini-3.1-flash-lite",
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        temperature=0.3,
        max_retries=5,
        timeout=60
    )

    # --- 1. AGENTS ---
    bull_agent = Agent(
        role='Growth Equity Analyst',
        goal='Identify positive business catalysts, competitive moats, and reasons why this stock is a strong BUY.',
        backstory='You are an optimistic growth equity investor who seeks high ROE compounders with expanding total addressable markets.',
        allow_delegation=False,
        verbose=True,
        llm=gemini_llm
    )

    bear_agent = Agent(
        role='Risk & Solvency Analyst',
        goal='Identify balance sheet risks, debt burdens, and reasons why this stock is a SELL.',
        backstory='You are a skeptical short-seller looking for balance sheet red flags, insolvency hazards, and revenue deceleration.',
        allow_delegation=False,
        verbose=True,
        llm=gemini_llm,
        tools=[calculate_altman_z_score]
    )

    technical_agent = Agent(
        role='Technical & Momentum Analyst',
        goal='Analyze moving average trends, RSI overbought/oversold status, and volume signals to assess execution timing.',
        backstory='You are a Chartered Market Technician (CMT). You ignore company hype and focus purely on price action, momentum exhaustion, support/resistance levels, and tactical entry points.',
        allow_delegation=False,
        verbose=True,
        llm=gemini_llm,
        tools=[calculate_technical_indicators]
    )

    arbiter_agent = Agent(
        role='Chief Investment Officer & Portfolio Manager',
        goal='Synthesize the fundamental 5-point bull/bear arguments with technical momentum and DCF valuation to issue a decisive BUY, HOLD, or SELL verdict.',
        backstory='You are an institutional fund manager. You recognize that even a great company is a bad investment at the wrong price or timing. You reconcile fundamental value with technical timing.',
        allow_delegation=False,
        verbose=True,
        llm=gemini_llm,
        tools=[calculate_dcf]
    )

    # --- 2. TASKS ---
    bull_task = Task(
        description=f'Write a comprehensive 5-point buy thesis for this stock based on this financial data: {financial_context}',
        expected_output='A structured 5-point bullish thesis detailing business catalysts, growth potential, and financial strength.',
        agent=bull_agent
    )

    bear_task = Task(
        description=f'Write a comprehensive 5-point sell thesis for this stock based on this financial data: {financial_context}. You MUST execute your Altman_Z_Score_Calculator tool to assess bankruptcy risk and include the explicit Z-Score in your report.',
        expected_output='A structured 5-point bearish thesis highlighting structural risks, headwinds, and the Altman Z-Score calculation.',
        agent=bear_agent
    )

    technical_task = Task(
        description=f'Execute your Technical_Indicator_Calculator tool to extract moving averages, RSI, and volume momentum for this stock. Provide a 3-point technical assessment explaining whether the current chart setup favors buying on a dip, taking profits, or staying on the sidelines.',
        expected_output='A technical momentum evaluation highlighting RSI conditions, 50/200 SMA trend alignment, and immediate entry timing recommendations.',
        agent=technical_agent
    )

    arbiter_task = Task(
        description='Review the 5-point Bull Thesis, 5-point Bear Thesis, and Technical Momentum Report. Run your DCF_Calculator tool to establish intrinsic value. Synthesize all perspectives and output a final justified verdict (BUY, HOLD, or SELL) with a clear target price and tactical entry guidance.',
        expected_output='A structured final Investment Committee Memorandum summarizing the Bull Case, Bear Case, Technical Outlook, DCF Intrinsic Value, and the Final Verdict (BUY/HOLD/SELL).',
        agent=arbiter_agent,
        context=[bull_task, bear_task, technical_task]
    )

    # --- 3. CREW ASSEMBLY ---
    investment_crew = Crew(
        agents=[bull_agent, bear_agent, technical_agent, arbiter_agent],
        tasks=[bull_task, bear_task, technical_task, arbiter_task],
        process=Process.sequential,
        max_rpm=4
    )
    
    return investment_crew