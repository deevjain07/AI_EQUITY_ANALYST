import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Import helper functions from your custom modules
from market_data import (
    get_stock_data, 
    get_stock_news, 
    get_competitor_data, 
    get_financial_statements, 
    get_shareholding_data
)
from debate_crew import create_investment_crew
from pdf_generator import generate_pdf_memorandum
from guardrails import validate_ai_output

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Equity Analyst & Investment Committee",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Equity Analyst & Multi-Agent Investment Committee")
st.markdown("Automated fundamental research, valuation modeling, and multi-perspective bull/bear debates for Indian equities (NSE/BSE).")

# =====================================================================
# SIDEBAR: CONFIGURATION & SEARCH
# =====================================================================
st.sidebar.header("🔍 Stock Selection")

# Smart Search & Typo Resolution Logic using yf.Search
user_input = st.sidebar.text_input("Enter Company Name or Ticker:", "Wipro")

def resolve_ticker(query: str):
    """Searches Yahoo Finance, handles typos, and returns matching symbols."""
    if not query:
        return None, "Please enter a search term."
    
    clean_query = query.strip()
    test_tickers = [clean_query.upper()]
    if not ("." in clean_query or "^" in clean_query):
        test_tickers.insert(0, f"{clean_query.upper()}.NS")
        
    # Try exact match first
    import yfinance as yf
    for sym in test_tickers:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if not hist.empty:
                return sym, f"Resolved: {sym}"
        except Exception:
            continue

    # Fallback to fuzzy search for typos or company names
    try:
        search_results = yf.Search(clean_query, max_results=5)
        quotes = search_results.quotes
        if not quotes:
            return None, f"No matching companies found for '{clean_query}'."
            
        options = {f"{q.get('longname', q.get('shortname', 'Unknown'))} ({q.get('symbol')})" : q.get('symbol') for q in quotes}
        return options, "multiple"
    except Exception as e:
        return None, f"Search error: {str(e)}"

resolved_ticker = None
if user_input:
    result, status = resolve_ticker(user_input)
    if status == "multiple":
        st.sidebar.warning("Multiple matches found. Select correct company:")
        selected_label = st.sidebar.selectbox("Matching Companies:", list(result.keys()))
        resolved_ticker = result[selected_label]
    elif result:
        resolved_ticker = result
    else:
        st.sidebar.error(status)

# Valuation Slider Settings
st.sidebar.header("⚙️ DCF Assumptions")
user_wacc = st.sidebar.slider("WACC (Discount Rate)", 0.05, 0.20, 0.11, 0.005, format="%.2f")
user_growth = st.sidebar.slider("Terminal / Short-Term Growth Rate", 0.01, 0.15, 0.06, 0.005, format="%.2f")

# =====================================================================
# MAIN DASHBOARD EXECUTION
# =====================================================================

# Global Tabs for App Navigation
main_tab_research, main_tab_scanner = st.tabs([
    "🔍 Individual Company Research", 
    "🚀 Market Breakout Scanner"
])

# ---------------------------------------------------------------------
# 1. INDIVIDUAL COMPANY RESEARCH TAB
# ---------------------------------------------------------------------
with main_tab_research:
    if resolved_ticker:
        st.sidebar.success(f"Active Ticker: **{resolved_ticker}**")
        
        # Fetch all stock data
        with st.spinner(f"Fetching live market data and filings for {resolved_ticker}..."):
            stock_data = get_stock_data(resolved_ticker)
            metrics = stock_data["metrics"]
            news_items = get_stock_news(resolved_ticker)
            competitors_df = get_competitor_data(resolved_ticker)
            fin_annual, bs_annual, cf_annual = get_financial_statements(resolved_ticker, frequency="Annual")
            fin_qtr, bs_qtr, cf_qtr = get_financial_statements(resolved_ticker, frequency="Quarterly")
            breakdown, major_holders, inst_holders = get_shareholding_data(resolved_ticker)
            
        # --- TOP METRICS OVERVIEW ---
        st.header(f"📊 {metrics['Company Name']} ({metrics['Ticker']})")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Format price cleanly
        current_price_str = f"₹{metrics['Current Price']:,.2f}" if isinstance(metrics['Current Price'], (int, float)) else "N/A"
        col1.metric("Current Price", current_price_str)
        
        # Explicitly show Market Cap with 'Cr' indication to prevent text cutoff
        col2.metric("Market Cap (₹ in Cr)", metrics['Market Cap (Cr)'])
        
        col3.metric("P/E Ratio", str(metrics['P/E Ratio']))
        col4.metric("ROE", str(metrics['ROE']))
        col5.metric("Debt to Equity", str(metrics['Debt to Equity']))
        
        # --- TABS FOR ORGANIZED VIEW ---
        tab_overview, tab_financials, tab_news, tab_committee = st.tabs([
            "Overview & Peers", 
            "Financial Statements", 
            "Catalysts & News", 
            "🏛️ AI Investment Committee"
        ])

        with tab_overview:
            # --- 1. PEER BENCHMARKING ---
            st.subheader("Industry Peer Benchmarking")
            if not competitors_df.empty:
                st.dataframe(competitors_df, use_container_width=True, hide_index=True)
            else:
                st.info("Peer benchmarking data currently unavailable.")

            # --- 2. PRICE TREND CHART ---
            st.subheader("Price Trend (1 Year)")
            if not stock_data["history"].empty:
                st.line_chart(stock_data["history"]["Close"], height=320)
            else:
                st.warning("Price history chart unavailable.")

            st.markdown("---")

            # --- 3. 52-WEEK RANGE & RETURN PERFORMANCE ---
            st.subheader("📈 Performance & Trading Range")
            hist = stock_data["history"]
            
            # Calculate returns if history exists
            ret_1m, ret_3m, ret_6m, ret_1y = "—", "—", "—", "—"
            if not hist.empty and len(hist) >= 20:
                latest_close = hist["Close"].iloc[-1]
                if len(hist) >= 22:
                    ret_1m = f"{((latest_close / hist['Close'].iloc[-22]) - 1) * 100:+.2f}%"
                if len(hist) >= 65:
                    ret_3m = f"{((latest_close / hist['Close'].iloc[-65]) - 1) * 100:+.2f}%"
                if len(hist) >= 125:
                    ret_6m = f"{((latest_close / hist['Close'].iloc[-125]) - 1) * 100:+.2f}%"
                ret_1y = f"{((latest_close / hist['Close'].iloc[0]) - 1) * 100:+.2f}%"

            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns(6)
            high_52 = metrics.get('52W High', 'N/A')
            low_52 = metrics.get('52W Low', 'N/A')
            
            r_col1.metric("52W High", f"₹{high_52:,.2f}" if isinstance(high_52, (int, float)) else "N/A")
            r_col2.metric("52W Low", f"₹{low_52:,.2f}" if isinstance(low_52, (int, float)) else "N/A")
            r_col3.metric("1-Month Return", ret_1m)
            r_col4.metric("3-Month Return", ret_3m)
            r_col5.metric("6-Month Return", ret_6m)
            r_col6.metric("1-Year Return", ret_1y)

            st.markdown("---")

            # --- 4. VALUATION & OWNERSHIP SUMMARY ---
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📑 Valuation & Trading Multiples")
                val_data = {
                    "Metric": [
                        "Price to Book (P/B)",
                        "Sector",
                        "Industry",
                        "Profit Margin",
                        "Free Cash Flow"
                    ],
                    "Value": [
                        str(metrics.get("P/B Ratio", "N/A")),
                        str(metrics.get("Sector", "N/A")),
                        str(metrics.get("Industry", "N/A")),
                        str(metrics.get("Profit Margin", "N/A")),
                        f"₹{metrics.get('Free Cash Flow', 0) / 1e7:,.2f} Cr" if isinstance(metrics.get('Free Cash Flow'), (int, float)) else "N/A"
                    ]
                }
                st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

            with col_right:
                st.subheader("👥 Shareholding Structure")
                if breakdown and sum(breakdown.values()) > 0:
                    sh_df = pd.DataFrame({
                        "Investor Category": breakdown.keys(),
                        "Holding (%)": list(breakdown.values())
                    })
                    
                    fig = px.pie(
                        sh_df, 
                        values="Holding (%)", 
                        names="Investor Category",
                        hole=0.4, 
                        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
                    )
                    
                    fig.update_traces(
                        textposition="inside", 
                        textinfo="percent", 
                        hovertemplate="<b>%{label}</b><br>Total Holding: <b>%{value}%</b><extra></extra>",
                        pull=[0.02, 0.02, 0.02], 
                        marker=dict(line=dict(color='#0e1117', width=2)) 
                    )
                    
                    fig.update_layout(
                        margin=dict(t=20, b=10, l=10, r=10),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif major_holders is not None and not major_holders.empty:
                    st.dataframe(major_holders, use_container_width=True)
                else:
                    st.info("Institutional breakdown not publicly available for this exchange ticker.")
                    
        with tab_financials:
            freq_choice = st.radio("Statement Frequency:", ["Annual", "Quarterly"], horizontal=True)
            
            st.subheader("Income Statement")
            st.dataframe(fin_annual if freq_choice == "Annual" else fin_qtr, use_container_width=True, height=350)
            
            st.subheader("Balance Sheet")
            st.dataframe(bs_annual if freq_choice == "Annual" else bs_qtr, use_container_width=True, height=350)
            
            st.subheader("Cash Flow Statement")
            st.dataframe(cf_annual if freq_choice == "Annual" else cf_qtr, use_container_width=True, height=350)

        with tab_news:
            st.subheader("Top Recent News & Announcements")
            if news_items:
                for item in news_items:
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    st.caption(f"Source: {item['publisher']} | Published: {item['published']}")
                    st.divider()
            else:
                st.info("No recent news articles found.")

        with tab_committee:
            st.subheader("🤖 Autonomous Multi-Agent Investment Committee")
            st.markdown("Your AI committee consists of a **Growth Analyst (Bull)**, **Risk Analyst (Bear)**, and a **Portfolio Manager (Arbiter)** running valuation checks.")

            if st.button("🚀 Run AI Investment Committee Debate", type="primary"):
                if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
                    st.error("Missing Google API Key! Please configure your GEMINI_API_KEY in your .env file.")
                else:
                    with st.spinner("AI Agents are analyzing balance sheets, calculating Altman Z-Scores, running DCF models, and debating..."):
                        try:
                            crew = create_investment_crew(
                                financial_context=stock_data["ai_context_string"],
                                custom_wacc=user_wacc,
                                custom_growth=user_growth
                            )
                            
                            debate_result = crew.kickoff()
                            validated_result = validate_ai_output(str(debate_result))
                            
                            st.success("Investment Committee Deliberation Complete!")
                            st.markdown("---")
                            st.markdown(validated_result)
                            
                            company_title = stock_data.get('metrics', {}).get('Company Name', resolved_ticker)
                            
                            pdf_data = generate_pdf_memorandum(
                                company_title, 
                                str(validated_result), 
                                user_wacc, 
                                user_growth
                            )
                            
                            st.download_button(
                                label="📥 Download Investment Memorandum (PDF)",
                                data=pdf_data,
                                file_name=f"{resolved_ticker}_Investment_Memo.pdf",
                                mime="application/pdf"
                            )
                            
                        except Exception as e:
                            st.error(f"An error occurred during agent execution: {str(e)}")
    else:
        st.info("👈 Please enter a stock ticker or company name in the sidebar to begin analysis.")


# ---------------------------------------------------------------------
# 2. MARKET BREAKOUT SCANNER TAB
# ---------------------------------------------------------------------
with main_tab_scanner:
    st.header("⚡ NSE Market-Wide Breakout Scanner")
    st.caption("Automated quantitative screener and AI catalyst evaluation for Nifty 500 equities. Flags volume anomalies, potential stock splits, upcoming bonus share issues, and undervalued momentum candidates.")

    data_file = "scanner_results.json"

    if not os.path.exists(data_file):
        st.info("Scanner data is not yet generated. Run `python scanner_engine.py` from your terminal to trigger the first market scan.")
    else:
        with open(data_file, "r") as f:
            scan_payload = json.load(f)

        st.caption(f"Last Scan Run: **{scan_payload.get('last_updated', 'Unknown')}** | Data Scope: **{scan_payload.get('session_mode', 'End of Day Data')}**")
        
        candidates = scan_payload.get("candidates", [])

        if not candidates:
            st.warning("No high-probability breakout setups met the quantitative thresholds in the latest scan.")
        else:
            # 1. Executive Summary Table
            table_rows = []
            for c in candidates:
                table_rows.append({
                    "Symbol": c["symbol"],
                    "Price (₹)": f"₹{c['price']}",
                    "Day Gain": f"+{c['day_change_pct']}%",
                    "Volume Spike": f"{c['vol_ratio']}x",
                    "Breakout Prob.": f"{c.get('breakout_probability', '—')}%",
                    "AI Recommendation": c.get("action", "WATCH"),
                    "Key Catalyst": c.get("primary_catalyst", "—")
                })

            df_summary = pd.DataFrame(table_rows)

            # Row-level color styling based on AI signal
            def style_signal_rows(row):
                action = str(row.get("AI Recommendation", "")).upper()
                if "BUY" in action:
                    bg_color = "rgba(39, 174, 96, 0.22)"   # Translucent Green
                elif "SELL" in action or "AVOID" in action:
                    bg_color = "rgba(231, 76, 60, 0.22)"  # Translucent Red
                else:
                    bg_color = "rgba(128, 128, 128, 0.12)" # Translucent Grey
                return [f"background-color: {bg_color}"] * len(row)

            styled_df = df_summary.style.apply(style_signal_rows, axis=1)

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("🔍 Detailed Catalyst & Technical Breakdown")

            # 2. Detailed Card Views
            for c in candidates:
                action_text = c.get("action", "WATCH")
                with st.expander(f"**{c['symbol']}** — Breakout Probability: {c.get('breakout_probability')}% ({action_text})"):
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Current Price", f"₹{c['price']}", f"+{c['day_change_pct']}%")
                    col_m2.metric("Volume Multiple", f"{c['vol_ratio']}x 20D Avg")
                    col_m3.metric("Breakout Setup", "Confirmed" if c.get("is_breakout") else "Consolidating")
                    col_m4.metric("AI Verdict", action_text)

                    st.markdown(f"**Primary Catalyst:** {c.get('primary_catalyst', 'N/A')}")
                    st.markdown(f"**Key Downside Risk:** {c.get('key_risk', 'N/A')}")

                    if c.get("headlines"):
                        st.markdown("**Recent News Headlines Analyzed:**")
                        for h in c["headlines"]:
                            st.write(f"- {h}")