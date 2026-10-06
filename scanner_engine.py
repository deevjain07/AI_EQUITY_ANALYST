import os
import json
from datetime import datetime, time
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure .env variables are loaded for standalone execution
load_dotenv()

# Setup IST timezone
try:
    from zoneinfo import ZoneInfo
    ist_tz = ZoneInfo("Asia/Kolkata")
except ImportError:
    import pytz
    ist_tz = pytz.timezone("Asia/Kolkata")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_nifty_universe() -> list:
    """Fetches broad NSE tickers with session support."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        res = session.get(url, headers=headers, timeout=10)
        if res.status_code == 200 and "Symbol" in res.text:
            df = pd.read_csv(io.StringIO(res.text))
            symbols = [f"{sym.strip()}.NS" for sym in df["Symbol"].dropna().tolist()]
            print(f"Loaded {len(symbols)} stocks from Nifty 500.")
            return symbols
    except Exception:
        pass

    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
        "LT.NS", "HINDUNILVR.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS",
        "TITAN.NS", "SUNPHARMA.NS", "ADANIENT.NS", "WIPRO.NS", "NESTLEIND.NS",
        "TRENT.NS", "BEL.NS", "HAL.NS", "DIXON.NS", "POLYCAB.NS", "PERSISTENT.NS",
        "COFORGE.NS", "BSE.NS", "CDSL.NS", "ZOMATO.NS", "JIOFIN.NS", "SUZLON.NS"
    ]


def run_quantitative_prefilter(tickers: list) -> tuple:
    print(f"Downloading market data for {len(tickers)} stocks...")
    
    data = yf.download(
        tickers=tickers, period="2mo", interval="1d",
        threads=True, progress=False
    )
    if data.empty:
        return [], "Unknown"

    flagged = []
    now_ist = datetime.now(ist_tz)
    market_closed_today = now_ist.time() >= time(15, 30)
    
    if market_closed_today:
        session_mode = "Today's Close (Completed Session)"
    else:
        session_mode = "Yesterday's Close (Market Currently Open)"
        
    print(f"Analysis Window: {session_mode}")

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.levels[0]:
                    close_s = data["Close"][ticker].dropna()
                    vol_s = data["Volume"][ticker].dropna()
                else:
                    close_s = data[ticker]["Close"].dropna()
                    vol_s = data[ticker]["Volume"].dropna()
            else:
                close_s = data["Close"].dropna()
                vol_s = data["Volume"].dropna()

            # Date slicing: drop partial day when running before 3:30 PM IST
            if not market_closed_today:
                today_date = now_ist.date()
                mask = close_s.index.date < today_date
                close_s = close_s[mask]
                vol_s = vol_s[mask]
                
            if len(close_s) < 22:
                continue

            current_price = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            day_return_pct = ((current_price - prev_close) / prev_close) * 100.0

            raw_vol = float(vol_s.iloc[-1])
            avg_vol_20 = float(vol_s.iloc[-21:-1].mean())

            if avg_vol_20 <= 0 or np.isnan(avg_vol_20):
                continue

            vol_ratio = round(raw_vol / avg_vol_20, 2)
            sma_20 = float(close_s.rolling(window=min(20, len(close_s))).mean().iloc[-1])
            high_20 = float(close_s.iloc[-min(21, len(close_s)):-1].max())
            is_breakout = current_price >= (high_20 * 0.985)

            if vol_ratio >= 1.5 and day_return_pct >= 0.8 and current_price >= sma_20:
                score = (vol_ratio * 12.0) + (day_return_pct * 8.0) + (15.0 if is_breakout else 0.0)
                flagged.append({
                    "ticker": ticker,
                    "symbol": ticker.replace(".NS", "").replace(".BO", ""),
                    "price": round(current_price, 2),
                    "day_change_pct": round(day_return_pct, 2),
                    "vol_ratio": vol_ratio,
                    "is_breakout": is_breakout,
                    "quant_score": round(score, 1),
                    "as_of_date": close_s.index[-1].strftime("%d %b %Y")
                })
        except Exception:
            continue

    flagged.sort(key=lambda x: x["quant_score"], reverse=True)
    selected = flagged[:12]
    print(f"Qualified {len(selected)} momentum candidates for AI review.")
    return selected, session_mode


def fetch_candidate_news(ticker: str) -> list:
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        headlines = [
            item.get("content", item).get("title")
            for item in news_items[:3]
            if item.get("content", item).get("title")
        ]
        return headlines
    except Exception:
        return []


def analyze_with_ai(candidates: list, session_mode: str) -> list:
    if not GEMINI_API_KEY:
        print("Missing GEMINI_API_KEY in environment. Falling back to default scoring.")
        for c in candidates:
            c["breakout_probability"] = min(95, int(c["quant_score"]))
            c["action"] = "WATCH"
            c["primary_catalyst"] = "Volume expansion setup"
            c["key_risk"] = "Index volatility"
        return candidates

    model = genai.GenerativeModel("gemini-2.5-flash")

    for c in candidates:
        headlines = fetch_candidate_news(c["ticker"])
        c["headlines"] = headlines

        prompt = f"""
You are an institutional momentum analyst for the Indian National Stock Exchange (NSE).
Analyze this candidate based on {session_mode} data:
Ticker: {c['ticker']} | Closing Price: ₹{c['price']} | Day Return: +{c['day_change_pct']}%
Relative Volume: {c['vol_ratio']}x 20D Avg | 20D Breakout: {'YES' if c['is_breakout'] else 'NO'}
Headlines: {json.dumps(headlines)}

Instructions:
Do not default to WATCH. If the setup is fundamentally strong with a positive catalyst, issue a BUY. If the volume spike appears to be a retail trap, exhaustion spike, or has adverse headline risk, issue a SELL. Otherwise, issue WATCH.

Return STRICT JSON:
{{
  "breakout_probability": <integer 0-100>,
  "action": "<BUY | SELL | WATCH>",
  "primary_catalyst": "<10-word summary of setup driver>",
  "key_risk": "<1-sentence key downside risk>"
}}
"""
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json", "temperature": 0.2}
            )
            parsed = json.loads(response.text)
            c["breakout_probability"] = parsed.get("breakout_probability", 50)
            c["action"] = parsed.get("action", "WATCH")
            c["primary_catalyst"] = parsed.get("primary_catalyst", "Momentum continuation")
            c["key_risk"] = parsed.get("key_risk", "Profit booking at resistance")
        except Exception:
            c["breakout_probability"] = min(92, int(c["quant_score"]))
            c["action"] = "WATCH"
            c["primary_catalyst"] = "Volume surge"
            c["key_risk"] = "Market pullback"

    candidates.sort(key=lambda x: x.get("breakout_probability", 0), reverse=True)
    return candidates


def main():
    print(f"\n=== Starting NSE Breakout Scanner ({datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')}) ===")
    universe = get_nifty_universe()
    quant_candidates, session_mode = run_quantitative_prefilter(universe)

    if not quant_candidates:
        print("No breakout setups detected matching the criteria.")
        scored = []
    else:
        scored = analyze_with_ai(quant_candidates, session_mode)

    results = {
        "last_updated": datetime.now(ist_tz).strftime("%d %b %Y, %I:%M %p IST"),
        "session_mode": session_mode,
        "candidates": scored
    }

    with open("scanner_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Scanner finished. Saved {len(scored)} candidates to scanner_results.json.\n")


if __name__ == "__main__":
    main()