[project_readme.md](https://github.com/user-attachments/files/33138111/project_readme.md)
# 📈 AI Equity Analyst & Autonomous Investment Committee

An institutional-grade equity analysis platform and automated market breakout scanner designed specifically for Indian Equities (NSE/BSE). Powered by **Streamlit**, **CrewAI**, **Google Gemini**, and **yfinance**.

## 🌟 Key Features

### 1. 🔍 Comprehensive Equity Research Dashboard

* **Dynamic Resolution:** Intelligent ticker search that automatically detects typos and resolves company names to their respective NSE/BSE symbols (e.g., `Wipro` $\rightarrow$ `WIPRO.NS`).
* **Financial & Valuation Analysis:** Real-time metrics including P/E, P/B, ROE, Debt-to-Equity, and multi-year financials formatted natively in Crores ($\text{₹ Cr}$) and Lakhs ($\text{₹ L}$).
* **Peer Benchmarking:** Automated multi-company comparison against relevant industry peers.
* **Institutional Ownership:** Scrapes SEBI-compliant shareholding patterns (Promoters, FIIs, DIIs, Retail Float) via Screener.in.
* **Technical & Performance Indicators:** 52-week ranges, 1M/3M/6M/1Y returns, RSI, 50/200-day SMAs, and volume momentum signals.

### 2. 🏛️ Multi-Agent AI Investment Committee

Orchestrated using **CrewAI** and powered by Google Gemini, the platform runs an autonomous 4-agent committee debate to evaluate equities from diverse financial angles:

* **Growth Equity Analyst (Bull):** Analyzes secular business tailwinds, competitive moats, market TAM expansion, and generates a structured 5-point buy thesis.
* **Risk & Solvency Analyst (Bear):** Evaluates debt exposure, forensic risks, and runs an automated **Altman Z-Score** calculation tool to detect bankruptcy hazards.
* **Technical & Momentum Analyst:** Executes algorithmic checks on RSI overbought/oversold levels, moving average trend alignments, and volume spikes to identify optimal entry/exit windows.
* **Chief Investment Officer (Arbiter):** Synthesizes the Bull, Bear, and Technical findings, runs a dynamic **Discounted Cash Flow (DCF)** valuation model based on customizable WACC and terminal growth sliders, and issues a final institutional **BUY, HOLD, or SELL** rating.

### 3. 📥 PDF Memorandum Generator

* Automatically exports the final synthesized committee deliberation, valuation assumptions, and investment verdict into a publication-ready corporate PDF document.

### 4. 🚀 Market-Wide Breakout Scanner (NSE)

* **Quantitative Pre-filtering:** Analyzes the broad Nifty universe using rolling volume anomalies, moving-average breakouts, and price action metrics.
* **AI Catalyst Verification:** Evaluates news catalysts, earnings sentiment, and risk factors for top setups, assigning a probability score and trade recommendation (`BUY`, `WATCH`, or `SELL`).

## 🏗️ Project Architecture

```
AI_EQUITY_ANALYST/
│
├── app.py                 # Streamlit interactive web dashboard
├── scanner_engine.py      # Quantitative + AI market breakout scanner
├── main.py                # Standalone CLI entrypoint for analysis
├── debate_crew.py         # CrewAI agent roles, tasks, and valuation tools (DCF, Altman Z)
├── market_data.py         # Financial data retrieval, formatting, and web scraping
├── guardrails.py          # Output validation and sanitization
├── pdf_generator.py       # Corporate PDF memorandum exporter (ReportLab)
├── requirements.txt       # Project dependencies
├── .env.example           # Template for environment configuration
└── .gitignore             # Excluded files and secret protection
```

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10 to 3.12 installed on your machine.
* A Google AI Studio API key ([Get a Gemini API Key](https://aistudio.google.com/)).

### 2. Clone the Repository

```bash
git clone https://github.com/deevjain07/AI_EQUITY_ANALYST.git
cd AI_EQUITY_ANALYST
```

### 3. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Open `.env` and insert your Gemini API Key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

## 🖥️ Running the Application

To ensure all dashboard sections are fully loaded—especially the **Market Breakout Scanner** tab—follow this execution order:

### Step 1: Run the Market Breakout Scanner

Run the standalone scanner engine first. It screens the Nifty universe for volume and price breakouts, evaluates sentiment using Google Gemini, and writes the output to `scanner_results.json`:

```bash
python scanner_engine.py
```

### Step 2: Launch the Streamlit Dashboard

Once the scan output is generated, start the interactive dashboard:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Both the **Individual Company Research** and **Market Breakout Scanner** tabs will be fully populated and ready for analysis.

---

### Alternative: Standalone CLI Analysis

If you want to run the multi-agent committee debate for a single company directly from the command line without opening the web interface:

```bash
python main.py
```

## ⚙️ Custom Valuation Engine

The platform integrates a dynamic DCF model:

$$
\text{Enterprise Value} = \sum_{t=1}^{5} \frac{\text{FCF}_t}{(1 + \text{WACC})^t} + \frac{\text{Terminal Value}}{(1 + \text{WACC})^5}
$$

$$
\text{Terminal Value} = \frac{\text{FCF}_5 \times (1 + g)}{\text{WACC} - g}
$$

Users can adjust the **WACC (Discount Rate)** and **Growth Rate ($g$)** in real time via the Streamlit sidebar sliders to recalculate intrinsic value on the fly.

## 🛡️ License & Disclaimer

*Disclaimer: This software is designed for educational and research purposes only. It does not constitute official financial advice or SEBI-registered investment advisory recommendations. Always conduct independent research before deploying capital in financial markets.*
