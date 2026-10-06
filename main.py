import os
from dotenv import load_dotenv

# These are the functions we wrote in the other files!
from market_data import get_stock_data
from debate_crew import create_investment_crew

# Load the secret API key from the .env file
load_dotenv()

def main():
    # 1. Ask the user which stock they want to analyze
    # (e.g., 'RELIANCE.NS' for Reliance Industries or 'AAPL' for Apple)
    ticker = input("Enter a stock ticker: ")
    
    # 2. Fetch the data
    print("\n--- Step 1: Gathering financial data ---")
    financial_data = get_stock_data(ticker)
    
    # 3. Setup the AI Crew
    print("--- Step 2: Starting the AI debate ---")
    crew = create_investment_crew(financial_data)
    
    # 4. Kickoff the process
    result = crew.kickoff()
    
    # 5. Print the final result
    print("\n==================================")
    print("FINAL ARBITER VERDICT:")
    print("==================================")
    print(result)

# This line just tells Python to run the main() function when we start the file
if __name__ == "__main__":
    main()