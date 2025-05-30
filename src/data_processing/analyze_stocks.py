import pandas as pd
import logging
from fair_value_calculator import FairValueCalculator
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_stocks_from_excel():
    """Analyze stocks listed in the Excel file."""
    try:
        # Initialize the calculator
        calculator = FairValueCalculator()
        
        # Read the Excel file
        excel_path = os.path.join('src', 'data_processing', 'psxsymbols.xlsx')
        logger.info(f"Reading stock symbols from {excel_path}, sheet 'KMI100'")
        df = pd.read_excel(excel_path, sheet_name='KMI100')
        
        # Print the contents of the KMI100 sheet
        print("\nKMI100 Sheet Contents:")
        print("=" * 50)
        print(df.head())  # Print first 5 rows
        print("\nTotal rows:", len(df))
        print("\nColumns:", df.columns.tolist())
        
        # Get the column name that contains stock symbols
        symbol_column = df.columns[0]  # Assuming symbols are in the first column
        logger.info(f"Found {len(df)} stocks to analyze from KMI100")
        
        # Analyze each stock
        results = []
        for symbol in df[symbol_column]:
            try:
                logger.info(f"Analyzing {symbol}...")
                analysis = calculator.analyze_stock(symbol)
                
                if analysis:
                    results.append({
                        'symbol': symbol,
                        'recommendation': analysis['recommendation']['action'],
                        'confidence': analysis['recommendation']['confidence'],
                        'technical_score': analysis['key_metrics']['technical_score'],
                        'financial_score': analysis['key_metrics']['financial_score'],
                        'overall_score': analysis['key_metrics']['overall_score']
                    })
                    logger.info(f"Analysis completed for {symbol}")
                else:
                    logger.warning(f"No analysis available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to Excel
        output_path = os.path.join('data', 'exports', 'kmi100_analysis_results.xlsx')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df.to_excel(output_path, index=False)
        logger.info(f"Analysis results saved to {output_path}")
        
        # Print summary
        print("\nKMI100 Analysis Summary:")
        print("=" * 50)
        print(f"Total stocks analyzed: {len(results)}")
        print("\nRecommendation Distribution:")
        print(results_df['recommendation'].value_counts())
        print("\nTop 5 Stocks by Overall Score:")
        print(results_df.nlargest(5, 'overall_score')[['symbol', 'overall_score', 'recommendation']])
        
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}")

if __name__ == "__main__":
    analyze_stocks_from_excel() 