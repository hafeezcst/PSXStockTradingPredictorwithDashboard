import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Import the frozen module
        import draw_indicator_trend_lines_v1_1_20250412 as analysis
        
        # Run the analysis
        dashboard_df = analysis.generate_stock_dashboard()
        if not dashboard_df.empty:
            # Generate recommendations
            portfolio_recommendations = analysis.generate_portfolio_recommendations(dashboard_df)
            
            # Create output directory
            output_dir = Path('outputs/recommendations')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write recommendations with UTF-8 encoding
            output_file = output_dir / f'portfolio_recommendations_{datetime.now().strftime("%Y%m%d")}.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(portfolio_recommendations)
            logging.info(f"Portfolio recommendations saved to {output_file}")
            
    except Exception as e:
        logging.error(f"Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 