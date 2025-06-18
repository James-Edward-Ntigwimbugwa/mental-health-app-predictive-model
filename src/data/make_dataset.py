# src/data/make_dataset.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from .external_data_collector import ExternalDataCollector

logger = logging.getLogger(__name__)

def main():
    """Collects and combines datasets from multiple sources"""
    logger.info('Creating comprehensive stress dataset')
    
    # Initialize data collector
    collector = ExternalDataCollector()
    
    # Combine datasets
    combined_df = collector.combine_all_datasets()
    
    # Save to processed data folder
    processed_dir = Path(__file__).resolve().parents[2] / 'data/processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / 'combined_stress_data.csv'
    combined_df.to_csv(output_path, index=False)
    
    logger.info(f'Dataset saved to {output_path} with {len(combined_df)} records')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()