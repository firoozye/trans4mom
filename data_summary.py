import pandas as pd
import numpy as np

def generate_data_summary(path):
    df = pd.read_parquet(path)
    summary = []
    
    for sym, group in df.groupby('symbol'):
        start = group.index.min()
        end = group.index.max()
        count = len(group)
        missing = group['close'].isnull().sum()
        summary.append({
            'Symbol': sym,
            'Start Date': start.date(),
            'End Date': end.date(),
            'Rows': count,
            'Missing': missing
        })
        
    summary_df = pd.DataFrame(summary).sort_values('Start Date')
    return summary_df

if __name__ == "__main__":
    path = 'data/processed/futures_1990_present.parquet'
    summary = generate_data_summary(path)
    # Manual table printing
    print(f"{'Symbol':<10} | {'Start':<12} | {'End':<12} | {'Rows':<6}")
    print("-" * 50)
    for _, row in summary.iterrows():
        print(f"{row['Symbol']:<10} | {str(row['Start Date']):<12} | {str(row['End Date']):<12} | {row['Rows']:<6}")
    
    print(f"\nTotal Symbols: {len(summary)}")
    print(f"Earliest Data: {summary['Start Date'].min()}")
    print(f"Latest Data: {summary['End Date'].max()}")
