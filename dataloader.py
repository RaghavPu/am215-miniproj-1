import pandas as pd
import numpy as np
from pathlib import Path


def load_international_matches(csv_path='data/international_matches.csv'):
    """
    Load international football matches data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing match data
        
    Returns:
        pd.DataFrame: DataFrame containing the match data with all features
        
    The dataset includes:
    - date: Match date
    - home_team, away_team: Team names
    - home_team_continent, away_team_continent: Continental affiliations
    - home_team_fifa_rank, away_team_fifa_rank: FIFA rankings
    - home_team_total_fifa_points, away_team_total_fifa_points: FIFA points
    - home_team_score, away_team_score: Match scores
    - tournament: Tournament name
    - city, country: Match location
    - neutral_location: Whether played at neutral venue
    - shoot_out: Whether match went to penalties
    - home_team_result: Win/Draw/Lose
    - Various team statistics (goalkeeper, defense, offense, midfield scores)
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Display basic info
    print(f"Loaded {len(df)} matches")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return df


def get_numpy_arrays(df, feature_columns=None):
    """
    Convert pandas DataFrame to numpy arrays for training.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        feature_columns (list): List of column names to extract. If None, uses numeric columns.
        
    Returns:
        np.ndarray: Numpy array of the selected features
    """
    if feature_columns is None:
        # Select only numeric columns by default
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return df[feature_columns].values


if __name__ == "__main__":
    # Example usage
    df = load_international_matches()
    print("\n" + "="*50)
    print("First few rows:")
    print(df.head())
    
    # Show numeric columns that could be used as features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n" + "="*50)
    print(f"Numeric columns available for modeling: {numeric_cols}")

