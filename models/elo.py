"""
Elo Rating System for Football Teams

This module implements an Elo rating system to calculate and track the relative
strength of football teams based on their match history.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .ranker_model import RankerModel

class EloRatingSystem(RankerModel):
    
    def __init__(self, k_factor: float = 32, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.rating_history: Dict[str, list] = {}
        
    def get_rating(self, team: str) -> float:
        # if this the first time we're seeing this team, set the initial rating
        if team not in self.ratings:
            self.ratings[team] = self.initial_rating
            self.rating_history[team] = []
        return self.ratings[team]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(
        self, 
        team_a: str, 
        team_b: str, 
        score_a: float,
        k_factor: Optional[float] = None
    ) -> Tuple[float, float]:
        k = k_factor if k_factor is not None else self.k_factor
        
        # Get current ratings
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a  # Expected scores sum to 1
        
        # Actual score for team B is the complement
        score_b = 1 - score_a
        
        # Update ratings
        new_rating_a = rating_a + k * (score_a - expected_a)
        new_rating_b = rating_b + k * (score_b - expected_b)
        
        # Store updated ratings
        self.ratings[team_a] = new_rating_a
        self.ratings[team_b] = new_rating_b
        
        return new_rating_a, new_rating_b
    
    def process_match(
        self, 
        home_team: str, 
        away_team: str, 
        home_score: int, 
        away_score: int,
        date: Optional[pd.Timestamp] = None,
        k_factor: Optional[float] = None
    ) -> Dict[str, float]:

        # get ratings
        old_home_rating = self.get_rating(home_team)
        old_away_rating = self.get_rating(away_team)
        
        # determine match outcome
        if home_score > away_score:
            home_actual_score = 1.0
        elif home_score < away_score:
            home_actual_score = 0.0 
        else:
            home_actual_score = 0.5  # when we get a draw
        
        new_home_rating, new_away_rating = self.update_ratings(
            home_team, away_team, home_actual_score, k_factor
        )
        
        # add to the team history (just for logs)
        self.rating_history[home_team].append({
            'date': date,
            'opponent': away_team,
            'rating': new_home_rating,
            'rating_change': new_home_rating - old_home_rating
        })
        self.rating_history[away_team].append({
            'date': date,
            'opponent': home_team,
            'rating': new_away_rating,
            'rating_change': new_away_rating - old_away_rating
        })
        

        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_rating_before': old_home_rating,
            'away_rating_before': old_away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_rating_change': new_home_rating - old_home_rating,
            'away_rating_change': new_away_rating - old_away_rating,
        }
    
    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:

        # Sort by date
        # we want to process in order so that later matches mean more than earlier matches
        df = matches_df.sort_values('date').copy()
        
        # Initialize result columns
        df['home_elo_before'] = 0.0
        df['away_elo_before'] = 0.0
        df['home_elo_after'] = 0.0
        df['away_elo_after'] = 0.0
        df['home_elo_change'] = 0.0
        df['away_elo_change'] = 0.0
        
        # Process each match
        for idx, row in df.iterrows():
            result = self.process_match(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=row['home_team_score'],
                away_score=row['away_team_score'],
                date=row['date']
            )
            
            # Update DataFrame with results
            df.loc[idx, 'home_elo_before'] = result['home_rating_before']
            df.loc[idx, 'away_elo_before'] = result['away_rating_before']
            df.loc[idx, 'home_elo_after'] = result['home_rating_after']
            df.loc[idx, 'away_elo_after'] = result['away_rating_after']
            df.loc[idx, 'home_elo_change'] = result['home_rating_change']
            df.loc[idx, 'away_elo_change'] = result['away_rating_change']
        
        return df
    
    def get_rankings(self, top_n: Optional[int] = None) -> pd.DataFrame:
        rankings = pd.DataFrame([
            {'team': team, 'elo_rating': rating, 'rank': 0}
            for team, rating in self.ratings.items()
        ])
        
        rankings = rankings.sort_values('elo_rating', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)
        rankings = rankings[['rank', 'team', 'elo_rating']]
        
        if top_n is not None:
            rankings = rankings.head(top_n)
        
        return rankings.reset_index(drop=True)
    
    def predict_match(self, team_a: str, team_b: str) -> Dict[str, float]:
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        
        prob_a_wins = self.expected_score(rating_a, rating_b)
        prob_b_wins = 1 - prob_a_wins
        
        return {
            'team_a': team_a,
            'team_b': team_b,
            'team_a_rating': rating_a,
            'team_b_rating': rating_b,
            'team_a_win_probability': prob_a_wins,
            'team_b_win_probability': prob_b_wins,
            'rating_difference': rating_a - rating_b
        }
    
    def get_probability_of_win(self, team_a: str, team_b: str) -> float:
        return self.predict_match(team_a, team_b)['team_a_win_probability']
    
    def reset(self):
        """Reset the Elo system to initial state"""
        self.ratings = {}
        self.rating_history = {}

if __name__ == "__main__":
    # Example usage
    from dataloader import load_international_matches
    df = load_international_matches()
    

    elo = EloRatingSystem(k_factor=32, initial_rating=1500)
    

    df_with_elo = elo.process_matches(df)
    rankings = elo.get_rankings(top_n=100)
    print(rankings.to_string(index=False))
    
    # Example prediction
    print("\n5. Example Match Prediction:")
    print("="*70)
    prediction = elo.predict_match("Brazil", "Argentina")
    print(f"Match: {prediction['team_a']} vs {prediction['team_b']}")
    print(f"{prediction['team_a']} rating: {prediction['team_a_rating']:.2f}")
    print(f"{prediction['team_b']} rating: {prediction['team_b_rating']:.2f}")
    print(f"{prediction['team_a']} win probability: {prediction['team_a_win_probability']:.2%}")
    print(f"{prediction['team_b']} win probability: {prediction['team_b_win_probability']:.2%}")
    
