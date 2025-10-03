"""
Elo Rating System with Time-Based Weighting

This model extends the basic Elo system by increasing the K-factor for more recent matches.
Recent games have more impact on ratings than older games.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .elo import EloRatingSystem


class EloTimeWeighted(EloRatingSystem):
    """
    Elo system where recent matches have higher weight.
    
    The K-factor increases exponentially as matches get more recent,
    giving more importance to recent performance.
    """
    
    def __init__(
        self, 
        k_factor: float = 32, 
        initial_rating: float = 1500,
        time_decay_rate: float = 0.5
    ):
        """
        Args:
            k_factor: Base K-factor for rating changes
            initial_rating: Starting rating for all teams
            time_decay_rate: How much to weight recent matches (0 = no decay, 1 = strong decay)
                           Higher values mean recent matches matter much more
        """
        super().__init__(k_factor, initial_rating)
        self.time_decay_rate = time_decay_rate
        self.first_match_date = None
        self.last_match_date = None
    
    def calculate_time_multiplier(self, match_date: pd.Timestamp) -> float:
        """
        Calculate K-factor multiplier based on how recent the match is.
        
        Uses exponential decay: older matches get lower multipliers.
        
        Args:
            match_date: Date of the match
            
        Returns:
            Multiplier for K-factor (between min_weight and 1.0)
            
        Formula:
            If time_decay_rate = 0: all matches weighted equally (multiplier = 1.0)
            If time_decay_rate > 0: multiplier = exp(-decay * normalized_age)
            where normalized_age is how far back the match is (0 = most recent, 1 = oldest)
        """
        if self.time_decay_rate == 0 or self.first_match_date is None or self.last_match_date is None:
            return 1.0
        
        # Calculate total time span
        total_days = (self.last_match_date - self.first_match_date).days
        
        if total_days == 0:
            return 1.0
        
        # Calculate how old this match is (0 = most recent, 1 = oldest)
        days_from_latest = (self.last_match_date - match_date).days
        normalized_age = days_from_latest / total_days
        
        # Exponential decay: recent matches get multiplier close to 1.0, old matches get lower
        # Using exp(-decay * age) gives smooth exponential decay
        multiplier = np.exp(-self.time_decay_rate * normalized_age)
        
        return multiplier
    
    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process multiple matches with time-based weighting.
        
        First pass: identify date range
        Second pass: process with time weights
        """
        # Sort by date
        df = matches_df.sort_values('date').copy()
        
        # Set date range for time weighting
        self.first_match_date = df['date'].min()
        self.last_match_date = df['date'].max()
        
        # Initialize result columns
        df['home_elo_before'] = 0.0
        df['away_elo_before'] = 0.0
        df['home_elo_after'] = 0.0
        df['away_elo_after'] = 0.0
        df['home_elo_change'] = 0.0
        df['away_elo_change'] = 0.0
        df['time_multiplier'] = 0.0
        
        # Process each match with time weighting
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
            df.loc[idx, 'time_multiplier'] = result.get('time_multiplier', 1.0)
        
        return df
    
    def process_match(
        self, 
        home_team: str, 
        away_team: str, 
        home_score: int, 
        away_score: int,
        date: Optional[pd.Timestamp] = None,
        k_factor: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Process a match with time-based K-factor weighting.
        """
        # Get ratings before update
        old_home_rating = self.get_rating(home_team)
        old_away_rating = self.get_rating(away_team)
        
        # Determine match outcome
        if home_score > away_score:
            home_actual_score = 1.0
        elif home_score < away_score:
            home_actual_score = 0.0 
        else:
            home_actual_score = 0.5
        
        # Calculate time-based multiplier
        time_multiplier = 1.0
        if date is not None:
            time_multiplier = self.calculate_time_multiplier(date)
        
        # Apply time-weighted K-factor
        effective_k = (k_factor if k_factor is not None else self.k_factor) * time_multiplier
        
        # Update ratings with time-weighted K-factor
        new_home_rating, new_away_rating = self.update_ratings(
            home_team, away_team, home_actual_score, k_factor=effective_k
        )
        
        # Add to team history
        self.rating_history[home_team].append({
            'date': date,
            'opponent': away_team,
            'rating': new_home_rating,
            'rating_change': new_home_rating - old_home_rating,
            'time_multiplier': time_multiplier
        })
        self.rating_history[away_team].append({
            'date': date,
            'opponent': home_team,
            'rating': new_away_rating,
            'rating_change': new_away_rating - old_away_rating,
            'time_multiplier': time_multiplier
        })
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'time_multiplier': time_multiplier,
            'effective_k': effective_k,
            'home_rating_before': old_home_rating,
            'away_rating_before': old_away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_rating_change': new_home_rating - old_home_rating,
            'away_rating_change': new_away_rating - old_away_rating,
        }
