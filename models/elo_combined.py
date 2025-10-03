"""
Combined Elo Rating System with Goal Differential and Time Weighting

This model combines both:
1. Goal differential weighting (bigger wins matter more)
2. Time-based weighting (recent matches matter more)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .elo import EloRatingSystem


class EloCombined(EloRatingSystem):
    """
    Elo system with both goal differential and time-based weighting.
    
    Combines multipliers from:
    - Goal differential (logarithmic scale)
    - Time decay (exponential recency weighting)
    """
    
    def __init__(
        self, 
        k_factor: float = 32, 
        initial_rating: float = 1500,
        goal_diff_importance: float = 1.0,
        time_decay_rate: float = 0.5
    ):
        """
        Args:
            k_factor: Base K-factor
            initial_rating: Starting rating
            goal_diff_importance: Weight for goal differential (0 = ignore, 1 = standard)
            time_decay_rate: Weight for recency (0 = all equal, 1 = strong decay)
        """
        super().__init__(k_factor, initial_rating)
        self.goal_diff_importance = goal_diff_importance
        self.time_decay_rate = time_decay_rate
        self.first_match_date = None
        self.last_match_date = None
    
    def calculate_goal_diff_multiplier(self, goal_diff: int) -> float:
        """Calculate multiplier based on goal differential."""
        if goal_diff <= 1:
            return 1.0
        
        multiplier = 1.0 + np.log2(goal_diff) * self.goal_diff_importance
        max_multiplier = 3.0
        return min(multiplier, max_multiplier)
    
    def calculate_time_multiplier(self, match_date: pd.Timestamp) -> float:
        """Calculate multiplier based on match recency."""
        if self.time_decay_rate == 0 or self.first_match_date is None or self.last_match_date is None:
            return 1.0
        
        total_days = (self.last_match_date - self.first_match_date).days
        
        if total_days == 0:
            return 1.0
        
        days_from_latest = (self.last_match_date - match_date).days
        normalized_age = days_from_latest / total_days
        
        multiplier = np.exp(-self.time_decay_rate * normalized_age)
        
        return multiplier
    
    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches with combined weighting."""
        df = matches_df.sort_values('date').copy()
        
        # Set date range
        self.first_match_date = df['date'].min()
        self.last_match_date = df['date'].max()
        
        # Initialize columns
        df['home_elo_before'] = 0.0
        df['away_elo_before'] = 0.0
        df['home_elo_after'] = 0.0
        df['away_elo_after'] = 0.0
        df['home_elo_change'] = 0.0
        df['away_elo_change'] = 0.0
        df['goal_diff_multiplier'] = 0.0
        df['time_multiplier'] = 0.0
        df['combined_multiplier'] = 0.0
        
        # Process each match
        for idx, row in df.iterrows():
            result = self.process_match(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=row['home_team_score'],
                away_score=row['away_team_score'],
                date=row['date']
            )
            
            df.loc[idx, 'home_elo_before'] = result['home_rating_before']
            df.loc[idx, 'away_elo_before'] = result['away_rating_before']
            df.loc[idx, 'home_elo_after'] = result['home_rating_after']
            df.loc[idx, 'away_elo_after'] = result['away_rating_after']
            df.loc[idx, 'home_elo_change'] = result['home_rating_change']
            df.loc[idx, 'away_elo_change'] = result['away_rating_change']
            df.loc[idx, 'goal_diff_multiplier'] = result.get('goal_diff_multiplier', 1.0)
            df.loc[idx, 'time_multiplier'] = result.get('time_multiplier', 1.0)
            df.loc[idx, 'combined_multiplier'] = result.get('combined_multiplier', 1.0)
        
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
        """Process a match with combined weighting."""
        old_home_rating = self.get_rating(home_team)
        old_away_rating = self.get_rating(away_team)
        
        # Calculate goal differential
        goal_diff = abs(home_score - away_score)
        
        # Determine outcome
        if home_score > away_score:
            home_actual_score = 1.0
        elif home_score < away_score:
            home_actual_score = 0.0 
        else:
            home_actual_score = 0.5
        
        # Calculate both multipliers
        goal_diff_multiplier = self.calculate_goal_diff_multiplier(goal_diff)
        
        time_multiplier = 1.0
        if date is not None:
            time_multiplier = self.calculate_time_multiplier(date)
        
        # Combine multipliers (multiply them together)
        combined_multiplier = goal_diff_multiplier * time_multiplier
        
        # Apply combined multiplier to K-factor
        effective_k = (k_factor if k_factor is not None else self.k_factor) * combined_multiplier
        
        # Update ratings
        new_home_rating, new_away_rating = self.update_ratings(
            home_team, away_team, home_actual_score, k_factor=effective_k
        )
        
        # Add to history
        self.rating_history[home_team].append({
            'date': date,
            'opponent': away_team,
            'rating': new_home_rating,
            'rating_change': new_home_rating - old_home_rating,
            'goal_diff': home_score - away_score,
            'goal_diff_multiplier': goal_diff_multiplier,
            'time_multiplier': time_multiplier,
            'combined_multiplier': combined_multiplier
        })
        self.rating_history[away_team].append({
            'date': date,
            'opponent': home_team,
            'rating': new_away_rating,
            'rating_change': new_away_rating - old_away_rating,
            'goal_diff': away_score - home_score,
            'goal_diff_multiplier': goal_diff_multiplier,
            'time_multiplier': time_multiplier,
            'combined_multiplier': combined_multiplier
        })
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'goal_diff': goal_diff,
            'goal_diff_multiplier': goal_diff_multiplier,
            'time_multiplier': time_multiplier,
            'combined_multiplier': combined_multiplier,
            'effective_k': effective_k,
            'home_rating_before': old_home_rating,
            'away_rating_before': old_away_rating,
            'home_rating_after': new_home_rating,
            'away_rating_after': new_away_rating,
            'home_rating_change': new_home_rating - old_home_rating,
            'away_rating_change': new_away_rating - old_away_rating,
        }
