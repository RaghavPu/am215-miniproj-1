import pandas as pd


# create an abstract base class for a ranker model
class RankerModel:
    def __init__(self):
        pass

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_probability_of_win(self, team_a: str, team_b: str) -> float:
        pass
    
    def reset(self):
        """Reset the model to initial state"""
        pass