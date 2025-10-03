import pandas as pd
import numpy as np
from models.elo import EloRatingSystem
from models.ranker_model import RankerModel
from typing import Dict, List, Tuple


def get_predicted_outcome(model: RankerModel, home_team: str, away_team: str, draw_threshold: float = 0.01) -> float:
    home_win_prob = model.get_probability_of_win(home_team, away_team)
    if home_win_prob > 0.5 + draw_threshold:
        return 'Home'
    elif home_win_prob < 0.5 - draw_threshold:
        return 'Away'
    else:
        return 'Draw'

def get_actual_outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return 'Home'
    elif away_score > home_score:
        return 'Away'
    else:
        return 'Draw'


def evaluate_predictions(
    predictions: List[str], 
    actuals: List[str],
    win_probabilities: List[float] = None
) -> Dict[str, float]:
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Overall accuracy
    accuracy = np.mean(predictions == actuals)
    
    # Accuracy excluding draws
    non_draw_mask = actuals != 'Draw'
    if non_draw_mask.sum() > 0:
        accuracy_no_draw = np.mean(predictions[non_draw_mask] == actuals[non_draw_mask])
    else:
        accuracy_no_draw = 0.0
    
    # Count statistics
    total = len(actuals)
    correct = np.sum(predictions == actuals)
    actual_draws = np.sum(actuals == 'Draw')
    predicted_draws = np.sum(predictions == 'Draw')
    
    # Draw prediction accuracy
    draw_pred_mask = predictions == 'Draw'
    if predicted_draws > 0:
        draw_precision = np.sum((predictions == 'Draw') & (actuals == 'Draw')) / predicted_draws
    else:
        draw_precision = 0.0
    
    # Brier score (error between predicted and actual probabilities)
    brier_score = None
    if win_probabilities is not None:
        binary_outcomes = (predictions == actuals).astype(float)
        brier_score = np.mean((np.array(win_probabilities) - binary_outcomes) ** 2)
    
    return {
        'accuracy': accuracy,
        'accuracy_no_draw': accuracy_no_draw,
        'correct': correct,
        'total': total,
        'actual_draws': actual_draws,
        'predicted_draws': predicted_draws,
        'draw_precision': draw_precision,
        'brier_score': brier_score
    }


def evaluate_on_test_period(
    model: RankerModel,
    test_df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[float], pd.DataFrame]:
    predictions = []
    actuals = []
    probabilities = []
    
    results_list = []
    
    for idx, row in test_df.iterrows():
        # Get win probability for home team
        home_win_prob = model.get_probability_of_win(row['home_team'], row['away_team'])
        
        # Make prediction (can be Home, Away, or Draw)
        prediction = get_predicted_outcome(model, row['home_team'], row['away_team'],
)
        
        # Get actual outcome
        actual = get_actual_outcome(row['home_team_score'], row['away_team_score'])
        
        predictions.append(prediction)
        actuals.append(actual)
        
        # Store probability of predicted outcome
        if prediction == 'Home':
            probabilities.append(home_win_prob)
        elif prediction == 'Away':
            probabilities.append(1 - home_win_prob)
        else:  # Draw
            # For draw predictions, use distance from 0.5 as confidence
            probabilities.append(1.0 - abs(home_win_prob - 0.5) / 0.5)
        
        # Store detailed results
        results_list.append({
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_score': row['home_team_score'],
            'away_score': row['away_team_score'],
            'home_win_prob': home_win_prob,
            'prediction': prediction,
            'actual': actual,
            'correct': prediction == actual
        })
    
    results_df = pd.DataFrame(results_list)
    return predictions, actuals, probabilities, results_df


def rolling_window_evaluation(
    df: pd.DataFrame,
    model: RankerModel,
    train_years: int = 4,
    test_years: int = 1,
    verbose: bool = True
) -> Dict:

    # Sort by date
    df = df.sort_values('date').copy()
    df['year'] = df['date'].dt.year
    
    # Get unique years
    years = sorted(df['year'].unique())
    
    if verbose:
        print(f"\nDataset spans {len(years)} years: {years[0]} to {years[-1]}")
        print(f"Training on {train_years} years, testing on {test_years} year(s)")
        print("="*80)
    
    # Create rolling windows
    all_metrics = []
    all_detailed_results = []
    
    window_start_idx = 0
    window_num = 1
    
    while window_start_idx + train_years + test_years <= len(years):
        # Define train and test years
        train_year_range = years[window_start_idx:window_start_idx + train_years]
        test_year_range = years[window_start_idx + train_years:window_start_idx + train_years + test_years]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"WINDOW {window_num}")
            print(f"{'='*80}")
            print(f"Train years: {train_year_range[0]}-{train_year_range[-1]}")
            print(f"Test years:  {test_year_range[0]}-{test_year_range[-1]}")
        
        # Split data
        train_df = df[df['year'].isin(train_year_range)].copy()
        test_df = df[df['year'].isin(test_year_range)].copy()
        
        if verbose:
            print(f"Train matches: {len(train_df)}")
            print(f"Test matches:  {len(test_df)}")
        
        # Reset model to fresh state for this window
        model.reset()
        
        # Train on training data
        model.process_matches(train_df)
        
        # Evaluate on test data
        if verbose:
            print(f"Evaluating on test period...")
        predictions, actuals, probabilities, detailed_results = evaluate_on_test_period(
            model, test_df
        )
        
        # Calculate metrics
        metrics = evaluate_predictions(predictions, actuals, probabilities)
        metrics['window'] = window_num
        metrics['train_years'] = f"{train_year_range[0]}-{train_year_range[-1]}"
        metrics['test_years'] = f"{test_year_range[0]}-{test_year_range[-1]}"
        
        if verbose:
            print(f"\nResults:")
            print(f"  Accuracy (overall):         {metrics['accuracy']:.2%}")
            print(f"  Accuracy (excluding draws): {metrics['accuracy_no_draw']:.2%}")
            print(f"  Correct predictions:        {metrics['correct']}/{metrics['total']}")
            print(f"  Actual draws in test set:   {metrics['actual_draws']}")
            print(f"  Predicted draws:            {metrics['predicted_draws']}")
            if metrics['predicted_draws'] > 0:
                print(f"  Draw precision:             {metrics['draw_precision']:.2%}")
            if metrics['brier_score'] is not None:
                print(f"  Brier Score:                {metrics['brier_score']:.4f}")
        
        all_metrics.append(metrics)
        
        # Add window info to detailed results
        detailed_results['window'] = window_num
        all_detailed_results.append(detailed_results)
        
        # Move to next window
        window_start_idx += train_years + test_years
        window_num += 1
    
    # Calculate average metrics
    if verbose:
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
    
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_accuracy_no_draw = np.mean([m['accuracy_no_draw'] for m in all_metrics])
    total_correct = sum([m['correct'] for m in all_metrics])
    total_predictions = sum([m['total'] for m in all_metrics])
    total_actual_draws = sum([m['actual_draws'] for m in all_metrics])
    total_predicted_draws = sum([m['predicted_draws'] for m in all_metrics])
    avg_draw_precision = np.mean([m['draw_precision'] for m in all_metrics if m['predicted_draws'] > 0]) if any(m['predicted_draws'] > 0 for m in all_metrics) else 0.0
    
    brier_scores = [m['brier_score'] for m in all_metrics if m['brier_score'] is not None]
    avg_brier = np.mean(brier_scores) if brier_scores else None
    
    if verbose:
        print(f"\nNumber of windows evaluated: {len(all_metrics)}")
        print(f"\nAverage Accuracy (overall):        {avg_accuracy:.2%}")
        print(f"Average Accuracy (excluding draws): {avg_accuracy_no_draw:.2%}")
        print(f"Total correct predictions:         {total_correct}/{total_predictions}")
        print(f"Total actual draws in test sets:   {total_actual_draws}")
        print(f"Total predicted draws:             {total_predicted_draws}")
        if total_predicted_draws > 0:
            print(f"Average draw precision:            {avg_draw_precision:.2%}")
        if avg_brier is not None:
            print(f"Average Brier Score:               {avg_brier:.4f}")
    
    # Combine all detailed results
    all_detailed_df = pd.concat(all_detailed_results, ignore_index=True)
    
    return {
        'avg_accuracy': avg_accuracy,
        'avg_accuracy_no_draw': avg_accuracy_no_draw,
        'total_correct': total_correct,
        'total_predictions': total_predictions,
        'total_actual_draws': total_actual_draws,
        'total_predicted_draws': total_predicted_draws,
        'avg_draw_precision': avg_draw_precision,
        'avg_brier_score': avg_brier,
        'window_metrics': all_metrics,
        'detailed_results': all_detailed_df,
        'num_windows': len(all_metrics)
    }
