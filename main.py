import pandas as pd
from models.elo import EloRatingSystem
from models.elo_weighted import EloWeighted
from models.elo_time_weighted import EloTimeWeighted
from models.elo_combined import EloCombined
from models.baseline_uniform import UniformBaseline
from models.baseline_random import RandomBaseline
from dataloader import load_international_matches
from evaluation import rolling_window_evaluation

def run_model(model, df, model_name):
    

    results = rolling_window_evaluation(
        df,
        train_years=4,
        test_years=1,
        model=model,
        verbose=False
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: {model_name}")
    print(f"{'='*80}")
    print(f"\nFinal Average Accuracy: {results['avg_accuracy']:.4f} ({results['avg_accuracy']:.2%})")
    print(f"  └─ Total: {results['total_correct']}/{results['total_predictions']}")
    print(f"\nFinal Average Accuracy (no draws): {results['avg_accuracy_no_draw']:.4f} ({results['avg_accuracy_no_draw']:.2%})")
    print(f"  └─ Total: {results['total_no_draw_correct']}/{results['total_no_draw_total']}")
    print(f"\nDraw Statistics:")
    print(f"  Actual draws in test sets: {results['total_actual_draws']}")
    print(f"  Predicted draws: {results['total_predicted_draws']}")
    if results['total_predicted_draws'] > 0:
        print(f"  Draw precision: {results['avg_draw_precision']:.2%}")
    
    # Show per-window breakdown
    print(f"\n{'='*80}")
    print("PER-WINDOW BREAKDOWN")
    print(f"{'='*80}")
    print(f"\n{'Window':<8} {'Train Years':<15} {'Test Years':<12} {'Accuracy':<12} {'Acc (no draw)':<15} {'Correct/Total':<15}")
    print("-"*80)
    
    for m in results['window_metrics']:
        print(f"{m['window']:<8} {m['train_years']:<15} {m['test_years']:<12} "
              f"{m['accuracy']:<12.2%} {m['accuracy_no_draw']:<15.2%} "
              f"{m['correct']}/{m['total']:<10}")
    
    print("\n" + "="*80)
    
    return results

if __name__ == "__main__":

    # Load data
    print("\nLoading match data...")
    df = load_international_matches()

    # Baselines
    run_model(UniformBaseline(), df, "Uniform Baseline")
    run_model(RandomBaseline(seed=42), df, "Random Baseline")
    
    # Standard Elo
    run_model(EloRatingSystem(k_factor=32, initial_rating=1500), df, "Elo (standard)")
    
    # Elo with goal differential weighting
    run_model(EloWeighted(k_factor=32, initial_rating=1500, goal_diff_importance=1.0), df, "Elo (goal diff weighted)")
    
    # Elo with time weighting (recent matches matter more)
    run_model(EloTimeWeighted(k_factor=32, initial_rating=1500, time_decay_rate=0.5), df, "Elo (time weighted)")
    
    # Elo with both goal diff and time weighting
    run_model(EloCombined(k_factor=32, initial_rating=1500, goal_diff_importance=1.0, time_decay_rate=0.5), df, "Elo (combined: goal+time)")
    