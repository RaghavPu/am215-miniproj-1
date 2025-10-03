import pandas as pd
import numpy as np
from models.elo import EloRatingSystem
from dataloader import load_international_matches
from evaluation import rolling_window_evaluation


if __name__ == "__main__":
    # Load data
    print("\nLoading match data...")
    df = load_international_matches()

    model = EloRatingSystem(k_factor=32, initial_rating=1500)

    results = rolling_window_evaluation(
        df,
        train_years=4,
        test_years=1,
        model=model,
        verbose=False
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal Average Accuracy: {results['avg_accuracy']:.4f} ({results['avg_accuracy']:.2%})")
    print(f"Final Average Accuracy (no draws): {results['avg_accuracy_no_draw']:.4f} ({results['avg_accuracy_no_draw']:.2%})")
    
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
