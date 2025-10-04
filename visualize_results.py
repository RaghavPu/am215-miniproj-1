"""
Visualization script for model comparison.

Runs all models and generates bar charts comparing their performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from models.elo import EloRatingSystem
from models.elo_weighted import EloWeighted
from models.elo_time_weighted import EloTimeWeighted
from models.elo_combined import EloCombined
from models.baseline_uniform import UniformBaseline
from models.baseline_random import RandomBaseline
from dataloader import load_international_matches
from evaluation import rolling_window_evaluation


def run_all_models(df, verbose=False):
    """Run all models and collect results."""
    
    models = [
        (UniformBaseline(), "Uniform Baseline"),
        (RandomBaseline(seed=42), "Random Baseline"),
        (EloRatingSystem(k_factor=32, initial_rating=1500), "Elo (standard)"),
        (EloWeighted(k_factor=32, initial_rating=1500, goal_diff_importance=1.0), "Elo (goal diff)"),
        (EloTimeWeighted(k_factor=32, initial_rating=1500, time_decay_rate=0.5), "Elo (time weighted)"),
        (EloCombined(k_factor=32, initial_rating=1500, goal_diff_importance=1.0, time_decay_rate=0.5), "Elo (combined)"),
    ]
    
    results = []
    
    for i, (model, name) in enumerate(models):
        print(f"\n{'='*80}")
        print(f"Running {name} ({i+1}/{len(models)})...")
        print(f"{'='*80}")
        
        result = rolling_window_evaluation(
            df,
            train_years=4,
            test_years=1,
            model=model,
            verbose=verbose
        )
        
        results.append({
            'model_name': name,
            'avg_accuracy': result['avg_accuracy'],
            'avg_accuracy_no_draw': result['avg_accuracy_no_draw'],
            'total_correct': result['total_correct'],
            'total_predictions': result['total_predictions'],
            'total_no_draw_correct': result['total_no_draw_correct'],
            'total_no_draw_total': result['total_no_draw_total'],
            'avg_brier_score': result.get('avg_brier_score', None),
        })
        
        print(f"✓ {name}: {result['avg_accuracy']:.2%} overall, {result['avg_accuracy_no_draw']:.2%} (no draws)")
    
    return pd.DataFrame(results)


def create_comparison_chart(results_df, output_file='model_comparison.png'):
    """Create a bar chart comparing all models."""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    models = results_df['model_name'].tolist()
    x_pos = np.arange(len(models))
    
    # Define colors
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
    
    # Plot 1: Overall Accuracy
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, results_df['avg_accuracy'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy (All Predictions)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline (50%)')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: No-Draw Accuracy
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, results_df['avg_accuracy_no_draw'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy on Decisive Matches (No Draws)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Baseline (50%)')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add overall title
    fig.suptitle('Football Match Prediction Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_file}")
    
    return fig


def create_detailed_chart(results_df, output_file='model_comparison_detailed.png'):
    """Create a more detailed comparison with multiple metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    models = results_df['model_name'].tolist()
    x_pos = np.arange(len(models))
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#1abc9c']
    
    # Plot 1: Overall Accuracy
    ax1 = axes[0, 0]
    bars1 = ax1.barh(x_pos, results_df['avg_accuracy'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_yticks(x_pos)
    ax1.set_yticklabels(models)
    ax1.set_xlim([0, 100])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Plot 2: No-Draw Accuracy
    ax2 = axes[0, 1]
    bars2 = ax2.barh(x_pos, results_df['avg_accuracy_no_draw'] * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy (No Draws)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(models)
    ax2.set_xlim([0, 100])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}%',
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Plot 3: Total Predictions Breakdown
    ax3 = axes[1, 0]
    correct = results_df['total_correct'].values
    incorrect = results_df['total_predictions'].values - correct
    
    bar_width = 0.35
    ax3.bar(x_pos, correct, bar_width, label='Correct', color='#2ecc71', alpha=0.8, edgecolor='black')
    ax3.bar(x_pos, incorrect, bar_width, bottom=correct, label='Incorrect', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Breakdown (All Matches)', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Plot 4: Brier Score (if available)
    ax4 = axes[1, 1]
    brier_scores = results_df['avg_brier_score'].fillna(0) * 1000  # Scale for visibility
    bars4 = ax4.bar(x_pos, brier_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Brier Score (×1000, lower is better)', fontsize=12, fontweight='bold')
    ax4.set_title('Brier Score (Prediction Quality)', fontsize=14, fontweight='bold', pad=15)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars4:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed chart saved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    print("="*80)
    print("MODEL COMPARISON & VISUALIZATION")
    print("="*80)
    
    # Load data
    print("\nLoading match data...")
    df = load_international_matches()
    
    # Run all models
    print("\nRunning all models...")
    results_df = run_all_models(df, verbose=False)
    
    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print(f"\n✓ Results saved to: model_results.csv")
    
    # Display results table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Create visualizations
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    create_comparison_chart(results_df, 'model_comparison.png')
    create_detailed_chart(results_df, 'model_comparison_detailed.png')
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - model_results.csv")
    print("  - model_comparison.png")
    print("  - model_comparison_detailed.png")
