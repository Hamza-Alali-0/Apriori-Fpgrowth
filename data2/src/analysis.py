"""
Analysis Module for Association Rule Mining
Provides sensitivity analysis, comparison, and visualization functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from apriori_fpgrowth import AprioriAlgorithm, FPGrowth, generate_association_rules, rules_to_dataframe


def sensitivity_analysis(
    transactions: List[List[str]],
    support_values: List[float],
    algorithm: str = 'apriori',
    min_confidence: float = 0.5
) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying min_support.
    
    Args:
        transactions: List of transactions
        support_values: List of support thresholds to test
        algorithm: 'apriori' or 'fpgrowth'
        min_confidence: Minimum confidence for rules
    
    Returns:
        DataFrame with results for each support value
    """
    results = []
    n_transactions = len(transactions)
    
    for min_sup in support_values:
        print(f"\nTesting min_support = {min_sup}...")
        
        # Run algorithm
        if algorithm.lower() == 'apriori':
            algo = AprioriAlgorithm(min_support=min_sup)
        else:
            algo = FPGrowth(min_support=min_sup)
        
        freq_itemsets = algo.fit(transactions)
        
        # Count itemsets
        total_itemsets = sum(len(itemsets) for itemsets in freq_itemsets.values())
        
        # Generate rules
        rules = generate_association_rules(
            freq_itemsets,
            algo.support_data,
            n_transactions,
            min_confidence=min_confidence
        )
        
        # Calculate average metrics
        if rules:
            avg_confidence = np.mean([r['confidence'] for r in rules])
            avg_lift = np.mean([r['lift'] for r in rules])
            max_lift = max([r['lift'] for r in rules])
            
            # Count trivial vs interesting rules
            trivial_rules = sum(1 for r in rules if r['lift'] < 1.5)
            interesting_rules = len(rules) - trivial_rules
        else:
            avg_confidence = avg_lift = max_lift = 0
            trivial_rules = interesting_rules = 0
        
        results.append({
            'min_support': min_sup,
            'num_frequent_itemsets': total_itemsets,
            'num_rules': len(rules),
            'trivial_rules': trivial_rules,
            'interesting_rules': interesting_rules,
            'avg_confidence': avg_confidence,
            'avg_lift': avg_lift,
            'max_lift': max_lift
        })
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(df_sensitivity: pd.DataFrame, save_path: str = None):
    """
    Create visualization for sensitivity analysis results.
    
    Args:
        df_sensitivity: DataFrame from sensitivity_analysis function
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensitivity Analysis: Impact of Min Support', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of frequent itemsets
    axes[0, 0].plot(df_sensitivity['min_support'], df_sensitivity['num_frequent_itemsets'], 
                    marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel('Minimum Support', fontsize=12)
    axes[0, 0].set_ylabel('Number of Frequent Itemsets', fontsize=12)
    axes[0, 0].set_title('Frequent Itemsets vs Min Support', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of rules
    axes[0, 1].plot(df_sensitivity['min_support'], df_sensitivity['num_rules'], 
                    marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel('Minimum Support', fontsize=12)
    axes[0, 1].set_ylabel('Number of Rules', fontsize=12)
    axes[0, 1].set_title('Association Rules vs Min Support', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Trivial vs Interesting rules
    width = (df_sensitivity['min_support'].iloc[1] - df_sensitivity['min_support'].iloc[0]) * 0.8 / 2
    axes[1, 0].bar(df_sensitivity['min_support'] - width/2, df_sensitivity['trivial_rules'], 
                   width=width, label='Trivial (lift < 1.5)', color='#F18F01', alpha=0.7)
    axes[1, 0].bar(df_sensitivity['min_support'] + width/2, df_sensitivity['interesting_rules'], 
                   width=width, label='Interesting (lift ≥ 1.5)', color='#06A77D', alpha=0.7)
    axes[1, 0].set_xlabel('Minimum Support', fontsize=12)
    axes[1, 0].set_ylabel('Number of Rules', fontsize=12)
    axes[1, 0].set_title('Rule Quality Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Average lift
    axes[1, 1].plot(df_sensitivity['min_support'], df_sensitivity['avg_lift'], 
                    marker='^', linewidth=2, markersize=8, color='#C73E1D', label='Average Lift')
    axes[1, 1].plot(df_sensitivity['min_support'], df_sensitivity['max_lift'], 
                    marker='v', linewidth=2, markersize=8, color='#6A994E', label='Max Lift')
    axes[1, 1].set_xlabel('Minimum Support', fontsize=12)
    axes[1, 1].set_ylabel('Lift Value', fontsize=12)
    axes[1, 1].set_title('Rule Lift Metrics', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sensitivity analysis plot to {save_path}")
    
    return fig


def compare_segments(
    segment1_rules: pd.DataFrame,
    segment2_rules: pd.DataFrame,
    segment1_name: str = "Segment 1",
    segment2_name: str = "Segment 2",
    top_n: int = 10
) -> Dict:
    """
    Compare association rules between two segments.
    
    Args:
        segment1_rules: Rules DataFrame for segment 1
        segment2_rules: Rules DataFrame for segment 2
        segment1_name: Name of segment 1
        segment2_name: Name of segment 2
        top_n: Number of top rules to show
    
    Returns:
        Dictionary with comparison results
    """
    # Create rule identifiers
    segment1_rules['rule'] = segment1_rules['antecedent_str'] + ' → ' + segment1_rules['consequent_str']
    segment2_rules['rule'] = segment2_rules['antecedent_str'] + ' → ' + segment2_rules['consequent_str']
    
    rules_seg1 = set(segment1_rules['rule'])
    rules_seg2 = set(segment2_rules['rule'])
    
    common_rules = rules_seg1 & rules_seg2
    unique_seg1 = rules_seg1 - rules_seg2
    unique_seg2 = rules_seg2 - rules_seg1
    
    comparison = {
        'total_rules_seg1': len(rules_seg1),
        'total_rules_seg2': len(rules_seg2),
        'common_rules': len(common_rules),
        'unique_to_seg1': len(unique_seg1),
        'unique_to_seg2': len(unique_seg2),
        'common_rules_list': list(common_rules)[:top_n],
        'unique_seg1_list': list(unique_seg1)[:top_n],
        'unique_seg2_list': list(unique_seg2)[:top_n]
    }
    
    print(f"\n{'='*60}")
    print(f"SEGMENT COMPARISON: {segment1_name} vs {segment2_name}")
    print(f"{'='*60}")
    print(f"Total rules in {segment1_name}: {comparison['total_rules_seg1']}")
    print(f"Total rules in {segment2_name}: {comparison['total_rules_seg2']}")
    print(f"Common rules: {comparison['common_rules']}")
    print(f"Unique to {segment1_name}: {comparison['unique_to_seg1']}")
    print(f"Unique to {segment2_name}: {comparison['unique_to_seg2']}")
    
    return comparison


def plot_top_rules(rules_df: pd.DataFrame, top_n: int = 10, 
                  metric: str = 'lift', title: str = None, save_path: str = None):
    """
    Create horizontal bar chart of top rules.
    
    Args:
        rules_df: DataFrame of rules
        top_n: Number of top rules to display
        metric: Metric to sort by ('lift', 'confidence', 'support')
        title: Plot title
        save_path: Optional path to save figure
    """
    # Get top rules
    top_rules = rules_df.nlargest(top_n, metric).copy()
    
    # Create rule labels
    top_rules['rule_label'] = (top_rules['antecedent_str'].str[:30] + ' → ' + 
                                top_rules['consequent_str'].str[:30])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.5)))
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_rules)), top_rules[metric], color='#3A86FF', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_rules[metric])):
        ax.text(value + 0.02, i, f'{value:.2f}', va='center', fontsize=10)
    
    # Customize plot
    ax.set_yticks(range(len(top_rules)))
    ax.set_yticklabels(top_rules['rule_label'], fontsize=10)
    ax.set_xlabel(metric.capitalize(), fontsize=12, fontweight='bold')
    ax.set_title(title or f'Top {top_n} Association Rules by {metric.capitalize()}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Highest value at top
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved top rules plot to {save_path}")
    
    return fig


def print_rule_insights(rules_df: pd.DataFrame, top_n: int = 5):
    """
    Print business insights from top rules.
    
    Args:
        rules_df: DataFrame of association rules
        top_n: Number of top rules to analyze
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_n} ASSOCIATION RULES - BUSINESS INSIGHTS")
    print(f"{'='*80}\n")
    
    # Sort by lift (most interesting correlations)
    top_rules = rules_df.nlargest(top_n, 'lift')
    
    for i, row in enumerate(top_rules.itertuples(), 1):
        print(f"{i}. Rule: {row.antecedent_str} → {row.consequent_str}")
        print(f"   Support: {row.support:.3f} ({row.support*100:.1f}% of transactions)")
        print(f"   Confidence: {row.confidence:.3f} ({row.confidence*100:.1f}% probability)")
        print(f"   Lift: {row.lift:.2f} (correlation strength)")
        
        # Interpret lift
        if row.lift > 2:
            interpretation = "STRONG positive correlation - highly recommended for bundling"
        elif row.lift > 1.5:
            interpretation = "MODERATE positive correlation - good candidate for recommendations"
        elif row.lift > 1:
            interpretation = "WEAK positive correlation - consider with other factors"
        else:
            interpretation = "Negative or no correlation - not recommended"
        
        print(f"   Interpretation: {interpretation}")
        print()


if __name__ == "__main__":
    print("Analysis module loaded successfully!")
    print("Available functions:")
    print("  - sensitivity_analysis()")
    print("  - plot_sensitivity_analysis()")
    print("  - compare_segments()")
    print("  - plot_top_rules()")
    print("  - print_rule_insights()")
