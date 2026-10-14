import pandas as pd
import numpy as np
from scipy import stats

def run_ab_test():
    df = pd.read_csv("data/processed/engineered_features.csv")
    
    g1 = df[df['length_category'] == 'Shorts']['engagement_rate'].dropna()
    g2 = df[df['length_category'] == 'Long']['engagement_rate'].dropna()
    
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    
    pool_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(g1) - np.mean(g2)) / pool_sd
    
    print("--- A/B Test: Shorts vs Long Videos Engagement ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value: {p_val:.4f}")
    print(f"Cohen's d: {cohens_d:.4f}")
    
    if p_val < 0.05:
        print("Result: Significant difference found.")
    else:
        print("Result: No significant difference.")

if __name__ == "__main__":
    run_ab_test()