#!/usr/bin/env python
"""
prepare_issues_binary.py
========================

Convert the 3-class issues dataset (bug/feature/question) into a binary
classification dataset (bug vs non-bug).
"""

import pandas as pd
from pathlib import Path

def prepare_binary_issues():
    """Convert multi-class issues to binary bug vs non-bug classification."""
    
    # Load the original dataset
    df = pd.read_csv('data/issues.csv')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original label distribution:\n{df['label'].value_counts()}")
    
    # Create binary labels
    df['binary_label'] = df['label'].apply(lambda x: 'bug' if x == 'bug' else 'non_bug')
    
    # Keep original columns plus binary label
    df_binary = df[['repo', 'created_at', 'title', 'body', 'binary_label']]
    df_binary = df_binary.rename(columns={'binary_label': 'label'})
    
    # Shuffle the dataset
    df_binary = df_binary.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the binary dataset
    output_path = 'data/issues_binary.csv'
    df_binary.to_csv(output_path, index=False)
    
    print(f"\nBinary dataset saved to: {output_path}")
    print(f"Binary label distribution:\n{df_binary['label'].value_counts()}")
    
    # Show some examples
    print("\n=== Sample bug issues ===")
    bug_samples = df_binary[df_binary['label'] == 'bug'].head(3)
    for idx, row in bug_samples.iterrows():
        print(f"Title: {row['title'][:80]}...")
        
    print("\n=== Sample non-bug issues ===")
    non_bug_samples = df_binary[df_binary['label'] == 'non_bug'].head(3)
    for idx, row in non_bug_samples.iterrows():
        print(f"Title: {row['title'][:80]}...")

if __name__ == "__main__":
    prepare_binary_issues()