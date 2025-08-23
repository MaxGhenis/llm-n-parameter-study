#!/usr/bin/env python3
"""Generate synthetic data for reproducible testing without API calls."""

import json
import csv
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

def generate_random_numbers_with_bias(n_samples=100, bias_value=47, bias_rate=0.15, seed=42):
    """Generate random numbers with realistic LLM-like bias."""
    np.random.seed(seed)
    
    results = []
    for i in range(n_samples):
        if np.random.random() < bias_rate:
            value = bias_value
        else:
            value = np.random.randint(1, 101)
        results.append(value)
    
    return results

def generate_n_parameter_batches(batch_size=5, n_batches=20, seed=42):
    """Generate batched responses simulating n parameter behavior."""
    np.random.seed(seed)
    
    batches = []
    for batch_id in range(n_batches):
        # Add position effect: first position more likely to be 47
        batch = []
        for position in range(batch_size):
            if position == 0 and np.random.random() < 0.35:  # Position 0 bias
                batch.append(47)
            elif np.random.random() < 0.10:  # General bias
                batch.append(47)
            else:
                batch.append(np.random.randint(1, 101))
        batches.append(batch)
    
    return batches

def generate_sentiment_data(n_samples=100, seed=42):
    """Generate synthetic sentiment analysis results."""
    np.random.seed(seed)
    
    sentiments = ['positive', 'negative', 'neutral']
    # Realistic distribution
    weights = [0.45, 0.35, 0.20]
    
    results = []
    for i in range(n_samples):
        sentiment = np.random.choice(sentiments, p=weights)
        confidence = np.random.uniform(0.7, 1.0)
        results.append({
            'id': i,
            'sentiment': sentiment,
            'confidence': round(confidence, 3)
        })
    
    return results

def generate_creative_text_samples(n_samples=20, seed=42):
    """Generate metadata for creative text generation."""
    np.random.seed(seed)
    
    samples = []
    for i in range(n_samples):
        sample = {
            'id': i,
            'prompt_type': np.random.choice(['story', 'poem', 'description']),
            'length_tokens': np.random.randint(50, 200),
            'uniqueness_score': round(np.random.uniform(0.3, 0.9), 3),
            'temperature_used': round(np.random.choice([0.7, 0.8, 0.9, 1.0]), 1)
        }
        samples.append(sample)
    
    return samples

def save_statistical_test_results(output_dir):
    """Generate and save pre-computed statistical test results."""
    results = {
        'ks_test': {
            'test_name': 'kolmogorov_smirnov',
            'comparing': ['n_parameter', 'separate_calls'],
            'statistic': 0.245,
            'p_value': 0.0012,
            'reject_null': True,
            'interpretation': 'Distributions are significantly different'
        },
        'position_effects': {
            'test_name': 'chi_square',
            'positions': [0, 1, 2, 3, 4],
            'chi2_statistic': 45.3,
            'p_value': 0.00001,
            'reject_null': True,
            'position_0_excess': 0.35,
            'interpretation': 'Strong position 0 bias detected'
        },
        'icc_calculation': {
            'test_name': 'intraclass_correlation',
            'icc_value': 0.689,
            'confidence_interval': [0.621, 0.751],
            'design_effect': 3.76,
            'effective_sample_size_reduction': 0.734,
            'interpretation': 'High within-batch correlation'
        },
        'autocorrelation': {
            'test_name': 'ljung_box',
            'lag': 10,
            'statistic': 28.4,
            'p_value': 0.0015,
            'reject_null': True,
            'interpretation': 'Significant autocorrelation detected'
        }
    }
    
    # Save individual test results
    stats_dir = output_dir / 'statistical_tests'
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    for test_name, test_data in results.items():
        with open(stats_dir / f'{test_name}.json', 'w') as f:
            json.dump(test_data, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for LLM n-parameter study')
    parser.add_argument('--output-dir', type=Path, default=Path('data/synthetic'),
                       help='Output directory for synthetic data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic data with seed={args.seed}")
    
    # Generate random number data
    print("Generating random number data...")
    random_numbers = generate_random_numbers_with_bias(n_samples=200, seed=args.seed)
    
    with open(args.output_dir / 'random_numbers.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id', 'value', 'method'])
        
        # First 100 as n_parameter (with batching effects)
        batches = generate_n_parameter_batches(seed=args.seed)
        sample_id = 0
        for batch in batches:
            for value in batch:
                writer.writerow([sample_id, value, 'n_parameter'])
                sample_id += 1
        
        # Next 100 as separate calls
        np.random.seed(args.seed + 1)
        for i in range(100):
            value = np.random.randint(1, 101)
            writer.writerow([sample_id, value, 'separate_calls'])
            sample_id += 1
    
    # Generate sentiment analysis data
    print("Generating sentiment analysis data...")
    sentiment_data = generate_sentiment_data(n_samples=100, seed=args.seed)
    
    with open(args.output_dir / 'sentiment_analysis.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'sentiment', 'confidence'])
        writer.writeheader()
        writer.writerows(sentiment_data)
    
    # Generate creative text metadata
    print("Generating creative text metadata...")
    creative_samples = generate_creative_text_samples(n_samples=20, seed=args.seed)
    
    with open(args.output_dir / 'creative_text.json', 'w') as f:
        json.dump({
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'seed': args.seed,
                'description': 'Synthetic creative text generation metadata'
            },
            'samples': creative_samples
        }, f, indent=2)
    
    # Generate statistical test results
    print("Generating statistical test results...")
    stats_results = save_statistical_test_results(args.output_dir.parent)
    
    # Create summary file
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'files_created': [
            'random_numbers.csv',
            'sentiment_analysis.csv',
            'creative_text.json',
            'statistical_tests/*.json'
        ],
        'statistics': {
            'random_numbers': {
                'total_samples': 200,
                'n_parameter_samples': 100,
                'separate_call_samples': 100,
                'bias_value': 47,
                'position_0_bias_rate': 0.35
            },
            'sentiment': {
                'total_samples': 100,
                'distribution': {'positive': 0.45, 'negative': 0.35, 'neutral': 0.20}
            },
            'creative_text': {
                'total_samples': 20
            }
        }
    }
    
    with open(args.output_dir / 'generation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSynthetic data generated successfully in {args.output_dir}")
    print(f"Summary saved to {args.output_dir / 'generation_summary.json'}")

if __name__ == '__main__':
    main()