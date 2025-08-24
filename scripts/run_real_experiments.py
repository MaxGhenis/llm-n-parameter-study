#!/usr/bin/env python3
"""
Run REAL experiments with actual API calls.

WARNING: This script makes real API calls and costs real money!
Only run if you have API keys configured and understand the costs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from llm_n_parameter.verified_experiments import (
    VerifiedOpenAIExperiment,
    VerifiedGeminiExperiment,
    save_verified_results,
    verify_saved_results
)


def estimate_cost(provider: str, n: int, num_batches: int, num_separate: int) -> float:
    """Estimate API costs before running."""
    
    if provider == "openai":
        # GPT-4o-mini pricing (as of 2024)
        input_price_per_1k = 0.00015  # $0.15 per 1M tokens
        output_price_per_1k = 0.0006   # $0.60 per 1M tokens
        
        # Rough estimates
        prompt_tokens = 10  # "Pick a random number..."
        output_tokens = 10  # Short response
        
        # n-parameter calls
        n_param_input = num_batches * prompt_tokens
        n_param_output = num_batches * n * output_tokens
        
        # Separate calls
        separate_input = num_separate * prompt_tokens
        separate_output = num_separate * output_tokens
        
        total_input = n_param_input + separate_input
        total_output = n_param_output + separate_output
        
        cost = (total_input * input_price_per_1k / 1000 + 
                total_output * output_price_per_1k / 1000)
        
        return cost
    
    elif provider == "gemini":
        # Gemini 1.5 Flash pricing (as of 2024)
        # Free tier: 15 RPM, 1M tokens/day
        # Assume free tier
        return 0.0
    
    return 0.0


def run_openai_experiments(args):
    """Run OpenAI experiments with n parameter."""
    
    print("\n" + "="*60)
    print("OPENAI EXPERIMENTS")
    print("="*60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return False
    
    # Estimate cost
    cost = estimate_cost("openai", args.n, args.batches, args.separate_calls)
    print(f"Estimated cost: ${cost:.4f}")
    
    if not args.skip_confirmation:
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return False
    
    # Create experiment
    exp = VerifiedOpenAIExperiment()
    
    # Create output directory
    output_dir = Path("data/real_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run n-parameter experiment
    print(f"\nRunning n={args.n} experiment with {args.batches} batches...")
    n_results = exp.run_n_parameter_experiment(
        n=args.n,
        num_batches=args.batches,
        model=args.model,
        temperature=args.temperature
    )
    
    # Save results
    n_file = output_dir / f"openai_n{args.n}_T{args.temperature}_real.json"
    save_verified_results(n_results, n_file)
    
    # Run separate calls experiment
    print(f"\nRunning {args.separate_calls} separate calls...")
    sep_results = exp.run_separate_calls_experiment(
        num_calls=args.separate_calls,
        model=args.model,
        temperature=args.temperature
    )
    
    # Save results
    sep_file = output_dir / f"openai_separate_T{args.temperature}_real.json"
    save_verified_results(sep_results, sep_file)
    
    # Verify both files
    print("\nVerifying saved data...")
    n_valid = verify_saved_results(n_file)
    sep_valid = verify_saved_results(sep_file)
    
    if n_valid and sep_valid:
        print("✅ All data verified as real API responses")
        
        # Quick analysis
        analyze_results(n_results, sep_results)
        
        return True
    else:
        print("❌ Verification failed")
        return False


def run_gemini_experiments(args):
    """Run Gemini experiments with candidateCount."""
    
    print("\n" + "="*60)
    print("GEMINI EXPERIMENTS")
    print("="*60)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set")
        return False
    
    # Note about Gemini limits
    print("Note: Gemini candidateCount is limited to 4 (not 5 like OpenAI)")
    candidate_count = min(args.n, 4)
    
    if not args.skip_confirmation:
        response = input(f"Run with candidateCount={candidate_count}? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return False
    
    # Create experiment
    exp = VerifiedGeminiExperiment()
    
    # Create output directory
    output_dir = Path("data/real_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run candidateCount experiment
    print(f"\nRunning candidateCount={candidate_count} experiment...")
    results = exp.run_candidate_count_experiment(
        candidate_count=candidate_count,
        num_batches=args.batches,
        temperature=args.temperature
    )
    
    # Save results
    output_file = output_dir / f"gemini_cc{candidate_count}_T{args.temperature}_real.json"
    save_verified_results(results, output_file)
    
    # Verify
    print("\nVerifying saved data...")
    if verify_saved_results(output_file):
        print("✅ Gemini data verified as real API responses")
        return True
    else:
        print("❌ Verification failed")
        return False


def analyze_results(n_results: dict, sep_results: dict):
    """Quick analysis of results."""
    
    print("\n" + "="*60)
    print("QUICK ANALYSIS")
    print("="*60)
    
    # Extract numbers from n-parameter results
    n_numbers = []
    for batch in n_results.get("batches", []):
        if "numbers" in batch:
            n_numbers.extend([n for n in batch["numbers"] if n is not None])
    
    # Extract numbers from separate calls
    sep_numbers = []
    for call in sep_results.get("calls", []):
        if "number" in call and call["number"] is not None:
            sep_numbers.append(call["number"])
    
    if n_numbers and sep_numbers:
        import numpy as np
        
        print(f"n-parameter results: {len(n_numbers)} numbers")
        print(f"  Mean: {np.mean(n_numbers):.2f}")
        print(f"  Std:  {np.std(n_numbers):.2f}")
        print(f"  Unique values: {len(set(n_numbers))}")
        
        # Check for suspicious patterns
        from collections import Counter
        n_counts = Counter(n_numbers)
        most_common = n_counts.most_common(1)[0]
        print(f"  Most common: {most_common[0]} (appears {most_common[1]} times)")
        
        print(f"\nSeparate calls results: {len(sep_numbers)} numbers")
        print(f"  Mean: {np.mean(sep_numbers):.2f}")
        print(f"  Std:  {np.std(sep_numbers):.2f}")
        print(f"  Unique values: {len(set(sep_numbers))}")
        
        sep_counts = Counter(sep_numbers)
        most_common_sep = sep_counts.most_common(1)[0]
        print(f"  Most common: {most_common_sep[0]} (appears {most_common_sep[1]} times)")
        
        # Position effects in n-parameter
        print("\nPosition effects in n-parameter batches:")
        n_value = len(n_results["batches"][0]["numbers"]) if n_results["batches"] else 0
        for pos in range(n_value):
            pos_values = [batch["numbers"][pos] 
                         for batch in n_results["batches"] 
                         if "numbers" in batch and len(batch["numbers"]) > pos
                         and batch["numbers"][pos] is not None]
            if pos_values:
                print(f"  Position {pos}: mean={np.mean(pos_values):.2f}, "
                      f"std={np.std(pos_values):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run REAL API experiments (costs money!)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini", "both"],
        default="openai",
        help="Which API provider to test"
    )
    
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Value for n parameter (OpenAI) or candidateCount (Gemini)"
    )
    
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches for n-parameter experiment"
    )
    
    parser.add_argument(
        "--separate-calls",
        type=int,
        default=50,
        help="Number of separate API calls to make"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature parameter"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (for OpenAI)"
    )
    
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip cost confirmation prompt"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("REAL API EXPERIMENT RUNNER")
    print("="*60)
    print("⚠️  This will make REAL API calls and cost REAL money!")
    print("⚠️  Make sure you have API keys configured")
    print("⚠️  Make sure you understand the costs")
    print("="*60)
    
    success = True
    
    if args.provider in ["openai", "both"]:
        success = success and run_openai_experiments(args)
    
    if args.provider in ["gemini", "both"]:
        success = success and run_gemini_experiments(args)
    
    if success:
        print("\n✅ Experiments completed successfully!")
        print("✅ All data is verified as real API responses")
        print("📁 Results saved in data/real_experiments/")
    else:
        print("\n❌ Some experiments failed or were aborted")
        sys.exit(1)


if __name__ == "__main__":
    main()