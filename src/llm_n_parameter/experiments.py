"""
Experiments comparing n parameter vs separate API calls.
"""

import openai
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    model: str = "gpt-4o-mini"
    temperature: float = 1.0
    max_tokens: int = 50
    batch_size: int = 5
    num_batches: int = 20
    delay_between_calls: float = 0.1


class NParameterExperiment:
    """Run experiments comparing n parameter vs separate calls."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    
    def query_n_parameter(
        self, 
        prompt: str = "Pick a random number between 1 and 100.",
        n: int = 5,
        config: Optional[ExperimentConfig] = None
    ) -> List[str]:
        """Query using n parameter for batch completions."""
        if config is None:
            config = ExperimentConfig()
        
        response = self.client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=n
        )
        
        return [choice.message.content.strip() for choice in response.choices]
    
    def query_separate(
        self,
        prompt: str = "Pick a random number between 1 and 100.",
        n: int = 5,
        config: Optional[ExperimentConfig] = None
    ) -> List[str]:
        """Query using separate API calls."""
        if config is None:
            config = ExperimentConfig()
        
        responses = []
        for _ in range(n):
            response = self.client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            responses.append(response.choices[0].message.content.strip())
            time.sleep(config.delay_between_calls)
        
        return responses
    
    def extract_numbers(self, responses: List[str]) -> List[Optional[int]]:
        """Extract numbers from text responses."""
        numbers = []
        for res in responses:
            matches = re.findall(r'\b\d+\b', res)
            if matches:
                try:
                    num = int(matches[0])
                    if 1 <= num <= 100:
                        numbers.append(num)
                    else:
                        numbers.append(None)
                except:
                    numbers.append(None)
            else:
                numbers.append(None)
        return numbers
    
    def run_full_experiment(
        self,
        prompt: str = "Pick a random number between 1 and 100.",
        config: Optional[ExperimentConfig] = None
    ) -> Dict[str, Any]:
        """Run full experiment comparing both methods."""
        if config is None:
            config = ExperimentConfig()
        
        print(f"Running experiment with {config.model} at temperature={config.temperature}")
        
        # Collect data using n parameter
        print("\nCollecting data using n parameter...")
        n_param_batches = []
        n_param_all = []
        
        for i in tqdm(range(config.num_batches)):
            batch = self.query_n_parameter(prompt, config.batch_size, config)
            n_param_batches.append(batch)
            n_param_all.extend(batch)
            time.sleep(config.delay_between_calls)
        
        # Collect data using separate calls
        print("\nCollecting data using separate calls...")
        separate_batches = []
        separate_all = []
        
        for i in tqdm(range(config.num_batches)):
            batch = self.query_separate(prompt, config.batch_size, config)
            separate_batches.append(batch)
            separate_all.extend(batch)
        
        # Extract numbers
        n_param_numbers = self.extract_numbers(n_param_all)
        separate_numbers = self.extract_numbers(separate_all)
        
        n_param_batch_numbers = [
            self.extract_numbers(batch) for batch in n_param_batches
        ]
        separate_batch_numbers = [
            self.extract_numbers(batch) for batch in separate_batches
        ]
        
        return {
            "config": config.__dict__,
            "n_parameter": {
                "all_responses": n_param_all,
                "batches": n_param_batches,
                "numbers": [n for n in n_param_numbers if n is not None],
                "batch_numbers": n_param_batch_numbers
            },
            "separate": {
                "all_responses": separate_all,
                "batches": separate_batches,
                "numbers": [n for n in separate_numbers if n is not None],
                "batch_numbers": separate_batch_numbers
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        with open(filename, 'r') as f:
            return json.load(f)