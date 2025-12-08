"""
Rerun Experiment from Hash

This script allows you to:
1. Load the original configuration from a result hash
2. View the configuration
3. Optionally re-run the experiment with the same or modified config

Usage:
    python rerun_from_hash.py <hash_or_directory>
    python rerun_from_hash.py a5d7fbc8
    python rerun_from_hash.py poisson_delta_comparison_a5d7fbc8
"""

import sys
from pathlib import Path
from simulation.cache import load_config_from_cache, find_cached_results
from simulation import run_simulation
import json


def find_result_directory(identifier: str, results_base: str = "results") -> Path:
    """
    Find result directory from hash or directory name.
    
    Parameters
    ----------
    identifier : str
        Either a hash (e.g., "a5d7fbc8") or full directory name
    results_base : str
        Base results directory
        
    Returns
    -------
    Path
        Path to results directory
    """
    results_path = Path(results_base)
    
    # Case 1: Full directory name provided
    full_path = results_path / identifier
    if full_path.exists():
        return full_path
    
    # Case 2: Hash provided - search for matching directory
    for result_dir in results_path.iterdir():
        if result_dir.is_dir() and identifier in result_dir.name:
            return result_dir
    
    raise FileNotFoundError(f"Could not find results directory for: {identifier}")


def display_config(config: dict) -> None:
    """Display configuration in a readable format."""
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    
    # Basic info
    print(f"\nExperiment Name: {config.get('name', 'N/A')}")
    print(f"Replications: {config.get('n_replications', 'N/A')}")
    print(f"Models: {', '.join(config.get('models', []))}")
    print(f"Optimization: {config.get('optimization', 'N/A')}")
    print(f"Metric: {config.get('optimize_metric', 'N/A')}")
    
    # Data configurations
    data_configs = config.get('data_configs', [])
    print(f"\nData Configurations: {len(data_configs)}")
    
    if len(data_configs) <= 5:
        # Show all if few
        for i, dc in enumerate(data_configs):
            print(f"\n  Config {i}:")
            for key, val in dc.items():
                print(f"    {key}: {val}")
    else:
        # Show summary if many
        print(f"  (Too many to display, showing first 3)")
        for i in range(3):
            print(f"\n  Config {i}:")
            for key, val in data_configs[i].items():
                print(f"    {key}: {val}")
        print(f"\n  ... and {len(data_configs) - 3} more")
    
    # Hyperparameter grid
    if 'hyperparameter_grid' in config:
        print("\nHyperparameter Grid:")
        for key, val in config['hyperparameter_grid'].items():
            print(f"  {key}: {val}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python rerun_from_hash.py <hash_or_directory>")
        print("\nExamples:")
        print("  python rerun_from_hash.py a5d7fbc8")
        print("  python rerun_from_hash.py poisson_delta_comparison_a5d7fbc8")
        sys.exit(1)
    
    identifier = sys.argv[1]
    
    # Find result directory
    try:
        result_path = find_result_directory(identifier)
        print(f"Found results: {result_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load config
    try:
        config = load_config_from_cache(result_path)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Display config
    display_config(config)
    
    # Ask user what to do
    print("\nOptions:")
    print("  1. View full config JSON")
    print("  2. Re-run experiment with same config")
    print("  3. Export config to file")
    print("  4. Exit")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        sys.exit(0)
    
    if choice == "1":
        # View full config
        print("\n" + "=" * 80)
        print("FULL CONFIGURATION (JSON)")
        print("=" * 80)
        print(json.dumps(config, indent=2))
    
    elif choice == "2":
        # Re-run experiment
        print("\n" + "=" * 80)
        print("RE-RUNNING EXPERIMENT")
        print("=" * 80)
        print("\nThis will create a new result directory with a different hash")
        print("(since we're re-running, the timestamp will be different)")
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm == 'y':
            print("\nStarting simulation...")
            results = run_simulation(config, cache=False, verbose=True)
            print(f"\n✓ Complete! New results saved to: {results.path}")
        else:
            print("Cancelled.")
    
    elif choice == "3":
        # Export config
        output_file = input("\nEnter output filename (e.g., config.json): ").strip()
        if not output_file:
            output_file = "exported_config.json"
        
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Config exported to: {output_file}")
        print("\nYou can now modify this file and run:")
        print(f"  python -c \"from simulation import run_simulation; import json;")
        print(f"  config = json.load(open('{output_file}')); run_simulation(config)\"")
    
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
