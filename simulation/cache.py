"""
Caching Utilities for Simulation Results

This module provides functions for caching simulation results based on
configuration hashing, allowing users to avoid re-running identical experiments.

Key functions:
- hash_config: Create deterministic hash of configuration
- find_cached_results: Search for existing results matching config
- save_config_hash: Save hash metadata to result directory
- load_config_from_cache: Load original config from cached results
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


def hash_config(config: Dict) -> str:
    """
    Create deterministic hash of configuration.
    
    Normalizes config dict, sorts keys, and computes SHA256 hash.
    The hash is based only on fields that affect simulation results,
    excluding things like n_jobs, output paths, etc.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (should be normalized first)
        
    Returns
    -------
    str
        64-character hex hash
        
    Examples
    --------
    >>> config = {'name': 'test', 'data_configs': [{'delta': 0.1}]}
    >>> hash1 = hash_config(config)
    >>> hash2 = hash_config(config)
    >>> hash1 == hash2
    True
    """
    # Import normalize function from config module
    from .config import normalize_config_for_hashing
    
    # Normalize config for consistent hashing
    normalized = normalize_config_for_hashing(config)
    
    # Convert to JSON string with sorted keys
    config_str = json.dumps(normalized, sort_keys=True)
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    
    return hash_obj.hexdigest()


def find_cached_results(
    config_hash: str,
    output_dir: str = "results"
) -> Optional[Path]:
    """
    Search for cached results matching config hash.
    
    Looks for directories in output_dir that contain a .cache.json file
    with a matching hash.
    
    Parameters
    ----------
    config_hash : str
        Configuration hash to search for
    output_dir : str, default="results"
        Base results directory to search in
        
    Returns
    -------
    Path or None
        Path to cached results directory, or None if not found
        
    Examples
    --------
    >>> config_hash = "abc123..."
    >>> cached = find_cached_results(config_hash)
    >>> if cached:
    ...     print(f"Found cached results: {cached}")
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Search all subdirectories
    for result_dir in output_path.iterdir():
        if not result_dir.is_dir():
            continue
        
        cache_file = result_dir / ".cache.json"
        if not cache_file.exists():
            continue
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            if cache_data.get('config_hash') == config_hash:
                return result_dir
        except (json.JSONDecodeError, KeyError):
            # Invalid or corrupted cache file, skip
            continue
    
    return None


def save_config_hash(
    result_path: Path,
    config_hash: str,
    config: Dict
) -> None:
    """
    Save config hash and full config to result directory.
    
    Creates .cache.json file with hash and original config
    for verification and debugging. This allows future runs to
    identify matching configurations.
    
    Parameters
    ----------
    result_path : Path
        Result directory path
    config_hash : str
        Configuration hash
    config : dict
        Original configuration dictionary
        
    Examples
    --------
    >>> result_path = Path("results/my_experiment_abc123")
    >>> save_config_hash(result_path, "abc123...", config)
    """
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'config_hash': config_hash,
        'config': config
    }
    
    cache_file = result_path / ".cache.json"
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)


def load_config_from_cache(result_path: Path) -> Dict:
    """
    Load original config from cached results.
    
    Parameters
    ----------
    result_path : Path
        Path to results directory
        
    Returns
    -------
    dict
        Original configuration dictionary
        
    Raises
    ------
    FileNotFoundError
        If .cache.json file doesn't exist
    KeyError
        If cache file doesn't contain 'config' key
        
    Examples
    --------
    >>> result_path = Path("results/my_experiment_abc123")
    >>> config = load_config_from_cache(result_path)
    """
    cache_file = Path(result_path) / ".cache.json"
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}")
    
    with open(cache_file, 'r') as f:
        cache_data = json.load(f)
    
    if 'config' not in cache_data:
        raise KeyError("Cache file does not contain 'config' key")
    
    return cache_data['config']
