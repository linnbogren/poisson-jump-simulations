"""
Data Generation for Poisson Jump Simulations

This module handles:
- HMM transition matrix generation
- State sequence sampling
- Poisson, Negative Binomial, and Gaussian HMM data generation
- Correlated noise generation using Gaussian Copula
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional
from .config import SimulationConfig


###############################################################################
# Transition Matrix Generation
###############################################################################

def generate_hmm_transition_matrix(n_states: int, 
                                   persistence: float = 0.97) -> np.ndarray:
    """
    Generate a transition matrix with high diagonal dominance.
    
    The matrix has diagonal elements approximately equal to `persistence`,
    with off-diagonal elements uniformly distributed to sum to (1 - persistence).
    
    Parameters:
    -----------
    n_states : int
        Number of states in the HMM.
    persistence : float
        Probability of staying in the same state (diagonal elements).
        
    Returns:
    --------
    np.ndarray
        Transition matrix of shape (n_states, n_states) where rows sum to 1.
    """
    assert 0 < persistence < 1, "Persistence must be in (0, 1)"
    assert n_states >= 2, "Must have at least 2 states"
    
    A = np.zeros((n_states, n_states))
    
    # Set diagonal to persistence
    np.fill_diagonal(A, persistence)
    
    # Distribute remaining probability uniformly to off-diagonal elements
    off_diag_prob = (1 - persistence) / (n_states - 1)
    A = A + off_diag_prob * (1 - np.eye(n_states))
    
    # Ensure rows sum to 1 (handle floating point errors)
    A = A / A.sum(axis=1, keepdims=True)
    
    return A


###############################################################################
# Lambda (Rate Parameter) Generation
###############################################################################

def compute_state_lambdas(lambda_0: float,
                          delta: float,
                          n_states: int,
                          n_features: int) -> np.ndarray:
    """
    Compute state-specific Poisson rates for the informative features.
    
    Following the pattern in simulation_instructions.tex:
    - State 1: λ₀(1 - δ) - low rate
    - State 2: λ₀ - baseline rate (overlaps with noise)
    - State 3: λ₀(1 + δ) - high rate
    - For more states, interpolate linearly
    
    Parameters:
    -----------
    lambda_0 : float
        Baseline Poisson rate.
    delta : float
        Signal strength parameter (0 <= delta < 1).
    n_states : int
        Number of states.
    n_features : int
        Number of informative features.
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_states, n_features) with state-specific rates.
    """
    assert 0 <= delta < 1, "delta must be in [0, 1)"
    assert lambda_0 > 0, "lambda_0 must be positive"
    
    lambdas = np.zeros((n_states, n_features))
    
    if n_states == 2:
        # Two states: low and high
        lambdas[0, :] = lambda_0 * (1 - delta)
        lambdas[1, :] = lambda_0 * (1 + delta)
    elif n_states == 3:
        # Three states: low, baseline, high
        lambdas[0, :] = lambda_0 * (1 - delta)
        lambdas[1, :] = lambda_0
        lambdas[2, :] = lambda_0 * (1 + delta)
    else:
        # More states: interpolate linearly from (1-δ) to (1+δ)
        multipliers = np.linspace(1 - delta, 1 + delta, n_states)
        for k in range(n_states):
            lambdas[k, :] = lambda_0 * multipliers[k]
    
    return lambdas


###############################################################################
# State Sequence Sampling
###############################################################################

def sample_state_sequence(n_samples: int,
                          n_states: int,
                          transition_matrix: np.ndarray,
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a state sequence from an HMM with given transition matrix.
    
    Parameters:
    -----------
    n_samples : int
        Length of the sequence.
    n_states : int
        Number of states.
    transition_matrix : np.ndarray
        Transition matrix of shape (n_states, n_states).
    random_state : int, optional
        Random seed.
        
    Returns:
    --------
    states : np.ndarray
        State sequence of shape (n_samples,).
    breakpoints : np.ndarray
        Indices where state changes occur.
    """
    rng = np.random.RandomState(random_state)
    
    states = np.zeros(n_samples, dtype=int)
    breakpoints = []
    
    # Start with uniform distribution over states
    states[0] = rng.choice(n_states)
    
    # Generate sequence using transition matrix
    for t in range(1, n_samples):
        states[t] = rng.choice(n_states, p=transition_matrix[states[t-1], :])
        if states[t] != states[t-1]:
            breakpoints.append(t)
    
    return states, np.array(breakpoints, dtype=int)


###############################################################################
# Poisson HMM Data Generation
###############################################################################

def generate_poisson_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate data from a Poisson HMM.
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object with all parameters.
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix of shape (n_samples, n_total_features).
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    
    # Sample state sequence
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute state-specific lambdas for informative features
    lambdas_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    # Initialize data matrix
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # Generate informative features
    for t in range(config.n_samples):
        state_t = states[t]
        X[t, :config.n_informative] = rng.poisson(lambdas_inform[state_t, :])
    
    # Generate noise features (constant rate λ₀ across all states)
    if config.n_noise > 0:
        if config.correlated_noise:
            X[:, config.n_informative:] = generate_correlated_noise(
                config.n_samples, config.n_noise, config.lambda_0,
                config.noise_correlation, rng
            )
        else:
            X[:, config.n_informative:] = rng.poisson(
                config.lambda_0, size=(config.n_samples, config.n_noise)
            )
    
    # Create DataFrame with proper column names
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


###############################################################################
# Negative Binomial HMM Data Generation
###############################################################################

def generate_negative_binomial_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate overdispersed count data using Negative Binomial distribution.
    
    The Negative Binomial is parameterized to match the Poisson mean but with
    increased variance (overdispersion).
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object. Uses nb_dispersion parameter for variance inflation.
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix with overdispersed counts.
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix and states
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute lambdas
    lambdas_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # For Negative Binomial: Var = μ + μ²/r
    # We want Var = φ * μ, so r = μ / (φ - 1)
    phi = config.nb_dispersion
    
    # Generate informative features
    for t in range(config.n_samples):
        state_t = states[t]
        for f in range(config.n_informative):
            mu = lambdas_inform[state_t, f]
            r = mu / (phi - 1) if phi > 1 else 1e6  # Large r ≈ Poisson
            p = r / (r + mu)
            X[t, f] = rng.negative_binomial(r, p)
    
    # Generate noise features
    if config.n_noise > 0:
        mu_noise = config.lambda_0
        r_noise = mu_noise / (phi - 1) if phi > 1 else 1e6
        p_noise = r_noise / (r_noise + mu_noise)
        X[:, config.n_informative:] = rng.negative_binomial(
            r_noise, p_noise, size=(config.n_samples, config.n_noise)
        )
    
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


###############################################################################
# Gaussian HMM Data Generation
###############################################################################

def generate_gaussian_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate data from a Gaussian HMM.
    
    Similar to the example in the paper attachment:
    y_t | s_t ~ N(μ_{s_t}, I_P)
    
    where μ_1 = μ·1_{p≤15}, μ_2 = 0, μ_3 = -μ·1_{p≤15}
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object with all parameters.
        Uses lambda_0 as the mean μ for state separation.
        Uses delta to control separation: μ = lambda_0 * (1 + delta) / (1 - delta)
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix of shape (n_samples, n_total_features).
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    
    # Sample state sequence
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute state-specific means for informative features
    # For Gaussian, we adapt the Poisson lambda computation
    # to create separated means: μ_k = λ_k for k=1,2,3
    means_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    # Initialize data matrix
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # Generate informative features: y_t ~ N(μ_{s_t}, I)
    for t in range(config.n_samples):
        state_t = states[t]
        # Sample from N(μ_{s_t}, I) where I is identity covariance
        X[t, :config.n_informative] = rng.normal(
            loc=means_inform[state_t, :],
            scale=1.0,  # Unit variance
            size=config.n_informative
        )
    
    # Generate noise features (zero mean across all states)
    if config.n_noise > 0:
        if config.correlated_noise:
            # Generate correlated Gaussian noise
            correlation = config.noise_correlation
            Sigma = np.eye(config.n_noise) * (1 - correlation) + correlation
            X[:, config.n_informative:] = rng.multivariate_normal(
                np.zeros(config.n_noise),
                Sigma,
                size=config.n_samples
            )
        else:
            # Independent Gaussian noise with zero mean, unit variance
            X[:, config.n_informative:] = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(config.n_samples, config.n_noise)
            )
    
    # Create DataFrame with proper column names
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


###############################################################################
# Correlated Noise Generation
###############################################################################

def generate_correlated_noise(n_samples: int,
                               n_features: int,
                               lambda_0: float,
                               correlation: float,
                               rng: np.random.RandomState) -> np.ndarray:
    """
    Generate correlated Poisson noise using Gaussian Copula (NORTA).
    
    Parameters:
    -----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of correlated noise features.
    lambda_0 : float
        Marginal Poisson rate for each feature.
    correlation : float
        Pairwise correlation between noise features.
    rng : np.random.RandomState
        Random number generator.
        
    Returns:
    --------
    np.ndarray
        Correlated count data of shape (n_samples, n_features).
    """
    # Create correlation matrix
    Sigma = np.eye(n_features) * (1 - correlation) + correlation
    
    # Generate correlated Gaussian latent variables
    Z = rng.multivariate_normal(np.zeros(n_features), Sigma, size=n_samples)
    
    # Transform to uniform [0,1] via standard normal CDF
    U = stats.norm.cdf(Z)
    
    # Transform to Poisson using inverse CDF
    # For Poisson, we use the ppf (percent point function)
    X = stats.poisson.ppf(U, lambda_0)
    
    return X


###############################################################################
# High-Level Data Generation Function
###############################################################################

def generate_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate data based on configuration distribution type.
    
    This is a convenience function that routes to the appropriate
    data generation function based on config.distribution_type.
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object with all parameters.
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix of shape (n_samples, n_total_features).
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    if config.distribution_type == "Poisson":
        return generate_poisson_hmm_data(config)
    elif config.distribution_type == "NegativeBinomial":
        return generate_negative_binomial_hmm_data(config)
    elif config.distribution_type == "Gaussian":
        return generate_gaussian_hmm_data(config)
    else:
        raise ValueError(f"Unknown distribution type: {config.distribution_type}")
