import kelly
import timeit
import math
import pytest


def test_kelly_basic():
    kwargs = {
        "price": 2.8,
        "is_back": True,
        "probability": 0.4,
        "other_probabilities": [0.6],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 1000.0,
        "kelly_fraction": 1.0,
        "verbose": False,
    }

    stake = kelly.calculate_kelly_stake(**kwargs)
    print("Basic test - stake:", stake)
    assert stake > 0, "Stake should be greater than 0"
    assert stake <= kwargs["bankroll"], "Stake should not exceed bankroll"


def test_liab_large_than_bank():
    kwargs = {
        "price": 100,
        "is_back": False,
        "probability": 1/100,
        "other_probabilities": [99/100],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 100.0,
        "kelly_fraction": 1.0,
        "verbose": True,
    }

    stake = round(kelly.calculate_kelly_stake(**kwargs))
    print("Large liab test - stake:", stake)
    assert stake < 1e-5, "Stake should be close to 0"



def test_kelly_edge_case_zero_probability():
    kwargs = {
        "price": 10.0,
        "is_back": True,
        "probability": 0.0,
        "other_probabilities": [1.0],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 500.0,
        "kelly_fraction": 1.0,
        "verbose": False,
    }

    stake = round(kelly.calculate_kelly_stake(**kwargs))
    print("Edge case zero probability - stake:", stake)
    assert math.isclose(
        stake, 0, rel_tol=1e-5
    ), "Stake should be close to 0 when probability is 0"


def test_kelly_fractional_stake():
    kwargs = {
        "price": 3.0,
        "is_back": False,
        "probability": 0.3,
        "other_probabilities": [0.4, 0.3],
        "position": 50.0,
        "other_positions": [10.0, -30.0],
        "bankroll": 1000.0,
        "kelly_fraction": 0.5,
        "verbose": False,
    }

    stake = kelly.calculate_kelly_stake(**kwargs)
    print("Fractional Kelly stake test - stake:", stake)
    assert stake > 0, "Stake should be greater than 0"
    assert stake <= 0.5 * kwargs["bankroll"], "Stake should respect the Kelly fraction"


def test_kelly_high_bankroll():
    kwargs = {
        "price": 2,
        "is_back": True,
        "probability": 0.55,
        "other_probabilities": [0.45],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 100000.0,
        "kelly_fraction": 1.0,
        "verbose": False,
    }

    stake = kelly.calculate_kelly_stake(**kwargs)
    print("High bankroll test - stake:", stake)
    assert stake > 0, "Stake should be greater than 0"
    assert stake <= kwargs["bankroll"], "Stake should not exceed bankroll"


def test_kelly_zero_bankroll():
    kwargs = {
        "price": 3.0,
        "is_back": True,
        "probability": 0.3,
        "other_probabilities": [0.7],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 0.5,
        "kelly_fraction": 1.0,
        "verbose": False,
    }

    stake = round(kelly.calculate_kelly_stake(**kwargs))
    print("Zero bankroll test - stake:", stake)
    assert math.isclose(
        stake, 0, rel_tol=1e-9
    ), "Stake should be close to 0 when bankroll is 0"


@pytest.mark.benchmark
def benchmark_kelly():
    kwargs = {
        "price": 4.0,
        "is_back": True,
        "probability": 0.25,
        "other_probabilities": [0.75],
        "position": 0.0,
        "other_positions": [0.0],
        "bankroll": 10000.0,
        "kelly_fraction": 1.0,
        "verbose": False,
    }

    # Benchmarking using timeit
    execution_time = timeit.timeit(
        lambda: kelly.calculate_kelly_stake(**kwargs), number=10000
    )
    print(
        f"Benchmark: Average time over 10000 runs: {execution_time:.6f} ms"
    )
