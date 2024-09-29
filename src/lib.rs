use argmin::core::Executor;
use argmin::solver::goldensectionsearch::GoldenSectionSearch;
use pyo3::prelude::*;

mod objectives;

use crate::objectives::LogExpectedWealthObjective;

///     Calculate the optimal fractional Kelly stake when placing a single bet on an event that has one winner
///
///     :param price: The price at which the bet is being placed
///     :param is_back: Whether you are backing (True) or laying (False)
///     :param probability: Your fair probability for the outcome you are betting on
///     :param other_probabilities: A list of probabilities for the other possible outcomes. sum(other_probabilities) + probability should be equal to 1 subject to floating-point error but this is not yet enforced
///     :param position: Your position - i.e. how much you are currently standing to win or lose - on the outcome you are betting on
///     :param other_positions: A list of positions corresponding to the other possible outcomes. It is assumed there is a 1:1 correspondence between the elements of this list and other_probabilities
///     :param bankroll: Your notional Kelly bankroll
///     :param kelly_fraction: A fraction to multiply the optimal stake by. Defaults to 1
///     :param verbose: Whether to generate log statements when numerically optimising the stake. Defaults to False
///
///     :return: The optimal fractional Kelly stake
#[pyfunction]
#[pyo3(signature = (price, is_back, probability, other_probabilities, position, other_positions, bankroll, kelly_fraction = 1.0, verbose = false))]
fn calculate_kelly_stake(
    price: f64,
    is_back: bool,
    probability: f64,
    other_probabilities: Vec<f64>,
    position: f64,
    other_positions: Vec<f64>,
    bankroll: f64,
    kelly_fraction: f64,
    verbose: bool,
) -> PyResult<f64> {
    let cost = LogExpectedWealthObjective {
        price,
        is_back,
        probability,
        other_probabilities,
        position,
        other_positions,
        bankroll,
    };

    let lower_bound = 0.;
    let upper_bound = bankroll;

    let solver = GoldenSectionSearch::new(lower_bound, upper_bound).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Error creating solver: {}", e))
    })?;

    let result = Executor::new(cost, solver)
        .configure(|state| state.param(bankroll / 100.).max_iters(500))
        .run()
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Optimization failed: {}", e))
        })?;

    // Retrieve the optimal stake
    let optimal_stake = result.state().param.unwrap_or(0.) * kelly_fraction;

    if verbose {
        println!("Optimal stake: {}", optimal_stake);
    }

    Ok(optimal_stake)
}

/// Fast Kelly staking calculations for a range of scenarios .
#[pymodule]
fn kelly(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_kelly_stake, m)?)?;
    Ok(())
}
