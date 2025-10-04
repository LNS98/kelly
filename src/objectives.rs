use argmin::core::{CostFunction, Error};

/// The objective function that calculates the negative log expected wealth
pub struct LogExpectedWealthObjective {
    pub price: f64,
    pub is_back: bool,
    pub probability: f64,
    pub other_probabilities: Vec<f64>,
    pub position: f64,
    pub other_positions: Vec<f64>,
    pub bankroll: f64,
}

impl CostFunction for LogExpectedWealthObjective {
    type Param = f64;
    type Output = f64;

    fn cost(&self, stake: &Self::Param) -> Result<Self::Output, Error> {
        // We want to minimize the negative expected log wealth
        let result = -calculate_log_expected_wealth(
            *stake,
            self.price,
            self.is_back,
            self.probability,
            &self.other_probabilities,
            self.position,
            &self.other_positions,
            self.bankroll,
        );
        Ok(result)
    }
}

fn calculate_log_expected_wealth(
    stake: f64,
    price: f64,
    is_back: bool,
    probability: f64,
    other_probabilities: &[f64],
    position: f64,
    other_positions: &[f64],
    bankroll: f64,
) -> f64 {
    let expected_log_wealth = if is_back {
        let wealth_if_back = bankroll + position + stake * (price - 1.0);
        if wealth_if_back <= 0.0 {
            return f64::NEG_INFINITY;
        }
        probability * wealth_if_back.ln()
            + (other_positions
                .iter()
                .zip(other_probabilities.iter())
                .map(|(other_position, other_probability)| {
                    other_probability * (bankroll + other_position - stake).ln()
                })
                .sum::<f64>())
    } else {
        let wealth_if_lay = bankroll + position - stake * (price - 1.0);
        if wealth_if_lay <= 0.0 {
            return f64::NEG_INFINITY;
        }
        probability * wealth_if_lay.ln()
            + (other_positions
                .iter()
                .zip(other_probabilities.iter())
                .map(|(other_position, other_probability)| {
                    other_probability * (bankroll + other_position + stake).ln()
                })
                .sum::<f64>())
    };

    expected_log_wealth
}
