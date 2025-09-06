use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io::Write;

// Import Pair from the lib crate
use croptimizer::{GameState, Pair};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
enum StartingPairKind {
    #[default]
    YY,
    BB,
    PP,
    YB,
    YP,
    BP,
}

// Starting conditions are wholly defined by the number of each distinct
// StartingPairKind in the Sacred Grove
type StartingCondition = [u32; 6];

fn format_starting_condition(cond: &StartingCondition) -> String {
    let mut s = String::new();
    for i in 0..6 {
        let kind = StartingPairKind::from(i);
        for _ in 0..cond[i] {
            s += &kind.to_string();
        }
    }
    return s;
}

impl fmt::Display for StartingPairKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<usize> for StartingPairKind {
    fn from(value: usize) -> Self {
        match value {
            0 => StartingPairKind::YY,
            1 => StartingPairKind::BB,
            2 => StartingPairKind::PP,
            3 => StartingPairKind::YB,
            4 => StartingPairKind::YP,
            5 => StartingPairKind::BP,
            _ => panic!("Invalid index {} for StartingPairKind", value),
        }
    }
}

impl From<StartingPairKind> for Pair {
    fn from(kind: StartingPairKind) -> Self {
        match kind {
            StartingPairKind::YY => Pair::YY,
            StartingPairKind::BB => Pair::BB,
            StartingPairKind::PP => Pair::PP,
            StartingPairKind::YB => Pair::YB,
            StartingPairKind::YP => Pair::YP,
            StartingPairKind::BP => Pair::BP,
        }
    }
}

/// Iterator that generates all possible ways of placing k balls into n bins.
struct BallsInBinsIter {
    current: Vec<u32>,
    k: u32,
    n: usize,
    finished: bool,
}

impl BallsInBinsIter {
    fn new(n: usize, k: u32) -> Self {
        let mut current = vec![0u32; n];
        if n > 0 {
            current[0] = k;
        }
        Self {
            current,
            k,
            n,
            finished: n == 0 && k > 0,
        }
    }
}

impl Iterator for BallsInBinsIter {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Return the current configuration
        let result = self.current.clone();

        if self.n == 0 {
            self.finished = true;
            return if self.k == 0 { Some(result) } else { None };
        }

        // Generate the next configuration
        // Find the rightmost non-zero element that's not in the last position
        let mut i = self.n - 2;
        loop {
            if self.current[i] > 0 {
                break;
            }
            if i == 0 {
                // We've generated all combinations
                self.finished = true;
                return Some(result);
            }
            i -= 1;
        }

        // Move one ball from position i to position i+1
        self.current[i] -= 1;
        self.current[i + 1] += 1;

        // Move all balls from positions > i+1 back to position i+1
        if i + 2 < self.n {
            let sum: u32 = self.current[i + 2..].iter().sum();
            self.current[i + 1] += sum;
            for j in i + 2..self.n {
                self.current[j] = 0;
            }
        }

        Some(result)
    }
}

/// Generates all possible ways of placing k balls into n bins.
/// This can also be used to model combinations-with-repetition.
/// For example, balls_in_bins(3, 1) returns an iterator that yields:
/// [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
fn balls_in_bins(n: usize, k: u32) -> impl Iterator<Item = Vec<u32>> {
    BallsInBinsIter::new(n, k)
}

const fn factorial(n: u32) -> u32 {
    match n {
        0u32 | 1u32 => 1,
        2u32..=20u32 => factorial(n - 1u32) * n,
        _ => 0,
    }
}

fn weights_to_probs(reduced_y: f32, reduced_b: f32, reduced_p: f32) -> (f32, f32, f32) {
    let y_weight = 1. - reduced_y;
    let b_weight = 1. - reduced_b;
    let p_weight = 1. - reduced_p;
    let sum = y_weight + b_weight + p_weight;
    return (y_weight / sum, b_weight / sum, p_weight / sum);
}

fn plot_prob(k: u32, y: f32, b: f32, p: f32, start: &StartingCondition) -> f32 {
    // Let's assign the 6 indices as follows:
    // [YY, BB, PP, YB, YP, BP]
    // The probability of seeing each kind of plot:
    // P(YY) = yy
    // P(BB) = bb
    // P(PP) = pp
    // P(YB) = 2yb
    // P(YP) = 2yp
    // P(BP) = 2bp

    let ps = [y * y, b * b, p * p, 2. * y * b, 2. * y * p, 2. * b * p];
    let num: u32 = factorial(k);
    let mut denom = 1;
    let mut prob: f32 = 1.;

    for i in 0..6 as usize {
        if start[i] == 0 {
            continue;
        }
        prob *= ps[i].powi(start[i] as i32);
        denom *= factorial(start[i] as u32);
    }
    let coef = num as f32 / denom as f32;

    return coef * prob;
}

fn main() {
    println!("-------------------------------------------------------------------------------");

    for y_r in [0.45, 0.35, 0.25, 0.0] {
        for b_r in [0.45, 0.35, 0.25, 0.0] {
            for p_r in [0.45, 0.35, 0.25, 0.0] {
                // if y_r > b_r {
                //     // If yellow is reduced more than blue, we know it's not optimal,
                //     // because yellow is better htan blue.
                //     continue;
                // }

                let filename = format!(
                    "y{}_b{}_p{}.csv",
                    (y_r * 100.) as u32,
                    (b_r * 100.) as u32,
                    (p_r * 100.) as u32
                );
                let mut file = File::create(&filename).expect("Unable to create file");
                println!("Writing to {}...", filename);

                let (y, b, p) = weights_to_probs(y_r, b_r, p_r);

                // Write CSV header to file
                writeln!(
                file,
                "starting_condition, probability, yellow_lifeforce, blue_lifeforce, purple_lifeforce"
            )
            .expect("Unable to write header");

                for k in [3, 4, 5] {
                    // harvest ordinarily has a 50/50 to be 3 or 4. There is an additional coin flip to add a harvest on the etlas tree ("Bumper Crop").
                    let multi = match k {
                        3 => 0.25,
                        4 => 0.50,
                        5 => 0.25,
                        _ => panic!("huh? only 3, 4, and 5-plot harvests are possible"),
                    };

                    let cases: Vec<Vec<u32>> = balls_in_bins(6, k).collect();
                    let results: Vec<String> = cases
                        .into_par_iter()
                        .map(|case| {
                            let start: StartingCondition =
                                [case[0], case[1], case[2], case[3], case[4], case[5]];

                            // Convert StartingCondition to Vec<Pair>
                            let mut pairs = Vec::new();
                            for i in 0..6 {
                                let kind = StartingPairKind::from(i);
                                let pair = Pair::from(kind);
                                for _ in 0..start[i] {
                                    pairs.push(pair);
                                }
                            }

                            // Create GameState and find optimal strategy
                            let mut game = GameState::from_starting_pairs(&pairs);
                            let optimal = game.find_optimal_strategy();

                            format!(
                                "{}, {}, {:.2}, {:.2}, {:.2}",
                                format_starting_condition(&start),
                                multi * plot_prob(k, y, b, p, &start),
                                optimal.ev_yellow,
                                optimal.ev_blue,
                                optimal.ev_purple
                            )
                        })
                        .collect();

                    for result in results {
                        writeln!(file, "{}", result).expect("Unable to write line");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Define a small epsilon for floating-point comparisons due to precision issues.
    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_plot_prob_k2() {
        // Define example probabilities for the base states
        let y = 0.5;
        let b = 0.3;
        let p = 0.2;

        // Calculate the probabilities of the 6 unique pair types for these base probabilities
        // P(YY) = y^2 = 0.5^2 = 0.25
        // P(BB) = b^2 = 0.3^2 = 0.09
        // P(PP) = p^2 = 0.2^2 = 0.04
        // P(YB) = 2yb = 2 * 0.5 * 0.3 = 0.30
        // P(YP) = 2yp = 2 * 0.5 * 0.2 = 0.20
        // P(BP) = 2bp = 2 * 0.3 * 0.2 = 0.12

        // Test Case 1: A set of two identical pairs (e.g., YY, YY)
        // Counts: [YY:2, BB:0, PP:0, YB:0, YP:0, BP:0]
        // Formula: (2! / (2! * 0! * 0! * 0! * 0! * 0!)) * (P_YY)^2
        //          = (2 / 2) * (0.25)^2 = 1 * 0.0625 = 0.0625
        let start1: StartingCondition = [2, 0, 0, 0, 0, 0];
        let prob1 = plot_prob(2, y, b, p, &start1);
        assert!(
            (prob1 - 0.0625).abs() < EPSILON,
            "Test Case 1 (YY, YY) Failed: Expected 0.0625, Got {}",
            prob1
        );

        // Test Case 2: A set of two different pairs (e.g., YY, BB)
        // Counts: [YY:1, BB:1, PP:0, YB:0, YP:0, BP:0]
        // Formula: (2! / (1! * 1! * 0! * 0! * 0! * 0!)) * (P_YY)^1 * (P_BB)^1
        //          = (2 / 1) * 0.25 * 0.09 = 2 * 0.0225 = 0.045
        let start2: StartingCondition = [1, 1, 0, 0, 0, 0];
        let prob2 = plot_prob(2, y, b, p, &start2);
        assert!(
            (prob2 - 0.045).abs() < EPSILON,
            "Test Case 2 (YY, BB) Failed: Expected 0.045, Got {}",
            prob2
        );

        // Test Case 3: A set with one identical and one mixed-state pair (e.g., YY, YB)
        // Counts: [YY:1, BB:0, PP:0, YB:1, YP:0, BP:0]
        // Formula: (2! / (1! * 1!)) * (P_YY)^1 * (P_YB)^1
        //          = (2 / 1) * 0.25 * 0.30 = 2 * 0.075 = 0.15
        let start3: StartingCondition = [1, 0, 0, 1, 0, 0];
        let prob3 = plot_prob(2, y, b, p, &start3);
        assert!(
            (prob3 - 0.15).abs() < EPSILON,
            "Test Case 3 (YY, YB) Failed: Expected 0.15, Got {}",
            prob3
        );

        // Test Case 4: A set of two different mixed-state pairs (e.g., YB, YP)
        // Counts: [YY:0, BB:0, PP:0, YB:1, YP:1, BP:0]
        // Formula: (2! / (1! * 1!)) * (P_YB)^1 * (P_YP)^1
        //          = (2 / 1) * 0.30 * 0.20 = 2 * 0.06 = 0.12
        let start4: StartingCondition = [0, 0, 0, 1, 1, 0];
        let prob4 = plot_prob(2, y, b, p, &start4);
        assert!(
            (prob4 - 0.12).abs() < EPSILON,
            "Test Case 4 (YB, YP) Failed: Expected 0.12, Got {}",
            prob4
        );

        // Test Case 5: A set of two identical mixed-state pairs (e.g., YB, YB)
        // Counts: [YY:0, BB:0, PP:0, YB:2, YP:0, BP:0]
        // Formula: (2! / (2!)) * (P_YB)^2
        //          = (2 / 2) * (0.30)^2 = 1 * 0.09 = 0.09
        let start5: StartingCondition = [0, 0, 0, 2, 0, 0];
        let prob5 = plot_prob(2, y, b, p, &start5);
        assert!(
            (prob5 - 0.09).abs() < EPSILON,
            "Test Case 5 (YB, YB) Failed: Expected 0.09, Got {}",
            prob5
        );
    }

    #[test]
    fn test_plot_prob_sum_k3() {
        // Test that the sum of plot_prob over all possible combinations for k=3 equals 1
        let y: f32 = 0.5;
        let b: f32 = 0.3;
        let p: f32 = 0.2;

        // Verify that y + b + p = 1
        assert!(
            (y + b + p - 1.0).abs() < EPSILON,
            "Base probabilities must sum to 1"
        );

        let mut total_prob: f32 = 0.0;

        for combination in balls_in_bins(6, 3) {
            let start: StartingCondition = [
                combination[0],
                combination[1],
                combination[2],
                combination[3],
                combination[4],
                combination[5],
            ];
            let prob = plot_prob(3, y, b, p, &start);
            total_prob += prob;
        }

        assert!(
            (total_prob - 1.0).abs() < EPSILON,
            "Sum of probabilities for k=3 should equal 1.0, got {}",
            total_prob
        );
    }

    #[test]
    fn test_plot_prob_sum_k4() {
        // Test that the sum of plot_prob over all possible combinations for k=4 equals 1
        let y: f32 = 0.5;
        let b: f32 = 0.3;
        let p: f32 = 0.2;

        // Verify that y + b + p = 1
        assert!(
            (y + b + p - 1.0).abs() < EPSILON,
            "Base probabilities must sum to 1"
        );

        let mut total_prob: f32 = 0.0;

        for combination in balls_in_bins(6, 4) {
            let start: StartingCondition = [
                combination[0],
                combination[1],
                combination[2],
                combination[3],
                combination[4],
                combination[5],
            ];
            let prob = plot_prob(4, y, b, p, &start);
            total_prob += prob;
        }

        assert!(
            (total_prob - 1.0).abs() < EPSILON,
            "Sum of probabilities for k=4 should equal 1.0, got {}",
            total_prob
        );
    }

    #[test]
    fn test_plot_prob_sum_k5() {
        // Test that the sum of plot_prob over all possible combinations for k=5 equals 1
        let y: f32 = 0.5;
        let b: f32 = 0.3;
        let p: f32 = 0.2;

        // Verify that y + b + p = 1
        assert!(
            (y + b + p - 1.0).abs() < EPSILON,
            "Base probabilities must sum to 1"
        );

        let mut total_prob: f32 = 0.0;

        for combination in balls_in_bins(6, 5) {
            let start: StartingCondition = [
                combination[0],
                combination[1],
                combination[2],
                combination[3],
                combination[4],
                combination[5],
            ];
            let prob = plot_prob(5, y, b, p, &start);
            total_prob += prob;
        }

        assert!(
            (total_prob - 1.0).abs() < EPSILON,
            "Sum of probabilities for k=5 should equal 1.0, got {}",
            total_prob
        );
    }

    #[test]
    fn test_plot_prob_sum_different_base_probs() {
        // Test with different base probabilities to ensure the sum still equals 1
        let test_cases: Vec<(f32, f32, f32)> = vec![
            (0.6, 0.3, 0.1),
            (0.4, 0.4, 0.2),
            (0.7, 0.2, 0.1),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ];

        for (y, b, p) in test_cases {
            // Verify that y + b + p = 1
            assert!(
                (y + b + p - 1.0).abs() < EPSILON,
                "Base probabilities ({}, {}, {}) must sum to 1",
                y,
                b,
                p
            );

            // Test k=3
            let mut total_prob: f32 = 0.0;

            for combination in balls_in_bins(6, 3) {
                let start: StartingCondition = [
                    combination[0],
                    combination[1],
                    combination[2],
                    combination[3],
                    combination[4],
                    combination[5],
                ];
                let prob = plot_prob(3, y, b, p, &start);
                total_prob += prob;
            }

            assert!(
                (total_prob - 1.0).abs() < EPSILON,
                "Sum of probabilities for k=3 with base probs ({}, {}, {}) should equal 1.0, got {}",
                y, b, p, total_prob
            );
        }
    }
}
