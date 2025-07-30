use core::f64;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Plot {
    Y,
    B,
    P,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Pair {
    YY,
    YP,
    YB,
    BB,
    BP,
    PP,
    YX,
    BX,
    PX,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Choice {
    YFromYY,
    YFromYB,
    YFromYP,
    PFromYP,
    PFromPP,
    PFromPB,
    BFromBB,
    BFromBY,
    BFromBP,
    YFromYX,
    BFromBX,
    PFromPX,
}

#[derive(Debug)]
struct Config {
    yellow_price: f64,
    blue_price: f64,
    purple_price: f64,
    wilt_rate: f64,
    t2_transition: f64,
    t3_transition: f64,
    t4_transition: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            yellow_price: 3900.0,
            blue_price: 5300.0,
            purple_price: 5900.0,
            wilt_rate: 0.4,
            t2_transition: 0.25,
            t3_transition: 0.20,
            t4_transition: 0.03,
        }
    }
}

impl Config {
    fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Path::new("conf.toml");

        if !config_path.exists() {
            eprintln!("Warning: conf.toml not found, using default values.");
            return Ok(Self::default());
        }

        let content = fs::read_to_string(config_path)?;
        Self::parse_toml(&content)
    }

    fn parse_toml(content: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = Self::default();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();

                // Remove quotes if present
                let value = if value.starts_with('"') && value.ends_with('"') {
                    &value[1..value.len() - 1]
                } else {
                    value
                };

                match key {
                    "yellow_price" => config.yellow_price = value.parse()?,
                    "blue_price" => config.blue_price = value.parse()?,
                    "purple_price" => config.purple_price = value.parse()?,
                    "wilt_rate" => config.wilt_rate = value.parse()?,
                    "t2_transition" => config.t2_transition = value.parse()?,
                    "t3_transition" => config.t3_transition = value.parse()?,
                    "t4_transition" => config.t4_transition = value.parse()?,
                    _ => eprintln!("Warning: Unknown config key '{}'", key),
                }
            }
        }

        // Validate config values
        config.validate()?;
        println!("Using values: {:?}", config);
        Ok(config)
    }

    fn validate(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.yellow_price <= 0.0 || self.blue_price <= 0.0 || self.purple_price <= 0.0 {
            return Err("All prices must be positive".into());
        }

        if !(0.0..=1.0).contains(&self.wilt_rate) {
            return Err("Wilt rate must be between 0.0 and 1.0".into());
        }

        if !(0.0..=1.0).contains(&self.t2_transition)
            || !(0.0..=1.0).contains(&self.t3_transition)
            || !(0.0..=1.0).contains(&self.t4_transition)
        {
            return Err("All transition rates must be between 0.0 and 1.0".into());
        }

        Ok(())
    }
}

impl Pair {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "YY" => Some(Self::YY),
            "YP" | "PY" => Some(Self::YP),
            "YB" | "BY" => Some(Self::YB),
            "BB" => Some(Self::BB),
            "BP" | "PB" => Some(Self::BP),
            "PP" => Some(Self::PP),
            "YX" | "XY" => Some(Self::YX),
            "BX" | "XB" => Some(Self::BX),
            "PX" | "XP" => Some(Self::PX),
            _ => None,
        }
    }

    fn is_single_plot(self) -> bool {
        matches!(self, Self::YX | Self::BX | Self::PX)
    }
}

// Precomputed lookup tables
struct LookupTables {
    evs: HashMap<Plot, Vec<f64>>,
    choice_from_pair: HashMap<Pair, Vec<Choice>>,
    pair_from_choice: HashMap<Choice, Pair>,
    upgrade_from_choice: HashMap<Choice, Plot>,
    leftover_from_choice: HashMap<Choice, Option<Choice>>,
    wilt_rate: f64,
}

static TABLES: OnceLock<LookupTables> = OnceLock::new();

fn get_tables() -> &'static LookupTables {
    TABLES.get_or_init(|| {
        let config = Config::load().unwrap_or_else(|e| {
            eprintln!("Error loading config: {}", e);
            eprintln!("Using default values");
            Config::default()
        });

        let mut tables = LookupTables {
            evs: calculate_evs(&config),
            choice_from_pair: HashMap::new(),
            pair_from_choice: HashMap::new(),
            upgrade_from_choice: HashMap::new(),
            leftover_from_choice: HashMap::new(),
            wilt_rate: config.wilt_rate,
        };

        tables.initialize_choice_mappings();
        tables.initialize_pair_mappings();
        tables.initialize_upgrade_mappings();
        tables.initialize_leftover_mappings();

        tables
    })
}

impl LookupTables {
    fn initialize_choice_mappings(&mut self) {
        let mappings = [
            (Pair::YY, vec![Choice::YFromYY]),
            (Pair::YP, vec![Choice::YFromYP, Choice::PFromYP]),
            (Pair::YB, vec![Choice::YFromYB, Choice::BFromBY]),
            (Pair::BB, vec![Choice::BFromBB]),
            (Pair::BP, vec![Choice::BFromBP, Choice::PFromPB]),
            (Pair::PP, vec![Choice::PFromPP]),
            (Pair::YX, vec![Choice::YFromYX]),
            (Pair::BX, vec![Choice::BFromBX]),
            (Pair::PX, vec![Choice::PFromPX]),
        ];

        self.choice_from_pair.extend(mappings);
    }

    fn initialize_pair_mappings(&mut self) {
        let mappings = [
            (Choice::YFromYY, Pair::YY),
            (Choice::YFromYB, Pair::YB),
            (Choice::YFromYP, Pair::YP),
            (Choice::PFromYP, Pair::YP),
            (Choice::PFromPP, Pair::PP),
            (Choice::PFromPB, Pair::BP),
            (Choice::BFromBB, Pair::BB),
            (Choice::BFromBY, Pair::YB),
            (Choice::BFromBP, Pair::BP),
            (Choice::YFromYX, Pair::YX),
            (Choice::PFromPX, Pair::PX),
            (Choice::BFromBX, Pair::BX),
        ];

        self.pair_from_choice.extend(mappings);
    }

    fn initialize_upgrade_mappings(&mut self) {
        let mappings = [
            (Choice::YFromYY, Plot::Y),
            (Choice::YFromYB, Plot::Y),
            (Choice::YFromYP, Plot::Y),
            (Choice::YFromYX, Plot::Y),
            (Choice::PFromYP, Plot::P),
            (Choice::PFromPP, Plot::P),
            (Choice::PFromPB, Plot::P),
            (Choice::PFromPX, Plot::P),
            (Choice::BFromBB, Plot::B),
            (Choice::BFromBY, Plot::B),
            (Choice::BFromBP, Plot::B),
            (Choice::BFromBX, Plot::B),
        ];

        self.upgrade_from_choice.extend(mappings);
    }

    fn initialize_leftover_mappings(&mut self) {
        let mappings = [
            (Choice::YFromYY, Some(Choice::YFromYX)),
            (Choice::YFromYB, Some(Choice::BFromBX)),
            (Choice::YFromYP, Some(Choice::PFromPX)),
            (Choice::PFromYP, Some(Choice::YFromYX)),
            (Choice::PFromPP, Some(Choice::PFromPX)),
            (Choice::PFromPB, Some(Choice::BFromBX)),
            (Choice::BFromBB, Some(Choice::BFromBX)),
            (Choice::BFromBY, Some(Choice::YFromYX)),
            (Choice::BFromBP, Some(Choice::PFromPX)),
            (Choice::YFromYX, None),
            (Choice::BFromBX, None),
            (Choice::PFromPX, None),
        ];

        self.leftover_from_choice.extend(mappings);
    }
}

#[derive(Debug, Clone)]
struct PlotState {
    count: u32,
    upgrades: HashMap<Plot, u32>,
}

impl PlotState {
    fn new() -> Self {
        Self {
            count: 0,
            upgrades: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct SeedLog {
    t1: usize,
    t2: usize,
    t3: usize,
    t4: usize,
    upgrades: usize,
    plot_type: Plot,
}
#[derive(Clone)]
struct GameState {
    plots: HashMap<Pair, PlotState>,
    harvested_value: f64,
    seedlog: Vec<SeedLog>,
    reset: bool,
}

impl GameState {
    fn new() -> Self {
        Self {
            plots: HashMap::new(),
            harvested_value: 0.0,
            seedlog: Vec::new(),
            reset: false,
        }
    }

    fn from_starting_pairs(pairs: &[Pair]) -> Self {
        let mut game = Self::new();
        for &pair in pairs {
            game.plots.entry(pair).or_insert_with(PlotState::new).count += 1;
        }
        game
    }

    fn get_available_choices(&self) -> Vec<Choice> {
        let tables = get_tables();
        let mut choices = Vec::new();

        for (&pair, plot_state) in &self.plots {
            if plot_state.count > 0 {
                if let Some(pair_choices) = tables.choice_from_pair.get(&pair) {
                    choices.extend(pair_choices);
                }
            }
        }
        choices
    }

    fn is_terminal(&self) -> bool {
        self.get_available_choices().is_empty()
    }

    fn calculate_harvest_value(&self, choice: Choice) -> f64 {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];
        let upgrade_plot = tables.upgrade_from_choice[&choice];

        let current_upgrades = self.plots[&pair]
            .upgrades
            .get(&upgrade_plot)
            .copied()
            .unwrap_or(0);

        let plot_evs = &tables.evs[&upgrade_plot];
        if (current_upgrades as usize) < plot_evs.len() {
            plot_evs[current_upgrades as usize]
        } else {
            *plot_evs.last().unwrap()
        }
    }

    fn choose(&mut self, choice: Choice, wither: bool) -> UndoInfo {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];

        let value_delta = self.calculate_harvest_value(choice);
        self.harvested_value += value_delta;

        // Store upgrade state before removing
        let harvested_plot_upgrades = self.plots[&pair].upgrades.clone();

        // Remove one plot of this pair
        self.plots.get_mut(&pair).unwrap().count -= 1;
        let pair_was_deleted = self.plots[&pair].count == 0;
        if pair_was_deleted {
            self.plots.remove(&pair);
        }

        // Add leftover plot if it doesn't wither
        let leftover_pair = self.handle_leftover_plot(choice, wither, &harvested_plot_upgrades);

        // Upgrade all remaining plots of other types
        let upgraded_plots = self.upgrade_remaining_plots(choice);

        UndoInfo {
            value_delta,
            pair,
            pair_was_deleted,
            leftover_pair,
            upgraded_plots,
            harvested_plot_upgrades,
        }
    }

    fn handle_leftover_plot(
        &mut self,
        choice: Choice,
        wither: bool,
        harvested_plot_upgrades: &HashMap<Plot, u32>,
    ) -> Option<Pair> {
        if wither {
            return None;
        }

        let tables = get_tables();
        if let Some(&Some(leftover_choice)) = tables.leftover_from_choice.get(&choice) {
            let leftover_pair = tables.pair_from_choice[&leftover_choice];
            let plot_state = self
                .plots
                .entry(leftover_pair)
                .or_insert_with(PlotState::new);
            let was_new = plot_state.count == 0;
            plot_state.count += 1;

            // Transfer upgrades from the harvested plot to the leftover plot
            // Only if this is a new plot
            if was_new {
                plot_state.upgrades = harvested_plot_upgrades.clone();
            }

            Some(leftover_pair)
        } else {
            None
        }
    }

    fn upgrade_remaining_plots(&mut self, choice: Choice) -> Vec<(Pair, Plot)> {
        let tables = get_tables();
        let upgrade_plot = tables.upgrade_from_choice[&choice];
        let mut upgraded_plots = Vec::new();

        let pairs: Vec<_> = self.plots.keys().cloned().collect();
        for pair in pairs {
            if self.plots[&pair].count > 0 {
                for &plot_type in &[Plot::Y, Plot::B, Plot::P] {
                    if plot_type != upgrade_plot {
                        *self
                            .plots
                            .get_mut(&pair)
                            .unwrap()
                            .upgrades
                            .entry(plot_type)
                            .or_insert(0) += 1;
                        upgraded_plots.push((pair, plot_type));
                    }
                }
            }
        }
        upgraded_plots
    }

    fn undo_choice(&mut self, undo_info: UndoInfo) {
        // Restore harvest value
        self.harvested_value -= undo_info.value_delta;

        // Undo upgrades
        for (pair, plot_type) in &undo_info.upgraded_plots {
            if let Some(plot_state) = self.plots.get_mut(pair) {
                if let Some(upgrade_count) = plot_state.upgrades.get_mut(plot_type) {
                    *upgrade_count -= 1;
                }
            }
        }

        // Remove leftover plot if it was added
        if let Some(leftover_pair) = undo_info.leftover_pair {
            if let Some(plot_state) = self.plots.get_mut(&leftover_pair) {
                plot_state.count -= 1;
                if plot_state.count == 0 {
                    self.plots.remove(&leftover_pair);
                }
            }
        }

        // Restore the original pair
        if undo_info.pair_was_deleted {
            let mut plot_state = PlotState::new();
            plot_state.count = 1;
            plot_state.upgrades = undo_info.harvested_plot_upgrades;
            self.plots.insert(undo_info.pair, plot_state);
        } else {
            self.plots.get_mut(&undo_info.pair).unwrap().count += 1;
        }
    }

    fn compute_optimal_value(&mut self) -> f64 {
        if self.is_terminal() {
            return self.harvested_value;
        }

        let mut best_ev = f64::NEG_INFINITY;
        let available_choices = self.get_available_choices();

        for choice in available_choices {
            let ev = self.calculate_choice_expected_value(choice);
            if ev > best_ev {
                best_ev = ev;
            }
        }

        best_ev
    }

    fn calculate_choice_expected_value(&mut self, choice: Choice) -> f64 {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];

        if pair.is_single_plot() {
            // Single plot, no wither possibility
            let undo_info = self.choose(choice, false);
            let value = self.compute_optimal_value();
            self.undo_choice(undo_info);
            value
        } else {
            // Paired plot, calculate wither probability
            let wilt_chance = tables.wilt_rate;

            // Calculate EV for no wither case
            let undo_info_no_wither = self.choose(choice, false);
            let no_wilt_value = self.compute_optimal_value();
            self.undo_choice(undo_info_no_wither);

            // Calculate EV for wither case
            let undo_info_wither = self.choose(choice, true);
            let wilt_value = self.compute_optimal_value();
            self.undo_choice(undo_info_wither);

            // Expected value
            wilt_chance * wilt_value + (1.0 - wilt_chance) * no_wilt_value
        }
    }

    fn find_optimal_strategy(&mut self) -> OptimalMove {
        if self.is_terminal() {
            return OptimalMove::terminal(self.harvested_value);
        }

        let mut best_choice = None;
        let mut best_ev = f64::NEG_INFINITY;
        let mut best_no_wilt_ev = f64::NEG_INFINITY;
        let mut best_wilt_ev = None;

        let available_choices = self.get_available_choices();
        for choice in available_choices {
            let move_analysis = self.analyze_choice(choice);

            // use gteq over gt because we want to favor picking the last plot
            // in the choices list which is optimal. The plot we just created
            // by harvesting will be at the end of the choices array, and we
            // want to favor it over other optimal moves.
            if move_analysis.expected_value >= best_ev {
                best_ev = move_analysis.expected_value;
                best_choice = Some(choice);
                best_no_wilt_ev = move_analysis.no_wilt_value;
                best_wilt_ev = move_analysis.wilt_value;
            }
        }

        OptimalMove {
            choice: best_choice,
            expected_value: best_ev,
            no_wilt_value: best_no_wilt_ev,
            wilt_value: best_wilt_ev,
        }
    }

    fn analyze_choice(&mut self, choice: Choice) -> MoveAnalysis {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];

        if pair.is_single_plot() {
            // Single plot, no wither possibility
            let undo_info = self.choose(choice, false);
            let value = self.compute_optimal_value();
            self.undo_choice(undo_info);

            MoveAnalysis {
                expected_value: value,
                no_wilt_value: value,
                wilt_value: None,
            }
        } else {
            let wilt_chance = tables.wilt_rate;

            // Calculate EV for no wither case
            let undo_info_no_wither = self.choose(choice, false);
            let no_wilt_value = self.compute_optimal_value();
            self.undo_choice(undo_info_no_wither);

            // Calculate EV for wither case
            let undo_info_wither = self.choose(choice, true);
            let wilt_value = self.compute_optimal_value();
            self.undo_choice(undo_info_wither);

            // Expected value
            let expected = wilt_chance * wilt_value + (1.0 - wilt_chance) * no_wilt_value;

            MoveAnalysis {
                expected_value: expected,
                no_wilt_value,
                wilt_value: Some(wilt_value),
            }
        }
    }

    fn get_total_value(&self) -> f64 {
        self.harvested_value
    }

    fn get_user_wither_input(&mut self) -> bool {
        print!("Did the other plot wither? (y/N): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim().to_lowercase();
        if input == "reset" {
            self.reset = true;
            println!("reset command received; starting a new game.");
            return false;
        }
        match input.as_str() {
            "y" | "yes" => {
                println!("Plot DID wither");
                true
            }
            _ => {
                println!("Plot did NOT wither");
                false
            }
        }
    }

    fn wait_for_user_input(&mut self) {
        println!("\nPress enter to continue.");
        let mut input = String::new();

        io::stdin().read_line(&mut input).unwrap();
        if input == "reset" {
            self.reset = true;
            println!("reset command received; starting a new game.");
        }
    }
}

struct UndoInfo {
    value_delta: f64,
    pair: Pair,
    pair_was_deleted: bool,
    leftover_pair: Option<Pair>,
    upgraded_plots: Vec<(Pair, Plot)>,
    harvested_plot_upgrades: HashMap<Plot, u32>,
}

struct MoveAnalysis {
    expected_value: f64,
    no_wilt_value: f64,
    wilt_value: Option<f64>,
}

struct OptimalMove {
    choice: Option<Choice>,
    expected_value: f64,
    no_wilt_value: f64,
    wilt_value: Option<f64>,
}

impl OptimalMove {
    fn terminal(value: f64) -> Self {
        Self {
            choice: None,
            expected_value: value,
            no_wilt_value: value,
            wilt_value: None,
        }
    }
}

// Calculate expected values
fn calculate_evs(config: &Config) -> HashMap<Plot, Vec<f64>> {
    const SEEDS_PER_PLOT: f64 = 23.0;
    const TIER_YIELD: [f64; 4] = [
        0.0073 * 12.7 * SEEDS_PER_PLOT,
        1.85 * 12.7 * SEEDS_PER_PLOT,
        47.0 * 12.7 * SEEDS_PER_PLOT,
        234.0 * 12.7 / 2.0 * SEEDS_PER_PLOT,
    ];

    let prices = [
        (Plot::Y, 1.0 / config.yellow_price),
        (Plot::B, 1.0 / config.blue_price),
        (Plot::P, 1.0 / config.purple_price),
    ];

    let mut evs = HashMap::new();
    for (plot, _) in prices {
        evs.insert(plot, Vec::new());
    }

    // Transition matrix P - calculated from config values
    let transition_matrix = [
        [1.0 - config.t2_transition, config.t2_transition, 0.00, 0.00],
        [0.00, 1.0 - config.t3_transition, config.t3_transition, 0.00],
        [0.00, 0.00, 1.0 - config.t4_transition, config.t4_transition],
        [0.00, 0.00, 0.00, 1.00],
    ];

    let mut state = [1.0, 0.0, 0.0, 0.0];

    for _ in 0..10 {
        let lifeforce: f64 = state
            .iter()
            .zip(TIER_YIELD.iter())
            .map(|(a, b)| a * b)
            .sum();

        for (plot, price) in prices {
            evs.get_mut(&plot).unwrap().push(lifeforce * price);
        }

        // Matrix multiplication: state = state @ transition_matrix
        let mut new_state = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                new_state[j] += state[i] * transition_matrix[i][j];
            }
        }
        state = new_state;
    }

    evs
}

fn parse_input_pairs(input: &str) -> Result<Vec<Pair>, String> {
    let pairs: Vec<&str> = input.trim().split_whitespace().collect();
    let mut parsed_pairs = Vec::new();

    for pair_str in pairs {
        if pair_str.is_empty() {
            continue;
        }

        match Pair::from_str(pair_str) {
            Some(pair) => parsed_pairs.push(pair),
            None => {
                return Err(format!(
                    "Unknown pair '{}'. Valid pairs are: YY, YP, YB, BB, BP, PP, YX, BX, PX",
                    pair_str
                ));
            }
        }
    }

    if parsed_pairs.is_empty() {
        Err("No valid pairs entered".to_string())
    } else {
        Ok(parsed_pairs)
    }
}

fn handle_turn(game: &mut GameState, turn: u32) -> bool {
    println!("=== TURN {} ===", turn);
    println!("Current value: {:.1}", game.get_total_value());

    let optimal_move = game.find_optimal_strategy();

    match optimal_move.choice {
        Some(choice) => {
            // Display expected value information
            if let Some(wilt) = optimal_move.wilt_value {
                println!(
                    "EV: {:.1} div. [{:.1}, {:.1}]",
                    optimal_move.expected_value, wilt, optimal_move.no_wilt_value
                );
            } else {
                println!("EV: {:.1}", optimal_move.expected_value);
            }
            println!("\nOPTIMAL MOVE: {:?}\n", choice);

            // Check if this is a paired plot and get wither result
            let tables = get_tables();
            let pair = tables.pair_from_choice[&choice];
            let wither = if pair.is_single_plot() {
                game.wait_for_user_input();
                false
            } else {
                game.get_user_wither_input()
            };
            if game.reset {
                return false;
            }

            // Make the move
            game.choose(choice, wither);
            true
        }
        None => {
            println!("No optimal move found (this shouldn't happen)");
            false
        }
    }
}

fn get_new_game_input() -> Option<GameState> {
    println!("NEW GAME: Enter the plots in the harvest as pairs of colors, e.g.: YY YP YB BB");
    print!("> ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match parse_input_pairs(&input) {
        Ok(pairs) => {
            println!();
            Some(GameState::from_starting_pairs(&pairs))
        }
        Err(error) => {
            eprintln!("{}", error);
            eprintln!("Try again or ctrl-c to exit.");
            None
        }
    }
}

fn main() {
    println!("=== Game State Optimizer ===\n");

    let mut turn = 1;
    let mut current_game: Option<GameState> = None;
    let _ = Config::load().unwrap_or_else(|e| {
        eprintln!("Error loading config: {}", e);
        Config::default()
    });

    // Game loop
    loop {
        match current_game {
            Some(ref mut game) => {
                if game.is_terminal() {
                    println!("\nGame over! No more moves available.");
                    println!("Final value: {:.1}", game.get_total_value());
                    current_game = None;
                    turn = 1;
                    continue;
                }

                if handle_turn(game, turn) {
                    turn += 1;
                } else {
                    turn = 1;
                    current_game = None;
                    continue;
                }
            }
            None => {
                current_game = get_new_game_input();
            }
        }
    }
}
