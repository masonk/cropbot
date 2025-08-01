use core::f64;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;
use toml;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Plot {
    Y,
    B,
    P,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pair {
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
pub enum Choice {
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

#[derive(Debug, Deserialize)]
pub struct Config {
    // Note that lifeforce is always priced in inverse divs
    pub yellow_price: f64,
    pub blue_price: f64,
    pub purple_price: f64,
    pub wilt_rate: f64,
    pub t2_transition: f64,
    pub t3_transition: f64,
    pub t4_transition: f64,
    pub doubling_scarab: Option<bool>,
    pub map_iiq: f64,
    pub map_pack_size: f64,
    pub atlas_lifeforce_quant: f64,
    pub atlas_additional_monster_chance: f64,
    pub atlas_monster_dupe_chance: f64,
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
            doubling_scarab: Some(true),
            map_iiq: 300.,
            map_pack_size: 110.,
            atlas_lifeforce_quant: 18.,
            atlas_additional_monster_chance: 10.,
            atlas_monster_dupe_chance: 6.,
        }
    }
}

impl Config {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Path::new("conf.toml");

        if !config_path.exists() {
            eprintln!("Warning: conf.toml not found in the current working directory; using default values.");
            let def = Self::default();
            println!("{def:?}");
            return Ok(def);
        }

        let config = fs::read_to_string(config_path)?;
        let config: Config = toml::from_str(&config)?;
        config.validate()?;
        println!("Valid conf.toml found. Using its values.");
        println!("{config:?}");
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
    pub fn from_str(s: &str) -> Option<Self> {
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

    pub fn is_single_plot(self) -> bool {
        matches!(self, Self::YX | Self::BX | Self::PX)
    }
}

// Precomputed lookup tables
pub struct LookupTables {
    pub evs: Vec<f64>, // Expected lifeforce per upgrade tier (independent of color)
    pub choice_from_pair: HashMap<Pair, Vec<Choice>>,
    pub pair_from_choice: HashMap<Choice, Pair>,
    pub upgrade_from_choice: HashMap<Choice, Plot>,
    pub leftover_from_choice: HashMap<Choice, Option<Choice>>,
    pub wilt_rate: f64,
    pub yellow_price_per_div: f64,
    pub blue_price_per_div: f64,
    pub purple_price_per_div: f64,
}

static TABLES: OnceLock<LookupTables> = OnceLock::new();

pub fn get_tables() -> &'static LookupTables {
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
            yellow_price_per_div: 1.0 / config.yellow_price,
            blue_price_per_div: 1.0 / config.blue_price,
            purple_price_per_div: 1.0 / config.purple_price,
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

    pub fn lifeforce_price_per_div(&self, plot: Plot) -> f64 {
        match plot {
            Plot::Y => self.yellow_price_per_div,
            Plot::B => self.blue_price_per_div,
            Plot::P => self.purple_price_per_div,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlotState {
    pub count: u32,
}

impl PlotState {
    pub fn new() -> Self {
        Self { count: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct SeedLog {
    pub t1: usize,
    pub t2: usize,
    pub t3: usize,
    pub t4: usize,
    pub upgrades: usize,
    pub plot_type: Plot,
}
#[derive(Clone)]
pub struct GameState {
    pub plots: HashMap<Pair, PlotState>,
    pub harvested_value: f64,               // Total price value harvested
    pub harvested_yellow: f64,              // Yellow lifeforce harvested
    pub harvested_blue: f64,                // Blue lifeforce harvested
    pub harvested_purple: f64,              // Purple lifeforce harvested
    pub color_upgrades: HashMap<Plot, u32>, // Upgrade tier for each color
    pub seedlog: Vec<SeedLog>,
    pub reset: bool,
}

impl GameState {
    pub fn new() -> Self {
        let mut color_upgrades = HashMap::new();
        color_upgrades.insert(Plot::Y, 0);
        color_upgrades.insert(Plot::B, 0);
        color_upgrades.insert(Plot::P, 0);

        Self {
            plots: HashMap::new(),
            harvested_value: 0.0,
            harvested_yellow: 0.0,
            harvested_blue: 0.0,
            harvested_purple: 0.0,
            color_upgrades,
            seedlog: Vec::new(),
            reset: false,
        }
    }

    pub fn from_starting_pairs(pairs: &[Pair]) -> Self {
        let mut game = Self::new();
        for &pair in pairs {
            game.plots.entry(pair).or_insert_with(PlotState::new).count += 1;
        }
        game
    }

    pub fn get_available_choices(&self) -> Vec<Choice> {
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

    pub fn is_terminal(&self) -> bool {
        self.get_available_choices().is_empty()
    }

    pub fn calculate_harvest_lifeforce(&self, choice: Choice) -> f64 {
        let tables = get_tables();
        let upgrade_plot = tables.upgrade_from_choice[&choice];

        let current_upgrades = self.color_upgrades.get(&upgrade_plot).copied().unwrap_or(0);

        if (current_upgrades as usize) < tables.evs.len() {
            tables.evs[current_upgrades as usize]
        } else {
            *tables.evs.last().unwrap()
        }
    }

    pub fn choose(&mut self, choice: Choice, wither: bool) -> UndoInfo {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];
        let upgrade_plot = tables.upgrade_from_choice[&choice];
        let price_per_lifeforce = tables.lifeforce_price_per_div(upgrade_plot);
        let lifeforce = self.calculate_harvest_lifeforce(choice);

        let value_delta = lifeforce * price_per_lifeforce;
        self.harvested_value += value_delta;

        match upgrade_plot {
            Plot::Y => self.harvested_yellow += lifeforce,
            Plot::B => self.harvested_blue += lifeforce,
            Plot::P => self.harvested_purple += lifeforce,
        }

        // Remove one plot of this pair
        self.plots.get_mut(&pair).unwrap().count -= 1;
        let pair_was_deleted = self.plots[&pair].count == 0;
        if pair_was_deleted {
            self.plots.remove(&pair);
        }

        // Add leftover plot if it doesn't wither
        let leftover_info = self.handle_leftover_plot(choice, wither);
        let (leftover_pair, leftover_was_new) = match leftover_info {
            Some((pair, was_new)) => (Some(pair), was_new),
            None => (None, false),
        };

        // Upgrade all other colors
        let upgraded_colors = self.upgrade_other_colors(upgrade_plot);

        UndoInfo {
            value_delta,
            lifeforce_delta: lifeforce,
            pair,
            pair_was_deleted,
            leftover_pair,
            leftover_was_new,
            upgraded_colors,
            harvested_plot_type: upgrade_plot,
        }
    }

    fn handle_leftover_plot(&mut self, choice: Choice, wither: bool) -> Option<(Pair, bool)> {
        if wither {
            return None;
        }

        let tables = get_tables();
        if let Some(&Some(leftover_choice)) = tables.leftover_from_choice.get(&choice) {
            let leftover_pair = tables.pair_from_choice[&leftover_choice];
            let was_new = !self.plots.contains_key(&leftover_pair);
            let plot_state = self
                .plots
                .entry(leftover_pair)
                .or_insert_with(PlotState::new);
            plot_state.count += 1;
            Some((leftover_pair, was_new))
        } else {
            None
        }
    }

    fn upgrade_other_colors(&mut self, harvested_color: Plot) -> Vec<Plot> {
        let mut upgraded_colors = Vec::new();

        for &color in &[Plot::Y, Plot::B, Plot::P] {
            if color != harvested_color {
                *self.color_upgrades.get_mut(&color).unwrap() += 1;
                upgraded_colors.push(color);
            }
        }

        upgraded_colors
    }

    pub fn undo_choice(&mut self, undo_info: UndoInfo) {
        // Restore harvest value
        self.harvested_value -= undo_info.value_delta;

        // Restore harvested lifeforce counts
        match undo_info.harvested_plot_type {
            Plot::Y => self.harvested_yellow -= undo_info.lifeforce_delta,
            Plot::B => self.harvested_blue -= undo_info.lifeforce_delta,
            Plot::P => self.harvested_purple -= undo_info.lifeforce_delta,
        }

        // Undo upgrades
        for &color in &undo_info.upgraded_colors {
            *self.color_upgrades.get_mut(&color).unwrap() -= 1;
        }

        // Remove leftover plot if it was added
        if let Some(leftover_pair) = undo_info.leftover_pair {
            if let Some(plot_state) = self.plots.get_mut(&leftover_pair) {
                plot_state.count -= 1;
                if plot_state.count == 0 && undo_info.leftover_was_new {
                    self.plots.remove(&leftover_pair);
                }
            }
        }

        // Restore the original pair
        if undo_info.pair_was_deleted {
            self.plots.insert(undo_info.pair, PlotState { count: 1 });
        } else {
            self.plots.get_mut(&undo_info.pair).unwrap().count += 1;
        }
    }

    pub fn compute_optimal_value(&mut self) -> (f64, f64, f64, f64) {
        if self.is_terminal() {
            return (
                self.harvested_value,
                self.harvested_yellow,
                self.harvested_blue,
                self.harvested_purple,
            );
        }

        let mut best_ev = f64::NEG_INFINITY;
        let mut best_yellow = 0.0;
        let mut best_blue = 0.0;
        let mut best_purple = 0.0;
        let available_choices = self.get_available_choices();

        for choice in available_choices {
            let (ev, yellow, blue, purple) =
                self.calculate_choice_expected_value_with_colors(choice);
            if ev > best_ev {
                best_ev = ev;
                best_yellow = yellow;
                best_blue = blue;
                best_purple = purple;
            }
        }

        (best_ev, best_yellow, best_blue, best_purple)
    }

    pub fn calculate_choice_expected_value(&mut self, choice: Choice) -> f64 {
        let (ev, _, _, _) = self.calculate_choice_expected_value_with_colors(choice);
        ev
    }

    pub fn calculate_choice_expected_value_with_colors(
        &mut self,
        choice: Choice,
    ) -> (f64, f64, f64, f64) {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];

        if pair.is_single_plot() {
            // Single plot, no wither possibility
            let undo_info = self.choose(choice, false);
            let (value, yellow, blue, purple) = self.compute_optimal_value();
            self.undo_choice(undo_info);
            (value, yellow, blue, purple)
        } else {
            // Paired plot, calculate wither probability
            let wilt_chance = tables.wilt_rate;

            // Calculate EV for no wither case
            let undo_info_no_wither = self.choose(choice, false);
            let (no_wilt_value, no_wilt_yellow, no_wilt_blue, no_wilt_purple) =
                self.compute_optimal_value();
            self.undo_choice(undo_info_no_wither);

            // Calculate EV for wither case
            let undo_info_wither = self.choose(choice, true);
            let (wilt_value, wilt_yellow, wilt_blue, wilt_purple) = self.compute_optimal_value();
            self.undo_choice(undo_info_wither);

            // Expected values
            let ev = wilt_chance * wilt_value + (1.0 - wilt_chance) * no_wilt_value;
            let yellow = wilt_chance * wilt_yellow + (1.0 - wilt_chance) * no_wilt_yellow;
            let blue = wilt_chance * wilt_blue + (1.0 - wilt_chance) * no_wilt_blue;
            let purple = wilt_chance * wilt_purple + (1.0 - wilt_chance) * no_wilt_purple;

            (ev, yellow, blue, purple)
        }
    }

    pub fn find_optimal_strategy(&mut self) -> OptimalMove {
        if self.is_terminal() {
            return OptimalMove::terminal(
                self.harvested_value,
                self.harvested_yellow,
                self.harvested_blue,
                self.harvested_purple,
            );
        }

        let mut best_choice = None;
        let mut best_ev = f64::NEG_INFINITY;
        let mut best_no_wilt_ev = f64::NEG_INFINITY;
        let mut best_wilt_ev = None;
        let mut best_yellow = 0.0;
        let mut best_blue = 0.0;
        let mut best_purple = 0.0;

        let available_choices = self.get_available_choices();
        for choice in available_choices {
            let move_analysis = self.analyze_choice_with_colors(choice);

            // use gteq over gt because we want to favor picking the last plot
            // in the choices list which is optimal. The plot we just created
            // by harvesting will be at the end of the choices array, and we
            // want to favor it over other optimal moves.
            if move_analysis.expected_value >= best_ev {
                best_ev = move_analysis.expected_value;
                best_choice = Some(choice);
                best_no_wilt_ev = move_analysis.no_wilt_value;
                best_wilt_ev = move_analysis.wilt_value;
                best_yellow = move_analysis.expected_yellow;
                best_blue = move_analysis.expected_blue;
                best_purple = move_analysis.expected_purple;
            }
        }

        OptimalMove {
            choice: best_choice,
            ev_divs: best_ev,
            ev_yellow: best_yellow,
            ev_blue: best_blue,
            ev_purple: best_purple,
            no_wilt_value: best_no_wilt_ev,
            wilt_value: best_wilt_ev,
        }
    }

    pub fn analyze_choice(&mut self, choice: Choice) -> MoveAnalysis {
        let analysis = self.analyze_choice_with_colors(choice);
        MoveAnalysis {
            expected_value: analysis.expected_value,
            no_wilt_value: analysis.no_wilt_value,
            wilt_value: analysis.wilt_value,
        }
    }

    pub fn analyze_choice_with_colors(&mut self, choice: Choice) -> MoveAnalysisWithColors {
        let tables = get_tables();
        let pair = tables.pair_from_choice[&choice];

        if pair.is_single_plot() {
            // Single plot, no wither possibility
            let undo_info = self.choose(choice, false);
            let (value, yellow, blue, purple) = self.compute_optimal_value();
            self.undo_choice(undo_info);

            MoveAnalysisWithColors {
                expected_value: value,
                no_wilt_value: value,
                wilt_value: None,
                expected_yellow: yellow,
                expected_blue: blue,
                expected_purple: purple,
            }
        } else {
            let wilt_chance = tables.wilt_rate;

            // Calculate EV for no wither case
            let undo_info_no_wither = self.choose(choice, false);
            let (no_wilt_value, no_wilt_yellow, no_wilt_blue, no_wilt_purple) =
                self.compute_optimal_value();
            self.undo_choice(undo_info_no_wither);

            // Calculate EV for wither case
            let undo_info_wither = self.choose(choice, true);
            let (wilt_value, wilt_yellow, wilt_blue, wilt_purple) = self.compute_optimal_value();
            self.undo_choice(undo_info_wither);

            // Expected values
            let expected = wilt_chance * wilt_value + (1.0 - wilt_chance) * no_wilt_value;
            let expected_yellow = wilt_chance * wilt_yellow + (1.0 - wilt_chance) * no_wilt_yellow;
            let expected_blue = wilt_chance * wilt_blue + (1.0 - wilt_chance) * no_wilt_blue;
            let expected_purple = wilt_chance * wilt_purple + (1.0 - wilt_chance) * no_wilt_purple;

            MoveAnalysisWithColors {
                expected_value: expected,
                no_wilt_value,
                wilt_value: Some(wilt_value),
                expected_yellow,
                expected_blue,
                expected_purple,
            }
        }
    }

    pub fn get_total_value(&self) -> f64 {
        self.harvested_value
    }

    pub fn get_total_price(&self) -> f64 {
        self.harvested_value // harvested_value now stores the total price
    }

    pub fn get_user_wither_input(&mut self) -> bool {
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

    pub fn wait_for_user_input(&mut self) {
        println!("\nPress enter to continue.");
        let mut input = String::new();

        io::stdin().read_line(&mut input).unwrap();
        if input == "reset" {
            self.reset = true;
            println!("reset command received; starting a new game.");
        }
    }
}

pub struct UndoInfo {
    pub value_delta: f64,
    pub lifeforce_delta: f64,
    pub pair: Pair,
    pub pair_was_deleted: bool,
    pub leftover_pair: Option<Pair>,
    pub leftover_was_new: bool,
    pub upgraded_colors: Vec<Plot>,
    pub harvested_plot_type: Plot,
}

pub struct MoveAnalysis {
    pub expected_value: f64,     // Expected price value
    pub no_wilt_value: f64,      // Price value if no wilt
    pub wilt_value: Option<f64>, // Price value if wilt
}

pub struct MoveAnalysisWithColors {
    pub expected_value: f64,     // Expected price value
    pub no_wilt_value: f64,      // Price value if no wilt
    pub wilt_value: Option<f64>, // Price value if wilt
    pub expected_yellow: f64,
    pub expected_blue: f64,
    pub expected_purple: f64,
}

pub struct OptimalMove {
    pub choice: Option<Choice>,
    pub ev_divs: f64,            // Expected price value (not lifeforce)
    pub ev_yellow: f64,          // Expected yellow lifeforce
    pub ev_blue: f64,            // Expected blue lifeforce
    pub ev_purple: f64,          // Expected purple lifeforce
    pub no_wilt_value: f64,      // Price value if no wilt
    pub wilt_value: Option<f64>, // Price value if wilt
}

impl OptimalMove {
    pub fn terminal(value: f64, y: f64, b: f64, p: f64) -> Self {
        Self {
            choice: None,
            ev_divs: value,
            ev_yellow: y,
            ev_blue: b,
            ev_purple: p,
            no_wilt_value: value,
            wilt_value: None,
        }
    }

    pub fn get_expected_price(&self) -> f64 {
        let tables = get_tables();
        self.ev_yellow * tables.yellow_price_per_div
            + self.ev_blue * tables.blue_price_per_div
            + self.ev_purple * tables.purple_price_per_div
    }
}

// Calculate expected lifeforce harvested per upgrade tier (independent of color).
pub fn calculate_evs(config: &Config) -> Vec<f64> {
    const SEEDS_PER_PLOT: f64 = 23.0;
    let doubling = match config.doubling_scarab {
        Some(false) => 1.0,
        _ => 2.0,
    };
    let pack_size = 1. + config.map_pack_size / 100.;
    let map_iiq = config.map_iiq;
    let dupe = 1. + config.atlas_monster_dupe_chance / 100.;
    let additional = 1. + config.atlas_additional_monster_chance / 100.;
    let atlas_quant = config.atlas_lifeforce_quant;

    let drop_multi = doubling * (1. + map_iiq / 200. + atlas_quant / 100.);

    /*
    https://forgottenarbiter.github.io/Poe-Harvest-Mechanics/#lifeforce-drop-chance-from-monsters
     Base lifeforce drop amounts:
        T4 monster: ~190 to ~280 (average around 225 to 245) (low confidence)
        T3 monster: 37 to 57 (average 47)
        T2 monster: 12 to 25 (average 18.5)
        T1 monster: ~4.5 to 10 (average around 7 to 7.5)

     Lifeforce drop chance
        A T4 or T3 seed always drops one stack of lifeforce.
        A T2 seed has a 10% chance to drop one stack of lifeforce.
        A T1 seed has a 2% chance to drop one stack of lifeforce.
    */

    let tier_yield: [f64; 4] = [
        0.02 * 7. * drop_multi * SEEDS_PER_PLOT * pack_size * dupe * additional,
        0.1 * 18.5 * drop_multi * SEEDS_PER_PLOT * pack_size * dupe * additional,
        1.0 * 47. * drop_multi * SEEDS_PER_PLOT * pack_size * dupe * additional,
        1.0 * 235. * drop_multi * SEEDS_PER_PLOT * dupe, // T4s don't benefit from 10% additional monsters or pack size
    ];

    // Transition matrix P - calculated from config values
    let transition_matrix = [
        [1.0 - config.t2_transition, config.t2_transition, 0.00, 0.00],
        [0.00, 1.0 - config.t3_transition, config.t3_transition, 0.00],
        [0.00, 0.00, 1.0 - config.t4_transition, config.t4_transition],
        [0.00, 0.00, 0.00, 1.00],
    ];

    let mut state = [1.0, 0.0, 0.0, 0.0];
    let mut lifeforce_yields_per_tier = vec![];

    for _ in 0..10 {
        let lifeforce: f64 = state
            .iter()
            .zip(tier_yield.iter())
            .map(|(a, b)| a * b)
            .sum();
        lifeforce_yields_per_tier.push(lifeforce);

        // Matrix multiplication: state = state @ transition_matrix
        let mut new_state = [0.0; 4];
        for i in 0..4 {
            for j in 0..4 {
                new_state[j] += state[i] * transition_matrix[i][j];
            }
        }
        state = new_state;
    }
    for (i, lifeforce) in lifeforce_yields_per_tier.iter().enumerate() {
        println!("{i} upgrades yields {lifeforce} lifeforce");
    }

    lifeforce_yields_per_tier
}

pub fn parse_input_pairs(input: &str) -> Result<Vec<Pair>, String> {
    let pairs: Vec<&str> = input.trim().split_whitespace().collect();
    let mut parsed_pairs = Vec::new();

    for pair_str in pairs {
        if pair_str.is_empty() {
            continue;
        }

        match Pair::from_str(&pair_str.to_uppercase()) {
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
