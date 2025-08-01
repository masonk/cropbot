use croptimizer::{GameState, Pair};

fn main() {
    // Test with 3 YY pairs
    let pairs = vec![Pair::YY, Pair::YY, Pair::YY];
    let mut game = GameState::from_starting_pairs(&pairs);

    println!("Initial state:");
    println!("  Plots: {:?}", game.plots);
    println!("  Color upgrades: {:?}", game.color_upgrades);
    println!("  Available choices: {:?}", game.get_available_choices());

    let optimal = game.find_optimal_strategy();
    println!("\nOptimal strategy:");
    println!("  Choice: {:?}", optimal.choice);
    println!("  EV divs: {}", optimal.ev_divs);
    println!("  EV yellow: {}", optimal.ev_yellow);
    println!("  EV blue: {}", optimal.ev_blue);
    println!("  EV purple: {}", optimal.ev_purple);

    // Let's manually trace through one harvest
    if let Some(choice) = optimal.choice {
        println!("\nManually tracing first harvest:");
        let lifeforce = game.calculate_harvest_lifeforce(choice);
        println!("  Lifeforce from harvest: {}", lifeforce);

        let undo_info = game.choose(choice, false); // assume no wither
        println!("  After harvest:");
        println!("    Plots: {:?}", game.plots);
        println!("    Color upgrades: {:?}", game.color_upgrades);
        println!("    Harvested yellow: {}", game.harvested_yellow);
        println!("    Available choices: {:?}", game.get_available_choices());

        game.undo_choice(undo_info);
    }
}
