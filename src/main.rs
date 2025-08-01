use croptimizer::{get_tables, Config, GameState, OptimalMove, Pair};
use std::io::{self, Write};

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
                    optimal_move.ev_divs, wilt, optimal_move.no_wilt_value
                );
            } else {
                println!("EV: {:.1}", optimal_move.ev_divs);
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

    match croptimizer::parse_input_pairs(&input) {
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
