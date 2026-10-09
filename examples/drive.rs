//! The working example: the `while playing { sim.step(); render(); }` loop.
//!
//! Runs the A01-Stadium car on the A01-Race track with a scripted input
//! schedule (launch, brake, steer), printing telemetry and writing a CSV
//! trajectory. Uses only public API — no game installation anywhere.
//!
//! ```text
//! cargo run --release --example drive
//! ```

use std::io::Write;
use tmnf_physics::sim::{Sim, SimOptions, TmnfInput};

fn repo() -> String {
    std::env::var("TMNF_REPO")
        .unwrap_or_else(|_| "../tmnf-physics".to_string())
}

fn main() {
    let repo = repo();
    let track = format!("{}/oracle/tracks/A01-Race.tmnftrack", repo);
    let vehicle = format!("{}/oracle/vehicles/A01-Stadium.tmnfvehicle", repo);
    let mask = format!("{}/local/game-mask.bin", repo);
    let route = format!("{}/oracle/routes/A01-Race.tmnfroute", repo);

    println!("TMNF physics — drive example");
    println!("  track:   {}", track);
    println!("  vehicle: {}", vehicle);

    let mut sim = Sim::load(
        &track,
        &vehicle,
        &SimOptions {
            fake_contact_mask: Some(mask),
            route: Some(route),
            pin_provenance: true,
        },
    );

    /* Scripted schedule (tick, input): launch hard, brake, wiggle, coast. */
    let input_at = |tick: u32| -> TmnfInput {
        match tick {
            t if t < 300 => TmnfInput::digital(1, 0, 0, 0), // full gas 3 s
            t if t < 350 => TmnfInput::digital(0, 1, 0, 0), // brake 0.5 s
            t if t < 500 => TmnfInput::digital(1, 0, 1, 0), // gas + left
            t if t < 600 => TmnfInput::digital(1, 0, 0, 1), // gas + right
            t if t < 700 => TmnfInput::digital(0, 0, 0, 0), // coast
            _ => TmnfInput::digital(1, 0, 0, 0),            // gas again
        }
    };

    let total_ticks = 1000; // 10 seconds
    let mut csv = String::from("t_ms,pos_x,pos_y,pos_z,speed_kmh,rpm,gear,on_ground\n");
    let mut max_speed = 0.0f32;

    let mut playing = true;
    let mut tick = 0u32;
    while playing {
        let input = input_at(tick);
        let state = sim.step(&input);

        let speed = (state.linear_speed.x * state.linear_speed.x
            + state.linear_speed.y * state.linear_speed.y
            + state.linear_speed.z * state.linear_speed.z)
            .sqrt();
        let kmh = speed * 3.6;
        if kmh > max_speed {
            max_speed = kmh;
        }
        let grounded = state.wheels.iter().filter(|w| w.on_ground).count();

        if tick % 100 == 0 {
            println!(
                "t={:5.1}s  pos=({:8.2},{:7.2},{:8.2})  v={:6.1} km/h  rpm={:7.0}  gear={}  wheels_down={}/{}",
                state.race_time_ms as f32 / 1000.0,
                state.position.x, state.position.y, state.position.z,
                kmh, state.engine_rpm, state.gear, grounded, 4
            );
        }
        csv.push_str(&format!(
            "{},{:.3},{:.3},{:.3},{:.3},{:.0},{},{}\n",
            state.race_time_ms, state.position.x, state.position.y,
            state.position.z, kmh, state.engine_rpm, state.gear, grounded
        ));

        tick += 1;
        playing = tick < total_ticks;
    }

    if let Some(race) = sim.race() {
        println!(
            "race: checkpoints taken {}, lap {}",
            race.visited_count, race.completed_laps
        );
    }
    println!("max speed: {:.1} km/h", max_speed);

    let out_path = "drive_telemetry.csv";
    let mut f = std::fs::File::create(out_path).expect("csv output");
    f.write_all(csv.as_bytes()).unwrap();
    println!("telemetry written to {}", out_path);

    /* The determinism proof: two FRESH sims, the same schedule, bit for bit. */
    let make = || Sim::load(
        &track,
        &vehicle,
        &SimOptions {
            fake_contact_mask: Some(format!("{}/local/game-mask.bin", repo)),
            route: Some(format!("{}/oracle/routes/A01-Race.tmnfroute", repo)),
            pin_provenance: true,
        },
    );
    let (mut sim_a, mut sim_b) = (make(), make());
    let mut final_a = None;
    let mut final_b = None;
    for t in 0..total_ticks {
        let a = sim_a.step(&input_at(t));
        let b = sim_b.step(&input_at(t));
        final_a = Some(a);
        final_b = Some(b);
    }
    let (a, b) = (final_a.unwrap(), final_b.unwrap());
    let same = a.position.x.to_bits() == b.position.x.to_bits()
        && a.position.y.to_bits() == b.position.y.to_bits()
        && a.position.z.to_bits() == b.position.z.to_bits()
        && a.linear_speed.x.to_bits() == b.linear_speed.x.to_bits()
        && a.linear_speed.y.to_bits() == b.linear_speed.y.to_bits()
        && a.linear_speed.z.to_bits() == b.linear_speed.z.to_bits();
    println!("determinism (two runs, same inputs): {}", if same { "BIT-IDENTICAL" } else { "DIVERGED" });
    assert!(same, "the simulation must be deterministic");
}
