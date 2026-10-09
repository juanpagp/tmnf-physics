//! The golden replay gate (the port of `tests/replay_tick.c`): replays the
//! captured A01 input schedules against the shipped reference records and
//! requires **byte-exact** 1,668-byte records, tick for tick.
//!
//! Fixtures live in the C repository (never redistributed): set
//! `TMNF_REPO` to override the default `../tmnf-physics`.

use std::path::PathBuf;
use tmnf_physics::gm::GmIso4;
use tmnf_physics::race::{tmnf_race_reset, tmnf_race_respawn_location,
    tmnf_race_step, TmnfRaceState};
use tmnf_physics::route::tmnf_route_load;
use tmnf_physics::track::tmnf_track_load;
use tmnf_physics::vehicle::TMNFRaceInputs;
use tmnf_physics::world::{World, WorldOptions};
use std::sync::Arc;

const TICK_MS: u32 = 10;
const RECORD_SIZE: usize = 1668;

const A01_SHA256: [u8; 32] = [
    0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
    0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
    0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
    0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
];

fn repo() -> PathBuf {
    std::env::var("TMNF_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../tmnf-physics"))
}

/// The record field table (replay_tick.c FIELDS) for mismatch reports.
const FIELDS: [(&str, usize); 16] = [
    ("race_time", 4),
    ("dyna.quat", 16),
    ("dyna.rotation", 36),
    ("dyna.position", 12),
    ("dyna.linear_speed", 12),
    ("dyna.add_linear_speed", 12),
    ("dyna.angular_speed", 12),
    ("dyna.force", 12),
    ("dyna.torque", 12),
    ("dyna.inverse_inertia_tensor", 36),
    ("dyna.not_tweaked_linear_speed", 12),
    ("scene_mobil.physics", 180),
    ("wheel[0].physics", 328),
    ("wheel[1].physics", 328),
    ("wheel[2].physics", 328),
    ("wheel[3].physics", 328),
];

fn report_mismatch(tick: u32, expected: &[u8], actual: &[u8]) {
    let mut byte = 0usize;
    while byte < RECORD_SIZE && expected[byte] == actual[byte] {
        byte += 1;
    }
    let mut field_start = 0usize;
    let mut field: Option<(&str, usize)> = None;
    for f in FIELDS {
        if byte < field_start + f.1 {
            field = Some(f);
            break;
        }
        field_start += f.1;
    }
    let (name, _) = field.expect("mismatch lies outside field table");
    let local = byte - field_start;
    let word = local & !3usize;
    let expected_bits = u32::from_le_bytes(
        expected[field_start + word..field_start + word + 4].try_into().unwrap());
    let actual_bits = u32::from_le_bytes(
        actual[field_start + word..field_start + word + 4].try_into().unwrap());
    panic!(
        "first divergence: tick {} (race time {} ms), field {}, byte {}, word {}\n  expected: 0x{:08x} ({})\n  actual:   0x{:08x} ({})",
        tick, (tick + 1) * TICK_MS, name, local, word,
        expected_bits, f32::from_bits(expected_bits),
        actual_bits, f32::from_bits(actual_bits),
    );
}

fn encode_scene(car: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(180);
    let ranges: [usize; 18] = [
        0x050, 0x2e0, 0x2e4, 0x59c, 0x5b0, 0x5c4, 0x5dc, 0x5e8, 0x5f4,
        0x5f8, 0x608, 0x628, 0x67c, 0x69c, 0x70c, 0x818, 0x844, 0x848,
    ];
    let sizes: [usize; 18] = [
        12, 4, 8, 4, 16, 8, 12, 4, 4, 12, 8, 4, 4, 4, 12, 12, 4, 48,
    ];
    for i in 0..18 {
        out.extend_from_slice(&car[ranges[i]..ranges[i] + sizes[i]]);
    }
    out
}

fn encode_wheel(wheel: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(328);
    let ranges: [usize; 18] = [
        0x004, 0x010, 0x040, 0x064, 0x070, 0x0a0, 0x0a8, 0x0b4, 0x0c0,
        0x0e4, 0x108, 0x120, 0x124, 0x130, 0x140, 0x144, 0x15c, 0x160,
    ];
    let sizes: [usize; 18] = [
        8, 48, 36, 12, 48, 8, 12, 12, 36, 36, 12, 4, 12, 12, 4, 12, 4, 12,
    ];
    for i in 0..18 {
        out.extend_from_slice(&wheel[ranges[i]..ranges[i] + sizes[i]]);
    }
    out
}

fn encode_tick(world: &World, tick: u32) -> [u8; RECORD_SIZE] {
    let (car, wheels) = world.write_player_game_state();
    let mut out = Vec::with_capacity(RECORD_SIZE);
    out.extend_from_slice(&(((tick + 1) * TICK_MS) as i32).to_le_bytes());
    let state = world.live();
    out.extend_from_slice(&state.to_bytes()[0x00..0xa0]);
    /* tail[1..4] as not_tweaked_linear_speed. */
    out.extend_from_slice(&state.to_bytes()[0xa4..0xb0]);
    out.extend_from_slice(&encode_scene(&car));
    for i in 0..4 {
        out.extend_from_slice(&encode_wheel(
            &wheels[i * 0x2fc..(i + 1) * 0x2fc]));
    }
    assert_eq!(out.len(), RECORD_SIZE, "record encoder produced the wrong size");
    let mut record = [0u8; RECORD_SIZE];
    record.copy_from_slice(&out);
    record
}

fn read_inputs(path: &std::path::Path) -> Vec<TMNFRaceInputs> {
    let bytes = std::fs::read(path).expect("input schedule");
    assert_eq!(bytes.len() % 0x48, 0, "input schedule is not a whole number of records");
    bytes
        .chunks(0x48)
        .map(|c| TMNFRaceInputs::from_bytes(c.try_into().unwrap()))
        .collect()
}

fn run_case(case: &str, results: &str, inputs: &str) {
    let repo = repo();
    let mask = std::fs::read(repo.join("local/game-mask.bin"))
        .expect("game mask");
    let mask = Arc::new(mask);
    let track = Arc::new(tmnf_track_load(
        repo.join("oracle/tracks/A01-Race.tmnftrack").to_str().unwrap(),
        Some(&A01_SHA256),
    ));
    let vehicle_blob = std::fs::read(
        repo.join("oracle/vehicles/A01-Stadium.tmnfvehicle"))
        .expect("vehicle snapshot");
    let route = Arc::new(tmnf_route_load(
        repo.join("oracle/routes/A01-Race.tmnfroute").to_str().unwrap(),
        Some(&A01_SHA256),
    ));

    let mut world = World::new(track, vehicle_blob, WorldOptions {
        fake_contact_mask: Some(mask),
        pin_provenance: true,
    });
    world.route = Some(route.clone());

    let mut race = TmnfRaceState::default();
    tmnf_race_reset(&route, &mut race, &world.live().rot_pos_iso());

    let reference = std::fs::read(repo.join(results)).expect("reference");
    assert_eq!(reference.len() % RECORD_SIZE, 0);
    let tick_count = reference.len() / RECORD_SIZE;
    let schedule = read_inputs(&repo.join(inputs));
    assert!(schedule.len() >= tick_count, "input schedule is shorter than the reference");

    let mut matched = 0usize;
    for tick in 0..tick_count {
        let input = schedule[tick];
        /* tracker_respawn: the press applies before the tick's inputs. */
        if input.respawn != 0 {
            match tmnf_race_respawn_location(&race) {
                Some(spawn) => world.respawn(spawn),
                None => panic!("respawn before any checkpoint restarts the race"),
            }
        }
        world.apply_inputs(&input);
        let actual = encode_tick(&world, tick as u32);
        let expected = &reference[tick * RECORD_SIZE..(tick + 1) * RECORD_SIZE];
        if actual != expected && tick == 1 && std::env::var("TMNF_DBG_DIFF").is_ok() {
            let mut o = 0usize;
            for f in FIELDS {
                let seg = &expected[o..o + f.1];
                let sega = &actual[o..o + f.1];
                if seg != sega {
                    eprintln!("field {} ({} B) differs:", f.0, f.1);
                    for k in (0..f.1).step_by(4) {
                        if seg[k..k + 4] != sega[k..k + 4] {
                            let e = u32::from_le_bytes(seg[k..k + 4].try_into().unwrap());
                            let a = u32::from_le_bytes(sega[k..k + 4].try_into().unwrap());
                            eprintln!("  +{}: {:08x} ({}) != {:08x} ({})", k, e,
                                f32::from_bits(e), a, f32::from_bits(a));
                        }
                    }
                }
                o += f.1;
            }
        }
        if actual != expected {
            report_mismatch(tick as u32, expected, &actual);
        }
        matched += 1;
        world.advance_timer(TICK_MS);
        world.physics_step2(TICK_MS);
        let car_iso: GmIso4 = world.live().rot_pos_iso();
        tmnf_race_step(&route, &mut race, world.trigger_contacts, &car_iso);
    }
    println!("{}: {}/{} ticks byte-exact", case, matched, tick_count);
}

#[test]
fn golden_a01_mixed() {
    run_case(
        "a01_mixed",
        "oracle/results/a01_mixed.bin",
        "oracle/results/a01_mixed_inputs.bin",
    );
}

#[test]
fn golden_a01_wall_contact() {
    run_case(
        "a01_wall_contact",
        "oracle/results/a01_wall_contact.bin",
        "oracle/results/a01_wall_contact_inputs.bin",
    );
}
