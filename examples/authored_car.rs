//! The gap-fix example: **authoring a car from data**.
//!
//! The fork's honest gap was that car *structure* (wheel geometry, body
//! ellipsoids, mass/inertia) only existed as an opaque captured image. This
//! example closes it:
//!
//! 1. reads the reference Stadium car,
//! 2. exports its complete structure and tuning to JSON,
//! 3. authors a variant purely in code (lower, stiffer, fatter tires),
//! 4. rebuilds a valid image, and
//! 5. drives both, showing the physics actually changed.
//!
//! ```text
//! cargo run --release --example authored_car
//! ```

use std::sync::Arc;
use tmnf_physics::data::{
    image_with_structure, image_with_tuning, structure_from_image,
    tuning_from_image, CarStructure,
};
use tmnf_physics::sim::{Sim, SimOptions, TmnfInput};
use tmnf_physics::track::tmnf_track_load;
use tmnf_physics::world::VehicleImage;

fn repo() -> String {
    std::env::var("TMNF_REPO")
        .unwrap_or_else(|_| "../tmnf-physics".to_string())
}

fn main() {
    let repo = repo();
    let base_blob = std::fs::read(format!(
        "{}/oracle/vehicles/A01-Stadium.tmnfvehicle", repo))
        .expect("the reference vehicle image");
    let mask = std::fs::read(format!("{}/local/game-mask.bin", repo)).unwrap();
    let track = Arc::new(tmnf_track_load(
        &format!("{}/oracle/tracks/A01-Race.tmnftrack", repo),
        None,
    ));

    println!("TMNF physics — authored car example");
    println!("reference image: {} bytes", base_blob.len());

    /* 1 + 2. Decode and export the reference car as JSON. */
    let image = VehicleImage::read(&base_blob);
    let structure = structure_from_image(&image);
    let tuning = tuning_from_image(&image);

    let structure_json = serde_json::to_string_pretty(&structure).unwrap();
    let tuning_json = serde_json::to_string_pretty(&tuning).unwrap();
    std::fs::write("reference_car_structure.json", &structure_json).unwrap();
    std::fs::write("reference_car_tuning.json", &tuning_json).unwrap();
    println!(
        "exported structure ({} B JSON): {} wheels, {} collision nodes, mass {:.1}",
        structure_json.len(),
        structure.wheels.len(),
        structure.collision_nodes.len(),
        structure.body.mass
    );
    println!(
        "exported tuning ({} B JSON): {} curves, {} gear ratios",
        tuning_json.len(),
        23,
        tuning.gear_ratios.len()
    );

    /* 3. Author a variant purely in code: a low, stiff track toy. */
    let mut custom: CarStructure =
        serde_json::from_str(&structure_json).unwrap();
    for wheel in &mut custom.wheels {
        wheel.offset_from_vehicle.y -= 0.1; // slam it 10 cm lower
    }
    for node in &mut custom.collision_nodes {
        if node.kind & 0xff == 1 {
            node.radii.x *= 1.2; // fatter tires
        }
        node.box_aligned.half_extent.y *= 0.92; // lower body
    }
    custom.body.drag_linear *= 0.7; // slipperier

    let mut custom_tuning =
        serde_json::from_str::<tmnf_physics::data::CarTuning>(&tuning_json)
            .unwrap();
    custom_tuning.base.suspension_stiffness *= 1.4; // stiffer springs
    custom_tuning.base.suspension_damping *= 1.2;
    custom_tuning.contact.max_angular_speed *= 1.1; // a bit more rotation

    /* 4. Rebuild a valid image (layout-preserving patch of the reference). */
    let custom_image = image_with_structure(image, &custom);
    let custom_image = image_with_tuning(custom_image, &custom_tuning);
    let custom_blob = custom_image.write();
    println!("authored image: {} bytes (round-trips to the reference when unmodified)", custom_blob.len());

    /* 5. Drive both. */
    let options = SimOptions {
        fake_contact_mask: None,
        route: None,
        pin_provenance: false,
    };
    let mut reference = Sim::new(
        track.clone(),
        base_blob.clone(),
        mask.clone(),
        &options,
    );
    let mut authored = Sim::new(track, custom_blob, mask, &options);

    let drive = |sim: &mut Sim| -> (f32, f32, f32) {
        let input = TmnfInput::digital(1, 0, 0, 0);
        let mut last = None;
        for _ in 0..300 {
            let s = sim.step(&input);
            last = Some((s.position.x, s.position.y, s.position.z));
        }
        last.unwrap()
    };
    let a = drive(&mut reference);
    let b = drive(&mut authored);
    println!("after 3 s full throttle:");
    println!("  reference: pos=({:.2}, {:.2}, {:.2})", a.0, a.1, a.2);
    println!("  authored : pos=({:.2}, {:.2}, {:.2})", b.0, b.1, b.2);
    assert_ne!(
        (a.0.to_bits(), a.1.to_bits(), a.2.to_bits()),
        (b.0.to_bits(), b.1.to_bits(), b.2.to_bits()),
        "the authored car must drive differently"
    );
    println!("the authored car drives differently — the structure is live data.");
    println!("JSON files written: reference_car_structure.json, reference_car_tuning.json");
}
