//! The data-lane round-trip gates: the vehicle image as data.
//!
//! The two headline guarantees (mirroring the C's `vehicle_image_test`):
//! 1. `write(read(image))` is **byte-identical** to the image.
//! 2. `image_with_structure(image, structure_from_image(image))` is
//!    **byte-identical** — i.e. the structure authoring path is lossless —
//!    and the same for tuning.
//!
//! Plus: an authored change (heavier car) provably changes the simulation,
//! so the equality above is not vacuous.

use std::sync::Arc;
use tmnf_physics::data::{
    image_with_structure, image_with_tuning, structure_from_image,
    tuning_from_image,
};
use tmnf_physics::track::tmnf_track_load;
use tmnf_physics::world::{VehicleImage, World, WorldOptions};

fn repo() -> String {
    std::env::var("TMNF_REPO")
        .unwrap_or_else(|_| "../tmnf-physics".to_string())
}

fn a01_sha() -> [u8; 32] {
    [
        0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
        0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
        0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
        0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
    ]
}

fn stadium_blob() -> Vec<u8> {
    std::fs::read(format!("{}/oracle/vehicles/A01-Stadium.tmnfvehicle", repo()))
        .expect("vehicle snapshot")
}

fn world_for(blob: &[u8]) -> World {
    let mask = Arc::new(
        std::fs::read(format!("{}/local/game-mask.bin", repo())).unwrap());
    let track = Arc::new(tmnf_track_load(
        &format!("{}/oracle/tracks/A01-Race.tmnftrack", repo()),
        Some(&a01_sha()),
    ));
    World::new(track, blob.to_vec(), WorldOptions {
        fake_contact_mask: Some(mask),
        pin_provenance: false,
    })
}

fn first_pos(world: &World) -> (f32, f32, f32) {
    let p = world.live().pos;
    (p.x, p.y, p.z)
}

#[test]
fn image_write_read_is_byte_identical() {
    let blob = stadium_blob();
    let image = VehicleImage::read(&blob);
    let rebuilt = image.write();
    assert_eq!(rebuilt.len(), blob.len());
    assert_eq!(rebuilt, blob, "write(read(b)) != b");
}

#[test]
fn structure_round_trip_is_byte_identical() {
    let blob = stadium_blob();
    let image = VehicleImage::read(&blob);
    let structure = structure_from_image(&image);
    let rebuilt = image_with_structure(image, &structure).write();
    assert_eq!(rebuilt, blob, "image_with_structure is not lossless");
}

#[test]
fn tuning_round_trip_is_byte_identical() {
    let blob = stadium_blob();
    let image = VehicleImage::read(&blob);
    let tuning = tuning_from_image(&image);
    let rebuilt = image_with_tuning(image, &tuning).write();
    assert_eq!(rebuilt, blob, "image_with_tuning is not lossless");
}

#[test]
fn tuning_json_round_trip() {
    let blob = stadium_blob();
    let image = VehicleImage::read(&blob);
    let tuning = tuning_from_image(&image);
    let json = serde_json::to_string_pretty(&tuning).unwrap();
    let back: tmnf_physics::data::CarTuning =
        serde_json::from_str(&json).unwrap();
    assert_eq!(back, tuning);
    assert!(json.len() > 5000, "tuning JSON suspiciously small");
}

#[test]
fn structure_json_round_trip() {
    let blob = stadium_blob();
    let image = VehicleImage::read(&blob);
    let structure = structure_from_image(&image);
    let json = serde_json::to_string(&structure).unwrap();
    let back: tmnf_physics::data::CarStructure =
        serde_json::from_str(&json).unwrap();
    assert_eq!(back, structure);
}

/// Drive a few ticks with full throttle and return the position +
/// velocity bits (a compact physics fingerprint).
fn drive_fingerprint(world: &mut World, ticks: usize) -> (u64, u64) {
    use tmnf_physics::vehicle::TMNFRaceInputs;
    let input = TMNFRaceInputs {
        accelerate: 1,
        ..Default::default()
    };
    for _ in 0..ticks {
        world.apply_inputs(&input);
        world.advance_timer(10);
        world.physics_step2(10);
    }
    let p = world.live().pos;
    let v = world.live().lin_vel;
    let pos_bits = (p.x.to_bits() as u64) ^ ((p.y.to_bits() as u64) << 32);
    let vel_bits = (v.x.to_bits() as u64) ^ ((v.y.to_bits() as u64) << 32);
    (pos_bits, vel_bits)
}

#[test]
fn authored_body_geometry_changes_physics() {
    /* Mass cancels out of gravity (F = m*g), so the honest probes are the
     * quantities the solver reads directly: the collision ellipsoids and
     * the inertia tensor. */
    let base = stadium_blob();
    let mut baseline_world = world_for(&base);
    let baseline = drive_fingerprint(&mut baseline_world, 30);

    let image = VehicleImage::read(&base);
    let mut structure = structure_from_image(&image);
    for node in &mut structure.collision_nodes {
        if node.kind & 0xff == 1 {
            /* Wheel leaves: fatter ellipsoids change the contact geometry. */
            node.radii.x *= 1.5;
            node.radii.y *= 1.5;
            node.radii.z *= 1.5;
        }
    }
    let fatter = image_with_structure(image, &structure).write();
    let mut fatter_world = world_for(&fatter);
    let fatter = drive_fingerprint(&mut fatter_world, 30);
    assert_ne!(baseline, fatter, "fatter wheel ellipsoids must change physics");
}

#[test]
fn authored_inertia_changes_physics() {
    let base = stadium_blob();
    let mut baseline_world = world_for(&base);
    let baseline = drive_fingerprint(&mut baseline_world, 30);

    let image = VehicleImage::read(&base);
    let mut structure = structure_from_image(&image);
    /* Half the inverse inertia = twice the rotational inertia. */
    for m in &mut structure.body.inverse_inertia.m {
        *m *= 0.5;
    }
    let stiffer = image_with_structure(image, &structure).write();
    let mut stiffer_world = world_for(&stiffer);
    let stiffer = drive_fingerprint(&mut stiffer_world, 30);
    assert_ne!(baseline, stiffer, "a different inertia must change physics");
}

#[test]
fn authored_suspension_changes_physics() {
    let base = stadium_blob();
    let mut baseline_world = world_for(&base);
    let baseline = drive_fingerprint(&mut baseline_world, 30);

    let image = VehicleImage::read(&base);
    let tuning = tuning_from_image(&image);
    let mut softer = tuning.clone();
    softer.base.suspension_stiffness *= 0.5;
    softer.base.suspension_damping *= 0.5;
    let softer_blob = image_with_tuning(image, &softer).write();
    let mut softer_world = world_for(&softer_blob);
    let softer = drive_fingerprint(&mut softer_world, 30);
    assert_ne!(baseline, softer, "softer suspension must change physics");
}
