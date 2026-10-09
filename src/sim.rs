//! The simulation facade — the public API the whole crate exists for:
//!
//! ```no_run
//! use tmnf_physics::sim::{Sim, SimOptions, TmnfInput};
//!
//! # fn main() {
//! let mut sim = Sim::load(
//!     "A01-Race.tmnftrack",
//!     "A01-Stadium.tmnfvehicle",
//!     &SimOptions {
//!         fake_contact_mask: Some("game-mask.bin".into()),
//!         ..Default::default()
//!     },
//! );
//! let mut playing = true;
//! while playing {
//!     let input = TmnfInput::digital(1, 0, 0, 0); // accelerate only
//!     let state = sim.step(&input); // one 10 ms tick
//!     playing = state.race_time_ms < 30_000;
//!     // render_car(&state) ...
//! }
//! # }
//! ```
//!
//! One `Sim` = one car on one track. Fixed 10 ms timestep by construction;
//! drive it from an accumulator in your game loop. Snapshots are `Copy`
//! bytes: rewind and branch-a-tree-of-futures are a memcpy.

use crate::gm::GmIso4;
use crate::race::{tmnf_race_reset, tmnf_race_respawn_location, tmnf_race_step,
    TmnfRaceState};
use crate::route::{tmnf_route_load, TmnfRoute};
use crate::track::tmnf_track_load;
use crate::vehicle::TMNFRaceInputs;
use crate::world::{VehicleImage, World, WorldOptions};
use std::path::Path;
use std::sync::Arc;

/// Simulation options: where the runtime data comes from.
#[derive(Clone, Default)]
pub struct SimOptions {
    /// Path to the 128x128 fake-contact mask (e.g. `local/game-mask.bin`).
    /// Required — in Rust the mask is always runtime data.
    pub fake_contact_mask: Option<String>,
    /// Path to a `TMNFROU1` route (checkpoints/finish). Without one the
    /// simulation is a sandbox (no race bookkeeping).
    pub route: Option<String>,
    /// Require the vehicle image to pin the canonical exe + track hashes
    /// (true for captured snapshots; false for authored images).
    pub pin_provenance: bool,
}

/// The player input for one tick. Digital buttons and an optional analog
/// axis (the game's 0x48 input packet).
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfInput {
    pub accelerate: bool,
    pub brake: bool,
    pub steer_left: bool,
    pub steer_right: bool,
    /// Analog gas [-1, 1] (0 = unused).
    pub gas_analog: f32,
    /// Analog steer [-1, 1] (0 = unused).
    pub steer_analog: f32,
    /// Respawn press (needs a route).
    pub respawn: bool,
}

impl TmnfInput {
    /// Digital input: (accelerate, brake, steer_left, steer_right).
    pub fn digital(accelerate: u8, brake: u8, steer_left: u8, steer_right: u8)
        -> TmnfInput {
        TmnfInput {
            accelerate: accelerate != 0,
            brake: brake != 0,
            steer_left: steer_left != 0,
            steer_right: steer_right != 0,
            ..Default::default()
        }
    }

    /// Coast (all controls released).
    pub fn idle() -> TmnfInput {
        TmnfInput::default()
    }

    fn to_packet(&self, tick: u32) -> TMNFRaceInputs {
        /* The static 0x004FE500 mapper reads the timestamp/value words; a
         * fresh packet with the current tick as every timestamp reproduces
         * the game's mapping for live input. */
        let t = tick.wrapping_mul(10);
        TMNFRaceInputs {
            steer_left_time: if self.steer_left { t } else { 0 },
            steer_left: self.steer_left as i32,
            steer_right_time: if self.steer_right { t } else { 0 },
            steer_right: self.steer_right as i32,
            steer_analog_time: if self.steer_analog != 0.0 { t } else { 0 },
            steer_analog: self.steer_analog,
            accelerate_time: if self.accelerate { t } else { 0 },
            accelerate: self.accelerate as i32,
            brake_time: if self.brake { t } else { 0 },
            brake: self.brake as i32,
            gas_analog_time: if self.gas_analog != 0.0 { t } else { 0 },
            respawn: self.respawn as u32,
            gas_analog: self.gas_analog,
            ..Default::default()
        }
    }
}

/// The per-tick observation: everything a renderer or game loop needs.
#[derive(Clone, Copy, Debug)]
pub struct VehicleState {
    /// Race time in ms since the start (or respawn).
    pub race_time_ms: u32,
    /// World-space position of the rigid body.
    pub position: crate::gm::GmVec3,
    /// Orientation quaternion in (x, y, z, w) field order — the game's
    /// scalar-first (w, x, y, z) byte order mapped to the conventional
    /// accessor order.
    pub quaternion: [f32; 4],
    /// World-space linear velocity (m/s).
    pub linear_speed: crate::gm::GmVec3,
    /// World-space angular velocity (rad/s).
    pub angular_speed: crate::gm::GmVec3,
    /// Per-wheel state: on-ground flag, material id, sliding, spin speed,
    /// contact point (world space).
    pub wheels: [WheelState; 4],
    /// Engine RPM.
    pub engine_rpm: f32,
    /// Gear (-1 reverse, 1..6 forward).
    pub gear: i32,
    /// The respawn the next respawn press applies, if any.
    pub can_respawn: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct WheelState {
    pub on_ground: bool,
    pub contact_material_id: i32,
    pub sliding: bool,
    pub spin_speed: f32,
}

/// One simulation: one car on one track. `step()` is one 10 ms tick.
pub struct Sim {
    world: World,
    route: Option<Arc<TmnfRoute>>,
    race: TmnfRaceState,
    tick: u32,
}

impl Sim {
    /// Builds a simulation from loaded pieces.
    pub fn new(
        track: Arc<crate::track::TmnfTrack>,
        vehicle_blob: Vec<u8>,
        fake_contact_mask: Vec<u8>,
        options: &SimOptions,
    ) -> Sim {
        let route = options.route.as_ref().map(|p| {
            Arc::new(tmnf_route_load(p, None))
        });
        let world = World::new(track, vehicle_blob, WorldOptions {
            fake_contact_mask: Some(Arc::new(fake_contact_mask)),
            pin_provenance: options.pin_provenance,
        });
        let mut sim = Sim {
            world,
            route: route.clone(),
            race: TmnfRaceState::default(),
            tick: 0,
        };
        if let Some(route) = &sim.route {
            let iso = sim.world.live().rot_pos_iso();
            tmnf_race_reset(route, &mut sim.race, &iso);
        }
        sim
    }

    /// Loads track + vehicle files (the reference lane).
    pub fn load<P: AsRef<Path>>(
        track_path: P,
        vehicle_path: P,
        options: &SimOptions,
    ) -> Sim {
        let track = Arc::new(tmnf_track_load(
            track_path.as_ref().to_str().unwrap(), None));
        let vehicle_blob = std::fs::read(vehicle_path)
            .expect("cannot read vehicle snapshot");
        let mask = std::fs::read(
            options.fake_contact_mask.as_deref().unwrap_or(
                "local/game-mask.bin"))
            .expect("fake-contact mask (see SimOptions::fake_contact_mask)");
        Sim::new(track, vehicle_blob, mask, options)
    }

    /// Builds a simulation from an authored vehicle image (the data lane).
    pub fn from_image(
        image: &VehicleImage,
        track: Arc<crate::track::TmnfTrack>,
        fake_contact_mask: Vec<u8>,
        options: &SimOptions,
    ) -> Sim {
        Sim::new(track, image.write(), fake_contact_mask, options)
    }

    /// One 10 ms tick: applies the input, steps the physics, advances the
    /// race, and returns the post-tick observation.
    pub fn step(&mut self, input: &TmnfInput) -> VehicleState {
        let packet = input.to_packet(self.tick);

        /* The respawn press applies before the tick's control mapping. */
        if packet.respawn != 0 {
            if let Some(spawn) = tmnf_race_respawn_location(&self.race) {
                let spawn: GmIso4 = *spawn;
                self.world.respawn(&spawn);
            }
        }
        self.world.route = self.route.clone();
        self.world.apply_inputs(&packet);
        self.tick = self.tick.wrapping_add(1);
        self.world.advance_timer(10);
        self.world.physics_step2(10);
        if let Some(route) = &self.route {
            if self.race.finished == 0 {
                let iso = self.world.live().rot_pos_iso();
                let contacts = self.world.trigger_contacts;
                tmnf_race_step(route, &mut self.race, contacts, &iso);
            }
        }
        self.state()
    }

    /// The observation of the current state (without stepping).
    pub fn state(&self) -> VehicleState {
        let s = self.world.live();
        /* The dyna quaternion bytes are (w, x, y, z); expose (x, y, z, w). */
        let q = [s.quat.y, s.quat.z, s.quat.w, s.quat.x];
        let mut wheels = [WheelState {
            on_ground: false,
            contact_material_id: 0,
            sliding: false,
            spin_speed: 0.0,
        }; 4];
        for (i, w) in wheels.iter_mut().enumerate() {
            let rt = &self.world.vehicle.wheels[i].real_time;
            w.on_ground = rt.has_ground_contact != 0;
            w.contact_material_id = rt.contact_material_id;
            w.sliding = rt.is_sliding != 0;
            w.spin_speed = rt.field6c;
        }
        VehicleState {
            race_time_ms: self.race.elapsed_ticks.wrapping_mul(10),
            position: s.pos,
            quaternion: q,
            linear_speed: s.lin_vel,
            angular_speed: s.ang_vel,
            wheels,
            engine_rpm: self.world.vehicle.engine.rpm,
            gear: self.world.vehicle.engine.gear,
            can_respawn: tmnf_race_respawn_location(&self.race).is_some(),
        }
    }

    /// Access to the underlying world (advanced use).
    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// The race state (checkpoints, laps, finish time), if a route is bound.
    pub fn race(&self) -> Option<&TmnfRaceState> {
        self.route.as_ref().map(|_| &self.race)
    }
}
