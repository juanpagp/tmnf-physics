//! The vehicle layer: `CSceneVehicleCar` and its force pipeline
//! (models 3–6), transliterated from `src/vehicle*.c/h`.
//!
//! # Architecture of this port
//!
//! The C code threads four big context structs (`CSceneVehicleCarAuxContext`,
//! `TMNFVehicleContactContext`, `TMNFVehicleComputeContext`,
//! `CSceneVehicleCarModel6Context`) that bundle pointers into one car object,
//! its tuning and per-subsystem state. In Rust:
//!
//! * [`Vehicle`] — ALL mutable per-simulation state (the car fields at their
//!   documented game offsets, the aux/contact/compute/model6 state blocks,
//!   and the four wheels, each folded together from `CSceneVehicleCarWheel`,
//!   `CSceneVehicleCarWheelAux` and `TMNFVehicleContactWheelState`).
//! * [`VehicleTuning`] — ALL immutable image-derived data (the five tuning
//!   views with their documented game offsets, the curve set, ground
//!   materials and the fake-contact mask). Shared by any number of sims.
//! * [`VehicleCtx`] — the bundle passed where the C passed a context struct:
//!   `{ &mut Vehicle, &VehicleTuning, &mut CHmsDyna }` (the C reached the
//!   dyna through `vehicle->dyna_state`/`dyna_params` pointers).
//!
//! Submodules (one per C file):
//! * [`curve`] — `vehicle_curve.c` (CFuncKeys interpolation).
//! * [`base`] — `vehicle.c` (input mapping 0x004FE500, engine integrate,
//!   wheel speed, add-force helpers).
//! * [`aux`] — `vehicle_aux.c` (turbo/roulette, air control, water forces,
//!   vehicle integration, fake contacts).
//! * [`contact`] — `vehicle_contact.c` (wheel contact absorption, friction,
//!   ground material blending).
//! * [`compute`] — `vehicle_compute.c` (the 0x007C69E0 force orchestrator).
//! * [`model6`] — `vehicle_model6.c` (the Stadium car model).
//!
//! # C field → Rust path map (for porters)
//!
//! | C access | Rust |
//! |---|---|
//! | `ctx->vehicle->input_gas` | `ctx.vehicle.input_gas` |
//! | `ctx->vehicle->engine` | `ctx.vehicle.engine` |
//! | `aux->turbo_epoch_tick` | `ctx.vehicle.turbo_epoch_tick` |
//! | `aux->air_control_immediate` | `ctx.vehicle.air_control_immediate` |
//! | `aux->steering_value` | `ctx.vehicle.steering_value` |
//! | `aux->roulette_modulus` | `ctx.vehicle.roulette_modulus` |
//! | `aux->body_box` | `ctx.vehicle.body_box` |
//! | `aux->wheels[i].surface_source` | `ctx.vehicle.wheels[i].surface_source` |
//! | `contact->friction_input_selector` | `ctx.vehicle.friction_input_selector` |
//! | `contact->event_source_c/ab` | `ctx.vehicle.compute.event_source_c/ab` |
//! | `contact->event_metric_a/b/c` | `ctx.vehicle.compute.event_metric_a/b/c` |
//! | `contact->air_control_immediate` | `ctx.vehicle.air_control_immediate` |
//! | `contact->contact_block_count` | `ctx.vehicle.model6.contact_block_count` |
//! | `contact->timer->tick_time` | `ctx.vehicle.tick_time` |
//! | `contact->wheel_contact_absorb_count` | `ctx.vehicle.wheel_contact_absorb_count` |
//! | `contact->body_contact_count` | `ctx.vehicle.body_contact_count` |
//! | `contact->body_contact_position_sum` | `ctx.vehicle.body_contact_position_sum` |
//! | `contact->body_contact_normal_sum` | `ctx.vehicle.body_contact_normal_sum` |
//! | `model6ctx->state->…` | `ctx.vehicle.model6.…` |
//! | `ctx->vehicle->dyna_state->…` | `ctx.dyna.live().…` |
//! | `ctx->vehicle->dyna_params->…` | `ctx.dyna.params.…` |
//! | tuning views | `ctx.tuning.aux.…` / `.contact` / `.compute` / `.model6` / `.curves` |
//!
//! External seams (supplied by the world at call sites): the water map
//! (`TmnfTrackWater` from the track), the model iso source and body
//! reference position (world/hms-item data), and the static body iso
//! resolver for `contact_body`.

pub mod aux;
pub mod base;
pub mod contact;
pub mod compute;
pub mod curve;
pub mod model6;

use crate::collision::GmBoxAligned;
use crate::dyna::{CHmsDyna, CHmsDynaParams};
use crate::gm::*;
use std::mem::{offset_of, size_of};

/* ===========================================================================
 * Input packet (vehicle.h)
 * ========================================================================= */

/// 0x48-byte input packet consumed by the static 0x004FE500 mapper.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TMNFRaceInputs {
    pub steer_left_time: u32,     /* 0x00 */
    pub reserved04: u32,
    pub steer_left: i32,          /* 0x08 */
    pub steer_right_time: u32,    /* 0x0c */
    pub reserved10: u32,
    pub steer_right: i32,         /* 0x14 */
    pub steer_analog_time: u32,   /* 0x18 */
    pub reserved1c: u32,
    pub steer_analog: f32,        /* 0x20 */
    pub accelerate_time: u32,     /* 0x24 */
    pub reserved28: u32,
    pub accelerate: i32,          /* 0x2c */
    pub brake_time: u32,          /* 0x30 */
    pub reserved34: u32,
    pub brake: i32,               /* 0x38 */
    pub gas_analog_time: u32,     /* 0x3c */
    /// Engine-defined: nonzero on the tick a respawn press applies.
    pub respawn: u32,             /* 0x40 */
    pub gas_analog: f32,          /* 0x44 */
}
const _: () = assert!(size_of::<TMNFRaceInputs>() == 0x48);
const _: () = assert!(offset_of!(TMNFRaceInputs, steer_left) == 0x08);
const _: () = assert!(offset_of!(TMNFRaceInputs, steer_right) == 0x14);
const _: () = assert!(offset_of!(TMNFRaceInputs, steer_analog) == 0x20);
const _: () = assert!(offset_of!(TMNFRaceInputs, accelerate) == 0x2c);
const _: () = assert!(offset_of!(TMNFRaceInputs, brake) == 0x38);
const _: () = assert!(offset_of!(TMNFRaceInputs, gas_analog) == 0x44);

impl TMNFRaceInputs {
    /// Reads the packet from raw little-endian bytes (replay schedules).
    pub fn from_bytes(bytes: &[u8; 0x48]) -> Self {
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        TMNFRaceInputs {
            steer_left_time: u32_at(0x00),
            reserved04: u32_at(0x04),
            steer_left: u32_at(0x08) as i32,
            steer_right_time: u32_at(0x0c),
            reserved10: u32_at(0x10),
            steer_right: u32_at(0x14) as i32,
            steer_analog_time: u32_at(0x18),
            reserved1c: u32_at(0x1c),
            steer_analog: f32_at(0x20),
            accelerate_time: u32_at(0x24),
            reserved28: u32_at(0x28),
            accelerate: u32_at(0x2c) as i32,
            brake_time: u32_at(0x30),
            reserved34: u32_at(0x34),
            brake: u32_at(0x38) as i32,
            gas_analog_time: u32_at(0x3c),
            respawn: u32_at(0x40),
            gas_analog: f32_at(0x44),
        }
    }
}

/* ===========================================================================
 * Curves (vehicle_curve.h)
 * ========================================================================= */

/// Immutable curve keys; `lower_bounds`/`upper_bounds` are the compiled
/// position bounds (positions[i] -/+ the key epsilon under PC=24).
#[derive(Clone, Debug, Default)]
pub struct CFuncKeys {
    pub count: u32,
    pub positions: Vec<f32>,
    pub lower_bounds: Vec<f32>,
    pub upper_bounds: Vec<f32>,
}

/// A real-valued curve.
#[derive(Clone, Debug, Default)]
pub struct CFuncKeysReal {
    pub keys: CFuncKeys,
    pub values: Vec<f32>,
    pub interpolation: i32,
}

/// The curve set referenced by the tuning views (21 curves; `None` where the
/// image carries no curve — e.g. `steering_angle` may be absent).
#[derive(Clone, Debug, Default)]
pub struct CSceneVehicleCarTuningCurveSet {
    pub accel_from_speed: Option<CFuncKeysReal>,
    pub rollover_lateral_from_speed: Option<CFuncKeysReal>,
    pub max_side_friction_from_speed: Option<CFuncKeysReal>,
    pub lateral_contact_slowdown_from_speed: Option<CFuncKeysReal>,
    pub steer_slowdown_from_speed: Option<CFuncKeysReal>,
    pub rollover_lateral_coef_from_angle: Option<CFuncKeysReal>,
    pub steer_drive_torque_from_speed: Option<CFuncKeysReal>,
    pub m4_steer_radius_from_speed: Option<CFuncKeysReal>,
    pub m4_max_friction_force_from_speed: Option<CFuncKeysReal>,
    pub m5_slipping_accel_from_speed: Option<CFuncKeysReal>,
    pub m5_slipping_accel_scale: f32,
    pub water_friction_from_speed: Option<CFuncKeysReal>,
    pub damper_max: f32,
    pub damper_min: f32,
    pub m6_damper_modulation: Option<CFuncKeysReal>,
    pub m6_rear_gear_accel_from_speed: Option<CFuncKeysReal>,
    pub m6_rollover_lateral_from_speed_ratio: Option<CFuncKeysReal>,
    pub m6_burnout_radius_from_speed: Option<CFuncKeysReal>,
    pub m6_lateral_speed_from_burnout_radius: Option<CFuncKeysReal>,
    pub m6_donut_rollover_from_speed: Option<CFuncKeysReal>,
    pub m6_burnout_rollover_from_speed: Option<CFuncKeysReal>,
}

/* ===========================================================================
 * Tuning views (vehicle.h / vehicle_aux.h / vehicle_contact.h /
 * vehicle_compute.h / vehicle_model6.h)
 * ========================================================================= */

/// `CSceneVehicleCarTuning`: the primary 0x3AC-byte tuning block's scalar
/// fields (curve members live in [`CSceneVehicleCarTuningCurveSet`]).
#[derive(Clone, Debug, Default)]
pub struct CSceneVehicleCarTuning {
    pub damper_max: f32,
    pub damper_min: f32,
    pub gear_ratios: Vec<f32>,
    pub gear_upshift: Vec<f32>,
    pub gear_downshift: Vec<f32>,
    pub gear_aux: Vec<f32>,
    pub gear_ratio_count: u32,
    pub engine_model: i32,
    pub forward_speed_limit_scale: f32, /* game +0x02c */
    pub engine_rpm_accel: f32,
    pub engine_rpm_decel: f32,
    pub engine_rpm_reverse_accel: f32,
    pub engine_rpm_high_decel: f32,
    pub engine_rpm_low_accel: f32,
    pub engine_rpm_follow_accel: f32,
    pub engine_rpm_turbo_decel: f32,
    pub speed_32c: f32,
    pub speed_330: f32,
    pub speed_334: f32,
    pub speed_338: f32,
    pub suspension_model: i32,
    pub suspension_stiffness: f32,
    pub suspension_damping: f32,
    pub suspension_rest_length: f32,
    pub suspension_scale: f32,
}

/// `CSceneVehicleCarTuningAux` (aux view; game offsets in comments).
/// `water_map` is NOT part of tuning — it is track data passed at call time.
#[derive(Clone, Debug, Default)]
pub struct CSceneVehicleCarTuningAux {
    pub steering_speed_base: f32,          /* game tuning +0x06c */
    pub steering_speed_scale: f32,         /* game tuning +0x070 */
    pub steering_slew_rate: f32,           /* game tuning +0x094 */
    pub air_torque_linear: f32,            /* game tuning +0x158 */
    pub air_torque_quadratic: f32,         /* game tuning +0x15c */
    pub suspension_follow_rate: f32,       /* game tuning +0x194 */
    pub air_control_window_ticks: u32,     /* game tuning +0x364 */
    pub air_reversal_threshold: f32,       /* game tuning +0x368 */
    pub air_vertical_curve: Option<CFuncKeysReal>,   /* game tuning +0x36c */
    pub steering_angle_curve: Option<CFuncKeysReal>, /* game tuning +0x378 */
    /* 0x007C2910 ApplyWaterForces tuning. */
    pub water_buoyancy: f32,               /* game tuning +0x204 */
    pub water_entry_speed_threshold: f32,  /* game tuning +0x208 */
    pub water_entry_speed_minimum: f32,    /* game tuning +0x20c */
    pub water_impulse_vertical_curve: Option<CFuncKeysReal>,   /* +0x210 */
    pub water_impulse_horizontal_curve: Option<CFuncKeysReal>, /* +0x214 */
    pub water_friction_curve: Option<CFuncKeysReal>,           /* +0x218 */
    pub water_angular_drag_linear: f32,    /* game tuning +0x21c */
    pub water_angular_drag_quadratic: f32, /* game tuning +0x220 */
}

/// `CSceneVehicleMaterialBlendableVals` (0x10 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CSceneVehicleMaterialBlendableVals {
    pub acceleration: f32,
    pub braking: f32,
    pub steering: f32,
    pub lateral_grip: f32,
}
const _: () = assert!(size_of::<CSceneVehicleMaterialBlendableVals>() == 0x10);

/// Per-ground-id material (`TMNFVehicleGroundMaterial`): blend values plus
/// the fake-contact descriptor.
#[derive(Clone, Debug)]
pub struct TMNFVehicleGroundMaterial {
    pub values: [f32; 4],
    /// Material +0x24 mask image (None: no bumps). Shared, immutable.
    pub fake_contact_mask: Option<std::sync::Arc<Vec<u8>>>,
    pub fake_contact_period_x: f32,
    pub fake_contact_period_z: f32,
    pub fake_contact_impulse_scale: f32,
    pub fake_contact_impulse_limit: f32,
}

impl Default for TMNFVehicleGroundMaterial {
    fn default() -> Self {
        TMNFVehicleGroundMaterial {
            values: [0.0; 4],
            fake_contact_mask: None,
            fake_contact_period_x: 0.0,
            fake_contact_period_z: 0.0,
            fake_contact_impulse_scale: 0.0,
            fake_contact_impulse_limit: 0.0,
        }
    }
}

/// `TMNFVehicleContactTuning`.
#[derive(Clone, Debug, Default)]
pub struct TMNFVehicleContactTuning {
    pub friction_force: f32,
    pub extra_friction_force: f32,
    pub slope_adherence_min: f32,
    pub slope_adherence_max: f32,
    pub slope_secondary_min: f32,
    pub slope_secondary_max: f32,
    pub angular_y_scale: f32,
    pub angular_xz_scale: f32,
    pub damper_max: f32,
    pub max_angular_speed: f32,
    pub max_linear_speed_delta: f32,
    pub body_tangent_ratio: f32,
    pub body_tangent_ratio_material4: f32,
    pub restitution_air_material4: f32,
    pub restitution_air: f32,
    pub restitution_ground: f32,
    pub restitution_ground_material4: f32,
    pub lateral_linear: f32,
    pub lateral_quadratic: f32,
    pub lateral_ground_scale: f32,
    pub lateral_contact_duration_ticks: u32,
    pub wheel_contact_model: i32,
    pub friction_model: i32,
}

/// `TMNFVehicleComputeTuning`.
#[derive(Clone, Debug, Default)]
pub struct TMNFVehicleComputeTuning {
    pub event_c_level1_max: f32,            /* tuning +0x028 */
    pub grounded_drag_term: f32,            /* tuning +0x058 */
    pub active_contact_stop_threshold: f32, /* tuning +0x0a4 */
    pub normal_turbo_factor: f32,           /* tuning +0x0f0 */
    pub roulette_turbo_factor: f32,         /* tuning +0x0f4 */
    pub normal_turbo_duration: u32,         /* tuning +0x0f8 */
    pub roulette_turbo_duration: u32,       /* tuning +0x0fc */
    pub air_impulse_scale: f32,             /* tuning +0x104 */
    pub special_force_field_scale: f32,     /* tuning +0x108 */
    pub airborne_linear_drag: f32,          /* tuning +0x154 */
    pub grounded_force_field_scale: f32,    /* tuning +0x160 */
    pub airborne_force_field_scale: f32,    /* tuning +0x164 */
    pub normalized_force_divisor: f32,      /* tuning +0x228 */
    pub effect_curve: Option<CFuncKeysReal>, /* tuning +0x380 */
    pub effect_curve_bias: f32,             /* tuning +0x384 */
    pub event_ab_level2_min: f32,           /* tuning +0x398 */
    pub event_ab_trigger: f32,              /* tuning +0x39c */
    pub event_c_trigger: f32,               /* tuning +0x3a0 */
    pub contact_decay_curve: Option<CFuncKeysReal>, /* primary tuning +0x044 */
    pub contact_rise_curve: Option<CFuncKeysReal>,  /* primary tuning +0x048 */
}

/// `CSceneVehicleCarModel6Tuning` — the model 6 scalar view (game offsets in
/// the C header; see `vehicle_model6.h` for the full table).
#[derive(Clone, Debug, Default)]
pub struct CSceneVehicleCarModel6Tuning {
    pub forward_speed_limit_scale: f32,
    pub reverse_speed_limit_scale: f32,
    pub brake_base: f32,
    pub brake_speed_scale: f32,
    pub forward_brake_limit_sliding: f32,
    pub forward_brake_limit: f32,
    pub speed_limit_force: f32,
    pub vertical_force_scale: f32,
    pub wheel_steer_sine_limit: f32,
    pub steer_slowdown_scale: f32,
    pub wheel_torque_scale: f32,
    pub sliding_steer_torque_scale: f32,
    pub lateral_force_scale: f32,
    pub sliding_lateral_limit_scale: f32,
    pub lateral_overflow_blend: f32,
    pub wheel_overflow_blend: f32,
    pub vertical_force_divisor: f32,
    pub traction_loss_scale: f32,
    pub burnout_trigger_scale: f32,
    pub burnout_trigger_limit: f32,
    pub rollover_axis_min_length: f32,
    pub rollover_torque_x_scale: f32,
    pub rollover_torque_z_scale: f32,
    pub sliding_brake_scale: f32,
    pub braking_lateral_limit_scale: f32,
    pub reverse_brake_limit_sliding: f32,
    pub reverse_brake_limit: f32,
    pub burnout_speed_max: f32,
    pub burnout_speed_min: f32,
    pub donut_lateral_force_scale: f32,
    pub donut_yaw_angle_scale: f32,
    pub donut_steer_linear: f32,
    pub donut_steer_quadratic: f32,
    pub donut_countersteer_scale: f32,
    pub donut_radius_exponent: f32,
    pub donut_radial_speed_exponent: f32,
    pub donut_radius_min: f32,
    pub donut_lateral_speed_limit: f32,
    pub donut_normal_angle_limit: f32,
    pub donut_angle_positive_limit: f32,
    pub donut_angle_negative_limit: f32,
    pub burnout_enter_ticks: u32,
    pub burnout_enter_accel_scale: f32,
    pub burnout_enter_lateral_scale: f32,
    pub burnout_exit_ticks: u32,
    pub burnout_exit_accel_scale: f32,
    pub burnout_exit_extra_accel: f32,
    pub material6_longitudinal_scale: f32,
    pub material6_gas_denominator: f32,
    pub material6_vertical_shape: f32,
    pub material6_vertical_scale: f32,
}

/// The complete immutable tuning bundle (all views over the one 0x3AC tuning
/// block + the ground materials of the car image). Duplicated views are
/// intentional: `world.rs` fills each exactly as the C decode does.
#[derive(Clone, Debug, Default)]
pub struct VehicleTuning {
    pub base: CSceneVehicleCarTuning,
    pub aux: CSceneVehicleCarTuningAux,
    pub contact: TMNFVehicleContactTuning,
    pub compute: TMNFVehicleComputeTuning,
    pub model6: CSceneVehicleCarModel6Tuning,
    pub curves: CSceneVehicleCarTuningCurveSet,
    /// 31-entry ground-id table (car +0x6c/+0x70) indexing ground_materials.
    pub ground_material_indices: Vec<u32>,
    pub ground_materials: Vec<TMNFVehicleGroundMaterial>,
    /// Wiring: object_refs of the four wheel collision trees (car image).
    pub wheel_tree_refs: Vec<u32>,
    /// Wiring: object_refs of the body collision trees.
    pub body_tree_refs: Vec<u32>,
}

/* ===========================================================================
 * Wheels (vehicle.h + vehicle_aux.h + vehicle_contact.h, folded)
 * ========================================================================= */

/// 0x30 bytes; pure-data engine state embedded at CSceneVehicleCar+0x59C.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CSceneVehicleCarEngine {
    pub max_rpm: f32,          /* 0x00 */
    pub reserved04: [u8; 0x10],
    pub braking_factor: f32,   /* 0x14 */
    pub rpm: f32,              /* 0x18 */
    pub target_rpm: f32,       /* 0x1c */
    pub clutch: f32,           /* 0x20 */
    pub shift_timer: f32,      /* 0x24 */
    pub reverse: i32,          /* 0x28 */
    pub gear: i32,             /* 0x2c */
}
const _: () = assert!(size_of::<CSceneVehicleCarEngine>() == 0x30);

/// 0xa8 bytes; pure-data real-time wheel state embedded at wheel+0xB4.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CSceneVehicleCarWheelRealTimeState {
    pub damper_absorb: f32,       /* 0x00 */
    pub field04: f32,
    pub field08: f32,
    pub basis0: GmMat3,           /* 0x0c */
    pub basis1: GmMat3,           /* 0x30 */
    pub field54: GmVec3,          /* 0x54 */
    pub reserved60: [u8; 0x0c],
    pub field6c: f32,
    pub has_ground_contact: i32,  /* 0x70 */
    pub contact_material_id: i32, /* 0x74 */
    pub is_sliding: i32,          /* 0x78 */
    pub relative_rotz_axis: GmVec3, /* 0x7c */
    /// Game 32-bit pointer — carried as the raw word by captures; the port
    /// tracks the contacted static body by corpus ref instead.
    pub contact_body_word: u32,   /* 0x88 */
    pub ground_contact_count: i32, /* 0x8c */
    pub field90: GmVec3,
    pub rotation_phase: f32,      /* 0x9c */
    pub blend_value: f32,         /* 0xa0 */
    pub blend_target: f32,        /* 0xa4 */
}
const _: () = assert!(size_of::<CSceneVehicleCarWheelRealTimeState>() == 0xa8);

/// 0x64 bytes of opaque per-history-slot wheel state.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CSceneVehicleCarWheelState {
    pub bytes: [u8; 0x64],
}
impl Default for CSceneVehicleCarWheelState {
    fn default() -> Self {
        CSceneVehicleCarWheelState { bytes: [0; 0x64] }
    }
}

/// 0x14 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSpringFloat {
    pub stiffness: f32, /* 0x00 */
    pub damping: f32,   /* 0x04 */
    pub value: f32,     /* 0x08 */
    pub target: f32,    /* 0x0c */
    pub velocity: f32,  /* 0x10 */
}
const _: () = assert!(size_of::<GmSpringFloat>() == 0x14);

/// One wheel: `CSceneVehicleCarWheel` (+aux surface isos, +contact fields),
/// with the 680-byte v6 history prefix preserved.
#[derive(Clone, Debug)]
pub struct CSceneVehicleCarWheel {
    pub active: i32,                     /* +0x000 */
    pub steerable: i32,                  /* +0x004 */
    pub radius: f32,                     /* +0x008 */
    pub surface_source: GmIso4,          /* +0x010 (aux view) */
    pub surface_location: GmIso4,        /* +0x040 (aux view) */
    pub field70: [f32; 12],              /* +0x070..+0x0a0 */
    pub fielda0: i32,
    pub fielda4: i32,
    pub offset_from_vehicle: GmVec3,     /* +0x0a8 */
    pub real_time: CSceneVehicleCarWheelRealTimeState, /* +0x0b4 */
    pub field15c: i32,                   /* +0x15c (contact_relative_valid) */
    pub contact_relative_local_distance: GmVec3, /* +0x160 */
    /// Wheel +0x64 impulse point (contact view's copy).
    pub impulse_point: GmVec3,
    /// Contacted static body, by corpus ref (the C stored a pointer).
    pub contact_body: Option<u32>,
    /* v6 snapshot history (400 bytes) — capture/restore/respawn only. */
    pub previous_sync: CSceneVehicleCarWheelState, /* +0x16c */
    pub sync: CSceneVehicleCarWheelState,          /* +0x1d0 */
    pub field234: CSceneVehicleCarWheelState,      /* +0x234 */
    pub async_state: CSceneVehicleCarWheelState,   /* +0x298 */
}

impl Default for CSceneVehicleCarWheel {
    fn default() -> Self {
        CSceneVehicleCarWheel {
            active: 0,
            steerable: 0,
            radius: 0.0,
            surface_source: GmIso4::IDENTITY,
            surface_location: GmIso4::IDENTITY,
            field70: [0.0; 12],
            fielda0: 0,
            fielda4: 0,
            offset_from_vehicle: GmVec3::ZERO,
            real_time: CSceneVehicleCarWheelRealTimeState::default(),
            field15c: 0,
            contact_relative_local_distance: GmVec3::ZERO,
            impulse_point: GmVec3::ZERO,
            contact_body: None,
            previous_sync: CSceneVehicleCarWheelState::default(),
            sync: CSceneVehicleCarWheelState::default(),
            field234: CSceneVehicleCarWheelState::default(),
            async_state: CSceneVehicleCarWheelState::default(),
        }
    }
}

/// Turbo type (vehicle_aux.h).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TMNFVehicleTurboType {
    #[default]
    None = 0,
    Normal = 1,
    Roulette = 2,
}

/* ===========================================================================
 * Model 6 + compute state blocks
 * ========================================================================= */

/// `CSceneVehicleCarModel6State` (game offsets in the C header).
#[derive(Clone, Debug)]
pub struct CSceneVehicleCarModel6State {
    pub pivot_position: GmVec3,       /* +1dc */
    pub pivot_axis: GmVec3,           /* +1e8 */
    pub reverse_mode: i32,            /* +5c4 */
    pub reverse_speed_threshold: f32, /* +5cc */
    pub contact_block_count: i32,     /* +5d8 */
    pub side_contact: i32,            /* +5dc */
    pub last_sliding_tick: u32,       /* +62c */
    pub sliding_start_tick: u32,      /* +630 */
    pub sliding_elapsed_ticks: u32,   /* +634 */
    pub model_iso: GmIso4,            /* +6a4 */
    pub rollover_axis: GmVec3,        /* +6d4 */
    pub orbit_center: GmVec3,         /* +6e0 */
    pub orbit_initial_radius: f32,    /* +6ec */
    pub orbit_radius: f32,            /* +6f0 */
    pub burnout_start_tick: u32,      /* +6f4 */
    pub burnout_transition_tick: u32, /* +6f8 */
    pub orbit_axis: GmVec3,           /* +6fc */
    pub orbit_sign: f32,              /* +708 */
    pub axle_width: f32,              /* +840 */
}

impl Default for CSceneVehicleCarModel6State {
    fn default() -> Self {
        CSceneVehicleCarModel6State {
            pivot_position: GmVec3::ZERO,
            pivot_axis: GmVec3::ZERO,
            reverse_mode: 0,
            reverse_speed_threshold: 0.0,
            contact_block_count: 0,
            side_contact: 0,
            last_sliding_tick: 0,
            sliding_start_tick: 0,
            sliding_elapsed_ticks: 0,
            model_iso: GmIso4::IDENTITY,
            rollover_axis: GmVec3::ZERO,
            orbit_center: GmVec3::ZERO,
            orbit_initial_radius: 0.0,
            orbit_radius: 0.0,
            burnout_start_tick: 0,
            burnout_transition_tick: 0,
            orbit_axis: GmVec3::ZERO,
            orbit_sign: 0.0,
            axle_width: 0.0,
        }
    }
}

/// `TMNFVehicleComputeState` (car fields used only by the top force caller).
#[derive(Clone, Debug, Default)]
pub struct TMNFVehicleComputeState {
    pub simulation_gate: f32,              /* car +0x1e8 */
    pub event_level_c: u32,                /* car +0x1fc */
    pub event_source_c: u8,                /* car +0x200 */
    pub event_source_ab: u8,               /* car +0x201 */
    pub spring_c: GmSpringFloat,           /* car +0x214 */
    pub spring_a: GmSpringFloat,           /* car +0x228 */
    pub contact_rise: f32,                 /* car +0x23c */
    pub contact_decay: f32,                /* car +0x240 */
    pub history_force_limit: f32,          /* car +0x244 */
    pub history_force_scale: f32,          /* car +0x24c */
    pub spring_value_limit: f32,           /* car +0x250 */
    pub local_speed_limit: f32,            /* car +0x2e0 */
    pub air_effect_threshold: f32,         /* car +0x05c */
    pub brake_input_scale: f32,            /* car +0x5a8 */
    pub grounded_drag_scale: f32,          /* car +0x5ac */
    pub computed_brake_force: f32,         /* car +0x5b0 */
    pub state_5d8: i32,                    /* car +0x5d8 */
    pub air_impulse_cooldown_tick: u32,    /* car +0x610 */
    pub effect_accumulator: f32,           /* car +0x624 */
    pub last_force_tick: u32,              /* car +0x650 */
    pub event_level_a: u32,                /* car +0x654 */
    pub event_level_b: u32,                /* car +0x658 */
    pub peak_event_level_b: u32,           /* car +0x660 */
    pub peak_event_level_a: u32,           /* car +0x664 */
    pub peak_event_level_c: u32,           /* car +0x668 */
    pub peak_event_source_ab: u8,          /* car +0x66c */
    pub peak_event_source_c: u8,           /* car +0x66d */
    pub event_metric_a: f32,               /* car +0x670 */
    pub event_metric_b: f32,               /* car +0x674 */
    pub event_metric_c: f32,               /* car +0x678 */
    pub normalized_force: GmVec3,          /* car +0x6d4 */
    pub air_effect_mode: i32,              /* car +0x74c */
}

/* ===========================================================================
 * The vehicle (all mutable state) and the context bundle
 * ========================================================================= */

/// ALL mutable per-simulation vehicle state: the `CSceneVehicleCar` fields
/// (game offsets preserved in comments) plus the aux/contact/compute/model6
/// state that the C kept in separate context structs.
#[derive(Clone, Debug, Default)]
pub struct Vehicle {
    /* --- CSceneVehicleCar --- */
    pub input_gas: f32,            /* +0x050 */
    pub input_brake: f32,          /* +0x054 */
    pub input_steer: f32,          /* +0x058 */
    pub wheels: Vec<CSceneVehicleCarWheel>,
    pub wheel_count: u32,          /* +0x2ec */
    pub engine: CSceneVehicleCarEngine, /* +0x59c */
    pub current_local_speed: GmVec3, /* +0x70c */
    pub total_force_added: GmVec3,   /* +0x818 */
    pub total_impulse_added: GmVec3, /* +0x824 */
    pub engine_mode: i32,          /* +0x2e4 */
    pub turbo_active: i32,         /* +0x628 */
    pub drive_mode: i32,           /* +0x69c */
    pub engine_limit_flag: i32,    /* +0x744 */
    pub gear_downshift_flag: i32,  /* +0x748 */
    pub force_wheel_speed: i32,    /* +0x6a0 */
    pub flag_60c: i32,             /* +0x60c */
    pub block_wheel_speed: i32,    /* +0x73c */
    pub forced_wheel_speed: f32,   /* tuning-selected source +0x2bc */

    /* --- CSceneVehicleCarAuxContext state --- */
    pub integration_flags: u32,    /* +0x2f4 */
    pub turbo_epoch_tick: u32,     /* +0x5d0 */
    pub air_control_immediate: i32, /* +0x5d4 */
    pub air_control_locked: i32,   /* +0x5e4 */
    pub steering_value: f32,       /* +0x5e8 */
    pub turbo_progress: f32,       /* +0x5f0 */
    pub turbo_factor: f32,         /* +0x5f4 */
    pub turbo_start_tick: u32,     /* +0x5f8 */
    pub turbo_end_tick: u32,       /* +0x5fc */
    pub turbo_type: TMNFVehicleTurboType, /* +0x600 */
    pub roulette_token: u32,       /* +0x604 */
    pub roulette_value: f32,       /* +0x608 */
    pub air_control_tick: u32,     /* +0x614 */
    pub air_control_speed: GmVec3, /* +0x618 */
    /// Runtime DAT_00d06a74.
    pub roulette_modulus: u32,
    /// Game car +0x1dc..+0x1f4: the body box read by ApplyWaterForces (the
    /// same words as model6 pivot_position/pivot_axis; both views decoded
    /// from the snapshot, never written).
    pub body_box: GmBoxAligned,
    /// Game car +0x26c != NULL.
    pub turbo_sound_attached: bool,

    /* --- TMNFVehicleContactContext state --- */
    pub friction_input_selector: i32, /* +0x5c4 */
    pub side_contact: i32,          /* +0x5dc */
    pub last_side_contact_tick: u32, /* +0x5e0 */
    pub airborne_friction_gate: i32, /* +0x5e4 */
    pub wheel_contact_absorb_count: u32,
    pub body_contact_count: u32,
    pub body_contact_position_sum: GmVec3,
    pub body_contact_normal_sum: GmVec3,
    /// TMNFVehicleContactTimer (CMwTimerAdapter tick).
    pub tick_time: u32,

    /* --- model 6 + compute state --- */
    pub model6: CSceneVehicleCarModel6State,
    pub compute: TMNFVehicleComputeState,
}

/// The bundle passed where the C passed one of its context structs.
/// The C reached the rigid body through `vehicle->dyna_state` /
/// `vehicle->dyna_params`; here the dyna travels beside the vehicle.
pub struct VehicleCtx<'a> {
    pub vehicle: &'a mut Vehicle,
    pub tuning: &'a VehicleTuning,
    pub dyna: &'a mut CHmsDyna,
}

impl VehicleCtx<'_> {
    /// The live dyna state (car->dyna_state).
    #[inline]
    pub fn live(&self) -> &crate::dyna::CHmsStateDyna {
        self.dyna.live()
    }

    #[inline]
    pub fn live_mut(&mut self) -> &mut crate::dyna::CHmsStateDyna {
        self.dyna.live_mut()
    }

    /// The dyna params (car->dyna_params).
    #[inline]
    pub fn params(&self) -> &CHmsDynaParams {
        &self.dyna.params
    }
}

/// Source of the virtual location copied to car+0x6A4 at model-6 entry and
/// the body reference position (`*(this->hms_item->field14)+0x50`): supplied
/// by the world at call time.
#[derive(Clone, Copy, Debug)]
pub struct Model6External {
    pub model_iso_source: GmIso4,
    pub body_reference_position: GmVec3,
}
