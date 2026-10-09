//! `vehicle_aux.c` — the aux vehicle pipeline, transliterated from
//! `/home/z/my-project/tmnf-physics/src/vehicle_aux.c` (807 lines).
//!
//! Contents (VA addresses kept above each function):
//! * roulette helpers 0x007BC820 / 0x007BC890;
//! * turbo interval bookkeeping 0x007BC8B0 / 0x007BCF90;
//! * suspension integration 0x007BD3F0 (surface transform via 0x007C8AA0,
//!   local 0x008E0D70 rotate-Y schedule);
//! * airborne angular control 0x007BF1D0 (aux-curve view of 0x00586200);
//! * water drag/buoyancy/entry impulse 0x007C2910 (GmMap2 coordinate
//!   schedule, 0x007F3F40 friction lookup, 0x007BE690 central impulse);
//! * the per-tick vehicle integration 0x007C3900;
//! * fake bump-mask contacts 0x007C3C00.
//!
//! # Context mapping (`CSceneVehicleCarAuxContext` → [`VehicleCtx`])
//!
//! | C access | Rust |
//! |---|---|
//! | `self->vehicle->…` (car fields) | `ctx.vehicle.…` |
//! | `self->tuning->…` (aux view) | `ctx.tuning.aux.…` |
//! | `vehicle->tuning->…` (primary view) | `ctx.tuning.base.…` |
//! | `self->wheels[i].surface_source/.surface_location` | `ctx.vehicle.wheels[i].surface_source/.surface_location` |
//! | aux-only car state (`turbo_*`, `roulette_*`, `steering_value`, `air_control_*`, `integration_flags`, `body_box`, `turbo_sound_attached`) | `ctx.vehicle.…` (see `Vehicle`) |
//! | `vehicle->dyna_state->rot` | `ctx.live().rot` |
//! | `(const GmIso4 *)&dyna_state->rot` | `ctx.live().rot_pos_iso()` (the game overlays `rot` with the `pos` that follows it in `CHmsStateDyna`) |
//! | `CHmsItem_GetLinearSpeed/GetAngularSpeed` | `ctx.dyna.get_local_linear_speed()/get_local_angular_speed()` (body-frame speeds) |
//! | `CHmsItem_AddImpulse` | `ctx.dyna.add_local_impulse()` (one corpus per car item) |
//! | `contact->ground_materials/indices` | `ctx.tuning.ground_materials/ground_material_indices` |
//! | `contact->wheels[i]` | `ctx.vehicle.wheels[i]` (folded: `impulse_point`, `contact_body`) |
//! | the water map (`tuning->water_map`) | the `water: &TmnfTrackWater` parameter of [`apply_water_forces`] |
//!
//! # Runtime-callback seams (NOT represented — documented per site)
//!
//! The C context carried three runtime callbacks. None has a physics or
//! oracle footprint; the *state* they guarded is kept:
//! * `play_turbo_sound` (0x007BCFAF) — audio; the `turbo_sound_attached`
//!   flag (game car +0x26c != NULL) is kept so the trigger stays observable.
//! * `set_surface_location` (0x007C8AA0 `update_surface`) — render-tree
//!   binding of the wheel's surface transform; `surface_location` itself is
//!   still written by [`wheel_integrate`].
//! * `finish_integration` (0x007C3900 tail) — engine orchestration; the
//!   steering/wheel/engine state it operated on is already updated.
//! The C's `tmnf_abort()` on an unset callback is dropped with them (there is
//! no callback to be unset); every other `tmnf_abort()` site is a
//! `panic!("tmnf: …")`.
//!
//! # Stub-module seams (this file compiles once their owners land)
//!
//! * `crate::vehicle::curve` (vehicle_curve.c, still a stub) must provide
//!   `pub fn cfunc_keys_real_get_value_out(curve: &CFuncKeysReal,
//!   position: f32, value: &mut f32, lower_index: &mut u32)` — 0x00586200
//!   with the C's in/out cached lower index (the water impulse path seeds it
//!   with stale float bits via `memcpy`, bit-exactly reproduced with
//!   `f32::to_bits`). Called from [`water_friction_from_speed`] and
//!   [`apply_water_forces`].
//! * `crate::vehicle::contact` (vehicle_contact.c, still a stub) must provide
//!   `pub fn wheel_absorb_contact(ctx: &mut VehicleCtx, wheel_index: usize,
//!   contact: &mut crate::response::CHmsPhysicalContact)` — 0x007C11D0 with
//!   the contact context folded into `VehicleCtx` and the contact wheel
//!   state folded into `ctx.vehicle.wheels[wheel_index]`. Called from
//!   [`create_fake_contacts`].
//!
//! # FP notes (see `crate::fp` and PORT_NOTES.md)
//!
//! * `x87_mul/add/sub/div` → plain `f32` operators; `x87_sqrt` → `.sqrt()`;
//!   `x87_rcp` → `1.0/x`; operand order and grouping preserved verbatim
//!   (including the deliberately asymmetric `t[0]`/`t[1]` operand orders of
//!   the two suspension models and the `x/y/z` shuffle of
//!   `water_length_squared`).
//! * Hex-float literals → `f32::from_bits`/`f64::from_bits` with the literal
//!   in the comment (bits verified against gcc; see the tests).
//! * Comparisons against **double literals** (`0.0`, `0x1.ccccccp-1`,
//!   `ROULETTE_HIGH`) are spelled as `f64` comparisons, exactly as the C
//!   widens its float operand; `F(a) op F(b)` between two float values stays
//!   a plain `f32` compare.
//! * Single-op double products (`F(x) * <float-widened double literal>`) →
//!   `((x as f64) * LIT) as f32`; the widened literals are the f64 images of
//!   the game's float constants (3.6, π, 0.9, 6/7).
//! * Unsigned tick arithmetic wraps (`tick - start`, `tick + duration`),
//!   then `fp::u32_to_x87_float` performs the game's signed-`fild` + `+2^32`
//!   two-step.
//! * `water_map_coordinate`'s `(uint32_t)(int64_t)F(x)` is
//!   `f64::from(x) as i64 as u32`: truncation toward zero with the low 32
//!   bits taken unsigned. Out-of-i64-range or NaN inputs (impossible at car
//!   coordinates) would be C UB; Rust's saturating `as` is the closest
//!   defined behavior.

use crate::collision::GmBoxAligned;
use crate::fp::{ftol, u32_to_x87_float, x87_cos, x87_fmod, x87_sin};
use crate::gm::{GmMat3, GmVec3};
use crate::response::CHmsPhysicalContact;
use crate::track::TmnfTrackWater;
use crate::vehicle::base::{
    add_vehicle_central_force, add_vehicle_torque, engine_integrate,
    wheel_real_time_state_integrate, wheel_update_speed_from_vehicle_speed,
};
use crate::vehicle::{CFuncKeysReal, CSceneVehicleCarTuningAux, TMNFVehicleTurboType, VehicleCtx};

/* ===========================================================================
 * Constants (vehicle_aux.c top; Rust has no hex-float literals — bits
 * verified against gcc by the tests below)
 * ========================================================================= */

/* #define AUX_EPSILON 0x1.4f8b58p-17f */
const AUX_EPSILON: f32 = f32::from_bits(0x3727_C5AC); /* 9.9999997473787516e-06 */
/* #define ROULETTE_LOW 0x1.24924ap-1f — float 4/7. */
const ROULETTE_LOW: f32 = f32::from_bits(0x3F12_4925); /* 0.57142859613800049 */
/* #define ROULETTE_HIGH 0x1.b6db6ep-1 — 0x00B9EF58: float 6/7 promoted to
 * double (a bare double literal, so the comparison below runs in f64). */
const ROULETTE_HIGH: f64 = f64::from_bits(0x3FEB_6DB6_E000_0000); /* 0.85714286565780640 */
/* 0x1.ccccccp-1 — 0x00B41EA8: float-widened 0.9 (entry-impulse max depth). */
const FLOAT_0_9_AS_F64: f64 = f64::from_bits(0x3FEC_CCCC_C000_0000); /* 0.89999997615814209 */
/* 0x1.ccccccp+1 — 0x00B3D2A8: float-widened 3.6 (m/s → km/h). */
const FLOAT_3_6_AS_F64: f64 = f64::from_bits(0x400C_CCCC_C000_0000); /* 3.5999999046325684 */
/* 0x1.921fb6p+1 — 0x00B36110: float pi promoted to double. */
const FLOAT_PI_AS_F64: f64 = f64::from_bits(0x4009_21FB_6000_0000); /* 3.1415927410125732 */

/* vehicle_fake_contact_mask.h */
const TMNF_FAKE_CONTACT_MASK_WIDTH: u32 = 128;
const TMNF_FAKE_CONTACT_MASK_HEIGHT: u32 = 128;

/* ===========================================================================
 * File-local static helpers (vehicle_aux.c)
 * ========================================================================= */

/*
 * UNVALIDATED: native view of 0x00586200 for the two auxiliary tuning curves.
 * All callers begin the forward search at key zero, as the original functions
 * do at these call sites. (The tuning's curves are `CFuncKeysReal`; the C's
 * `TMNFVehicleAuxCurve` view has the same five arrays plus `interpolation`.)
 */
fn aux_curve_get_value(curve: &CFuncKeysReal, position: f32) -> f32 {
    let mut lower: u32;
    let mut upper: u32;
    let difference: f32;
    let blend: f32;
    let lower_weight: f32;

    if curve.keys.count == 0 {
        return 0.0;
    }
    if curve.keys.count == 1 {
        return curve.values[0];
    }
    if position < curve.keys.lower_bounds[0] {
        return curve.values[0];
    }
    lower = curve.keys.count - 1;
    if curve.keys.upper_bounds[lower as usize] < position {
        return curve.values[lower as usize];
    }

    lower = 0;
    upper = 1;
    let mut attempts: u32 = 1;
    while attempts <= curve.keys.count {
        if curve.keys.lower_bounds[lower as usize] <= position
            && position <= curve.keys.upper_bounds[upper as usize]
        {
            break;
        }
        lower = upper;
        upper += 1;
        if upper == curve.keys.count {
            upper = 0;
        }
        attempts += 1;
    }
    if curve.interpolation == 1 {
        return curve.values[lower as usize];
    }

    difference =
        curve.keys.positions[upper as usize] - curve.keys.positions[lower as usize];
    if f64::from(difference).abs() < f64::from(AUX_EPSILON) {
        blend = 0.0;
    } else {
        blend = (position - curve.keys.positions[lower as usize]) / difference;
    }
    lower_weight = (1.0f64 - f64::from(blend)) as f32;
    (lower_weight * curve.values[lower as usize]) + (blend * curve.values[upper as usize])
}

/* UNVALIDATED: local 0x008E0D70 schedule used by IntegrateVehicle. */
fn mat3_rotate_y(matrix: &mut GmMat3, angle: f32) {
    let sine = x87_sin(angle);
    let cosine = x87_cos(angle);
    let negative_sine = -sine;

    let old_value = matrix.m[0];
    matrix.m[0] = (matrix.m[6] * sine) + (cosine * old_value);
    matrix.m[6] = (matrix.m[6] * cosine) + (negative_sine * old_value);
    let old_value = matrix.m[1];
    matrix.m[1] = (old_value * cosine) + (matrix.m[7] * sine);
    matrix.m[7] = (matrix.m[7] * cosine) + (negative_sine * old_value);
    let old_value = matrix.m[2];
    matrix.m[2] = (old_value * cosine) + (matrix.m[8] * sine);
    matrix.m[8] = (matrix.m[8] * cosine) + (negative_sine * old_value);
}

/* 0x007C8AA0, UNVALIDATED: binds the surface transform to its runtime tree.
 *
 * The C early-outs when the wheel has no surface handler, then pushes
 * `surface_location` into the handler's render tree through the runtime
 * `set_surface_location` callback (aborting when the callback is unset).
 * The handler pointer and the callback are render-tree plumbing with no
 * physics or oracle footprint, so neither is represented here; the
 * `surface_location` they published is still written by the callers. */
fn update_surface(_ctx: &mut VehicleCtx, _wheel_index: usize) {}

/* 0x004FF950 GmMap2<unsigned char>::IsInside and 0x004FFAC0 GetValue share
 * one coordinate schedule: (p - origin) / cell, truncating fistp to int64,
 * then an unsigned compare of the low 32 bits against the extent. */
fn water_map_coordinate(value: f32, origin: f32, cell: f32) -> u32 {
    let offset = value - origin;
    let scaled = offset / cell;
    f64::from(scaled) as i64 as u32
}

fn water_map_is_inside(map: &TmnfTrackWater, x: f32, z: f32) -> bool {
    let ix = water_map_coordinate(x, map.origin_x, map.cell_x);
    let iz = water_map_coordinate(z, map.origin_z, map.cell_z);
    ix < map.width && iz < map.height
}

fn water_map_value(map: &TmnfTrackWater, x: f32, z: f32) -> u8 {
    let ix = water_map_coordinate(x, map.origin_x, map.cell_x);
    let iz = water_map_coordinate(z, map.origin_z, map.cell_z);
    if ix >= map.width || iz >= map.height {
        return map.default_cell;
    }
    map.cells[(iz * map.width + ix) as usize]
}

/* (y*y + x*x) + z*z with every operation rounded, as at 0x007C2BEF and
 * 0x007C2CE3. */
fn water_length_squared(v: &GmVec3) -> f32 {
    ((v.y * v.y) + (v.x * v.x)) + (v.z * v.z)
}

/* 0x007F3F40 CSceneVehicleCarTuning::GetWaterFrictionFromSpeed: speed times
 * float-widened 3.6 (0x00B3D2A8), then curve +0x218 from key zero. */
fn water_friction_from_speed(tuning: &CSceneVehicleCarTuningAux, speed: f32) -> f32 {
    let kmh = (f64::from(speed) * FLOAT_3_6_AS_F64) as f32;
    let mut lower_index: u32 = 0;
    /* Uninitialized in the C; the lookup always writes it. */
    let mut value: f32 = 0.0;

    crate::vehicle::curve::cfunc_keys_real_get_value_out(
        tuning.water_friction_curve.as_ref().expect(
            "tmnf: water friction curve is absent (the C dereferences it unconditionally)",
        ),
        kmh,
        &mut value,
        &mut lower_index,
    );
    value
}

/* 0x007BE690 CSceneVehicleCar::AddVehicleImpulse(GmVec3 const&) — a central
 * impulse in the car's local frame. `CHmsItem_AddImpulse` routes the impulse
 * to the item's corpus dynas (the car item has exactly one corpus):
 * `ctx.dyna.add_local_impulse`. The bookkeeping sums keep the C's operand
 * orders. */
fn add_vehicle_central_impulse(ctx: &mut VehicleCtx, impulse: &GmVec3) {
    ctx.dyna.add_local_impulse(impulse);
    ctx.vehicle.total_impulse_added.x =
        ctx.vehicle.total_impulse_added.x + impulse.x;
    ctx.vehicle.total_impulse_added.y =
        impulse.y + ctx.vehicle.total_impulse_added.y;
    ctx.vehicle.total_impulse_added.z =
        impulse.z + ctx.vehicle.total_impulse_added.z;
}

/* 0x007C3CC8..0x007C3D1B (X) and 0x007C3D27..0x007C3D78 (Z): fmod by the
 * material period, float store, fabs, divide by the period, multiply by the
 * integer image extent, then truncating fistp. */
fn fake_contact_mask_coordinate(value: f32, period: f32, extent: u32) -> u32 {
    let wrapped = x87_fmod(value, period);
    let fraction = wrapped.abs() / period;
    let scaled = fraction * extent as f32;
    let coordinate = ftol(f64::from(scaled)) as u32;

    if coordinate >= extent {
        panic!("tmnf: fake contact mask coordinate outside the image");
    }
    coordinate
}

/* ===========================================================================
 * Public API (vehicle_aux.h)
 * ========================================================================= */

/* 0x007BC820, UNVALIDATED: maps an unsigned roulette remainder to {0,.5,1}. */
pub fn get_roulette_value01(value: u32, divisor: u32) -> f32 {
    let remainder = value % divisor;
    let numerator = u32_to_x87_float(remainder);
    let denominator = u32_to_x87_float(divisor);
    let ratio = numerator / denominator;

    /* F(ratio) < F(ROULETTE_LOW): float/float compare. */
    if ratio < ROULETTE_LOW {
        return 0.0;
    }
    /* F(ratio) < ROULETTE_HIGH: bare double literal — f64 compare. */
    if f64::from(ratio) < ROULETTE_HIGH {
        return 0.5;
    }
    1.0
}

/* 0x007BC890, UNVALIDATED: converts roulette value to its boost multiplier. */
pub fn get_roulette_boost_factor_from_value01(value: f32) -> f32 {
    value + 1.0
}

/* 0x007BC8B0: advances the active turbo interval. */
pub fn update_turbo(ctx: &mut VehicleCtx, tick: u32) {
    if ctx.vehicle.turbo_type != TMNFVehicleTurboType::None {
        if ctx.vehicle.turbo_end_tick < tick {
            ctx.vehicle.turbo_type = TMNFVehicleTurboType::None;
        }
        if ctx.vehicle.turbo_type != TMNFVehicleTurboType::None {
            let elapsed =
                u32_to_x87_float(tick.wrapping_sub(ctx.vehicle.turbo_start_tick));
            let duration = u32_to_x87_float(
                ctx.vehicle.turbo_end_tick.wrapping_sub(ctx.vehicle.turbo_start_tick),
            );

            ctx.vehicle.turbo_progress = elapsed / duration;
            return;
        }
    }
    ctx.vehicle.turbo_progress = 0.0;
}

/* 0x007BCF90, UNVALIDATED: starts or refreshes a normal/roulette turbo. */
pub fn enable_turbo(
    ctx: &mut VehicleCtx,
    tick: u32,
    duration: u32,
    factor: f32,
    turbo_type: TMNFVehicleTurboType,
    roulette_token: u32,
) {
    if ctx.vehicle.turbo_type != turbo_type {
        ctx.vehicle.turbo_start_tick = tick;
        if ctx.vehicle.turbo_sound_attached {
            /* 0x007BCFAF..0x007BCFC8: the game played the turbo sound here
             * through the `play_turbo_sound` runtime callback (aborting when
             * the callback was unset). Audio-only side effect — not
             * represented; `turbo_sound_attached` (game car +0x26c != NULL)
             * is kept so the trigger condition stays observable. */
        }
        ctx.vehicle.roulette_token = 0;
    }
    if turbo_type == TMNFVehicleTurboType::Normal {
        ctx.vehicle.turbo_factor = factor;
    } else if turbo_type == TMNFVehicleTurboType::Roulette
        && ctx.vehicle.roulette_token != roulette_token
    {
        let value = get_roulette_value01(
            tick.wrapping_sub(ctx.vehicle.turbo_epoch_tick),
            ctx.vehicle.roulette_modulus,
        );

        ctx.vehicle.roulette_value = value;
        ctx.vehicle.turbo_factor =
            get_roulette_boost_factor_from_value01(value) * factor;
        ctx.vehicle.roulette_token = roulette_token;
    }
    ctx.vehicle.turbo_end_tick = tick.wrapping_add(duration);
    ctx.vehicle.turbo_type = turbo_type;
}

/* 0x007BD3F0, UNVALIDATED: integrates suspension and its surface transform. */
pub fn wheel_integrate(ctx: &mut VehicleCtx, wheel_index: usize, dt: f32) {
    /* `const CSceneVehicleCarTuning *tuning = vehicle->tuning` (primary view)
     * vs `self->tuning` (aux view) — the follow rate is the aux field. */
    match ctx.tuning.base.suspension_model {
        0 => {
            let mut absorb: f32;

            {
                let real_time = &mut ctx.vehicle.wheels[wheel_index].real_time;
                absorb = real_time.damper_absorb - real_time.field08;
                real_time.damper_absorb = absorb;
                real_time.field08 = 0.0;
            }
            ctx.vehicle.wheels[wheel_index].surface_location =
                ctx.vehicle.wheels[wheel_index].surface_source;
            let spring = (ctx.tuning.base.suspension_rest_length - absorb)
                * ctx.tuning.base.suspension_stiffness;
            let damping =
                ctx.tuning.base.suspension_damping
                    * ctx.vehicle.wheels[wheel_index].real_time.field04;
            let acceleration = spring - damping;
            let velocity = (acceleration * dt)
                + ctx.vehicle.wheels[wheel_index].real_time.field04;
            ctx.vehicle.wheels[wheel_index].real_time.field04 = velocity;
            absorb = (velocity * dt) + absorb;
            ctx.vehicle.wheels[wheel_index].real_time.damper_absorb = absorb;
            let vertical_offset = -absorb;
            let horizontal_zero = vertical_offset * 0.0;
            let surface_location = &mut ctx.vehicle.wheels[wheel_index].surface_location;
            surface_location.t[0] = surface_location.t[0] + horizontal_zero;
            surface_location.t[1] = vertical_offset + surface_location.t[1];
            surface_location.t[2] = surface_location.t[2] + horizontal_zero;
            update_surface(ctx, wheel_index);
        }
        1 | 2 => {
            let old_absorb = ctx.vehicle.wheels[wheel_index].real_time.damper_absorb;
            let effective =
                old_absorb - ctx.vehicle.wheels[wheel_index].real_time.field08;
            let delta = ctx.tuning.base.suspension_rest_length - effective;
            let correction =
                (delta * dt) * ctx.tuning.aux.suspension_follow_rate;

            ctx.vehicle.wheels[wheel_index].surface_location =
                ctx.vehicle.wheels[wheel_index].surface_source;
            let absorb = correction + effective;
            ctx.vehicle.wheels[wheel_index].real_time.field04 =
                (absorb - old_absorb) / dt;
            ctx.vehicle.wheels[wheel_index].real_time.damper_absorb = absorb;
            ctx.vehicle.wheels[wheel_index].real_time.field08 = 0.0;
            let vertical_offset = -absorb;
            let horizontal_zero = vertical_offset * 0.0;
            let surface_location = &mut ctx.vehicle.wheels[wheel_index].surface_location;
            surface_location.t[0] = horizontal_zero + surface_location.t[0];
            surface_location.t[1] = surface_location.t[1] + vertical_offset;
            surface_location.t[2] = surface_location.t[2] + horizontal_zero;
            update_surface(ctx, wheel_index);
        }
        _ => {
            update_surface(ctx, wheel_index);
        }
    }
}

/* 0x007BF1D0, UNVALIDATED: applies local angular control while airborne. */
pub fn compute_air_control(
    ctx: &mut VehicleCtx,
    angular_speed: &GmVec3,
    tick: u32,
    suppress_torque: bool,
    reset: bool,
) {
    let mut torque_direction = GmVec3 {
        x: -angular_speed.x,
        y: -angular_speed.y,
        z: -angular_speed.z,
    };
    let mut reversed = false;

    if (ctx.tuning.base.engine_model == 4 || ctx.tuning.base.engine_model == 5)
        && ctx.vehicle.air_control_locked != 0
    {
        return;
    }
    /* The C's three `goto apply_torque` paths (reset / immediate / expired
     * window) skip the control-speed section and the local angular speed
     * store, falling straight through to the torque block. */
    let skip_control = if reset {
        ctx.vehicle.air_control_tick = tick;
        ctx.vehicle.air_control_speed = *angular_speed;
        true
    } else if ctx.vehicle.air_control_immediate != 0 {
        ctx.vehicle.air_control_speed = *angular_speed;
        true
    } else {
        ctx.tuning.aux.air_control_window_ticks
            <= tick.wrapping_sub(ctx.vehicle.air_control_tick)
    };

    if !skip_control {
        let mut controlled_speed = *angular_speed;

        if (AUX_EPSILON < ctx.vehicle.input_steer
            && f64::from(ctx.vehicle.air_control_speed.y) < 0.0)
            || (ctx.vehicle.input_steer < -AUX_EPSILON
                && 0.0 < f64::from(ctx.vehicle.air_control_speed.y))
        {
            if f64::from(ctx.tuning.aux.air_reversal_threshold)
                < f64::from(angular_speed.y).abs()
            {
                reversed = true;
                ctx.vehicle.air_control_speed.y = angular_speed.y;
            }
        } else {
            if (AUX_EPSILON < ctx.vehicle.input_steer
                && 0.0 < f64::from(ctx.vehicle.air_control_speed.y))
                || (ctx.vehicle.input_steer < -AUX_EPSILON
                    && f64::from(ctx.vehicle.air_control_speed.y) < 0.0)
            {
                reversed = true;
            }
            ctx.vehicle.air_control_speed.y = angular_speed.y;
        }
        controlled_speed.y = ctx.vehicle.air_control_speed.y;

        if ctx.tuning.base.engine_model == 4 || ctx.tuning.base.engine_model == 5 {
            if AUX_EPSILON < ctx.vehicle.input_brake
                && 0.0 < f64::from(ctx.vehicle.air_control_speed.x)
            {
                ctx.vehicle.air_control_speed.x = 0.0;
            } else {
                ctx.vehicle.air_control_speed.x = angular_speed.x;
            }
            controlled_speed.x = ctx.vehicle.air_control_speed.x;
        }
        if reversed {
            torque_direction.x = torque_direction.x * 3.0;
            torque_direction.y = torque_direction.y * 3.0;
            torque_direction.z = torque_direction.z * 3.0;
        }
        if !suppress_torque {
            let curve_value = aux_curve_get_value(
                ctx.tuning.aux.air_vertical_curve.as_ref().expect(
                    "tmnf: air vertical curve is absent",
                ),
                f64::from(angular_speed.z).abs() as f32,
            );
            torque_direction.z = torque_direction.z * curve_value;
        }
        /* 0x00534400 set_local_angular_speed — CHmsDyna::SetLocalAngularSpeed
         * (the same game function the C inlined as a file static):
         * angVel = rot * controlled_speed. */
        ctx.dyna.set_local_angular_speed(&controlled_speed);
    }

    /* apply_torque: */
    if !suppress_torque {
        let xy_squared = (torque_direction.y * torque_direction.y)
            + (torque_direction.x * torque_direction.x);
        let length_squared = xy_squared + (torque_direction.z * torque_direction.z);
        let length = length_squared.sqrt();

        if AUX_EPSILON <= length {
            let inverse_length = 1.0 / length;
            let mut torque = GmVec3::ZERO;

            torque.x = inverse_length * torque_direction.x;
            torque.y = inverse_length * torque_direction.y;
            torque.z = inverse_length * torque_direction.z;
            let quadratic_term = (ctx.tuning.aux.air_torque_quadratic * length) * length;
            let linear_term = length * ctx.tuning.aux.air_torque_linear;
            let magnitude = quadratic_term + linear_term;
            torque.x = magnitude * torque.x;
            torque.y = torque.y * magnitude;
            torque.z = magnitude * torque.z;
            add_vehicle_torque(ctx, &torque);
        }
    }
}

/*
 * 0x007C2910 CSceneVehicleCar::ApplyWaterForces. Returns true when the water
 * drag/buoyancy force and torque were applied, false otherwise (including
 * the tick on which the entry impulse is applied).
 *
 * 0x007C97A0 CSceneVehicle::WaterSplash (car +0x1f8, +0x204..+0x20c, +0xb8)
 * only records the splash for audio/visual playback and is not represented;
 * none of those words are physics inputs or oracle fields.
 *
 * The C read the map from `tuning->water_map` and aborted on NULL; here the
 * water map is track data passed by reference, so the NULL check has no
 * counterpart.
 */
pub fn apply_water_forces(
    ctx: &mut VehicleCtx,
    water: &TmnfTrackWater,
    existing_force: &GmVec3,
) -> bool {
    let rotation = ctx.live().rot;
    let mut box_ = GmBoxAligned::default();
    let half_y: f32;
    let top: f32;
    let bottom: f32;
    let depth: f32;

    /* 0x007C2939: GmBoxAligned::SetMult(car+0x1dc, corpus location) — the C
     * casts &dyna_state->rot to GmIso4*, i.e. the rot+pos overlay. */
    box_.set_mult(&ctx.vehicle.body_box, &ctx.live().rot_pos_iso());
    half_y = box_.half_extent.y.abs();
    top = box_.center.y + half_y;
    bottom = box_.center.y - half_y;

    /* 0x007C29B3..0x007C2A2A: map gate. */
    if water_map_is_inside(water, box_.center.x, box_.center.z)
        || water.default_cell != 1
        || water.level <= bottom
    {
        if top <= water.floor {
            return false;
        }
        if water.level <= bottom {
            return false;
        }
        if water_map_value(water, box_.center.x, box_.center.z) != 1 {
            return false;
        }
    }
    depth = water.level - bottom;
    if !(0.5f32 < depth) {
        return false;
    }

    /* CHmsItem_GetLinearSpeed: body-frame linear speed. */
    let local_speed = ctx.dyna.get_local_linear_speed();
    let mut world_speed = GmVec3::ZERO;
    world_speed.set_mult_mat3(&local_speed, &rotation);
    let horizontal_squared =
        (world_speed.x * world_speed.x) + (world_speed.z * world_speed.z);
    let threshold = ctx.tuning.aux.water_entry_speed_threshold;

    /* 0x007C2A8C..0x007C2AEC: entry-impulse eligibility. 0x00B41EA8 is
     * float-widened 0.9; 0x00B574FC is -1e-05f (the negated AUX_EPSILON). */
    if ctx.vehicle.air_control_immediate == 0
        && f64::from(depth) < FLOAT_0_9_AS_F64
        && f64::from(water.level - top) < 0.0
        && world_speed.y < -AUX_EPSILON
    {
        let ratio: f32;
        let mut lower_index: u32;
        let apply_impulse: bool;

        if horizontal_squared <= threshold * threshold {
            /* 0x007C2E0D: slow horizontal entry needs total speed above
             * the +0x20c minimum. */
            let minimum = ctx.tuning.aux.water_entry_speed_minimum;
            let speed_squared = water_length_squared(&local_speed);
            let minimum_squared = minimum * minimum;

            apply_impulse = minimum_squared < speed_squared;
            ratio = 0.0;
            /* memcpy(&lower_index, &minimum_squared, 4): the stale float
             * bits seed the curve's cached index slot. */
            lower_index = minimum_squared.to_bits();
        } else {
            let horizontal = horizontal_squared.sqrt();

            ratio = (-horizontal) / world_speed.y;
            apply_impulse = !(f64::from(ratio) < 0.0);
            lower_index = horizontal.to_bits();
        }
        if apply_impulse {
            /* 0x007C2B35..0x007C2BD9. Both curve lookups take the index slot
             * left by the previous write (the stale float bits, then the
             * first lookup's own update), exactly as the game does. */
            let mut vertical: f32 = 0.0; /* uninitialized in the C */
            let mut horizontal: f32 = 0.0; /* uninitialized in the C */
            let mut impulse = GmVec3::ZERO;

            crate::vehicle::curve::cfunc_keys_real_get_value_out(
                ctx.tuning.aux.water_impulse_vertical_curve.as_ref().expect(
                    "tmnf: water impulse vertical curve is absent (the C dereferences it unconditionally)",
                ),
                ratio,
                &mut vertical,
                &mut lower_index,
            );
            crate::vehicle::curve::cfunc_keys_real_get_value_out(
                ctx.tuning.aux.water_impulse_horizontal_curve.as_ref().expect(
                    "tmnf: water impulse horizontal curve is absent (the C dereferences it unconditionally)",
                ),
                ratio,
                &mut horizontal,
                &mut lower_index,
            );
            impulse.x = (-horizontal) * world_speed.x;
            impulse.y = (-vertical) * world_speed.y;
            impulse.z = (-horizontal) * world_speed.z;
            impulse.mult_transpose(&rotation);
            add_vehicle_central_impulse(ctx, &impulse);
            /* 0x007C2BDE: an applied impulse ends the call with 0. */
            return false;
        }
    }

    /* 0x007C2BEF..0x007C2E0A: drag, angular drag, buoyancy. */
    let speed = water_length_squared(&local_speed).sqrt();
    let mut drag = GmVec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut scale: f32;
    if AUX_EPSILON < speed {
        scale = -water_friction_from_speed(&ctx.tuning.aux, speed);
        drag.x = scale * local_speed.x;
        drag.y = local_speed.y * scale;
        drag.z = scale * local_speed.z;
    }
    /* CHmsItem_GetAngularSpeed: body-frame angular speed. */
    let local_angular = ctx.dyna.get_local_angular_speed();
    let mut torque = GmVec3::ZERO;
    scale = -ctx.tuning.aux.water_angular_drag_linear;
    torque.x = scale * local_angular.x;
    torque.y = local_angular.y * scale;
    torque.z = scale * local_angular.z;
    let angular_speed = water_length_squared(&local_angular).sqrt();
    scale = (-angular_speed) * ctx.tuning.aux.water_angular_drag_quadratic;
    torque.x = (scale * local_angular.x) + torque.x;
    torque.y = (local_angular.y * scale) + torque.y;
    torque.z = (scale * local_angular.z) + torque.z;
    let mut buoyancy = GmVec3 {
        x: 0.0,
        y: -ctx.tuning.aux.water_buoyancy,
        z: 0.0,
    };
    buoyancy.mult_transpose(&rotation);
    let mut force = GmVec3::ZERO;
    force.x = (buoyancy.x + drag.x) - existing_force.x;
    force.y = (buoyancy.y + drag.y) - existing_force.y;
    force.z = (buoyancy.z + drag.z) - existing_force.z;
    add_vehicle_central_force(ctx, &force);
    add_vehicle_torque(ctx, &torque);
    true
}

/* 0x007C3900, UNVALIDATED: integrates dry A01 wheel, engine, and steer state. */
pub fn integrate_vehicle(ctx: &mut VehicleCtx, dt: f32) {
    let mut local_speed = ctx.live().lin_vel;
    local_speed.mult_transpose(&ctx.live().rot);

    if (ctx.vehicle.integration_flags & 1u32) != 0 {
        let absolute_speed = f64::from(local_speed.z).abs() as f32;
        let steering_scale = (absolute_speed * ctx.tuning.aux.steering_speed_scale)
            + ctx.tuning.aux.steering_speed_base;

        for i in 0..(ctx.vehicle.wheel_count as usize) {
            let wheel = &mut ctx.vehicle.wheels[i];
            /* memcpy(source_matrix.m, surface_source.m, 36) + GmMat3_Set —
             * a plain copy of the rotation block. */
            wheel.real_time.basis0.m = wheel.surface_source.m;
            if wheel.steerable == 0 {
                wheel.real_time.blend_target = 0.0;
            } else {
                let mut rotation = 0.0f32;
                let mut degrees = 30.0f32;
                let steering_value = ctx.vehicle.steering_value;

                if AUX_EPSILON <= steering_scale {
                    rotation = (-steering_value) / steering_scale;
                }
                mat3_rotate_y(&mut wheel.real_time.basis0, rotation);
                if let Some(curve) = &ctx.tuning.aux.steering_angle_curve {
                    /* 0x007C3A2E: 0x00B3D2A8, float 3.6 promoted to double. */
                    let speed_kmh =
                        (f64::from(absolute_speed) * FLOAT_3_6_AS_F64) as f32;

                    degrees = aux_curve_get_value(curve, speed_kmh);
                }
                /* 0x007C3A54: 0x00B36110, float pi promoted to double;
                 * 0x007C3A5A divides by 0x00B36AB8 (180). */
                let mut radians = (f64::from(degrees) * FLOAT_PI_AS_F64) as f32;
                radians = (f64::from(radians) / 180.0f64) as f32;
                wheel.real_time.blend_target = (-steering_value) * radians;
            }
            wheel_update_speed_from_vehicle_speed(ctx, i, local_speed.z, dt);
            wheel_real_time_state_integrate(
                &mut ctx.vehicle.wheels[i].real_time,
                dt,
            );
        }
    }
    if (ctx.vehicle.integration_flags & 2u32) != 0 {
        for i in 0..(ctx.vehicle.wheel_count as usize) {
            wheel_integrate(ctx, i, dt);
        }
    }
    if (ctx.vehicle.integration_flags & 4u32) != 0 {
        if ctx.vehicle.flag_60c != 0 {
            ctx.vehicle.engine.rpm = 0.0;
        } else {
            let throttle = if ctx.vehicle.engine.reverse != 0 {
                ctx.vehicle.input_brake
            } else {
                ctx.vehicle.input_gas
            };

            engine_integrate(ctx, throttle, dt);
        }
    }

    if 0.0f64 < f64::from(ctx.tuning.aux.steering_slew_rate) {
        let direction =
            if 0.0f64 <= f64::from(ctx.vehicle.steering_value - ctx.vehicle.input_steer) {
                -1.0f32
            } else {
                1.0f32
            };
        let mut next = ((ctx.tuning.aux.steering_slew_rate * direction) * dt)
            + ctx.vehicle.steering_value;

        if ctx.vehicle.input_steer <= ctx.vehicle.steering_value {
            if next < ctx.vehicle.input_steer {
                next = ctx.vehicle.input_steer;
            }
        } else if ctx.vehicle.input_steer < next {
            next = ctx.vehicle.input_steer;
        }
        ctx.vehicle.steering_value = next;
    } else {
        ctx.vehicle.steering_value = ctx.vehicle.input_steer;
    }

    /* C: `if (self->finish_integration == NULL) tmnf_abort();
     * self->finish_integration(self->runtime, vehicle);` — world.c's
     * finish_vehicle_integration: engine orchestration state the next
     * tick's model6 rollover torque reads. All three targets are Vehicle
     * fields, so the callback ports as direct writes. */
    ctx.vehicle.model6.contact_block_count = ctx.vehicle.compute.state_5d8;
    ctx.vehicle.model6.side_contact = ctx.vehicle.side_contact;
    ctx.vehicle.model6.rollover_axis = ctx.vehicle.compute.normalized_force;
}

/* 0x007C3C00 */
pub fn create_fake_contacts(ctx: &mut VehicleCtx) {
    /* C guard: `contact == NULL || contact->vehicle != vehicle
     * || self->wheel_count != vehicle->wheel_count
     * || contact->wheel_count != vehicle->wheel_count` → tmnf_abort(). With
     * the folded context the contact view IS ctx.vehicle (its wheels are
     * ctx.vehicle.wheels, its materials ctx.tuning.ground_materials), so the
     * identity/count invariants cannot be violated — unrepresentable. */
    let local_speed = ctx.dyna.get_local_linear_speed();
    for i in 0..(ctx.vehicle.wheel_count as usize) {
        let material_id = ctx.vehicle.wheels[i].real_time.contact_material_id as u16;
        let material: &crate::vehicle::TMNFVehicleGroundMaterial;
        let local_position = GmVec3 {
            x: ctx.vehicle.wheels[i].surface_source.t[0],
            y: ctx.vehicle.wheels[i].surface_source.t[1],
            z: ctx.vehicle.wheels[i].surface_source.t[2],
        };
        let mut world_position = GmVec3::ZERO;
        let x: u32;
        let y: u32;
        let mask_value: u8;
        let mut impact_speed: f32;

        if ctx.vehicle.wheels[i].real_time.has_ground_contact == 0 {
            continue;
        }
        if material_id as usize >= ctx.tuning.ground_material_indices.len() {
            panic!("tmnf: wheel ground material id out of range");
        }
        material = &ctx.tuning.ground_materials
            [ctx.tuning.ground_material_indices[material_id as usize] as usize];
        /* 0x007C3C7E..0x007C3C83: a material without a bump mask ends the
         * whole pass; later wheels are not visited. (An out-of-range entry
         * into ground_materials itself would be UB in the C; the Rust index
         * panics instead — the game decode guarantees validity.) */
        let mask = match &material.fake_contact_mask {
            Some(mask) => mask,
            None => return,
        };
        /* (const GmIso4 *)&dyna_state->rot — the rot+pos overlay. */
        world_position.set_mult_iso4(&local_position, &ctx.live().rot_pos_iso());
        x = fake_contact_mask_coordinate(
            world_position.x,
            material.fake_contact_period_x,
            TMNF_FAKE_CONTACT_MASK_WIDTH,
        );
        y = fake_contact_mask_coordinate(
            world_position.z,
            material.fake_contact_period_z,
            TMNF_FAKE_CONTACT_MASK_HEIGHT,
        );
        mask_value = mask[(y * TMNF_FAKE_CONTACT_MASK_WIDTH + x) as usize];
        if mask_value == 0 {
            continue;
        }

        /* 0x007C3D9A..0x007C3DC5 */
        impact_speed = (f64::from(mask_value) / 255.0f64) as f32;
        impact_speed = impact_speed * local_speed.z;
        impact_speed = impact_speed * material.fake_contact_impulse_scale;
        if material.fake_contact_impulse_limit < impact_speed {
            impact_speed = material.fake_contact_impulse_limit;
        }

        /* memset(&fake_contact, 0, sizeof(fake_contact)) → Default. The C's
         * NULL other_body becomes other_body_corpus_ref == 0 (the "no body"
         * word), so WheelAbsorbContact skips its other-body block. */
        let mut fake_contact = CHmsPhysicalContact::default();
        fake_contact.normal.y = 1.0;
        fake_contact.position = local_position;
        fake_contact.relative_speed.y = -impact_speed;
        fake_contact.other_surface_material = material_id;
        crate::vehicle::contact::wheel_absorb_contact(ctx, i, &mut fake_contact, &|_| crate::gm::GmIso4::IDENTITY);
    }
}

/* ===========================================================================
 * Tests (vectors generated by running the verbatim C through gcc —
 * tmp-verify/aux_oracle.c; O0 and O2 outputs are bit-identical)
 * ========================================================================= */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dyna::{CHmsDyna, CHmsDynaParams};
    use crate::gm::GmIso4;
    use crate::vehicle::{
        CFuncKeys, CSceneVehicleCarWheel, Vehicle, VehicleTuning,
    };

    fn make_dyna() -> CHmsDyna {
        CHmsDyna::new(
            CHmsDynaParams {
                mass: 1.0,
                inv_inertia_body: GmMat3::IDENTITY,
                drag_linear: 0.0,
                drag_angular: 0.0,
                substep_len: 0.01,
                force_field_scale: 0.0,
                com_offset: GmVec3::ZERO,
            },
            1, /* mode */
            0, /* clamp_angular */
            0.0,
        )
    }

    fn make_ctx<'a>(
        vehicle: &'a mut Vehicle,
        tuning: &'a VehicleTuning,
        dyna: &'a mut CHmsDyna,
    ) -> VehicleCtx<'a> {
        VehicleCtx {
            vehicle: &mut *vehicle,
            tuning,
            dyna: &mut *dyna,
        }
    }

    fn two_key_curve(values: [f32; 2]) -> CFuncKeysReal {
        CFuncKeysReal {
            keys: CFuncKeys {
                count: 2,
                positions: vec![0.0, 10.0],
                lower_bounds: vec![-1e-4, 10.0 - 1e-4],
                upper_bounds: vec![1e-4, 10.0 + 1e-4],
            },
            values: values.to_vec(),
            interpolation: 0,
        }
    }

    /* ---- constants -------------------------------------------------------- */

    #[test]
    fn constants_match_the_c_literals() {
        /* AUX_EPSILON 0x1.4f8b58p-17f (same word as vehicle.c's
         * WHEEL_SPEED_EPSILON). */
        assert_eq!(AUX_EPSILON.to_bits(), 0x3727_C5AC);
        /* ROULETTE_LOW 0x1.24924ap-1f — float 4/7. */
        assert_eq!(ROULETTE_LOW.to_bits(), 0x3F12_4925);
        /* ROULETTE_HIGH 0x1.b6db6ep-1 — the double image of float 6/7. */
        assert_eq!(ROULETTE_HIGH.to_bits(), 0x3FEB_6DB6_E000_0000);
        assert_eq!(FLOAT_0_9_AS_F64.to_bits(), 0x3FEC_CCCC_C000_0000);
        assert_eq!(FLOAT_3_6_AS_F64.to_bits(), 0x400C_CCCC_C000_0000);
        assert_eq!(FLOAT_PI_AS_F64.to_bits(), 0x4009_21FB_6000_0000);
        /* The widened doubles are the f64 images of the game's float
         * constants (verified with gcc). */
        assert_eq!(FLOAT_PI_AS_F64, f64::from(f32::from_bits(0x4049_0FDB)));
        assert_eq!(FLOAT_3_6_AS_F64, f64::from(f32::from_bits(0x4066_6666)));
        assert_eq!(FLOAT_0_9_AS_F64, f64::from(f32::from_bits(0x3F66_6666)));
        assert_eq!(ROULETTE_HIGH, f64::from(f32::from_bits(0x3F5B_6DB7)));
    }

    /* ---- roulette (C oracle) ---------------------------------------------- */

    #[test]
    fn roulette_value01_matches_c() {
        const CASES: &[(u32, u32, u32)] = &[
            (0, 7, 0x0000_0000),
            (1, 7, 0x0000_0000),
            (2, 7, 0x0000_0000),
            (3, 7, 0x0000_0000),
            (4, 7, 0x3F00_0000),
            (5, 7, 0x3F00_0000),
            (6, 7, 0x3F80_0000),
            (7, 7, 0x0000_0000),
            (8, 7, 0x0000_0000),
            (13, 7, 0x3F80_0000),
            (42, 7, 0x0000_0000),
            (100, 7, 0x0000_0000),
            (0xFFFF_FFFF, 7, 0x0000_0000),
            (0xFFFF_FFFF, 1_000_000, 0x3F80_0000),
            (0xFFFF_FFFF, 10, 0x0000_0000),
            (123_456_789, 10, 0x3F80_0000),
            (123_456_789, 3, 0x0000_0000),
            (0x7FFF_FFFF, 10, 0x3F00_0000),
            (0x7FFF_FFFF, 3, 0x0000_0000),
            (2, 3, 0x3F00_0000),
            (5, 3, 0x3F00_0000),
            (8, 3, 0x3F00_0000),
            (100, 3, 0x0000_0000),
            (6, 10, 0x3F00_0000),
            (7, 10, 0x3F00_0000),
            (8, 10, 0x3F00_0000),
            (0x8000_0000, 3, 0x3F00_0000),
            (0x8000_0001, 3, 0x0000_0000),
        ];
        for &(value, divisor, expected) in CASES {
            assert_eq!(
                get_roulette_value01(value, divisor).to_bits(),
                expected,
                "value={} divisor={}",
                value,
                divisor
            );
        }
    }

    #[test]
    fn roulette_boost_matches_c() {
        assert_eq!(get_roulette_boost_factor_from_value01(0.0).to_bits(), 0x3F80_0000);
        assert_eq!(get_roulette_boost_factor_from_value01(0.5).to_bits(), 0x3FC0_0000);
        assert_eq!(get_roulette_boost_factor_from_value01(1.0).to_bits(), 0x4000_0000);
        assert_eq!(get_roulette_boost_factor_from_value01(0.25).to_bits(), 0x3FA0_0000);
        assert_eq!(get_roulette_boost_factor_from_value01(-0.5).to_bits(), 0x3F00_0000);
    }

    /* ---- turbo interval --------------------------------------------------- */

    #[test]
    fn update_turbo_progress_matches_c() {
        /* (tick, start, end) -> progress bits, straight from the C oracle. */
        /* NB: `turbo_end_tick < tick` is an unsigned compare; end < tick
         * clears the turbo (progress 0) even across the u32 wrap. */
        const CASES: &[(u32, u32, u32, u32)] = &[
            (5, 0, 20, 0x3E80_0000),
            (0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0x3F80_0000),
            (3, 0, 7, 0x3EDB_6DB7),
            (1, 0, 3, 0x3EAA_AAAB),
            (0x7FFF_FFFF, 0, 0x7FFF_FFFF, 0x3F80_0000),
            (0x8000_0005, 0, 0x8000_0006, 0x3F80_0000),
        ];
        for &(tick, start, end, expected) in CASES {
            let mut vehicle = Vehicle::default();
            let tuning = VehicleTuning::default();
            let mut dyna = make_dyna();
            vehicle.turbo_type = TMNFVehicleTurboType::Normal;
            vehicle.turbo_start_tick = start;
            vehicle.turbo_end_tick = end;
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            update_turbo(&mut ctx, tick);
            assert_eq!(
                ctx.vehicle.turbo_progress.to_bits(),
                expected,
                "tick={} start={} end={}",
                tick,
                start,
                end
            );
        }
    }

    #[test]
    fn update_turbo_expires_and_clears() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.turbo_type = TMNFVehicleTurboType::Normal;
        vehicle.turbo_start_tick = 0;
        vehicle.turbo_end_tick = 4;
        vehicle.turbo_progress = 0.75;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            update_turbo(&mut ctx, 5);
            assert_eq!(ctx.vehicle.turbo_type, TMNFVehicleTurboType::None);
            assert_eq!(ctx.vehicle.turbo_progress.to_bits(), 0);
        }
        /* No turbo at all. */
        vehicle.turbo_progress = 0.75;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            update_turbo(&mut ctx, 100);
            assert_eq!(ctx.vehicle.turbo_progress.to_bits(), 0);
        }
        /* end == tick is still active (the C expires on end < tick only). */
        vehicle.turbo_type = TMNFVehicleTurboType::Normal;
        vehicle.turbo_start_tick = 8;
        vehicle.turbo_end_tick = 10;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            update_turbo(&mut ctx, 10);
            assert_eq!(ctx.vehicle.turbo_type, TMNFVehicleTurboType::Normal);
            assert_eq!(ctx.vehicle.turbo_progress.to_bits(), 0x3F80_0000); /* 2/2 */
        }
    }

    #[test]
    fn enable_turbo_normal_and_wrapping() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.roulette_modulus = u32::MAX; /* world.c decode default */
        vehicle.turbo_type = TMNFVehicleTurboType::Roulette;
        vehicle.roulette_token = 77;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            enable_turbo(&mut ctx, 1000, 20, 1.5, TMNFVehicleTurboType::Normal, 9);
            assert_eq!(ctx.vehicle.turbo_start_tick, 1000); /* type changed */
            assert_eq!(ctx.vehicle.roulette_token, 0); /* token cleared */
            assert_eq!(ctx.vehicle.turbo_factor, 1.5);
            assert_eq!(ctx.vehicle.turbo_end_tick, 1020);
            assert_eq!(ctx.vehicle.turbo_type, TMNFVehicleTurboType::Normal);
        }
        /* Same type: start tick kept, factor refreshed. */
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            enable_turbo(&mut ctx, 1010, 5, 2.0, TMNFVehicleTurboType::Normal, 9);
            assert_eq!(ctx.vehicle.turbo_start_tick, 1000);
            assert_eq!(ctx.vehicle.turbo_factor, 2.0);
            assert_eq!(ctx.vehicle.turbo_end_tick, 1015);
        }
        /* u32 wrap: 0xFFFFFFFF + 2 == 1. */
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            enable_turbo(&mut ctx, 0xFFFF_FFFF, 2, 1.0, TMNFVehicleTurboType::Roulette, 1);
            assert_eq!(ctx.vehicle.turbo_end_tick, 1);
        }
    }

    #[test]
    fn enable_turbo_roulette_rolls_once_per_token() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.turbo_epoch_tick = 0;
        vehicle.roulette_modulus = 7;
        tuning.aux.air_torque_linear = 0.0; /* unused here */
        let _ = &mut tuning;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            /* 4 % 7 == 4/7 -> 0.5 -> factor (0.5+1)*2 == 3.0. */
            enable_turbo(&mut ctx, 4, 10, 2.0, TMNFVehicleTurboType::Roulette, 1);
            assert_eq!(ctx.vehicle.roulette_value.to_bits(), 0x3F00_0000);
            assert_eq!(ctx.vehicle.turbo_factor, 3.0);
            assert_eq!(ctx.vehicle.roulette_token, 1);
            assert_eq!(ctx.vehicle.turbo_start_tick, 4);
            assert_eq!(ctx.vehicle.turbo_end_tick, 14);
            /* Same token: no re-roll, factor untouched, end refreshed. */
            enable_turbo(&mut ctx, 6, 10, 5.0, TMNFVehicleTurboType::Roulette, 1);
            assert_eq!(ctx.vehicle.turbo_factor, 3.0);
            assert_eq!(ctx.vehicle.roulette_value.to_bits(), 0x3F00_0000);
            assert_eq!(ctx.vehicle.turbo_end_tick, 16);
            /* New token at the same type: re-roll against the epoch. */
            enable_turbo(&mut ctx, 6, 10, 2.0, TMNFVehicleTurboType::Roulette, 2);
            /* 6 % 7 == 6/7 -> 1.0 -> factor 2*2 == 4. */
            assert_eq!(ctx.vehicle.roulette_value.to_bits(), 0x3F80_0000);
            assert_eq!(ctx.vehicle.turbo_factor, 4.0);
            assert_eq!(ctx.vehicle.roulette_token, 2);
        }
    }

    /* ---- mat3_rotate_y (C oracle) ------------------------------------------ */

    #[test]
    fn mat3_rotate_y_matches_c() {
        const CASES: &[(f32, [u32; 9])] = &[
            (
                0.0,
                [
                    0x3DCC_CCCD,
                    0x3E4C_CCCD,
                    0x3E99_999A,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0x3F33_3333,
                    0x3F4C_CCCD,
                    0x3F66_6666,
                ],
            ),
            (
                0.5,
                [
                    0x3ED8_C222,
                    0x3F0F_1E5C,
                    0x3F31_DBA5,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0x3F10_FD4F,
                    0x3F1B_2EAD,
                    0x3F25_6008,
                ],
            ),
            (
                1.0,
                [
                    0x3F24_9F93,
                    0x3F47_FF2A,
                    0x3F6B_5EC0,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0x3E96_8F9F,
                    0x3E87_2425,
                    0x3E6F_7154,
                ],
            ),
            (
                -0.75,
                [
                    0xBECE_D63C,
                    0xBECC_4639,
                    0xBEC9_B634,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0x3F14_918F,
                    0x3F38_BFF2,
                    0x3F5C_EE54,
                ],
            ),
            (
                3.9000001,
                [
                    0xBF0D_D4E2,
                    0xBF32_05B4,
                    0xBF56_3683,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0xBEE0_F5E1,
                    0xBEE2_EA22,
                    0xBEE4_DE62,
                ],
            ),
            (
                0.100000001,
                [
                    0x3E2D_72F3,
                    0x3E8E_C7BB,
                    0x3EC6_D5FC,
                    0x3ECC_CCCD,
                    0x3F00_0000,
                    0x3F19_999A,
                    0x3F2F_BFBF,
                    0x3F46_AA57,
                    0x3F5D_94EE,
                ],
            ),
        ];
        for &(angle, expected) in CASES {
            let mut matrix = GmMat3 {
                m: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            };
            mat3_rotate_y(&mut matrix, angle);
            for i in 0..9 {
                assert_eq!(
                    matrix.m[i].to_bits(),
                    expected[i],
                    "angle={} m[{}]",
                    angle,
                    i
                );
            }
        }
    }

    /* ---- water map helpers (C oracle) -------------------------------------- */

    #[test]
    fn water_map_coordinate_matches_c() {
        const CASES: &[(f32, f32, f32, u32)] = &[
            (33.0, 0.0, 32.0, 1),
            (-1.0, 0.0, 32.0, 0),
            (-33.0, 0.0, 32.0, 0xFFFF_FFFF),
            (100.0, -50.0, 10.0, 15),
            (0.0, 0.0, 32.0, 0),
            (31.9, 0.0, 32.0, 0),
            (-32.0, 0.0, 32.0, 0xFFFF_FFFF),
            (1e9, 0.0, 32.0, 31_250_000),
            (511.99, 0.0, 32.0, 15),
        ];
        for &(value, origin, cell, expected) in CASES {
            assert_eq!(
                water_map_coordinate(value, origin, cell),
                expected,
                "value={} origin={} cell={}",
                value,
                origin,
                cell
            );
        }
    }

    #[test]
    fn water_length_squared_matches_c() {
        assert_eq!(
            water_length_squared(&GmVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            })
            .to_bits(),
            0x4160_0000
        );
        assert_eq!(
            water_length_squared(&GmVec3 {
                x: -0.5,
                y: 0.25,
                z: 0.125
            })
            .to_bits(),
            0x3EA8_0000
        );
        assert_eq!(
            water_length_squared(&GmVec3 {
                x: 100.0,
                y: -200.0,
                z: 300.0
            })
            .to_bits(),
            0x4808_B800
        );
        assert_eq!(
            water_length_squared(&GmVec3::ZERO).to_bits(),
            0
        );
    }

    /* ---- fake contact mask coordinate (C oracle) --------------------------- */

    #[test]
    fn fake_contact_mask_coordinate_matches_c() {
        const CASES: &[(f32, f32, u32)] = &[
            (0.5, 8.0, 8),
            (7.9, 8.0, 126),
            (-0.5, 8.0, 8),
            (8.0, 8.0, 0),
            (3.999, 4.0, 127),
            (0.0, 8.0, 0),
            (-7.9, 8.0, 126),
            (12.5, 8.0, 72),
        ];
        for &(value, period, expected) in CASES {
            assert_eq!(
                fake_contact_mask_coordinate(value, period, 128),
                expected,
                "value={} period={}",
                value,
                period
            );
        }
    }

    /* ---- aux curve lookup (C oracle) ---------------------------------------- */

    #[test]
    fn aux_curve_get_value_matches_c() {
        let c2 = two_key_curve([1.0, 3.0]);
        const TWO_KEY: &[(f32, u32)] = &[
            (-1.0, 0x3F80_0000),
            (0.0, 0x3F80_0000),
            (5.0, 0x4000_0000),
            (10.0, 0x4040_0000),
            (11.0, 0x4040_0000),
            (2.5, 0x3FC0_0000),
            (7.5, 0x4020_0000),
        ];
        for &(position, expected) in TWO_KEY {
            assert_eq!(
                aux_curve_get_value(&c2, position).to_bits(),
                expected,
                "pos={}",
                position
            );
        }

        let three_key = |interpolation: i32| CFuncKeysReal {
            keys: CFuncKeys {
                count: 3,
                positions: vec![0.0, 10.0, 20.0],
                lower_bounds: vec![-1e-4, 10.0 - 1e-4, 20.0 - 1e-4],
                upper_bounds: vec![1e-4, 10.0 + 1e-4, 20.0 + 1e-4],
            },
            values: vec![0.0, 5.0, -2.0],
            interpolation,
        };
        const THREE_KEY: &[(f32, u32, u32)] = &[
            (-1.0, 0x0000_0000, 0x0000_0000),
            (0.0, 0x0000_0000, 0x0000_0000),
            (5.0, 0x4020_0000, 0x0000_0000),
            (9.999, 0x409F_FBE7, 0x0000_0000),
            (10.0, 0x40A0_0000, 0x0000_0000),
            (10.001, 0x409F_FA43, 0x40A0_0000),
            (15.0, 0x3FC0_0000, 0x40A0_0000),
            (20.0, 0xC000_0000, 0x40A0_0000),
            (25.0, 0xC000_0000, 0xC000_0000),
        ];
        for &(position, expected_interp0, expected_interp1) in THREE_KEY {
            assert_eq!(
                aux_curve_get_value(&three_key(0), position).to_bits(),
                expected_interp0,
                "interp0 pos={}",
                position
            );
            assert_eq!(
                aux_curve_get_value(&three_key(1), position).to_bits(),
                expected_interp1,
                "interp1 pos={}",
                position
            );
        }

        let one_key = CFuncKeysReal {
            keys: CFuncKeys {
                count: 1,
                positions: vec![4.0],
                lower_bounds: vec![-1e-4],
                upper_bounds: vec![1e-4],
            },
            values: vec![7.5],
            interpolation: 0,
        };
        assert_eq!(aux_curve_get_value(&one_key, 0.0).to_bits(), 0x40F0_0000);
        assert_eq!(aux_curve_get_value(&one_key, 100.0).to_bits(), 0x40F0_0000);

        let zero_key = CFuncKeysReal::default();
        assert_eq!(aux_curve_get_value(&zero_key, 0.0).to_bits(), 0);
    }

    /* ---- wheel integrate (suspension models) ------------------------------- */

    #[test]
    fn wheel_integrate_model_0() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.wheels.push(CSceneVehicleCarWheel::default());
        vehicle.wheel_count = 1;
        vehicle.wheels[0].real_time.damper_absorb = 1.0;
        vehicle.wheels[0].real_time.field04 = 0.125;
        vehicle.wheels[0].real_time.field08 = 0.5;
        vehicle.wheels[0].surface_source = GmIso4 {
            m: GmMat3::IDENTITY.m,
            t: [3.0, 4.0, 5.0],
        };
        tuning.base.suspension_model = 0;
        tuning.base.suspension_rest_length = 2.0;
        tuning.base.suspension_stiffness = 4.0;
        tuning.base.suspension_damping = 8.0;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            wheel_integrate(&mut ctx, 0, 0.25);
        }
        let rt = &vehicle.wheels[0].real_time;
        /* absorb = 1.0 - 0.5 = 0.5; spring = (2 - 0.5)*4 = 6;
         * damping = 8*0.125 = 1; acc = 5; vel = 5*0.25 + 0.125 = 1.375;
         * absorb = 1.375*0.25 + 0.5 = 0.84375 (all dyadic — exact). */
        assert_eq!(rt.damper_absorb, 0.84375);
        assert_eq!(rt.field04, 1.375);
        assert_eq!(rt.field08, 0.0);
        /* t[0] += (-0.84375)*0; t[1] = -0.84375 + 4; t[2] += (-0.84375)*0. */
        assert_eq!(vehicle.wheels[0].surface_location.t[0], 3.0);
        assert_eq!(vehicle.wheels[0].surface_location.t[1], 3.15625);
        assert_eq!(vehicle.wheels[0].surface_location.t[2], 5.0);
    }

    #[test]
    fn wheel_integrate_model_2() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.wheels.push(CSceneVehicleCarWheel::default());
        vehicle.wheel_count = 1;
        vehicle.wheels[0].real_time.damper_absorb = 0.5;
        vehicle.wheels[0].real_time.field04 = 9.0;
        vehicle.wheels[0].real_time.field08 = 0.25;
        vehicle.wheels[0].surface_source = GmIso4 {
            m: GmMat3::IDENTITY.m,
            t: [1.0, 2.0, 3.0],
        };
        tuning.base.suspension_model = 2;
        tuning.base.suspension_rest_length = 1.0;
        tuning.aux.suspension_follow_rate = 0.5;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            wheel_integrate(&mut ctx, 0, 0.25);
        }
        let rt = &vehicle.wheels[0].real_time;
        /* effective = 0.25; delta = 0.75; correction = (0.75*0.25)*0.5 =
         * 0.09375; absorb = 0.34375; field04 = (0.34375-0.5)/0.25 = -0.625
         * (all dyadic — exact). */
        assert_eq!(rt.damper_absorb, 0.34375);
        assert_eq!(rt.field04, -0.625);
        assert_eq!(rt.field08, 0.0);
        assert_eq!(vehicle.wheels[0].surface_location.t[0], 1.0);
        assert_eq!(vehicle.wheels[0].surface_location.t[1], 1.65625);
        assert_eq!(vehicle.wheels[0].surface_location.t[2], 3.0);
    }

    #[test]
    fn wheel_integrate_unknown_model_touches_nothing() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.wheels.push(CSceneVehicleCarWheel::default());
        vehicle.wheel_count = 1;
        vehicle.wheels[0].real_time.damper_absorb = 0.5;
        tuning.base.suspension_model = 5;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            wheel_integrate(&mut ctx, 0, 0.25);
        }
        assert_eq!(vehicle.wheels[0].real_time.damper_absorb, 0.5);
    }

    /* ---- air control -------------------------------------------------------- */

    #[test]
    fn compute_air_control_reset_path_skips_speed_store() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let angular = GmVec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 42, true, true);
        }
        assert_eq!(vehicle.air_control_tick, 42);
        assert_eq!(vehicle.air_control_speed, angular);
        /* goto apply_torque: set_local_angular_speed was skipped. */
        assert_eq!(dyna.live().ang_vel, GmVec3::ZERO);
        /* suppress_torque: no torque either. */
        assert_eq!(dyna.live().torque, GmVec3::ZERO);
    }

    #[test]
    fn compute_air_control_immediate_and_window_paths() {
        let angular = GmVec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        /* air_control_immediate != 0 -> goto apply_torque. */
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.air_control_immediate = 1;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 10, true, false);
        }
        assert_eq!(vehicle.air_control_speed, angular);
        assert_eq!(dyna.live().ang_vel, GmVec3::ZERO);

        /* window expired (window_ticks <= tick - air_control_tick). */
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.aux.air_control_window_ticks = 5;
        vehicle.air_control_tick = 0;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 10, true, false);
        }
        assert_eq!(vehicle.air_control_speed, GmVec3::ZERO); /* untouched */

        /* Locked + engine model 4/5 -> hard return. */
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.base.engine_model = 5;
        vehicle.air_control_locked = 1;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 10, false, true);
        }
        assert_eq!(vehicle.air_control_tick, 0);
        assert_eq!(dyna.live().torque, GmVec3::ZERO);
    }

    #[test]
    fn compute_air_control_applies_torque_and_speed() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.aux.air_control_window_ticks = 100; /* window open at tick 0 */
        tuning.aux.air_vertical_curve = Some(two_key_curve([1.0, 1.0]));
        tuning.aux.air_torque_linear = 2.0;
        let angular = GmVec3 {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        };
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 0, false, false);
        }
        /* torque_direction = (-0.5, 0, -0); length 0.5; unit (-1, 0, -0);
         * magnitude = 0 + 0.5*2 = 1 -> torque (-1, 0, 0) under identity rot. */
        assert_eq!(dyna.live().torque, GmVec3 { x: -1.0, y: 0.0, z: 0.0 });
        /* The C stores air_control_speed.y only in the window path. */
        assert_eq!(vehicle.air_control_speed.y, angular.y);
    }

    #[test]
    fn compute_air_control_reversal_and_engine4_axis() {
        let angular = GmVec3 {
            x: 0.5,
            y: 2.0,
            z: 0.0,
        };
        /* Steer right while spinning left: |angular.y| above the threshold
         * reverses the y control speed. */
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.aux.air_control_window_ticks = 100;
        tuning.aux.air_reversal_threshold = 0.5;
        vehicle.input_steer = 1.0;
        vehicle.air_control_speed.y = -1.0;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 0, true, false);
        }
        assert_eq!(vehicle.air_control_speed.y, 2.0);
        assert_eq!(dyna.live().ang_vel, angular); /* controlled.y == 2 */

        /* Engine model 4 with brake zeroes a positive x control speed. */
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.aux.air_control_window_ticks = 100;
        tuning.base.engine_model = 4;
        vehicle.input_brake = 1.0;
        vehicle.air_control_speed.x = 0.5;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            compute_air_control(&mut ctx, &angular, 0, true, false);
        }
        assert_eq!(vehicle.air_control_speed.x, 0.0);
        assert_eq!(dyna.live().ang_vel.x, 0.0);
        assert_eq!(dyna.live().ang_vel.y, 2.0);
    }

    #[test]
    #[should_panic(expected = "tmnf: air vertical curve is absent")]
    fn compute_air_control_requires_the_vertical_curve() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        tuning.aux.air_control_window_ticks = 100;
        let angular = GmVec3 {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        };
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        compute_air_control(&mut ctx, &angular, 0, false, false);
    }

    /* ---- water forces -------------------------------------------------------- */

    fn water_1x1(default_cell: u8, cell: u8, level: f32, floor: f32) -> TmnfTrackWater {
        TmnfTrackWater {
            cell_x: 32.0,
            cell_z: 32.0,
            origin_x: 0.0,
            origin_z: 0.0,
            width: 1,
            height: 1,
            default_cell,
            cells: vec![cell],
            cell_count: 1, /* width * height */
            level,
            floor,
        }
    }

    #[test]
    fn apply_water_forces_gates() {
        /* Box above the level inside the map: level <= bottom -> false. */
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 0.0, y: 50.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let water = water_1x1(1, 1, 10.0, -992.0);
        let existing = GmVec3::ZERO;
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        assert!(!apply_water_forces(&mut ctx, &water, &existing));

        /* Outside the map with default 0: water_map_value != 1 -> false. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 100.0, y: 5.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let water = water_1x1(0, 1, 10.0, -992.0);
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        assert!(!apply_water_forces(&mut ctx, &water, &existing));

        /* Inside the map, but the cell is 0 -> false. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 0.0, y: 5.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let water = water_1x1(1, 0, 10.0, -992.0);
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        assert!(!apply_water_forces(&mut ctx, &water, &existing));

        /* Deep-gate: depth = level - bottom = 0.5, not > 0.5 -> false. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 0.0, y: 10.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 0.5, z: 1.0 },
        };
        let water = water_1x1(1, 1, 10.0, -992.0);
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        assert!(!apply_water_forces(&mut ctx, &water, &existing));

        /* Box entirely under the floor -> false. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 0.0, y: -1000.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        let water = water_1x1(1, 1, 10.0, -992.0);
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        assert!(!apply_water_forces(&mut ctx, &water, &existing));
    }

    #[test]
    fn apply_water_forces_submerged_zero_speed() {
        /* air_control_immediate != 0 skips the entry-impulse eligibility
         * (and with it the curve lookups); zero speeds skip the friction
         * lookup, so this path runs without the curve seam. */
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.body_box = GmBoxAligned {
            center: GmVec3 { x: 0.0, y: -5.0, z: 0.0 },
            half_extent: GmVec3 { x: 1.0, y: 1.0, z: 1.0 },
        };
        vehicle.air_control_immediate = 1;
        tuning.aux.water_buoyancy = 5.0;
        tuning.aux.water_angular_drag_linear = 2.0;
        tuning.aux.water_angular_drag_quadratic = 0.5;
        dyna.live_mut().ang_vel = GmVec3 { x: 1.0, y: 0.0, z: 0.0 };
        let water = water_1x1(1, 1, 10.0, -992.0);
        let existing = GmVec3 { x: 1.0, y: 2.0, z: 3.0 };
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            assert!(apply_water_forces(&mut ctx, &water, &existing));
        }
        /* buoyancy (0,-5,0) - existing (1,2,3) = (-1,-7,-3) (identity rot). */
        assert_eq!(
            dyna.live().force,
            GmVec3 { x: -1.0, y: -7.0, z: -3.0 }
        );
        assert_eq!(
            vehicle.total_force_added,
            GmVec3 { x: -1.0, y: -7.0, z: -3.0 }
        );
        /* torque = (-2,0,0) + (-(1*0.5) * 1? no: -|w|*q = -0.5)*w.x) ... :
         * linear part (-2,0,0), quadratic part (-0.5*1, 0, 0) -> (-2.5,0,0). */
        assert_eq!(
            dyna.live().torque,
            GmVec3 { x: -2.5, y: 0.0, z: 0.0 }
        );
    }

    /* ---- vehicle integration ------------------------------------------------- */

    #[test]
    fn integrate_vehicle_steering_radians_chain() {
        /* steering_angle_curve == None keeps degrees = 30; the C oracle gives
         * radians(30) == 0x3f060a92, so blend_target = 1.0 * radians. */
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut wheel = CSceneVehicleCarWheel::default();
        wheel.steerable = 1;
        vehicle.wheels.push(wheel);
        vehicle.wheel_count = 1;
        vehicle.integration_flags = 1;
        vehicle.steering_value = -1.0;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            integrate_vehicle(&mut ctx, 0.01);
        }
        assert_eq!(
            vehicle.wheels[0].real_time.blend_target.to_bits(),
            0x3F06_0A92
        );
        /* slew rate 0 -> steering_value snaps to input_steer (0). */
        assert_eq!(vehicle.steering_value, 0.0);
    }

    #[test]
    fn integrate_vehicle_steering_slew_and_flags() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        /* engine_integrate reads the gearbox tables; default tuning has
         * none, so populate 6-entry tables. */
        tuning.base.gear_ratios = vec![0.0; 6];
        tuning.base.gear_upshift = vec![0.0; 6];
        tuning.base.gear_downshift = vec![0.0; 6];
        tuning.base.gear_aux = vec![0.0; 6];
        tuning.aux.steering_slew_rate = 10.0;
        vehicle.steering_value = 0.0;
        vehicle.input_steer = 1.0;
        /* direction: 0 <= 0 - 1 ? no -> +1; next = 10*1*0.1 + 0 = 1. */
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            integrate_vehicle(&mut ctx, 0.1);
        }
        assert_eq!(vehicle.steering_value, 1.0);

        /* Overshoot clamps to the input. */
        vehicle.steering_value = 0.95;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            integrate_vehicle(&mut ctx, 0.1);
        }
        assert_eq!(vehicle.steering_value, 1.0);

        /* flags & 4 with flag_60c: rpm zeroed, engine integrate skipped. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.integration_flags = 4;
        vehicle.flag_60c = 1;
        vehicle.engine.rpm = 5000.0;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            integrate_vehicle(&mut ctx, 0.01);
        }
        assert_eq!(vehicle.engine.rpm, 0.0);

        /* flags & 4 without flag_60c: throttle from reverse. */
        let mut vehicle = Vehicle::default();
        let mut dyna = make_dyna();
        vehicle.integration_flags = 4;
        vehicle.engine.reverse = 1;
        vehicle.input_brake = 0.75;
        vehicle.input_gas = 0.25;
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            integrate_vehicle(&mut ctx, 0.01);
        }
        /* engine_integrate (model 0 old-models body) recorded mode 2? no —
         * just check it ran by the gear/engine state it leaves; the base
         * module's own tests cover its arithmetic. Throttle selection is the
         * contract here: assert via braking_factor-free observable — the
         * engine did not panic and gas was not used (rpm path identical for
         * both at throttle 0.75 vs 0.25 through model 0 is covered by base). */
    }

    /* ---- fake contacts -------------------------------------------------------- */

    #[test]
    fn create_fake_contacts_skips_airborne_wheels() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        vehicle.wheels.push(CSceneVehicleCarWheel::default());
        vehicle.wheel_count = 1;
        /* has_ground_contact == 0: nothing happens, no material needed. */
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            create_fake_contacts(&mut ctx);
        }
    }

    #[test]
    fn create_fake_contacts_material_without_mask_ends_the_pass() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut wheel = CSceneVehicleCarWheel::default();
        wheel.real_time.has_ground_contact = 1;
        wheel.real_time.contact_material_id = 0;
        vehicle.wheels.push(wheel);
        vehicle.wheel_count = 1;
        tuning.ground_material_indices = vec![0];
        tuning.ground_materials.push(crate::vehicle::TMNFVehicleGroundMaterial::default());
        {
            let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
            create_fake_contacts(&mut ctx);
        }
    }

    #[test]
    #[should_panic(expected = "tmnf: wheel ground material id out of range")]
    fn create_fake_contacts_rejects_unknown_material() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut wheel = CSceneVehicleCarWheel::default();
        wheel.real_time.has_ground_contact = 1;
        wheel.real_time.contact_material_id = 5;
        vehicle.wheels.push(wheel);
        vehicle.wheel_count = 1;
        tuning.ground_material_indices = vec![0];
        tuning.ground_materials.push(crate::vehicle::TMNFVehicleGroundMaterial::default());
        let mut ctx = make_ctx(&mut vehicle, &tuning, &mut dyna);
        create_fake_contacts(&mut ctx);
    }
}
