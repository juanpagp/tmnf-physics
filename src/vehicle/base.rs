//! `vehicle.c` — the vehicle base layer, transliterated from
//! `/home/z/my-project/tmnf-physics/src/vehicle.c` (755 lines) plus the
//! declarations of `src/vehicle.h`.
//!
//! Contents (VA addresses kept above each function):
//! * input mapping — 0x004FE500 (`update_vehicle_state_from_inputs`) and the
//!   three `VehicleInput*Set` setters (0x007BFD80/90/A0);
//! * wheel real-time state integrate (0x007C1060) and wheel angular speed
//!   (0x007C0EC0);
//! * `GmSpringFloat::Integrate` (0x008F46C0);
//! * `SDynaMath_ComputeImpulse` (0x007BD090);
//! * the local→world force pipeline (0x007BE2C0/0x007BE310/0x007BE360);
//! * engine/gearbox integration (0x007BD700) with the Steer01..Steer05
//!   old-models body (0x007BE00B) — both live in `vehicle.c`, so both are
//!   ported here; `engine_model != 5` dispatches to the old-models body
//!   exactly as the C does (no unsupported-model abort exists in the C);
//! * wheel suspension force application (0x007C1810).
//!
//! # Floating-point policy (see `crate::fp` and PORT_NOTES.md)
//!
//! * `x87_mul/add/sub/div` → plain `f32` operators; `x87_sqrt` → `.sqrt()`;
//!   `x87_rcp(x)` → `1.0/x`. Grouping and operand order preserved verbatim
//!   (including the deliberately asymmetric operand orders in `add_vec3`,
//!   `normalize_if_nonzero` and `mat3_set_up_v_and_dov` — multiplication is
//!   bit-commutative, but the shapes are kept mechanically identical).
//! * Hex-float literals → `f32::from_bits` (Rust has no hex-float syntax);
//!   the original literal sits in the comment. Bits verified against gcc.
//! * **Bare double literals in comparisons** (`0.01 < fabs(F(x))`,
//!   `0.3 <= F(x)`, `0.1 < F(throttle)`, `F(clutch) < 1.15`,
//!   `F(rpm) < 1000.0`, `… 0.0`, `… 1.0`): C widens the float operand to
//!   double and compares against the double literal, so the port spells these
//!   as `f64` comparisons. For most of these literals the f32 comparison is
//!   provably equivalent, but the port does not rely on that (see the
//!   `engine_turbo_clutch_double_literal_comparison` test for the one corner
//!   — `clutch == 1.15f32`, `dt` NaN — where the two forms diverge).
//!   Comparisons `F(a) op F(b)` between two float lvalues stay plain `f32`.
//! * `x87_r24(F(a) * <double literal>)` single-op double products are spelled
//!   as `((a as f64) * lit) as f32` (bit-identical to the f32 op because the
//!   f64 product of two binary32 values is exact, but kept literal).
//! * `fmod` (double) → `crate::fp::x87_fmod`; `fabs(F(x))` → `f64::from(x).abs()`
//!   where the C compares in double, `f32::abs` where it narrows back.
//!
//! # Context mapping (see the table in `super`'s module docs)
//!
//! `CSceneVehicleCar *self` → `VehicleCtx`: car fields via `ctx.vehicle.*`,
//! the live dyna via `ctx.live()/ctx.live_mut()`, params via `ctx.params()`,
//! tuning via `ctx.tuning.base.*`. The C's `wheel` pointer argument becomes a
//! `wheel_index` into `ctx.vehicle.wheels`.

use crate::fp::x87_fmod;
use crate::gm::{GmMat3, GmVec3};
use crate::vehicle::{
    CSceneVehicleCarWheelRealTimeState, GmSpringFloat, TMNFRaceInputs, VehicleCtx,
};

/* ===========================================================================
 * Constants (vehicle.c top; Rust has no hex-float literals — bits verified
 * against gcc for the exact literal spellings)
 * ========================================================================= */

/* #define WHEEL_ANGLE_PERIOD 0x1.921fb6p+10f */
const WHEEL_ANGLE_PERIOD: f32 = f32::from_bits(0x44C9_0FDB); /* 1608.4954833984375 */
/* #define NORMALIZE_EPSILON 0x1.b7cdfcp-34f */
const NORMALIZE_EPSILON: f32 = f32::from_bits(0x2EDB_E6FE); /* 9.9999994396249292e-11 */
/* #define WHEEL_SPEED_EPSILON 0x1.4f8b58p-17f */
const WHEEL_SPEED_EPSILON: f32 = f32::from_bits(0x3727_C5AC); /* 9.9999997473787516e-06 */
/* #define WHEEL_SPEED_DAMPING 0x1.fd70a4p-1 */
const WHEEL_SPEED_DAMPING: f32 = f32::from_bits(0x3F7E_B852); /* 0.99500000476837158203 */

/* ===========================================================================
 * Static helpers (vehicle.c file-local)
 * ========================================================================= */

/* gm_mod: the file-local re-implementation of GmFunc_Mod. */
fn gm_mod(value: f32, lower: f32, upper: f32) -> f32 {
    if (lower < value) && (value < upper) {
        return value;
    }
    let period = upper - lower;
    let relative = value - lower;
    let mut result = x87_fmod(relative, period);
    if result < 0.0f32 {
        result = result + period;
    }
    result + lower
}

fn vec_length_squared(value: &GmVec3) -> f32 {
    ((value.x * value.x) + (value.y * value.y)) + (value.z * value.z)
}

fn normalize_if_nonzero(value: &mut GmVec3) {
    let length_squared = vec_length_squared(value);

    if NORMALIZE_EPSILON < length_squared {
        let length = length_squared.sqrt();
        let inverse_length = 1.0f32 / length;
        value.x = inverse_length * value.x;
        value.y = value.y * inverse_length;
        value.z = inverse_length * value.z;
    }
}

fn mat3_set_up_v_and_dov(matrix: &mut GmMat3, up_value: &GmVec3, dov: &GmVec3) {
    let mut line0 = GmVec3::ZERO;
    let mut line1 = *up_value;
    let mut line2 = GmVec3::ZERO;

    line0.x = (up_value.y * dov.z) - (up_value.z * dov.y);
    line0.y = (up_value.z * dov.x) - (up_value.x * dov.z);
    line0.z = (up_value.x * dov.y) - (dov.x * up_value.y);
    normalize_if_nonzero(&mut line0);
    normalize_if_nonzero(&mut line1);
    line2.x = (line0.y * line1.z) - (line0.z * line1.y);
    line2.y = (line1.x * line0.z) - (line0.x * line1.z);
    line2.z = (line0.x * line1.y) - (line0.y * line1.x);
    matrix.m[0] = line0.x;
    matrix.m[3] = line0.y;
    matrix.m[6] = line0.z;
    matrix.m[1] = line1.x;
    matrix.m[4] = line1.y;
    matrix.m[7] = line1.z;
    matrix.m[2] = line2.x;
    matrix.m[5] = line2.y;
    matrix.m[8] = line2.z;
}

fn add_vec3(sum: &mut GmVec3, value: &GmVec3) {
    sum.x = sum.x + value.x;
    sum.y = value.y + sum.y;
    sum.z = value.z + sum.z;
}

/* ===========================================================================
 * Input pipeline
 * ========================================================================= */

/* 0x007BFD80 */
pub fn vehicle_input_steer_set(ctx: &mut VehicleCtx, value: f32) {
    ctx.vehicle.input_steer = value;
}

/* 0x007BFD90 */
pub fn vehicle_input_gas_set(ctx: &mut VehicleCtx, value: f32) {
    ctx.vehicle.input_gas = value;
}

/* 0x007BFDA0 */
pub fn vehicle_input_brake_set(ctx: &mut VehicleCtx, value: f32) {
    ctx.vehicle.input_brake = value;
}

/* 0x004FE500  Maps timestamped digital/analog race inputs to car controls. */
pub fn update_vehicle_state_from_inputs(inputs: &TMNFRaceInputs, ctx: &mut VehicleCtx) {
    let mut latest: u32;
    let mut event_time: u32;
    let gas: f32;
    let brake: f32;
    let steer: f32;

    latest = inputs.accelerate_time;
    if latest < inputs.brake_time {
        latest = inputs.brake_time;
    }
    event_time = inputs.gas_analog_time;
    if event_time == latest
        && inputs.accelerate == 0
        && inputs.brake == 0
        && 0.01f64 < f64::from(inputs.gas_analog).abs()
    {
        event_time = latest.wrapping_add(1);
    }

    if event_time <= latest {
        gas = if inputs.accelerate != 0 { 1.0f32 } else { 0.0f32 };
        brake = if inputs.brake != 0 { 1.0f32 } else { 0.0f32 };
    } else if !inputs.gas_analog.is_nan() && 0.3f64 <= f64::from(inputs.gas_analog) {
        gas = 1.0f32;
        brake = 0.0f32;
    } else if !inputs.gas_analog.is_nan() && f64::from(inputs.gas_analog) <= -0.3f64 {
        gas = 0.0f32;
        brake = 1.0f32;
    } else {
        gas = 0.0f32;
        brake = 0.0f32;
    }
    vehicle_input_gas_set(ctx, gas);
    vehicle_input_brake_set(ctx, brake);

    latest = inputs.steer_left_time;
    if latest < inputs.steer_right_time {
        latest = inputs.steer_right_time;
    }
    event_time = inputs.steer_analog_time;
    if event_time == latest
        && inputs.steer_left == 0
        && inputs.steer_right == 0
        && 0.01f64 < f64::from(inputs.steer_analog).abs()
    {
        event_time = latest.wrapping_add(1);
    }

    if event_time <= latest {
        if inputs.steer_left != 0 {
            steer = -1.0f32;
        } else if inputs.steer_right != 0 {
            steer = 1.0f32;
        } else {
            steer = 0.0f32;
        }
    } else {
        /* steer = (float)-F(inputs->steer_analog): exact negation. */
        steer = -inputs.steer_analog;
    }
    vehicle_input_steer_set(ctx, steer);
}

/* ===========================================================================
 * Wheel real-time state
 * ========================================================================= */

/* 0x007C1060  Integrates one wheel's angular and contact blend state. */
pub fn wheel_real_time_state_integrate(state: &mut CSceneVehicleCarWheelRealTimeState, dt: f32) {
    let phase_input = (state.field6c * dt) + state.rotation_phase;
    let axis_length_squared: f32;

    state.rotation_phase = gm_mod(phase_input, 0.0f32, WHEEL_ANGLE_PERIOD);
    axis_length_squared = vec_length_squared(&state.field90);
    if NORMALIZE_EPSILON < axis_length_squared {
        let mut dov = GmVec3::ZERO;

        normalize_if_nonzero(&mut state.field90);
        dov.x = (state.field90.z * 0.0f32) - (state.field90.y * 0.0f32);
        dov.y = (state.field90.x * 0.0f32) - state.field90.z;
        dov.z = state.field90.y - (state.field90.x * 0.0f32);
        mat3_set_up_v_and_dov(&mut state.basis1, &state.field90, &dov);
    }
    if state.blend_target <= state.blend_value {
        state.blend_value = state.blend_value - dt;
        if state.blend_value < state.blend_target {
            state.blend_value = state.blend_target;
        }
    } else {
        state.blend_value = state.blend_value + dt;
        if state.blend_target < state.blend_value {
            state.blend_value = state.blend_target;
        }
    }
}

/* 0x007C0EC0  Updates angular wheel speed from contact and driver inputs. */
pub fn wheel_update_speed_from_vehicle_speed(
    ctx: &mut VehicleCtx,
    wheel_index: usize,
    vehicle_speed: f32,
    dt: f32,
) {
    let mut acceleration: f32;
    let mut target: f32;
    let speed: f32;

    if ctx.vehicle.wheels[wheel_index].real_time.has_ground_contact != 0 {
        if ctx.vehicle.force_wheel_speed != 0 && ctx.vehicle.block_wheel_speed == 0 {
            ctx.vehicle.wheels[wheel_index].real_time.field6c =
                ctx.vehicle.forced_wheel_speed;
        } else {
            ctx.vehicle.wheels[wheel_index].real_time.field6c =
                vehicle_speed / ctx.vehicle.wheels[wheel_index].radius;
        }
        return;
    }

    acceleration = 0.0f32;
    target = 0.0f32;
    if ctx.vehicle.input_brake <= WHEEL_SPEED_EPSILON {
        if ctx.vehicle.input_gas <= WHEEL_SPEED_EPSILON
            || ctx.vehicle.block_wheel_speed != 0
            || ctx.vehicle.flag_60c != 0
        {
            ctx.vehicle.wheels[wheel_index].real_time.field6c = (f64::from(
                ctx.vehicle.wheels[wheel_index].real_time.field6c,
            ) * f64::from(WHEEL_SPEED_DAMPING)) as f32;
        } else {
            target = (f64::from(ctx.vehicle.input_gas) * 200.0f64) as f32;
            acceleration = 100.0f32;
        }
    } else {
        target = 1.0f32 - ctx.vehicle.input_brake;
        if !target.is_nan() {
            if f64::from(target) <= 0.0f64 {
                target = 0.0f32;
            } else if 1.0f64 <= f64::from(target) {
                target = 1.0f32;
            }
        }
        acceleration = -100.0f32;
    }

    if f64::from(WHEEL_SPEED_EPSILON) <= f64::from(acceleration).abs() {
        speed = (acceleration * dt) + ctx.vehicle.wheels[wheel_index].real_time.field6c;
        ctx.vehicle.wheels[wheel_index].real_time.field6c = speed;
        if (0.0f64 < f64::from(acceleration) && (target < speed))
            || ((f64::from(acceleration) < 0.0f64) && (speed < target))
        {
            ctx.vehicle.wheels[wheel_index].real_time.field6c = target;
        }
    }
}

/* ===========================================================================
 * Spring
 * ========================================================================= */

/* 0x008F46C0  Integrates a scalar damped spring by one simulation step. */
pub fn gm_spring_float_integrate(spring: &mut GmSpringFloat, dt: f32) {
    let displacement: f32;
    let spring_force: f32;
    let damping_force: f32;
    let acceleration: f32;
    let velocity: f32;

    displacement = spring.target - spring.value;
    spring_force = displacement * spring.stiffness;
    damping_force = spring.damping * spring.velocity;
    acceleration = spring_force - damping_force;
    velocity = (acceleration * dt) + spring.velocity;
    spring.velocity = velocity;
    spring.value = (dt * velocity) + spring.value;
}

/* ===========================================================================
 * Impulse math
 * ========================================================================= */

/* 0x007BD090  Computes one contact impulse from effective point mass. */
pub fn sdyna_math_compute_impulse(
    mass: f32,
    inverse_inertia: &GmMat3,
    restitution: f32,
    relative_speed: &GmVec3,
    normal: &GmVec3,
    lever_arm: &GmVec3,
) -> GmVec3 {
    let mut angular = GmVec3::ZERO;
    let mut projected = GmVec3::ZERO;
    let relative_normal_speed: f32;
    let rotational_mass: f32;
    let denominator: f32;
    let numerator: f32;
    let scale: f32;

    angular.x = (normal.z * lever_arm.y) - (normal.y * lever_arm.z);
    angular.y = (normal.x * lever_arm.z) - (lever_arm.x * normal.z);
    angular.z = (normal.y * lever_arm.x) - (normal.x * lever_arm.y);
    angular.mult_mat3(inverse_inertia);

    projected.x = (lever_arm.z * angular.y) - (angular.z * lever_arm.y);
    projected.y = (lever_arm.x * angular.z) - (lever_arm.z * angular.x);
    projected.z = (angular.x * lever_arm.y) - (lever_arm.x * angular.y);
    relative_normal_speed = ((relative_speed.y * normal.y) + (relative_speed.x * normal.x))
        + (relative_speed.z * normal.z);
    rotational_mass = ((normal.x * projected.x) + (normal.y * projected.y))
        + (projected.z * normal.z);
    denominator = (1.0f32 / mass) + rotational_mass;
    numerator = (restitution - 1.0f32) * relative_normal_speed;
    scale = numerator / denominator;
    GmVec3 {
        x: normal.x * scale,
        y: normal.y * scale,
        z: scale * normal.z,
    }
}

/* ===========================================================================
 * Local force pipeline
 * ========================================================================= */

/* 0x007BE310  Adds a local central force to the vehicle rigid body. */
pub fn add_vehicle_central_force(ctx: &mut VehicleCtx, force: &GmVec3) {
    let mut world_force = GmVec3::ZERO;

    world_force.set_mult_mat3(force, &ctx.live().rot);
    add_vec3(&mut ctx.live_mut().force, &world_force);
    add_vec3(&mut ctx.vehicle.total_force_added, force);
}

/* 0x007BE360  Adds a local torque to the vehicle rigid body. */
pub fn add_vehicle_torque(ctx: &mut VehicleCtx, torque: &GmVec3) {
    let mut world_torque = GmVec3::ZERO;

    world_torque.set_mult_mat3(torque, &ctx.live().rot);
    add_vec3(&mut ctx.live_mut().torque, &world_torque);
}

/* 0x007BE2C0  Adds a local force at a local point on the vehicle. */
pub fn add_vehicle_force(ctx: &mut VehicleCtx, force: &GmVec3, point: &GmVec3) {
    let mut world_force = GmVec3::ZERO;
    let mut world_point = GmVec3::ZERO;
    let mut center_of_mass = GmVec3::ZERO;
    let mut lever = GmVec3::ZERO;
    let mut torque = GmVec3::ZERO;

    world_force.set_mult_mat3(force, &ctx.live().rot);
    world_point.set_mult_mat3(point, &ctx.live().rot);
    world_point.x = world_point.x + ctx.live().pos.x;
    world_point.y = world_point.y + ctx.live().pos.y;
    world_point.z = world_point.z + ctx.live().pos.z;
    center_of_mass.set_mult_mat3(&ctx.params().com_offset, &ctx.live().rot);
    center_of_mass.x = center_of_mass.x + ctx.live().pos.x;
    center_of_mass.y = center_of_mass.y + ctx.live().pos.y;
    center_of_mass.z = center_of_mass.z + ctx.live().pos.z;
    lever.x = world_point.x - center_of_mass.x;
    lever.y = world_point.y - center_of_mass.y;
    lever.z = world_point.z - center_of_mass.z;
    torque.x = (lever.y * world_force.z) - (lever.z * world_force.y);
    torque.y = (lever.z * world_force.x) - (lever.x * world_force.z);
    torque.z = (lever.x * world_force.y) - (lever.y * world_force.x);
    add_vec3(&mut ctx.live_mut().force, &world_force);
    add_vec3(&mut ctx.live_mut().torque, &torque);
    add_vec3(&mut ctx.vehicle.total_force_added, force);
}

/* ===========================================================================
 * Engine / gearbox
 * ========================================================================= */

/* 0x007BE00B  The Steer01..Steer05 engine (tuning +0x354 != 5): a speed
 * ratio drives the rpm towards max_rpm with a fixed gain and shifts through
 * the upshift/downshift tables; clutch, target_rpm and engine_mode are not
 * touched. */
fn engine_integrate_oldmodels(
    ctx: &mut VehicleCtx,
    throttle: f32,
    dt: f32,
    shift_ready: bool,
) {
    let engine = &mut ctx.vehicle.engine;
    let tuning = &ctx.tuning.base;
    let speed = ctx.vehicle.current_local_speed;
    /* 0x00B43310 0.2, 0x00B5B8E0 0.3, 0x00B362C0 0.1, 0x00B9EFB8 1.9:
     * floats widened to double; 0x00B80D18 0.04f, 0x00B3D274 12.0f,
     * 0x00B36AE8 3.5f. */
    let limit = tuning.forward_speed_limit_scale * 0.2f32;
    let magnitude = throttle.abs();
    let can_shift = !shift_ready && !(0.0f64 < f64::from(engine.shift_timer));
    let index: u32 = if engine.gear > 1 {
        (engine.gear as u32) - 1
    } else {
        0
    };
    let upshift = tuning.gear_upshift[index as usize];
    let downshift = tuning.gear_downshift[index as usize];
    let ratio = tuning.gear_ratios[index as usize];
    let sum = (((speed.x * speed.x) * 0.3f32) + (speed.z * speed.z))
        + ((speed.y * speed.y) * 0.1f32);
    let ratio_speed = (sum.sqrt() / limit) * ratio;
    let drive: f32;
    let target: f32;

    if can_shift {
        drive = ratio_speed;
        if engine.reverse != 0 {
            if engine.gear != 0 {
                engine.gear = 0;
                engine.shift_timer = 0.04f32;
            }
        } else if engine.gear == 0 {
            engine.gear = 1;
            engine.shift_timer = 0.04f32;
        } else if (upshift < ratio_speed) && engine.gear < 5 {
            engine.gear += 1;
            engine.shift_timer = 0.04f32;
        } else if (ratio_speed < downshift) && engine.gear > 1 {
            engine.gear -= 1;
            engine.shift_timer = 0.04f32;
        }
    } else {
        drive = magnitude;
        if (0.0f64 <= f64::from(engine.shift_timer))
            && !((dt + dt) < engine.shift_timer)
        {
            engine.rpm = engine.rpm - ((engine.max_rpm * dt) * 1.9f32);
        }
    }
    target = (engine.max_rpm * drive) - engine.rpm;
    engine.rpm =
        ((target * dt) * if can_shift { 12.0f32 } else { 3.5f32 }) + engine.rpm;
}

/* 0x007BD700  Integrates the engine and gearbox state; the model-5 body is
 * 0x007BD7C5.., every other tuning takes 0x007BE00B. */
pub fn engine_integrate(ctx: &mut VehicleCtx, throttle: f32, dt: f32) {
    let throttle_on = 0.1f64 < f64::from(throttle);
    let mut all_airborne = true;
    let shift_ready: bool;
    let mut value: f32;
    let mut gear_index: u32;

    for i in 0..ctx.vehicle.wheel_count {
        if ctx.vehicle.wheels[i as usize].real_time.has_ground_contact != 0 {
            all_airborne = false;
            break;
        }
    }
    if 0.0f64 < f64::from(ctx.vehicle.engine.shift_timer) {
        ctx.vehicle.engine.shift_timer = ctx.vehicle.engine.shift_timer - dt;
    }
    shift_ready = all_airborne || (0.0f64 < f64::from(ctx.vehicle.engine.shift_timer));

    'clamp_rpm: {
        'shift_checks: {
            if ctx.tuning.base.engine_model != 5 {
                engine_integrate_oldmodels(ctx, throttle, dt, shift_ready);
                break 'clamp_rpm;
            }

            if shift_ready {
                if throttle_on {
                    ctx.vehicle.engine.rpm =
                        (ctx.tuning.base.engine_rpm_accel * dt) + ctx.vehicle.engine.rpm;
                } else {
                    ctx.vehicle.engine.rpm = ctx.vehicle.engine.rpm
                        - (ctx.tuning.base.engine_rpm_decel * dt);
                }
                break 'clamp_rpm;
            }

            if ctx.vehicle.drive_mode == 1 || ctx.vehicle.drive_mode == 2 {
                if ctx.vehicle.engine.gear != 0 {
                    ctx.vehicle.engine_mode = 4;
                }
            } else if ctx.vehicle.engine_mode == 4 {
                ctx.vehicle.engine_mode = 0;
            }

            if ctx.vehicle.engine_mode == 2 {
                ctx.vehicle.engine_limit_flag = ((ctx.tuning.base.speed_32c
                    < ctx.vehicle.current_local_speed.z)
                    || (ctx.vehicle.current_local_speed.z < ctx.tuning.base.speed_330))
                    as i32;
                ctx.vehicle.engine.clutch = 1.0f32;
                ctx.vehicle.engine.target_rpm = (ctx.tuning.base.gear_aux[1] * 0.0f32)
                    + (ctx.vehicle.current_local_speed.z.abs()
                        * ctx.tuning.base.gear_ratios[1]);
                /* 0x007BDD33..0x007BDD54: the active-throttle branch stores
                 * engine_rpm_low_accel * dt + rpm and falls through to the shift
                 * checks. Only the deceleration branches (0x007BDCB2 and
                 * 0x007BDD0C) reach the target comparison at 0x007BDCDA. */
                if ctx.vehicle.engine_limit_flag == 0 && throttle_on {
                    ctx.vehicle.engine.rpm = (ctx.tuning.base.engine_rpm_low_accel * dt)
                        + ctx.vehicle.engine.rpm;
                    break 'shift_checks;
                }
                value = ctx.vehicle.engine.rpm
                    - (ctx.tuning.base.engine_rpm_high_decel * dt);
                ctx.vehicle.engine.rpm = value;
                if value <= ctx.vehicle.engine.target_rpm {
                    ctx.vehicle.engine.rpm = ctx.vehicle.engine.target_rpm;
                    ctx.vehicle.engine_mode = 0;
                    ctx.vehicle.engine_limit_flag = 0;
                }
                break 'shift_checks;
            }

            if ctx.vehicle.engine_mode == 3 {
                ctx.vehicle.engine_limit_flag = ((ctx.vehicle.current_local_speed.z
                    < ctx.tuning.base.speed_338)
                    || (ctx.tuning.base.speed_334 < ctx.vehicle.current_local_speed.z))
                    as i32;
                ctx.vehicle.engine.clutch = 1.0f32;
                ctx.vehicle.engine.target_rpm = (ctx.tuning.base.gear_aux[0] * 0.0f32)
                    + (ctx.vehicle.current_local_speed.z.abs()
                        * ctx.tuning.base.gear_ratios[0]);
                /* 0x007BDBC3 jumps to the same unclamped active-throttle store
                 * at 0x007BDD33; only the deceleration branches reach the target
                 * comparison at 0x007BDB8C. */
                if ctx.vehicle.engine_limit_flag == 0 && throttle_on {
                    ctx.vehicle.engine.rpm = (ctx.tuning.base.engine_rpm_low_accel * dt)
                        + ctx.vehicle.engine.rpm;
                    break 'shift_checks;
                }
                value = ctx.vehicle.engine.rpm
                    - (ctx.tuning.base.engine_rpm_high_decel * dt);
                ctx.vehicle.engine.rpm = value;
                if !(ctx.vehicle.engine.target_rpm < value) {
                    ctx.vehicle.engine.rpm = ctx.vehicle.engine.target_rpm;
                    ctx.vehicle.engine_mode = 0;
                    ctx.vehicle.engine_limit_flag = 0;
                }
                break 'shift_checks;
            }

            if ctx.vehicle.engine_mode == 4 {
                ctx.vehicle.engine.target_rpm = ctx.vehicle.engine.max_rpm;
                ctx.vehicle.engine.clutch = 1.15f32;
                if ctx.vehicle.engine.max_rpm <= ctx.vehicle.engine.rpm {
                    if ctx.vehicle.engine.rpm <= ctx.vehicle.engine.max_rpm {
                        break 'shift_checks;
                    }
                    value = ctx.tuning.base.engine_rpm_decel;
                } else {
                    value = ctx.tuning.base.engine_rpm_reverse_accel;
                }
                ctx.vehicle.engine.rpm = (value * dt) + ctx.vehicle.engine.rpm;
                break 'shift_checks;
            }

            if ctx.vehicle.turbo_active != 0 && throttle_on {
                value = 1.15f32;
                if f64::from(ctx.vehicle.engine.clutch) < 1.15f64 {
                    value = (((1.15f32 - ctx.vehicle.engine.clutch) * 0.3f32) * dt)
                        + ctx.vehicle.engine.clutch;
                }
            } else {
                value = 1.0f32;
            }
            ctx.vehicle.engine.clutch = value;
            gear_index = ctx.vehicle.engine.gear as u32;
            let mut target = (ctx.tuning.base.gear_aux[gear_index as usize] * 0.0f32)
                + ((ctx.vehicle.current_local_speed.z * ctx.vehicle.engine.clutch).abs()
                    * ctx.tuning.base.gear_ratios[gear_index as usize]);
            ctx.vehicle.engine.target_rpm = target;
            if (ctx.vehicle.engine.reverse == 0 && gear_index == 0)
                || (ctx.vehicle.engine.reverse != 0 && gear_index != 0)
            {
                ctx.vehicle.engine.target_rpm = 0.0f32;
                target = 0.0f32;
            }
            if target <= ctx.vehicle.engine.rpm {
                if throttle_on {
                    if ctx.vehicle.turbo_active == 0
                        || ctx.vehicle.drive_mode == 1
                        || ctx.vehicle.drive_mode == 2
                    {
                        value = ctx.tuning.base.engine_rpm_turbo_decel;
                    } else {
                        value = ctx.tuning.base.engine_rpm_decel;
                    }
                } else {
                    value = ctx.tuning.base.engine_rpm_high_decel;
                }
                ctx.vehicle.engine.rpm =
                    ctx.vehicle.engine.rpm - (value * dt);
                if ctx.vehicle.engine.rpm < ctx.vehicle.engine.target_rpm {
                    ctx.vehicle.engine_mode = 0;
                }
            } else {
                ctx.vehicle.engine.rpm = (ctx.tuning.base.engine_rpm_follow_accel * dt)
                    + ctx.vehicle.engine.rpm;
                if ctx.vehicle.engine.target_rpm < ctx.vehicle.engine.rpm {
                    ctx.vehicle.engine_mode = 0;
                }
            }
        }

        /* shift_checks: */
        if !throttle_on || ctx.vehicle.engine.reverse != 0 || ctx.vehicle.engine_mode != 0 {
            if (ctx.tuning.base.speed_338 < ctx.vehicle.current_local_speed.z)
                && (ctx.vehicle.current_local_speed.z < ctx.tuning.base.speed_334)
                && throttle_on
                && ctx.vehicle.engine.reverse != 0
                && ctx.vehicle.engine_mode == 0
            {
                ctx.vehicle.engine_mode = 3;
                ctx.vehicle.engine_limit_flag = 0;
                if ctx.vehicle.engine.gear != 0 {
                    ctx.vehicle.engine.gear = 0;
                    ctx.vehicle.engine.shift_timer = 0.025f32;
                }
            }
        } else if (ctx.tuning.base.speed_330 < ctx.vehicle.current_local_speed.z)
            && (ctx.vehicle.current_local_speed.z < ctx.tuning.base.speed_32c)
        {
            ctx.vehicle.engine_mode = 2;
            ctx.vehicle.engine_limit_flag = 0;
            if ctx.vehicle.engine.gear == 0 {
                ctx.vehicle.engine.shift_timer = 0.025f32;
                ctx.vehicle.engine.gear = 1;
            }
        }

        if ctx.vehicle.engine_mode == 0 || ctx.vehicle.engine_mode == 1 {
            if ctx.vehicle.engine.reverse == 0 {
                if ctx.vehicle.engine.gear == 0 {
                    ctx.vehicle.engine_mode = 1;
                    ctx.vehicle.gear_downshift_flag = 0;
                    if f64::from(ctx.vehicle.engine.rpm) < 1000.0f64 {
                        ctx.vehicle.engine.shift_timer = 0.025f32;
                        ctx.vehicle.engine.gear = 1;
                    }
                }
                gear_index = ctx.vehicle.engine.gear as u32;
                if 0 < ctx.vehicle.engine.gear {
                    if (ctx.vehicle.engine.target_rpm
                        <= (ctx.tuning.base.gear_upshift[gear_index as usize]
                            * ctx.vehicle.engine.max_rpm))
                        || 4 < ctx.vehicle.engine.gear
                    {
                        if (ctx.vehicle.engine.target_rpm
                            < (ctx.tuning.base.gear_downshift[gear_index as usize]
                                * ctx.vehicle.engine.max_rpm))
                            && 1 < ctx.vehicle.engine.gear
                        {
                            ctx.vehicle.engine.shift_timer = 0.025f32;
                            ctx.vehicle.engine.gear -= 1;
                            ctx.vehicle.engine_mode = 1;
                            ctx.vehicle.gear_downshift_flag = 1;
                        }
                    } else {
                        ctx.vehicle.engine.shift_timer = 0.025f32;
                        ctx.vehicle.engine.gear += 1;
                        ctx.vehicle.engine_mode = 1;
                        ctx.vehicle.gear_downshift_flag = 0;
                    }
                }
            } else if ctx.vehicle.engine.gear != 0 {
                ctx.vehicle.engine_mode = 1;
                ctx.vehicle.gear_downshift_flag = 1;
                if f64::from(ctx.vehicle.engine.rpm) < 1000.0f64 {
                    ctx.vehicle.engine.gear = 0;
                    ctx.vehicle.engine.shift_timer =
                        if all_airborne { 0.002f32 } else { 0.025f32 };
                }
            }
        }
    }

    /* clamp_rpm: */
    value = ctx.vehicle.engine.rpm;
    if 0.0f64 < f64::from(value) {
        if ctx.vehicle.engine.max_rpm < value {
            ctx.vehicle.engine.rpm = ctx.vehicle.engine.max_rpm;
        }
    } else {
        ctx.vehicle.engine.rpm = 0.0f32;
    }
}

/* ===========================================================================
 * Suspension force
 * ========================================================================= */

/* 0x007C1810  Applies one wheel's suspension force to the vehicle. */
pub fn wheel_add_force_to_vehicle(ctx: &mut VehicleCtx, wheel_index: usize) {
    let mut force = GmVec3 {
        x: 0.0f32,
        y: 0.0f32,
        z: 0.0f32,
    };
    let delta: f32;

    let wheel = &ctx.vehicle.wheels[wheel_index];
    if wheel.real_time.has_ground_contact == 0 {
        return;
    }
    delta = ctx.tuning.base.suspension_rest_length - wheel.real_time.damper_absorb;
    match ctx.tuning.base.suspension_model {
        0 => {
            force.y = (ctx.tuning.base.suspension_stiffness
                * ctx.tuning.base.suspension_scale)
                * delta;
        }
        1 | 2 => {
            force.y = (delta * ctx.tuning.base.suspension_stiffness)
                - (ctx.tuning.base.suspension_damping * wheel.real_time.field04);
        }
        _ => {
            return;
        }
    }
    let point = wheel.offset_from_vehicle;
    add_vehicle_force(ctx, &force, &point);
}

/* ===========================================================================
 * Tests
 * ========================================================================= */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dyna::{CHmsDyna, CHmsDynaParams};
    use crate::vehicle::{CSceneVehicleCarWheel, Vehicle, VehicleTuning};

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

    fn grounded_vehicle() -> Vehicle {
        let mut v = Vehicle::default();
        for _ in 0..4 {
            let mut w = CSceneVehicleCarWheel::default();
            w.real_time.has_ground_contact = 1;
            v.wheels.push(w);
        }
        v.wheel_count = 4;
        v
    }

    fn ri(
        steer_left_time: u32,
        steer_left: i32,
        steer_right_time: u32,
        steer_right: i32,
        steer_analog_time: u32,
        steer_analog: f32,
        accelerate_time: u32,
        accelerate: i32,
        brake_time: u32,
        brake: i32,
        gas_analog_time: u32,
        gas_analog: f32,
    ) -> TMNFRaceInputs {
        TMNFRaceInputs {
            steer_left_time,
            steer_left,
            steer_right_time,
            steer_right,
            steer_analog_time,
            steer_analog,
            accelerate_time,
            accelerate,
            brake_time,
            brake,
            gas_analog_time,
            gas_analog,
            ..Default::default()
        }
    }

    /* ---- input mapper: first press, hold, release ---------------------- */

    #[test]
    fn input_mapper_digital_press_hold_release() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };

        /* First press: accelerate down at t=5, analog silent (t=0). */
        let pressed = ri(5, 0, 5, 0, 0, 0.0, 5, 1, 0, 0, 0, 0.0);
        update_vehicle_state_from_inputs(&pressed, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (1.0, 0.0));
        assert_eq!(ctx.vehicle.input_steer, 0.0);

        /* Hold: same packet, same outputs (the mapper is stateless). */
        update_vehicle_state_from_inputs(&pressed, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (1.0, 0.0));

        /* Release: accelerator word back to 0. */
        let released = ri(5, 0, 5, 0, 0, 0.0, 5, 0, 0, 0, 0, 0.0);
        update_vehicle_state_from_inputs(&released, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));

        /* Brake digital press. */
        let braking = ri(5, 0, 5, 0, 0, 0.0, 5, 0, 7, 1, 0, 0.0);
        update_vehicle_state_from_inputs(&braking, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 1.0));

        /* Steering digital: left, right, both (left wins), none. */
        let left = ri(9, 1, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0);
        update_vehicle_state_from_inputs(&left, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, -1.0);
        let right = ri(0, 0, 9, 1, 0, 0.0, 0, 0, 0, 0, 0, 0.0);
        update_vehicle_state_from_inputs(&right, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, 1.0);
        let both = ri(9, 1, 9, 1, 0, 0.0, 0, 0, 0, 0, 0, 0.0);
        update_vehicle_state_from_inputs(&both, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, -1.0);
    }

    /* ---- input mapper: analog vs digital precedence -------------------- */

    #[test]
    fn input_mapper_analog_vs_digital_precedence() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };

        /* Analog NEWER than every digital word: analog wins (the digital
         * brake=1@5 would give gas=0/brake=1; analog +0.5@10 overrides). */
        let newer = ri(0, 0, 0, 0, 10, 0.7, 5, 0, 5, 1, 10, 0.5);
        update_vehicle_state_from_inputs(&newer, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (1.0, 0.0));
        assert_eq!(ctx.vehicle.input_steer, -0.7);

        /* Timestamp TIE with |analog| > 0.01 and no digital word pressed:
         * the tie-break bumps the analog event past `latest` -> analog. */
        let tie = ri(5, 0, 5, 0, 5, 0.7, 5, 0, 5, 0, 5, 0.5);
        update_vehicle_state_from_inputs(&tie, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (1.0, 0.0));
        assert_eq!(ctx.vehicle.input_steer, -0.7);

        /* Timestamp tie but a digital GAS word is pressed: no gas bump ->
         * digital gas. (The STEER tie-break checks only the steer words,
         * which are still idle, so the steer analog still bumps and wins:
         * steer = -steer_analog = +0.7.) */
        let tie_digital = ri(5, 0, 5, 0, 5, -0.7, 5, 0, 5, 1, 5, 0.5);
        update_vehicle_state_from_inputs(&tie_digital, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 1.0));
        assert_eq!(ctx.vehicle.input_steer, 0.7);

        /* Timestamp tie with a STEER digital word pressed: no steer bump
         * -> digital steering wins, analog ignored. */
        let tie_steer_digital = ri(9, 1, 9, 0, 9, 0.7, 9, 0, 9, 0, 9, 0.5);
        update_vehicle_state_from_inputs(&tie_steer_digital, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, -1.0);

        /* Tie, no digital, but |analog| <= 0.01: no bump -> digital zero. */
        let tie_small = ri(5, 0, 5, 0, 5, 0.005, 5, 0, 5, 0, 5, 0.005);
        update_vehicle_state_from_inputs(&tie_small, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));
        assert_eq!(ctx.vehicle.input_steer, 0.0);

        /* Digital NEWER than analog: digital wins (analog +0.5 ignored). */
        let digital_newer = ri(0, 0, 0, 0, 5, 0.7, 10, 0, 10, 1, 5, 0.5);
        update_vehicle_state_from_inputs(&digital_newer, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 1.0));

        /* Analog signs / thresholds on the gas axis (all newer). */
        let neg = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, -0.4);
        update_vehicle_state_from_inputs(&neg, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 1.0));
        let mid = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, 0.2);
        update_vehicle_state_from_inputs(&mid, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));
        let quiet = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, -0.29);
        update_vehicle_state_from_inputs(&quiet, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));
        let nan = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, f32::NAN);
        update_vehicle_state_from_inputs(&nan, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));
    }

    /* ---- input mapper: 0.01 / 0.3 threshold edges ---------------------- */

    #[test]
    fn input_mapper_threshold_edges() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };

        /* The 0.01 tie-break gate is a *double* comparison in the C
         * (0.01 < fabs(F(x))). The f32 just above/below the double 0.01
         * pin the boundary: 0x3C23D70B = 0.010000000707805157 is above
         * 0.01 (double), 0x3C23D70A = 0.01f32 (0.009999999776482582) is
         * below it. */
        let above = ri(5, 0, 5, 0, 5, f32::from_bits(0x3C23D70B), 5, 0, 5, 0, 5, 0.0);
        update_vehicle_state_from_inputs(&above, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, -f32::from_bits(0x3C23D70B));
        let below = ri(5, 0, 5, 0, 5, f32::from_bits(0x3C23D70A), 5, 0, 5, 0, 5, 0.0);
        update_vehicle_state_from_inputs(&below, &mut ctx);
        assert_eq!(ctx.vehicle.input_steer, 0.0);

        /* Gas analog thresholds are double comparisons too:
         * 0x3E99999A = 0.3f32 passes `0.3 <= x`; 0x3E999999 does not. */
        let at = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, f32::from_bits(0x3E99999A));
        update_vehicle_state_from_inputs(&at, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (1.0, 0.0));
        let under = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, f32::from_bits(0x3E999999));
        update_vehicle_state_from_inputs(&under, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 0.0));
        /* -0.3f32 passes `x <= -0.3` (double). */
        let rev = ri(0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 10, -f32::from_bits(0x3E99999A));
        update_vehicle_state_from_inputs(&rev, &mut ctx);
        assert_eq!((ctx.vehicle.input_gas, ctx.vehicle.input_brake), (0.0, 1.0));
    }

    /* ---- spring ---------------------------------------------------------- */

    #[test]
    fn spring_integrate_exact_step_and_zero_dt() {
        let mut spring = GmSpringFloat {
            stiffness: 10.0,
            damping: 2.0,
            value: 0.0,
            target: 1.0,
            velocity: 0.0,
        };
        gm_spring_float_integrate(&mut spring, 0.1);
        /* displacement=1, spring=10, damping=0, accel=10;
         * velocity = 10*0.1f32 + 0 == 1.0 (exact product rounds to 1);
         * value = 0.1f32*1.0 + 0 == 0.1f32. */
        assert_eq!(spring.velocity, 1.0);
        assert_eq!(spring.value.to_bits(), 0x3DCC_CCCD);

        /* dt = 0: nothing moves. */
        let mut frozen = GmSpringFloat {
            stiffness: 10.0,
            damping: 2.0,
            value: 3.0,
            target: 1.0,
            velocity: -0.5,
        };
        gm_spring_float_integrate(&mut frozen, 0.0);
        assert_eq!(frozen.value, 3.0);
        assert_eq!(frozen.velocity, -0.5);
    }

    #[test]
    fn spring_integrate_stability() {
        /* Overdamped-ish spring under symplectic Euler: converges to the
         * target, stays finite, no NaN/inf. */
        let mut spring = GmSpringFloat {
            stiffness: 170.0,
            damping: 22.0,
            value: -1.0,
            target: 1.0,
            velocity: 0.0,
        };
        for _ in 0..2000 {
            gm_spring_float_integrate(&mut spring, 0.01);
        }
        assert!(spring.value.is_finite());
        assert!(spring.velocity.is_finite());
        assert!((spring.value - 1.0).abs() < 1e-3, "value = {}", spring.value);
        assert!(spring.velocity.abs() < 1e-3, "velocity = {}", spring.velocity);
    }

    /* ---- wheel real-time state ------------------------------------------- */

    #[test]
    fn wheel_real_time_state_integrate_basis_phase_blend() {
        let mut state = CSceneVehicleCarWheelRealTimeState::default();
        state.field90 = GmVec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        state.field6c = 2000.0;
        state.rotation_phase = 0.0;
        state.blend_value = 0.2;
        state.blend_target = 0.5;
        wheel_real_time_state_integrate(&mut state, 1.0);

        /* up = +Y, dov = (0,0,1) -> orthonormal frame = identity. */
        assert_eq!(state.basis1, GmMat3::IDENTITY);
        /* 2000 wraps once against WHEEL_ANGLE_PERIOD (1608.4954833984375):
         * fmod(2000, 1608.4954833984375) = 391.5045166015625 exactly. */
        assert_eq!(state.rotation_phase, 391.5045166015625);
        /* blend rises by dt=1.0 and clamps at the target 0.5. */
        assert_eq!(state.blend_value, 0.5);

        /* With a small dt the blend rises one step without clamping:
         * 0.2f32 + 0.1f32 = 0.30000001192092896. */
        let mut rising = CSceneVehicleCarWheelRealTimeState::default();
        rising.field90 = GmVec3::ZERO;
        rising.blend_value = 0.2;
        rising.blend_target = 0.5;
        wheel_real_time_state_integrate(&mut rising, 0.1);
        assert_eq!(rising.blend_value.to_bits(), 0x3E99_999A);

        /* Negative phase wraps up into the period. */
        let mut neg = CSceneVehicleCarWheelRealTimeState::default();
        neg.field90 = GmVec3::ZERO;
        neg.field6c = -100.0;
        wheel_real_time_state_integrate(&mut neg, 1.0);
        assert_eq!(neg.rotation_phase, 1508.4954833984375);

        /* Blend deceleration clamps at the target. */
        let mut down = CSceneVehicleCarWheelRealTimeState::default();
        down.field90 = GmVec3::ZERO;
        down.blend_value = 0.2;
        down.blend_target = 0.15;
        wheel_real_time_state_integrate(&mut down, 0.1);
        assert_eq!(down.blend_value, 0.15);
        /* Exact equality at the target: stays put (target <= value, and the
         * decremented value is not below target). */
        let mut hold = CSceneVehicleCarWheelRealTimeState::default();
        hold.field90 = GmVec3::ZERO;
        hold.blend_value = 0.15;
        hold.blend_target = 0.15;
        wheel_real_time_state_integrate(&mut hold, 0.1);
        assert_eq!(hold.blend_value, 0.15);
    }

    /* ---- wheel speed ------------------------------------------------------ */

    #[test]
    fn wheel_speed_from_vehicle_speed_paths() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };
        ctx.vehicle.wheels.push(CSceneVehicleCarWheel::default());
        ctx.vehicle.wheel_count = 1;
        ctx.vehicle.wheels[0].radius = 0.5;

        /* Grounded: field6c = vehicle_speed / radius. */
        ctx.vehicle.wheels[0].real_time.has_ground_contact = 1;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 20.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 40.0);

        /* Grounded + force_wheel_speed (unblocked): forced value wins. */
        ctx.vehicle.force_wheel_speed = 1;
        ctx.vehicle.forced_wheel_speed = 123.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 20.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 123.0);

        /* Grounded + blocked: back to vehicle_speed / radius. */
        ctx.vehicle.block_wheel_speed = 1;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 20.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 40.0);
        ctx.vehicle.force_wheel_speed = 0;
        ctx.vehicle.block_wheel_speed = 0;

        /* Airborne + gas: spin up towards input_gas * 200. */
        ctx.vehicle.wheels[0].real_time.has_ground_contact = 0;
        ctx.vehicle.wheels[0].real_time.field6c = 10.0;
        ctx.vehicle.input_gas = 1.0;
        ctx.vehicle.input_brake = 0.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 11.0);
        /* Overshoot clamps at the target (200). */
        ctx.vehicle.wheels[0].real_time.field6c = 250.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 200.0);

        /* Airborne + no gas: damped towards zero (100 * 0.995f32 -> 99.5). */
        ctx.vehicle.input_gas = 0.0;
        ctx.vehicle.wheels[0].real_time.field6c = 100.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 99.5);

        /* Airborne + brake: decelerate towards 1 - input_brake. */
        ctx.vehicle.input_brake = 0.5;
        ctx.vehicle.wheels[0].real_time.field6c = 10.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 9.0);
        ctx.vehicle.wheels[0].real_time.field6c = 0.3;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 0.5);
        /* Over-brake clamps the target at 0. */
        ctx.vehicle.input_brake = 1.5;
        ctx.vehicle.wheels[0].real_time.field6c = 0.005;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 0.0);
        /* Negative brake passes `input_brake <= epsilon` and therefore takes
         * the DAMPING branch, not the brake branch (1 - input_brake > 1 is
         * unreachable: input_brake < 0 implies input_brake <= epsilon).
         * 250 * 0.99500000476837158203 -> 248.75. */
        ctx.vehicle.input_brake = -0.5;
        ctx.vehicle.wheels[0].real_time.field6c = 250.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 248.75);

        /* Airborne + gas, but block_wheel_speed / flag_60c: damping instead. */
        ctx.vehicle.input_brake = 0.0;
        ctx.vehicle.input_gas = 1.0;
        ctx.vehicle.block_wheel_speed = 1;
        ctx.vehicle.wheels[0].real_time.field6c = 100.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 99.5);
        ctx.vehicle.block_wheel_speed = 0;
        ctx.vehicle.flag_60c = 1;
        ctx.vehicle.wheels[0].real_time.field6c = 100.0;
        wheel_update_speed_from_vehicle_speed(&mut ctx, 0, 0.0, 0.01);
        assert_eq!(ctx.vehicle.wheels[0].real_time.field6c, 99.5);
    }

    /* ---- impulse ---------------------------------------------------------- */

    #[test]
    fn sdyna_math_compute_impulse_identity_inertia() {
        /* Central hit (zero lever arm): impulse = normal * (e-1)*v_n / (1/m). */
        let j = sdyna_math_compute_impulse(
            4.0,
            &GmMat3::IDENTITY,
            0.0,
            &GmVec3 {
                x: 0.0,
                y: -2.0,
                z: 0.0,
            },
            &GmVec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            &GmVec3::ZERO,
        );
        assert_eq!(j, GmVec3 { x: 0.0, y: 8.0, z: 0.0 });

        /* Off-center hit: rotational mass adds 1/2 via lever=(1,0,0),
         * angular=(0,0,1), projected=(0,1,0) with identity inertia. */
        let j = sdyna_math_compute_impulse(
            2.0,
            &GmMat3::IDENTITY,
            0.5,
            &GmVec3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
            &GmVec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            &GmVec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
        );
        /* scale = (0.5-1)*(-1) / (1/2 + 1) = 0.5/1.5 = 1/3 (f32). */
        assert_eq!(j.x, 0.0);
        assert_eq!(j.y.to_bits(), (1.0f32 / 3.0f32).to_bits());
        assert_eq!(j.z, 0.0);
    }

    /* ---- force pipeline ---------------------------------------------------- */

    #[test]
    fn add_vehicle_force_torque_and_central() {
        let mut vehicle = Vehicle::default();
        let tuning = VehicleTuning::default();
        let mut dyna = make_dyna();
        {
            let mut ctx = VehicleCtx {
                vehicle: &mut vehicle,
                tuning: &tuning,
                dyna: &mut dyna,
            };
            /* Identity rotation, pos=(10,20,30), com=(1,1,1):
             * force=(0,10,0) at point=(1,0,0) -> lever=(0,-1,-1),
             * torque=(10,0,0). */
            ctx.dyna.live_mut().pos = GmVec3 {
                x: 10.0,
                y: 20.0,
                z: 30.0,
            };
            ctx.dyna.params.com_offset = GmVec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            };
            add_vehicle_force(
                &mut ctx,
                &GmVec3 {
                    x: 0.0,
                    y: 10.0,
                    z: 0.0,
                },
                &GmVec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            );
            assert_eq!(ctx.live().force, GmVec3 { x: 0.0, y: 10.0, z: 0.0 });
            assert_eq!(ctx.live().torque, GmVec3 { x: 10.0, y: 0.0, z: 0.0 });
            assert_eq!(
                ctx.vehicle.total_force_added,
                GmVec3 {
                    x: 0.0,
                    y: 10.0,
                    z: 0.0
                }
            );

            /* Central force + torque rotate through the dyna rotation
             * (90 degrees about Z: x-axis -> y-axis). */
            ctx.dyna.live_mut().rot = GmMat3 {
                m: [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            };
            add_vehicle_central_force(
                &mut ctx,
                &GmVec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            );
            assert_eq!(
                ctx.live().force,
                GmVec3 {
                    x: 0.0,
                    y: 11.0,
                    z: 0.0
                }
            );
            add_vehicle_torque(
                &mut ctx,
                &GmVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 5.0,
                },
            );
            assert_eq!(
                ctx.live().torque,
                GmVec3 {
                    x: 10.0,
                    y: 0.0,
                    z: 5.0
                }
            );
            assert_eq!(
                ctx.vehicle.total_force_added,
                GmVec3 {
                    x: 1.0,
                    y: 10.0,
                    z: 0.0
                }
            );
        }
    }

    /* ---- engine ------------------------------------------------------------ */

    fn model5_tuning() -> VehicleTuning {
        let mut t = VehicleTuning::default();
        t.base.engine_model = 5;
        t.base.gear_ratios = vec![0.03, 60.0, 40.0, 30.0, 20.0, 10.0];
        t.base.gear_upshift = vec![0.8, 0.9, 0.95, 0.97, 0.99, 1.0];
        t.base.gear_downshift = vec![0.3, 0.35, 0.4, 0.45, 0.5, 0.55];
        t.base.gear_aux = vec![0.0; 6];
        t.base.engine_rpm_accel = 4000.0;
        t.base.engine_rpm_decel = 3000.0;
        t.base.engine_rpm_reverse_accel = 2000.0;
        t.base.engine_rpm_high_decel = 2500.0;
        t.base.engine_rpm_low_accel = 3000.0;
        t.base.engine_rpm_follow_accel = 3500.0;
        t.base.engine_rpm_turbo_decel = 1500.0;
        t.base.speed_32c = 100.0;
        t.base.speed_330 = -1.0;
        t.base.speed_334 = 60.0;
        t.base.speed_338 = 5.0;
        t
    }

    #[test]
    fn engine_integrate_model5_launch() {
        let mut vehicle = grounded_vehicle();
        vehicle.engine.max_rpm = 8000.0;
        let tuning = model5_tuning();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };
        engine_integrate(&mut ctx, 1.0, 0.01);

        /* rpm dips to -15 (turbo_decel * dt) then clamps to 0; the speed
         * window engages engine_mode 2 and first gear. */
        assert_eq!(ctx.vehicle.engine.rpm, 0.0);
        assert_eq!(ctx.vehicle.engine.gear, 1);
        assert_eq!(ctx.vehicle.engine_mode, 2);
        assert_eq!(ctx.vehicle.engine.shift_timer, 0.025);
        assert_eq!(ctx.vehicle.engine.clutch, 1.0);
        assert_eq!(ctx.vehicle.engine_limit_flag, 0);
    }

    #[test]
    fn engine_integrate_model5_mode2_accel() {
        let mut vehicle = grounded_vehicle();
        vehicle.engine.max_rpm = 8000.0;
        vehicle.engine.rpm = 1000.0;
        vehicle.engine.gear = 1;
        vehicle.engine_mode = 2;
        vehicle.current_local_speed.z = 50.0;
        let tuning = model5_tuning();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };
        engine_integrate(&mut ctx, 1.0, 0.01);

        /* target_rpm = |50| * 60 = 3000; low_accel path:
         * rpm = 3000*0.01f32 + 1000 = 1030 (3000*0.01f32 == 30 exactly). */
        assert_eq!(ctx.vehicle.engine.rpm, 1030.0);
        assert_eq!(ctx.vehicle.engine.target_rpm, 3000.0);
        assert_eq!(ctx.vehicle.engine.clutch, 1.0);
        assert_eq!(ctx.vehicle.engine_mode, 2);
    }

    #[test]
    fn engine_integrate_oldmodels_path() {
        let mut vehicle = grounded_vehicle();
        vehicle.engine.max_rpm = 8000.0;
        vehicle.engine.rpm = 1000.0;
        vehicle.engine.gear = 1;
        let mut tuning = model5_tuning();
        tuning.base.engine_model = 0; /* != 5 -> 0x007BE00B body */
        tuning.base.forward_speed_limit_scale = 100.0;
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };
        engine_integrate(&mut ctx, 1.0, 0.01);

        /* Zero speed: drive = 0, no shift; target = -1000;
         * rpm = ((-1000 * 0.01f32) * 12.0f32) + 1000 = 880. */
        assert_eq!(ctx.vehicle.engine.rpm, 880.0);
        assert_eq!(ctx.vehicle.engine.gear, 1);
        assert_eq!(ctx.vehicle.engine.shift_timer, 0.0);
    }

    #[test]
    fn engine_turbo_clutch_double_literal_comparison() {
        /* The C compares F(clutch) < 1.15 against the *double* 1.15. With
         * clutch == 1.15f32 (which is BELOW 1.15 as a double) the branch is
         * taken, and a NaN dt poisons the product: value = 0*NaN + 1.15f32
         * = NaN. A port that (wrongly) compares in f32 would leave the
         * clutch at exactly 1.15f32 — this test fails for that port. */
        let mut vehicle = grounded_vehicle();
        vehicle.engine.max_rpm = 8000.0;
        vehicle.engine.clutch = 1.15f32;
        vehicle.turbo_active = 1;
        let tuning = model5_tuning();
        let mut dyna = make_dyna();
        let mut ctx = VehicleCtx {
            vehicle: &mut vehicle,
            tuning: &tuning,
            dyna: &mut dyna,
        };
        engine_integrate(&mut ctx, 1.0, f32::NAN);

        assert!(ctx.vehicle.engine.clutch.is_nan());
        /* rpm goes NaN through the turbo_decel*dt store and clamps to 0. */
        assert_eq!(ctx.vehicle.engine.rpm, 0.0);
        assert_eq!(ctx.vehicle.engine.gear, 1);
        assert_eq!(ctx.vehicle.engine_mode, 2);
    }

    /* ---- suspension -------------------------------------------------------- */

    #[test]
    fn wheel_add_force_to_vehicle_models() {
        let mut vehicle = Vehicle::default();
        let mut tuning = VehicleTuning::default();
        let mut dyna = make_dyna();

        vehicle.wheels.push(CSceneVehicleCarWheel::default());
        vehicle.wheel_count = 1;
        vehicle.wheels[0].real_time.has_ground_contact = 1;
        vehicle.wheels[0].real_time.damper_absorb = 0.25;
        vehicle.wheels[0].real_time.field04 = 0.5;
        vehicle.wheels[0].offset_from_vehicle = GmVec3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        };

        /* Model 1: delta*stiffness - damping*field04 = 0.25*100 - 10*0.5 = 20. */
        tuning.base.suspension_model = 1;
        tuning.base.suspension_stiffness = 100.0;
        tuning.base.suspension_damping = 10.0;
        tuning.base.suspension_rest_length = 0.5;
        {
            let mut ctx = VehicleCtx {
                vehicle: &mut vehicle,
                tuning: &tuning,
                dyna: &mut dyna,
            };
            wheel_add_force_to_vehicle(&mut ctx, 0);
            assert_eq!(ctx.live().force, GmVec3 { x: 0.0, y: 20.0, z: 0.0 });
            /* lever = (0,0,-1) x force (0,20,0) = (20, 0, 0). */
            assert_eq!(ctx.live().torque, GmVec3 { x: 20.0, y: 0.0, z: 0.0 });
            assert_eq!(
                ctx.vehicle.total_force_added,
                GmVec3 {
                    x: 0.0,
                    y: 20.0,
                    z: 0.0
                }
            );
        }

        /* Model 0: (stiffness * scale) * delta = (100 * 0.5) * 0.25 = 12.5. */
        tuning.base.suspension_model = 0;
        tuning.base.suspension_scale = 0.5;
        {
            let mut ctx = VehicleCtx {
                vehicle: &mut vehicle,
                tuning: &tuning,
                dyna: &mut dyna,
            };
            wheel_add_force_to_vehicle(&mut ctx, 0);
            assert_eq!(ctx.live().force, GmVec3 { x: 0.0, y: 32.5, z: 0.0 });
            assert_eq!(
                ctx.live().torque,
                GmVec3 {
                    x: 32.5,
                    y: 0.0,
                    z: 0.0
                }
            );
        }

        /* No ground contact: nothing is added. */
        vehicle.wheels[0].real_time.has_ground_contact = 0;
        {
            let mut ctx = VehicleCtx {
                vehicle: &mut vehicle,
                tuning: &tuning,
                dyna: &mut dyna,
            };
            wheel_add_force_to_vehicle(&mut ctx, 0);
            assert_eq!(ctx.live().force, GmVec3 { x: 0.0, y: 32.5, z: 0.0 });
            assert_eq!(ctx.live().torque, GmVec3 { x: 32.5, y: 0.0, z: 0.0 });
        }

        /* Unknown suspension model: early return, nothing added. */
        vehicle.wheels[0].real_time.has_ground_contact = 1;
        tuning.base.suspension_model = 3;
        {
            let mut ctx = VehicleCtx {
                vehicle: &mut vehicle,
                tuning: &tuning,
                dyna: &mut dyna,
            };
            wheel_add_force_to_vehicle(&mut ctx, 0);
            assert_eq!(ctx.live().force, GmVec3 { x: 0.0, y: 32.5, z: 0.0 });
        }
    }

    /* ---- gm_mod wrap (via the public wheel integrate path above) ---------- */

    #[test]
    fn gm_mod_wraps_into_range() {
        /* Directly exercised through wheel_real_time_state_integrate in
         * wheel_real_time_state_integrate_basis_phase_blend; the identity
         * check here pins the early-out branch (value already in range). */
        let mut state = CSceneVehicleCarWheelRealTimeState::default();
        state.field90 = GmVec3::ZERO;
        state.field6c = 100.0;
        state.rotation_phase = 5.0;
        wheel_real_time_state_integrate(&mut state, 0.0);
        assert_eq!(state.rotation_phase, 5.0);
    }
}
