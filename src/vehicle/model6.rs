//! Model 6 — the Stadium car force pipeline (`src/vehicle_model6.c`),
//! transliterated from the 2.11.26 disassembly (0x007C3E80).
//!
//! This is the most-validated file in the repository. Every constant here is
//! the game's own, including the "float 0.1/π/3.6 promoted to double" forms
//! (they are NOT the decimal 0.1/π/3.6), and the operation schedule (which
//! products feed which sums, in which order) is the physics.

use crate::fp::{u32_to_x87_float, x87_cos, x87_exp, x87_sin};
use crate::gm::*;
use crate::vehicle::base::{
    add_vehicle_central_force, add_vehicle_force, add_vehicle_torque,
    wheel_add_force_to_vehicle,
};
use crate::vehicle::curve::cfunc_keys_real_get_value;
use crate::vehicle::{CSceneVehicleMaterialBlendableVals, Model6External, VehicleCtx};

/* ---------------------------------------------------------------------------
 * Constants (bit-exact; Rust has no hex-float literals)
 * ------------------------------------------------------------------------- */

/// `0x1.b7cdfcp-34f` — length epsilon.
const MODEL6_LENGTH_EPSILON: f32 = f32::from_bits(0x2EB7_CDFC);
/// `0x1.4f8b58p-17f` — control epsilon.
const MODEL6_CONTROL_EPSILON: f32 = f32::from_bits(0x374F_8B58);
/// `0x1.99999ap-4` — float 0.1 promoted to double (NOT double 0.1).
const MODEL6_POINT_ONE: f64 = f64::from_bits(0x3FB9_9999_A000_0000);
/// `0x1.666666p-1f` — 0.7f.
const MODEL6_POINT_SEVEN: f32 = f32::from_bits(0x3F33_3333);
/// `0x1.8p-1f` — 0.75f.
const MODEL6_POINT_SEVEN_FIVE: f32 = f32::from_bits(0x3F40_0000);
/// `0x1.921fb6p+1` — float pi promoted to double.
const MODEL6_PI: f64 = f64::from_bits(0x4009_21FB_6000_0000);
/// `0x1.ccccccp+1` — float 3.6 promoted to double (m/s → km/h).
const MODEL6_METERS_PER_SECOND_TO_KMH: f64 = f64::from_bits(0x400C_CCCC_C000_0000);
/// `0x1p32f`.
const MODEL6_UINT32_RANGE: f32 = f32::from_bits(0x4F80_0000);

/// `x87_mul_double(value, multiplier)`: one multiply in double, rounded once.
#[inline]
fn x87_mul_double(value: f32, multiplier: f64) -> f32 {
    ((value as f64) * multiplier) as f32
}

/// `x87_add_double`.
#[inline]
fn x87_add_double(value: f32, addend: f64) -> f32 {
    ((value as f64) + addend) as f32
}

/// `x87_sub_double`.
#[inline]
fn x87_sub_double(value: f32, subtrahend: f64) -> f32 {
    ((value as f64) - subtrahend) as f32
}

#[inline]
fn model6_abs(value: f32) -> f32 {
    value.abs()
}

#[inline]
fn model6_sign_bits(value: f32) -> f32 {
    crate::gm::gmfunc_sign(value)
}

#[inline]
fn model6_u32_float(value: u32) -> f32 {
    u32_to_x87_float(value)
}

#[inline]
fn model6_sin(value: f32) -> f32 {
    x87_sin(value)
}

#[inline]
fn model6_cos(value: f32) -> f32 {
    x87_cos(value)
}

#[inline]
fn model6_exp(value: f32) -> f32 {
    x87_exp(value)
}

use crate::vehicle::CFuncKeysReal;

fn model6_curve_value(curve: &CFuncKeysReal, position: f32) -> f32 {
    let mut lower_index = 0u32;
    cfunc_keys_real_get_value(curve, position, &mut lower_index)
}

fn model6_speed_position(speed: f32) -> f32 {
    x87_mul_double(speed, MODEL6_METERS_PER_SECOND_TO_KMH)
}

fn model6_curve_from_speed(curve: &CFuncKeysReal, speed: f32) -> f32 {
    model6_curve_value(curve, model6_speed_position(speed))
}

fn model6_max_side_friction(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.max_side_friction_from_speed.as_ref().unwrap(), speed)
}

fn model6_steer_drive_torque(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.steer_drive_torque_from_speed.as_ref().unwrap(), speed)
}

fn model6_slipping_accel(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    curves.m5_slipping_accel_scale
        * model6_curve_from_speed(curves.m5_slipping_accel_from_speed.as_ref().unwrap(), speed)
}

fn model6_accel(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.accel_from_speed.as_ref().unwrap(), speed)
}

fn model6_steer_slowdown(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.steer_slowdown_from_speed.as_ref().unwrap(), speed)
}

fn model6_rear_gear_accel(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.m6_rear_gear_accel_from_speed.as_ref().unwrap(), speed)
}

fn model6_burnout_radius(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.m6_burnout_radius_from_speed.as_ref().unwrap(), speed)
}

fn model6_burnout_rollover(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.m6_burnout_rollover_from_speed.as_ref().unwrap(), speed)
}

fn model6_donut_rollover(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed: f32) -> f32 {
    model6_curve_from_speed(curves.m6_donut_rollover_from_speed.as_ref().unwrap(), speed)
}

fn model6_rollover_ratio(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, speed_ratio: f32) -> f32 {
    model6_curve_from_speed(curves.m6_rollover_lateral_from_speed_ratio.as_ref().unwrap(), speed_ratio)
}

fn model6_lateral_speed_from_radius(curves: &crate::vehicle::CSceneVehicleCarTuningCurveSet, radius: f32) -> f32 {
    let value = model6_curve_value(
        curves.m6_lateral_speed_from_burnout_radius.as_ref().unwrap(),
        radius,
    );
    value / (MODEL6_METERS_PER_SECOND_TO_KMH as f32)
}

fn model6_length_squared(value: &GmVec3) -> f32 {
    let xy = (value.x * value.x) + (value.y * value.y);
    xy + (value.z * value.z)
}

fn model6_length(value: &GmVec3) -> f32 {
    model6_length_squared(value).sqrt()
}

fn model6_normalize(value: &mut GmVec3) -> bool {
    let length_squared = model6_length_squared(value);
    if !((MODEL6_LENGTH_EPSILON as f64) < (length_squared as f64)) {
        return false;
    }
    let inverse = 1.0f32 / length_squared.sqrt();
    value.x = inverse * value.x;
    value.y = value.y * inverse;
    value.z = inverse * value.z;
    true
}

fn model6_sub_vec3(left: &GmVec3, right: &GmVec3) -> GmVec3 {
    GmVec3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    }
}

fn model6_cross(left: &GmVec3, right: &GmVec3) -> GmVec3 {
    GmVec3 {
        x: (left.y * right.z) - (left.z * right.y),
        y: (left.z * right.x) - (left.x * right.z),
        z: (left.x * right.y) - (left.y * right.x),
    }
}

/* Schedule at 0x007C4B7B: x*x + y*y, then z*z. */
fn model6_dot_xyz(left: &GmVec3, right: &GmVec3) -> f32 {
    let xy = (left.x * right.x) + (left.y * right.y);
    xy + (left.z * right.z)
}

/* Schedule at 0x007C4378: y*y + x*x, then z*z. */
fn model6_dot_yxz(left: &GmVec3, right: &GmVec3) -> f32 {
    let yx = (left.y * right.y) + (left.x * right.x);
    yx + (left.z * right.z)
}

fn model6_wheel_material_values<'a>(
    ctx: &'a VehicleCtx,
    contact_material_id: i32,
) -> &'a [f32; 4] {
    let material_id = contact_material_id as u16;
    let material_index = ctx.tuning.ground_material_indices[material_id as usize];
    &ctx.tuning.ground_materials[material_index as usize].values
}

fn model6_enter_wave(ctx: &VehicleCtx, tick: u32) -> f32 {
    let tuning = &ctx.tuning.model6;
    let elapsed = tick.wrapping_sub(ctx.vehicle.model6.burnout_start_tick);
    let angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);
    let denominator = model6_u32_float(tuning.burnout_enter_ticks * 2u32);

    let angle = angle / denominator;
    ((tuning.burnout_enter_lateral_scale - 1.0f32) * model6_cos(angle)) + 1.0f32
}

fn model6_accel_wave(ctx: &VehicleCtx, tick: u32) -> f32 {
    let tuning = &ctx.tuning.model6;
    let elapsed = tick.wrapping_sub(ctx.vehicle.model6.burnout_start_tick);
    let angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);

    let angle = angle / model6_u32_float(tuning.burnout_enter_ticks);
    ((tuning.burnout_enter_accel_scale - 1.0f32) * model6_sin(angle)) + 1.0f32
}

fn model6_exit_wave(ctx: &VehicleCtx, tick: u32, extra_acceleration: &mut f32) -> f32 {
    let tuning = &ctx.tuning.model6;
    let elapsed = tick.wrapping_sub(ctx.vehicle.model6.burnout_transition_tick);
    let angle = x87_mul_double(model6_u32_float(elapsed), MODEL6_PI);
    let quotient = elapsed / tuning.burnout_exit_ticks;
    let phase_count = model6_u32_float(quotient);

    let angle = angle / model6_u32_float(tuning.burnout_exit_ticks);
    let phase_delta = x87_sub_double(phase_count, 1.0);
    *extra_acceleration =
        (phase_delta * phase_delta) * tuning.burnout_exit_extra_accel;
    ((tuning.burnout_exit_accel_scale - 1.0f32) * model6_sin(angle)) + 1.0f32
}

fn model6_add_material6_forces(
    ctx: &mut VehicleCtx,
    local_speed: &GmVec3,
    side_force: &GmVec3,
) {
    let tuning = &ctx.tuning.model6;
    let mut normalized_speed = *local_speed;
    let mut front_force = GmVec3::ZERO;
    let mut rear_force = GmVec3::ZERO;
    let length_squared = model6_length_squared(&normalized_speed);

    if (MODEL6_LENGTH_EPSILON as f64) < (length_squared as f64) {
        let inverse = 1.0f32 / length_squared.sqrt();
        /* 0x007C50C7 stores only the x component. */
        normalized_speed.x = inverse * normalized_speed.x;
    }
    if (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64) {
        let drag = GmVec3 {
            x: x87_mul_double(local_speed.x, -MODEL6_POINT_ONE),
            y: x87_mul_double(local_speed.y, -MODEL6_POINT_ONE),
            z: x87_mul_double(local_speed.z, -MODEL6_POINT_ONE),
        };
        add_vehicle_central_force(ctx, &drag);
    }
    if ctx.vehicle.flag_60c == 0
        && (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_gas as f64)
        && (ctx.vehicle.input_brake as f64) < MODEL6_POINT_ONE
    {
        let abs_x = model6_abs(normalized_speed.x);
        let speed_shape =
            x87_add_double(x87_mul_double(abs_x, 20.0), 1.0);
        let absolute_forward = x87_add_double(model6_abs(local_speed.z), 1.0);
        let vertical_divisor = absolute_forward * absolute_forward;
        let gas_divisor = x87_add_double(
            tuning.material6_gas_denominator * ctx.vehicle.input_gas,
            1.0,
        );
        let mut front_common = x87_mul_double(
            tuning.material6_vertical_scale * ctx.vehicle.input_gas,
            1.5,
        );
        let mut rear_common =
            tuning.material6_vertical_scale * ctx.vehicle.input_gas;

        front_common = front_common * abs_x;
        front_common = front_common * speed_shape;
        front_common = front_common * tuning.material6_vertical_shape;
        rear_common = ((rear_common * abs_x) * speed_shape)
            * tuning.material6_vertical_shape;

        front_force.x = (tuning.material6_longitudinal_scale * normalized_speed.x)
            / gas_divisor;
        front_force.z = front_common / vertical_divisor;
        rear_force.x = ((normalized_speed.x * -1.0f32)
            * tuning.material6_longitudinal_scale)
            / gas_divisor;
        rear_force.z = rear_common / vertical_divisor;
    }

    let wheel_count = ctx.vehicle.wheels.len();
    for index in 0..wheel_count {
        let (is_sliding, offset_from_vehicle) = {
            let wheel = &ctx.vehicle.wheels[index];
            (wheel.real_time.is_sliding, wheel.offset_from_vehicle)
        };

        if (ctx.vehicle.input_brake as f64) < MODEL6_POINT_ONE
            && is_sliding != 0
        {
            if index == 0 || index == 1 {
                add_vehicle_force(ctx, &front_force, &offset_from_vehicle);
            }
            if index == 2 || index == 3 {
                add_vehicle_force(ctx, &rear_force, &offset_from_vehicle);
            }
        }
    }
    add_vehicle_central_force(ctx, side_force);
}

#[allow(clippy::too_many_arguments)]
fn model6_continue_donut(
    ctx: &mut VehicleCtx,
    ext: &Model6External,
    existing_force: &GmVec3,
    local_speed: &GmVec3,
    local_angular_speed: &GmVec3,
    steering_angle: f32,
    grounded: bool,
    water_contact: bool,
    upright_value: f32,
) {
    let _ = existing_force;
    let _ = upright_value;

    let curves = &ctx.tuning.curves;

    'body: {
        if (ctx.vehicle.input_gas as f64) < MODEL6_POINT_ONE
            || (ctx.vehicle.input_brake as f64) < MODEL6_POINT_ONE
            || (model6_abs(steering_angle) as f64)
                < (MODEL6_CONTROL_EPSILON as f64)
        {
            ctx.vehicle.drive_mode = 0;
            break 'body; /* goto transition */
        }
        if ctx.vehicle.model6.orbit_sign != model6_sign_bits(steering_angle)
            || ctx.vehicle.model6.side_contact != 0
            || 0 < ctx.vehicle.model6.contact_block_count
            || !grounded
            || water_contact
            || ctx.vehicle.flag_60c != 0
        {
            ctx.vehicle.drive_mode = 0;
            break 'body; /* goto transition */
        }

        let mut wheel_normal = GmVec3::ZERO;
        for index in 0..ctx.vehicle.wheels.len() {
            let normal = ctx.vehicle.wheels[index].real_time.field90;
            wheel_normal.x = normal.x + wheel_normal.x;
            wheel_normal.y = normal.y + wheel_normal.y;
            wheel_normal.z = normal.z + wheel_normal.z;
        }
        if !model6_normalize(&mut wheel_normal) {
            wheel_normal = GmVec3 { x: 0.0, y: 1.0, z: 0.0 };
        }
        let mut world_normal = GmVec3::ZERO;
        world_normal.set_mult_mat3(
            &wheel_normal,
            &GmMat3 { m: ctx.vehicle.model6.model_iso.m },
        );
        let normal_angle = model6_abs(gmvec3_get_angle(
            &world_normal,
            &ctx.vehicle.model6.orbit_axis,
        ));
        if (ctx.tuning.model6.donut_normal_angle_limit as f64)
            < (normal_angle as f64)
        {
            ctx.vehicle.drive_mode = 0;
            break 'body; /* goto transition */
        }

        let mut inverse = GmIso4::default();
        inverse.set_inverse(&ctx.vehicle.model6.model_iso);
        let mut local_radius = GmVec3::ZERO;
        local_radius.set_mult_iso4(&ctx.vehicle.model6.orbit_center, &inverse);
        let local_radius = model6_sub_vec3(&local_radius, &ext.body_reference_position);
        let radius = model6_length(&local_radius);
        let mut radial_direction = local_radius;
        let _ = model6_normalize(&mut radial_direction);
        let tangent = model6_cross(&wheel_normal, &radial_direction);
        let lateral_speed = model6_dot_yxz(&tangent, local_speed);

        if (ctx.tuning.model6.donut_lateral_speed_limit as f64)
            < (model6_abs(lateral_speed) as f64)
        {
            ctx.vehicle.drive_mode = 0;
        }
        if (ctx.vehicle.model6.orbit_initial_radius as f64) < (radius as f64)
            || (radius as f64) < (ctx.tuning.model6.donut_radius_min as f64)
        {
            let radius_delta = radius - ctx.vehicle.model6.orbit_radius;
            let radial_speed = model6_dot_yxz(&radial_direction, local_speed);
            let exponent = (radius_delta * ctx.tuning.model6.donut_radius_exponent)
                - (radial_speed * ctx.tuning.model6.donut_radial_speed_exponent);
            let magnitude = ((lateral_speed * lateral_speed) / radius)
                * model6_exp(exponent);

            let force = GmVec3 {
                x: magnitude * radial_direction.x,
                y: magnitude * radial_direction.y,
                z: magnitude * radial_direction.z,
            };
            add_vehicle_central_force(ctx, &force);
        } else {
            ctx.vehicle.drive_mode = 0;
        }

        for index in 0..ctx.vehicle.wheels.len() {
            wheel_add_force_to_vehicle(ctx, index);
            ctx.vehicle.wheels[index].real_time.is_sliding = 0;
        }

        {
            let direction = crate::gm::gmfunc_sign(steering_angle);
            let target = model6_lateral_speed_from_radius(curves, radius);
            let difference = (-direction * target) - lateral_speed;
            let magnitude =
                ctx.tuning.model6.donut_lateral_force_scale * difference;

            let force = GmVec3 {
                x: magnitude * tangent.x,
                y: magnitude * tangent.y,
                z: magnitude * tangent.z,
            };
            add_vehicle_central_force(ctx, &force);

            let forward_axis = GmVec3 { x: 0.0, y: 0.0, z: 1.0 };
            let signed_angle = gmvec3_get_angle(&forward_axis, &radial_direction)
                / (MODEL6_PI as f32);
            if !crate::gm::gmfunc_is_a_number(signed_angle) {
                ctx.vehicle.drive_mode = 0;
            } else {
                let angle_radians = x87_mul_double(
                    direction * signed_angle,
                    MODEL6_PI,
                );

                if (angle_radians as f64)
                    < (-(ctx.tuning.model6.donut_angle_negative_limit) as f64)
                    || (ctx.tuning.model6.donut_angle_positive_limit as f64)
                        < (angle_radians as f64)
                {
                    ctx.vehicle.drive_mode = 0;
                } else {
                    let mut steer_torque;
                    let mut countersteer = 0.0f32;
                    if (signed_angle as f64) <= 0.0 {
                        if (local_angular_speed.y as f64) <= 0.0 {
                            let shifted = signed_angle + 1.0f32;
                            let shifted_squared = shifted * shifted;
                            steer_torque = ((-(ctx.tuning.model6.donut_steer_quadratic)
                                * local_angular_speed.y)
                                * shifted_squared);
                        } else {
                            steer_torque = -(ctx.tuning.model6.donut_steer_linear)
                                * local_angular_speed.y;
                        }
                    } else if 0.0 <= (local_angular_speed.y as f64) {
                        let shifted = signed_angle - 1.0f32;
                        let shifted_squared = shifted * shifted;
                        steer_torque = ((-(ctx.tuning.model6.donut_steer_quadratic)
                            * local_angular_speed.y)
                            * shifted_squared);
                    } else {
                        steer_torque = -(ctx.tuning.model6.donut_steer_linear)
                            * local_angular_speed.y;
                    }
                    let radial_ratio = lateral_speed / radius;
                    if ((signed_angle as f64) <= 0.0 && 0.0 < (radial_ratio as f64))
                        || (0.0 < (signed_angle as f64)
                            && (radial_ratio as f64) < 0.0)
                    {
                        countersteer = -(ctx.tuning.model6.donut_countersteer_scale)
                            * radial_ratio;
                    }
                    let torque = GmVec3 {
                        x: (((ctx.tuning.model6.donut_yaw_angle_scale * signed_angle)
                            + steer_torque)
                            + countersteer)
                            * 0.0f32,
                        y: ((ctx.tuning.model6.donut_yaw_angle_scale * signed_angle)
                            + steer_torque)
                            + countersteer,
                        z: 0.0,
                    };
                    let torque_y = torque.y;
                    let torque = GmVec3 {
                        x: torque.x,
                        y: torque_y,
                        z: torque_y * 0.0f32,
                    };
                    add_vehicle_torque(ctx, &torque);
                }
            }
        }

        {
            let tz = -(crate::gm::gmfunc_sign(local_speed.x))
                * model6_donut_rollover(curves, model6_abs(local_speed.x));
            let torque = GmVec3 { x: tz * 0.0f32, y: tz * 0.0f32, z: tz };
            let torque = GmVec3 { x: torque.x, y: torque.x, z: torque.z };
            add_vehicle_torque(ctx, &torque);
            let torque = GmVec3 {
                x: model6_burnout_rollover(curves, local_speed.z),
                y: 0.0,
                z: 0.0,
            };
            add_vehicle_torque(ctx, &torque);
        }
    }

    /* transition: */
    if ctx.vehicle.drive_mode != 2 {
        let tick = ctx.vehicle.tick_time;

        ctx.vehicle.drive_mode = 1;
        ctx.vehicle.model6.burnout_start_tick = tick;
    }
}

/* 0x007C3E80 */
#[allow(clippy::too_many_arguments)]
pub fn compute_forces_model6(
    ctx: &mut VehicleCtx,
    ext: &Model6External,
    water: &crate::track::TmnfTrackWater,
    model_value: f32,
    existing_force: &GmVec3,
    lateral_force_factor: f32,
    longitudinal_force_factor: f32,
    local_speed: &GmVec3,
    local_angular_speed: &GmVec3,
    steering_angle: f32,
    grounded: bool,
    material: &CSceneVehicleMaterialBlendableVals,
    sliding: &mut i32,
    brake_force: &mut f32,
) {
    let _ = model_value;
    ctx.vehicle.model6.model_iso = ext.model_iso_source;
    let upright_value = ctx.vehicle.model6.model_iso.m[4];
    let water_contact = crate::vehicle::aux::apply_water_forces(ctx, water, existing_force);
    /* 0x007C3ED8 stores car +0x5e4, which is both the air-control lock and
     * the airborne friction gate read at 0x007C3916 by ApplyFrictionForces. */
    ctx.vehicle.air_control_locked = water_contact as i32;
    ctx.vehicle.airborne_friction_gate = water_contact as i32;

    let wheel_count = ctx.vehicle.wheels.len();
    let mut all_material6 = true;
    let mut any_sliding = false;
    let was_sliding = ctx.vehicle.turbo_active != 0;
    let mut jumped_to_finish = false;

    for index in 0..wheel_count {
        let wheel = &ctx.vehicle.wheels[index];
        if wheel.real_time.has_ground_contact == 0
            || wheel.real_time.contact_material_id as u16 != 6u16
        {
            all_material6 = false;
        }
    }

    if ctx.vehicle.drive_mode == 2 {
        model6_continue_donut(
            ctx, ext, existing_force, local_speed, local_angular_speed,
            steering_angle, grounded, water_contact, upright_value,
        );
        jumped_to_finish = true; /* goto finish */
    } else {
        let tick = ctx.vehicle.tick_time;
        if ctx.vehicle.drive_mode == 1 {
            let start = ctx.vehicle.model6.burnout_start_tick;

            if tick < start
                || ctx.tuning.model6.burnout_enter_ticks
                    <= tick.wrapping_sub(start)
            {
                ctx.vehicle.model6.burnout_transition_tick = tick;
                ctx.vehicle.drive_mode = 3;
            } else {
                any_sliding = true;
            }
        }
        if ctx.vehicle.drive_mode == 3 {
            let start = ctx.vehicle.model6.burnout_transition_tick;

            if tick < start
                || ctx.tuning.model6.burnout_exit_ticks <= tick.wrapping_sub(start)
            {
                ctx.vehicle.drive_mode = 0;
                ctx.vehicle.force_wheel_speed = 0;
            } else {
                for index in 0..wheel_count {
                    ctx.vehicle.wheels[index].real_time.is_sliding = 1;
                }
            }
        }

        for index in 0..wheel_count {
            wheel_add_force_to_vehicle(ctx, index);
            let wheel = &ctx.vehicle.wheels[index];
            let has_ground_contact = wheel.real_time.has_ground_contact;
            let contact_material_id = wheel.real_time.contact_material_id;
            if has_ground_contact == 0
                || !((ctx.tuning.model6.lateral_force_scale as f64) > 0.0)
            {
                continue;
            }
            let _ = contact_material_id;
            let field54 = wheel.real_time.field54;
            let field90 = wheel.real_time.field90;
            let damper_absorb = wheel.real_time.damper_absorb;
            let is_sliding_r = wheel.real_time.is_sliding;
            let steerable = wheel.steerable;

            let relative_contact =
                model6_sub_vec3(&field54, &ext.body_reference_position);
            let mut wheel_axis = GmVec3 {
                x: field90.y - (field90.z * 0.0f32),
                y: (field90.z * 0.0f32) - field90.x,
                z: (field90.x * 0.0f32) - (field90.y * 0.0f32),
            };
            let axis_length_squared = model6_length_squared(&wheel_axis);
            if (axis_length_squared as f64) <= (MODEL6_LENGTH_EPSILON as f64) {
                wheel_axis = GmVec3 { x: 1.0, y: 0.0, z: 0.0 };
            } else {
                let inverse = 1.0f32 / axis_length_squared.sqrt();
                wheel_axis.x = inverse * wheel_axis.x;
                wheel_axis.y = inverse * wheel_axis.y;
                wheel_axis.z = inverse * wheel_axis.z;
            }
            if steerable != 0 {
                let cosine = model6_cos(steering_angle);
                let negative_sine = -model6_sin(steering_angle);

                let rotated = GmVec3 {
                    x: (cosine * wheel_axis.x) + (negative_sine * 0.0f32),
                    y: (negative_sine * 0.0f32) + (cosine * wheel_axis.y),
                    z: negative_sine + (cosine * wheel_axis.z),
                };
                wheel_axis = rotated;
            }
            let velocity_on_axis = model6_dot_xyz(local_speed, &wheel_axis);

            {
                let axis_scale = -ctx.tuning.model6.burnout_trigger_scale;
                let mut rollover_axis = GmVec3 {
                    x: axis_scale * ctx.vehicle.model6.rollover_axis.x,
                    y: ctx.vehicle.model6.rollover_axis.y * axis_scale,
                    z: axis_scale * ctx.vehicle.model6.rollover_axis.z,
                };
                let axis_length = model6_length(&rollover_axis);
                if (axis_length as f64)
                    < (ctx.tuning.model6.rollover_axis_min_length as f64)
                {
                    rollover_axis = GmVec3::ZERO;
                }
                let mut torque = model6_cross(&relative_contact, &rollover_axis);
                torque.x = (torque.x * -1.0f32)
                    * ctx.tuning.model6.rollover_torque_x_scale;
                torque.y = 0.0;
                torque.z = (torque.z * -1.0f32)
                    * ctx.tuning.model6.rollover_torque_z_scale;
                add_vehicle_torque(ctx, &torque);
            }
            let mut enter_scale = 1.0f32;
            if ctx.vehicle.drive_mode == 1 {
                let tx = model6_burnout_rollover(&ctx.tuning.curves, local_speed.z);
                let torque = GmVec3 { x: tx, y: 0.0, z: 0.0 };
                add_vehicle_torque(ctx, &torque);
                enter_scale = model6_enter_wave(ctx, tick);
            }

            let damper_modulation = crate::vehicle::curve::cscene_vehicle_car_tuning_m6_get_modulation_from_damper_absorb_val(
                &ctx.tuning.curves, damper_absorb,
            );
            let sliding_scale = if is_sliding_r == 0 {
                1.0f32
            } else {
                ctx.tuning.model6.sliding_lateral_limit_scale
            };
            let braking_scale = if is_sliding_r != 0
                && (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64)
            {
                ctx.tuning.model6.braking_lateral_limit_scale
            } else {
                1.0f32
            };
            let material_values =
                *model6_wheel_material_values(ctx, contact_material_id);
            let max_side = model6_max_side_friction(&ctx.tuning.curves, local_speed.z);
            let maximum_force = (((material_values[3] * lateral_force_factor)
                * max_side)
                * sliding_scale)
                * braking_scale
                * damper_modulation;
            let mut requested_force = ((-(ctx.tuning.model6.lateral_force_scale)
                * 0.5f32)
                * velocity_on_axis)
                * enter_scale;
            if (model6_abs(requested_force) as f64) <= (maximum_force as f64) {
                ctx.vehicle.wheels[index].real_time.is_sliding = 0;
            } else {
                let signed_limit = if (requested_force as f64) <= 0.0 {
                    -maximum_force
                } else {
                    maximum_force
                };

                ctx.vehicle.wheels[index].real_time.is_sliding = 1;
                requested_force = ((1.0f32 - ctx.tuning.model6.lateral_overflow_blend)
                    * signed_limit)
                    + (ctx.tuning.model6.lateral_overflow_blend * requested_force);
            }
            if ctx.vehicle.wheels[index].real_time.is_sliding != 0 {
                *sliding = 1;
            }
            let force = GmVec3 {
                x: requested_force * wheel_axis.x,
                y: requested_force * wheel_axis.y,
                z: wheel_axis.z * requested_force,
            };
            if all_material6 && 6.0 < (local_speed.z as f64) {
                model6_add_material6_forces(ctx, local_speed, &force);
                continue;
            }
            add_vehicle_central_force(ctx, &force);
        }

        if !grounded {
            if ctx.vehicle.drive_mode == 1 {
                ctx.vehicle.model6.burnout_transition_tick = tick;
                ctx.vehicle.drive_mode = 3;
            }
            /* goto store_sliding */
        } else {
            let speed_length = model6_length(local_speed);

            if ctx.vehicle.flag_60c == 0
                && (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64)
                && (ctx.vehicle.input_gas as f64) < MODEL6_POINT_ONE
                && ctx.vehicle.drive_mode == 1
            {
                ctx.vehicle.model6.burnout_transition_tick = tick;
                ctx.vehicle.drive_mode = 3;
            }
            if ctx.vehicle.flag_60c == 0 {
                if (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_gas as f64)
                    && (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64)
                    && (local_speed.z as f64)
                        < (ctx.tuning.model6.burnout_speed_min as f64)
                    && (MODEL6_POINT_SEVEN_FIVE as f64)
                        < (upright_value as f64)
                {
                    ctx.vehicle.drive_mode = 1;
                    ctx.vehicle.force_wheel_speed = 1;
                    ctx.vehicle.model6.burnout_start_tick = tick;
                }
                if (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_gas as f64)
                    && (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64)
                    && (local_speed.z as f64)
                        < (ctx.tuning.model6.burnout_speed_max as f64)
                    && (ctx.tuning.model6.burnout_speed_min as f64)
                        < (local_speed.z as f64)
                    && (MODEL6_CONTROL_EPSILON as f64)
                        <= (model6_abs(steering_angle) as f64)
                {
                    let mut accumulated = GmVec3::ZERO;
                    let direction = model6_sign_bits(steering_angle);

                    ctx.vehicle.drive_mode = 2;
                    ctx.vehicle.model6.orbit_sign = direction;
                    for index in 0..wheel_count {
                        let wheel = &ctx.vehicle.wheels[index];
                        if wheel.real_time.has_ground_contact != 0 {
                            accumulated.x =
                                wheel.real_time.field90.x + accumulated.x;
                            accumulated.y =
                                wheel.real_time.field90.y + accumulated.y;
                            accumulated.z =
                                wheel.real_time.field90.z + accumulated.z;
                        }
                    }
                    if !model6_normalize(&mut accumulated) {
                        ctx.vehicle.drive_mode = 0;
                        accumulated = GmVec3 { x: 0.0, y: 1.0, z: 0.0 };
                    }
                    ctx.vehicle.model6.orbit_axis = accumulated;
                    let mut pivot_delta = model6_sub_vec3(
                        &ctx.vehicle.model6.pivot_position,
                        &ext.body_reference_position,
                    );
                    let pivot_axis = GmVec3 {
                        x: ctx.vehicle.model6.pivot_axis.y * 0.0f32,
                        y: ctx.vehicle.model6.pivot_axis.x * 0.0f32,
                        z: ctx.vehicle.model6.pivot_axis.z,
                    };
                    pivot_delta.x = pivot_axis.x + pivot_delta.x;
                    pivot_delta.y = pivot_axis.y + pivot_delta.y;
                    pivot_delta.z = pivot_axis.z + pivot_delta.z;
                    ctx.vehicle.model6.orbit_initial_radius =
                        model6_length(&pivot_delta);
                    let mut tangent =
                        model6_cross(&ctx.vehicle.model6.orbit_axis, local_speed);
                    tangent.x = direction * tangent.x;
                    tangent.y = direction * tangent.y;
                    tangent.z = direction * tangent.z;
                    let _ = model6_normalize(&mut tangent);
                    ctx.vehicle.model6.orbit_axis
                        .mult_mat3(&GmMat3 { m: ctx.vehicle.model6.model_iso.m });
                    if (ctx.vehicle.model6.orbit_axis.y as f64)
                        < (MODEL6_POINT_SEVEN_FIVE as f64)
                    {
                        ctx.vehicle.drive_mode = 0;
                        ctx.vehicle.model6.orbit_axis =
                            GmVec3 { x: 0.0, y: 1.0, z: 0.0 };
                    }
                    let forward_axis = GmVec3 { x: 0.0, y: 0.0, z: 1.0 };
                    let mut angle =
                        gmvec3_get_angle(&forward_axis, &tangent);
                    angle = direction * angle;
                    if (ctx.tuning.model6.donut_angle_positive_limit as f64)
                        < (angle as f64)
                        || (angle as f64)
                            < (-(ctx.tuning.model6.donut_angle_negative_limit) as f64)
                    {
                        ctx.vehicle.drive_mode = 0;
                    } else {
                        let radius =
                            model6_burnout_radius(&ctx.tuning.curves, local_speed.z);

                        let mut local_orbit_direction = GmVec3::ZERO;
                        local_orbit_direction.set_mult_mat3(
                            &tangent,
                            &GmMat3 { m: ctx.vehicle.model6.model_iso.m },
                        );
                        let mut transformed_reference = GmVec3::ZERO;
                        transformed_reference.set_mult_iso4(
                            &ext.body_reference_position,
                            &ctx.vehicle.model6.model_iso,
                        );
                        let radius =
                            radius + ctx.vehicle.model6.orbit_initial_radius;
                        ctx.vehicle.model6.orbit_radius = radius;
                        ctx.vehicle.model6.orbit_center = GmVec3 {
                            x: transformed_reference.x
                                + (radius * local_orbit_direction.x),
                            y: (radius * local_orbit_direction.y)
                                + transformed_reference.y,
                            z: transformed_reference.z
                                + (radius * local_orbit_direction.z),
                        };
                    }
                    ctx.vehicle.force_wheel_speed =
                        (ctx.vehicle.drive_mode == 2) as i32;
                }
            }

            if ctx.vehicle.drive_mode == 0 {
                if (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_brake as f64)
                    && (local_speed.z as f64)
                        < (ctx.vehicle.model6.reverse_speed_threshold as f64)
                    && (model6_abs(local_speed.x) as f64) < 2.0
                {
                    ctx.vehicle.model6.reverse_mode = 1;
                }
                if (MODEL6_POINT_ONE as f64) < (ctx.vehicle.input_gas as f64)
                    && (0.0 < (local_speed.z as f64)
                        || 2.0 < (model6_abs(local_speed.x) as f64))
                {
                    ctx.vehicle.model6.reverse_mode = 0;
                }
                if (ctx.vehicle.input_gas as f64) < MODEL6_POINT_ONE
                    && (ctx.vehicle.input_brake as f64) < MODEL6_POINT_ONE
                {
                    if 0.0 < (local_speed.z as f64)
                        || (model6_abs(local_speed.z) as f64) < 2.0
                    {
                        ctx.vehicle.model6.reverse_mode = 0;
                    } else {
                        ctx.vehicle.model6.reverse_mode = 1;
                    }
                }
                if 0.0 < (local_speed.z as f64)
                    && ctx.vehicle.turbo_type != crate::vehicle::TMNFVehicleTurboType::None
                {
                    ctx.vehicle.model6.reverse_mode = 0;
                }
            } else {
                ctx.vehicle.model6.reverse_mode = 0;
            }

            {
                let speed_ratio = (local_speed.x * local_speed.x)
                    / x87_add_double(model6_abs(local_speed.z), 1.0);
                let direction = model6_sign_bits(-local_speed.x);
                let ratio =
                    model6_rollover_ratio(&ctx.tuning.curves, speed_ratio);
                let torque = GmVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: ratio * direction,
                };
                add_vehicle_torque(ctx, &torque);
            }

            {
                let mut total_requested = 0.0f32;
                let mut total_limited = 0.0f32;
                for index in 0..wheel_count {
                    let steerable = ctx.vehicle.wheels[index].steerable;
                    let is_sliding_w =
                        ctx.vehicle.wheels[index].real_time.is_sliding;
                    let half_width = x87_mul_double(
                        if steerable != 0 {
                            ctx.vehicle.model6.axle_width
                        } else {
                            -ctx.vehicle.model6.axle_width
                        },
                        0.5,
                    );

                    let offset_speed = GmVec3 {
                        x: (local_angular_speed.y * half_width) + local_speed.x,
                        y: local_speed.y + 0.0f32,
                        z: local_speed.z + 0.0f32,
                    };
                    let steer_modulation;
                    if (speed_length as f64) < (MODEL6_POINT_SEVEN as f64) {
                        steer_modulation = 0.0f32;
                    } else if (speed_length as f64)
                        <= (ctx.tuning.model6.wheel_steer_sine_limit as f64)
                    {
                        let angle = x87_mul_double(
                            speed_length / ctx.tuning.model6.wheel_steer_sine_limit,
                            MODEL6_PI,
                        );
                        steer_modulation =
                            model6_sin(x87_mul_double(angle, 0.5));
                    } else {
                        steer_modulation = 1.0f32;
                    }
                    let maximum = model6_max_side_friction(
                        &ctx.tuning.curves, local_speed.z,
                    ) * material.lateral_grip;
                    let mut requested = (-(ctx.tuning.model6.lateral_force_scale)
                        * 0.5f32)
                        * offset_speed.x;
                    if (maximum as f64) < (model6_abs(requested) as f64) {
                        let absolute_request = model6_abs(requested);
                        let blended = (maximum
                            * (1.0f32 - ctx.tuning.model6.wheel_overflow_blend))
                            + (absolute_request
                                * ctx.tuning.model6.wheel_overflow_blend);

                        total_limited = total_limited + maximum;
                        total_requested = absolute_request + total_requested;
                        requested = model6_sign_bits(requested) * blended;
                        any_sliding = true;
                    }
                    let mut requested =
                        ctx.tuning.model6.wheel_torque_scale * requested;
                    if steerable != 0 {
                        let reverse_direction = if ctx.vehicle.model6.reverse_mode == 0 {
                            1.0f32
                        } else {
                            -1.0f32
                        };
                        let wheel_slide_scale = if is_sliding_w == 0 {
                            1.0f32
                        } else {
                            ctx.tuning.model6.sliding_steer_torque_scale
                        };
                        let drive_torque = model6_steer_drive_torque(
                            &ctx.tuning.curves, local_speed.z,
                        );
                        let correction = (((reverse_direction * steer_modulation)
                            * ctx.vehicle.steering_value)
                            * drive_torque)
                            * wheel_slide_scale;

                        requested = requested - correction;
                    }
                    let torque = GmVec3 {
                        x: (0.0f32 * 0.0f32) - (0.0f32 * half_width),
                        y: (requested * half_width) - (0.0f32 * 0.0f32),
                        z: (0.0f32 * 0.0f32) - (requested * 0.0f32),
                    };
                    add_vehicle_torque(ctx, &torque);
                }

                if any_sliding {
                    ctx.vehicle.model6.last_sliding_tick = tick;
                    if !was_sliding {
                        ctx.vehicle.model6.sliding_start_tick = tick;
                    }
                    ctx.vehicle.model6.sliding_elapsed_ticks = tick
                        .wrapping_sub(ctx.vehicle.model6.sliding_start_tick);
                }

                {
                    let mut traction = 1.0f32;
                    let reverse_bias = if ctx.vehicle.model6.reverse_mode == 0 {
                        0.0f32
                    } else {
                        -1.0f32
                    };
                    let turbo_factor = if ctx.vehicle.turbo_type
                        == crate::vehicle::TMNFVehicleTurboType::None
                    {
                        0.0f32
                    } else {
                        ctx.vehicle.turbo_factor
                    };
                    let direction = if ctx.vehicle.model6.reverse_mode == 0 {
                        1.0f32
                    } else {
                        -1.0f32
                    };
                    let mut brake = 0.0f32;

                    if tick == ctx.vehicle.model6.last_sliding_tick
                        && (MODEL6_CONTROL_EPSILON as f64)
                            < (total_limited as f64)
                    {
                        let mut loss = ((total_requested - total_limited)
                            / total_limited)
                            / ctx.tuning.model6.traction_loss_scale;

                        if (loss as f64) < 0.0 {
                            loss = 0.0f32;
                        } else if 1.0 < (loss as f64) {
                            loss = 1.0f32;
                        }
                        traction = 1.0f32 - loss;
                    }
                    let slipping_acceleration = model6_slipping_accel(
                        &ctx.tuning.curves, local_speed.z,
                    );
                    let normal_acceleration = if ctx.vehicle.model6.reverse_mode == 0 {
                        model6_accel(&ctx.tuning.curves, local_speed.z)
                    } else {
                        model6_rear_gear_accel(&ctx.tuning.curves, local_speed.z)
                    };
                    let acceleration;
                    if ctx.vehicle.engine_mode == 1 {
                        acceleration = 0.0f32;
                    } else {
                        acceleration = ((1.0f32 - traction) * slipping_acceleration)
                            + (normal_acceleration * traction);
                    }
                    let steer_slowdown = (ctx.tuning.model6.steer_slowdown_scale
                        * model6_abs(ctx.vehicle.steering_value))
                        * model6_steer_slowdown(&ctx.tuning.curves, local_speed.z);
                    let mut mode_scale = 1.0f32;
                    let mut mode_extra = 0.0f32;
                    if ctx.vehicle.drive_mode == 1 {
                        mode_scale = model6_accel_wave(ctx, tick);
                    }
                    if ctx.vehicle.drive_mode == 3 {
                        mode_scale = model6_exit_wave(ctx, tick, &mut mode_extra);
                    }
                    /* The C's grouping (lines 1230-1252): the (gas*braking +
                     * rev_bias*braking*brake) sum MULTIPLIES the blended
                     * acceleration — it is the second operand of the x87_mul
                     * opened at 1239, not an addend. */
                    let acceleration = mode_extra
                        + ((mode_scale
                            * ((normal_acceleration * turbo_factor)
                                + (((ctx.vehicle.input_gas * material.braking)
                                    + ((reverse_bias * material.braking)
                                        * ctx.vehicle.input_brake))
                                    * acceleration)))
                            - (steer_slowdown * direction));
                    let acceleration = if water_contact {
                        x87_mul_double(acceleration, 0.5)
                    } else {
                        acceleration
                    };
                    let acceleration = if ctx.vehicle.flag_60c != 0 {
                        if ctx.vehicle.turbo_type
                            == crate::vehicle::TMNFVehicleTurboType::None
                        {
                            normal_acceleration * 0.0f32
                        } else {
                            normal_acceleration * ctx.vehicle.turbo_factor
                        }
                    } else {
                        acceleration
                    };

                    if 0.0 < (local_speed.z as f64) {
                        let mut product = 1.0f32;

                        for index in 0..wheel_count {
                            if ctx.vehicle.wheels[index].real_time.is_sliding != 0 {
                                product = ctx.tuning.model6.sliding_brake_scale
                                    * product;
                            }
                        }
                        brake = (((ctx.tuning.model6.brake_speed_scale
                            * local_speed.z)
                            + ctx.tuning.model6.brake_base)
                            * ctx.vehicle.input_brake)
                            * product;
                        let limit = material.steering
                            * if *sliding != 0 {
                                ctx.tuning.model6.forward_brake_limit_sliding
                            } else {
                                ctx.tuning.model6.forward_brake_limit
                            };
                        if (limit as f64) < (brake as f64) {
                            brake = limit;
                            any_sliding = true;
                            for index in 0..wheel_count {
                                ctx.vehicle.wheels[index].real_time.is_sliding = 1;
                            }
                        }
                    }
                    if (local_speed.z as f64) < 0.0
                        && (MODEL6_POINT_ONE as f64)
                            < (ctx.vehicle.input_gas as f64)
                    {
                        let mut product = 1.0f32;

                        if ctx.vehicle.flag_60c == 0 {
                            let trigger = ((-acceleration
                                * ctx.tuning.model6.burnout_trigger_scale)
                                * local_speed.z);

                            if (ctx.tuning.model6.burnout_trigger_limit as f64)
                                < (trigger as f64)
                                && (MODEL6_POINT_SEVEN_FIVE as f64)
                                    < (upright_value as f64)
                            {
                                ctx.vehicle.model6.burnout_start_tick = tick;
                                ctx.vehicle.drive_mode = 1;
                                ctx.vehicle.force_wheel_speed = 1;
                            }
                        }
                        for index in 0..wheel_count {
                            if ctx.vehicle.wheels[index].real_time.is_sliding != 0 {
                                product = ctx.tuning.model6.sliding_brake_scale
                                    * product;
                            }
                        }
                        brake = ((ctx.tuning.model6.brake_base
                            - (ctx.tuning.model6.brake_speed_scale * local_speed.z))
                            * ctx.vehicle.input_gas)
                            * product;
                        let limit = material.steering
                            * if *sliding != 0 {
                                ctx.tuning.model6.reverse_brake_limit_sliding
                            } else {
                                ctx.tuning.model6.reverse_brake_limit
                            };
                        if (limit as f64) < (brake as f64) {
                            brake = limit;
                            any_sliding = true;
                            for index in 0..wheel_count {
                                ctx.vehicle.wheels[index].real_time.is_sliding = 1;
                            }
                        }
                    }

                    *brake_force = brake;
                    let net = acceleration
                        - (model6_sign_bits(local_speed.z) * brake);
                    let net = if ((ctx.tuning.model6.forward_speed_limit_scale
                        * material.acceleration)
                        as f64)
                        < (local_speed.z as f64)
                    {
                        if 0.0 <= (net as f64) {
                            -ctx.tuning.model6.speed_limit_force
                        } else {
                            net - ctx.tuning.model6.speed_limit_force
                        }
                    } else {
                        net
                    };
                    let net = if (local_speed.z as f64)
                        < ((-(ctx.tuning.model6.reverse_speed_limit_scale
                            * material.acceleration))
                            as f64)
                    {
                        if (net as f64) <= 0.0 {
                            ctx.tuning.model6.speed_limit_force
                        } else {
                            ctx.tuning.model6.speed_limit_force + net
                        }
                    } else {
                        net
                    };
                    let force = GmVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: net * longitudinal_force_factor,
                    };
                    add_vehicle_central_force(ctx, &force);
                    let force = GmVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: (-(ctx.tuning.model6.vertical_force_scale)
                            * existing_force.z)
                            / ctx.tuning.model6.vertical_force_divisor,
                    };
                    add_vehicle_central_force(ctx, &force);
                }
            }
        }
    }

    /* store_sliding: (skipped by the donut path's goto finish) */
    if !jumped_to_finish {
        ctx.vehicle.turbo_active = any_sliding as i32;
    }

    /* finish: */
    ctx.vehicle.engine.reverse = ctx.vehicle.model6.reverse_mode;
    ctx.vehicle.friction_input_selector = ctx.vehicle.model6.reverse_mode;
    ctx.vehicle.current_local_speed = *local_speed;
}
