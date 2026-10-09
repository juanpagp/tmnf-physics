//! Wheel/body contact absorption, friction and ground-material blending
//! (`src/vehicle_contact.c`), transliterated.
//!
//! The C's `TMNFVehicleContactContext` maps to [`VehicleCtx`]; the
//! `resolve_body_iso` function pointer becomes an explicit resolver
//! parameter (`&dyn Fn(u32) -> GmIso4`) on the functions that need it, and
//! the wheel/body tree wiring (`wheel_tree_refs` / `body_tree_refs`) lives
//! in `VehicleTuning`.

use crate::fp::{u32_to_x87_float, x87_cos, x87_sin};
use crate::gm::*;
use crate::response::CHmsPhysicalContact;
use crate::vehicle::base::{add_vehicle_central_force, sdyna_math_compute_impulse};
use crate::vehicle::curve::{
    cscene_vehicle_car_tuning_get_lateral_contact_slow_down_from_speed,
    cscene_vehicle_car_tuning_m4_get_max_friction_force_from_speed,
    cscene_vehicle_car_tuning_m5_get_lateral_contact_slow_down_from_speed,
};
use crate::vehicle::{CSceneVehicleMaterialBlendableVals, VehicleCtx};

/* `0x1.b7cdfcp-34f` */
const CONTACT_LENGTH_EPSILON: f32 = f32::from_bits(0x2EB7_CDFC);
/* `0x1.4f8b58p-17f` */
const FRICTION_SPEED_EPSILON: f32 = f32::from_bits(0x374F_8B58);
/* `0x1.921fb60000000p-1` — float pi/2 promoted to double. */
const CONTACT_ANGLE_RADIANS: f64 = f64::from_bits(0x3FE9_21FB_6000_0000);
/* `0x1.921fb60000000p+1` — float pi promoted to double. */
const TMNF_PI: f64 = f64::from_bits(0x4009_21FB_6000_0000);

fn vec_dot_yxz(left: &GmVec3, right: &GmVec3) -> f32 {
    ((left.y * right.y) + (left.x * right.x)) + (left.z * right.z)
}

fn vec_length_squared_yxz(value: &GmVec3) -> f32 {
    vec_dot_yxz(value, value)
}

fn vec_length_squared_xyz(value: &GmVec3) -> f32 {
    (value.z * value.z) + ((value.x * value.x) + (value.y * value.y))
}

fn dyna_iso(state: &crate::dyna::CHmsStateDyna) -> GmIso4 {
    GmIso4 {
        m: state.rot.m,
        t: [state.pos.x, state.pos.y, state.pos.z],
    }
}

fn slope_curve(ratio: f32, lower: f32, upper: f32) -> f32 {
    let numerator = ratio - lower;
    let denominator = upper - lower;
    let normalized = numerator / denominator;
    let radians = ((normalized as f64) * TMNF_PI) as f32;
    let half_radians = ((radians as f64) * 0.5) as f32;
    let cosine = x87_cos(half_radians);

    1.0f32 - cosine
}

/* 0x0093A4A0, UNVALIDATED: returns the timer's current simulation tick. */
pub fn cmw_timer_adapter_get_tick_time(ctx: &VehicleCtx) -> u32 {
    ctx.vehicle.tick_time
}

/// 0x007BE390, UNVALIDATED: applies a local impulse at a local point and
/// clamps the resulting rigid-body speeds with the selected vehicle tuning.
pub fn add_vehicle_impulse(ctx: &mut VehicleCtx, impulse: &GmVec3, point: &GmVec3) {
    let inverse_mass = 1.0f32 / ctx.dyna.params.mass;
    let (rot, iso) = {
        let state = ctx.dyna.live();
        (state.rot, dyna_iso(state))
    };
    let mut world_impulse = GmVec3::ZERO;
    world_impulse.set_mult_mat3(impulse, &rot);
    let mut world_point = GmVec3::ZERO;
    world_point.set_mult_iso4(point, &iso);
    let linear_delta = GmVec3 {
        x: world_impulse.x * inverse_mass,
        y: world_impulse.y * inverse_mass,
        z: inverse_mass * world_impulse.z,
    };
    let new_linear_speed = {
        let state = ctx.dyna.live();
        GmVec3 {
            x: state.lin_vel.x + linear_delta.x,
            y: state.lin_vel.y + linear_delta.y,
            z: state.lin_vel.z + linear_delta.z,
        }
    };
    let old_speed_squared = {
        let state = ctx.dyna.live();
        vec_length_squared_yxz(&state.lin_vel)
    };
    let new_speed_squared = vec_length_squared_yxz(&new_linear_speed);
    let new_linear_speed = if (old_speed_squared as f64) < (new_speed_squared as f64)
        && (ctx.tuning.contact.max_linear_speed_delta as f64)
            < ((new_speed_squared - old_speed_squared) as f64)
    {
        GmVec3::ZERO
    } else {
        new_linear_speed
    };
    ctx.dyna.live_mut().lin_vel = new_linear_speed;

    let mut world_center = GmVec3::ZERO;
    world_center.set_mult_iso4(&ctx.dyna.params.com_offset, &iso);
    let lever = GmVec3 {
        x: world_point.x - world_center.x,
        y: world_point.y - world_center.y,
        z: world_point.z - world_center.z,
    };
    let mut angular_delta = GmVec3 {
        x: (world_impulse.z * lever.y) - (world_impulse.y * lever.z),
        y: (world_impulse.x * lever.z) - (lever.x * world_impulse.z),
        z: (lever.x * world_impulse.y) - (world_impulse.x * lever.y),
    };
    {
        let state = ctx.dyna.live();
        angular_delta.mult_mat3(&state.inv_inertia_world);
    }
    let angular_delta = GmVec3 {
        x: ctx.tuning.contact.angular_xz_scale * angular_delta.x,
        y: ctx.tuning.contact.angular_xz_scale * angular_delta.y,
        z: ctx.tuning.contact.angular_xz_scale * angular_delta.z,
    };
    let angular_delta = GmVec3 {
        x: angular_delta.x,
        y: ctx.tuning.contact.angular_y_scale * angular_delta.y,
        z: angular_delta.z,
    };
    {
        let state = ctx.dyna.live_mut();
        state.ang_vel.x = state.ang_vel.x + angular_delta.x;
        state.ang_vel.y = state.ang_vel.y + angular_delta.y;
        state.ang_vel.z = state.ang_vel.z + angular_delta.z;
    }

    let angular_speed_squared = {
        let state = ctx.dyna.live();
        vec_length_squared_yxz(&state.ang_vel)
    };
    let max_angular_squared =
        ctx.tuning.contact.max_angular_speed * ctx.tuning.contact.max_angular_speed;
    if (max_angular_squared as f64) < (angular_speed_squared as f64) {
        let angular_speed = angular_speed_squared.sqrt();
        let ratio = ctx.tuning.contact.max_angular_speed / angular_speed;

        let state = ctx.dyna.live_mut();
        state.ang_vel.x = state.ang_vel.x * ratio;
        state.ang_vel.y = state.ang_vel.y * ratio;
        state.ang_vel.z = ratio * state.ang_vel.z;
    }

    ctx.vehicle.total_impulse_added.x =
        impulse.x + ctx.vehicle.total_impulse_added.x;
    ctx.vehicle.total_impulse_added.y =
        impulse.y + ctx.vehicle.total_impulse_added.y;
    ctx.vehicle.total_impulse_added.z =
        impulse.z + ctx.vehicle.total_impulse_added.z;
}

fn apply_wheel_impulse(
    ctx: &mut VehicleCtx,
    relative_speed: &GmVec3,
    normal: &GmVec3,
    point: &GmVec3,
    restitution: f32,
) {
    let params = &ctx.dyna.params;
    let lever = GmVec3 {
        x: point.x - params.com_offset.x,
        y: point.y - params.com_offset.y,
        z: point.z - params.com_offset.z,
    };
    let impulse = sdyna_math_compute_impulse(
        params.mass,
        &params.inv_inertia_body,
        -restitution,
        relative_speed,
        normal,
        &lever,
    );
    add_vehicle_impulse(ctx, &impulse, point);
}

/// 0x007C11D0, UNVALIDATED: classifies a wheel contact, accumulates its
/// contact state, removes accepted replacement, and applies the collision
/// impulse.
pub fn wheel_absorb_contact(
    ctx: &mut VehicleCtx,
    wheel_index: usize,
    contact: &mut CHmsPhysicalContact,
    resolve_body_iso: &dyn Fn(u32) -> GmIso4,
) {
    /* The C's `vehicle_contact_rotation` is the COMMITTED state's rotation
     * (world->committed_state.rot), not the live one. */
    let contact_rotation = ctx.dyna.state_b().rot;
    let ground_threshold = x87_sin(CONTACT_ANGLE_RADIANS as f32);
    let ground_contact =
        (contact.normal.x.abs() as f64) < (ground_threshold as f64);

    ctx.vehicle.wheels[wheel_index].real_time.has_ground_contact =
        ground_contact as i32;
    if !ground_contact {
        ctx.vehicle.side_contact = 1;
        ctx.vehicle.wheels[wheel_index].contact_relative_local_distance =
            contact.position;
        ctx.vehicle.wheels[wheel_index].field15c = 1;
    }
    if ground_contact {
        let real_time = &mut ctx.vehicle.wheels[wheel_index].real_time;
        real_time.ground_contact_count += 1;
        real_time.field90.x = real_time.field90.x + contact.normal.x;
        real_time.field90.y = contact.normal.y + real_time.field90.y;
        real_time.field90.z = contact.normal.z + real_time.field90.z;
        /* set_contact_material_id: u16 store into the low half. */
        real_time.contact_material_id &= !0xFFFFi32;
        real_time.contact_material_id |= contact.other_surface_material as i32;
    }
    contact.accepted = 0;

    if let Some(other_corpus) = contact.other_body_corpus_ref {
        let other_iso = resolve_body_iso(other_corpus);
        let other_up = GmVec3 {
            x: other_iso.m[2],
            y: other_iso.m[5],
            z: other_iso.m[8],
        };
        ctx.vehicle.wheels[wheel_index].real_time.relative_rotz_axis = other_up;
        ctx.vehicle.wheels[wheel_index].contact_body = Some(other_corpus);
        ctx.vehicle.wheels[wheel_index]
            .real_time
            .relative_rotz_axis
            .mult_transpose(&contact_rotation);
    }
    ctx.vehicle.wheels[wheel_index].real_time.field54 = contact.position;

    if ctx.tuning.contact.wheel_contact_model == 2 {
        let mut replacement_y = ((contact.replacement.x * 0.0f32)
            + contact.replacement.y)
            + (0.0f32 * contact.replacement.z);
        let replacement_accepted;

        if !(0.0 < (replacement_y as f64)) {
            replacement_accepted = true;
        } else {
            let mut accepted = false;
            if !((ctx.tuning.contact.damper_max as f64)
                < (-(FRICTION_SPEED_EPSILON) as f64))
            {
                let damper_replacement = ctx.vehicle.wheels[wheel_index]
                    .real_time
                    .damper_absorb
                    - ctx.tuning.contact.damper_max;

                if !((replacement_y as f64) < (damper_replacement as f64)) {
                    accepted = true;
                    replacement_y = damper_replacement;
                }
            }
            replacement_accepted = accepted;
            let contact_replacement = replacement_y;
            if (replacement_y as f64)
                <= (ctx.vehicle.wheels[wheel_index].real_time.field08 as f64)
            {
                replacement_y = ctx.vehicle.wheels[wheel_index].real_time.field08;
            }
            ctx.vehicle.wheels[wheel_index].real_time.field08 = replacement_y;
            {
                let zero_replacement = 0.0f32 * contact_replacement;
                contact.replacement.x =
                    contact.replacement.x - zero_replacement;
                contact.replacement.y =
                    contact.replacement.y - contact_replacement;
                contact.replacement.z =
                    contact.replacement.z - zero_replacement;
            }
        }

        if !ground_contact {
            let restitution = if contact.other_surface_material == 4 {
                ctx.tuning.contact.restitution_air_material4
            } else {
                ctx.tuning.contact.restitution_air
            };
            let normal_speed = vec_dot_yxz(&contact.normal, &contact.relative_speed);

            if (normal_speed as f64) < 0.0 {
                let mut impulse_point = contact.position;

                impulse_point.y = ctx.dyna.params.com_offset.y;
                apply_wheel_impulse(
                    ctx,
                    &contact.relative_speed,
                    &contact.normal,
                    &impulse_point,
                    restitution,
                );
            }
        } else {
            let restitution = if contact.other_surface_material == 4 {
                ctx.tuning.contact.restitution_ground_material4
            } else {
                ctx.tuning.contact.restitution_ground
            };
            let normal_speed = vec_dot_yxz(&contact.normal, &contact.relative_speed);

            if (normal_speed as f64) < 0.0 {
                let projected = GmVec3 {
                    x: contact.normal.x * normal_speed,
                    y: contact.normal.y * normal_speed,
                    z: normal_speed * contact.normal.z,
                };
                let vertical_projection = ((projected.x * 0.0f32) + projected.y)
                    + (projected.z * 0.0f32);
                if replacement_accepted
                    || !((vertical_projection as f64) < 0.0)
                {
                    apply_wheel_impulse(
                        ctx,
                        &contact.relative_speed,
                        &contact.normal,
                        &contact.position,
                        restitution,
                    );
                } else {
                    let horizontal_projection =
                        vertical_projection * 0.0f32;

                    let adjusted_speed = GmVec3 {
                        x: contact.relative_speed.x - horizontal_projection,
                        y: contact.relative_speed.y - vertical_projection,
                        z: contact.relative_speed.z - horizontal_projection,
                    };
                    if (vec_dot_yxz(&contact.normal, &adjusted_speed) as f64) < 0.0 {
                        let impulse_point =
                            ctx.vehicle.wheels[wheel_index].impulse_point;
                        apply_wheel_impulse(
                            ctx,
                            &adjusted_speed,
                            &contact.normal,
                            &impulse_point,
                            restitution,
                        );
                    }
                }
            }
        }
    }
}

fn wheel_index_from_tree(ctx: &VehicleCtx, tree_ref: u32) -> i32 {
    for (i, &r) in ctx.tuning.wheel_tree_refs.iter().enumerate() {
        if r == tree_ref {
            return i as i32;
        }
    }
    for &r in &ctx.tuning.body_tree_refs {
        if r == tree_ref {
            return -1;
        }
    }
    panic!("tmnf: contact tree ref {} is neither wheel nor body", tree_ref);
}

fn absorb_body_contact(ctx: &mut VehicleCtx, contact: &mut CHmsPhysicalContact) {
    'reject: {
        if (contact.normal.y as f64) < -0.75
            && ctx.tuning.contact.friction_model == 5
        {
            let replacement_projection =
                vec_dot_yxz(&contact.normal, &contact.replacement);

            contact.replacement.x = contact.normal.x * replacement_projection;
            contact.replacement.y = contact.normal.y * replacement_projection;
            contact.replacement.z = replacement_projection * contact.normal.z;
        }

        ctx.vehicle.body_contact_position_sum.x = ctx
            .vehicle
            .body_contact_position_sum
            .x
            + contact.position.x;
        ctx.vehicle.body_contact_position_sum.y =
            contact.position.y + ctx.vehicle.body_contact_position_sum.y;
        ctx.vehicle.body_contact_position_sum.z = ctx
            .vehicle
            .body_contact_position_sum
            .z
            + contact.position.z;
        ctx.vehicle.body_contact_normal_sum.x =
            contact.normal.x + ctx.vehicle.body_contact_normal_sum.x;
        ctx.vehicle.body_contact_normal_sum.y =
            contact.normal.y + ctx.vehicle.body_contact_normal_sum.y;
        ctx.vehicle.body_contact_normal_sum.z =
            contact.normal.z + ctx.vehicle.body_contact_normal_sum.z;
        ctx.vehicle.body_contact_count += 1;
        ctx.vehicle.model6.contact_block_count = 1;
        ctx.vehicle.compute.event_source_c = contact.other_surface_material as u8;

        if ctx.tuning.contact.wheel_contact_model != 2 {
            break 'reject;
        }
        let normal_speed = vec_dot_yxz(&contact.normal, &contact.relative_speed);
        if !((normal_speed as f64) < 0.0) {
            break 'reject;
        }

        let (tangent_ratio, restitution) = if contact.other_surface_material == 4 {
            (
                ctx.tuning.contact.body_tangent_ratio_material4,
                ctx.tuning.contact.restitution_air_material4,
            )
        } else {
            (ctx.tuning.contact.body_tangent_ratio, ctx.tuning.contact.restitution_air)
        };
        let normal_projection = GmVec3 {
            x: contact.normal.x * normal_speed,
            y: contact.normal.y * normal_speed,
            z: normal_speed * contact.normal.z,
        };
        let mut tangent = GmVec3 {
            x: contact.relative_speed.x - normal_projection.x,
            y: contact.relative_speed.y - normal_projection.y,
            z: contact.relative_speed.z - normal_projection.z,
        };
        let normal_length = vec_length_squared_xyz(&normal_projection).sqrt();
        let tangent_length = vec_length_squared_xyz(&tangent).sqrt();
        {
            let tangent_limit = normal_length * tangent_ratio;

            if (tangent_limit as f64) < (tangent_length as f64) {
                let scale = tangent_limit / tangent_length;

                tangent.x = scale * tangent.x;
                tangent.y = tangent.y * scale;
                tangent.z = scale * tangent.z;
            }
        }
        let mut impulse_direction = GmVec3 {
            x: -(tangent.x + normal_projection.x),
            y: -(tangent.y + normal_projection.y),
            z: -(tangent.z + normal_projection.z),
        };
        let impulse_length = vec_length_squared_xyz(&impulse_direction).sqrt();
        if !((FRICTION_SPEED_EPSILON as f64) < (impulse_length as f64)) {
            break 'reject;
        }
        {
            let inverse_length = 1.0f32 / impulse_length;

            impulse_direction.x = inverse_length * impulse_direction.x;
            impulse_direction.y = impulse_direction.y * inverse_length;
            impulse_direction.z = inverse_length * impulse_direction.z;
        }
        let params = &ctx.dyna.params;
        let lever = GmVec3 {
            x: contact.position.x - params.com_offset.x,
            y: contact.position.y - params.com_offset.y,
            z: contact.position.z - params.com_offset.z,
        };
        let impulse = sdyna_math_compute_impulse(
            params.mass,
            &params.inv_inertia_body,
            -restitution,
            &contact.relative_speed,
            &impulse_direction,
            &lever,
        );
        add_vehicle_impulse(ctx, &impulse, &contact.position);
    }

    /* reject: */
    contact.accepted = 0;
}

/// 0x007C3410: filters scene contacts, records impact metrics, maps the
/// collision tree to a wheel, and dispatches wheel or body response.
pub fn absorb_contact(
    ctx: &mut VehicleCtx,
    contact: &mut CHmsPhysicalContact,
    resolve_body_iso: &dyn Fn(u32) -> GmIso4,
) {
    if contact.other_surface_material == 13u16
        || contact.other_surface_material == 23u16
    {
        contact.replacement = GmVec3::ZERO;
        contact.accepted = 0;
        return;
    }

    ctx.vehicle.air_control_immediate = 1;
    let wheel_index = wheel_index_from_tree(ctx, contact.tree_ref);
    let impact = vec_dot_yxz(&contact.normal, &contact.relative_speed).abs();
    if wheel_index >= 0 && !((contact.normal.y as f64) < 0.2f64) {
        if ctx.vehicle.wheels[wheel_index as usize].steerable != 0 {
            ctx.vehicle.compute.event_metric_a =
                impact + ctx.vehicle.compute.event_metric_a;
        } else {
            ctx.vehicle.compute.event_metric_b =
                ctx.vehicle.compute.event_metric_b + impact;
        }
    } else {
        ctx.vehicle.compute.event_metric_c =
            ctx.vehicle.compute.event_metric_c + impact;
    }

    if wheel_index < 0 {
        absorb_body_contact(ctx, contact);
        return;
    }
    ctx.vehicle.compute.event_source_ab = contact.other_surface_material as u8;
    wheel_absorb_contact(ctx, wheel_index as usize, contact, resolve_body_iso);
    ctx.vehicle.wheel_contact_absorb_count += 1;
}

/// 0x007BEB40, UNVALIDATED: maps the contact-normal slope to two configured
/// adherence ramps.
pub fn get_slope_adherence(ctx: &VehicleCtx, normal: &GmVec3) -> (f32, f32) {
    let length_squared = vec_length_squared_yxz(normal);

    if (CONTACT_LENGTH_EPSILON as f64) < (length_squared as f64) {
        let length = length_squared.sqrt();
        let mut ratio = ((normal.y / length) as f64).abs() as f32;

        let mut adherence = ratio;
        if (ratio as f64) < (ctx.tuning.contact.slope_adherence_min as f64) {
            adherence = 0.0f32;
        } else if (ctx.tuning.contact.slope_adherence_max as f64)
            < (ratio as f64)
        {
            adherence = 1.0f32;
        } else {
            adherence = slope_curve(
                ratio,
                ctx.tuning.contact.slope_adherence_min,
                ctx.tuning.contact.slope_adherence_max,
            );
        }

        ratio = ((normal.y / length) as f64).abs() as f32;
        let mut secondary = ratio;
        if (ratio as f64) < (ctx.tuning.contact.slope_secondary_min as f64) {
            secondary = 0.0f32;
            return (adherence, secondary);
        }
        if (ctx.tuning.contact.slope_secondary_max as f64) < (ratio as f64) {
            secondary = 1.0f32;
            return (adherence, secondary);
        }
        secondary = slope_curve(
            ratio,
            ctx.tuning.contact.slope_secondary_min,
            ctx.tuning.contact.slope_secondary_max,
        );
        (adherence, secondary)
    } else {
        (0.0, 0.0)
    }
}

/// 0x007BD1E0, UNVALIDATED: reports whether any wheel has ground contact.
pub fn is_ground_contact(ctx: &VehicleCtx) -> bool {
    for wheel in &ctx.vehicle.wheels {
        if wheel.real_time.has_ground_contact != 0 {
            return true;
        }
    }
    false
}

/// 0x007BF5C0, UNVALIDATED: reports whether every contacting wheel has the
/// requested material ID and at least one wheel is contacting.
pub fn is_all_wheel_ground_contact_id(ctx: &VehicleCtx, material_id: u8) -> bool {
    let mut inactive_count = 0u32;

    for wheel in &ctx.vehicle.wheels {
        if wheel.real_time.has_ground_contact == 0 {
            inactive_count += 1;
        } else if wheel.real_time.contact_material_id as u16 != material_id as u16 {
            return false;
        }
    }
    inactive_count < ctx.vehicle.wheels.len() as u32
}

/// 0x007BF620, UNVALIDATED: returns the first contacting wheel with the
/// requested material ID: (relative_axis, contact body corpus ref).
pub fn is_ground_contact_id(
    ctx: &VehicleCtx,
    material_id: u8,
    _resolve_body_iso: &dyn Fn(u32) -> GmIso4,
) -> Option<(GmVec3, Option<u32>)> {
    for wheel in &ctx.vehicle.wheels {
        if wheel.real_time.has_ground_contact != 0
            && wheel.real_time.contact_material_id as u16 == material_id as u16
        {
            return Some((
                wheel.real_time.relative_rotz_axis,
                wheel.contact_body,
            ));
        }
    }
    None
}

/// 0x007BED10, UNVALIDATED: applies central drag and timed lateral contact
/// slowdown forces.
pub fn apply_friction_forces(ctx: &mut VehicleCtx, velocity: &GmVec3) {
    if (ctx.tuning.contact.friction_model == 4
        || ctx.tuning.contact.friction_model == 5)
        && ctx.vehicle.airborne_friction_gate != 0
        && !is_ground_contact(ctx)
    {
        return;
    }

    let selected_input = if ctx.vehicle.friction_input_selector == 0 {
        ctx.vehicle.input_gas
    } else {
        ctx.vehicle.input_brake
    };
    if (selected_input as f64) < (FRICTION_SPEED_EPSILON as f64)
        || ctx.vehicle.flag_60c != 0
    {
        let mut force = *velocity;
        let length_squared = vec_length_squared_yxz(&force);

        if (CONTACT_LENGTH_EPSILON as f64) < (length_squared as f64) {
            let length = length_squared.sqrt();
            let inverse_length = 1.0f32 / length;
            let mut scale;

            force.x = inverse_length * force.x;
            force.y = force.y * inverse_length;
            force.z = inverse_length * force.z;
            scale = -(ctx.tuning.contact.friction_force);
            force.x = scale * force.x;
            force.y = force.y * scale;
            force.z = scale * force.z;
            if ctx.vehicle.flag_60c == 0 {
                let mut extra;

                scale = -(ctx.tuning.contact.extra_friction_force);
                extra = GmVec3 {
                    x: velocity.x * scale,
                    y: velocity.y * scale,
                    z: scale * velocity.z,
                };
                force.x = extra.x + force.x;
                force.y = extra.y + force.y;
                force.z = extra.z + force.z;
            }
            add_vehicle_central_force(ctx, &force);
        }
    }

    if ctx.tuning.contact.friction_model < 4 {
        if ctx.vehicle.side_contact != 0 {
            /* The game stores interpolation mode 1 into the tuning curve
             * here; the step accessor evaluates with mode 1 regardless and
             * nothing else on a model < 4 car reads the field, so the tuning
             * stays read-only (cold). */
            let scale = -(cscene_vehicle_car_tuning_get_lateral_contact_slow_down_from_speed(
                &ctx.tuning.curves,
                velocity.z,
            ));
            let force = GmVec3 {
                x: scale * velocity.x,
                y: velocity.y * scale,
                z: scale * velocity.z,
            };
            add_vehicle_central_force(ctx, &force);
        }
        return;
    }

    {
        let tick = ctx.vehicle.tick_time;

        if ctx.vehicle.side_contact != 0 {
            ctx.vehicle.last_side_contact_tick = tick;
        }
        if ctx.vehicle.last_side_contact_tick <= tick
            && tick.wrapping_sub(ctx.vehicle.last_side_contact_tick)
                < ctx.tuning.contact.lateral_contact_duration_ticks
        {
            let length_squared = vec_length_squared_xyz(velocity);
            let length = length_squared.sqrt();

            if (FRICTION_SPEED_EPSILON as f64) < (length as f64) {
                let inverse_length = 1.0f32 / length;
                let normalized = GmVec3 {
                    x: velocity.x * inverse_length,
                    y: velocity.y * inverse_length,
                    z: inverse_length * velocity.z,
                };
                let scale = -(cscene_vehicle_car_tuning_m5_get_lateral_contact_slow_down_from_speed(
                    &ctx.tuning.curves,
                    length,
                ));
                let force = GmVec3 {
                    x: scale * normalized.x,
                    y: normalized.y * scale,
                    z: scale * normalized.z,
                };
                add_vehicle_central_force(ctx, &force);
            }
        }
    }
}

/// 0x007BF080, UNVALIDATED: computes and clamps the Model 4 lateral friction
/// force for one material blend.
pub fn get_lateral_friction(
    ctx: &VehicleCtx,
    velocity: &GmVec3,
    lateral_axis: &GmVec3,
    material: &CSceneVehicleMaterialBlendableVals,
    load: f32,
    grounded: bool,
) -> (f32, bool) {
    let lateral_speed = vec_dot_yxz(velocity, lateral_axis);
    let absolute_speed = lateral_speed.abs();
    let linear = ctx.tuning.contact.lateral_linear * (-lateral_speed);
    let quadratic = (lateral_speed * absolute_speed)
        * ctx.tuning.contact.lateral_quadratic;
    let requested = linear - quadratic;
    let ground_scale = if !grounded {
        1.0f32
    } else {
        ctx.tuning.contact.lateral_ground_scale
    };
    let maximum = cscene_vehicle_car_tuning_m4_get_max_friction_force_from_speed(
        &ctx.tuning.curves,
        absolute_speed,
    );
    let mut limit = material.lateral_grip * load;

    limit = limit * maximum;
    limit = limit * ground_scale;
    if (limit as f64) < (requested.abs() as f64) {
        let sign = crate::gm::gmfunc_sign(requested);
        return (limit * sign, true);
    }
    (requested, false)
}

/// 0x007C2800, UNVALIDATED: averages the material blend values for
/// contacting wheels. The original routine indexes material selection with
/// wheel zero.
pub fn compute_vehicle_ground_material_vals(
    ctx: &VehicleCtx,
) -> (CSceneVehicleMaterialBlendableVals, bool) {
    let mut output = [0.0f32; 4];
    let mut contact_count = 0u32;
    let mut has_material = false;

    for index in 0..ctx.vehicle.wheels.len() {
        if ctx.vehicle.wheels[index].real_time.has_ground_contact != 0 {
            let material_id =
                ctx.vehicle.wheels[0].real_time.contact_material_id as u16;
            let material_index =
                ctx.tuning.ground_material_indices[material_id as usize];
            let material =
                &ctx.tuning.ground_materials[material_index as usize];

            contact_count += 1;
            output[0] = material.values[0] + output[0];
            output[1] = material.values[1] + output[1];
            output[2] = material.values[2] + output[2];
            output[3] = material.values[3] + output[3];
            has_material = true;
        }
    }
    if contact_count != 0 {
        let float_count = u32_to_x87_float(contact_count);
        let inverse_count = 1.0f32 / float_count;
        output[0] = inverse_count * output[0];
        output[1] = output[1] * inverse_count;
        output[2] = inverse_count * output[2];
        output[3] = inverse_count * output[3];
    }
    (
        CSceneVehicleMaterialBlendableVals {
            acceleration: output[0],
            braking: output[1],
            steering: output[2],
            lateral_grip: output[3],
        },
        has_material,
    )
}
