//! The top-level vehicle force orchestrator (`src/vehicle_compute.c`,
//! 0x007C69E0), transliterated.
//!
//! The C's `TMNFVehicleComputeContext` composition root becomes
//! [`ComputeExternal`]: the world-supplied seams (model iso source, water
//! map, post-force callback, contact-token resolver) plus the two
//! adapter-supplied branch facts. The `CHmsItem` accessors route straight to
//! the single car dyna (`ctx.dyna`), exactly as `hms_item.c` does for a
//! one-corpus item.

use crate::gm::*;
use crate::track::TmnfTrackWater;
use crate::vehicle::aux::{
    compute_air_control, create_fake_contacts, enable_turbo, integrate_vehicle,
    update_turbo,
};
use crate::vehicle::base::gm_spring_float_integrate;
use crate::vehicle::contact::{
    apply_friction_forces, compute_vehicle_ground_material_vals,
    cmw_timer_adapter_get_tick_time, get_slope_adherence, is_all_wheel_ground_contact_id,
    is_ground_contact, is_ground_contact_id,
};
use crate::vehicle::curve::cfunc_keys_real_get_value;
use crate::vehicle::model6::compute_forces_model6;
use crate::vehicle::{Model6External, VehicleCtx, TMNFVehicleTurboType};

/* `0x1.b7cdfcp-34f` */
const COMPUTE_LENGTH_EPSILON: f32 = f32::from_bits(0x2EB7_CDFC);
/* `0x1.4f8b58p-17f` */
const COMPUTE_VALUE_EPSILON: f32 = f32::from_bits(0x374F_8B58);
/* 0x00B3D2A8: float 3.6 promoted to double (0x007C777E, 0x007C7803). */
const KMH_PER_MPS: f64 = f64::from_bits(0x400C_CCCC_C000_0000);

/// World-supplied seams for one force computation (the C's
/// `TMNFVehicleComputeContext` externals).
pub struct ComputeExternal<'a> {
    pub ext: &'a Model6External,
    pub water: &'a TmnfTrackWater,
    /// `context->post_force` — invoked after all forces are accumulated.
    pub post_force: &'a mut dyn FnMut(&mut VehicleCtx),
    /// `context->contact_token` — corpus ref → token for turbo roulette.
    pub contact_token: &'a dyn Fn(u32) -> u32,
    /// The static-body iso resolver (`resolve_body_iso`), used by the
    /// ground-contact-id queries for turbo surface detection.
    pub body_iso: &'a dyn Fn(u32) -> GmIso4,
    pub fake_contacts_active: bool,
    pub water_forces_active: bool,
}

fn vec_length_squared_yxz(value: &GmVec3) -> f32 {
    ((value.y * value.y) + (value.x * value.x)) + (value.z * value.z)
}

fn clamp01(value: f32) -> f32 {
    if value.is_nan() || (value as f64) < 0.0 {
        return 0.0;
    }
    if 1.0 < (value as f64) {
        return 1.0;
    }
    value
}

fn clamp_symmetric(value: f32, limit: f32) -> f32 {
    let lower = -limit;
    if (value as f64) < (lower as f64) {
        return lower;
    }
    if (limit as f64) < (value as f64) {
        return limit;
    }
    value
}

/// `CHmsItem_SetLinearSpeed` for the single-corpus car item.
fn set_vehicle_linear_speed(ctx: &mut VehicleCtx, speed: &GmVec3) {
    ctx.dyna.set_local_linear_speed(speed);
}

/// `CHmsItem_AddImpulse` + the car's impulse accumulator.
fn add_vehicle_impulse(ctx: &mut VehicleCtx, impulse: &GmVec3) {
    ctx.dyna.add_local_impulse(impulse);
    ctx.vehicle.total_impulse_added.x =
        ctx.vehicle.total_impulse_added.x + impulse.x;
    ctx.vehicle.total_impulse_added.y =
        impulse.y + ctx.vehicle.total_impulse_added.y;
    ctx.vehicle.total_impulse_added.z =
        impulse.z + ctx.vehicle.total_impulse_added.z;
}

fn clamp_vehicle_linear_speed(ctx: &mut VehicleCtx, speed: &mut GmVec3) {
    let maximum = ctx.vehicle.compute.local_speed_limit;
    let maximum_squared = maximum * maximum;
    let length_squared = vec_length_squared_yxz(speed);

    if (maximum_squared as f64) < (length_squared as f64)
        && (COMPUTE_LENGTH_EPSILON as f64) < (maximum_squared as f64)
    {
        let length = length_squared.sqrt();
        let scale = maximum / length;

        speed.x = scale * speed.x;
        speed.y = speed.y * scale;
        speed.z = scale * speed.z;
        set_vehicle_linear_speed(ctx, speed);
    }
}

fn compute_steering_angle(ctx: &VehicleCtx, linear_speed: &GmVec3) -> f32 {
    let tuning = &ctx.tuning.aux;
    let denominator = (linear_speed.z.abs() * tuning.steering_speed_scale)
        + tuning.steering_speed_base;
    let angle;

    if (COMPUTE_VALUE_EPSILON as f64) <= (denominator as f64) {
        let reciprocal = 1.0f32 / denominator;
        angle = crate::gm::gmfunc_asin_safe(reciprocal);
    } else {
        angle = 0.0f32;
    }
    -(ctx.vehicle.steering_value) * angle
}

fn scan_wheel_contacts(ctx: &VehicleCtx) -> (bool, bool) {
    let mut any_contact = false;
    let mut any_active_contact = false;
    for wheel in &ctx.vehicle.wheels {
        if wheel.real_time.has_ground_contact != 0 {
            any_contact = true;
            if wheel.active != 0 {
                any_active_contact = true;
            }
        }
    }
    (any_contact, any_active_contact)
}

fn apply_air_effect(
    ctx: &mut VehicleCtx,
    external: &ComputeExternal,
    existing_force: &GmVec3,
    tick: u32,
    grounded: bool,
) {
    let air_effect_threshold = ctx.vehicle.compute.air_effect_threshold;
    if !((COMPUTE_VALUE_EPSILON as f64) < (air_effect_threshold as f64)) {
        return;
    }

    match ctx.vehicle.compute.air_effect_mode {
        1 => {
            if grounded
                && ctx.vehicle.compute.air_impulse_cooldown_tick < tick
            {
                let mut impulse = GmVec3 {
                    x: -existing_force.x,
                    y: -existing_force.y,
                    z: -existing_force.z,
                };
                let length_squared = vec_length_squared_yxz(&impulse);

                if (COMPUTE_LENGTH_EPSILON as f64) < (length_squared as f64) {
                    let length = length_squared.sqrt();
                    let inverse_length = 1.0f32 / length;

                    impulse.x = inverse_length * impulse.x;
                    impulse.y = impulse.y * inverse_length;
                    impulse.z = inverse_length * impulse.z;
                }
                impulse.x = ctx.tuning.compute.air_impulse_scale * impulse.x;
                impulse.y = impulse.y * ctx.tuning.compute.air_impulse_scale;
                impulse.z = ctx.tuning.compute.air_impulse_scale * impulse.z;
                add_vehicle_impulse(ctx, &impulse);
                ctx.vehicle.compute.air_impulse_cooldown_tick = tick + 100;
            }
        }
        2 => {
            ctx.dyna.params.force_field_scale =
                ctx.tuning.compute.special_force_field_scale;
        }
        3 => {
            ctx.vehicle.turbo_type = TMNFVehicleTurboType::Normal;
            ctx.vehicle.turbo_factor = ctx.tuning.compute.normal_turbo_factor;
        }
        _ => {}
    }
}

fn contact_token(external: &ComputeExternal, body: Option<u32>) -> u32 {
    match body {
        None => 0,
        Some(corpus_ref) => (external.contact_token)(corpus_ref),
    }
}

fn update_surface_effects(
    ctx: &mut VehicleCtx,
    external: &ComputeExternal,
    tick: u32,
) {
    if let Some((relative_axis, _body)) =
        is_ground_contact_id(ctx, 7, external.body_iso)
    {
        let factor = ctx.tuning.compute.normal_turbo_factor * relative_axis.z;
        enable_turbo(
            ctx,
            tick,
            ctx.tuning.compute.normal_turbo_duration,
            factor,
            TMNFVehicleTurboType::Normal,
            0,
        );
    }
    if let Some((relative_axis, _body)) =
        is_ground_contact_id(ctx, 26, external.body_iso)
    {
        let factor = ctx.tuning.compute.roulette_turbo_factor * relative_axis.z;
        enable_turbo(
            ctx,
            tick,
            ctx.tuning.compute.roulette_turbo_duration,
            factor,
            TMNFVehicleTurboType::Normal,
            0,
        );
    }
    if let Some((relative_axis, body)) =
        is_ground_contact_id(ctx, 30, external.body_iso)
    {
        let factor = ctx.tuning.compute.normal_turbo_factor * relative_axis.z;
        let token = contact_token(external, body);

        enable_turbo(
            ctx,
            tick,
            ctx.tuning.compute.normal_turbo_duration,
            factor,
            TMNFVehicleTurboType::Roulette,
            token,
        );
    }
    if is_ground_contact_id(ctx, 29, external.body_iso).is_some() {
        ctx.vehicle.flag_60c = 1;
    }
    update_turbo(ctx, tick);
}

fn update_event_levels(ctx: &mut VehicleCtx, tick: u32) {
    let (metric_a, metric_b, metric_c) = (
        ctx.vehicle.compute.event_metric_a,
        ctx.vehicle.compute.event_metric_b,
        ctx.vehicle.compute.event_metric_c,
    );
    let (ab_trigger, ab_level2_min, c_trigger, c_level1_max) = (
        ctx.tuning.compute.event_ab_trigger,
        ctx.tuning.compute.event_ab_level2_min,
        ctx.tuning.compute.event_c_trigger,
        ctx.tuning.compute.event_c_level1_max,
    );

    if (ab_trigger as f64) < (metric_a as f64) {
        if (metric_a as f64) <= (ab_level2_min as f64) {
            if ctx.vehicle.compute.event_level_a == 0 {
                ctx.vehicle.compute.event_level_a = 1;
            }
        } else if ctx.vehicle.compute.event_level_a < 2 {
            ctx.vehicle.compute.event_level_a = 2;
        }
    }
    if (ab_trigger as f64) < (metric_b as f64) {
        if (metric_b as f64) <= (ab_level2_min as f64) {
            if ctx.vehicle.compute.event_level_b == 0 {
                ctx.vehicle.compute.event_level_b = 1;
            }
        } else if ctx.vehicle.compute.event_level_b < 2 {
            ctx.vehicle.compute.event_level_b = 2;
        }
    }
    if (c_trigger as f64) < (metric_c as f64) {
        if (metric_c as f64) <= (c_level1_max as f64) {
            if ctx.vehicle.compute.event_level_c == 0 {
                ctx.vehicle.compute.event_level_c = 1;
            }
        } else if ctx.vehicle.compute.event_level_c < 2 {
            ctx.vehicle.compute.event_level_c = 2;
        }
    }

    if ctx.vehicle.compute.peak_event_level_b < ctx.vehicle.compute.event_level_b {
        ctx.vehicle.compute.peak_event_level_b = ctx.vehicle.compute.event_level_b;
        ctx.vehicle.compute.peak_event_source_ab =
            ctx.vehicle.compute.event_source_ab;
    }
    if ctx.vehicle.compute.peak_event_level_a < ctx.vehicle.compute.event_level_a {
        ctx.vehicle.compute.peak_event_level_a = ctx.vehicle.compute.event_level_a;
        ctx.vehicle.compute.peak_event_source_ab =
            ctx.vehicle.compute.event_source_ab;
    }
    if ctx.vehicle.compute.peak_event_level_c < ctx.vehicle.compute.event_level_c {
        ctx.vehicle.compute.peak_event_level_c = ctx.vehicle.compute.event_level_c;
        ctx.vehicle.compute.peak_event_source_c =
            ctx.vehicle.compute.event_source_c;
    }
    ctx.vehicle.compute.last_force_tick = tick;
}

fn update_force_history(
    ctx: &mut VehicleCtx,
    dt: f32,
    old_force: &GmVec3,
    old_impulse: &GmVec3,
) {
    gm_spring_float_integrate(&mut ctx.vehicle.compute.spring_a, dt);
    let mut history = (old_force.x * dt) + old_impulse.x;
    history = clamp_symmetric(history, ctx.vehicle.compute.history_force_limit);
    let adjustment = (history / ctx.vehicle.compute.history_force_limit)
        * ctx.vehicle.compute.history_force_scale;
    ctx.vehicle.compute.spring_a.value = clamp_symmetric(
        ctx.vehicle.compute.spring_a.value,
        ctx.vehicle.compute.spring_value_limit,
    );
    ctx.vehicle.compute.spring_a.velocity = clamp_symmetric(
        ctx.vehicle.compute.spring_a.velocity + adjustment,
        ctx.vehicle.compute.history_force_scale,
    );

    gm_spring_float_integrate(&mut ctx.vehicle.compute.spring_c, dt);
    let mut history = (-old_force.z * dt) - old_impulse.z;
    history = clamp_symmetric(history, ctx.vehicle.compute.history_force_limit);
    let adjustment = (history / ctx.vehicle.compute.history_force_limit)
        * ctx.vehicle.compute.history_force_scale;
    ctx.vehicle.compute.spring_c.value = clamp_symmetric(
        ctx.vehicle.compute.spring_c.value,
        ctx.vehicle.compute.spring_value_limit,
    );
    ctx.vehicle.compute.spring_c.velocity = clamp_symmetric(
        ctx.vehicle.compute.spring_c.velocity + adjustment,
        ctx.vehicle.compute.history_force_scale,
    );
}

fn update_contact_accumulators(ctx: &mut VehicleCtx, dt: f32, linear_speed: &GmVec3) {
    let direction = if is_all_wheel_ground_contact_id(ctx, 6) {
        1.0f32
    } else {
        -1.0f32
    };
    let speed_kmh =
        (((linear_speed.z as f64) * KMH_PER_MPS) as f32).abs();
    let (rise_curve, decay_curve) = (
        ctx.tuning.compute.contact_rise_curve.as_ref(),
        ctx.tuning.compute.contact_decay_curve.as_ref(),
    );
    let rise_curve = match rise_curve {
        Some(c) => c,
        None => panic!("tmnf: contact rise curve missing"),
    };
    let decay_curve = match decay_curve {
        Some(c) => c,
        None => panic!("tmnf: contact decay curve missing"),
    };

    let mut lower = 0u32;
    let curve_value = cfunc_keys_real_get_value(rise_curve, speed_kmh, &mut lower);
    let delta = (curve_value * dt) * direction;
    ctx.vehicle.compute.contact_rise =
        clamp01(delta + ctx.vehicle.compute.contact_rise);

    let mut lower = 0u32;
    let curve_value = cfunc_keys_real_get_value(decay_curve, speed_kmh, &mut lower);
    let delta = (curve_value * dt) * direction;
    ctx.vehicle.compute.contact_decay =
        clamp01(delta + ctx.vehicle.compute.contact_decay);
}

fn reset_contact_frame(ctx: &mut VehicleCtx) {
    for wheel in &mut ctx.vehicle.wheels {
        wheel.real_time.has_ground_contact = 0;
        wheel.real_time.ground_contact_count = 0;
        wheel.real_time.field54 = GmVec3::ZERO;
        wheel.real_time.field90 = GmVec3::ZERO;
        /* memcpy of a zero u16 into the low half of contact_material_id. */
        wheel.real_time.contact_material_id &= !0xFFFFi32;
        wheel.field15c = 0;
    }
    ctx.vehicle.compute.event_metric_a = 0.0;
    ctx.vehicle.compute.state_5d8 = 0;
    ctx.vehicle.compute.event_metric_b = 0.0;
    ctx.vehicle.air_control_immediate = 0;
    ctx.vehicle.compute.event_metric_c = 0.0;
    ctx.vehicle.side_contact = 0;
    ctx.vehicle.wheel_contact_absorb_count = 0;
    ctx.vehicle.body_contact_count = 0;
    ctx.vehicle.body_contact_position_sum = GmVec3::ZERO;
    ctx.vehicle.body_contact_normal_sum = GmVec3::ZERO;
}

fn finish_active_frame(
    ctx: &mut VehicleCtx,
    dt: f32,
    linear_speed: &GmVec3,
    old_force: &GmVec3,
    old_impulse: &GmVec3,
    effect_curve_position: f32,
) {
    let force = ctx.dyna.get_local_force();
    let inverse_divisor = 1.0f32 / ctx.tuning.compute.normalized_force_divisor;
    ctx.vehicle.compute.normalized_force = GmVec3 {
        x: inverse_divisor * force.x,
        y: force.y * inverse_divisor,
        z: inverse_divisor * force.z,
    };

    let effect_curve = match ctx.tuning.compute.effect_curve.as_ref() {
        Some(c) => c,
        None => panic!("tmnf: effect curve missing"),
    };
    let mut lower = 0u32;
    let curve_value =
        cfunc_keys_real_get_value(effect_curve, effect_curve_position, &mut lower);
    let delta = dt * (curve_value + ctx.tuning.compute.effect_curve_bias);
    ctx.vehicle.compute.effect_accumulator =
        clamp01(ctx.vehicle.compute.effect_accumulator + delta);

    update_force_history(ctx, dt, old_force, old_impulse);
    update_contact_accumulators(ctx, dt, linear_speed);
    reset_contact_frame(ctx);
}

/// 0x007C69E0 — the top force caller for one 10 ms tick.
pub fn cscene_vehicle_car_compute_forces(
    ctx: &mut VehicleCtx,
    external: &mut ComputeExternal,
    dt: f32,
) {
    let old_impulse = ctx.vehicle.total_impulse_added;
    let old_force = ctx.vehicle.total_force_added;
    let integration_flags = ctx.vehicle.integration_flags;

    ctx.vehicle.total_force_added = GmVec3::ZERO;
    ctx.vehicle.total_impulse_added = GmVec3::ZERO;

    if (integration_flags & 0x30) != 0
        || (ctx.vehicle.compute.simulation_gate as f64) < 0.0
    {
        let zero = GmVec3::ZERO;
        ctx.dyna.set_local_linear_speed(&zero);
        ctx.dyna.set_local_angular_speed(&zero);
        ctx.dyna.set_local_force(&zero);
        ctx.dyna.set_local_torque(&zero);
        return;
    }

    if external.fake_contacts_active {
        panic!("tmnf: fake-contacts branch not ported");
    }
    create_fake_contacts(ctx);
    integrate_vehicle(ctx, dt);

    let tick = cmw_timer_adapter_get_tick_time(ctx);
    let grounded = is_ground_contact(ctx);
    ctx.dyna.params.force_field_scale = if grounded {
        ctx.tuning.compute.grounded_force_field_scale
    } else {
        ctx.tuning.compute.airborne_force_field_scale
    };
    ctx.dyna.params.drag_linear = if grounded {
        0.0f32
    } else {
        ctx.tuning.compute.airborne_linear_drag
    };

    if (integration_flags & 2) == 0 {
        return;
    }

    let mut linear_speed = ctx.dyna.get_local_linear_speed();
    if (integration_flags & 8) != 0 {
        linear_speed.x = 0.0;
        linear_speed.z = 0.0;
        set_vehicle_linear_speed(ctx, &linear_speed);
        finish_active_frame(
            ctx, dt, &linear_speed, &old_force, &old_impulse, 0.0,
        );
        return;
    }

    let angular_speed = ctx.dyna.get_local_angular_speed();
    let existing_force = ctx.dyna.get_local_force();
    ctx.vehicle.compute.computed_brake_force = 0.0;
    apply_friction_forces(ctx, &linear_speed);
    clamp_vehicle_linear_speed(ctx, &mut linear_speed);
    let (ground_material, has_ground_material) =
        compute_vehicle_ground_material_vals(ctx);
    let (slope_adherence, slope_secondary) =
        get_slope_adherence(ctx, &existing_force);
    let steering_angle = compute_steering_angle(ctx, &linear_speed);
    ctx.vehicle.current_local_speed = linear_speed;

    if external.water_forces_active {
        panic!("tmnf: water-forces branch not ported");
    }
    /* 0x007C69E0 tests +0x354 for 3, 4, 5 in that order; the else branch is
     * Model 3 (values 0, 1, 2) — the legacy models are not ported. */
    let mut model6_sliding: i32 = 0;
    let mut effect_curve_position: f32 = 0.0;
    if ctx.tuning.contact.friction_model == 5 {
        compute_forces_model6(
            ctx,
            external.ext,
            external.water,
            dt, /* model_value */
            &existing_force,
            slope_adherence,       /* lateral_force_factor */
            slope_secondary,       /* longitudinal_force_factor */
            &linear_speed,
            &angular_speed,
            steering_angle,
            has_ground_material,   /* grounded */
            &ground_material,
            &mut model6_sliding,
            &mut effect_curve_position,
        );
    } else {
        panic!("tmnf: legacy car models (friction_model != 5) are not ported");
    }

    {
        let (any_contact, any_active_contact) = scan_wheel_contacts(ctx);
        if any_contact {
            let braking = -(ctx.vehicle.input_brake)
                * ctx.vehicle.compute.brake_input_scale;
            let grounded_drag = ctx.tuning.compute.grounded_drag_term
                * ctx.vehicle.compute.grounded_drag_scale;

            ctx.vehicle.compute.computed_brake_force =
                braking - grounded_drag;
        }
        if any_active_contact
            && (ctx.tuning.compute.active_contact_stop_threshold as f64) < 0.0
        {
            linear_speed.x = 0.0;
            set_vehicle_linear_speed(ctx, &linear_speed);
        }
        /* 0x007C6ED6..0x007C6F9F: the reset argument is [esp+0x2c], set at
         * 0x007C6F0F when a wheel has contact (+0x124) and is active (+0x0);
         * the Model6 sliding out-parameter is the separate [esp+0x28]. */
        compute_air_control(
            ctx,
            &angular_speed,
            tick,
            grounded,
            any_active_contact,
        );
    }

    apply_air_effect(ctx, external, &existing_force, tick, grounded);
    update_event_levels(ctx, tick);
    update_surface_effects(ctx, external, tick);
    (external.post_force)(ctx);

    finish_active_frame(
        ctx, dt, &linear_speed, &old_force, &old_impulse,
        effect_curve_position,
    );
}

/// 0x007C7D40 — the scene callback form (item identity check is the world's
/// wiring invariant in this port; kept for provenance).
pub fn ccallback_scene_vehicle_car_compute_forces_compute_forces(
    ctx: &mut VehicleCtx,
    external: &mut ComputeExternal,
    dt: f32,
) {
    cscene_vehicle_car_compute_forces(ctx, external, dt);
}
