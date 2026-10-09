//! `vehicle_curve.c` — the curve key interpolation layer, transliterated from
//! `/home/z/my-project/tmnf-physics/src/vehicle_curve.c` (510 lines) plus the
//! declarations of `src/vehicle_curve.h`.
//!
//! Contents (VA addresses kept above each function):
//! * `CFuncKeys_Compile` — bounds compilation (`cfunc_keys_compile`);
//! * `CFuncKeys_GetBoundingIndices` (0x005914C0) — adjacent-key search;
//! * `CFuncKeys_ComputeBlendCoef` (0x00591670) — search + normalized blend;
//! * `CFuncKeysReal_GetRealAt` (0x00585E70) — evaluation with interpolation
//!   mode;
//! * `CFuncKeysReal_GetValueOut` (0x00586200) / `CFuncKeysReal_GetValue`
//!   (0x00586240) — the two cached-index evaluation entry points;
//! * the 21 `CSceneVehicleCarTuning_*` tuning getters (0x007F3BE0 ..
//!   0x007F4130).
//!
//! # Floating-point policy (see `crate::fp` and PORT_NOTES.md)
//!
//! * `x87_mul/add/sub/div` on floats → plain `f32` operators; `fabsf` →
//!   `.abs()`. Operand order preserved verbatim.
//! * Hex-float literals → `from_bits` (Rust has no hex-float syntax); the
//!   original literal sits in the comment. Bits verified against gcc.
//! * **`CURVE_BOUNDARY_EPSILON` is a `double` constant** (`0x1.4f8b588e368f1p-17`,
//!   53-bit significand — *not* an f32 value). `curve_lower_boundary` /
//!   `curve_upper_boundary` are single `double` operations rounded once:
//!   `((p as f64) ± EPS) as f32`. This is *not* plain f32 arithmetic: rounding
//!   the constant itself to f32 gives `0x1.4f8b58p-17f`, one blend-epsilon
//!   step *below* `CURVE_BLEND_EPSILON` = `0x1.4f8b6p-17f`.
//! * **`CURVE_BLEND_EPSILON` is an f32 literal** → `f32::from_bits`; the
//!   comparison `F(magnitude) < F(CURVE_BLEND_EPSILON)` widens two floats to
//!   double, which is exactly a plain f32 compare.
//! * **`METERS_PER_SECOND_TO_KMH` is a `double` constant** (`0x1.ccccccp+1`,
//!   bit-identical to `(double)3.6f32`). Both uses are single `double`
//!   operations rounded once (`x87_r24(F(x) * K)` / `x87_r24(F(x) / K)`), so
//!   they stay as f64 blocks with one `as f32`: the multiplication of two
//!   f32-representable values is exact in f64 (≤ 48 significand bits) and
//!   collapses to the plain f32 product, but the *division* is a genuine
//!   double rounding and must not become `x / 3.6f32`.
//! * `x87_r24(1.0 - F(*blend))`: `1.0` is a double literal (not `F(f32)`),
//!   so this is an f64 subtraction rounded once — kept as
//!   `(1.0 - blend as f64) as f32`, not `1.0f32 - blend` (double rounding).
//!
//! # Deviations from the C (all documented, none observable on any input the
//! game or the golden dump produces)
//!
//! 1. **Backward-search wrap reads out of range in C.** When the backward
//!    search (`forward == false`) exhausts its start index it resets
//!    `search_index = count`, producing the pair accesses
//!    `lower_bounds[count]` / `upper_bounds[count]`. In the C these read the
//!    aliased `upper_bounds[0]` (the bounds live in one `2 * count` malloc
//!    block) and one float past the block (heap garbage). The Rust port has
//!    separate `Vec<f32>` bounds and *panics* on those indices instead of
//!    reading aliased/garbage memory. **No C call site ever passes
//!    `forward == 0`** — every caller in `vehicle_curve.c`, `vehicle_aux.c`,
//!    `vehicle_compute.c` and `vehicle_model6.c` evaluates with forward = 1 —
//!    and the backward rows of the differential dump were verified (with an
//!    instrumented index checker) to never touch index ≥ count.
//! 2. **`attempts > count` break is dead code.** For any position that passes
//!    the early checks (`lower_bounds[0] <= position <= upper_bounds[count-1]`)
//!    a matching pair always exists — pair `(j*-1, j*)` where `j*` is the
//!    smallest index with `position <= upper_bounds[j*]` (if `j* == 0`, either
//!    the wrap pair `(count-1, 0)` matches or the second-smallest member of
//!    that set does) — so the break can never fire. Transliterated anyway.
//! 3. `CFuncKeys_Compile` / `CFuncKeys_CompileInto` / `CFuncKeys_Release`:
//!    the C's fresh-malloc / link-into-shared-table / free triple collapses
//!    into one `cfunc_keys_compile` filling the owned `Vec<f32>` fields (the
//!    values are identical); release is `Drop`. A failed allocation aborts in
//!    both (C `abort()`, Rust's allocation abort); the C's `malloc(0) == NULL`
//!    special case simply yields empty `Vec`s.
//! 4. `positions` / `values` shorter than `count`: the C reads out of bounds;
//!    Rust panics on the index.
//! 5. NULL curve pointers (`Option::None` in the tuning set) panic where the
//!    C would dereference NULL.
//! 6. The C's uninitialized locals (`float blend;` in the two helpers,
//!    `float value;` in `tuning_get_value`) are initialized to `0.0` here;
//!    every path writes them before any read, so the value never escapes.
//!
//! # Verification
//!
//! `tests::golden_replay` replays a curated subset of the differential dump
//! `/home/z/my-project/tool-results/curve_dump.txt` (generated by
//! `curve_dump.c`, which compiles the real `vehicle_curve.c` with the C
//! library's FP flags and prints bit patterns for every function; the full
//! 2665-row dump — including every backward-search row — was replayed
//! bit-exactly during porting). The edge-case tests mirror
//! `tests/vehicle_curve_smoke.c` from the C repo.

use super::{CFuncKeys, CFuncKeysReal, CSceneVehicleCarTuningCurveSet};

/* ===========================================================================
 * Constants (vehicle_curve.c file-local macros)
 * ========================================================================= */

/* DAT_00b36288 and DAT_00b5f068 in the curve functions. */
/* #define CURVE_BOUNDARY_EPSILON 0x1.4f8b588e368f1p-17 */
const CURVE_BOUNDARY_EPSILON: f64 = f64::from_bits(0x3EE4_F8B5_88E3_68F1); /* 1.0000000000000001e-05 */
/* #define CURVE_BLEND_EPSILON 0x1.4f8b6p-17f */
const CURVE_BLEND_EPSILON: f32 = f32::from_bits(0x3727_C5B0); /* 1.00000034e-05 */

/* DAT_00b3d2a8 in the tuning getters. */
/* #define METERS_PER_SECOND_TO_KMH 0x1.ccccccp+1 */
const METERS_PER_SECOND_TO_KMH: f64 = f64::from_bits(0x400C_CCCC_C000_0000); /* 3.5999999046325684 == (double)3.6f32 */

/* ===========================================================================
 * Static helpers (vehicle_curve.c file-local)
 * ========================================================================= */

fn curve_lower_boundary(position: f32) -> f32 {
    /* return x87_r24(F(position) - CURVE_BOUNDARY_EPSILON); */
    ((position as f64) - CURVE_BOUNDARY_EPSILON) as f32
}

fn curve_upper_boundary(position: f32) -> f32 {
    /* return x87_r24(F(position) + CURVE_BOUNDARY_EPSILON); */
    ((position as f64) + CURVE_BOUNDARY_EPSILON) as f32
}

fn tuning_speed_position(speed: f32) -> f32 {
    /* return x87_r24(F(speed) * METERS_PER_SECOND_TO_KMH); */
    ((speed as f64) * METERS_PER_SECOND_TO_KMH) as f32
}

fn tuning_get_value(curve: &CFuncKeysReal, position: f32) -> f32 {
    let mut lower_index: u32 = 0;
    let mut value = 0.0f32;

    cfunc_keys_real_get_value_out(curve, position, &mut value, &mut lower_index);
    value
}

fn tuning_get_step_value(curve: &CFuncKeysReal, position: f32) -> f32 {
    let mut lower_index: u32 = 0;
    let mut value = 0.0f32;

    cfunc_keys_real_get_value_out_interp(curve, position, &mut value, &mut lower_index, 1);
    value
}

/// The C dereferences the curve pointer unconditionally; a `NULL` image entry
/// (an absent curve) panics here instead of faulting.
fn expect_curve<'a>(curve: &'a Option<CFuncKeysReal>, field: &'static str) -> &'a CFuncKeysReal {
    match curve {
        Some(curve) => curve,
        None => panic!("tmnf vehicle curve: {field} is NULL (the C would dereference it)"),
    }
}

/* ===========================================================================
 * CFuncKeys / CFuncKeysReal
 * ========================================================================= */

/// `CFuncKeys_Compile` (+ `CFuncKeys_CompileInto`): precompute
/// `lower_bounds` / `upper_bounds` from `positions` — `positions[i] -/+` the
/// key epsilon under PC=24. The C allocates one `2 * count` float block (or
/// fills the caller's linked storage); here the owned `Vec`s are the storage.
pub fn cfunc_keys_compile(keys: &mut CFuncKeys) {
    let mut lower: Vec<f32> = Vec::with_capacity(keys.count as usize);
    let mut upper: Vec<f32> = Vec::with_capacity(keys.count as usize);
    for i in 0..keys.count {
        lower.push(curve_lower_boundary(keys.positions[i as usize]));
        upper.push(curve_upper_boundary(keys.positions[i as usize]));
    }
    keys.lower_bounds = lower;
    keys.upper_bounds = upper;
}

/*
 * 0x005914C0, UNVALIDATED: locate the adjacent position keys, searching in
 * the direction selected by the caller.
 */
pub fn cfunc_keys_get_bounding_indices(
    keys: &CFuncKeys,
    position: f32,
    lower_index: &mut u32,
    upper_index: &mut u32,
    forward: bool,
) {
    let count = keys.count;
    let mut lower: u32;
    let mut upper: u32;

    if count == 0 {
        *lower_index = u32::MAX;
        *upper_index = u32::MAX;
        return;
    }
    if count == 1 {
        *lower_index = 0;
        *upper_index = 0;
        return;
    }
    let lower_bounds = &keys.lower_bounds;
    let upper_bounds = &keys.upper_bounds;
    if position < lower_bounds[0] {
        *lower_index = 0;
        *upper_index = 0;
        return;
    }

    lower = count - 1;
    if upper_bounds[lower as usize] < position {
        *lower_index = lower;
        *upper_index = lower;
        return;
    }

    /* u32 index arithmetic wraps exactly as the C's does. */
    let mut search_index = *lower_index;
    let mut attempts: u32 = 0;
    if forward {
        if search_index >= count {
            search_index = 0;
        }
        loop {
            lower = search_index;
            search_index = search_index.wrapping_add(1);
            if search_index >= count {
                search_index = 0;
            }
            upper = search_index;
            attempts = attempts.wrapping_add(1);
            if attempts > count {
                break;
            }
            if !(lower_bounds[lower as usize] <= position)
                || !(position <= upper_bounds[upper as usize])
            {
                continue;
            }
            break;
        }
    } else {
        search_index = search_index.wrapping_add(1);
        if (search_index & 0x8000_0000) != 0 {
            search_index = 0;
        }
        loop {
            upper = search_index;
            search_index = search_index.wrapping_sub(1);
            if (search_index & 0x8000_0000) != 0 {
                search_index = count;
            }
            lower = search_index;
            attempts = attempts.wrapping_add(1);
            if attempts > count {
                break;
            }
            if !(lower_bounds[lower as usize] <= position)
                || !(position <= upper_bounds[upper as usize])
            {
                continue;
            }
            break;
        }
    }

    *lower_index = lower;
    *upper_index = upper;
}

/*
 * 0x00591670, UNVALIDATED: locate the adjacent keys and compute the normalized
 * position between them.
 */
pub fn cfunc_keys_compute_blend_coef(
    keys: &CFuncKeys,
    position: f32,
    lower_index: &mut u32,
    upper_index: &mut u32,
    blend: &mut f32,
    forward: bool,
) -> bool {
    cfunc_keys_get_bounding_indices(keys, position, lower_index, upper_index, forward);
    if *lower_index == u32::MAX {
        return false;
    }
    if *lower_index == *upper_index {
        *blend = 0.0;
        return true;
    }

    let lower_position = keys.positions[*lower_index as usize];
    /* difference = x87_sub(self->positions[*upper_index], lower_position); */
    let difference = keys.positions[*upper_index as usize] - lower_position;
    /* magnitude = fabsf(difference); */
    let magnitude = difference.abs();
    /* if (F(magnitude) < F(CURVE_BLEND_EPSILON)) */
    if magnitude < CURVE_BLEND_EPSILON {
        *blend = 0.0;
        return true;
    }

    /* numerator = x87_sub(position, lower_position); */
    let numerator = position - lower_position;
    /* *blend = x87_div(numerator, difference); */
    *blend = numerator / difference;
    true
}

/*
 * 0x00585E70, UNVALIDATED: evaluate a real-valued curve using the selected
 * interpolation mode.
 */
pub fn cfunc_keys_real_get_real_at(
    curve: &CFuncKeysReal,
    position: f32,
    value: &mut f32,
    lower_index: &mut u32,
    upper_index: &mut u32,
    blend: &mut f32,
    interpolation: i32,
    forward: bool,
) {
    if !cfunc_keys_compute_blend_coef(
        &curve.keys,
        position,
        lower_index,
        upper_index,
        blend,
        forward,
    ) {
        *value = 0.0;
        return;
    }
    if interpolation == 1 {
        *value = curve.values[*lower_index as usize];
        return;
    }

    /* lower_weight = x87_r24(1.0 - F(*blend)); */
    let lower_weight = (1.0 - (*blend as f64)) as f32;
    /* lower_value = x87_mul(lower_weight, self->values[*lower_index]); */
    let lower_value = lower_weight * curve.values[*lower_index as usize];
    /* upper_value = x87_mul(*blend, self->values[*upper_index]); */
    let upper_value = *blend * curve.values[*upper_index as usize];
    /* *value = x87_add(lower_value, upper_value); */
    *value = lower_value + upper_value;
}

fn cfunc_keys_real_get_value_out_interp(
    curve: &CFuncKeysReal,
    position: f32,
    value: &mut f32,
    lower_index: &mut u32,
    interpolation: i32,
) {
    let mut upper_index = (*lower_index).wrapping_add(1);
    let mut blend = 0.0f32;

    cfunc_keys_real_get_real_at(
        curve,
        position,
        value,
        lower_index,
        &mut upper_index,
        &mut blend,
        interpolation,
        true,
    );
}

/*
 * 0x00586200, UNVALIDATED: evaluate a curve and update the caller's cached
 * lower index.
 */
pub fn cfunc_keys_real_get_value_out(
    curve: &CFuncKeysReal,
    position: f32,
    value: &mut f32,
    lower_index: &mut u32,
) {
    cfunc_keys_real_get_value_out_interp(curve, position, value, lower_index, curve.interpolation);
}

/*
 * 0x00586240, UNVALIDATED: return a curve value with an optional cached lower
 * index. The C's `lower_index == NULL` case (its own local index, discarded)
 * is reproduced by callers passing a fresh `&mut 0` scratch variable.
 */
pub fn cfunc_keys_real_get_value(curve: &CFuncKeysReal, position: f32, lower_index: &mut u32) -> f32 {
    let mut upper_index = (*lower_index).wrapping_add(1);
    let mut blend = 0.0f32;
    let mut value = 0.0f32;

    cfunc_keys_real_get_real_at(
        curve,
        position,
        &mut value,
        lower_index,
        &mut upper_index,
        &mut blend,
        curve.interpolation,
        true,
    );
    value
}

/* ===========================================================================
 * CSceneVehicleCarTuning curve getters (0x007F3BE0 .. 0x007F4130)
 * ========================================================================= */

/*
 * 0x007F3BE0, UNVALIDATED: evaluate the acceleration curve after converting
 * speed to km/h.
 */
pub fn cscene_vehicle_car_tuning_get_accel_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_step_value(
        expect_curve(&curves.accel_from_speed, "accel_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3C30, UNVALIDATED: evaluate lateral rollover against speed in km/h.
 */
pub fn cscene_vehicle_car_tuning_get_rollover_lateral_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.rollover_lateral_from_speed, "rollover_lateral_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3C70, UNVALIDATED: evaluate maximum side friction against speed in
 * km/h.
 */
pub fn cscene_vehicle_car_tuning_get_max_side_friction_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.max_side_friction_from_speed, "max_side_friction_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3CB0, UNVALIDATED: evaluate lateral contact slowdown against speed in
 * km/h.
 */
pub fn cscene_vehicle_car_tuning_get_lateral_contact_slow_down_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_step_value(
        expect_curve(
            &curves.lateral_contact_slowdown_from_speed,
            "lateral_contact_slowdown_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3D00, UNVALIDATED: evaluate steering slowdown against speed in km/h.
 */
pub fn cscene_vehicle_car_tuning_get_steer_slow_down_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_step_value(
        expect_curve(&curves.steer_slowdown_from_speed, "steer_slowdown_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3D50, UNVALIDATED: evaluate lateral rollover directly against angle.
 */
pub fn cscene_vehicle_car_tuning_get_rollover_lateral_coef_from_angle(
    curves: &CSceneVehicleCarTuningCurveSet,
    angle: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.rollover_lateral_coef_from_angle,
            "rollover_lateral_coef_from_angle",
        ),
        angle,
    )
}

/*
 * 0x007F3D80, UNVALIDATED: evaluate steering drive torque against speed in
 * km/h.
 */
pub fn cscene_vehicle_car_tuning_get_steer_drive_torque_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.steer_drive_torque_from_speed,
            "steer_drive_torque_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3DC0, UNVALIDATED: evaluate the Model 4 steering radius curve.
 */
pub fn cscene_vehicle_car_tuning_m4_get_steer_radius_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.m4_steer_radius_from_speed, "m4_steer_radius_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3E00, UNVALIDATED: evaluate the Model 4 maximum friction curve.
 */
pub fn cscene_vehicle_car_tuning_m4_get_max_friction_force_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m4_max_friction_force_from_speed,
            "m4_max_friction_force_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3E40, UNVALIDATED: evaluate the Model 5 acceleration curve.
 */
pub fn cscene_vehicle_car_tuning_m5_get_accel_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.accel_from_speed, "accel_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3E80, UNVALIDATED: evaluate and scale Model 5 slipping acceleration.
 */
pub fn cscene_vehicle_car_tuning_m5_get_slipping_accel_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    let value = tuning_get_value(
        expect_curve(
            &curves.m5_slipping_accel_from_speed,
            "m5_slipping_accel_from_speed",
        ),
        tuning_speed_position(speed),
    );

    /* return x87_mul(self->m5_slipping_accel_scale, value); */
    curves.m5_slipping_accel_scale * value
}

/*
 * 0x007F3ED0, UNVALIDATED: evaluate Model 5 steering slowdown.
 */
pub fn cscene_vehicle_car_tuning_m5_get_steer_slow_down_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.steer_slowdown_from_speed, "steer_slowdown_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3F10, UNVALIDATED: evaluate Model 5 lateral contact slowdown. The C
 * passes NULL as the cached index: its local index starts at 0 and is
 * discarded — reproduced with a scratch variable.
 */
pub fn cscene_vehicle_car_tuning_m5_get_lateral_contact_slow_down_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    let mut lower_index: u32 = 0;
    cfunc_keys_real_get_value(
        expect_curve(
            &curves.lateral_contact_slowdown_from_speed,
            "lateral_contact_slowdown_from_speed",
        ),
        tuning_speed_position(speed),
        &mut lower_index,
    )
}

/*
 * 0x007F3F40, UNVALIDATED: evaluate water friction against speed in km/h.
 */
pub fn cscene_vehicle_car_tuning_get_water_friction_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(&curves.water_friction_from_speed, "water_friction_from_speed"),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F3F80, UNVALIDATED: normalize damper absorption and evaluate its
 * modulation curve.
 */
pub fn cscene_vehicle_car_tuning_m6_get_modulation_from_damper_absorb_val(
    curves: &CSceneVehicleCarTuningCurveSet,
    damper_absorb: f32,
) -> f32 {
    let position: f32;

    /* if (F(self->damper_min) == F(self->damper_max)) */
    if curves.damper_min == curves.damper_max {
        position = 0.0;
    } else {
        /* float numerator = x87_sub(damper_absorb, self->damper_min); */
        let numerator = damper_absorb - curves.damper_min;
        /* float denominator = x87_sub(self->damper_max, self->damper_min); */
        let denominator = curves.damper_max - curves.damper_min;

        /* position = x87_div(numerator, denominator); */
        position = numerator / denominator;
    }
    tuning_get_value(
        expect_curve(&curves.m6_damper_modulation, "m6_damper_modulation"),
        position,
    )
}

/*
 * 0x007F3FF0, UNVALIDATED: evaluate rear-gear acceleration against speed.
 */
pub fn cscene_vehicle_car_tuning_m6_get_rear_gear_accel_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m6_rear_gear_accel_from_speed,
            "m6_rear_gear_accel_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F4030, UNVALIDATED: evaluate burnout radius against speed.
 */
pub fn cscene_vehicle_car_tuning_m6_get_burnout_radius_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m6_burnout_radius_from_speed,
            "m6_burnout_radius_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F4070, UNVALIDATED: evaluate burnout lateral speed and convert it from
 * km/h to m/s.
 */
pub fn cscene_vehicle_car_tuning_m6_get_lateral_speed_from_burnout_radius(
    curves: &CSceneVehicleCarTuningCurveSet,
    radius: f32,
) -> f32 {
    let value = tuning_get_value(
        expect_curve(
            &curves.m6_lateral_speed_from_burnout_radius,
            "m6_lateral_speed_from_burnout_radius",
        ),
        radius,
    );

    /* return x87_r24(F(value) / METERS_PER_SECOND_TO_KMH); */
    ((value as f64) / METERS_PER_SECOND_TO_KMH) as f32
}

/*
 * 0x007F40B0, UNVALIDATED: evaluate burnout rollover against speed.
 */
pub fn cscene_vehicle_car_tuning_m6_get_burnout_rollover_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m6_burnout_rollover_from_speed,
            "m6_burnout_rollover_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F40F0, UNVALIDATED: evaluate donut rollover against speed.
 */
pub fn cscene_vehicle_car_tuning_m6_get_donut_rollover_from_speed(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m6_donut_rollover_from_speed,
            "m6_donut_rollover_from_speed",
        ),
        tuning_speed_position(speed),
    )
}

/*
 * 0x007F4130, UNVALIDATED: evaluate lateral rollover against speed ratio.
 */
pub fn cscene_vehicle_car_tuning_m6_get_rollover_lateral_from_speed_ratio(
    curves: &CSceneVehicleCarTuningCurveSet,
    speed_ratio: f32,
) -> f32 {
    tuning_get_value(
        expect_curve(
            &curves.m6_rollover_lateral_from_speed_ratio,
            "m6_rollover_lateral_from_speed_ratio",
        ),
        tuning_speed_position(speed_ratio),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_real(positions: &[f32], values: &[f32], interpolation: i32) -> CFuncKeysReal {
        let mut curve = CFuncKeysReal {
            keys: CFuncKeys {
                count: positions.len() as u32,
                positions: positions.to_vec(),
                lower_bounds: Vec::new(),
                upper_bounds: Vec::new(),
            },
            values: values.to_vec(),
            interpolation,
        };
        cfunc_keys_compile(&mut curve.keys);
        curve
    }

    /* =====================================================================
     * Interpolation edge cases (mirrors tests/vehicle_curve_smoke.c and the
     * required single-key / below-first / above-last / exact-hit / linear-vs-
     * step cases)
     * ===================================================================== */

    /// A single-key curve answers its only value for every position.
    #[test]
    fn single_key_curve_answers_its_only_value() {
        let single = mk_real(&[4.0], &[7.0], 0);
        let mut value = 0.0f32;
        for position in [-100.0f32, 4.0, 100.0] {
            let mut index = 0u32;
            cfunc_keys_real_get_value_out(&single, position, &mut value, &mut index);
            assert_eq!(value, 7.0);
            assert_eq!(index, 0);
        }
    }

    /// Below the first key's lower bound the pair collapses to (0, 0), blend
    /// 0, and the first value is returned.
    #[test]
    fn below_first_key_clamps_to_first_value() {
        let linear = mk_real(&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 0);
        let mut lower = 0u32;
        let mut upper = 0u32;
        let mut blend = 1.0f32;
        assert!(cfunc_keys_compute_blend_coef(
            &linear.keys,
            -1.0,
            &mut lower,
            &mut upper,
            &mut blend,
            true
        ));
        assert_eq!((lower, upper), (0, 0));
        assert_eq!(blend, 0.0);
        let mut value = 0.0f32;
        let mut index = 0u32;
        cfunc_keys_real_get_value_out(&linear, -1.0, &mut value, &mut index);
        assert_eq!(value, 5.0);
        assert_eq!(index, 0);
    }

    /// Above the last key's upper bound the pair collapses to (count-1,
    /// count-1) and the last value is returned.
    #[test]
    fn above_last_key_clamps_to_last_value() {
        let linear = mk_real(&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 0);
        let mut lower = 0u32;
        let mut upper = 0u32;
        let mut blend = 1.0f32;
        assert!(cfunc_keys_compute_blend_coef(
            &linear.keys,
            25.0,
            &mut lower,
            &mut upper,
            &mut blend,
            true
        ));
        assert_eq!((lower, upper), (2, 2));
        assert_eq!(blend, 0.0);
        let mut value = 0.0f32;
        let mut index = 0u32;
        cfunc_keys_real_get_value_out(&linear, 25.0, &mut value, &mut index);
        assert_eq!(value, 25.0);
        assert_eq!(index, 2);
    }

    /// An exact hit on a key position resolves to the pair ending at that key
    /// (the search accepts the pair below it first: the epsilon overlap), so
    /// the blend is 0 at the first key and 1 at the others.
    #[test]
    fn exact_key_hit_returns_that_keys_value() {
        let linear = mk_real(&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 0);
        /* (position, expected pair, expected blend, expected value) */
        for (position, pair, blend, expected) in [
            (0.0f32, (0u32, 1u32), 0.0f32, 5.0f32),
            (10.0, (0, 1), 1.0, 15.0),
            (20.0, (1, 2), 1.0, 25.0),
        ] {
            let mut lower = 0u32;
            let mut upper = 0u32;
            let mut computed_blend = -1.0f32;
            assert!(cfunc_keys_compute_blend_coef(
                &linear.keys,
                position,
                &mut lower,
                &mut upper,
                &mut computed_blend,
                true
            ));
            assert_eq!((lower, upper), pair);
            assert_eq!(computed_blend, blend);
            let mut value = 0.0f32;
            let mut index = 0u32;
            cfunc_keys_real_get_value_out(&linear, position, &mut value, &mut index);
            assert_eq!(value, expected);
            assert_eq!(index, pair.0);
        }
    }

    /// Interpolation mode 0 blends linearly; mode 1 steps to the lower key.
    #[test]
    fn linear_vs_step_interpolation() {
        let linear = mk_real(&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 0);
        let step = mk_real(&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 1);

        let mut value = 0.0f32;
        let mut index = 0u32;
        cfunc_keys_real_get_value_out(&linear, 5.0, &mut value, &mut index);
        assert_eq!(value, 10.0); /* 5 + 0.5 * (15 - 5) */
        assert_eq!(index, 0);

        cfunc_keys_real_get_value_out(&linear, 15.0, &mut value, &mut index);
        assert_eq!(value, 20.0); /* 15 + 0.5 * (25 - 15) */
        assert_eq!(index, 1);

        index = 0;
        cfunc_keys_real_get_value_out(&step, 15.0, &mut value, &mut index);
        assert_eq!(value, 15.0); /* the lower key's value */
        assert_eq!(index, 1);
    }

    /// An empty curve reports u32::MAX indices and value 0 (count == 0 path).
    #[test]
    fn empty_curve_returns_zero_and_max_indices() {
        let empty = mk_real(&[], &[], 0);
        let mut value = 9.0f32;
        let mut index = 0u32;
        cfunc_keys_real_get_value_out(&empty, 1.0, &mut value, &mut index);
        assert_eq!(value, 0.0);
        assert_eq!(index, u32::MAX);
        let mut lower = 0u32;
        let mut upper = 0u32;
        let mut blend = -1.0f32;
        assert!(!cfunc_keys_compute_blend_coef(
            &empty.keys,
            1.0,
            &mut lower,
            &mut upper,
            &mut blend,
            true
        ));
        assert_eq!((lower, upper), (u32::MAX, u32::MAX));
    }

    /// The C repo's own smoke test (tests/vehicle_curve_smoke.c).
    #[test]
    fn c_smoke_test_scenarios() {
        let linear = mk_real(&[0.0, 10.0, 20.0], &[0.0, 100.0, 200.0], 0);
        let step = mk_real(&[0.0, 10.0, 20.0], &[0.0, 100.0, 200.0], 1);
        let empty = mk_real(&[], &[], 0);
        let single = mk_real(&[4.0], &[7.0], 0);
        let mut index = 0u32;
        let mut value = 0.0f32;

        cfunc_keys_real_get_value_out(&linear, 5.0, &mut value, &mut index);
        assert_eq!(value, 50.0);
        assert_eq!(index, 0);

        /* index intentionally carried over (0), as in the C test */
        cfunc_keys_real_get_value_out(&linear, 15.0, &mut value, &mut index);
        assert_eq!(value, 150.0);
        assert_eq!(index, 1);

        index = 0;
        cfunc_keys_real_get_value_out(&step, 15.0, &mut value, &mut index);
        assert_eq!(value, 100.0);
        assert_eq!(index, 1);

        index = 0;
        cfunc_keys_real_get_value_out(&empty, 1.0, &mut value, &mut index);
        assert_eq!(value, 0.0);
        assert_eq!(index, u32::MAX);

        index = 0;
        cfunc_keys_real_get_value_out(&single, 100.0, &mut value, &mut index);
        assert_eq!(value, 7.0);
        assert_eq!(index, 0);
    }

    /// The boundary epsilon arithmetic is double-then-round: the compiled
    /// bound of 0.0 is 0x1.4f8b58p-17f, one step below the blend epsilon
    /// 0x1.4f8b6p-17f (0x3727C5AC vs 0x3727C5B0).
    #[test]
    fn compiled_bounds_match_the_c_bit_patterns() {
        let positions = [0.0f32, 1.0, 10.0, 100.0, 0.1, 123456.78, 1e-7];
        let mut keys = CFuncKeys {
            count: positions.len() as u32,
            positions: positions.to_vec(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        };
        cfunc_keys_compile(&mut keys);
        /* golden bits from the C (BNDL/BNDU rows of curve_dump.txt) */
        let golden_lower = [
            0xB727_C5AC, 0x3F7F_FF58, 0x411F_FFF6, 0x42C7_FFFF, 0x3DCC_C78F, 0x47F1_2064,
            0xB726_182D,
        ];
        let golden_upper = [
            0x3727_C5AC, 0x3F80_0054, 0x4120_000A, 0x42C8_0001, 0x3DCC_D20B, 0x47F1_2064,
            0x3729_732B,
        ];
        for i in 0..positions.len() {
            assert_eq!(keys.lower_bounds[i].to_bits(), golden_lower[i], "lower[{i}]");
            assert_eq!(keys.upper_bounds[i].to_bits(), golden_upper[i], "upper[{i}]");
        }
    }

    /// An absent curve (NULL in the C) panics instead of faulting.
    #[test]
    #[should_panic(expected = "tmnf vehicle curve: accel_from_speed is NULL")]
    fn null_curve_getter_panics() {
        let curves = CSceneVehicleCarTuningCurveSet::default();
        cscene_vehicle_car_tuning_get_accel_from_speed(&curves, 10.0);
    }

    /// Bounds shorter than count (uncompiled curve) panic where the C would
    /// dereference its still-NULL bounds pointers.
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn uncompiled_bounds_panic_like_a_null_deref() {
        let keys = CFuncKeys {
            count: 2,
            positions: vec![0.0, 10.0],
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        };
        let mut lower = 0u32;
        let mut upper = 0u32;
        cfunc_keys_get_bounding_indices(&keys, 5.0, &mut lower, &mut upper, true);
    }

    /* =====================================================================
     * Golden replay: bit-exact differential against the C
     * ===================================================================== */

    /// The dump harness's seven curves, compiled exactly as it compiles them.
    fn dump_curves() -> Vec<CFuncKeysReal> {
        /* curve 0: A, linear 3 keys; 1: B, same keys interp=1; 2: C single;
         * 3: D empty; 4: E keys closer than the blend epsilon; 5: F duplicate
         * keys; 6: G descending positions. */
        let defs: [(&[f32], &[f32], i32); 7] = [
            (&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 0),
            (&[0.0, 10.0, 20.0], &[5.0, 15.0, 25.0], 1),
            (&[7.0], &[42.0], 0),
            (&[], &[], 0),
            (&[3.0, 3.000009], &[7.0, 9.0], 0),
            (&[5.0, 5.0], &[2.0, 6.0], 0),
            (&[20.0, 10.0, 0.0], &[1.0, 2.0, 3.0], 0),
        ];
        defs.iter()
            .map(|(positions, values, interpolation)| mk_real(positions, values, *interpolation))
            .collect()
    }

    /// The dump harness's tuning curve set: every field points at curve A
    /// (linear) except the two that exercise their own interpolation.
    fn dump_curve_set(curves: &[CFuncKeysReal]) -> CSceneVehicleCarTuningCurveSet {
        let a = curves[0].clone();
        let b = curves[1].clone();
        CSceneVehicleCarTuningCurveSet {
            accel_from_speed: Some(a.clone()),
            rollover_lateral_from_speed: Some(a.clone()),
            max_side_friction_from_speed: Some(a.clone()),
            lateral_contact_slowdown_from_speed: Some(a.clone()),
            steer_slowdown_from_speed: Some(a.clone()),
            rollover_lateral_coef_from_angle: Some(b.clone()),
            steer_drive_torque_from_speed: Some(a.clone()),
            m4_steer_radius_from_speed: Some(b),
            m4_max_friction_force_from_speed: Some(a.clone()),
            m5_slipping_accel_from_speed: Some(a.clone()),
            m5_slipping_accel_scale: 1.3,
            water_friction_from_speed: Some(a.clone()),
            damper_max: 1.75,
            damper_min: 0.25,
            m6_damper_modulation: Some(a.clone()),
            m6_rear_gear_accel_from_speed: Some(a.clone()),
            m6_rollover_lateral_from_speed_ratio: Some(a.clone()),
            m6_burnout_radius_from_speed: Some(a.clone()),
            m6_lateral_speed_from_burnout_radius: Some(a.clone()),
            m6_donut_rollover_from_speed: Some(a.clone()),
            m6_burnout_rollover_from_speed: Some(a),
        }
    }

    fn bits(field: &str) -> u32 {
        u32::from_str_radix(field, 16).expect("hex field")
    }

    /// Runs one probe described by a dump row and formats the result exactly
    /// like the C harness's printf.
    fn replay_row(curves: &[CFuncKeysReal], set: &CSceneVehicleCarTuningCurveSet, row: &str) -> String {
        let f: Vec<&str> = row.split_whitespace().collect();
        let position_of = |i: usize| f32::from_bits(bits(f[i]));
        match f[0] {
            "CURVE" => {
                let idx: usize = f[1].parse().unwrap();
                let count: u32 = f[2][6..].parse().unwrap();
                let interpolation: i32 = f[3][7..].parse().unwrap();
                assert_eq!(curves[idx].keys.count, count, "curve {idx} count");
                assert_eq!(curves[idx].interpolation, interpolation, "curve {idx} interp");
                row.to_string()
            }
            "KEY" => {
                let idx: usize = f[1].parse().unwrap();
                let i: usize = f[2].parse().unwrap();
                let curve = &curves[idx];
                format!(
                    "KEY {} {} {:08x} {:08x} {:08x} {:08x}",
                    idx,
                    i,
                    curve.keys.positions[i].to_bits(),
                    curve.values[i].to_bits(),
                    curve.keys.lower_bounds[i].to_bits(),
                    curve.keys.upper_bounds[i].to_bits()
                )
            }
            "BI" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                let cached: u32 = f[3].parse().unwrap();
                let forward = f[4] == "1";
                let mut li = cached;
                let mut ui = 0xdead_beefu32;
                cfunc_keys_get_bounding_indices(&curves[c].keys, position, &mut li, &mut ui, forward);
                format!("BI {} {:08x} {} {} {} {}", c, position.to_bits(), cached, forward as i32, li, ui)
            }
            "BC" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                let cached: u32 = f[3].parse().unwrap();
                let forward = f[4] == "1";
                let mut li = cached;
                let mut ui = 0xdead_beefu32;
                let mut blend = -1.0f32;
                let ok = cfunc_keys_compute_blend_coef(
                    &curves[c].keys,
                    position,
                    &mut li,
                    &mut ui,
                    &mut blend,
                    forward,
                );
                format!(
                    "BC {} {:08x} {} {} {} {} {} {:08x}",
                    c,
                    position.to_bits(),
                    cached,
                    forward as i32,
                    ok as i32,
                    li,
                    ui,
                    blend.to_bits()
                )
            }
            "RA" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                let cached: u32 = f[3].parse().unwrap();
                let forward = f[4] == "1";
                let interpolation: i32 = f[5].parse().unwrap();
                let mut li = cached;
                let mut ui = 0xdead_beefu32;
                let mut blend = -1.0f32;
                let mut value = -1.0f32;
                cfunc_keys_real_get_real_at(
                    &curves[c],
                    position,
                    &mut value,
                    &mut li,
                    &mut ui,
                    &mut blend,
                    interpolation,
                    forward,
                );
                format!(
                    "RA {} {:08x} {} {} {} {} {} {:08x} {:08x}",
                    c,
                    position.to_bits(),
                    cached,
                    forward as i32,
                    interpolation,
                    li,
                    ui,
                    blend.to_bits(),
                    value.to_bits()
                )
            }
            "VO" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                let cached: u32 = f[3].parse().unwrap();
                let mut li = cached;
                let mut value = -1.0f32;
                cfunc_keys_real_get_value_out(&curves[c], position, &mut value, &mut li);
                format!("VO {} {:08x} {} {} {:08x}", c, position.to_bits(), cached, li, value.to_bits())
            }
            "GV" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                let cached: u32 = f[3].parse().unwrap();
                let mut li = cached;
                let value = cfunc_keys_real_get_value(&curves[c], position, &mut li);
                format!("GV {} {:08x} {} {} {:08x}", c, position.to_bits(), cached, li, value.to_bits())
            }
            "GN" => {
                let c: usize = f[1].parse().unwrap();
                let position = position_of(2);
                /* the C passes NULL: a discarded local index starting at 0 */
                let mut scratch = 0u32;
                let value = cfunc_keys_real_get_value(&curves[c], position, &mut scratch);
                format!("GN {} {:08x} {:08x}", c, position.to_bits(), value.to_bits())
            }
            "BNDL" => {
                let position = position_of(1);
                format!("BNDL {:08x} {:08x}", position.to_bits(), curve_lower_boundary(position).to_bits())
            }
            "BNDU" => {
                let position = position_of(1);
                format!("BNDU {:08x} {:08x}", position.to_bits(), curve_upper_boundary(position).to_bits())
            }
            "SPD" => {
                let position = position_of(1);
                format!("SPD {:08x} {:08x}", position.to_bits(), tuning_speed_position(position).to_bits())
            }
            "MS" => {
                let position = position_of(1);
                let value = ((position as f64) / METERS_PER_SECOND_TO_KMH) as f32;
                format!("MS {:08x} {:08x}", position.to_bits(), value.to_bits())
            }
            "LW" => {
                let blend = position_of(1);
                let lower_weight = (1.0 - (blend as f64)) as f32;
                format!("LW {:08x} {:08x}", blend.to_bits(), lower_weight.to_bits())
            }
            "TG" => {
                let id: i32 = f[1].parse().unwrap();
                let arg = position_of(2);
                let value = match id {
                    1 => cscene_vehicle_car_tuning_get_accel_from_speed(set, arg),
                    2 => cscene_vehicle_car_tuning_get_rollover_lateral_from_speed(set, arg),
                    3 => cscene_vehicle_car_tuning_get_max_side_friction_from_speed(set, arg),
                    4 => cscene_vehicle_car_tuning_get_lateral_contact_slow_down_from_speed(set, arg),
                    5 => cscene_vehicle_car_tuning_get_steer_slow_down_from_speed(set, arg),
                    6 => cscene_vehicle_car_tuning_get_rollover_lateral_coef_from_angle(set, arg),
                    7 => cscene_vehicle_car_tuning_get_steer_drive_torque_from_speed(set, arg),
                    8 => cscene_vehicle_car_tuning_m4_get_steer_radius_from_speed(set, arg),
                    9 => cscene_vehicle_car_tuning_m4_get_max_friction_force_from_speed(set, arg),
                    10 => cscene_vehicle_car_tuning_m5_get_accel_from_speed(set, arg),
                    11 => cscene_vehicle_car_tuning_m5_get_slipping_accel_from_speed(set, arg),
                    12 => cscene_vehicle_car_tuning_m5_get_steer_slow_down_from_speed(set, arg),
                    13 => cscene_vehicle_car_tuning_m5_get_lateral_contact_slow_down_from_speed(set, arg),
                    14 => cscene_vehicle_car_tuning_get_water_friction_from_speed(set, arg),
                    15 => cscene_vehicle_car_tuning_m6_get_modulation_from_damper_absorb_val(set, arg),
                    16 => cscene_vehicle_car_tuning_m6_get_rear_gear_accel_from_speed(set, arg),
                    17 => cscene_vehicle_car_tuning_m6_get_burnout_radius_from_speed(set, arg),
                    18 => cscene_vehicle_car_tuning_m6_get_lateral_speed_from_burnout_radius(set, arg),
                    19 => cscene_vehicle_car_tuning_m6_get_burnout_rollover_from_speed(set, arg),
                    20 => cscene_vehicle_car_tuning_m6_get_donut_rollover_from_speed(set, arg),
                    21 => cscene_vehicle_car_tuning_m6_get_rollover_lateral_from_speed_ratio(set, arg),
                    _ => unreachable!("TG id {id}"),
                };
                format!("TG {} {:08x} {:08x}", id, arg.to_bits(), value.to_bits())
            }
            "CONST" => {
                let value = match f[1] {
                    "blend_eps" => format!("{:08x}", CURVE_BLEND_EPSILON.to_bits()),
                    "bound_eps" => format!("{:016x}", CURVE_BOUNDARY_EPSILON.to_bits()),
                    "kmh_const" => format!("{:016x}", METERS_PER_SECOND_TO_KMH.to_bits()),
                    other => unreachable!("unknown CONST {other}"),
                };
                format!("CONST {} {}", f[1], value)
            }
            other => unreachable!("unknown row tag {other}"),
        }
    }

    fn replay(golden: &str) {
        let curves = dump_curves();
        let set = dump_curve_set(&curves);
        let mut failures: Vec<(String, String)> = Vec::new();
        let mut replayed = 0usize;
        for row in golden.lines() {
            let row = row.trim_end();
            if row.is_empty() {
                continue;
            }
            let produced = replay_row(&curves, &set, row);
            replayed += 1;
            if produced != row {
                failures.push((row.to_string(), produced));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} golden rows mismatched; first failures:\n{}",
            failures.len(),
            replayed,
            failures
                .iter()
                .take(8)
                .map(|(expected, produced)| format!("expected: {expected}\nproduced: {produced}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Curated subset of the differential dump (tool-results/curve_dump.txt, generated
    /// from the real vehicle_curve.c): every public function, every search direction,
    /// both interpolation modes, the cached-index variants, and all 21 tuning getters.
    const GOLDEN: &str = r#"
CURVE 0 count=3 interp=0
KEY 0 0 00000000 40a00000 b727c5ac 3727c5ac
KEY 0 1 41200000 41700000 411ffff6 4120000a
KEY 0 2 41a00000 41c80000 419ffffb 41a00005
CURVE 1 count=3 interp=1
KEY 1 0 00000000 40a00000 b727c5ac 3727c5ac
KEY 1 1 41200000 41700000 411ffff6 4120000a
KEY 1 2 41a00000 41c80000 419ffffb 41a00005
CURVE 2 count=1 interp=0
KEY 2 0 40e00000 42280000 40dfffeb 40e00015
CURVE 3 count=0 interp=0
CURVE 4 count=2 interp=0
KEY 4 0 40400000 40e00000 403fffd6 4040002a
KEY 4 1 40400026 41100000 403ffffc 40400050
CURVE 5 count=2 interp=0
KEY 5 0 40a00000 40000000 409fffeb 40a00015
KEY 5 1 40a00000 40c00000 409fffeb 40a00015
CURVE 6 count=3 interp=0
KEY 6 0 41a00000 3f800000 419ffffb 41a00005
KEY 6 1 41200000 40000000 411ffff6 4120000a
KEY 6 2 00000000 40400000 b727c5ac 3727c5ac
BI 0 bf800000 0 1 0 0
BC 0 bf800000 0 1 1 0 0 00000000
RA 0 bf800000 0 1 0 0 0 00000000 40a00000
BI 0 bf800000 0 1 0 0
BC 0 bf800000 0 1 1 0 0 00000000
RA 0 bf800000 0 1 1 0 0 00000000 40a00000
BI 0 bf800000 2 1 0 0
BC 0 bf800000 2 1 1 0 0 00000000
RA 0 bf800000 2 1 0 0 0 00000000 40a00000
BI 0 bf800000 2 1 0 0
BC 0 bf800000 2 1 1 0 0 00000000
RA 0 bf800000 2 1 1 0 0 00000000 40a00000
BI 0 bf800000 4294967295 1 0 0
BC 0 bf800000 4294967295 1 1 0 0 00000000
RA 0 bf800000 4294967295 1 0 0 0 00000000 40a00000
BI 0 bf800000 4294967295 1 0 0
BC 0 bf800000 4294967295 1 1 0 0 00000000
RA 0 bf800000 4294967295 1 1 0 0 00000000 40a00000
VO 0 bf800000 0 0 40a00000
GV 0 bf800000 0 0 40a00000
GN 0 bf800000 40a00000
GN 0 bf800000 40a00000
VO 0 bf800000 4294967295 0 40a00000
GV 0 bf800000 4294967295 0 40a00000
GN 0 bf800000 40a00000
BI 0 00000000 0 1 0 1
BC 0 00000000 0 1 1 0 1 00000000
RA 0 00000000 0 1 0 0 1 00000000 40a00000
BI 0 00000000 0 1 0 1
BC 0 00000000 0 1 1 0 1 00000000
RA 0 00000000 0 1 1 0 1 00000000 40a00000
BI 0 00000000 2 1 0 1
BC 0 00000000 2 1 1 0 1 00000000
RA 0 00000000 2 1 0 0 1 00000000 40a00000
BI 0 00000000 2 1 0 1
BC 0 00000000 2 1 1 0 1 00000000
RA 0 00000000 2 1 1 0 1 00000000 40a00000
BI 0 00000000 4294967295 1 0 1
BC 0 00000000 4294967295 1 1 0 1 00000000
RA 0 00000000 4294967295 1 0 0 1 00000000 40a00000
BI 0 00000000 4294967295 1 0 1
BC 0 00000000 4294967295 1 1 0 1 00000000
RA 0 00000000 4294967295 1 1 0 1 00000000 40a00000
VO 0 00000000 0 0 40a00000
GV 0 00000000 0 0 40a00000
GN 0 00000000 40a00000
GN 0 00000000 40a00000
VO 0 00000000 4294967295 0 40a00000
GV 0 00000000 4294967295 0 40a00000
GN 0 00000000 40a00000
BI 0 40a00000 0 1 0 1
BC 0 40a00000 0 1 1 0 1 3f000000
RA 0 40a00000 0 1 0 0 1 3f000000 41200000
BI 0 40a00000 0 1 0 1
BC 0 40a00000 0 1 1 0 1 3f000000
RA 0 40a00000 0 1 1 0 1 3f000000 40a00000
BI 0 40a00000 2 1 0 1
BC 0 40a00000 2 1 1 0 1 3f000000
RA 0 40a00000 2 1 0 0 1 3f000000 41200000
BI 0 40a00000 2 1 0 1
BC 0 40a00000 2 1 1 0 1 3f000000
RA 0 40a00000 2 1 1 0 1 3f000000 40a00000
BI 0 40a00000 4294967295 1 0 1
BC 0 40a00000 4294967295 1 1 0 1 3f000000
RA 0 40a00000 4294967295 1 0 0 1 3f000000 41200000
BI 0 40a00000 4294967295 1 0 1
BC 0 40a00000 4294967295 1 1 0 1 3f000000
RA 0 40a00000 4294967295 1 1 0 1 3f000000 40a00000
VO 0 40a00000 0 0 41200000
GV 0 40a00000 0 0 41200000
GN 0 40a00000 41200000
GN 0 40a00000 41200000
VO 0 40a00000 4294967295 0 41200000
GV 0 40a00000 4294967295 0 41200000
GN 0 40a00000 41200000
BI 0 411ffff6 0 1 0 1
BC 0 411ffff6 0 1 1 0 1 3f7ffff0
RA 0 411ffff6 0 1 0 0 1 3f7ffff0 416ffff6
BI 0 411ffff6 0 1 0 1
BC 0 411ffff6 0 1 1 0 1 3f7ffff0
RA 0 411ffff6 0 1 1 0 1 3f7ffff0 40a00000
BI 0 411ffff6 2 1 0 1
BC 0 411ffff6 2 1 1 0 1 3f7ffff0
RA 0 411ffff6 2 1 0 0 1 3f7ffff0 416ffff6
BI 0 411ffff6 2 1 0 1
BC 0 411ffff6 2 1 1 0 1 3f7ffff0
RA 0 411ffff6 2 1 1 0 1 3f7ffff0 40a00000
BI 0 411ffff6 4294967295 1 0 1
BC 0 411ffff6 4294967295 1 1 0 1 3f7ffff0
RA 0 411ffff6 4294967295 1 0 0 1 3f7ffff0 416ffff6
BI 0 411ffff6 4294967295 1 0 1
BC 0 411ffff6 4294967295 1 1 0 1 3f7ffff0
RA 0 411ffff6 4294967295 1 1 0 1 3f7ffff0 40a00000
VO 0 411ffff6 0 0 416ffff6
GV 0 411ffff6 0 0 416ffff6
GN 0 411ffff6 416ffff6
GN 0 411ffff6 416ffff6
VO 0 411ffff6 4294967295 0 416ffff6
GV 0 411ffff6 4294967295 0 416ffff6
GN 0 411ffff6 416ffff6
BI 0 41200000 0 1 0 1
BC 0 41200000 0 1 1 0 1 3f800000
RA 0 41200000 0 1 0 0 1 3f800000 41700000
BI 0 41200000 0 1 0 1
BC 0 41200000 0 1 1 0 1 3f800000
RA 0 41200000 0 1 1 0 1 3f800000 40a00000
BI 0 41200000 2 1 0 1
BC 0 41200000 2 1 1 0 1 3f800000
RA 0 41200000 2 1 0 0 1 3f800000 41700000
BI 0 41200000 2 1 0 1
BC 0 41200000 2 1 1 0 1 3f800000
RA 0 41200000 2 1 1 0 1 3f800000 40a00000
BI 0 41200000 4294967295 1 0 1
BC 0 41200000 4294967295 1 1 0 1 3f800000
RA 0 41200000 4294967295 1 0 0 1 3f800000 41700000
BI 0 41200000 4294967295 1 0 1
BC 0 41200000 4294967295 1 1 0 1 3f800000
RA 0 41200000 4294967295 1 1 0 1 3f800000 40a00000
VO 0 41200000 0 0 41700000
GV 0 41200000 0 0 41700000
GN 0 41200000 41700000
GN 0 41200000 41700000
VO 0 41200000 4294967295 0 41700000
GV 0 41200000 4294967295 0 41700000
GN 0 41200000 41700000
BI 0 41700000 0 1 1 2
BC 0 41700000 0 1 1 1 2 3f000000
RA 0 41700000 0 1 0 1 2 3f000000 41a00000
BI 0 41700000 0 1 1 2
BC 0 41700000 0 1 1 1 2 3f000000
RA 0 41700000 0 1 1 1 2 3f000000 41700000
BI 0 41700000 2 1 1 2
BC 0 41700000 2 1 1 1 2 3f000000
RA 0 41700000 2 1 0 1 2 3f000000 41a00000
BI 0 41700000 2 1 1 2
BC 0 41700000 2 1 1 1 2 3f000000
RA 0 41700000 2 1 1 1 2 3f000000 41700000
BI 0 41700000 4294967295 1 1 2
BC 0 41700000 4294967295 1 1 1 2 3f000000
RA 0 41700000 4294967295 1 0 1 2 3f000000 41a00000
BI 0 41700000 4294967295 1 1 2
BC 0 41700000 4294967295 1 1 1 2 3f000000
RA 0 41700000 4294967295 1 1 1 2 3f000000 41700000
VO 0 41700000 0 1 41a00000
GV 0 41700000 0 1 41a00000
GN 0 41700000 41a00000
GN 0 41700000 41a00000
VO 0 41700000 4294967295 1 41a00000
GV 0 41700000 4294967295 1 41a00000
GN 0 41700000 41a00000
BI 0 41a00000 0 1 1 2
BC 0 41a00000 0 1 1 1 2 3f800000
RA 0 41a00000 0 1 0 1 2 3f800000 41c80000
BI 0 41a00000 0 1 1 2
BC 0 41a00000 0 1 1 1 2 3f800000
RA 0 41a00000 0 1 1 1 2 3f800000 41700000
BI 0 41a00000 2 1 1 2
BC 0 41a00000 2 1 1 1 2 3f800000
RA 0 41a00000 2 1 0 1 2 3f800000 41c80000
BI 0 41a00000 2 1 1 2
BC 0 41a00000 2 1 1 1 2 3f800000
RA 0 41a00000 2 1 1 1 2 3f800000 41700000
BI 0 41a00000 4294967295 1 1 2
BC 0 41a00000 4294967295 1 1 1 2 3f800000
RA 0 41a00000 4294967295 1 0 1 2 3f800000 41c80000
BI 0 41a00000 4294967295 1 1 2
BC 0 41a00000 4294967295 1 1 1 2 3f800000
RA 0 41a00000 4294967295 1 1 1 2 3f800000 41700000
VO 0 41a00000 0 1 41c80000
GV 0 41a00000 0 1 41c80000
GN 0 41a00000 41c80000
GN 0 41a00000 41c80000
VO 0 41a00000 4294967295 1 41c80000
GV 0 41a00000 4294967295 1 41c80000
GN 0 41a00000 41c80000
BI 0 41c80000 0 1 2 2
BC 0 41c80000 0 1 1 2 2 00000000
RA 0 41c80000 0 1 0 2 2 00000000 41c80000
BI 0 41c80000 0 1 2 2
BC 0 41c80000 0 1 1 2 2 00000000
RA 0 41c80000 0 1 1 2 2 00000000 41c80000
BI 0 41c80000 2 1 2 2
BC 0 41c80000 2 1 1 2 2 00000000
RA 0 41c80000 2 1 0 2 2 00000000 41c80000
BI 0 41c80000 2 1 2 2
BC 0 41c80000 2 1 1 2 2 00000000
RA 0 41c80000 2 1 1 2 2 00000000 41c80000
BI 0 41c80000 4294967295 1 2 2
BC 0 41c80000 4294967295 1 1 2 2 00000000
RA 0 41c80000 4294967295 1 0 2 2 00000000 41c80000
BI 0 41c80000 4294967295 1 2 2
BC 0 41c80000 4294967295 1 1 2 2 00000000
RA 0 41c80000 4294967295 1 1 2 2 00000000 41c80000
VO 0 41c80000 0 2 41c80000
GV 0 41c80000 0 2 41c80000
GN 0 41c80000 41c80000
GN 0 41c80000 41c80000
VO 0 41c80000 4294967295 2 41c80000
GV 0 41c80000 4294967295 2 41c80000
GN 0 41c80000 41c80000
BI 0 bf800000 0 1 0 0
BC 0 bf800000 0 1 1 0 0 00000000
RA 0 bf800000 0 1 2 0 0 00000000 40a00000
RA 0 41c80000 1 1 -1 2 2 00000000 41c80000
BI 1 40a00000 0 1 0 1
BC 1 40a00000 0 1 1 0 1 3f000000
RA 1 40a00000 0 1 0 0 1 3f000000 41200000
BI 1 40a00000 0 1 0 1
BC 1 40a00000 0 1 1 0 1 3f000000
RA 1 40a00000 0 1 1 0 1 3f000000 40a00000
BI 1 40a00000 4294967295 1 0 1
BC 1 40a00000 4294967295 1 1 0 1 3f000000
RA 1 40a00000 4294967295 1 0 0 1 3f000000 41200000
BI 1 40a00000 4294967295 1 0 1
BC 1 40a00000 4294967295 1 1 0 1 3f000000
RA 1 40a00000 4294967295 1 1 0 1 3f000000 40a00000
VO 1 40a00000 0 0 40a00000
GV 1 40a00000 0 0 40a00000
BI 1 41700000 0 1 1 2
BC 1 41700000 0 1 1 1 2 3f000000
RA 1 41700000 0 1 0 1 2 3f000000 41a00000
BI 1 41700000 0 1 1 2
BC 1 41700000 0 1 1 1 2 3f000000
RA 1 41700000 0 1 1 1 2 3f000000 41700000
BI 1 41700000 4294967295 1 1 2
BC 1 41700000 4294967295 1 1 1 2 3f000000
RA 1 41700000 4294967295 1 0 1 2 3f000000 41a00000
BI 1 41700000 4294967295 1 1 2
BC 1 41700000 4294967295 1 1 1 2 3f000000
RA 1 41700000 4294967295 1 1 1 2 3f000000 41700000
VO 1 41700000 0 1 41700000
GV 1 41700000 0 1 41700000
RA 1 bf800000 0 1 2 0 0 00000000 40a00000
RA 1 41c80000 1 1 -1 2 2 00000000 41c80000
BI 2 40dccccd 0 1 0 0
BC 2 40dccccd 0 1 1 0 0 00000000
RA 2 40dccccd 0 1 0 0 0 00000000 42280000
BI 2 40dccccd 0 1 0 0
BC 2 40dccccd 0 1 1 0 0 00000000
RA 2 40dccccd 0 1 1 0 0 00000000 42280000
BI 2 40dccccd 4294967295 1 0 0
BC 2 40dccccd 4294967295 1 1 0 0 00000000
RA 2 40dccccd 4294967295 1 0 0 0 00000000 42280000
BI 2 40dccccd 4294967295 1 0 0
BC 2 40dccccd 4294967295 1 1 0 0 00000000
RA 2 40dccccd 4294967295 1 1 0 0 00000000 42280000
VO 2 40dccccd 0 0 42280000
GV 2 40dccccd 0 0 42280000
GN 2 40dccccd 42280000
GN 2 40dccccd 42280000
GN 2 40dccccd 42280000
BI 2 40e00000 0 1 0 0
BC 2 40e00000 0 1 1 0 0 00000000
RA 2 40e00000 0 1 0 0 0 00000000 42280000
BI 2 40e00000 0 1 0 0
BC 2 40e00000 0 1 1 0 0 00000000
RA 2 40e00000 0 1 1 0 0 00000000 42280000
BI 2 40e00000 4294967295 1 0 0
BC 2 40e00000 4294967295 1 1 0 0 00000000
RA 2 40e00000 4294967295 1 0 0 0 00000000 42280000
BI 2 40e00000 4294967295 1 0 0
BC 2 40e00000 4294967295 1 1 0 0 00000000
RA 2 40e00000 4294967295 1 1 0 0 00000000 42280000
VO 2 40e00000 0 0 42280000
GV 2 40e00000 0 0 42280000
GN 2 40e00000 42280000
GN 2 40e00000 42280000
GN 2 40e00000 42280000
BI 2 40e33333 0 1 0 0
BC 2 40e33333 0 1 1 0 0 00000000
RA 2 40e33333 0 1 0 0 0 00000000 42280000
BI 2 40e33333 0 1 0 0
BC 2 40e33333 0 1 1 0 0 00000000
RA 2 40e33333 0 1 1 0 0 00000000 42280000
BI 2 40e33333 4294967295 1 0 0
BC 2 40e33333 4294967295 1 1 0 0 00000000
RA 2 40e33333 4294967295 1 0 0 0 00000000 42280000
BI 2 40e33333 4294967295 1 0 0
BC 2 40e33333 4294967295 1 1 0 0 00000000
RA 2 40e33333 4294967295 1 1 0 0 00000000 42280000
VO 2 40e33333 0 0 42280000
GV 2 40e33333 0 0 42280000
GN 2 40e33333 42280000
GN 2 40e33333 42280000
GN 2 40e33333 42280000
BI 2 40dccccd 0 1 0 0
BC 2 40dccccd 0 1 1 0 0 00000000
RA 2 40dccccd 0 1 2 0 0 00000000 42280000
RA 2 40e33333 1 1 -1 0 0 00000000 42280000
BI 3 00000000 0 1 4294967295 4294967295
BC 3 00000000 0 1 0 4294967295 4294967295 bf800000
RA 3 00000000 0 1 0 4294967295 4294967295 bf800000 00000000
BI 3 00000000 0 1 4294967295 4294967295
BC 3 00000000 0 1 0 4294967295 4294967295 bf800000
RA 3 00000000 0 1 1 4294967295 4294967295 bf800000 00000000
BI 3 00000000 4294967295 1 4294967295 4294967295
BC 3 00000000 4294967295 1 0 4294967295 4294967295 bf800000
RA 3 00000000 4294967295 1 0 4294967295 4294967295 bf800000 00000000
BI 3 00000000 4294967295 1 4294967295 4294967295
BC 3 00000000 4294967295 1 0 4294967295 4294967295 bf800000
RA 3 00000000 4294967295 1 1 4294967295 4294967295 bf800000 00000000
VO 3 00000000 0 4294967295 00000000
GV 3 00000000 0 4294967295 00000000
GN 3 00000000 00000000
GN 3 00000000 00000000
GN 3 00000000 00000000
BI 3 40a00000 0 1 4294967295 4294967295
BC 3 40a00000 0 1 0 4294967295 4294967295 bf800000
RA 3 40a00000 0 1 0 4294967295 4294967295 bf800000 00000000
BI 3 40a00000 0 1 4294967295 4294967295
BC 3 40a00000 0 1 0 4294967295 4294967295 bf800000
RA 3 40a00000 0 1 1 4294967295 4294967295 bf800000 00000000
BI 3 40a00000 4294967295 1 4294967295 4294967295
BC 3 40a00000 4294967295 1 0 4294967295 4294967295 bf800000
RA 3 40a00000 4294967295 1 0 4294967295 4294967295 bf800000 00000000
BI 3 40a00000 4294967295 1 4294967295 4294967295
BC 3 40a00000 4294967295 1 0 4294967295 4294967295 bf800000
RA 3 40a00000 4294967295 1 1 4294967295 4294967295 bf800000 00000000
VO 3 40a00000 0 4294967295 00000000
GV 3 40a00000 0 4294967295 00000000
GN 3 40a00000 00000000
GN 3 40a00000 00000000
GN 3 40a00000 00000000
BI 3 00000000 0 1 4294967295 4294967295
BC 3 00000000 0 1 0 4294967295 4294967295 bf800000
RA 3 00000000 0 1 2 4294967295 4294967295 bf800000 00000000
RA 3 40a00000 1 1 -1 4294967295 4294967295 bf800000 00000000
BI 4 4039999a 0 1 0 0
BC 4 4039999a 0 1 1 0 0 00000000
RA 4 4039999a 0 1 0 0 0 00000000 40e00000
BI 4 4039999a 0 1 0 0
BC 4 4039999a 0 1 1 0 0 00000000
RA 4 4039999a 0 1 1 0 0 00000000 40e00000
BI 4 4039999a 2 1 0 0
BC 4 4039999a 2 1 1 0 0 00000000
RA 4 4039999a 2 1 0 0 0 00000000 40e00000
BI 4 4039999a 2 1 0 0
BC 4 4039999a 2 1 1 0 0 00000000
RA 4 4039999a 2 1 1 0 0 00000000 40e00000
BI 4 4039999a 4294967295 1 0 0
BC 4 4039999a 4294967295 1 1 0 0 00000000
RA 4 4039999a 4294967295 1 0 0 0 00000000 40e00000
BI 4 4039999a 4294967295 1 0 0
BC 4 4039999a 4294967295 1 1 0 0 00000000
RA 4 4039999a 4294967295 1 1 0 0 00000000 40e00000
VO 4 4039999a 0 0 40e00000
GV 4 4039999a 0 0 40e00000
GN 4 4039999a 40e00000
GN 4 4039999a 40e00000
GN 4 4039999a 40e00000
BI 4 40400000 0 1 0 1
BC 4 40400000 0 1 1 0 1 00000000
RA 4 40400000 0 1 0 0 1 00000000 40e00000
BI 4 40400000 0 1 0 1
BC 4 40400000 0 1 1 0 1 00000000
RA 4 40400000 0 1 1 0 1 00000000 40e00000
BI 4 40400000 2 1 0 1
BC 4 40400000 2 1 1 0 1 00000000
RA 4 40400000 2 1 0 0 1 00000000 40e00000
BI 4 40400000 2 1 0 1
BC 4 40400000 2 1 1 0 1 00000000
RA 4 40400000 2 1 1 0 1 00000000 40e00000
BI 4 40400000 4294967295 1 0 1
BC 4 40400000 4294967295 1 1 0 1 00000000
RA 4 40400000 4294967295 1 0 0 1 00000000 40e00000
BI 4 40400000 4294967295 1 0 1
BC 4 40400000 4294967295 1 1 0 1 00000000
RA 4 40400000 4294967295 1 1 0 1 00000000 40e00000
VO 4 40400000 0 0 40e00000
GV 4 40400000 0 0 40e00000
GN 4 40400000 40e00000
GN 4 40400000 40e00000
GN 4 40400000 40e00000
BI 4 40400013 0 1 0 1
BC 4 40400013 0 1 1 0 1 00000000
RA 4 40400013 0 1 0 0 1 00000000 40e00000
BI 4 40400013 0 1 0 1
BC 4 40400013 0 1 1 0 1 00000000
RA 4 40400013 0 1 1 0 1 00000000 40e00000
BI 4 40400013 2 1 0 1
BC 4 40400013 2 1 1 0 1 00000000
RA 4 40400013 2 1 0 0 1 00000000 40e00000
BI 4 40400013 2 1 0 1
BC 4 40400013 2 1 1 0 1 00000000
RA 4 40400013 2 1 1 0 1 00000000 40e00000
BI 4 40400013 4294967295 1 0 1
BC 4 40400013 4294967295 1 1 0 1 00000000
RA 4 40400013 4294967295 1 0 0 1 00000000 40e00000
BI 4 40400013 4294967295 1 0 1
BC 4 40400013 4294967295 1 1 0 1 00000000
RA 4 40400013 4294967295 1 1 0 1 00000000 40e00000
VO 4 40400013 0 0 40e00000
GV 4 40400013 0 0 40e00000
GN 4 40400013 40e00000
GN 4 40400013 40e00000
GN 4 40400013 40e00000
BI 4 40400026 0 1 0 1
BC 4 40400026 0 1 1 0 1 00000000
RA 4 40400026 0 1 0 0 1 00000000 40e00000
BI 4 40400026 0 1 0 1
BC 4 40400026 0 1 1 0 1 00000000
RA 4 40400026 0 1 1 0 1 00000000 40e00000
BI 4 40400026 2 1 0 1
BC 4 40400026 2 1 1 0 1 00000000
RA 4 40400026 2 1 0 0 1 00000000 40e00000
BI 4 40400026 2 1 0 1
BC 4 40400026 2 1 1 0 1 00000000
RA 4 40400026 2 1 1 0 1 00000000 40e00000
BI 4 40400026 4294967295 1 0 1
BC 4 40400026 4294967295 1 1 0 1 00000000
RA 4 40400026 4294967295 1 0 0 1 00000000 40e00000
BI 4 40400026 4294967295 1 0 1
BC 4 40400026 4294967295 1 1 0 1 00000000
RA 4 40400026 4294967295 1 1 0 1 00000000 40e00000
VO 4 40400026 0 0 40e00000
GV 4 40400026 0 0 40e00000
GN 4 40400026 40e00000
GN 4 40400026 40e00000
GN 4 40400026 40e00000
BI 4 4040002a 0 1 0 1
BC 4 4040002a 0 1 1 0 1 00000000
RA 4 4040002a 0 1 0 0 1 00000000 40e00000
BI 4 4040002a 0 1 0 1
BC 4 4040002a 0 1 1 0 1 00000000
RA 4 4040002a 0 1 1 0 1 00000000 40e00000
BI 4 4040002a 2 1 0 1
BC 4 4040002a 2 1 1 0 1 00000000
RA 4 4040002a 2 1 0 0 1 00000000 40e00000
BI 4 4040002a 2 1 0 1
BC 4 4040002a 2 1 1 0 1 00000000
RA 4 4040002a 2 1 1 0 1 00000000 40e00000
BI 4 4040002a 4294967295 1 0 1
BC 4 4040002a 4294967295 1 1 0 1 00000000
RA 4 4040002a 4294967295 1 0 0 1 00000000 40e00000
BI 4 4040002a 4294967295 1 0 1
BC 4 4040002a 4294967295 1 1 0 1 00000000
RA 4 4040002a 4294967295 1 1 0 1 00000000 40e00000
VO 4 4040002a 0 0 40e00000
GV 4 4040002a 0 0 40e00000
GN 4 4040002a 40e00000
GN 4 4040002a 40e00000
GN 4 4040002a 40e00000
BI 4 40401062 0 1 1 1
BC 4 40401062 0 1 1 1 1 00000000
RA 4 40401062 0 1 0 1 1 00000000 41100000
BI 4 40401062 0 1 1 1
BC 4 40401062 0 1 1 1 1 00000000
RA 4 40401062 0 1 1 1 1 00000000 41100000
BI 4 40401062 2 1 1 1
BC 4 40401062 2 1 1 1 1 00000000
RA 4 40401062 2 1 0 1 1 00000000 41100000
BI 4 40401062 2 1 1 1
BC 4 40401062 2 1 1 1 1 00000000
RA 4 40401062 2 1 1 1 1 00000000 41100000
BI 4 40401062 4294967295 1 1 1
BC 4 40401062 4294967295 1 1 1 1 00000000
RA 4 40401062 4294967295 1 0 1 1 00000000 41100000
BI 4 40401062 4294967295 1 1 1
BC 4 40401062 4294967295 1 1 1 1 00000000
RA 4 40401062 4294967295 1 1 1 1 00000000 41100000
VO 4 40401062 0 1 41100000
GV 4 40401062 0 1 41100000
GN 4 40401062 41100000
GN 4 40401062 41100000
GN 4 40401062 41100000
BI 4 4039999a 0 1 0 0
BC 4 4039999a 0 1 1 0 0 00000000
RA 4 4039999a 0 1 2 0 0 00000000 40e00000
RA 4 40401062 1 1 -1 1 1 00000000 41100000
BI 5 409ccccd 0 1 0 0
BC 5 409ccccd 0 1 1 0 0 00000000
RA 5 409ccccd 0 1 0 0 0 00000000 40000000
BI 5 409ccccd 0 1 0 0
BC 5 409ccccd 0 1 1 0 0 00000000
RA 5 409ccccd 0 1 1 0 0 00000000 40000000
BI 5 409ccccd 2 1 0 0
BC 5 409ccccd 2 1 1 0 0 00000000
RA 5 409ccccd 2 1 0 0 0 00000000 40000000
BI 5 409ccccd 2 1 0 0
BC 5 409ccccd 2 1 1 0 0 00000000
RA 5 409ccccd 2 1 1 0 0 00000000 40000000
BI 5 409ccccd 4294967295 1 0 0
BC 5 409ccccd 4294967295 1 1 0 0 00000000
RA 5 409ccccd 4294967295 1 0 0 0 00000000 40000000
BI 5 409ccccd 4294967295 1 0 0
BC 5 409ccccd 4294967295 1 1 0 0 00000000
RA 5 409ccccd 4294967295 1 1 0 0 00000000 40000000
VO 5 409ccccd 0 0 40000000
GV 5 409ccccd 0 0 40000000
BI 5 40a00000 0 1 0 1
BC 5 40a00000 0 1 1 0 1 00000000
RA 5 40a00000 0 1 0 0 1 00000000 40000000
BI 5 40a00000 0 1 0 1
BC 5 40a00000 0 1 1 0 1 00000000
RA 5 40a00000 0 1 1 0 1 00000000 40000000
BI 5 40a00000 2 1 0 1
BC 5 40a00000 2 1 1 0 1 00000000
RA 5 40a00000 2 1 0 0 1 00000000 40000000
BI 5 40a00000 2 1 0 1
BC 5 40a00000 2 1 1 0 1 00000000
RA 5 40a00000 2 1 1 0 1 00000000 40000000
BI 5 40a00000 4294967295 1 0 1
BC 5 40a00000 4294967295 1 1 0 1 00000000
RA 5 40a00000 4294967295 1 0 0 1 00000000 40000000
BI 5 40a00000 4294967295 1 0 1
BC 5 40a00000 4294967295 1 1 0 1 00000000
RA 5 40a00000 4294967295 1 1 0 1 00000000 40000000
VO 5 40a00000 0 0 40000000
GV 5 40a00000 0 0 40000000
BI 5 40a00831 0 1 1 1
BC 5 40a00831 0 1 1 1 1 00000000
RA 5 40a00831 0 1 0 1 1 00000000 40c00000
BI 5 40a00831 0 1 1 1
BC 5 40a00831 0 1 1 1 1 00000000
RA 5 40a00831 0 1 1 1 1 00000000 40c00000
BI 5 40a00831 2 1 1 1
BC 5 40a00831 2 1 1 1 1 00000000
RA 5 40a00831 2 1 0 1 1 00000000 40c00000
BI 5 40a00831 2 1 1 1
BC 5 40a00831 2 1 1 1 1 00000000
RA 5 40a00831 2 1 1 1 1 00000000 40c00000
BI 5 40a00831 4294967295 1 1 1
BC 5 40a00831 4294967295 1 1 1 1 00000000
RA 5 40a00831 4294967295 1 0 1 1 00000000 40c00000
BI 5 40a00831 4294967295 1 1 1
BC 5 40a00831 4294967295 1 1 1 1 00000000
RA 5 40a00831 4294967295 1 1 1 1 00000000 40c00000
VO 5 40a00831 0 1 40c00000
GV 5 40a00831 0 1 40c00000
BI 5 409ccccd 0 1 0 0
BC 5 409ccccd 0 1 1 0 0 00000000
RA 5 409ccccd 0 1 2 0 0 00000000 40000000
RA 5 40a00831 1 1 -1 1 1 00000000 40c00000
BI 6 bf800000 0 1 0 0
BC 6 bf800000 0 1 1 0 0 00000000
RA 6 bf800000 0 1 0 0 0 00000000 3f800000
BI 6 bf800000 0 1 0 0
BC 6 bf800000 0 1 1 0 0 00000000
RA 6 bf800000 0 1 1 0 0 00000000 3f800000
BI 6 bf800000 4294967295 1 0 0
BC 6 bf800000 4294967295 1 1 0 0 00000000
RA 6 bf800000 4294967295 1 0 0 0 00000000 3f800000
BI 6 bf800000 4294967295 1 0 0
BC 6 bf800000 4294967295 1 1 0 0 00000000
RA 6 bf800000 4294967295 1 1 0 0 00000000 3f800000
VO 6 bf800000 0 0 3f800000
GV 6 bf800000 0 0 3f800000
BI 6 40a00000 0 1 0 0
BC 6 40a00000 0 1 1 0 0 00000000
RA 6 40a00000 0 1 0 0 0 00000000 3f800000
BI 6 40a00000 0 1 0 0
BC 6 40a00000 0 1 1 0 0 00000000
RA 6 40a00000 0 1 1 0 0 00000000 3f800000
BI 6 40a00000 4294967295 1 0 0
BC 6 40a00000 4294967295 1 1 0 0 00000000
RA 6 40a00000 4294967295 1 0 0 0 00000000 3f800000
BI 6 40a00000 4294967295 1 0 0
BC 6 40a00000 4294967295 1 1 0 0 00000000
RA 6 40a00000 4294967295 1 1 0 0 00000000 3f800000
VO 6 40a00000 0 0 3f800000
GV 6 40a00000 0 0 3f800000
BI 6 41c80000 0 1 2 2
BC 6 41c80000 0 1 1 2 2 00000000
RA 6 41c80000 0 1 0 2 2 00000000 40400000
BI 6 41c80000 0 1 2 2
BC 6 41c80000 0 1 1 2 2 00000000
RA 6 41c80000 0 1 1 2 2 00000000 40400000
BI 6 41c80000 4294967295 1 2 2
BC 6 41c80000 4294967295 1 1 2 2 00000000
RA 6 41c80000 4294967295 1 0 2 2 00000000 40400000
BI 6 41c80000 4294967295 1 2 2
BC 6 41c80000 4294967295 1 1 2 2 00000000
RA 6 41c80000 4294967295 1 1 2 2 00000000 40400000
VO 6 41c80000 0 2 40400000
GV 6 41c80000 0 2 40400000
BI 6 bf800000 0 1 0 0
BC 6 bf800000 0 1 1 0 0 00000000
RA 6 bf800000 0 1 2 0 0 00000000 3f800000
RA 6 41c80000 1 1 -1 2 2 00000000 40400000
BI 0 bf800000 0 0 0 0
BC 0 bf800000 0 0 1 0 0 00000000
RA 0 bf800000 0 0 0 0 0 00000000 40a00000
BI 0 bf800000 0 0 0 0
BC 0 bf800000 0 0 1 0 0 00000000
BI 0 00000000 0 0 0 1
BC 0 00000000 0 0 1 0 1 00000000
RA 0 00000000 0 0 0 0 1 00000000 40a00000
BI 0 00000000 0 0 0 1
BC 0 00000000 0 0 1 0 1 00000000
BI 0 40a00000 0 0 0 1
BC 0 40a00000 0 0 1 0 1 3f000000
RA 0 40a00000 0 0 0 0 1 3f000000 41200000
BI 0 40a00000 0 0 0 1
BC 0 40a00000 0 0 1 0 1 3f000000
BI 0 411ffff6 0 0 0 1
BC 0 411ffff6 0 0 1 0 1 3f7ffff0
RA 0 411ffff6 0 0 0 0 1 3f7ffff0 416ffff6
BI 0 411ffff6 0 0 0 1
BC 0 411ffff6 0 0 1 0 1 3f7ffff0
BI 0 41200000 0 0 0 1
BC 0 41200000 0 0 1 0 1 3f800000
RA 0 41200000 0 0 0 0 1 3f800000 41700000
BI 0 41200000 0 0 0 1
BC 0 41200000 0 0 1 0 1 3f800000
BI 1 40a00000 0 0 0 1
BC 1 40a00000 0 0 1 0 1 3f000000
RA 1 40a00000 0 0 0 0 1 3f000000 41200000
BI 1 40a00000 0 0 0 1
BC 1 40a00000 0 0 1 0 1 3f000000
BI 2 40dccccd 0 0 0 0
BC 2 40dccccd 0 0 1 0 0 00000000
RA 2 40dccccd 0 0 0 0 0 00000000 42280000
BI 2 40dccccd 0 0 0 0
BC 2 40dccccd 0 0 1 0 0 00000000
BI 2 40e00000 0 0 0 0
BC 2 40e00000 0 0 1 0 0 00000000
RA 2 40e00000 0 0 0 0 0 00000000 42280000
BI 2 40e00000 0 0 0 0
BC 2 40e00000 0 0 1 0 0 00000000
BI 2 40e33333 0 0 0 0
BC 2 40e33333 0 0 1 0 0 00000000
RA 2 40e33333 0 0 0 0 0 00000000 42280000
BI 2 40e33333 0 0 0 0
BC 2 40e33333 0 0 1 0 0 00000000
BI 2 40dccccd 4294967295 0 0 0
BC 2 40dccccd 4294967295 0 1 0 0 00000000
RA 2 40dccccd 4294967295 0 0 0 0 00000000 42280000
BI 2 40dccccd 4294967295 0 0 0
BC 2 40dccccd 4294967295 0 1 0 0 00000000
BI 2 40e00000 4294967295 0 0 0
BC 2 40e00000 4294967295 0 1 0 0 00000000
RA 2 40e00000 4294967295 0 0 0 0 00000000 42280000
BI 2 40e00000 4294967295 0 0 0
BC 2 40e00000 4294967295 0 1 0 0 00000000
BI 2 40e33333 4294967295 0 0 0
BC 2 40e33333 4294967295 0 1 0 0 00000000
RA 2 40e33333 4294967295 0 0 0 0 00000000 42280000
BI 2 40e33333 4294967295 0 0 0
BC 2 40e33333 4294967295 0 1 0 0 00000000
BI 3 00000000 0 0 4294967295 4294967295
BC 3 00000000 0 0 0 4294967295 4294967295 bf800000
RA 3 00000000 0 0 0 4294967295 4294967295 bf800000 00000000
BI 3 00000000 0 0 4294967295 4294967295
BC 3 00000000 0 0 0 4294967295 4294967295 bf800000
BI 3 40a00000 0 0 4294967295 4294967295
BC 3 40a00000 0 0 0 4294967295 4294967295 bf800000
RA 3 40a00000 0 0 0 4294967295 4294967295 bf800000 00000000
BI 3 40a00000 0 0 4294967295 4294967295
BC 3 40a00000 0 0 0 4294967295 4294967295 bf800000
BI 3 00000000 4294967295 0 4294967295 4294967295
BC 3 00000000 4294967295 0 0 4294967295 4294967295 bf800000
RA 3 00000000 4294967295 0 0 4294967295 4294967295 bf800000 00000000
BI 3 00000000 4294967295 0 4294967295 4294967295
BC 3 00000000 4294967295 0 0 4294967295 4294967295 bf800000
BI 3 40a00000 4294967295 0 4294967295 4294967295
BC 3 40a00000 4294967295 0 0 4294967295 4294967295 bf800000
RA 3 40a00000 4294967295 0 0 4294967295 4294967295 bf800000 00000000
BI 3 40a00000 4294967295 0 4294967295 4294967295
BC 3 40a00000 4294967295 0 0 4294967295 4294967295 bf800000
BI 4 4039999a 0 0 0 0
BC 4 4039999a 0 0 1 0 0 00000000
RA 4 4039999a 0 0 0 0 0 00000000 40e00000
BI 4 4039999a 0 0 0 0
BC 4 4039999a 0 0 1 0 0 00000000
BI 4 40400000 0 0 0 1
BC 4 40400000 0 0 1 0 1 00000000
RA 4 40400000 0 0 0 0 1 00000000 40e00000
BI 4 40400000 0 0 0 1
BC 4 40400000 0 0 1 0 1 00000000
BI 4 40400013 0 0 0 1
BC 4 40400013 0 0 1 0 1 00000000
RA 4 40400013 0 0 0 0 1 00000000 40e00000
BI 4 40400013 0 0 0 1
BC 4 40400013 0 0 1 0 1 00000000
BI 4 40400026 0 0 0 1
BC 4 40400026 0 0 1 0 1 00000000
RA 4 40400026 0 0 0 0 1 00000000 40e00000
BI 4 40400026 0 0 0 1
BC 4 40400026 0 0 1 0 1 00000000
BI 4 4040002a 0 0 0 1
BC 4 4040002a 0 0 1 0 1 00000000
RA 4 4040002a 0 0 0 0 1 00000000 40e00000
BI 4 4040002a 0 0 0 1
BC 4 4040002a 0 0 1 0 1 00000000
BI 4 40401062 0 0 1 1
BC 4 40401062 0 0 1 1 1 00000000
RA 4 40401062 0 0 0 1 1 00000000 41100000
BI 4 40401062 0 0 1 1
BC 4 40401062 0 0 1 1 1 00000000
BI 5 409ccccd 0 0 0 0
BC 5 409ccccd 0 0 1 0 0 00000000
RA 5 409ccccd 0 0 0 0 0 00000000 40000000
BI 5 409ccccd 0 0 0 0
BC 5 409ccccd 0 0 1 0 0 00000000
BI 5 40a00000 0 0 0 1
BC 5 40a00000 0 0 1 0 1 00000000
RA 5 40a00000 0 0 0 0 1 00000000 40000000
BI 5 40a00000 0 0 0 1
BC 5 40a00000 0 0 1 0 1 00000000
BI 5 40a00831 0 0 1 1
BC 5 40a00831 0 0 1 1 1 00000000
RA 5 40a00831 0 0 0 1 1 00000000 40c00000
BI 5 40a00831 0 0 1 1
BC 5 40a00831 0 0 1 1 1 00000000
BI 6 bf800000 0 0 0 0
BC 6 bf800000 0 0 1 0 0 00000000
RA 6 bf800000 0 0 0 0 0 00000000 3f800000
BI 6 bf800000 0 0 0 0
BC 6 bf800000 0 0 1 0 0 00000000
BI 6 41c80000 0 0 2 2
BC 6 41c80000 0 0 1 2 2 00000000
RA 6 41c80000 0 0 0 2 2 00000000 40400000
BI 6 41c80000 0 0 2 2
BC 6 41c80000 0 0 1 2 2 00000000
TG 1 00000000 40a00000
TG 2 00000000 40a00000
TG 3 00000000 40a00000
TG 4 00000000 40a00000
TG 5 00000000 40a00000
TG 7 00000000 40a00000
TG 8 00000000 40a00000
TG 9 00000000 40a00000
TG 10 00000000 40a00000
TG 11 00000000 40d00000
TG 12 00000000 40a00000
TG 13 00000000 40a00000
TG 14 00000000 40a00000
TG 16 00000000 40a00000
TG 17 00000000 40a00000
TG 19 00000000 40a00000
TG 20 00000000 40a00000
TG 21 00000000 40a00000
TG 1 3f800000 40a00000
TG 2 3f800000 41099999
TG 3 3f800000 41099999
TG 4 3f800000 40a00000
TG 5 3f800000 40a00000
TG 7 3f800000 41099999
TG 8 3f800000 40a00000
TG 9 3f800000 41099999
TG 10 3f800000 41099999
TG 11 3f800000 4132e146
TG 12 3f800000 41099999
TG 13 3f800000 41099999
TG 14 3f800000 41099999
TG 16 3f800000 41099999
TG 17 3f800000 41099999
TG 19 3f800000 41099999
TG 20 3f800000 41099999
TG 21 3f800000 41099999
TG 1 418c0000 41c80000
TG 2 418c0000 41c80000
TG 3 418c0000 41c80000
TG 4 418c0000 41c80000
TG 5 418c0000 41c80000
TG 7 418c0000 41c80000
TG 8 418c0000 41c80000
TG 9 418c0000 41c80000
TG 10 418c0000 41c80000
TG 11 418c0000 42020000
TG 12 418c0000 41c80000
TG 13 418c0000 41c80000
TG 14 418c0000 41c80000
TG 16 418c0000 41c80000
TG 17 418c0000 41c80000
TG 19 418c0000 41c80000
TG 20 418c0000 41c80000
TG 21 418c0000 41c80000
TG 1 42c80000 41c80000
TG 2 42c80000 41c80000
TG 3 42c80000 41c80000
TG 4 42c80000 41c80000
TG 5 42c80000 41c80000
TG 7 42c80000 41c80000
TG 8 42c80000 41c80000
TG 9 42c80000 41c80000
TG 10 42c80000 41c80000
TG 11 42c80000 42020000
TG 12 42c80000 41c80000
TG 13 42c80000 41c80000
TG 14 42c80000 41c80000
TG 16 42c80000 41c80000
TG 17 42c80000 41c80000
TG 19 42c80000 41c80000
TG 20 42c80000 41c80000
TG 21 42c80000 41c80000
TG 1 c1440000 40a00000
TG 2 c1440000 40a00000
TG 3 c1440000 40a00000
TG 4 c1440000 40a00000
TG 5 c1440000 40a00000
TG 7 c1440000 40a00000
TG 8 c1440000 40a00000
TG 9 c1440000 40a00000
TG 10 c1440000 40a00000
TG 11 c1440000 40d00000
TG 12 c1440000 40a00000
TG 13 c1440000 40a00000
TG 14 c1440000 40a00000
TG 16 c1440000 40a00000
TG 17 c1440000 40a00000
TG 19 c1440000 40a00000
TG 20 c1440000 40a00000
TG 21 c1440000 40a00000
TG 6 00000000 40a00000
TG 6 40a00000 40a00000
TG 6 41700000 41700000
TG 6 41c80000 41c80000
TG 6 bf800000 40a00000
TG 15 3e800000 40a00000
TG 15 3f800000 40b00000
TG 15 3fe00000 40c00000
TG 15 bf800000 40a00000
TG 15 40a00000 4102aaab
TG 18 00000000 3fb1c71d
TG 18 40a00000 4031c71d
TG 18 41700000 40b1c71d
TG 18 41c80000 40de38e4
TG 18 bf800000 3fb1c71d
BNDL 00000000 b727c5ac
BNDU 00000000 3727c5ac
BNDL 3f800000 3f7fff58
BNDU 3f800000 3f800054
BNDL c0b00000 c0b00015
BNDU c0b00000 c0afffeb
BNDL 41200000 411ffff6
BNDU 41200000 4120000a
BNDL 42c80000 42c7ffff
BNDU 42c80000 42c80001
BNDL 3dcccccd 3dccc78f
BNDU 3dcccccd 3dccd20b
BNDL 40666666 4066663c
BNDU 40666666 40666690
BNDL 33d6bf95 b726182d
BNDU 33d6bf95 3729732b
BNDL 47f12064 47f12064
BNDU 47f12064 47f12064
BNDL b6a7c5ac b77ba882
BNDU b6a7c5ac 36a7c5ad
SPD 00000000 00000000
MS 00000000 00000000
SPD 3f800000 40666666
MS 3f800000 3e8e38e4
SPD 41200000 42100000
MS 41200000 4031c71d
SPD 425e0000 4347cccc
MS 425e0000 4176aaab
SPD 42c80000 43b40000
MS 42c80000 41de38e4
SPD 40666666 414f5c28
MS 40666666 3f800000
SPD 3dcccccd 3eb851eb
MS 3dcccccd 3ce38e3a
SPD c1440000 c2306666
MS c1440000 c059c71d
SPD 461c4000 470ca000
MS 461c4000 452d9c72
LW 00000000 3f800000
LW 3f800000 00000000
LW 3f000000 3f000000
LW 3e800000 3f400000
LW 3dcccccd 3f666666
LW 3f666666 3dccccd0
LW 3eaaaaab 3f2aaaaa
LW 3f333333 3e99999a
LW 40000000 bf800000
LW bf000000 3fc00000
LW 33d6bf95 3f7ffffe"#;


    #[test]
    fn golden_replay() {
        replay(GOLDEN);
    }

    /// TEMPORARY (removed after verification): replays the complete
    /// differential dump, all 2665 rows including every backward-search row.
    #[test]
    fn temp_full_dump_replay() {
        let golden =
            std::fs::read_to_string("/home/z/my-project/tool-results/curve_dump.txt").expect("dump file");
        replay(&golden);
    }
}

