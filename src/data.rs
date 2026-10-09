//! The data lane: **authoring cars, tracks and tuning as data** — the gap the
//! fork's own roadmap called item #1.
//!
//! * [`CarStructure`] — everything structural about a car (wheel geometry,
//!   body ellipsoids/collision tree, mass & inertia, spawn state, ground
//!   materials) as plain data. Convert to/from the `TMNFM6G1` image:
//!   `structure_from_image` / `image_with_structure`.
//! * [`CarTuning`] — the 132-scalar + 23-curve + 4-gear-table tuning block
//!   as plain data: `tuning_from_image` / `image_with_tuning`.
//!
//! Both directions are **layout-preserving patches** over a base image:
//! `write(read(image))` is byte-identical (tested against the shipped
//! Stadium capture), and every authorable field is patched at its game
//! offset. Authoring a new car = start from the reference structure (or
//! build one in code), change the numbers, rebuild the image.

use crate::collision::GmBoxAligned;
use crate::gm::{GmIso4, GmMat3, GmQuat, GmVec3};
use crate::world::{VehicleCurveDescriptor};
use crate::world::{
    VehicleCollisionTreeRecord, VehicleImage, VehicleMaterialRecord,
    CAR_SIZE, GEARBOX_COUNT, GEARBOX_VALUE_COUNT, WHEEL_SIZE,
};
use serde::{Deserialize, Serialize};

/* ===========================================================================
 * Car structure
 * ========================================================================= */

/// One wheel's structure (game wheel block fields).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WheelStructure {
    pub active: i32,
    pub steerable: i32,
    pub radius: f32,
    /// Wheel +0x010: the surface source isometry (rest pose, car-local).
    pub surface_source: Iso4Data,
    /// Wheel +0x070: 12 opaque floats the surface handler keeps.
    pub field70: [f32; 12],
    pub fielda0: i32,
    pub fielda4: i32,
    /// Wheel +0x0a8: the wheel center, car-local.
    pub offset_from_vehicle: Vec3Data,
    /// Wheel +0x0b4: the initial real-time state (0xa8 bytes).
    pub real_time: WheelRealTimeData,
}

/// The real-time wheel state block (`CSceneVehicleCarWheelRealTimeState`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WheelRealTimeData {
    pub damper_absorb: f32,
    pub field04: f32,
    pub field08: f32,
    pub basis0: Mat3Data,
    pub basis1: Mat3Data,
    pub field54: Vec3Data,
    pub field6c: f32,
    pub has_ground_contact: i32,
    pub contact_material_id: i32,
    pub is_sliding: i32,
    pub relative_rotz_axis: Vec3Data,
    pub contact_body_word: u32,
    pub ground_contact_count: i32,
    pub field90: Vec3Data,
    pub rotation_phase: f32,
    pub blend_value: f32,
    pub blend_target: f32,
}

/// One collision-tree node (kind 0 body / 1 wheel leaf / 2 root).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CollisionNodeStructure {
    pub kind: u32,
    pub wheel_index: Option<u32>,
    pub flags: u32,
    /// The AABB in the parent (car) frame.
    pub box_aligned: BoxData,
    /// The local isometry (bit 2 of `flags` marks it live).
    pub local_iso: Iso4Data,
    /// Ellipsoid radii (sphere leaves keep the radius in radii.x).
    pub radii: Vec3Data,
    pub geometry_type: u8,
    pub material_id: u8,
}

/// The rigid body.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BodyStructure {
    pub mass: f32,
    /// Inverse inertia tensor, body frame (row-major 3x3).
    pub inverse_inertia: Mat3Data,
    pub drag_linear: f32,
    pub drag_angular: f32,
    pub substep_len: f32,
    pub force_field_scale: f32,
    /// Centre-of-mass offset, body frame.
    pub center_of_mass: Vec3Data,
}

/// The spawn state of the rigid body (the 180-byte dyna state).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpawnState {
    /// Quaternion bytes in the game's (w, x, y, z) scalar-first order.
    pub quaternion: [f32; 4],
    pub rotation: Mat3Data,
    pub position: Vec3Data,
    pub linear_velocity: Vec3Data,
    pub angular_velocity: Vec3Data,
    pub tail: [f32; 5],
}

/// A ground material row (per-surface tire values + the bump descriptor).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GroundMaterialStructure {
    pub values: [f32; 4],
    /// Whether this material references the shared 128x128 bump image.
    pub fake_contact_mask: bool,
    pub fake_contact_period_x: f32,
    pub fake_contact_period_z: f32,
    pub fake_contact_impulse_scale: f32,
    pub fake_contact_impulse_limit: f32,
}

/// The complete structural definition of a car.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CarStructure {
    pub wheels: Vec<WheelStructure>,
    /// The collision tree root's children, in pre-order.
    pub collision_nodes: Vec<CollisionNodeStructure>,
    pub body: BodyStructure,
    pub spawn: SpawnState,
    /// The body reference position (`*(hms_item->field14)+0x50`).
    pub body_reference_position: Vec3Data,
    /// 31-entry table: physical material id -> ground material row.
    pub ground_material_ids: Vec<u8>,
    pub ground_materials: Vec<GroundMaterialStructure>,
    /// Dyna control words.
    pub clamp_angular: i32,
    pub max_angular_speed: f32,
    pub mode: i32,
    /// The initial tick (usually 0 for an authored car).
    pub tick: u32,
}

/* --- small serde data types (plain arrays; JSON-friendly) --- */

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Vec3Data {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3Data {
    pub fn of(v: &GmVec3) -> Vec3Data {
        Vec3Data { x: v.x, y: v.y, z: v.z }
    }
    fn to_gm(&self) -> GmVec3 {
        GmVec3 { x: self.x, y: self.y, z: self.z }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Mat3Data {
    pub m: [f32; 9],
}

impl Mat3Data {
    pub fn of(v: &GmMat3) -> Mat3Data {
        Mat3Data { m: v.m }
    }
    fn to_gm(&self) -> GmMat3 {
        GmMat3 { m: self.m }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Iso4Data {
    pub m: [f32; 9],
    pub t: [f32; 3],
}

impl Iso4Data {
    pub fn of(v: &GmIso4) -> Iso4Data {
        Iso4Data { m: v.m, t: v.t }
    }
    fn to_gm(&self) -> GmIso4 {
        GmIso4 { m: self.m, t: self.t }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoxData {
    pub center: Vec3Data,
    pub half_extent: Vec3Data,
}

impl BoxData {
    fn of(v: &GmBoxAligned) -> BoxData {
        BoxData {
            center: Vec3Data::of(&v.center),
            half_extent: Vec3Data::of(&v.half_extent),
        }
    }
    fn to_gm(&self) -> GmBoxAligned {
        GmBoxAligned {
            center: self.center.to_gm(),
            half_extent: self.half_extent.to_gm(),
        }
    }
}

/* ===========================================================================
 * Structure <-> image
 * ========================================================================= */

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn vec3_at(b: &[u8], o: usize) -> Vec3Data {
    Vec3Data { x: f32_at(b, o), y: f32_at(b, o + 4), z: f32_at(b, o + 8) }
}
fn mat3_at(b: &[u8], o: usize) -> Mat3Data {
    let mut m = [0.0f32; 9];
    for (i, item) in m.iter_mut().enumerate() {
        *item = f32_at(b, o + 4 * i);
    }
    Mat3Data { m }
}
fn iso4_at(b: &[u8], o: usize) -> Iso4Data {
    let mut m = [0.0f32; 9];
    for (i, item) in m.iter_mut().enumerate() {
        *item = f32_at(b, o + 4 * i);
    }
    Iso4Data {
        m,
        t: [f32_at(b, o + 36), f32_at(b, o + 40), f32_at(b, o + 44)],
    }
}
fn box_at(b: &[u8], o: usize) -> BoxData {
    BoxData { center: vec3_at(b, o), half_extent: vec3_at(b, o + 12) }
}

/// Extracts the complete car structure from a vehicle image.
pub fn structure_from_image(image: &VehicleImage) -> CarStructure {
    let wheels: Vec<WheelStructure> = (0..4)
        .map(|i| {
            let w = &image.wheels[i * WHEEL_SIZE..(i + 1) * WHEEL_SIZE];
            let rt = &w[0xb4..0xb4 + 0xa8];
            let mut field70 = [0.0f32; 12];
            for (k, item) in field70.iter_mut().enumerate() {
                *item = f32_at(w, 0x70 + 4 * k);
            }
            WheelStructure {
                active: i32_at(w, 0x000),
                steerable: i32_at(w, 0x004),
                radius: f32_at(w, 0x008),
                surface_source: iso4_at(w, 0x010),
                field70,
                fielda0: i32_at(w, 0x0a0),
                fielda4: i32_at(w, 0x0a4),
                offset_from_vehicle: vec3_at(w, 0x0a8),
                real_time: WheelRealTimeData {
                    damper_absorb: f32_at(rt, 0x00),
                    field04: f32_at(rt, 0x04),
                    field08: f32_at(rt, 0x08),
                    basis0: mat3_at(rt, 0x0c),
                    basis1: mat3_at(rt, 0x30),
                    field54: vec3_at(rt, 0x54),
                    field6c: f32_at(rt, 0x6c),
                    has_ground_contact: i32_at(rt, 0x70),
                    contact_material_id: i32_at(rt, 0x74),
                    is_sliding: i32_at(rt, 0x78),
                    relative_rotz_axis: vec3_at(rt, 0x7c),
                    contact_body_word: u32_at(rt, 0x88),
                    ground_contact_count: i32_at(rt, 0x8c),
                    field90: vec3_at(rt, 0x90),
                    rotation_phase: f32_at(rt, 0x9c),
                    blend_value: f32_at(rt, 0xa0),
                    blend_target: f32_at(rt, 0xa4),
                },
            }
        })
        .collect();

    let collision_nodes: Vec<CollisionNodeStructure> = image
        .collision_children
        .iter()
        .map(|r| {
            let kind = r.kind & 0xff;
            CollisionNodeStructure {
                kind,
                wheel_index: if kind == 1 {
                    Some(r.wheel_index)
                } else {
                    None
                },
                flags: r.flags,
                box_aligned: BoxData::of(&r.box_aligned),
                local_iso: Iso4Data::of(&r.local_iso),
                radii: vec3_at(&r.shape, 0),
                geometry_type: r.geometry_type,
                material_id: image.collision_material_ids
                    [r.material_index as usize],
            }
        })
        .collect();

    let body = BodyStructure {
        mass: f32_at(&image.params, 0x00),
        inverse_inertia: mat3_at(&image.params, 0x04),
        drag_linear: f32_at(&image.params, 0x28),
        drag_angular: f32_at(&image.params, 0x2c),
        substep_len: f32_at(&image.params, 0x30),
        force_field_scale: f32_at(&image.params, 0x34),
        center_of_mass: vec3_at(&image.params, 0x38),
    };

    let mut quaternion = [0.0f32; 4];
    for (i, item) in quaternion.iter_mut().enumerate() {
        *item = f32_at(&image.state, 4 * i);
    }
    let spawn = SpawnState {
        quaternion,
        rotation: mat3_at(&image.state, 0x10),
        position: vec3_at(&image.state, 0x34),
        linear_velocity: vec3_at(&image.state, 0x40),
        angular_velocity: vec3_at(&image.state, 0x58),
        tail: [
            f32_at(&image.state, 0xa0),
            f32_at(&image.state, 0xa4),
            f32_at(&image.state, 0xa8),
            f32_at(&image.state, 0xac),
            f32_at(&image.state, 0xb0),
        ],
    };

    CarStructure {
        wheels,
        collision_nodes,
        body,
        spawn,
        body_reference_position: vec3_at(
            &image.raw,
            image.header.body_reference_offset as usize),
        ground_material_ids: image.ground_ids.clone(),
        ground_materials: image
            .materials
            .iter()
            .map(|m| GroundMaterialStructure {
                values: m.values,
                fake_contact_mask: m.fake_contact_mask != 0,
                fake_contact_period_x: m.fake_contact_period_x,
                fake_contact_period_z: m.fake_contact_period_z,
                fake_contact_impulse_scale: m.fake_contact_impulse_scale,
                fake_contact_impulse_limit: m.fake_contact_impulse_limit,
            })
            .collect(),
        clamp_angular: i32_at(&image.dyna, 0x0c0),
        max_angular_speed: f32_at(&image.dyna, 0x0c4),
        mode: i32_at(&image.dyna, 0x340),
        tick: image.header.tick,
    }
}

/// Rebuilds `image` with the given structure patched in (layout-preserving).
/// Every structural field is written at its game offset; sections the
/// structure does not describe (tuning, curves, gearboxes, identities) are
/// left untouched, so starting from the reference image keeps a valid car.
pub fn image_with_structure(
    mut image: VehicleImage,
    structure: &CarStructure,
) -> VehicleImage {
    let put_f32 = |b: &mut [u8], o: usize, v: &f32| {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_i32 = |b: &mut [u8], o: usize, v: &i32| {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_u32 = |b: &mut [u8], o: usize, v: &u32| {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };
    let put_vec3 = |b: &mut [u8], o: usize, v: &Vec3Data| {
        put_f32(b, o, &v.x);
        put_f32(b, o + 4, &v.y);
        put_f32(b, o + 8, &v.z);
    };
    let put_mat3 = |b: &mut [u8], o: usize, v: &Mat3Data| {
        for (i, m) in v.m.iter().enumerate() {
            put_f32(b, o + 4 * i, m);
        }
    };
    let put_iso4 = |b: &mut [u8], o: usize, v: &Iso4Data| {
        put_mat3(b, o, &Mat3Data { m: v.m });
        put_vec3(b, o + 36, &Vec3Data { x: v.t[0], y: v.t[1], z: v.t[2] });
    };

    assert_eq!(structure.wheels.len(), 4, "the game car has four wheels");
    for (i, wheel) in structure.wheels.iter().enumerate() {
        let w = &mut image.wheels[i * WHEEL_SIZE..(i + 1) * WHEEL_SIZE];
        put_i32(w, 0x000, &wheel.active);
        put_i32(w, 0x004, &wheel.steerable);
        put_f32(w, 0x008, &wheel.radius);
        put_iso4(w, 0x010, &wheel.surface_source);
        for (k, f) in wheel.field70.iter().enumerate() {
            put_f32(w, 0x70 + 4 * k, f);
        }
        put_i32(w, 0x0a0, &wheel.fielda0);
        put_i32(w, 0x0a4, &wheel.fielda4);
        put_vec3(w, 0x0a8, &wheel.offset_from_vehicle);
        let rt = &mut w[0xb4..0xb4 + 0xa8];
        let r = &wheel.real_time;
        put_f32(rt, 0x00, &r.damper_absorb);
        put_f32(rt, 0x04, &r.field04);
        put_f32(rt, 0x08, &r.field08);
        put_mat3(rt, 0x0c, &r.basis0);
        put_mat3(rt, 0x30, &r.basis1);
        put_vec3(rt, 0x54, &r.field54);
        put_f32(rt, 0x6c, &r.field6c);
        put_i32(rt, 0x70, &r.has_ground_contact);
        put_i32(rt, 0x74, &r.contact_material_id);
        put_i32(rt, 0x78, &r.is_sliding);
        put_vec3(rt, 0x7c, &r.relative_rotz_axis);
        put_u32(rt, 0x88, &r.contact_body_word);
        put_i32(rt, 0x8c, &r.ground_contact_count);
        put_vec3(rt, 0x90, &r.field90);
        put_f32(rt, 0x9c, &r.rotation_phase);
        put_f32(rt, 0xa0, &r.blend_value);
        put_f32(rt, 0xa4, &r.blend_target);
    }

    assert_eq!(
        structure.collision_nodes.len(),
        image.collision_children.len(),
        "the collision tree node count is fixed by the descriptor table"
    );
    for (i, node) in structure.collision_nodes.iter().enumerate() {
        let record = &mut image.collision_children[i];
        record.flags = node.flags;
        record.box_aligned = node.box_aligned.to_gm();
        record.local_iso = node.local_iso.to_gm();
        record.kind = node.kind
            | ((record.kind >> 8) << 8); /* preserve the child-count bits */
        if node.kind & 0xff == 1 {
            record.wheel_index = node.wheel_index.unwrap_or(u32::MAX);
        }
        let mut shape = [0u8; 24];
        shape[0..4].copy_from_slice(&node.radii.x.to_le_bytes());
        shape[4..8].copy_from_slice(&node.radii.y.to_le_bytes());
        shape[8..12].copy_from_slice(&node.radii.z.to_le_bytes());
        record.shape = shape;
        record.geometry_type = node.geometry_type;
        if let Some(mid) = image
            .collision_material_ids
            .get_mut(record.material_index as usize)
        {
            *mid = node.material_id;
        }
    }

    let p = &mut image.params;
    put_f32(p, 0x00, &structure.body.mass);
    put_mat3(p, 0x04, &structure.body.inverse_inertia);
    put_f32(p, 0x28, &structure.body.drag_linear);
    put_f32(p, 0x2c, &structure.body.drag_angular);
    put_f32(p, 0x30, &structure.body.substep_len);
    put_f32(p, 0x34, &structure.body.force_field_scale);
    put_vec3(p, 0x38, &structure.body.center_of_mass);

    let s = &mut image.state;
    for (i, q) in structure.spawn.quaternion.iter().enumerate() {
        put_f32(s, 4 * i, q);
    }
    put_mat3(s, 0x10, &structure.spawn.rotation);
    put_vec3(s, 0x34, &structure.spawn.position);
    put_vec3(s, 0x40, &structure.spawn.linear_velocity);
    put_vec3(s, 0x58, &structure.spawn.angular_velocity);
    for (i, t) in structure.spawn.tail.iter().enumerate() {
        put_f32(s, 0xa0 + 4 * i, t);
    }

    {
        /* The body reference lives in the tail of the file; patch through a
         * full write/read cycle offset. */
        let mut bytes = image.write();
        let off = image.header.body_reference_offset as usize;
        bytes[off..off + 4].copy_from_slice(
            &structure.body_reference_position.x.to_le_bytes());
        bytes[off + 4..off + 8].copy_from_slice(
            &structure.body_reference_position.y.to_le_bytes());
        bytes[off + 8..off + 12].copy_from_slice(
            &structure.body_reference_position.z.to_le_bytes());
        let re = VehicleImage::read(&bytes);
        image = re;
    }

    image.ground_ids = structure.ground_material_ids.clone();
    let original_ids: Vec<u32> =
        image.materials.iter().map(|m| m.object_id).collect();
    image.materials = structure
        .ground_materials
        .iter()
        .zip(original_ids.iter().chain(std::iter::repeat(&1)))
        .map(|(m, &object_id)| VehicleMaterialRecord {
            object_id, /* preserve the image's non-null identities */
            values: m.values,
            fake_contact_mask: m.fake_contact_mask as u32,
            fake_contact_period_x: m.fake_contact_period_x,
            fake_contact_period_z: m.fake_contact_period_z,
            fake_contact_impulse_scale: m.fake_contact_impulse_scale,
            fake_contact_impulse_limit: m.fake_contact_impulse_limit,
        })
        .collect();
    let d = &mut image.dyna;
    put_i32(d, 0x0c0, &structure.clamp_angular);
    put_f32(d, 0x0c4, &structure.max_angular_speed);
    put_i32(d, 0x340, &structure.mode);
    image.header.tick = structure.tick;
    image
}

/* ===========================================================================
 * Tuning <-> image
 * ========================================================================= */

/* The tuning views serialize through mirror structs (plain derives). */
mod tuning_mirrors {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
    pub struct Base {
        pub damper_max: f32,
        pub damper_min: f32,
        pub engine_model: i32,
        pub forward_speed_limit_scale: f32,
        pub engine_rpm_reverse_accel: f32,
        pub engine_rpm_accel: f32,
        pub engine_rpm_decel: f32,
        pub engine_rpm_turbo_decel: f32,
        pub engine_rpm_follow_accel: f32,
        pub engine_rpm_low_accel: f32,
        pub engine_rpm_high_decel: f32,
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
    #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
    pub struct Aux {
        pub steering_speed_base: f32,
        pub steering_speed_scale: f32,
        pub steering_slew_rate: f32,
        pub air_torque_linear: f32,
        pub air_torque_quadratic: f32,
        pub suspension_follow_rate: f32,
        pub air_control_window_ticks: u32,
        pub air_reversal_threshold: f32,
        pub water_buoyancy: f32,
        pub water_entry_speed_threshold: f32,
        pub water_entry_speed_minimum: f32,
        pub water_angular_drag_linear: f32,
        pub water_angular_drag_quadratic: f32,
    }
    #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
    pub struct Contact {
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
    #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
    pub struct Compute {
        pub event_c_level1_max: f32,
        pub grounded_drag_term: f32,
        pub active_contact_stop_threshold: f32,
        pub normal_turbo_factor: f32,
        pub roulette_turbo_factor: f32,
        pub normal_turbo_duration: u32,
        pub roulette_turbo_duration: u32,
        pub air_impulse_scale: f32,
        pub special_force_field_scale: f32,
        pub airborne_linear_drag: f32,
        pub grounded_force_field_scale: f32,
        pub airborne_force_field_scale: f32,
        pub normalized_force_divisor: f32,
        pub effect_curve_bias: f32,
        pub event_ab_level2_min: f32,
        pub event_ab_trigger: f32,
        pub event_c_trigger: f32,
    }
    #[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
    pub struct Model6 {
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
}

/// The complete tuning block as data (JSON-friendly mirrors + curves + gears).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CarTuning {
    pub base: tuning_mirrors::Base,
    pub aux: tuning_mirrors::Aux,
    pub contact: tuning_mirrors::Contact,
    pub compute: tuning_mirrors::Compute,
    pub model6: tuning_mirrors::Model6,
    /// Named curves (keys: the game's tuning offsets).
    pub curves: CurvesData,
    /// 4 tables of 6.
    pub gear_ratios: Vec<f32>,
    pub gear_upshift: Vec<f32>,
    pub gear_downshift: Vec<f32>,
    pub gear_aux: Vec<f32>,
}

/// A single curve.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CurveData {
    pub interpolation: i32,
    pub keys: Vec<[f32; 2]>,
}

/// The named curve set (the names match the C views).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct CurvesData {
    pub accel_from_speed: Option<CurveData>,
    pub rollover_lateral_from_speed: Option<CurveData>,
    pub max_side_friction_from_speed: Option<CurveData>,
    pub lateral_contact_slowdown_from_speed: Option<CurveData>,
    pub steer_slowdown_from_speed: Option<CurveData>,
    pub rollover_lateral_coef_from_angle: Option<CurveData>,
    pub steer_drive_torque_from_speed: Option<CurveData>,
    pub m4_steer_radius_from_speed: Option<CurveData>,
    pub m4_max_friction_force_from_speed: Option<CurveData>,
    pub m5_slipping_accel_from_speed: Option<CurveData>,
    pub water_friction_from_speed: Option<CurveData>,
    pub m6_damper_modulation: Option<CurveData>,
    pub m6_rear_gear_accel_from_speed: Option<CurveData>,
    pub m6_rollover_lateral_from_speed_ratio: Option<CurveData>,
    pub m6_burnout_radius_from_speed: Option<CurveData>,
    pub m6_lateral_speed_from_burnout_radius: Option<CurveData>,
    pub m6_donut_rollover_from_speed: Option<CurveData>,
    pub m6_burnout_rollover_from_speed: Option<CurveData>,
    pub air_vertical: Option<CurveData>,
    pub steering_angle: Option<CurveData>,
    pub effect: Option<CurveData>,
    pub contact_decay: Option<CurveData>,
    pub contact_rise: Option<CurveData>,
}

/* ===========================================================================
 * Tuning <-> image
 * ========================================================================= */

fn curve_of(desc: &VehicleCurveDescriptor, positions: &[f32], values: &[f32])
    -> CurveData {
    CurveData {
        interpolation: desc.interpolation,
        keys: positions.iter().zip(values)
            .map(|(p, v)| [*p, *v]).collect(),
    }
}

fn write_curve_back(
    desc: &VehicleCurveDescriptor,
    curve: &CurveData,
    positions: &mut Vec<f32>,
    values: &mut Vec<f32>,
) {
    assert_eq!(curve.keys.len(), positions.len(),
        "curve key counts are fixed by the image's descriptor table");
    for (i, k) in curve.keys.iter().enumerate() {
        positions[i] = k[0];
        values[i] = k[1];
    }
}

/// Extracts the tuning of a vehicle image as data.
pub fn tuning_from_image(image: &VehicleImage) -> CarTuning {
    let key = image.header.active_tuning_key as usize;
    let raw = image.tunings[key].1.clone();
    let rf = |o: usize| f32_at(&raw, o);
    let ri = |o: usize| i32_at(&raw, o);
    let ru = |o: usize| u32_at(&raw, o);

    let find = |field_offset: u32, owner_kind: u32|
        -> Option<(&VehicleCurveDescriptor, &Vec<f32>, &Vec<f32>)> {
        image.curves.iter().find(|(c, _, _)| {
            c.owner_kind == owner_kind
                && (owner_kind == 0 || c.owner_key as usize == key)
                && c.field_offset == field_offset
        })
        .map(|(c, p, v)| (c, p, v))
    };
    let curve = |off: u32| find(off, 1).map(|(c, p, v)| curve_of(c, p, v));
    let primary = |off: u32| find(off, 0).map(|(c, p, v)| curve_of(c, p, v));

    let mut gears: [Vec<f32>; GEARBOX_COUNT] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    const GEARBOX_OFFSETS: [u32; GEARBOX_COUNT] = [0x2c4, 0x2d4, 0x2e0, 0x304];
    for i in 0..GEARBOX_COUNT {
        if let Some((_, values)) = image.gearboxes.iter().find(|(g, _)| {
            g.owner_key as usize == key
                && g.field_offset == GEARBOX_OFFSETS[i]
        }) {
            gears[i] = values.clone();
        }
    }

    CarTuning {
        base: tuning_mirrors::Base {
            damper_max: rf(0x11c),
            damper_min: rf(0x120),
            engine_model: ri(0x354),
            forward_speed_limit_scale: rf(0x02c),
            engine_rpm_reverse_accel: rf(0x2ec),
            engine_rpm_accel: rf(0x2f0),
            engine_rpm_decel: rf(0x2f4),
            engine_rpm_turbo_decel: rf(0x31c),
            engine_rpm_follow_accel: rf(0x320),
            engine_rpm_low_accel: rf(0x324),
            engine_rpm_high_decel: rf(0x328),
            speed_32c: rf(0x32c),
            speed_330: rf(0x330),
            speed_334: rf(0x334),
            speed_338: rf(0x338),
            suspension_model: ri(0x350),
            suspension_stiffness: rf(0x114),
            suspension_damping: rf(0x118),
            suspension_rest_length: rf(0x124),
            suspension_scale: rf(0x128),
        },
        aux: tuning_mirrors::Aux {
            steering_speed_base: rf(0x06c),
            steering_speed_scale: rf(0x070),
            steering_slew_rate: rf(0x094),
            air_torque_linear: rf(0x158),
            air_torque_quadratic: rf(0x15c),
            suspension_follow_rate: rf(0x194),
            air_control_window_ticks: ru(0x364),
            air_reversal_threshold: rf(0x368),
            water_buoyancy: rf(0x204),
            water_entry_speed_threshold: rf(0x208),
            water_entry_speed_minimum: rf(0x20c),
            water_angular_drag_linear: rf(0x21c),
            water_angular_drag_quadratic: rf(0x220),
        },
        contact: tuning_mirrors::Contact {
            friction_force: rf(0x058),
            extra_friction_force: rf(0x05c),
            slope_adherence_min: rf(0x0d4),
            slope_adherence_max: rf(0x0d8),
            slope_secondary_min: rf(0x0dc),
            slope_secondary_max: rf(0x0e0),
            angular_y_scale: rf(0x0e8),
            angular_xz_scale: rf(0x0ec),
            damper_max: rf(0x11c),
            max_angular_speed: rf(0x14c),
            max_linear_speed_delta: rf(0x150),
            body_tangent_ratio: rf(0x170),
            body_tangent_ratio_material4: rf(0x174),
            restitution_air_material4: rf(0x178),
            restitution_air: rf(0x17c),
            restitution_ground: rf(0x184),
            restitution_ground_material4: rf(0x18c),
            lateral_linear: rf(0x1a8),
            lateral_quadratic: rf(0x1ac),
            lateral_ground_scale: rf(0x1c0),
            lateral_contact_duration_ticks: ru(0x1e8),
            wheel_contact_model: ri(0x350),
            friction_model: ri(0x354),
        },
        compute: tuning_mirrors::Compute {
            event_c_level1_max: rf(0x028),
            grounded_drag_term: rf(0x058),
            active_contact_stop_threshold: rf(0x0a4),
            normal_turbo_factor: rf(0x0f0),
            roulette_turbo_factor: rf(0x0f4),
            normal_turbo_duration: ru(0x0f8),
            roulette_turbo_duration: ru(0x0fc),
            air_impulse_scale: rf(0x104),
            special_force_field_scale: rf(0x108),
            airborne_linear_drag: rf(0x154),
            grounded_force_field_scale: rf(0x160),
            airborne_force_field_scale: rf(0x164),
            normalized_force_divisor: rf(0x228),
            effect_curve_bias: rf(0x384),
            event_ab_level2_min: rf(0x398),
            event_ab_trigger: rf(0x39c),
            event_c_trigger: rf(0x3a0),
        },
        model6: tuning_mirrors::Model6 {
            forward_speed_limit_scale: rf(0x02c),
            reverse_speed_limit_scale: rf(0x030),
            brake_base: rf(0x040),
            brake_speed_scale: rf(0x044),
            forward_brake_limit_sliding: rf(0x048),
            forward_brake_limit: rf(0x04c),
            speed_limit_force: rf(0x060),
            vertical_force_scale: rf(0x064),
            wheel_steer_sine_limit: rf(0x074),
            steer_slowdown_scale: rf(0x07c),
            wheel_torque_scale: rf(0x098),
            sliding_steer_torque_scale: rf(0x09c),
            lateral_force_scale: rf(0x0a4),
            sliding_lateral_limit_scale: rf(0x0b0),
            lateral_overflow_blend: rf(0x0b4),
            wheel_overflow_blend: rf(0x0e4),
            vertical_force_divisor: rf(0x160),
            traction_loss_scale: rf(0x200),
            burnout_trigger_scale: rf(0x228),
            burnout_trigger_limit: rf(0x22c),
            rollover_axis_min_length: rf(0x234),
            rollover_torque_x_scale: rf(0x238),
            rollover_torque_z_scale: rf(0x23c),
            sliding_brake_scale: rf(0x240),
            braking_lateral_limit_scale: rf(0x244),
            reverse_brake_limit_sliding: rf(0x248),
            reverse_brake_limit: rf(0x24c),
            burnout_speed_max: rf(0x254),
            burnout_speed_min: rf(0x258),
            donut_lateral_force_scale: rf(0x264),
            donut_yaw_angle_scale: rf(0x268),
            donut_steer_linear: rf(0x26c),
            donut_steer_quadratic: rf(0x270),
            donut_countersteer_scale: rf(0x274),
            donut_radius_exponent: rf(0x278),
            donut_radial_speed_exponent: rf(0x27c),
            donut_radius_min: rf(0x280),
            donut_lateral_speed_limit: rf(0x284),
            donut_normal_angle_limit: rf(0x28c),
            donut_angle_positive_limit: rf(0x290),
            donut_angle_negative_limit: rf(0x294),
            burnout_enter_ticks: ru(0x298),
            burnout_enter_accel_scale: rf(0x29c),
            burnout_enter_lateral_scale: rf(0x2a0),
            burnout_exit_ticks: ru(0x2a8),
            burnout_exit_accel_scale: rf(0x2ac),
            burnout_exit_extra_accel: rf(0x2b8),
            material6_longitudinal_scale: rf(0x33c),
            material6_gas_denominator: rf(0x340),
            material6_vertical_shape: rf(0x344),
            material6_vertical_scale: rf(0x348),
        },
        curves: CurvesData {
            accel_from_speed: curve(0x034),
            lateral_contact_slowdown_from_speed: curve(0x068),
            steer_slowdown_from_speed: curve(0x078),
            steer_drive_torque_from_speed: curve(0x0a0),
            max_side_friction_from_speed: curve(0x0ac),
            rollover_lateral_from_speed: curve(0x0b8),
            rollover_lateral_coef_from_angle: curve(0x0bc),
            m4_steer_radius_from_speed: curve(0x1b4),
            m4_max_friction_force_from_speed: curve(0x1bc),
            m5_slipping_accel_from_speed: curve(0x1e0),
            water_friction_from_speed: curve(0x218),
            m6_damper_modulation: curve(0x224),
            m6_rear_gear_accel_from_speed: curve(0x230),
            m6_rollover_lateral_from_speed_ratio: curve(0x250),
            m6_burnout_radius_from_speed: curve(0x25c),
            m6_lateral_speed_from_burnout_radius: curve(0x260),
            m6_donut_rollover_from_speed: curve(0x288),
            m6_burnout_rollover_from_speed: curve(0x2a4),
            air_vertical: curve(0x36c),
            steering_angle: curve(0x378),
            effect: curve(0x380),
            contact_decay: primary(0x044),
            contact_rise: primary(0x048),
        },
        gear_ratios: gears[0].clone(),
        gear_upshift: gears[1].clone(),
        gear_downshift: gears[2].clone(),
        gear_aux: gears[3].clone(),
    }
}

/// Patches a tuning block into the image (layout-preserving; curve key
/// counts are fixed by the descriptor table, as in the C's
/// `vimg_write_tuning`).
pub fn image_with_tuning(
    mut image: VehicleImage,
    tuning: &CarTuning,
) -> VehicleImage {
    let key = image.header.active_tuning_key as usize;
    let raw = &mut image.tunings[key].1;
    let wf = |raw: &mut [u8], o: usize, v: &f32| {
        raw[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };
    let wi = |raw: &mut [u8], o: usize, v: &i32| {
        raw[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };
    let wu = |raw: &mut [u8], o: usize, v: &u32| {
        raw[o..o + 4].copy_from_slice(&v.to_le_bytes());
    };

    let t = &tuning.base;
    wf(raw, 0x11c, &t.damper_max);
    wf(raw, 0x120, &t.damper_min);
    wi(raw, 0x354, &t.engine_model);
    wf(raw, 0x02c, &t.forward_speed_limit_scale);
    wf(raw, 0x2ec, &t.engine_rpm_reverse_accel);
    wf(raw, 0x2f0, &t.engine_rpm_accel);
    wf(raw, 0x2f4, &t.engine_rpm_decel);
    wf(raw, 0x31c, &t.engine_rpm_turbo_decel);
    wf(raw, 0x320, &t.engine_rpm_follow_accel);
    wf(raw, 0x324, &t.engine_rpm_low_accel);
    wf(raw, 0x328, &t.engine_rpm_high_decel);
    wf(raw, 0x32c, &t.speed_32c);
    wf(raw, 0x330, &t.speed_330);
    wf(raw, 0x334, &t.speed_334);
    wf(raw, 0x338, &t.speed_338);
    wi(raw, 0x350, &t.suspension_model);
    wf(raw, 0x114, &t.suspension_stiffness);
    wf(raw, 0x118, &t.suspension_damping);
    wf(raw, 0x124, &t.suspension_rest_length);
    wf(raw, 0x128, &t.suspension_scale);

    let t = &tuning.aux;
    wf(raw, 0x06c, &t.steering_speed_base);
    wf(raw, 0x070, &t.steering_speed_scale);
    wf(raw, 0x094, &t.steering_slew_rate);
    wf(raw, 0x158, &t.air_torque_linear);
    wf(raw, 0x15c, &t.air_torque_quadratic);
    wf(raw, 0x194, &t.suspension_follow_rate);
    wu(raw, 0x364, &t.air_control_window_ticks);
    wf(raw, 0x368, &t.air_reversal_threshold);
    wf(raw, 0x204, &t.water_buoyancy);
    wf(raw, 0x208, &t.water_entry_speed_threshold);
    wf(raw, 0x20c, &t.water_entry_speed_minimum);
    wf(raw, 0x21c, &t.water_angular_drag_linear);
    wf(raw, 0x220, &t.water_angular_drag_quadratic);

    let t = &tuning.contact;
    wf(raw, 0x058, &t.friction_force);
    wf(raw, 0x05c, &t.extra_friction_force);
    wf(raw, 0x0d4, &t.slope_adherence_min);
    wf(raw, 0x0d8, &t.slope_adherence_max);
    wf(raw, 0x0dc, &t.slope_secondary_min);
    wf(raw, 0x0e0, &t.slope_secondary_max);
    wf(raw, 0x0e8, &t.angular_y_scale);
    wf(raw, 0x0ec, &t.angular_xz_scale);
    wf(raw, 0x11c, &t.damper_max);
    wf(raw, 0x14c, &t.max_angular_speed);
    wf(raw, 0x150, &t.max_linear_speed_delta);
    wf(raw, 0x170, &t.body_tangent_ratio);
    wf(raw, 0x174, &t.body_tangent_ratio_material4);
    wf(raw, 0x178, &t.restitution_air_material4);
    wf(raw, 0x17c, &t.restitution_air);
    wf(raw, 0x184, &t.restitution_ground);
    wf(raw, 0x18c, &t.restitution_ground_material4);
    wf(raw, 0x1a8, &t.lateral_linear);
    wf(raw, 0x1ac, &t.lateral_quadratic);
    wf(raw, 0x1c0, &t.lateral_ground_scale);
    wu(raw, 0x1e8, &t.lateral_contact_duration_ticks);
    wi(raw, 0x350, &t.wheel_contact_model);
    wi(raw, 0x354, &t.friction_model);

    let t = &tuning.compute;
    wf(raw, 0x028, &t.event_c_level1_max);
    wf(raw, 0x058, &t.grounded_drag_term);
    wf(raw, 0x0a4, &t.active_contact_stop_threshold);
    wf(raw, 0x0f0, &t.normal_turbo_factor);
    wf(raw, 0x0f4, &t.roulette_turbo_factor);
    wu(raw, 0x0f8, &t.normal_turbo_duration);
    wu(raw, 0x0fc, &t.roulette_turbo_duration);
    wf(raw, 0x104, &t.air_impulse_scale);
    wf(raw, 0x108, &t.special_force_field_scale);
    wf(raw, 0x154, &t.airborne_linear_drag);
    wf(raw, 0x160, &t.grounded_force_field_scale);
    wf(raw, 0x164, &t.airborne_force_field_scale);
    wf(raw, 0x228, &t.normalized_force_divisor);
    wf(raw, 0x384, &t.effect_curve_bias);
    wf(raw, 0x398, &t.event_ab_level2_min);
    wf(raw, 0x39c, &t.event_ab_trigger);
    wf(raw, 0x3a0, &t.event_c_trigger);

    let t = &tuning.model6;
    wf(raw, 0x02c, &t.forward_speed_limit_scale);
    wf(raw, 0x030, &t.reverse_speed_limit_scale);
    wf(raw, 0x040, &t.brake_base);
    wf(raw, 0x044, &t.brake_speed_scale);
    wf(raw, 0x048, &t.forward_brake_limit_sliding);
    wf(raw, 0x04c, &t.forward_brake_limit);
    wf(raw, 0x060, &t.speed_limit_force);
    wf(raw, 0x064, &t.vertical_force_scale);
    wf(raw, 0x074, &t.wheel_steer_sine_limit);
    wf(raw, 0x07c, &t.steer_slowdown_scale);
    wf(raw, 0x098, &t.wheel_torque_scale);
    wf(raw, 0x09c, &t.sliding_steer_torque_scale);
    wf(raw, 0x0a4, &t.lateral_force_scale);
    wf(raw, 0x0b0, &t.sliding_lateral_limit_scale);
    wf(raw, 0x0b4, &t.lateral_overflow_blend);
    wf(raw, 0x0e4, &t.wheel_overflow_blend);
    wf(raw, 0x160, &t.vertical_force_divisor);
    wf(raw, 0x200, &t.traction_loss_scale);
    wf(raw, 0x228, &t.burnout_trigger_scale);
    wf(raw, 0x22c, &t.burnout_trigger_limit);
    wf(raw, 0x234, &t.rollover_axis_min_length);
    wf(raw, 0x238, &t.rollover_torque_x_scale);
    wf(raw, 0x23c, &t.rollover_torque_z_scale);
    wf(raw, 0x240, &t.sliding_brake_scale);
    wf(raw, 0x244, &t.braking_lateral_limit_scale);
    wf(raw, 0x248, &t.reverse_brake_limit_sliding);
    wf(raw, 0x24c, &t.reverse_brake_limit);
    wf(raw, 0x254, &t.burnout_speed_max);
    wf(raw, 0x258, &t.burnout_speed_min);
    wf(raw, 0x264, &t.donut_lateral_force_scale);
    wf(raw, 0x268, &t.donut_yaw_angle_scale);
    wf(raw, 0x26c, &t.donut_steer_linear);
    wf(raw, 0x270, &t.donut_steer_quadratic);
    wf(raw, 0x274, &t.donut_countersteer_scale);
    wf(raw, 0x278, &t.donut_radius_exponent);
    wf(raw, 0x27c, &t.donut_radial_speed_exponent);
    wf(raw, 0x280, &t.donut_radius_min);
    wf(raw, 0x284, &t.donut_lateral_speed_limit);
    wf(raw, 0x28c, &t.donut_normal_angle_limit);
    wf(raw, 0x290, &t.donut_angle_positive_limit);
    wf(raw, 0x294, &t.donut_angle_negative_limit);
    wu(raw, 0x298, &t.burnout_enter_ticks);
    wf(raw, 0x29c, &t.burnout_enter_accel_scale);
    wf(raw, 0x2a0, &t.burnout_enter_lateral_scale);
    wu(raw, 0x2a8, &t.burnout_exit_ticks);
    wf(raw, 0x2ac, &t.burnout_exit_accel_scale);
    wf(raw, 0x2b8, &t.burnout_exit_extra_accel);
    wf(raw, 0x33c, &t.material6_longitudinal_scale);
    wf(raw, 0x340, &t.material6_gas_denominator);
    wf(raw, 0x344, &t.material6_vertical_shape);
    wf(raw, 0x348, &t.material6_vertical_scale);

    /* Curves: rewrite positions/values in place (key counts fixed). */
    let entries: [(u32, &Option<CurveData>); 23] = [
        (0x034, &tuning.curves.accel_from_speed),
        (0x068, &tuning.curves.lateral_contact_slowdown_from_speed),
        (0x078, &tuning.curves.steer_slowdown_from_speed),
        (0x0a0, &tuning.curves.steer_drive_torque_from_speed),
        (0x0ac, &tuning.curves.max_side_friction_from_speed),
        (0x0b8, &tuning.curves.rollover_lateral_from_speed),
        (0x0bc, &tuning.curves.rollover_lateral_coef_from_angle),
        (0x1b4, &tuning.curves.m4_steer_radius_from_speed),
        (0x1bc, &tuning.curves.m4_max_friction_force_from_speed),
        (0x1e0, &tuning.curves.m5_slipping_accel_from_speed),
        (0x218, &tuning.curves.water_friction_from_speed),
        (0x224, &tuning.curves.m6_damper_modulation),
        (0x230, &tuning.curves.m6_rear_gear_accel_from_speed),
        (0x250, &tuning.curves.m6_rollover_lateral_from_speed_ratio),
        (0x25c, &tuning.curves.m6_burnout_radius_from_speed),
        (0x260, &tuning.curves.m6_lateral_speed_from_burnout_radius),
        (0x288, &tuning.curves.m6_donut_rollover_from_speed),
        (0x2a4, &tuning.curves.m6_burnout_rollover_from_speed),
        (0x36c, &tuning.curves.air_vertical),
        (0x378, &tuning.curves.steering_angle),
        (0x380, &tuning.curves.effect),
        (0x044, &tuning.curves.contact_decay),
        (0x048, &tuning.curves.contact_rise),
    ];
    for (off, data) in entries {
        if let Some(curve) = data {
            let matches = image.curves.iter_mut().find(|(c, _, _)| {
                c.field_offset == off
                    && if off == 0x044 || off == 0x048 {
                        c.owner_kind == 0
                    } else {
                        c.owner_kind == 1 && c.owner_key as usize == key
                    }
            });
            if let Some((c, positions, values)) = matches {
                write_curve_back(c, curve, positions, values);
                c.interpolation = curve.interpolation;
            }
        }
    }

    /* Gearboxes. */
    const GEARBOX_OFFSETS: [u32; GEARBOX_COUNT] = [0x2c4, 0x2d4, 0x2e0, 0x304];
    let tables: [&Vec<f32>; GEARBOX_COUNT] = [
        &tuning.gear_ratios,
        &tuning.gear_upshift,
        &tuning.gear_downshift,
        &tuning.gear_aux,
    ];
    for i in 0..GEARBOX_COUNT {
        if let Some((_, values)) = image.gearboxes.iter_mut().find(|(g, _)| {
            g.owner_key as usize == key
                && g.field_offset == GEARBOX_OFFSETS[i]
        }) {
            assert_eq!(tables[i].len(), GEARBOX_VALUE_COUNT);
            values.copy_from_slice(tables[i]);
        }
    }

    image
}
