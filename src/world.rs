//! The world: `TMNFM6G1` vehicle-image decode, the collision zone, and the
//! 100 Hz tick (`src/world.c` + `src/physics.c` + `src/hms_item.c` +
//! `src/vehicle_respawn.c`, transliterated).
//!
//! # Layout
//!
//! * [`WorldCold`] — everything immutable after creation (the vehicle blob,
//!   the decoded [`VehicleTuning`], the player response body, gravity),
//!   shareable across simulations of the same car.
//! * [`World`] — the mutable per-simulation state: the dyna, the vehicle,
//!   the collision trees (root + pre-order nodes + wheel slots), contact
//!   buffers, the zone state and the collision buffer.
//!
//! The C's pointer topology (`World_LinkPointers` and its device relinking)
//! disappears: ownership replaces relinking.

use crate::buffer::{CHmsCollisionBuffer, SHmsSphereBufferContact};
use crate::collision::shms_sphere_buffer_contact_merge_and_add_to_collisions;
use crate::collision::*;
use crate::dyna::{CHmsDyna, CHmsDynaParams, CHmsStateDyna};
use crate::gm::*;
use crate::response::*;
use crate::track::TmnfTrack;
use crate::vehicle::contact::absorb_contact as vehicle_absorb_contact;
use crate::vehicle::compute::{cscene_vehicle_car_compute_forces, ComputeExternal};
use crate::vehicle::curve::{cfunc_keys_compile, cfunc_keys_real_get_value};
use crate::vehicle::{CFuncKeysReal, GmSpringFloat, Model6External,
    TMNFVehicleTurboType, Vehicle, VehicleCtx, VehicleTuning};
use std::mem::size_of;
use std::sync::Arc;

pub const CAR_SIZE: usize = 0x878;
pub const VEHICLE_STRUCT_SIZE: usize = 0x50;
pub const TUNING_SIZE: usize = 0x3ac;
pub const WHEEL_SIZE: usize = 0x2fc;
pub const DYNA_SIZE: usize = 0x344;
pub const PARAMS_SIZE: usize = 0x5c;
pub const STATE_SIZE: usize = 0xb4;
pub const ACTIVE_CURVE_COUNT: usize = 21;
pub const PRIMARY_CURVE_COUNT: usize = 2;
pub const GEARBOX_COUNT: usize = 4;
pub const GEARBOX_VALUE_COUNT: usize = 6;
pub const VEHICLE_COLLISION_MAX_NODES: usize = 8;

const CURVE_ACCEL: usize = 0;
const CURVE_LATERAL_CONTACT_SLOWDOWN: usize = 1;
const CURVE_STEER_SLOWDOWN: usize = 2;
const CURVE_STEER_DRIVE_TORQUE: usize = 3;
const CURVE_MAX_SIDE_FRICTION: usize = 4;
const CURVE_ROLLOVER_LATERAL: usize = 5;
const CURVE_ROLLOVER_ANGLE: usize = 6;
const CURVE_M4_STEER_RADIUS: usize = 7;
const CURVE_M4_MAX_FRICTION: usize = 8;
const CURVE_M5_SLIPPING_ACCEL: usize = 9;
const CURVE_WATER_FRICTION: usize = 10;
const CURVE_M6_DAMPER_MODULATION: usize = 11;
const CURVE_M6_REAR_GEAR_ACCEL: usize = 12;
const CURVE_M6_ROLLOVER_RATIO: usize = 13;
const CURVE_M6_BURNOUT_RADIUS: usize = 14;
const CURVE_M6_BURNOUT_LATERAL_SPEED: usize = 15;
const CURVE_M6_DONUT_ROLLOVER: usize = 16;
const CURVE_M6_BURNOUT_ROLLOVER: usize = 17;
const CURVE_AIR_VERTICAL: usize = 18;
const CURVE_STEERING_ANGLE: usize = 19;
const CURVE_EFFECT: usize = 20;

/// Tuning field offsets of the 21 active curves and the two primary curves.
const ACTIVE_CURVE_OFFSETS: [u32; ACTIVE_CURVE_COUNT] = [
    0x034, 0x068, 0x078, 0x0a0, 0x0ac, 0x0b8, 0x0bc,
    0x1b4, 0x1bc, 0x1e0, 0x218, 0x224, 0x230, 0x250,
    0x25c, 0x260, 0x288, 0x2a4, 0x36c, 0x378, 0x380,
];
const PRIMARY_CURVE_OFFSETS: [u32; PRIMARY_CURVE_COUNT] = [0x044, 0x048];
const GEARBOX_OFFSETS: [u32; GEARBOX_COUNT] = [0x2c4, 0x2d4, 0x2e0, 0x304];

const VEHICLE_CURVE_OWNER_STRUCT: u32 = 0;
const VEHICLE_CURVE_OWNER_TUNING: u32 = 1;

pub const TMNF_VEHICLE_SNAPSHOT_MAGIC: &[u8; 8] = b"TMNFM6G1";
pub const TMNF_VEHICLE_SNAPSHOT_VERSION: u32 = 4;

const VEHICLE_COLLISION_TREE_BODY: u32 = 0;
const VEHICLE_COLLISION_TREE_WHEEL: u32 = 1;
const VEHICLE_COLLISION_TREE_ROOT: u32 = 2;
const VEHICLE_COLLISION_KIND_MASK: u32 = 0xff;
const VEHICLE_COLLISION_CHILD_SHIFT: u32 = 8;

/// `TMNF_WATER_IMPULSE_POSITIONS` (vehicle_water_tuning.h), float bits.
pub const TMNF_WATER_IMPULSE_POSITIONS: [f32; 4] = [
    0.0,
    f32::from_bits(0x3E99_999A), /* 0x1.99999ap-2f */
    0.5,
    1.0,
];
/// `TMNF_WATER_IMPULSE_VERTICAL_VALUES`.
pub const TMNF_WATER_IMPULSE_VERTICAL_VALUES: [f32; 4] = [
    f32::from_bits(0x3FCC_CCCC), /* 0x1.ccccccp-1f */
    f32::from_bits(0x3FCC_CCCC),
    1.5,
    1.5,
];
/// `TMNF_WATER_IMPULSE_HORIZONTAL_VALUES`.
pub const TMNF_WATER_IMPULSE_HORIZONTAL_VALUES: [f32; 4] = [
    f32::from_bits(0x3F66_6666), /* 0x1.666666p-1f */
    f32::from_bits(0x3F66_6666),
    f32::from_bits(0x3F33_3334), /* 0x1.333334p-1f */
    f32::from_bits(0x3F33_3334),
];

/// SHA-256 of the canonical TMNF 2.11.26 executable.
pub const TMNF_21126_EXE_SHA256: [u8; 32] = [
    0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
    0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
    0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
    0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
];

/* ---------------------------------------------------------------------------
 * On-disk layouts (packed, little-endian, read by offset)
 * ------------------------------------------------------------------------- */

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn vec3_at(b: &[u8], o: usize) -> GmVec3 {
    GmVec3 { x: f32_at(b, o), y: f32_at(b, o + 4), z: f32_at(b, o + 8) }
}
fn iso4_at(b: &[u8], o: usize) -> GmIso4 {
    let mut m = [0.0f32; 9];
    for i in 0..9 {
        m[i] = f32_at(b, o + 4 * i);
    }
    GmIso4 { m, t: [f32_at(b, o + 36), f32_at(b, o + 40), f32_at(b, o + 44)] }
}
fn box_at(b: &[u8], o: usize) -> GmBoxAligned {
    GmBoxAligned { center: vec3_at(b, o), half_extent: vec3_at(b, o + 12) }
}

/// The 320-byte vehicle snapshot header.
#[derive(Clone, Debug)]
pub struct VehicleSnapshotHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub total_size: u32,
    pub header_size: u32,
    pub phase: u32,
    pub tick: u32,
    pub car_id: u32,
    pub vehicle_struct_id: u32,
    pub tuning_container_id: u32,
    pub wheels_id: u32,
    pub item_id: u32,
    pub corpus_id: u32,
    pub dyna_id: u32,
    pub params_id: u32,
    pub state_id: u32,
    pub model_iso_id: u32,
    pub body_reference_id: u32,
    pub wheel_count: u32,
    pub ground_id_count: u32,
    pub ground_material_count: u32,
    pub tuning_count: u32,
    pub active_tuning_key: u32,
    pub curve_count: u32,
    pub gearbox_count: u32,
    pub car_offset: u32,
    pub vehicle_struct_offset: u32,
    pub tuning_descriptors_offset: u32,
    pub wheels_offset: u32,
    pub dyna_offset: u32,
    pub params_offset: u32,
    pub state_offset: u32,
    pub curve_descriptors_offset: u32,
    pub gearbox_descriptors_offset: u32,
    pub ground_ids_offset: u32,
    pub ground_materials_offset: u32,
    pub model_iso_offset: u32,
    pub body_reference_offset: u32,
    pub curve_data_offset: u32,
    pub gearbox_data_offset: u32,
    pub collision_child_count: u32,
    pub collision_root_offset: u32,
    pub collision_children_offset: u32,
    pub collision_material_ids_offset: u32,
    pub model_value: f32,
    pub lateral_force_factor: f32,
    pub longitudinal_force_factor: f32,
    pub steering_angle: f32,
    pub grounded: i32,
    pub existing_force: GmVec3,
    pub local_speed: GmVec3,
    pub local_angular_speed: GmVec3,
    pub material: [f32; 4],
    pub sliding: i32,
    pub brake_force: f32,
    pub source_exe_sha256: [u8; 32],
    pub source_track_sha256: [u8; 32],
}

impl VehicleSnapshotHeader {
    /// Packed little-endian layout (verified: 0xc4+12=0xd0, 0xe8+16=0xf8,
    /// 0x100+32=0x120, total 0x140 = 320).
    pub fn from_bytes(b: &[u8]) -> VehicleSnapshotHeader {
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&b[0..8]);
        let g = |o: usize| u32_at(b, o);
        let mut material = [0.0f32; 4];
        for i in 0..4 {
            material[i] = f32_at(b, 0xe8 + 4 * i);
        }
        let mut exe = [0u8; 32];
        exe.copy_from_slice(&b[0x100..0x120]);
        let mut trk = [0u8; 32];
        trk.copy_from_slice(&b[0x120..0x140]);
        VehicleSnapshotHeader {
            magic,
            version: g(0x08),
            total_size: g(0x0c),
            header_size: g(0x10),
            phase: g(0x14),
            tick: g(0x18),
            car_id: g(0x1c),
            vehicle_struct_id: g(0x20),
            tuning_container_id: g(0x24),
            wheels_id: g(0x28),
            item_id: g(0x2c),
            corpus_id: g(0x30),
            dyna_id: g(0x34),
            params_id: g(0x38),
            state_id: g(0x3c),
            model_iso_id: g(0x40),
            body_reference_id: g(0x44),
            wheel_count: g(0x48),
            ground_id_count: g(0x4c),
            ground_material_count: g(0x50),
            tuning_count: g(0x54),
            active_tuning_key: g(0x58),
            curve_count: g(0x5c),
            gearbox_count: g(0x60),
            car_offset: g(0x64),
            vehicle_struct_offset: g(0x68),
            tuning_descriptors_offset: g(0x6c),
            wheels_offset: g(0x70),
            dyna_offset: g(0x74),
            params_offset: g(0x78),
            state_offset: g(0x7c),
            curve_descriptors_offset: g(0x80),
            gearbox_descriptors_offset: g(0x84),
            ground_ids_offset: g(0x88),
            ground_materials_offset: g(0x8c),
            model_iso_offset: g(0x90),
            body_reference_offset: g(0x94),
            curve_data_offset: g(0x98),
            gearbox_data_offset: g(0x9c),
            collision_child_count: g(0xa0),
            collision_root_offset: g(0xa4),
            collision_children_offset: g(0xa8),
            collision_material_ids_offset: g(0xac),
            model_value: f32_at(b, 0xb0),
            lateral_force_factor: f32_at(b, 0xb4),
            longitudinal_force_factor: f32_at(b, 0xb8),
            steering_angle: f32_at(b, 0xbc),
            grounded: i32_at(b, 0xc0),
            existing_force: vec3_at(b, 0xc4),
            local_speed: vec3_at(b, 0xd0),
            local_angular_speed: vec3_at(b, 0xdc),
            material,
            sliding: i32_at(b, 0xf8),
            brake_force: f32_at(b, 0xfc),
            source_exe_sha256: exe,
            source_track_sha256: trk,
        }
    }
}

/// 16-byte tuning descriptor.
#[derive(Clone, Copy, Debug)]
pub struct VehicleTuningDescriptor {
    pub key: u32,
    pub object_id: u32,
    pub raw_offset: u32,
    pub raw_size: u32,
}

/// 40-byte curve descriptor.
#[derive(Clone, Copy, Debug)]
pub struct VehicleCurveDescriptor {
    pub owner_kind: u32,
    pub owner_key: u32,
    pub field_offset: u32,
    pub object_id: u32,
    pub positions_id: u32,
    pub values_id: u32,
    pub count: u32,
    pub interpolation: i32,
    pub positions_offset: u32,
    pub values_offset: u32,
}

/// 24-byte gearbox descriptor.
#[derive(Clone, Copy, Debug)]
pub struct VehicleGearboxDescriptor {
    pub owner_key: u32,
    pub field_offset: u32,
    pub buffer_id: u32,
    pub data_id: u32,
    pub count: u32,
    pub data_offset: u32,
}

/// 40-byte ground material record.
#[derive(Clone, Copy, Debug)]
pub struct VehicleMaterialRecord {
    pub object_id: u32,
    pub values: [f32; 4],
    /// Flag: the material references the shared 128x128 fake-contact image.
    pub fake_contact_mask: u32,
    pub fake_contact_period_x: f32,
    pub fake_contact_period_z: f32,
    pub fake_contact_impulse_scale: f32,
    pub fake_contact_impulse_limit: f32,
}

/// 132-byte collision tree record.
#[derive(Clone, Debug)]
pub struct VehicleCollisionTreeRecord {
    pub object_id: u32,
    pub flags: u32,
    pub surface_id: u32,
    pub geometry_id: u32,
    pub kind: u32,
    pub wheel_index: u32,
    pub material_index: u32,
    pub material_count: u32,
    pub geometry_material_index: u16,
    pub geometry_type: u8,
    pub geometry_reserved: u8,
    pub box_aligned: GmBoxAligned,
    pub local_iso: GmIso4,
    pub shape: [u8; 24],
}

fn read_tuning_descriptor(b: &[u8], o: usize) -> VehicleTuningDescriptor {
    VehicleTuningDescriptor {
        key: u32_at(b, o),
        object_id: u32_at(b, o + 4),
        raw_offset: u32_at(b, o + 8),
        raw_size: u32_at(b, o + 12),
    }
}

fn read_curve_descriptor(b: &[u8], o: usize) -> VehicleCurveDescriptor {
    VehicleCurveDescriptor {
        owner_kind: u32_at(b, o),
        owner_key: u32_at(b, o + 4),
        field_offset: u32_at(b, o + 8),
        object_id: u32_at(b, o + 12),
        positions_id: u32_at(b, o + 16),
        values_id: u32_at(b, o + 20),
        count: u32_at(b, o + 24),
        interpolation: i32_at(b, o + 28),
        positions_offset: u32_at(b, o + 32),
        values_offset: u32_at(b, o + 36),
    }
}

fn read_gearbox_descriptor(b: &[u8], o: usize) -> VehicleGearboxDescriptor {
    VehicleGearboxDescriptor {
        owner_key: u32_at(b, o),
        field_offset: u32_at(b, o + 4),
        buffer_id: u32_at(b, o + 8),
        data_id: u32_at(b, o + 12),
        count: u32_at(b, o + 16),
        data_offset: u32_at(b, o + 20),
    }
}

fn read_material_record(b: &[u8], o: usize) -> VehicleMaterialRecord {
    let mut values = [0.0f32; 4];
    for i in 0..4 {
        values[i] = f32_at(b, o + 4 + 4 * i);
    }
    VehicleMaterialRecord {
        object_id: u32_at(b, o),
        values,
        fake_contact_mask: u32_at(b, o + 20),
        fake_contact_period_x: f32_at(b, o + 24),
        fake_contact_period_z: f32_at(b, o + 28),
        fake_contact_impulse_scale: f32_at(b, o + 32),
        fake_contact_impulse_limit: f32_at(b, o + 36),
    }
}

fn read_collision_tree_record(b: &[u8], o: usize) -> VehicleCollisionTreeRecord {
    let mut shape = [0u8; 24];
    shape.copy_from_slice(&b[o + 0x6c..o + 0x84]);
    VehicleCollisionTreeRecord {
        object_id: u32_at(b, o),
        flags: u32_at(b, o + 4),
        surface_id: u32_at(b, o + 8),
        geometry_id: u32_at(b, o + 12),
        kind: u32_at(b, o + 16),
        wheel_index: u32_at(b, o + 20),
        material_index: u32_at(b, o + 24),
        material_count: u32_at(b, o + 28),
        geometry_material_index: u16::from_le_bytes(
            b[o + 32..o + 34].try_into().unwrap()),
        geometry_type: b[o + 34],
        geometry_reserved: b[o + 35],
        box_aligned: box_at(b, o + 36),
        local_iso: iso4_at(b, o + 60),
        shape,
    }
}

/// `tree_world_iso` for the world's [`VehicleTree`] nodes.
fn vehicle_tree_world_iso(node: &VehicleTree, parent: &GmIso4) -> GmIso4 {
    if (node.flags & 4u32) == 0 {
        return *parent;
    }
    let mut out = node.local_iso;
    out.mult(parent);
    out
}

fn graph_section(blob: &[u8], offset: u32, length: u32, what: &str) -> usize {
    let size = blob.len() as u32;
    if offset > size || length > size - offset {
        panic!("tmnf world: vehicle graph section is out of bounds: {}", what);
    }
    offset as usize
}

fn contains_offset(offsets: &[u32], offset: u32) -> bool {
    offsets.contains(&offset)
}

/* ---------------------------------------------------------------------------
 * Validation (validate_vehicle_graph)
 * ------------------------------------------------------------------------- */

/// The validated graph: header + descriptor views into the blob.
pub(crate) struct VehicleGraph<'a> {
    pub header: VehicleSnapshotHeader,
    pub tuning_descriptors: Vec<VehicleTuningDescriptor>,
    pub curve_descriptors: Vec<VehicleCurveDescriptor>,
    pub gearbox_descriptors: Vec<VehicleGearboxDescriptor>,
    pub materials: Vec<VehicleMaterialRecord>,
    pub collision_root: VehicleCollisionTreeRecord,
    pub collision_children: Vec<VehicleCollisionTreeRecord>,
    pub blob: &'a [u8],
}

pub(crate) fn validate_vehicle_graph<'a>(
    blob: &'a [u8],
    expected_track_sha256: Option<&[u8; 32]>,
    pin_provenance: bool,
) -> VehicleGraph<'a> {
    if blob.len() < 320 {
        panic!("tmnf world: truncated vehicle graph");
    }
    let header = VehicleSnapshotHeader::from_bytes(blob);
    if &header.magic != TMNF_VEHICLE_SNAPSHOT_MAGIC
        || header.version != TMNF_VEHICLE_SNAPSHOT_VERSION
        || header.total_size != blob.len() as u32
        || header.header_size != 320
        || header.phase != 0
        || header.car_id == 0
        || header.vehicle_struct_id == 0
        || header.tuning_container_id == 0
        || header.wheels_id == 0
        || header.item_id == 0
        || header.corpus_id == 0
        || header.dyna_id == 0
        || header.params_id == 0
        || header.state_id == 0
        || header.model_iso_id == 0
        || header.body_reference_id == 0
        || header.wheel_count != 4
        || header.ground_id_count != 31
        || header.ground_material_count == 0
        || header.ground_material_count > crate::surface::TMNF_MAX_GROUND_MATERIALS
        || header.tuning_count == 0
        || header.tuning_count > 64
        || header.active_tuning_key >= header.tuning_count
        || header.curve_count == 0
        || header.curve_count > 4096
        || header.collision_child_count == 0
        || header.collision_child_count as usize > VEHICLE_COLLISION_MAX_NODES
        || header.gearbox_count != header.tuning_count * GEARBOX_COUNT as u32
        || (pin_provenance
            && (header.source_exe_sha256 != TMNF_21126_EXE_SHA256
                || expected_track_sha256
                    .map(|t| header.source_track_sha256 != *t)
                    .unwrap_or(false)))
    {
        panic!("tmnf world: invalid vehicle graph header");
    }
    graph_section(blob, header.car_offset, CAR_SIZE as u32, "car");
    graph_section(blob, header.vehicle_struct_offset,
        VEHICLE_STRUCT_SIZE as u32, "vehicle struct");
    graph_section(blob, header.wheels_offset, 4 * WHEEL_SIZE as u32, "wheels");
    graph_section(blob, header.dyna_offset, DYNA_SIZE as u32, "dyna");
    graph_section(blob, header.params_offset, PARAMS_SIZE as u32, "params");
    graph_section(blob, header.state_offset, STATE_SIZE as u32, "state");
    graph_section(blob, header.ground_ids_offset, 31 * 4, "ground ids");
    graph_section(blob, header.ground_materials_offset,
        header.ground_material_count * 40, "ground materials");
    graph_section(blob, header.model_iso_offset, 48, "model iso");
    graph_section(blob, header.body_reference_offset, 12, "body reference");
    graph_section(blob, header.curve_data_offset, 0, "curve data");
    graph_section(blob, header.gearbox_data_offset, 0, "gearbox data");
    let root_off = graph_section(blob, header.collision_root_offset, 132,
        "collision root");
    let children_off = graph_section(blob, header.collision_children_offset,
        header.collision_child_count * 132, "collision children");
    let collision_root = read_collision_tree_record(blob, root_off);
    let mut collision_children = Vec::new();
    for i in 0..header.collision_child_count as usize {
        collision_children.push(read_collision_tree_record(
            blob, children_off + i * 132));
    }
    if collision_root.object_id == 0
        || (collision_root.flags & 0x80) == 0
        || collision_root.surface_id != 0
        || collision_root.geometry_id != 0
        || collision_root.kind != VEHICLE_COLLISION_TREE_ROOT
        || collision_root.wheel_index != u32::MAX
        || collision_root.material_count != 0
    {
        panic!("tmnf world: invalid vehicle collision root");
    }
    let mut wheel_mask = 0u32;
    let mut collision_material_count = 0u32;
    let mut open_children = 0u32;
    for tree in &collision_children {
        let kind = tree.kind & VEHICLE_COLLISION_KIND_MASK;
        let child_count = tree.kind >> VEHICLE_COLLISION_CHILD_SHIFT;
        if tree.object_id == 0
            || (tree.flags & 0x80) == 0
            || child_count > 0xff
            || tree.material_index != collision_material_count
        {
            panic!("tmnf world: invalid vehicle collision node");
        }
        /* Pre-order: a record either fills a slot opened by an earlier
         * parent or is a direct child of the root. */
        if open_children != 0 {
            open_children -= 1;
        }
        open_children += child_count;
        if tree.surface_id == 0 {
            if kind != VEHICLE_COLLISION_TREE_BODY
                || tree.geometry_id != 0
                || tree.wheel_index != u32::MAX
                || tree.material_count != 0
                || child_count == 0
            {
                panic!("tmnf world: invalid surface-less vehicle collision node");
            }
            continue;
        }
        if tree.geometry_id == 0
            || (tree.flags & 0x04) == 0
            || (tree.geometry_type != GM_SURF_ELLIPSOID
                && tree.geometry_type != GM_SURF_SPHERE)
            || tree.geometry_material_index != 0
            || tree.material_count != 1
        {
            panic!("tmnf world: invalid vehicle collision shape");
        }
        let material = graph_section(blob,
            header.collision_material_ids_offset + tree.material_index,
            tree.material_count, "collision material");
        if blob[material] as u32 >= 31 {
            panic!("tmnf world: invalid vehicle collision material");
        }
        if kind == VEHICLE_COLLISION_TREE_BODY {
            if tree.wheel_index != u32::MAX {
                panic!("tmnf world: invalid vehicle body collision node");
            }
        } else if kind == VEHICLE_COLLISION_TREE_WHEEL {
            if tree.wheel_index >= 4
                || child_count != 0
                || (wheel_mask & (1u32 << tree.wheel_index)) != 0
            {
                panic!("tmnf world: invalid vehicle wheel collision node");
            }
            wheel_mask |= 1u32 << tree.wheel_index;
        } else {
            panic!("tmnf world: invalid vehicle collision node kind");
        }
        collision_material_count += tree.material_count;
    }
    if wheel_mask != 0x0f || open_children != 0 {
        panic!("tmnf world: incomplete vehicle collision tree");
    }
    graph_section(blob, header.collision_material_ids_offset,
        collision_material_count, "collision material ids");

    let tunings_off = graph_section(blob, header.tuning_descriptors_offset,
        header.tuning_count * 16, "tuning descriptors");
    let mut tuning_descriptors = Vec::new();
    for i in 0..header.tuning_count as usize {
        let d = read_tuning_descriptor(blob, tunings_off + i * 16);
        if d.key != i as u32 || d.object_id == 0 || d.raw_size != TUNING_SIZE as u32 {
            panic!("tmnf world: invalid vehicle tuning descriptor");
        }
        graph_section(blob, d.raw_offset, TUNING_SIZE as u32, "tuning raw");
        tuning_descriptors.push(d);
    }

    let curves_off = graph_section(blob, header.curve_descriptors_offset,
        header.curve_count * 40, "curve descriptors");
    let mut curve_descriptors: Vec<VehicleCurveDescriptor> = Vec::new();
    for i in 0..header.curve_count as usize {
        let c = read_curve_descriptor(blob, curves_off + i * 40);
        let owner: &[u8];
        if c.owner_kind == VEHICLE_CURVE_OWNER_STRUCT {
            if c.owner_key != 0
                || !contains_offset(&PRIMARY_CURVE_OFFSETS, c.field_offset)
            {
                panic!("tmnf world: invalid vehicle-struct curve owner");
            }
            owner = &blob[graph_section(blob, header.vehicle_struct_offset,
                VEHICLE_STRUCT_SIZE as u32, "vehicle struct")..];
        } else if c.owner_kind == VEHICLE_CURVE_OWNER_TUNING {
            if c.owner_key >= header.tuning_count
                || !contains_offset(&ACTIVE_CURVE_OFFSETS, c.field_offset)
            {
                panic!("tmnf world: invalid tuning curve owner");
            }
            let raw_off = tuning_descriptors[c.owner_key as usize].raw_offset;
            owner = &blob[raw_off as usize..];
        } else {
            panic!("tmnf world: invalid vehicle curve owner kind");
        }
        if c.object_id == 0 || c.positions_id == 0 || c.values_id == 0
            || c.count == 0 || c.count > u32::MAX / 4
            || u32_at(owner, c.field_offset as usize) == 0
        {
            panic!("tmnf world: invalid vehicle curve descriptor");
        }
        graph_section(blob, c.positions_offset, c.count * 4, "curve positions");
        graph_section(blob, c.values_offset, c.count * 4, "curve values");
        for j in 0..i {
            let other = &curve_descriptors[j];
            if other.owner_kind == c.owner_kind
                && other.owner_key == c.owner_key
                && other.field_offset == c.field_offset
            {
                panic!("tmnf world: duplicate vehicle curve descriptor");
            }
        }
        curve_descriptors.push(c);
    }

    let gearboxes_off = graph_section(blob, header.gearbox_descriptors_offset,
        header.gearbox_count * 24, "gearbox descriptors");
    let mut gearbox_descriptors: Vec<VehicleGearboxDescriptor> = Vec::new();
    for i in 0..header.gearbox_count as usize {
        let g = read_gearbox_descriptor(blob, gearboxes_off + i * 24);
        if g.owner_key >= header.tuning_count
            || !contains_offset(&GEARBOX_OFFSETS, g.field_offset)
            || g.buffer_id == 0
            || g.data_id == 0
            || g.count != GEARBOX_VALUE_COUNT as u32
        {
            panic!("tmnf world: invalid vehicle gearbox descriptor");
        }
        let raw_off = tuning_descriptors[g.owner_key as usize].raw_offset as usize;
        if u32_at(blob, raw_off + g.field_offset as usize)
            != GEARBOX_VALUE_COUNT as u32
            || u32_at(blob, raw_off + g.field_offset as usize + 4) == 0
        {
            panic!("tmnf world: vehicle gearbox does not match raw tuning");
        }
        graph_section(blob, g.data_offset, 24, "gearbox data");
        for j in 0..i {
            let other = &gearbox_descriptors[j];
            if other.owner_key == g.owner_key
                && other.field_offset == g.field_offset
            {
                panic!("tmnf world: duplicate vehicle gearbox descriptor");
            }
        }
        gearbox_descriptors.push(g);
    }

    let materials_off = graph_section(blob, header.ground_materials_offset,
        header.ground_material_count * 40, "ground materials");
    let mut materials = Vec::new();
    for i in 0..header.ground_material_count as usize {
        let m = read_material_record(blob, materials_off + i * 40);
        if m.object_id == 0 {
            panic!("tmnf world: vehicle ground material has null identity");
        }
        if m.fake_contact_mask > 1 {
            panic!("tmnf world: vehicle ground material mask is not a flag");
        }
        materials.push(m);
    }

    VehicleGraph {
        header,
        tuning_descriptors,
        curve_descriptors,
        gearbox_descriptors,
        materials,
        collision_root,
        collision_children,
        blob,
    }
}

/* ---------------------------------------------------------------------------
 * The world structures
 * ------------------------------------------------------------------------- */

/// One node of the vehicle collision tree (the C's `CPlugTree` + per-node
/// contact buffer). `wheel` set on wheel leaves.
#[derive(Clone, Debug)]
pub struct VehicleTree {
    pub object_ref: u32,
    pub flags: u32,
    pub box_aligned: GmBoxAligned,
    pub local_iso: GmIso4,
    /// The ellipsoid (or sphere-as-ellipsoid) surface, when this node has one.
    pub surface: Option<CPlugSurface>,
    pub children: Vec<usize>,
    pub wheel: Option<u32>,
}

/// The immutable half of a world.
pub struct WorldCold {
    pub vehicle_blob: Vec<u8>,
    pub header: VehicleSnapshotHeader,
    pub tuning: VehicleTuning,
    pub player_response_body: CHmsResponseBody,
    pub response_material: CHmsResponseMaterial,
    /// 0x0055F3B0-style uniform gravity field.
    pub gravity_active: i32,
    pub gravity_value: GmVec3,
    /// `*(hms_item->field14)+0x50` — the body reference position (image data).
    pub body_reference_position: GmVec3,
}

/// Creation options (`TmnfWorldOptions`).
#[derive(Clone, Default)]
pub struct WorldOptions {
    /// 128x128 fake-contact mask; REQUIRED (the Rust port has no compiled-in
    /// game image — data is always runtime data).
    pub fake_contact_mask: Option<Arc<Vec<u8>>>,
    /// Require the image to declare the canonical exe and this track's hash.
    pub pin_provenance: bool,
}

/// The mutable simulation world: one car + one track.
pub struct World {
    pub cold: Arc<WorldCold>,
    pub track: Arc<TmnfTrack>,
    pub route: Option<Arc<crate::route::TmnfRoute>>,

    pub dyna: CHmsDyna,
    pub vehicle: Vehicle,

    /// Collision trees: node 0..n-1 in pre-order (wheels included), plus the
    /// root's direct children list.
    pub nodes: Vec<VehicleTree>,
    pub root_children: Vec<usize>,
    pub root_object_ref: u32,
    pub root_flags: u32,
    pub root_box: GmBoxAligned,
    pub root_local_iso: GmIso4,
    /// Node index of each wheel leaf.
    pub wheel_node_of: [usize; 4],
    /// One contact buffer per node (root has none).
    pub contact_buffers: Vec<SHmsSphereBufferContact>,

    /* Zone state (CHmsCollisionManager_SZone mutable fields). */
    pub current_material: u32,
    pub current_corpus1: u32,
    pub current_corpus2: u32,
    pub static_group_bound: bool,
    pub merge_buffers: Vec<usize>,
    pub collision_buffer: CHmsCollisionBuffer,

    /* Physics world state (TmnfPhysicsWorld). */
    pub linear_drag_scale: f32,
    pub angular_drag_scale: f32,
    pub trigger_contacts: u64,
    /// Game scene object +0x18 of the car corpus.
    pub scene_flags: u32,
    pub tick_time: u32,

    /// The car corpus' object_ref (== track.static_response_count).
    pub player_corpus_ref: u32,
}

fn read_curve_real(
    blob: &[u8],
    descriptor: &VehicleCurveDescriptor,
) -> CFuncKeysReal {
    let count = descriptor.count as usize;
    let mut positions = Vec::with_capacity(count);
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        positions.push(f32_at(blob, descriptor.positions_offset as usize + 4 * i));
        values.push(f32_at(blob, descriptor.values_offset as usize + 4 * i));
    }
    let mut curve = CFuncKeysReal {
        keys: crate::vehicle::CFuncKeys {
            count: descriptor.count,
            positions,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        },
        values,
        interpolation: descriptor.interpolation,
    };
    cfunc_keys_compile(&mut curve.keys);
    curve
}

fn lookup_curve_descriptor(
    descriptors: &[VehicleCurveDescriptor],
    owner_kind: u32,
    owner_key: u32,
    field_offset: u32,
) -> Option<usize> {
    descriptors.iter().position(|c| {
        c.owner_kind == owner_kind
            && c.owner_key == owner_key
            && c.field_offset == field_offset
    })
}

/* ---------------------------------------------------------------------------
 * Decode (the C's decode_* + link_* stages, fused)
 * ------------------------------------------------------------------------- */

fn decode_tuning(graph: &VehicleGraph) -> (VehicleTuning, Vehicle) {
    let blob = graph.blob;
    let key = graph.header.active_tuning_key;
    let raw_off =
        graph.tuning_descriptors[key as usize].raw_offset as usize;
    let raw = &blob[raw_off..raw_off + TUNING_SIZE];
    let car_off = graph.header.car_offset as usize;
    let car = &blob[car_off..car_off + CAR_SIZE];
    let rawf = |o: usize| f32_at(raw, o);
    let rawu = |o: usize| u32_at(raw, o);
    let rawi = |o: usize| i32_at(raw, o);
    let carf = |o: usize| f32_at(car, o);
    let cari = |o: usize| i32_at(car, o);
    let caru = |o: usize| u32_at(car, o);

    let mut tuning = VehicleTuning::default();

    /* --- active + primary curves --- */
    let mut active: Vec<Option<CFuncKeysReal>> = Vec::with_capacity(ACTIVE_CURVE_COUNT);
    for i in 0..ACTIVE_CURVE_COUNT {
        match lookup_curve_descriptor(&graph.curve_descriptors,
            VEHICLE_CURVE_OWNER_TUNING, key, ACTIVE_CURVE_OFFSETS[i])
        {
            None if i == CURVE_STEERING_ANGLE => active.push(None),
            None => panic!("tmnf world: vehicle curve lookup failed"),
            Some(idx) => active.push(Some(read_curve_real(blob,
                &graph.curve_descriptors[idx]))),
        }
    }
    let mut primary: Vec<CFuncKeysReal> = Vec::with_capacity(PRIMARY_CURVE_COUNT);
    for i in 0..PRIMARY_CURVE_COUNT {
        let idx = lookup_curve_descriptor(&graph.curve_descriptors,
            VEHICLE_CURVE_OWNER_STRUCT, 0, PRIMARY_CURVE_OFFSETS[i])
            .unwrap_or_else(|| panic!("tmnf world: vehicle curve lookup failed"));
        primary.push(read_curve_real(blob, &graph.curve_descriptors[idx]));
    }

    /* --- curve set --- */
    {
        let set = &mut tuning.curves;
        set.accel_from_speed = active[CURVE_ACCEL].clone();
        set.rollover_lateral_from_speed = active[CURVE_ROLLOVER_LATERAL].clone();
        set.max_side_friction_from_speed = active[CURVE_MAX_SIDE_FRICTION].clone();
        set.lateral_contact_slowdown_from_speed =
            active[CURVE_LATERAL_CONTACT_SLOWDOWN].clone();
        set.steer_slowdown_from_speed = active[CURVE_STEER_SLOWDOWN].clone();
        set.rollover_lateral_coef_from_angle = active[CURVE_ROLLOVER_ANGLE].clone();
        set.steer_drive_torque_from_speed = active[CURVE_STEER_DRIVE_TORQUE].clone();
        set.m4_steer_radius_from_speed = active[CURVE_M4_STEER_RADIUS].clone();
        set.m4_max_friction_force_from_speed = active[CURVE_M4_MAX_FRICTION].clone();
        set.m5_slipping_accel_from_speed = active[CURVE_M5_SLIPPING_ACCEL].clone();
        set.m5_slipping_accel_scale = rawf(0x1e4);
        set.water_friction_from_speed = active[CURVE_WATER_FRICTION].clone();
        set.damper_max = rawf(0x11c);
        set.damper_min = rawf(0x120);
        set.m6_damper_modulation = active[CURVE_M6_DAMPER_MODULATION].clone();
        set.m6_rear_gear_accel_from_speed =
            active[CURVE_M6_REAR_GEAR_ACCEL].clone();
        set.m6_rollover_lateral_from_speed_ratio =
            active[CURVE_M6_ROLLOVER_RATIO].clone();
        set.m6_burnout_radius_from_speed = active[CURVE_M6_BURNOUT_RADIUS].clone();
        set.m6_lateral_speed_from_burnout_radius =
            active[CURVE_M6_BURNOUT_LATERAL_SPEED].clone();
        set.m6_donut_rollover_from_speed =
            active[CURVE_M6_DONUT_ROLLOVER].clone();
        set.m6_burnout_rollover_from_speed =
            active[CURVE_M6_BURNOUT_ROLLOVER].clone();
    }

    /* --- gearboxes --- */
    {
        let destinations: [&mut Vec<f32>; GEARBOX_COUNT] =
            [&mut tuning.base.gear_ratios, &mut tuning.base.gear_upshift,
             &mut tuning.base.gear_downshift, &mut tuning.base.gear_aux];
        for i in 0..GEARBOX_COUNT {
            let idx = graph.gearbox_descriptors.iter().position(|g| {
                g.owner_key == key && g.field_offset == GEARBOX_OFFSETS[i]
            }).unwrap_or_else(|| panic!("tmnf world: vehicle gearbox lookup failed"));
            let g = &graph.gearbox_descriptors[idx];
            for k in 0..GEARBOX_VALUE_COUNT {
                destinations[i].push(f32_at(blob, g.data_offset as usize + 4 * k));
            }
        }
    }

    /* --- base tuning (decode_vehicle_tuning) --- */
    {
        let t = &mut tuning.base;
        t.damper_max = rawf(0x11c);
        t.damper_min = rawf(0x120);
        t.gear_ratio_count = 6;
        t.engine_model = rawi(0x354);
        t.forward_speed_limit_scale = rawf(0x02c);
        t.engine_rpm_reverse_accel = rawf(0x2ec);
        t.engine_rpm_accel = rawf(0x2f0);
        t.engine_rpm_decel = rawf(0x2f4);
        t.engine_rpm_turbo_decel = rawf(0x31c);
        t.engine_rpm_follow_accel = rawf(0x320);
        t.engine_rpm_low_accel = rawf(0x324);
        t.engine_rpm_high_decel = rawf(0x328);
        t.speed_32c = rawf(0x32c);
        t.speed_330 = rawf(0x330);
        t.speed_334 = rawf(0x334);
        t.speed_338 = rawf(0x338);
        t.suspension_model = rawi(0x350);
        t.suspension_stiffness = rawf(0x114);
        t.suspension_damping = rawf(0x118);
        t.suspension_rest_length = rawf(0x124);
        t.suspension_scale = rawf(0x128);
    }

    /* --- aux tuning (decode_aux) --- */
    {
        let t = &mut tuning.aux;
        t.steering_speed_base = rawf(0x06c);
        t.steering_speed_scale = rawf(0x070);
        t.steering_slew_rate = rawf(0x094);
        t.air_torque_linear = rawf(0x158);
        t.air_torque_quadratic = rawf(0x15c);
        t.suspension_follow_rate = rawf(0x194);
        t.air_control_window_ticks = rawu(0x364);
        t.air_reversal_threshold = rawf(0x368);
        t.water_buoyancy = rawf(0x204);
        t.water_entry_speed_threshold = rawf(0x208);
        t.water_entry_speed_minimum = rawf(0x20c);
        t.water_angular_drag_linear = rawf(0x21c);
        t.water_angular_drag_quadratic = rawf(0x220);
        /* Tuning +0x210/+0x214 are absent from the snapshot; the embedded
         * water-impulse tables stand in (vehicle_water_tuning.h). */
        t.air_vertical_curve = active[CURVE_AIR_VERTICAL].clone();
        t.steering_angle_curve = active[CURVE_STEERING_ANGLE].clone();
        let mk = |values: [f32; 4]| CFuncKeysReal {
            keys: crate::vehicle::CFuncKeys {
                count: 4,
                positions: TMNF_WATER_IMPULSE_POSITIONS.to_vec(),
                lower_bounds: Vec::new(),
                upper_bounds: Vec::new(),
            },
            values: values.to_vec(),
            interpolation: 0,
        };
        let mut vertical = mk(TMNF_WATER_IMPULSE_VERTICAL_VALUES);
        let mut horizontal = mk(TMNF_WATER_IMPULSE_HORIZONTAL_VALUES);
        cfunc_keys_compile(&mut vertical.keys);
        cfunc_keys_compile(&mut horizontal.keys);
        t.water_impulse_vertical_curve = Some(vertical);
        t.water_impulse_horizontal_curve = Some(horizontal);
    }

    /* --- contact tuning (decode_contact_tuning) --- */
    {
        let t = &mut tuning.contact;
        t.friction_force = rawf(0x058);
        t.extra_friction_force = rawf(0x05c);
        t.slope_adherence_min = rawf(0x0d4);
        t.slope_adherence_max = rawf(0x0d8);
        t.slope_secondary_min = rawf(0x0dc);
        t.slope_secondary_max = rawf(0x0e0);
        t.angular_y_scale = rawf(0x0e8);
        t.angular_xz_scale = rawf(0x0ec);
        t.damper_max = rawf(0x11c);
        t.max_angular_speed = rawf(0x14c);
        t.max_linear_speed_delta = rawf(0x150);
        t.body_tangent_ratio = rawf(0x170);
        t.body_tangent_ratio_material4 = rawf(0x174);
        t.restitution_air_material4 = rawf(0x178);
        t.restitution_air = rawf(0x17c);
        t.restitution_ground = rawf(0x184);
        t.restitution_ground_material4 = rawf(0x18c);
        t.lateral_linear = rawf(0x1a8);
        t.lateral_quadratic = rawf(0x1ac);
        t.lateral_ground_scale = rawf(0x1c0);
        t.lateral_contact_duration_ticks = rawu(0x1e8);
        t.wheel_contact_model = rawi(0x350);
        t.friction_model = rawi(0x354);
    }

    /* --- model 6 tuning (decode_model6) --- */
    {
        let t = &mut tuning.model6;
        t.forward_speed_limit_scale = rawf(0x02c);
        t.reverse_speed_limit_scale = rawf(0x030);
        t.brake_base = rawf(0x040);
        t.brake_speed_scale = rawf(0x044);
        t.forward_brake_limit_sliding = rawf(0x048);
        t.forward_brake_limit = rawf(0x04c);
        t.speed_limit_force = rawf(0x060);
        t.vertical_force_scale = rawf(0x064);
        t.wheel_steer_sine_limit = rawf(0x074);
        t.steer_slowdown_scale = rawf(0x07c);
        t.wheel_torque_scale = rawf(0x098);
        t.sliding_steer_torque_scale = rawf(0x09c);
        t.lateral_force_scale = rawf(0x0a4);
        t.sliding_lateral_limit_scale = rawf(0x0b0);
        t.lateral_overflow_blend = rawf(0x0b4);
        t.wheel_overflow_blend = rawf(0x0e4);
        t.vertical_force_divisor = rawf(0x160);
        t.traction_loss_scale = rawf(0x200);
        t.burnout_trigger_scale = rawf(0x228);
        t.burnout_trigger_limit = rawf(0x22c);
        t.rollover_axis_min_length = rawf(0x234);
        t.rollover_torque_x_scale = rawf(0x238);
        t.rollover_torque_z_scale = rawf(0x23c);
        t.sliding_brake_scale = rawf(0x240);
        t.braking_lateral_limit_scale = rawf(0x244);
        t.reverse_brake_limit_sliding = rawf(0x248);
        t.reverse_brake_limit = rawf(0x24c);
        t.burnout_speed_max = rawf(0x254);
        t.burnout_speed_min = rawf(0x258);
        t.donut_lateral_force_scale = rawf(0x264);
        t.donut_yaw_angle_scale = rawf(0x268);
        t.donut_steer_linear = rawf(0x26c);
        t.donut_steer_quadratic = rawf(0x270);
        t.donut_countersteer_scale = rawf(0x274);
        t.donut_radius_exponent = rawf(0x278);
        t.donut_radial_speed_exponent = rawf(0x27c);
        t.donut_radius_min = rawf(0x280);
        t.donut_lateral_speed_limit = rawf(0x284);
        t.donut_normal_angle_limit = rawf(0x28c);
        t.donut_angle_positive_limit = rawf(0x290);
        t.donut_angle_negative_limit = rawf(0x294);
        t.burnout_enter_ticks = rawu(0x298);
        t.burnout_enter_accel_scale = rawf(0x29c);
        t.burnout_enter_lateral_scale = rawf(0x2a0);
        t.burnout_exit_ticks = rawu(0x2a8);
        t.burnout_exit_accel_scale = rawf(0x2ac);
        t.burnout_exit_extra_accel = rawf(0x2b8);
        t.material6_longitudinal_scale = rawf(0x33c);
        t.material6_gas_denominator = rawf(0x340);
        t.material6_vertical_shape = rawf(0x344);
        t.material6_vertical_scale = rawf(0x348);
    }

    /* --- compute tuning (decode_compute) --- */
    {
        let t = &mut tuning.compute;
        t.event_c_level1_max = rawf(0x028);
        t.grounded_drag_term = rawf(0x058);
        t.active_contact_stop_threshold = rawf(0x0a4);
        t.normal_turbo_factor = rawf(0x0f0);
        t.roulette_turbo_factor = rawf(0x0f4);
        t.normal_turbo_duration = rawu(0x0f8);
        t.roulette_turbo_duration = rawu(0x0fc);
        t.air_impulse_scale = rawf(0x104);
        t.special_force_field_scale = rawf(0x108);
        t.airborne_linear_drag = rawf(0x154);
        t.grounded_force_field_scale = rawf(0x160);
        t.airborne_force_field_scale = rawf(0x164);
        t.normalized_force_divisor = rawf(0x228);
        t.effect_curve_bias = rawf(0x384);
        t.event_ab_level2_min = rawf(0x398);
        t.event_ab_trigger = rawf(0x39c);
        t.event_c_trigger = rawf(0x3a0);
        t.effect_curve = active[CURVE_EFFECT].clone();
        t.contact_decay_curve = Some(primary[0].clone());
        t.contact_rise_curve = Some(primary[1].clone());
    }

    /* --- ground materials + ids --- */
    {
        let ids_off = graph.header.ground_ids_offset as usize;
        for i in 0..31usize {
            tuning.ground_material_indices.push(u32_at(blob, ids_off + 4 * i));
        }
        for id in &tuning.ground_material_indices {
            if *id >= graph.header.ground_material_count {
                panic!("tmnf world: ground-id table maps a physical material outside the material manager");
            }
        }
        for raw in &graph.materials {
            tuning.ground_materials.push(crate::vehicle::TMNFVehicleGroundMaterial {
                values: raw.values,
                fake_contact_mask: None, /* bound to the world mask below */
                fake_contact_period_x: raw.fake_contact_period_x,
                fake_contact_period_z: raw.fake_contact_period_z,
                fake_contact_impulse_scale: raw.fake_contact_impulse_scale,
                fake_contact_impulse_limit: raw.fake_contact_impulse_limit,
            });
        }
    }

    /* --- vehicle mutable initial state (decode_vehicle, minus wheels) --- */
    /* (returned via the tuple below) */
    let mut vehicle = Vehicle::default();
    vehicle.input_gas = carf(0x050);
    vehicle.input_brake = carf(0x054);
    vehicle.input_steer = carf(0x058);
    vehicle.wheel_count = 4;
    {
        let mut engine = &mut vehicle.engine;
        engine.max_rpm = carf(0x59c);
        engine.braking_factor = carf(0x59c + 0x14);
        engine.rpm = carf(0x59c + 0x18);
        engine.target_rpm = carf(0x59c + 0x1c);
        engine.clutch = carf(0x59c + 0x20);
        engine.shift_timer = carf(0x59c + 0x24);
        engine.reverse = cari(0x59c + 0x28);
        engine.gear = cari(0x59c + 0x2c);
    }
    vehicle.current_local_speed = vec3_at(car, 0x70c);
    vehicle.total_force_added = vec3_at(car, 0x818);
    vehicle.total_impulse_added = vec3_at(car, 0x824);
    vehicle.engine_mode = cari(0x2e4);
    vehicle.turbo_active = cari(0x628);
    vehicle.drive_mode = cari(0x69c);
    vehicle.engine_limit_flag = cari(0x744);
    vehicle.gear_downshift_flag = cari(0x748);
    vehicle.force_wheel_speed = cari(0x6a0);
    vehicle.flag_60c = cari(0x60c);
    vehicle.block_wheel_speed = cari(0x73c);
    vehicle.forced_wheel_speed = rawf(0x2bc);

    /* aux state (decode_aux) */
    vehicle.integration_flags = caru(0x2f4);
    vehicle.turbo_epoch_tick = caru(0x5d0);
    vehicle.air_control_immediate = cari(0x5d4);
    vehicle.air_control_locked = cari(0x5e4);
    vehicle.steering_value = carf(0x5e8);
    vehicle.turbo_progress = carf(0x5f0);
    vehicle.turbo_factor = carf(0x5f4);
    vehicle.turbo_start_tick = caru(0x5f8);
    vehicle.turbo_end_tick = caru(0x5fc);
    vehicle.turbo_type = match cari(0x600) {
        1 => TMNFVehicleTurboType::Normal,
        2 => TMNFVehicleTurboType::Roulette,
        _ => TMNFVehicleTurboType::None,
    };
    vehicle.roulette_token = caru(0x604);
    vehicle.roulette_value = carf(0x608);
    vehicle.air_control_tick = caru(0x614);
    vehicle.air_control_speed = vec3_at(car, 0x618);
    vehicle.roulette_modulus = u32::MAX;
    vehicle.body_box = box_at(car, 0x1dc);
    vehicle.turbo_sound_attached = false;

    /* contact state */
    vehicle.friction_input_selector = cari(0x5c4);
    vehicle.side_contact = cari(0x5dc);
    vehicle.last_side_contact_tick = caru(0x5e0);
    vehicle.airborne_friction_gate = cari(0x5e4);
    vehicle.wheel_contact_absorb_count = caru(0x67c);
    vehicle.body_contact_count = caru(0x680);
    vehicle.body_contact_position_sum = vec3_at(car, 0x684);
    vehicle.body_contact_normal_sum = vec3_at(car, 0x690);
    vehicle.tick_time = graph.header.tick;

    /* model 6 state (decode_model6) */
    {
        let s = &mut vehicle.model6;
        s.pivot_position = vec3_at(car, 0x1dc);
        s.pivot_axis = vec3_at(car, 0x1e8);
        s.reverse_mode = cari(0x5c4);
        s.reverse_speed_threshold = carf(0x5cc);
        s.contact_block_count = cari(0x5d8);
        s.side_contact = cari(0x5dc);
        s.last_sliding_tick = caru(0x62c);
        s.sliding_start_tick = caru(0x630);
        s.sliding_elapsed_ticks = caru(0x634);
        s.model_iso = iso4_at(car, 0x6a4);
        s.rollover_axis = vec3_at(car, 0x6d4);
        s.orbit_center = vec3_at(car, 0x6e0);
        s.orbit_initial_radius = carf(0x6ec);
        s.orbit_radius = carf(0x6f0);
        s.burnout_start_tick = caru(0x6f4);
        s.burnout_transition_tick = caru(0x6f8);
        s.orbit_axis = vec3_at(car, 0x6fc);
        s.orbit_sign = carf(0x708);
        s.axle_width = carf(0x840);
    }

    /* compute state (decode_compute) */
    {
        let s = &mut vehicle.compute;
        s.simulation_gate = carf(0x1e8);
        s.event_level_c = caru(0x1fc);
        s.event_source_c = car[0x200];
        s.event_source_ab = car[0x201];
        s.spring_c = GmSpringFloat {
            stiffness: carf(0x214), damping: carf(0x218),
            value: carf(0x21c), target: carf(0x220), velocity: carf(0x224),
        };
        s.spring_a = GmSpringFloat {
            stiffness: carf(0x228), damping: carf(0x22c),
            value: carf(0x230), target: carf(0x234), velocity: carf(0x238),
        };
        s.contact_rise = carf(0x23c);
        s.contact_decay = carf(0x240);
        s.history_force_limit = carf(0x244);
        s.history_force_scale = carf(0x24c);
        s.spring_value_limit = carf(0x250);
        s.local_speed_limit = carf(0x2e0);
        s.air_effect_threshold = carf(0x05c);
        s.brake_input_scale = carf(0x5a8);
        s.grounded_drag_scale = carf(0x5ac);
        s.computed_brake_force = carf(0x5b0);
        s.state_5d8 = cari(0x5d8);
        s.air_impulse_cooldown_tick = caru(0x610);
        s.effect_accumulator = carf(0x624);
        s.last_force_tick = caru(0x650);
        s.event_level_a = caru(0x654);
        s.event_level_b = caru(0x658);
        s.peak_event_level_b = caru(0x660);
        s.peak_event_level_a = caru(0x664);
        s.peak_event_level_c = caru(0x668);
        s.peak_event_source_ab = car[0x66c];
        s.peak_event_source_c = car[0x66d];
        s.event_metric_a = carf(0x670);
        s.event_metric_b = carf(0x674);
        s.event_metric_c = carf(0x678);
        s.normalized_force = vec3_at(car, 0x6d4);
        s.air_effect_mode = cari(0x74c);
    }

    /* wheels (decode_wheel + aux surface isos) */
    for i in 0..4usize {
        let w = graph.header.wheels_offset as usize + i * WHEEL_SIZE;
        let wheel_bytes = &blob[w..w + WHEEL_SIZE];
        let mut wheel = crate::vehicle::CSceneVehicleCarWheel::default();
        wheel.active = i32_at(wheel_bytes, 0x000);
        wheel.steerable = i32_at(wheel_bytes, 0x004);
        wheel.radius = f32_at(wheel_bytes, 0x008);
        for k in 0..12 {
            wheel.field70[k] = f32_at(wheel_bytes, 0x070 + 4 * k);
        }
        wheel.fielda0 = i32_at(wheel_bytes, 0x0a0);
        wheel.fielda4 = i32_at(wheel_bytes, 0x0a4);
        wheel.offset_from_vehicle = vec3_at(wheel_bytes, 0x0a8);
        wheel.real_time = decode_wheel_real_time(wheel_bytes, 0x0b4);
        wheel.field15c = i32_at(wheel_bytes, 0x15c);
        wheel.contact_relative_local_distance = vec3_at(wheel_bytes, 0x160);
        wheel.surface_source = iso4_at(wheel_bytes, 0x010);
        wheel.surface_location = iso4_at(wheel_bytes, 0x040);
        /* History (previous_sync..async_state) from +0x16c. */
        wheel.previous_sync = decode_wheel_state(wheel_bytes, 0x16c);
        wheel.sync = decode_wheel_state(wheel_bytes, 0x1d0);
        wheel.field234 = decode_wheel_state(wheel_bytes, 0x234);
        wheel.async_state = decode_wheel_state(wheel_bytes, 0x298);
        vehicle.wheels.push(wheel);
    }

    (tuning, vehicle)
}

fn decode_wheel_real_time(b: &[u8], o: usize) -> crate::vehicle::CSceneVehicleCarWheelRealTimeState {
    use crate::vehicle::CSceneVehicleCarWheelRealTimeState;
    let f = |o: usize| f32_at(b, o);
    let vec = |o: usize| GmVec3 { x: f(o), y: f(o + 4), z: f(o + 8) };
    let mat = |o: usize| {
        let mut m = [0.0f32; 9];
        for (i, item) in m.iter_mut().enumerate() {
            *item = f(o + 4 * i);
        }
        GmMat3 { m }
    };
    let mut reserved60 = [0u8; 0x0c];
    reserved60.copy_from_slice(&b[o + 0x60..o + 0x6c]);
    CSceneVehicleCarWheelRealTimeState {
        damper_absorb: f(o + 0x00),
        field04: f(o + 0x04),
        field08: f(o + 0x08),
        basis0: mat(o + 0x0c),
        basis1: mat(o + 0x30),
        field54: vec(o + 0x54),
        reserved60,
        field6c: f(o + 0x6c),
        has_ground_contact: i32_at(b, o + 0x70),
        contact_material_id: i32_at(b, o + 0x74),
        is_sliding: i32_at(b, o + 0x78),
        relative_rotz_axis: vec(o + 0x7c),
        contact_body_word: u32_at(b, o + 0x88),
        ground_contact_count: i32_at(b, o + 0x8c),
        field90: vec(o + 0x90),
        rotation_phase: f(o + 0x9c),
        blend_value: f(o + 0xa0),
        blend_target: f(o + 0xa4),
    }
}

fn decode_wheel_state(b: &[u8], o: usize) -> crate::vehicle::CSceneVehicleCarWheelState {
    let mut bytes = [0u8; 0x64];
    bytes.copy_from_slice(&b[o..o + 0x64]);
    crate::vehicle::CSceneVehicleCarWheelState { bytes }
}

/* ---------------------------------------------------------------------------
 * World creation (World_CreateFromBlob)
 * ------------------------------------------------------------------------- */

impl World {
    /// `World_CreateFromBlob`: build a world from the track, the vehicle
    /// image bytes and runtime data. Panics exactly where the C aborts.
    pub fn new(
        track: Arc<TmnfTrack>,
        vehicle_blob: Vec<u8>,
        options: WorldOptions,
    ) -> World {
        let mask = options.fake_contact_mask.unwrap_or_else(|| {
            panic!("tmnf world: fake-contact mask missing; supply one at runtime (local/game-mask.bin)");
        });
        if mask.len() != 128 * 128 {
            panic!("tmnf world: fake-contact mask must be 128x128 bytes");
        }
        let graph = validate_vehicle_graph(
            &vehicle_blob,
            Some(&track.header.track_sha256),
            options.pin_provenance,
        );

        let (mut tuning, mut vehicle) = decode_tuning(&graph);
        /* The wheel/body tree wiring (decode_collision_world's
         * wheel_tree_refs / body_tree_refs fill). */
        tuning.wheel_tree_refs = vec![0u32; 4];
        for raw in &graph.collision_children {
            let kind = raw.kind & VEHICLE_COLLISION_KIND_MASK;
            if kind == VEHICLE_COLLISION_TREE_WHEEL {
                tuning.wheel_tree_refs[raw.wheel_index as usize] = raw.object_id;
            } else {
                tuning.body_tree_refs.push(raw.object_id);
            }
        }
        /* Bind the fake-contact masks (the C's link_vehicle). */
        for (i, raw) in graph.materials.iter().enumerate() {
            if raw.fake_contact_mask != 0 {
                tuning.ground_materials[i].fake_contact_mask = Some(mask.clone());
            }
        }

        /* The dyna. */
        let state_off = graph.header.state_offset as usize;
        let mut live_state_bytes = [0u8; 180];
        live_state_bytes.copy_from_slice(
            &vehicle_blob[state_off..state_off + 180]);
        let live_state = CHmsStateDyna::from_bytes(&live_state_bytes);
        let params_off = graph.header.params_offset as usize;
        let dyna_params = CHmsDynaParams::from_bytes(
            &vehicle_blob[params_off..params_off + PARAMS_SIZE]);
        let dyna_raw = graph.header.dyna_offset as usize;
        let mut temp_bytes = [0u8; 180];
        temp_bytes.copy_from_slice(
            &vehicle_blob[dyna_raw + 0x274..dyna_raw + 0x274 + 180]);
        let temp_state = CHmsStateDyna::from_bytes(&temp_bytes);
        let clamp_angular = i32_at(&vehicle_blob, dyna_raw + 0x0c0);
        let max_angular_speed = f32_at(&vehicle_blob, dyna_raw + 0x0c4);
        let dirty_flag = i32_at(&vehicle_blob, dyna_raw + 0x33c);
        let mode = i32_at(&vehicle_blob, dyna_raw + 0x340);
        let mut dyna = CHmsDyna::new(dyna_params, mode, clamp_angular,
            max_angular_speed);
        dyna.dirty_flag = dirty_flag;
        dyna.set_live(live_state);
        dyna.set_state_b(temp_state);
        dyna.temp_state = temp_state;

        /* The collision trees (decode_collision_world + link_collision). */
        let node_count = graph.header.collision_child_count as usize;
        let mut nodes: Vec<VehicleTree> = Vec::with_capacity(node_count);
        let mut wheel_node_of = [usize::MAX; 4];
        let mut body_tree_count = 0usize;
        for (i, raw) in graph.collision_children.iter().enumerate() {
            let kind = raw.kind & VEHICLE_COLLISION_KIND_MASK;
            let mut node = VehicleTree {
                object_ref: raw.object_id,
                flags: raw.flags,
                box_aligned: raw.box_aligned,
                local_iso: raw.local_iso,
                surface: None,
                children: Vec::new(),
                wheel: None,
            };
            if kind == VEHICLE_COLLISION_TREE_WHEEL {
                let wheel_index = raw.wheel_index as usize;
                wheel_node_of[wheel_index] = i;
                node.wheel = Some(raw.wheel_index);
                /* Wheel impulse point starts at the box center. */
                vehicle.wheels[wheel_index].impulse_point = raw.box_aligned.center;
            } else {
                body_tree_count += 1;
            }
            if raw.surface_id != 0 {
                /* The shape payload: 12 bytes of radii at offset 0 of `shape`. */
                let radii = vec3_at(&raw.shape, 0);
                let geom = if raw.geometry_type == GM_SURF_SPHERE {
                    GmSurfGeom::Sphere(GmSurfSphere {
                        base: GmSurf {
                            vtable: 0,
                            material_index: raw.geometry_material_index,
                            surf_type: raw.geometry_type,
                            reserved: raw.geometry_reserved,
                        },
                        radius: radii.x,
                    })
                } else {
                    GmSurfGeom::Ellipsoid(GmSurfEllipsoid {
                        base: GmSurf {
                            vtable: 0,
                            material_index: raw.geometry_material_index,
                            surf_type: raw.geometry_type,
                            reserved: raw.geometry_reserved,
                        },
                        radii,
                    })
                };
                let mat_off = graph.header.collision_material_ids_offset as usize
                    + raw.material_index as usize;
                node.surface = Some(CPlugSurface {
                    geom,
                    material_ids: vehicle_blob[mat_off..mat_off
                        + raw.material_count as usize].to_vec(),
                    material_count: raw.material_count,
                });
            }
            nodes.push(node);
        }

        /* Pre-order linking: the root owns every record not claimed as a
         * child by an earlier node; each node's children follow it in
         * pre-order (child, then the child's subtree, then the next child). */
        let child_count_of =
            |i: usize| graph.collision_children[i].kind >> VEHICLE_COLLISION_CHILD_SHIFT;
        let mut root_children: Vec<usize> = Vec::new();
        {
            let mut pending = 0u32;
            for i in 0..node_count {
                if pending != 0 {
                    pending -= 1;
                } else {
                    root_children.push(i);
                }
                pending += child_count_of(i);
            }
        }
        {
            fn assign(
                nodes: &mut [VehicleTree],
                child_count_of: &dyn Fn(usize) -> u32,
                parent: usize,
                next: &mut usize,
            ) {
                let count = child_count_of(parent);
                let mut kids = Vec::new();
                for _ in 0..count {
                    if *next >= nodes.len() {
                        panic!("tmnf world: vehicle collision node is missing children");
                    }
                    let child = *next;
                    *next += 1;
                    assign(nodes, child_count_of, child, next);
                    kids.push(child);
                }
                nodes[parent].children = kids;
            }
            let mut next = 0usize;
            let roots = root_children.clone();
            for r in roots {
                if r != next {
                    panic!("tmnf world: vehicle collision tree construction failed");
                }
                next += 1;
                assign(&mut nodes, &child_count_of, r, &mut next);
            }
            if next != node_count {
                panic!("tmnf world: vehicle collision tree construction failed");
            }
        }

        let player_corpus_ref = track.static_response_count;

        let contact_buffers: Vec<SHmsSphereBufferContact> =
            (0..node_count).map(|_| SHmsSphereBufferContact::new()).collect();

        let body_reference_position = vec3_at(
            &vehicle_blob, graph.header.body_reference_offset as usize);
        let cold_header = graph.header.clone();
        let snapshot_tick = graph.header.tick;
        let root_object_ref = graph.collision_root.object_id;
        let root_flags = graph.collision_root.flags;
        let root_box = graph.collision_root.box_aligned;
        let root_local_iso = graph.collision_root.local_iso;
        drop(graph);

        let cold = WorldCold {
            vehicle_blob,
            header: cold_header,
            tuning,
            player_response_body: CHmsResponseBody {
                corpus_ref: player_corpus_ref,
                classification_flags: 0x19847004u32,
                response_flags: 0xff618001u32,
                response_weight: 1.0,
                iso: GmIso4::default(),
                dyna: Some(0),
                has_contact_sink: true,
                sink: AbsorbSink::VehicleBody,
            },
            response_material: CHmsResponseMaterial {
                category: 3,
                response_mode: 1,
                side_enabled: [1, 1],
            },
            gravity_active: 1,
            gravity_value: GmVec3 { x: 0.0, y: -10.0, z: 0.0 },
            body_reference_position,
        };

        let mut world = World {
            cold: Arc::new(cold),
            track,
            route: None,
            dyna,
            vehicle,
            nodes,
            root_children,
            root_object_ref,
            root_flags,
            root_box,
            root_local_iso,
            wheel_node_of,
            contact_buffers,
            current_material: 0,
            current_corpus1: 0,
            current_corpus2: 0,
            static_group_bound: false,
            merge_buffers: Vec::new(),
            collision_buffer: CHmsCollisionBuffer::new(),
            linear_drag_scale: 1.0,
            angular_drag_scale: 1.0,
            trigger_contacts: 0,
            scene_flags: 0x2000u32,
            tick_time: snapshot_tick,
            player_corpus_ref,
        };

        /* The snapshot begins at Model6 entry, after ComputeForces changed
         * this field from the preceding grounded value to the airborne value.
         * Restore the value consumed by the in-progress ComputeCorpusForces. */
        world.dyna.params.force_field_scale =
            world.cold.tuning.compute.grounded_force_field_scale;

        /* finish_snapshot_tick: run one tick to settle the captured state. */
        world.physics_step2(10);
        world
    }

    /// The live rigid-body state.
    pub fn live(&self) -> &CHmsStateDyna {
        self.dyna.live()
    }

    /// A `&[CPlugTree]` view of the vehicle collision tree for the race
    /// trigger test (root at index 0, nodes at 1..; children shifted +1).
    fn car_tree_snapshot(&self) -> Vec<CPlugTree> {
        let mut trees = Vec::with_capacity(1 + self.nodes.len());
        trees.push(CPlugTree {
            object_ref: self.root_object_ref,
            flags: self.root_flags,
            box_aligned: self.root_box,
            local_iso: self.root_local_iso,
            surface: None,
            children: self.root_children.iter().map(|&c| c + 1).collect(),
        });
        for n in &self.nodes {
            trees.push(CPlugTree {
                object_ref: n.object_ref,
                flags: n.flags,
                box_aligned: n.box_aligned,
                local_iso: n.local_iso,
                surface: n.surface.clone(),
                children: n.children.iter().map(|&c| c + 1).collect(),
            });
        }
        trees
    }

    /// 0x0055F3B0: the uniform gravity field value.
    pub fn gravity(&self) -> Option<GmVec3> {
        if self.cold.gravity_active != 0 {
            Some(self.cold.gravity_value)
        } else {
            None
        }
    }

    /// `World_AdvanceTimer` — the game timer counts *milliseconds*
    /// (TmnfPhysicsCorpus_AdvanceTimer: timer->tick_time += tick_ms), and the
    /// physics corpus mirrors it.
    pub fn advance_timer(&mut self, tick_ms: u32) {
        self.tick_time = self.tick_time.wrapping_add(tick_ms);
        self.vehicle.tick_time = self.tick_time;
    }

    /// 0x004FE500 `UpdateVehicleStateFromInputs` on the player vehicle.
    pub fn apply_inputs(&mut self, inputs: &crate::vehicle::TMNFRaceInputs) {
        let World { vehicle, dyna, cold, .. } = self;
        let mut ctx = VehicleCtx { vehicle, tuning: &cold.tuning, dyna };
        crate::vehicle::base::update_vehicle_state_from_inputs(inputs, &mut ctx);
    }

    /// The wheel trees follow the suspension: `set_surface_location`
    /// (0x007C8AA0) applied after the wheel integrates.
    pub(crate) fn sync_wheel_surfaces(&mut self) {
        for w in 0..4 {
            let node = self.wheel_node_of[w];
            if node == usize::MAX {
                continue;
            }
            let location = self.vehicle.wheels[w].surface_location;
            self.nodes[node].local_iso = location;
            self.nodes[node].box_aligned.center = GmVec3 {
                x: location.t[0], y: location.t[1], z: location.t[2],
            };
            self.vehicle.wheels[w].impulse_point = GmVec3 {
                x: location.t[0], y: location.t[1], z: location.t[2],
            };
        }
    }
}

/* ---------------------------------------------------------------------------
 * The response context (the C's resolver wiring)
 * ------------------------------------------------------------------------- */

impl World {
    fn static_body(&self, corpus_ref: u32) -> Option<&CHmsResponseBody> {
        let track = &self.track;
        if (corpus_ref as usize) < track.static_response_bodies.len()
            && track.static_response_present[corpus_ref as usize] != 0
        {
            Some(&track.static_response_bodies[corpus_ref as usize])
        } else {
            None
        }
    }

    pub(crate) fn resolve_body_iso(&self, corpus_ref: u32) -> GmIso4 {
        if corpus_ref == self.player_corpus_ref {
            self.dyna.live().rot_pos_iso()
        } else if let Some(body) = self.static_body(corpus_ref) {
            body.iso
        } else {
            panic!("tmnf world: contact delivered to an unknown response body")
        }
    }
}

impl ResponseCtx for World {
    fn body(&self, corpus_ref: u32) -> &CHmsResponseBody {
        if corpus_ref == self.player_corpus_ref {
            &self.cold.player_response_body
        } else if let Some(body) = self.static_body(corpus_ref) {
            body
        } else {
            panic!("tmnf world: contact delivered to an unknown response body");
        }
    }

    fn body_iso(&self, corpus_ref: u32) -> GmIso4 {
        self.resolve_body_iso(corpus_ref)
    }

    fn get_speed(&self, corpus_ref: u32, position: &GmVec3) -> GmVec3 {
        if corpus_ref == self.player_corpus_ref {
            response_get_speed(&self.cold.player_response_body,
                Some(&self.dyna), position)
        } else if let Some(body) = self.static_body(corpus_ref) {
            response_get_speed(body, None, position)
        } else {
            panic!("tmnf world: contact delivered to an unknown response body");
        }
    }

    fn add_replacement(&mut self, corpus_ref: u32, d: &GmVec3) {
        if corpus_ref == self.player_corpus_ref {
            self.dyna.add_replacement(d);
        }
    }

    fn solve_body_impulse(&mut self, corpus_ref: u32, collision: &GmCollision,
                          speed: &GmVec3, friction_product: f32, restitution: f32) {
        if corpus_ref == self.player_corpus_ref {
            let body = self.cold.player_response_body;
            crate::response::solve_body_impulse(&body, Some(&mut self.dyna),
                collision, speed, friction_product, restitution);
        }
    }

    fn absorb_contact(&mut self, corpus_ref: u32, contact: &mut CHmsPhysicalContact) {
        if corpus_ref != self.player_corpus_ref {
            panic!("tmnf world: contact delivered to the wrong response body");
        }
        /* Split borrows: the vehicle ctx takes &mut vehicle + &mut dyna while
         * the static body resolver reads the immutable track. The player's
         * contact `other_body` is always a static corpus. */
        {
            let World { vehicle, dyna, cold, track, .. } = self;
            let track_ref: &TmnfTrack = track;
            let static_iso = |corpus: u32| -> GmIso4 {
                match track_ref.static_response_bodies.get(corpus as usize) {
                    Some(b) if track_ref.static_response_present[corpus as usize] != 0 => {
                        b.iso
                    }
                    _ => panic!("tmnf world: contact delivered to an unknown response body"),
                }
            };
            let mut ctx = VehicleCtx { vehicle, tuning: &cold.tuning, dyna };
            vehicle_absorb_contact(&mut ctx, contact, &static_iso);
        }
        self.vehicle.model6.side_contact = self.vehicle.side_contact;
    }

    fn response_material(&self, material_ref: u32) -> &CHmsResponseMaterial {
        if material_ref == 1 {
            &self.cold.response_material
        } else {
            panic!("tmnf world: unknown response material");
        }
    }

    fn surface_materials(&self) -> &[CPlugSurfaceMaterialData] {
        &self.track.materials
    }

    fn collisions(&mut self) -> &mut crate::buffer::CFastBufferShmsPhysicalCollision {
        &mut self.collision_buffer.collisions
    }
}

/* ---------------------------------------------------------------------------
 * Detection (the C's compute_surface_pair + static_tree_detect)
 * ------------------------------------------------------------------------- */

impl World {
    /// `compute_surface_pair`: surface1 is a vehicle node's surface at iso1;
    /// surface2 is a static entry's surface at iso2 with optional accel.
    #[allow(clippy::too_many_arguments)]
    fn compute_surface_pair(
        &mut self,
        node: usize,
        iso1: &GmIso4,
        entry_surface: Option<&CPlugSurface>,
        entry_iso: &GmIso4,
        accel2: Option<&TmnfMeshQueryAccel>,
        corpus2: u32,
        tree2_ref: u32,
    ) -> bool {
        let surface1 = match &self.nodes[node].surface {
            Some(s) => s.clone(),
            None => return false,
        };
        let surface2 = match entry_surface {
            Some(s) => s,
            None => return false,
        };

        /* Mergeable buffer when the vehicle shape is a sphere or ellipsoid. */
        let mergeable = matches!(
            self.nodes[node].surface.as_ref().map(|s| s.geom.surf_type()),
            Some(GM_SURF_SPHERE) | Some(GM_SURF_ELLIPSOID));

        let buffer_is_mergeable = mergeable;
        let node_count_before = if mergeable {
            self.contact_buffers[node].base.get_count()
        } else {
            self.collision_buffer.get_count()
        };
        let ok = if mergeable {
            compute_surface_collision(&surface1, iso1, surface2, entry_iso,
                accel2, &mut self.contact_buffers[node].base,
                &CollisionShapeDispatch::default()) != 0
        } else {
            compute_surface_collision(&surface1, iso1, surface2, entry_iso,
                accel2, &mut self.collision_buffer,
                &CollisionShapeDispatch::default()) != 0
        };
        if !ok {
            return false;
        }

        if buffer_is_mergeable && self.contact_buffers[node].active == 0 {
            self.contact_buffers[node].active = 1;
            self.merge_buffers.push(node);
        }

        let end;
        if buffer_is_mergeable {
            end = self.contact_buffers[node].base.get_count();
        } else {
            end = self.collision_buffer.get_count();
        }
        let (corpus1, tree1_ref, current_material) =
            (self.current_corpus1, self.nodes[node].object_ref,
             self.current_material);
        let buffer_target_merge = buffer_is_mergeable;
        for i in node_count_before..end {
            if buffer_target_merge {
                let record = self.contact_buffers[node].base.collisions.at_mut(i as usize);
                record.corpus1 = corpus1;
                record.tree1 = tree1_ref;
                record.corpus2 = corpus2;
                record.tree2 = tree2_ref;
                record.material = current_material;
            } else {
                let record = self.collision_buffer.collisions.at_mut(i as usize);
                record.corpus1 = corpus1;
                record.tree1 = tree1_ref;
                record.corpus2 = corpus2;
                record.tree2 = tree2_ref;
                record.material = current_material;
            }
        }
        true
    }

    /// 0x0053A120 `static_tree_detect`: the car tree vs the static group.
    fn static_tree_detect(&mut self, iso: &GmIso4, node: usize) {
        let flags = self.nodes[node].flags;
        if (flags & 0x80) == 0 {
            return;
        }

        let world = vehicle_tree_world_iso(&self.nodes[node], iso);
        let children = self.nodes[node].children.clone();
        for child in children {
            self.static_tree_detect(&world, child);
        }
        if self.nodes[node].surface.is_none() {
            return;
        }

        let mut world_box = GmBoxAligned::default();
        world_box.set_mult(&self.nodes[node].box_aligned, iso);
        if self.track.grid.cell_count != 0 {
            let cell = static_grid_locate(&self.track.grid, &world_box,
                self.nodes[node].surface.as_ref().unwrap());
            if let Some(cell) = cell {
                self.static_grid_scan(cell, &world_box, node, &world);
                return;
            }
        }

        let entry_count = self.track.entries.len();
        let mut i = 0usize;
        while i < entry_count {
            let entry = self.track.entries[i].clone();
            if !world_box.test_inter(&entry.box_aligned) {
                i += entry.skip_count as usize;
                continue;
            }
            if entry.surface.is_some() && (entry.tree_flags & 0x80) != 0 {
                let surface = self.track.surfaces
                    [entry.surface.unwrap() as usize].clone();
                self.compute_surface_pair(
                    node, &world, Some(&surface), &entry.iso, None,
                    entry.corpus_ref, entry.tree_ref);
            }
            i += 1;
        }
    }

    /// `static_grid_scan`: one cell of the region-restricted static grid.
    fn static_grid_scan(
        &mut self,
        cell: u32,
        world_query: &GmBoxAligned,
        node: usize,
        world: &GmIso4,
    ) {
        /* A local Arc handle keeps the immutable grid borrow off `self`. */
        let track = self.track.clone();
        let grid: &TmnfStaticGrid = &track.grid;
        let nodes_off = grid.cell_offsets[cell as usize] as usize;
        let count = grid.cell_counts[cell as usize] as usize;
        let mut i = 0usize;
        while i < count {
            let cell_node = grid.nodes[nodes_off + i];
            if !world_query.test_inter(&cell_node.box_aligned) {
                i += cell_node.skip as usize;
                continue;
            }
            i += 1;
            if cell_node.entry_index >= TMNF_CELL_NODE_EMPTY {
                continue;
            }
            let entry_index = cell_node.entry_index as usize;
            let entry = self.track.entries[entry_index].clone();
            let mesh_grid = track.grid.entry_mesh_grids[entry_index];
            let inverse_iso = track.grid.entry_inverse_isos[entry_index];
            let accel = TmnfMeshQueryAccel {
                inverse_iso: Some(&inverse_iso),
                mesh_grid: mesh_grid.map(|g| &track.grid.mesh_grids[g]),
            };
            let surface = entry.surface.map(|s| {
                track.surfaces[s as usize].clone()
            });
            self.compute_surface_pair(
                node, world, surface.as_ref(), &entry.iso, Some(&accel),
                entry.corpus_ref, entry.tree_ref);
        }
    }

    /// 0x0053B1C0 `DetectCollisionsCorpus` for the single car corpus.
    fn detect_collisions_corpus(&mut self, active: bool) {
        /* group_index = (flags >> 13) & 0xf == 1 -> groups[0]; the dynamic
         * group holds one device mat: the static group, perform rows=1,
         * columns=0 (no dynamic-dynamic pairs). */
        if active {
            self.current_material = 1;
            self.static_group_bound = true;
            self.current_corpus1 = self.player_corpus_ref;
        }
        /* static_entry_count > 1 in practice. */
        let corpus_iso = self.dyna.live().rot_pos_iso();
        /* The corpus tree is the root: detect from the root's children. */
        let root_children = self.root_children.clone();
        for child in root_children {
            self.static_tree_detect(&corpus_iso, child);
        }

        if !active {
            return;
        }
        let merge: Vec<usize> = std::mem::take(&mut self.merge_buffers);
        for node in merge {
            let mut buffer = std::mem::replace(
                &mut self.contact_buffers[node], SHmsSphereBufferContact::new());
            shms_sphere_buffer_contact_merge_and_add_to_collisions(
                &mut buffer, &mut self.collision_buffer);
            self.contact_buffers[node] = buffer;
        }
    }
}

/* ---------------------------------------------------------------------------
 * The tick (physics.c: ComputeCorpusForces + step_dynamic_corpus +
 * PhysicsStep2)
 * ------------------------------------------------------------------------- */

impl World {
    /// 0x005481A0 `CHmsZoneDynamic_ComputeCorpusForces`.
    fn compute_corpus_forces(&mut self, dt: f32) {
        self.dyna.validate_dynamic_state();
        if (self.scene_flags & 0x00100000) != 0 {
            self.dyna.set_force(&GmVec3::ZERO);
            self.dyna.set_torque(&GmVec3::ZERO);
            return;
        }

        let mut force = GmVec3::ZERO;
        if let Some(value) = self.gravity() {
            let scale = self.dyna.params.force_field_scale * self.dyna.params.mass;
            force.x = (scale * value.x) + force.x;
            force.y = (value.y * scale) + force.y;
            force.z = (scale * value.z) + force.z;
        }

        let linear_speed = self.dyna.get_linear_speed();
        let linear_drag = -(self.linear_drag_scale * self.dyna.params.drag_linear);
        force.x = (linear_drag * linear_speed.x) + force.x;
        force.y = (linear_speed.y * linear_drag) + force.y;
        force.z = (linear_drag * linear_speed.z) + force.z;
        self.dyna.set_force(&force);

        if self.dyna.mode == 1 {
            let angular_speed = self.dyna.get_angular_speed();
            let angular_drag = -(self.angular_drag_scale * self.dyna.params.drag_angular);
            let torque = GmVec3 {
                x: angular_drag * angular_speed.x,
                y: angular_speed.y * angular_drag,
                z: angular_drag * angular_speed.z,
            };
            self.dyna.set_torque(&torque);
        }

        /* The vehicle force pipeline (TMNFVehicleComputeForces adapter). */
        {
            let ext = Model6External {
                model_iso_source: self.dyna.live().rot_pos_iso(),
                body_reference_position: self.cold.body_reference_position,
            };
            let water = &self.track.water;
            let cold = self.cold.clone();
            let track = self.track.clone();
            let mut post = |ctx: &mut VehicleCtx| {
                /* post_vehicle_force. */
                ctx.vehicle.engine.braking_factor =
                    ctx.vehicle.compute.computed_brake_force;
            };
            let contact_token = |corpus: u32| corpus;
            let body_iso = |corpus: u32| -> GmIso4 {
                match track.static_response_bodies.get(corpus as usize) {
                    Some(b) if track.static_response_present[corpus as usize] != 0 => b.iso,
                    _ => panic!("tmnf world: contact delivered to an unknown response body"),
                }
            };
            let mut external = ComputeExternal {
                ext: &ext,
                water,
                post_force: &mut post,
                contact_token: &contact_token,
                body_iso: &body_iso,
                fake_contacts_active: false,
                water_forces_active: false,
            };
            let World { vehicle, dyna, .. } = self;
            let mut ctx = VehicleCtx {
                vehicle,
                tuning: &cold.tuning,
                dyna,
            };
            cscene_vehicle_car_compute_forces(&mut ctx, &mut external, dt);
        }

        /* set_surface_location for every wheel whose suspension integrated
         * inside the force pass (equivalent point: nothing reads the trees
         * until detection). */
        self.sync_wheel_surfaces();
    }

    /// `step_dynamic_corpus`: one adaptive integration/collision sequence.
    fn step_dynamic_corpus(&mut self, dt: f32) {
        let active = self.dyna.dirty_flag != 0;
        let mut substeps: u32 = 0;
        let mut remaining = dt;
        let mut step = 0.0f32;
        if active {
            self.dyna.copy_state_to_temp();

            let linear_speed = self.dyna.get_linear_speed();
            let angular_speed = self.dyna.get_angular_speed();
            let speed_sum = length3(&linear_speed) + length3(&angular_speed);
            let numerator = dt * speed_sum;
            let quotient = numerator / self.dyna.params.substep_len;
            substeps = (crate::fp::ftol(quotient as f64) as u32).wrapping_add(1);
            if substeps > 1000 {
                substeps = 1000;
            }
            if substeps > 1 {
                step = ((dt as f64) / (substeps as f64)) as f32;
            }
        }

        let rounds = substeps;
        for i in 1..rounds {
            let stepping = active && i < substeps;
            if stepping {
                self.compute_corpus_forces(step);
                self.dyna.do_pre_collision_dynamic(step);
            }
            self.detect_and_respond(stepping);
            if stepping {
                self.dyna.do_post_collision_dynamic();
                remaining = remaining - step;
            }
        }

        if active {
            self.compute_corpus_forces(remaining);
            self.dyna.do_pre_collision_dynamic(remaining);
        }
        self.detect_and_respond(active);
        if active {
            self.dyna.do_post_collision_dynamic();
            self.dyna.copy_temp_to_state();
        }
    }

    /// `detect_and_respond`: one collision detection + response pass.
    fn detect_and_respond(&mut self, active: bool) {
        if active {
            self.collision_buffer.collisions.data.clear();
        }
        self.detect_collisions_corpus(active);
        if !active {
            return;
        }
        /* The route's waypoint triggers stand in for the game's checkpoint
         * corpora. */
        if let Some(route) = &self.route {
            let corpus_iso = self.dyna.live().rot_pos_iso();
            let car_trees = self.car_tree_snapshot();
            let mask = crate::race::tmnf_race_trigger_contact_mask(
                route, &car_trees, 0, &corpus_iso);
            self.trigger_contacts |= mask;
        }
        crate::response::compute_collision_response(self);

    }

    /// 0x00549C90 `CHmsZoneDynamic_PhysicsStep2`: one physics tick.
    pub fn physics_step2(&mut self, tick_ms: u32) {
        let dt = (tick_ms as i32) as f32 * 0.001f32;
        self.trigger_contacts = 0;

        /* Initial force/integration pass over the zone's dynamic corpora. */
        {
            let scene_flags = self.scene_flags;
            if (scene_flags & 0x0001E000) == 0 {
                self.compute_corpus_forces(dt);
                self.dyna.do_pre_collision_dynamic(dt);
            }
        }

        /* PrepareCollisions: the single dynamic group's speed table. */

        /* The game has five collision groups; the world binds group 0
         * (dynamic, one corpus: the car) and group 4 (static, from the
         * track). The loop only steps non-static groups. */
        {
            /* Group 0: the car corpus. */
            self.step_dynamic_corpus(dt);
        }
    }
}

fn length3(v: &GmVec3) -> f32 {
    (((v.y * v.y) + (v.x * v.x)) + (v.z * v.z)).sqrt()
}

/* ---------------------------------------------------------------------------
 * Respawn (vehicle_respawn.c)
 * ------------------------------------------------------------------------- */

fn wheel_state_reset(state: &mut crate::vehicle::CSceneVehicleCarWheelState) {
    /* 0x007BCA20: zero the listed dwords, the 16-bit word at +0x0c, and
     * store the identity matrix at +0x30. */
    const ZEROED: [usize; 15] = [
        0x00, 0x04, 0x08, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28,
        0x2c, 0x54, 0x58, 0x5c, 0x60,
    ];
    for o in ZEROED {
        state.bytes[o..o + 4].copy_from_slice(&0u32.to_le_bytes());
    }
    state.bytes[0x0c..0x0e].copy_from_slice(&0u16.to_le_bytes());
    let identity: [u8; 36] = {
        let mut b = [0u8; 36];
        let one = 1.0f32.to_le_bytes();
        let zero = 0.0f32.to_le_bytes();
        for i in 0..9 {
            let src = if i % 4 == 0 { &one } else { &zero };
            b[4 * i..4 * i + 4].copy_from_slice(src);
        }
        b
    };
    state.bytes[0x30..0x54].copy_from_slice(&identity);
}

impl World {
    /// 0x0047BF00 `World_Respawn`: reset the car and place it at `spawn`.
    pub fn respawn(&mut self, spawn: &GmIso4) {
        /* vehicle_reset (0x007C0320 + base 0x007CB6B0). */
        {
            let grounded_force_field_scale =
                self.cold.tuning.compute.grounded_force_field_scale;

            let v = &mut self.vehicle;
            v.input_gas = 0.0;
            v.input_brake = 0.0;
            v.input_steer = 0.0;
            v.compute.air_effect_threshold = 0.0;
            v.compute.spring_a.value = 0.0;
            v.compute.spring_a.target = 0.0;
            v.compute.spring_a.velocity = 0.0;
            v.compute.spring_c.value = 0.0;
            v.compute.spring_c.target = 0.0;
            v.compute.spring_c.velocity = 0.0;
            v.compute.contact_rise = 0.0;
            v.compute.contact_decay = 0.0;
        }
        dyna_reset(&mut self.dyna);
        {
            let v = &mut self.vehicle;
            v.steering_value = 0.0;
            v.turbo_progress = 0.0;
            v.turbo_type = TMNFVehicleTurboType::None;
            v.compute.air_impulse_cooldown_tick = 0;
            v.air_control_immediate = 0;
            v.compute.state_5d8 = 0;
            v.model6.contact_block_count = 0;
            v.side_contact = 0;
            v.model6.side_contact = 0;
            v.air_control_tick = 0;
            v.air_control_speed = GmVec3::ZERO;
            v.turbo_factor = 0.0;
            v.turbo_active = 0;
            v.last_side_contact_tick = u32::MAX;
            v.model6.last_sliding_tick = u32::MAX;
            v.model6.sliding_start_tick = u32::MAX;
            v.air_control_locked = 0;
            v.airborne_friction_gate = 0;
            v.force_wheel_speed = 0;
            v.model6.burnout_start_tick = u32::MAX;
            v.model6.burnout_transition_tick = u32::MAX;
            v.model6.model_iso = GmIso4::IDENTITY;
            v.model6.orbit_axis = GmVec3::ZERO;
            v.current_local_speed = GmVec3::ZERO;
            v.compute.effect_accumulator = 0.0;
            v.drive_mode = 0;
            v.engine_mode = 0;
            v.engine_limit_flag = 0;
            v.block_wheel_speed = 0;
            v.compute.event_level_a = 0;
            v.compute.event_level_b = 0;
            v.compute.event_level_c = 0;
            v.compute.event_source_ab = 0;
            v.compute.event_source_c = 0;
            v.compute.peak_event_level_b = 0;
            v.compute.peak_event_level_a = 0;
            v.compute.peak_event_level_c = 0;
            v.compute.event_metric_a = 0.0;
            v.compute.peak_event_source_ab = 0;
            v.compute.event_metric_b = 0.0;
            v.compute.peak_event_source_c = 0;
            v.compute.event_metric_c = 0.0;
            v.compute.last_force_tick = 0;
            v.total_force_added = GmVec3::ZERO;
            v.total_impulse_added = GmVec3::ZERO;
        }

        /* wheel_reset (0x007BD2A0) per wheel. */
        let rest = self.cold.tuning.base.suspension_rest_length;
        for index in 0..4 {
            let rest = rest;
            let minus_rest = -rest;
            let minus_zero = minus_rest * 0.0f32;

            let v = &mut self.vehicle;
            let wheel = &mut v.wheels[index];
            wheel.real_time.field08 = 0.0;
            wheel.real_time.field04 = 0.0;
            wheel.real_time.damper_absorb = rest;
            wheel.surface_location = wheel.surface_source;
            wheel.surface_location.t[0] = wheel.surface_location.t[0] + minus_zero;
            wheel.surface_location.t[1] = wheel.surface_location.t[1] + minus_rest;
            wheel.surface_location.t[2] = wheel.surface_location.t[2] + minus_zero;
            wheel.real_time.field6c = 0.0;
            wheel.real_time.rotation_phase = 0.0;
            wheel.real_time.has_ground_contact = 0;
            wheel.real_time.basis1 = GmMat3::IDENTITY;
            wheel.real_time.field54 = GmVec3::ZERO;
            wheel.real_time.field90 = GmVec3::ZERO;
            wheel.real_time.reserved60 = [0; 0x0c];
            wheel.real_time.blend_value = 0.0;
            wheel.real_time.blend_target = 0.0;
            /* 16-bit store: the upper half of the material word is kept. */
            wheel.real_time.contact_material_id &= !0xFFFFi32;
            wheel.real_time.is_sliding = 0;
            wheel.real_time.ground_contact_count = 0;
            wheel.field15c = 0;
            wheel.contact_relative_local_distance = GmVec3::ZERO;
            wheel_state_reset(&mut wheel.field234);
            wheel_state_reset(&mut wheel.async_state);
            wheel_state_reset(&mut wheel.previous_sync);
            wheel_state_reset(&mut wheel.sync);
        }
        self.sync_wheel_surfaces();

        /* SEngine::Reset (0x007BC9A0). */
        {
            let v = &mut self.vehicle;
            v.engine.reverse = 0;
            v.engine.rpm = 0.0;
            v.engine.gear = 1;
            v.engine.target_rpm = 0.0;
            v.engine.braking_factor = 0.0;
            v.engine.shift_timer = 0.0;
            v.engine.clutch = 1.0;
        }
        /* The dyna model words the force caller rewrites every tick. */
        self.dyna.params.force_field_scale =
            self.cold.tuning.compute.grounded_force_field_scale;
        self.dyna.params.drag_linear = 0.0;
        {
            let v = &mut self.vehicle;
            v.body_contact_position_sum = GmVec3::ZERO;
            v.body_contact_normal_sum = GmVec3::ZERO;
            v.body_contact_count = 0;
            v.wheel_contact_absorb_count = 0;
            v.flag_60c = 0;
        }

        dyna_reset(&mut self.dyna);
        /* 0x007BC800 VehicleBlockSpeed2Set(0). */
        self.vehicle.integration_flags &= !0x20;
        dyna_set_location(&mut self.dyna, spawn);
    }
}

fn dyna_state_reset(state: &mut CHmsStateDyna) {
    state.lin_vel = GmVec3::ZERO;
    state.ang_vel = GmVec3::ZERO;
    state.force = GmVec3::ZERO;
    state.torque = GmVec3::ZERO;
    state.lin_vel_added = GmVec3::ZERO;
    state.tail[0] = 0.0;
    state.tail[1] = 0.0;
    state.tail[2] = 0.0;
    state.tail[3] = 0.0;
}

fn dyna_reset(dyna: &mut CHmsDyna) {
    let live = *dyna.live();
    dyna_state_reset(dyna.live_mut());
    let state_b = *dyna.state_b();
    let mut state_b = state_b;
    dyna_state_reset(&mut state_b);
    dyna.set_state_b(state_b);
    let mut temp = dyna.temp_state;
    dyna_state_reset(&mut temp);
    dyna.temp_state = temp;
    let _ = live;
    dyna.replacement_buf.clear();
}

/// 0x005338F0 `CHmsDyna::SetLocation`: state B first, then the live copy.
fn dyna_set_location(dyna: &mut CHmsDyna, spawn: &GmIso4) {
    let mut b = *dyna.state_b();
    b.quat.set_from_mat3(&GmMat3 { m: spawn.m });
    b.rot.m = spawn.m;
    b.pos = GmVec3 { x: spawn.t[0], y: spawn.t[1], z: spawn.t[2] };
    let mut inv = GmMat3 { m: b.rot.m };
    inv.set_mult(&GmMat3 { m: b.rot.m }, &dyna.params.inv_inertia_body);
    // GmMat3_SetMult(&b.invInertiaWorld, &b.rot, &params.invInertiaBody)
    // then GmMat3_MultTranspose(&b.invInertiaWorld, &b.rot):
    //   invInertiaWorld = (rot * Iinv_body) then this = rot^T * itself.
    {
        let mut iw = GmMat3 { m: b.rot.m };
        iw.set_mult(&GmMat3 { m: b.rot.m }, &dyna.params.inv_inertia_body);
        iw.mult_transpose(&GmMat3 { m: b.rot.m });
        b.inv_inertia_world = iw;
    }
    let _ = inv;
    dyna.set_state_b(b);
    let b = *dyna.state_b();
    let mut live = *dyna.live();
    live.quat = b.quat;
    live.rot.m = spawn.m;
    live.pos = GmVec3 { x: spawn.t[0], y: spawn.t[1], z: spawn.t[2] };
    live.inv_inertia_world = b.inv_inertia_world;
    dyna.set_live(live);
}

/* ---------------------------------------------------------------------------
 * Game-state export (World_WritePlayerGameState) + observation
 * ------------------------------------------------------------------------- */

impl World {
    /// `World_WritePlayerGameState`: the 0x878-byte car block and the
    /// 4x0x2fc-byte wheel blocks, with every simulated field patched in.
    pub fn write_player_game_state(&self) -> ([u8; CAR_SIZE], [u8; 4 * WHEEL_SIZE]) {
        let cold = &self.cold;
        let car_off = cold.header.car_offset as usize;
        let mut car = [0u8; CAR_SIZE];
        car.copy_from_slice(
            &cold.vehicle_blob[car_off..car_off + CAR_SIZE]);
        let wheels_off = cold.header.wheels_offset as usize;
        let mut wheels = [0u8; 4 * WHEEL_SIZE];
        wheels.copy_from_slice(
            &cold.vehicle_blob[wheels_off..wheels_off + 4 * WHEEL_SIZE]);

        let put_f32 = |b: &mut [u8], o: usize, v: f32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_i32 = |b: &mut [u8], o: usize, v: i32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_u32 = |b: &mut [u8], o: usize, v: u32| {
            b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        };
        let put_vec3 = |b: &mut [u8], o: usize, v: &GmVec3| {
            put_f32(b, o, v.x);
            put_f32(b, o + 4, v.y);
            put_f32(b, o + 8, v.z);
        };

        /* patch_model6_state */
        {
            let s = &self.vehicle.model6;
            put_vec3(&mut car, 0x1dc, &s.pivot_position);
            put_vec3(&mut car, 0x1e8, &s.pivot_axis);
            put_i32(&mut car, 0x5c4, s.reverse_mode);
            put_f32(&mut car, 0x5cc, s.reverse_speed_threshold);
            put_i32(&mut car, 0x5d8, s.contact_block_count);
            put_i32(&mut car, 0x5dc, s.side_contact);
            put_u32(&mut car, 0x62c, s.last_sliding_tick);
            put_u32(&mut car, 0x630, s.sliding_start_tick);
            put_u32(&mut car, 0x634, s.sliding_elapsed_ticks);
            for i in 0..9 {
                put_f32(&mut car, 0x6a4 + 4 * i, s.model_iso.m[i]);
            }
            for i in 0..3 {
                put_f32(&mut car, 0x6a4 + 36 + 4 * i, s.model_iso.t[i]);
            }
            put_vec3(&mut car, 0x6d4, &s.rollover_axis);
            put_vec3(&mut car, 0x6e0, &s.orbit_center);
            put_f32(&mut car, 0x6ec, s.orbit_initial_radius);
            put_f32(&mut car, 0x6f0, s.orbit_radius);
            put_u32(&mut car, 0x6f4, s.burnout_start_tick);
            put_u32(&mut car, 0x6f8, s.burnout_transition_tick);
            put_vec3(&mut car, 0x6fc, &s.orbit_axis);
            put_f32(&mut car, 0x708, s.orbit_sign);
            put_f32(&mut car, 0x840, s.axle_width);
        }
        /* patch_compute_state */
        {
            let s = &self.vehicle.compute;
            put_f32(&mut car, 0x1e8, s.simulation_gate);
            put_u32(&mut car, 0x1fc, s.event_level_c);
            car[0x200] = s.event_source_c;
            car[0x201] = s.event_source_ab;
            car[0x214..0x228].copy_from_slice(&spring_bytes(&s.spring_c));
            car[0x228..0x23c].copy_from_slice(&spring_bytes(&s.spring_a));
            put_f32(&mut car, 0x23c, s.contact_rise);
            put_f32(&mut car, 0x240, s.contact_decay);
            put_f32(&mut car, 0x5b0, s.computed_brake_force);
            put_i32(&mut car, 0x5d8, s.state_5d8);
            put_u32(&mut car, 0x610, s.air_impulse_cooldown_tick);
            put_f32(&mut car, 0x624, s.effect_accumulator);
            put_u32(&mut car, 0x650, s.last_force_tick);
            put_u32(&mut car, 0x654, s.event_level_a);
            put_u32(&mut car, 0x658, s.event_level_b);
            put_u32(&mut car, 0x660, s.peak_event_level_b);
            put_u32(&mut car, 0x664, s.peak_event_level_a);
            put_u32(&mut car, 0x668, s.peak_event_level_c);
            car[0x66c] = s.peak_event_source_ab;
            car[0x66d] = s.peak_event_source_c;
            put_f32(&mut car, 0x670, s.event_metric_a);
            put_f32(&mut car, 0x674, s.event_metric_b);
            put_f32(&mut car, 0x678, s.event_metric_c);
            put_vec3(&mut car, 0x6d4, &s.normalized_force);
            put_i32(&mut car, 0x74c, s.air_effect_mode);
        }

        let v = &self.vehicle;
        put_f32(&mut car, 0x050, v.input_gas);
        put_f32(&mut car, 0x054, v.input_brake);
        put_f32(&mut car, 0x058, v.input_steer);
        put_f32(&mut car, 0x2e0, v.compute.local_speed_limit);
        put_i32(&mut car, 0x2e4, v.engine_mode);
        put_u32(&mut car, 0x2e8, v.wheel_count);
        car[0x59c..0x5cc].copy_from_slice(&engine_bytes(&v.engine));
        put_u32(&mut car, 0x5d0, v.turbo_epoch_tick);
        put_i32(&mut car, 0x5d4, v.air_control_immediate);
        put_i32(&mut car, 0x5d8, v.compute.state_5d8);
        put_i32(&mut car, 0x5dc, v.side_contact);
        put_u32(&mut car, 0x5e0, v.last_side_contact_tick);
        put_i32(&mut car, 0x5e4, v.air_control_locked);
        put_f32(&mut car, 0x5e8, v.steering_value);
        put_f32(&mut car, 0x5f0, v.turbo_progress);
        put_f32(&mut car, 0x5f4, v.turbo_factor);
        put_u32(&mut car, 0x5f8, v.turbo_start_tick);
        put_u32(&mut car, 0x5fc, v.turbo_end_tick);
        put_i32(&mut car, 0x600, v.turbo_type as i32);
        put_u32(&mut car, 0x604, v.roulette_token);
        put_f32(&mut car, 0x608, v.roulette_value);
        put_i32(&mut car, 0x60c, v.flag_60c);
        put_u32(&mut car, 0x614, v.air_control_tick);
        put_vec3(&mut car, 0x618, &v.air_control_speed);
        put_i32(&mut car, 0x628, v.turbo_active);
        put_i32(&mut car, 0x69c, v.drive_mode);
        put_i32(&mut car, 0x6a0, v.force_wheel_speed);
        put_vec3(&mut car, 0x70c, &v.current_local_speed);
        put_i32(&mut car, 0x73c, v.block_wheel_speed);
        put_i32(&mut car, 0x744, v.engine_limit_flag);
        put_i32(&mut car, 0x748, v.gear_downshift_flag);
        put_vec3(&mut car, 0x818, &v.total_force_added);
        put_vec3(&mut car, 0x824, &v.total_impulse_added);

        for i in 0..4usize {
            let raw = &mut wheels[i * WHEEL_SIZE..(i + 1) * WHEEL_SIZE];
            let wheel = &self.vehicle.wheels[i];
            put_i32(raw, 0x000, wheel.active);
            put_i32(raw, 0x004, wheel.steerable);
            put_f32(raw, 0x008, wheel.radius);
            for k in 0..9 {
                put_f32(raw, 0x10 + 4 * k, wheel.surface_source.m[k]);
            }
            put_f32(raw, 0x10 + 36, wheel.surface_source.t[0]);
            put_f32(raw, 0x10 + 40, wheel.surface_source.t[1]);
            put_f32(raw, 0x10 + 44, wheel.surface_source.t[2]);
            for k in 0..9 {
                put_f32(raw, 0x40 + 4 * k, wheel.surface_location.m[k]);
            }
            put_f32(raw, 0x40 + 36, wheel.surface_location.t[0]);
            put_f32(raw, 0x40 + 40, wheel.surface_location.t[1]);
            put_f32(raw, 0x40 + 44, wheel.surface_location.t[2]);
            for k in 0..12 {
                put_f32(raw, 0x70 + 4 * k, wheel.field70[k]);
            }
            put_i32(raw, 0xa0, wheel.fielda0);
            put_i32(raw, 0xa4, wheel.fielda4);
            put_vec3(raw, 0xa8, &wheel.offset_from_vehicle);
            raw[0xb4..0xb4 + 0xa8].copy_from_slice(
                &real_time_bytes(&wheel.real_time));
            put_i32(raw, 0x15c, wheel.field15c);
            put_vec3(raw, 0x160, &wheel.contact_relative_local_distance);
            raw[0x16c..0x1d0].copy_from_slice(&wheel.previous_sync.bytes);
            raw[0x1d0..0x234].copy_from_slice(&wheel.sync.bytes);
            raw[0x234..0x298].copy_from_slice(&wheel.field234.bytes);
            raw[0x298..0x2fc].copy_from_slice(&wheel.async_state.bytes);
        }

        (car, wheels)
    }

    /// `World_GetPlayerObservation`.
    pub fn player_observation(&self) -> Observation {
        let state = self.dyna.live();
        Observation {
            position: state.pos,
            rotation: state.quat,
            linear_speed: state.lin_vel,
            angular_speed: state.ang_vel,
            wheel_speed: [
                self.vehicle.wheels[0].real_time.field6c,
                self.vehicle.wheels[1].real_time.field6c,
                self.vehicle.wheels[2].real_time.field6c,
                self.vehicle.wheels[3].real_time.field6c,
            ],
            engine_rpm: self.vehicle.engine.rpm,
            gear: self.vehicle.engine.gear,
        }
    }
}

/// `TmnfObservation`.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub position: GmVec3,
    pub rotation: GmQuat,
    pub linear_speed: GmVec3,
    pub angular_speed: GmVec3,
    pub wheel_speed: [f32; 4],
    pub engine_rpm: f32,
    pub gear: i32,
}

fn spring_bytes(s: &GmSpringFloat) -> [u8; 0x14] {
    let mut b = [0u8; 0x14];
    b[0..4].copy_from_slice(&s.stiffness.to_le_bytes());
    b[4..8].copy_from_slice(&s.damping.to_le_bytes());
    b[8..12].copy_from_slice(&s.value.to_le_bytes());
    b[12..16].copy_from_slice(&s.target.to_le_bytes());
    b[16..20].copy_from_slice(&s.velocity.to_le_bytes());
    b
}

fn engine_bytes(e: &crate::vehicle::CSceneVehicleCarEngine) -> [u8; 0x30] {
    let mut b = [0u8; 0x30];
    b[0..4].copy_from_slice(&e.max_rpm.to_le_bytes());
    b[0x14..0x18].copy_from_slice(&e.braking_factor.to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&e.rpm.to_le_bytes());
    b[0x1c..0x20].copy_from_slice(&e.target_rpm.to_le_bytes());
    b[0x20..0x24].copy_from_slice(&e.clutch.to_le_bytes());
    b[0x24..0x28].copy_from_slice(&e.shift_timer.to_le_bytes());
    b[0x28..0x2c].copy_from_slice(&e.reverse.to_le_bytes());
    b[0x2c..0x30].copy_from_slice(&e.gear.to_le_bytes());
    b
}

fn real_time_bytes(r: &crate::vehicle::CSceneVehicleCarWheelRealTimeState) -> [u8; 0xa8] {
    let f = |v: f32| v.to_le_bytes();
    let mut b = [0u8; 0xa8];
    b[0x00..0x04].copy_from_slice(&f(r.damper_absorb));
    b[0x04..0x08].copy_from_slice(&f(r.field04));
    b[0x08..0x0c].copy_from_slice(&f(r.field08));
    for i in 0..9 {
        b[0x0c + 4 * i..0x10 + 4 * i].copy_from_slice(&f(r.basis0.m[i]));
    }
    for i in 0..9 {
        b[0x30 + 4 * i..0x34 + 4 * i].copy_from_slice(&f(r.basis1.m[i]));
    }
    b[0x54..0x60].copy_from_slice(&r.field54.as_bytes());
    b[0x60..0x6c].copy_from_slice(&r.reserved60);
    b[0x6c..0x70].copy_from_slice(&f(r.field6c));
    b[0x70..0x74].copy_from_slice(&r.has_ground_contact.to_le_bytes());
    b[0x74..0x78].copy_from_slice(&r.contact_material_id.to_le_bytes());
    b[0x78..0x7c].copy_from_slice(&r.is_sliding.to_le_bytes());
    b[0x7c..0x88].copy_from_slice(&r.relative_rotz_axis.as_bytes());
    b[0x88..0x8c].copy_from_slice(&r.contact_body_word.to_le_bytes());
    b[0x8c..0x90].copy_from_slice(&r.ground_contact_count.to_le_bytes());
    b[0x90..0x9c].copy_from_slice(&r.field90.as_bytes());
    b[0x9c..0xa0].copy_from_slice(&f(r.rotation_phase));
    b[0xa0..0xa4].copy_from_slice(&f(r.blend_value));
    b[0xa4..0xa8].copy_from_slice(&f(r.blend_target));
    b
}

/* ---------------------------------------------------------------------------
 * The vehicle image as data (THE GAP FIX): full read/write of TMNFM6G1
 * ------------------------------------------------------------------------- */

/// The parsed vehicle image: every section kept raw (byte-exact round-trip)
/// with typed views for the authoring lane.
pub struct VehicleImage {
    /// The original bytes: `write` patches sections over this buffer, so
    /// every reserved byte and section gap round-trips exactly.
    pub raw: Vec<u8>,
    pub header: VehicleSnapshotHeader,
    /// Raw section bytes in file order, with their offsets — `write` rebuilds
    /// the file by rewriting these buffers in place.
    pub car: Vec<u8>,                    /* 0x878 */
    pub vehicle_struct: Vec<u8>,         /* 0x50 */
    pub tunings: Vec<(VehicleTuningDescriptor, Vec<u8>)>, /* 0x3ac each */
    pub wheels: Vec<u8>,                 /* 4 x 0x2fc */
    pub dyna: Vec<u8>,                   /* 0x344 */
    pub params: Vec<u8>,                 /* 0x5c */
    pub state: Vec<u8>,                  /* 0xb4 (180-byte CHmsStateDyna) */
    pub curves: Vec<(VehicleCurveDescriptor, Vec<f32>, Vec<f32>)>,
    pub gearboxes: Vec<(VehicleGearboxDescriptor, Vec<f32>)>,
    pub ground_ids: Vec<u8>,             /* 31 */
    pub materials: Vec<VehicleMaterialRecord>,
    pub collision_root: VehicleCollisionTreeRecord,
    pub collision_children: Vec<VehicleCollisionTreeRecord>,
    pub collision_material_ids: Vec<u8>,
}

impl VehicleImage {
    /// Reads and validates a TMNFM6G1 image (`World_CreateFromBlob`'s
    /// validator, minus the track-hash pin).
    pub fn read(blob: &[u8]) -> VehicleImage {
        let graph = validate_vehicle_graph(blob, None, false);
        let h = &graph.header;
        let section = |off: u32, len: usize| -> Vec<u8> {
            blob[off as usize..off as usize + len].to_vec()
        };
        VehicleImage {
            raw: blob.to_vec(),
            header: h.clone(),
            car: section(h.car_offset, CAR_SIZE),
            vehicle_struct: section(h.vehicle_struct_offset, VEHICLE_STRUCT_SIZE),
            tunings: graph.tuning_descriptors.iter()
                .map(|d| (*d, section(d.raw_offset, TUNING_SIZE)))
                .collect(),
            wheels: section(h.wheels_offset, 4 * WHEEL_SIZE),
            dyna: section(h.dyna_offset, DYNA_SIZE),
            params: section(h.params_offset, PARAMS_SIZE),
            state: section(h.state_offset, STATE_SIZE),
            curves: graph.curve_descriptors.iter()
                .map(|c| {
                    let n = c.count as usize;
                    let mut positions = Vec::with_capacity(n);
                    let mut values = Vec::with_capacity(n);
                    for i in 0..n {
                        positions.push(f32_at(blob, c.positions_offset as usize + 4 * i));
                        values.push(f32_at(blob, c.values_offset as usize + 4 * i));
                    }
                    (*c, positions, values)
                })
                .collect(),
            gearboxes: graph.gearbox_descriptors.iter()
                .map(|g| {
                    let mut values = Vec::with_capacity(GEARBOX_VALUE_COUNT);
                    for i in 0..GEARBOX_VALUE_COUNT {
                        values.push(f32_at(blob, g.data_offset as usize + 4 * i));
                    }
                    (*g, values)
                })
                .collect(),
            ground_ids: section(h.ground_ids_offset, 31),
            materials: graph.materials.clone(),
            collision_root: graph.collision_root.clone(),
            collision_children: graph.collision_children.clone(),
            collision_material_ids: blob[h.collision_material_ids_offset as usize
                ..h.collision_material_ids_offset as usize
                + graph.collision_children.iter()
                    .map(|t| t.material_count as usize).sum::<usize>()].to_vec(),
        }
    }

    /// Rebuilds the image bytes. Layout-preserving: every section is written
    /// back at its original offset, so `write(read(b)) == b` for any valid
    /// image whose section table is unchanged.
    pub fn write(&self) -> Vec<u8> {
        let mut out = self.raw.clone();
        debug_assert_eq!(out.len(), self.header.total_size as usize);
        let put = |out: &mut Vec<u8>, off: u32, bytes: &[u8]| {
            out[off as usize..off as usize + bytes.len()]
                .copy_from_slice(bytes);
        };
        let h = &self.header;
        /* Header. */
        let mut head = vec![0u8; 320];
        let g = |b: &[u8], o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f = |b: &[u8], o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        /* Rebuild the header from the live fields (round-trips because the
         * reader filled them from the same bytes). */

        head[0..8].copy_from_slice(&self.header.magic);
        head[0x08..0x08+4].copy_from_slice(&(self.header.version).to_le_bytes());
        head[0x0c..0x0c+4].copy_from_slice(&(self.header.total_size).to_le_bytes());
        head[0x10..0x10+4].copy_from_slice(&(self.header.header_size).to_le_bytes());
        head[0x14..0x14+4].copy_from_slice(&(self.header.phase).to_le_bytes());
        head[0x18..0x18+4].copy_from_slice(&(self.header.tick).to_le_bytes());
        head[0x1c..0x1c+4].copy_from_slice(&(self.header.car_id).to_le_bytes());
        head[0x20..0x20+4].copy_from_slice(&(self.header.vehicle_struct_id).to_le_bytes());
        head[0x24..0x24+4].copy_from_slice(&(self.header.tuning_container_id).to_le_bytes());
        head[0x28..0x28+4].copy_from_slice(&(self.header.wheels_id).to_le_bytes());
        head[0x2c..0x2c+4].copy_from_slice(&(self.header.item_id).to_le_bytes());
        head[0x30..0x30+4].copy_from_slice(&(self.header.corpus_id).to_le_bytes());
        head[0x34..0x34+4].copy_from_slice(&(self.header.dyna_id).to_le_bytes());
        head[0x38..0x38+4].copy_from_slice(&(self.header.params_id).to_le_bytes());
        head[0x3c..0x3c+4].copy_from_slice(&(self.header.state_id).to_le_bytes());
        head[0x40..0x40+4].copy_from_slice(&(self.header.model_iso_id).to_le_bytes());
        head[0x44..0x44+4].copy_from_slice(&(self.header.body_reference_id).to_le_bytes());
        head[0x48..0x48+4].copy_from_slice(&(self.header.wheel_count).to_le_bytes());
        head[0x4c..0x4c+4].copy_from_slice(&(self.header.ground_id_count).to_le_bytes());
        head[0x50..0x50+4].copy_from_slice(&(self.header.ground_material_count).to_le_bytes());
        head[0x54..0x54+4].copy_from_slice(&(self.header.tuning_count).to_le_bytes());
        head[0x58..0x58+4].copy_from_slice(&(self.header.active_tuning_key).to_le_bytes());
        head[0x5c..0x5c+4].copy_from_slice(&(self.header.curve_count).to_le_bytes());
        head[0x60..0x60+4].copy_from_slice(&(self.header.gearbox_count).to_le_bytes());
        head[0x64..0x64+4].copy_from_slice(&(self.header.car_offset).to_le_bytes());
        head[0x68..0x68+4].copy_from_slice(&(self.header.vehicle_struct_offset).to_le_bytes());
        head[0x6c..0x6c+4].copy_from_slice(&(self.header.tuning_descriptors_offset).to_le_bytes());
        head[0x70..0x70+4].copy_from_slice(&(self.header.wheels_offset).to_le_bytes());
        head[0x74..0x74+4].copy_from_slice(&(self.header.dyna_offset).to_le_bytes());
        head[0x78..0x78+4].copy_from_slice(&(self.header.params_offset).to_le_bytes());
        head[0x7c..0x7c+4].copy_from_slice(&(self.header.state_offset).to_le_bytes());
        head[0x80..0x80+4].copy_from_slice(&(self.header.curve_descriptors_offset).to_le_bytes());
        head[0x84..0x84+4].copy_from_slice(&(self.header.gearbox_descriptors_offset).to_le_bytes());
        head[0x88..0x88+4].copy_from_slice(&(self.header.ground_ids_offset).to_le_bytes());
        head[0x8c..0x8c+4].copy_from_slice(&(self.header.ground_materials_offset).to_le_bytes());
        head[0x90..0x90+4].copy_from_slice(&(self.header.model_iso_offset).to_le_bytes());
        head[0x94..0x94+4].copy_from_slice(&(self.header.body_reference_offset).to_le_bytes());
        head[0x98..0x98+4].copy_from_slice(&(self.header.curve_data_offset).to_le_bytes());
        head[0x9c..0x9c+4].copy_from_slice(&(self.header.gearbox_data_offset).to_le_bytes());
        head[0xa0..0xa0+4].copy_from_slice(&(self.header.collision_child_count).to_le_bytes());
        head[0xa4..0xa4+4].copy_from_slice(&(self.header.collision_root_offset).to_le_bytes());
        head[0xa8..0xa8+4].copy_from_slice(&(self.header.collision_children_offset).to_le_bytes());
        head[0xac..0xac+4].copy_from_slice(&(self.header.collision_material_ids_offset).to_le_bytes());
        for (i, v) in [self.header.model_value, self.header.lateral_force_factor,
            self.header.longitudinal_force_factor, self.header.steering_angle]
            .iter().enumerate() {
            head[0xb0 + 4 * i..0xb4 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        head[0xc0..0xc4].copy_from_slice(&self.header.grounded.to_le_bytes());
        head[0xc4..0xd0].copy_from_slice(&self.header.existing_force.as_bytes());
        head[0xd0..0xdc].copy_from_slice(&self.header.local_speed.as_bytes());
        head[0xdc..0xe8].copy_from_slice(&self.header.local_angular_speed.as_bytes());
        for i in 0..4 {
            head[0xe8 + 4 * i..0xec + 4 * i]
                .copy_from_slice(&self.header.material[i].to_le_bytes());
        }
        head[0xf8..0xfc].copy_from_slice(&self.header.sliding.to_le_bytes());
        head[0xfc..0x100].copy_from_slice(&self.header.brake_force.to_le_bytes());
        head[0x100..0x120].copy_from_slice(&self.header.source_exe_sha256);
        head[0x120..0x140].copy_from_slice(&self.header.source_track_sha256);
        let _ = (g, f);
        put(&mut out, 0, &head);

        put(&mut out, h.car_offset, &self.car);
        put(&mut out, h.vehicle_struct_offset, &self.vehicle_struct);
        for (d, bytes) in &self.tunings {
            put(&mut out, d.raw_offset, bytes);
            let mut db = [0u8; 16];
            db[0..4].copy_from_slice(&d.key.to_le_bytes());
            db[4..8].copy_from_slice(&d.object_id.to_le_bytes());
            db[8..12].copy_from_slice(&d.raw_offset.to_le_bytes());
            db[12..16].copy_from_slice(&d.raw_size.to_le_bytes());
            put(&mut out, h.tuning_descriptors_offset
                + 16 * d.key as u32, &db);
        }
        put(&mut out, h.wheels_offset, &self.wheels);
        put(&mut out, h.dyna_offset, &self.dyna);
        put(&mut out, h.params_offset, &self.params);
        put(&mut out, h.state_offset, &self.state);
        for (i, (c, positions, values)) in self.curves.iter().enumerate() {
            let mut cb = [0u8; 40];
            for (o, v) in [(0usize, c.owner_kind), (4, c.owner_key), (8, c.field_offset),
                           (12, c.object_id), (16, c.positions_id), (20, c.values_id),
                           (24, c.count), (32, c.positions_offset), (36, c.values_offset)] {
                cb[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
            cb[28..32].copy_from_slice(&c.interpolation.to_le_bytes());
            put(&mut out, h.curve_descriptors_offset + 40 * i as u32, &cb);
            for (k, p) in positions.iter().enumerate() {
                put(&mut out, c.positions_offset + 4 * k as u32,
                    &p.to_le_bytes());
            }
            for (k, v) in values.iter().enumerate() {
                put(&mut out, c.values_offset + 4 * k as u32,
                    &v.to_le_bytes());
            }
        }
        for (i, (g, values)) in self.gearboxes.iter().enumerate() {
            let mut gb = [0u8; 24];
            for (o, v) in [(0usize, g.owner_key), (4, g.field_offset), (8, g.buffer_id),
                           (12, g.data_id), (16, g.count), (20, g.data_offset)] {
                gb[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
            put(&mut out, h.gearbox_descriptors_offset + 24 * i as u32, &gb);
            for (k, v) in values.iter().enumerate() {
                put(&mut out, g.data_offset + 4 * k as u32,
                    &v.to_le_bytes());
            }
        }
        put(&mut out, h.ground_ids_offset, &self.ground_ids);
        for (i, m) in self.materials.iter().enumerate() {
            let mut mb = [0u8; 40];
            mb[0..4].copy_from_slice(&m.object_id.to_le_bytes());
            for k in 0..4 {
                mb[4 + 4 * k..8 + 4 * k]
                    .copy_from_slice(&m.values[k].to_le_bytes());
            }
            mb[20..24].copy_from_slice(&m.fake_contact_mask.to_le_bytes());
            mb[24..28].copy_from_slice(&m.fake_contact_period_x.to_le_bytes());
            mb[28..32].copy_from_slice(&m.fake_contact_period_z.to_le_bytes());
            mb[32..36].copy_from_slice(&m.fake_contact_impulse_scale.to_le_bytes());
            mb[36..40].copy_from_slice(&m.fake_contact_impulse_limit.to_le_bytes());
            put(&mut out, h.ground_materials_offset + 40 * i as u32, &mb);
        }
        /* Collision tree records. */
        let write_record = |out: &mut Vec<u8>, off: u32, r: &VehicleCollisionTreeRecord| {
            let mut b = [0u8; 132];
            for (o, v) in [(0x00usize, r.object_id), (0x04, r.flags), (0x08, r.surface_id),
                           (0x0c, r.geometry_id), (0x10, r.kind), (0x14, r.wheel_index),
                           (0x18, r.material_index), (0x1c, r.material_count)] {
                b[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
            b[0x20..0x22].copy_from_slice(&r.geometry_material_index.to_le_bytes());
            b[0x22] = r.geometry_type;
            b[0x23] = r.geometry_reserved;
            b[0x24..0x3c].copy_from_slice(&r.box_aligned.as_bytes());
            let iso = iso4_bytes(&r.local_iso);
            b[0x3c..0x6c].copy_from_slice(&iso);
            b[0x6c..0x84].copy_from_slice(&r.shape);
            put(out, off, &b);
        };
        write_record(&mut out, h.collision_root_offset, &self.collision_root);
        for (i, r) in self.collision_children.iter().enumerate() {
            write_record(&mut out, h.collision_children_offset
                + 132 * i as u32, r);
        }
        put(&mut out, h.collision_material_ids_offset, &self.collision_material_ids);
        out
    }
}

fn iso4_bytes(iso: &GmIso4) -> [u8; 48] {
    let mut b = [0u8; 48];
    for i in 0..9 {
        b[4 * i..4 * i + 4].copy_from_slice(&iso.m[i].to_le_bytes());
    }
    for (i, t) in iso.t.iter().enumerate() {
        b[36 + 4 * i..40 + 4 * i].copy_from_slice(&t.to_le_bytes());
    }
    b
}
