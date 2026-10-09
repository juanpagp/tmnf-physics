//! `TMNFROU1` immutable race-route snapshot loader (`src/route.h` /
//! `src/route.c`), transliterated.
//!
//! The C loader mmaps the snapshot, fixes the on-disk relative offsets
//! (`metadata->start_rel` etc.) up to native pointers inside a private
//! mapping, validates, then mprotects the mapping read-only. This port reads
//! the whole file into a `Vec<u8>` and decodes **owned** structures:
//!
//! * the four `*_rel` fixups become validated byte offsets (the resolved
//!   arrays are parsed into `TmnfRoute`'s owned fields); `TmnfRouteMetadata`
//!   keeps the on-disk offset values, since the C's fixed-up values are just
//!   `base + offset` and the owned port has no `base`;
//! * `TmnfRoute` drops `mapping`/`mapping_size` (no mmap) — ownership replaces
//!   the read-only mapping, and `TmnfRoute_Unload` becomes plain `Drop`;
//! * the projection BVH is built exactly as the C builds it (same median
//!   splits, same node order, same `projection_bvh_count`/`height`).
//!
//! Every validation, its order and its panic message mirror the C `fail()`
//! (`tmnf_fail` prints `tmnf: <message>` and aborts; this port panics with
//! `tmnf route: <message>`).
//!
//! # Documented deviations (all structural, none arithmetic)
//!
//! * `expected_track_sha256` is `Option`: the C would read through a NULL
//!   pointer, so `None` (like `tmnf_track_load`) skips only the track-hash
//!   comparison.
//! * `project_in_memory_route` (the BVH-less fallback for a GPU device copy
//!   of the route, guarded by `route->mapping == NULL`) is **not** ported:
//!   an owned `TmnfRoute` built by [`tmnf_route_load`] always carries its
//!   BVH, so the branch is unreachable; the CUDA-device lane is out of scope
//!   for this crate.
//! * Allocation-failure fails ("route allocation failed", "projection BVH
//!   allocation failed") are unreachable: Rust aborts on OOM.
//! * `mprotect`/`munmap` failure handling is dropped with the mmap itself.

use crate::collision::GmBoxAligned;
use crate::gm::{GmIso4, GmMat3, GmQuat, GmVec3};
use crate::track::{sha256_digest, Reader, TMNF_21126_EXE_SHA256};
use std::fs;
use std::mem::size_of;

/* ---------------------------------------------------------------------------
 * Constants (route.h)
 * ------------------------------------------------------------------------- */

pub const TMNF_ROUTE_VERSION: u32 = 3;
pub const TMNF_ROUTE_SECTION_COUNT: usize = 5;

pub const TMNF_ROUTE_METADATA: usize = 0;
pub const TMNF_ROUTE_START: usize = 1;
pub const TMNF_ROUTE_CHECKPOINTS: usize = 2;
pub const TMNF_ROUTE_FINISH: usize = 3;
pub const TMNF_ROUTE_REFERENCE: usize = 4;

pub const TMNF_ROUTE_WAYPOINT_START: u32 = 0;
pub const TMNF_ROUTE_WAYPOINT_FINISH: u32 = 1;
pub const TMNF_ROUTE_WAYPOINT_CHECKPOINT: u32 = 2;
pub const TMNF_ROUTE_WAYPOINT_NONE: u32 = 3;
pub const TMNF_ROUTE_WAYPOINT_START_FINISH: u32 = 4;

/// `TMNF_ROUTE_MULTILAP` (1u << 0).
pub const TMNF_ROUTE_MULTILAP: u32 = 1 << 0;

/* ---------------------------------------------------------------------------
 * Byte-contract structs (on-disk layouts, parsed field-by-field)
 * ------------------------------------------------------------------------- */

/// 0x10 bytes. `TmnfRouteSection`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteSection {
    pub offset: u64,
    pub count: u32,
    pub stride: u32,
}
const _: () = assert!(size_of::<TmnfRouteSection>() == 0x10);

/// 0xE0 bytes. `TmnfRouteHeader`.
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TmnfRouteHeader {
    pub magic: [u8; 8], /* "TMNFROU1" */
    pub version: u32,
    pub endian: u32, /* 0x12345678 */
    pub header_size: u32, /* 0xE0 */
    pub section_count: u32, /* 5 */
    pub file_size: u64,
    pub exe_sha256: [u8; 32],
    pub track_sha256: [u8; 32],
    pub sections: [TmnfRouteSection; TMNF_ROUTE_SECTION_COUNT],
    pub payload_sha256: [u8; 32],
    pub reserved: [u8; 16],
}
const _: () = assert!(size_of::<TmnfRouteHeader>() == 0xE0);

/// 0x40 bytes. `TmnfRouteMetadata`. On disk the four `*_rel` fields are
/// offsets from the file base (the C loader fixes them up to pointers; this
/// port validates them and keeps the on-disk offsets).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteMetadata {
    pub lap_count: u32,
    pub checkpoint_count: u32,
    /// Alternatives; `checkpoint_count + finish_count <= 64`.
    pub finish_count: u32,
    /// Serialized `checkpoint_count + 2`.
    pub reference_count: u32,
    pub total_race_checkpoints: u32,
    pub race_checkpoint_limit: u32,
    pub flags: u32,
    /// Dense reference-section records.
    pub centerline_count: u32,
    pub start_rel: u64,
    pub checkpoints_rel: u64,
    pub finish_rel: u64,
    pub reference_rel: u64,
}
const _: () = assert!(size_of::<TmnfRouteMetadata>() == 0x40);

/// 0xAC bytes. `TmnfRouteInitialState` — deterministic, pointer-free prefix
/// of `CHmsStateDyna` (the two trailing game dwords are excluded because the
/// final one is a process-local pointer).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteInitialState {
    pub quat: GmQuat,
    pub rot: GmMat3,
    pub pos: GmVec3,
    pub lin_vel: GmVec3,
    pub lin_vel_added: GmVec3,
    pub ang_vel: GmVec3,
    pub force: GmVec3,
    pub torque: GmVec3,
    pub inv_inertia_world: GmMat3,
    pub not_tweaked_lin_vel: GmVec3,
}
const _: () = assert!(size_of::<TmnfRouteInitialState>() == 0xAC);

/// 0x114 bytes. `TmnfRouteStart`. `spawn` is `CGameCtnBlock::GetSpawnLoc`
/// (0x0060B410) of the start block.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteStart {
    pub transform: GmIso4,
    pub initial_state: TmnfRouteInitialState,
    pub block_index: u32,
    pub waypoint_type: u32,
    pub spawn: GmIso4,
}
const _: () = assert!(size_of::<TmnfRouteStart>() == 0x114);

/// 0x90 bytes. `TmnfRouteTrigger`. `box` is the exact local root CPlugTree
/// AABB used by TMNF collision, in the tree's parent frame (`transform` is
/// the corpus isometry); `spawn` is the checkpoint block's GetSpawnLoc;
/// `no_respawn` is CGameCtnBlockInfo+0x120 (OnCheckpoint 0x0047C330 then
/// keeps the previous spawn).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteTrigger {
    pub race_index: u32,
    pub block_index: u32,
    pub waypoint_type: u32,
    pub tree_flags: u32,
    pub box_aligned: GmBoxAligned,
    pub transform: GmIso4,
    pub spawn: GmIso4,
    pub no_respawn: u32,
    pub reserved: u32,
}
const _: () = assert!(size_of::<TmnfRouteTrigger>() == 0x90);

/// 0x18 bytes. `TmnfRouteReferencePoint`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteReferencePoint {
    pub position: GmVec3,
    pub arc_length: f32,
    pub half_width: f32,
    pub leg_index: u32,
}
const _: () = assert!(size_of::<TmnfRouteReferencePoint>() == 0x18);

/// `TmnfRouteProjection` (plain struct in the C, no size contract).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteProjection {
    pub arc_length: f32,
    pub lateral_offset: f32,
    pub half_width: f32,
    pub segment_index: u32,
    pub centerline_segment_index: u32,
    pub segments_tested: u32,
}

/// 36 bytes. `TmnfRouteBvhNode` — projection BVH node; leaves carry a
/// centerline segment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRouteBvhNode {
    pub minimum: [f32; 3],
    pub maximum: [f32; 3],
    pub left: u32,
    pub right: u32,
    pub segment: u32,
}
const _: () = assert!(size_of::<TmnfRouteBvhNode>() == 36);

/* ---------------------------------------------------------------------------
 * Owned route
 * ------------------------------------------------------------------------- */

/// `TmnfRoute` — the loaded snapshot. Field-for-field mirror of the C struct
/// except `mapping`/`mapping_size` (dropped: the arrays are owned) — see the
/// module docs. `finish` has `finish_count` entries; index zero owns the
/// reference line.
#[derive(Clone, Debug, Default)]
pub struct TmnfRoute {
    pub header: TmnfRouteHeader,
    pub metadata: TmnfRouteMetadata,
    pub start: TmnfRouteStart,
    pub checkpoints: Vec<TmnfRouteTrigger>,
    pub finish: Vec<TmnfRouteTrigger>,
    pub centerline: Vec<TmnfRouteReferencePoint>,
    pub projection_bvh: Vec<TmnfRouteBvhNode>,
    pub projection_bvh_count: u32,
    pub projection_bvh_height: u32,
}

/* ---------------------------------------------------------------------------
 * Parsers
 * ------------------------------------------------------------------------- */

fn read_quat(r: &mut Reader) -> GmQuat {
    GmQuat {
        x: r.f32(),
        y: r.f32(),
        z: r.f32(),
        w: r.f32(),
    }
}

fn read_mat3(r: &mut Reader) -> GmMat3 {
    let mut m = [0f32; 9];
    for v in &mut m {
        *v = r.f32();
    }
    GmMat3 { m }
}

fn read_box_aligned(r: &mut Reader) -> GmBoxAligned {
    GmBoxAligned {
        center: r.gm_vec3(),
        half_extent: r.gm_vec3(),
    }
}

fn parse_route_header(data: &[u8]) -> TmnfRouteHeader {
    let mut r = Reader::new(data);
    let mut magic = [0u8; 8];
    magic.copy_from_slice(r.bytes(8));
    let version = r.u32();
    let endian = r.u32();
    let header_size = r.u32();
    let section_count = r.u32();
    let file_size = r.u64();
    let mut exe_sha256 = [0u8; 32];
    exe_sha256.copy_from_slice(r.bytes(32));
    let mut track_sha256 = [0u8; 32];
    track_sha256.copy_from_slice(r.bytes(32));
    let mut sections = [TmnfRouteSection::default(); TMNF_ROUTE_SECTION_COUNT];
    for s in &mut sections {
        s.offset = r.u64();
        s.count = r.u32();
        s.stride = r.u32();
    }
    let mut payload_sha256 = [0u8; 32];
    payload_sha256.copy_from_slice(r.bytes(32));
    let mut reserved = [0u8; 16];
    reserved.copy_from_slice(r.bytes(16));
    TmnfRouteHeader {
        magic,
        version,
        endian,
        header_size,
        section_count,
        file_size,
        exe_sha256,
        track_sha256,
        sections,
        payload_sha256,
        reserved,
    }
}

fn parse_route_metadata(r: &mut Reader) -> TmnfRouteMetadata {
    let lap_count = r.u32();
    let checkpoint_count = r.u32();
    let finish_count = r.u32();
    let reference_count = r.u32();
    let total_race_checkpoints = r.u32();
    let race_checkpoint_limit = r.u32();
    let flags = r.u32();
    let centerline_count = r.u32();
    let start_rel = r.u64();
    let checkpoints_rel = r.u64();
    let finish_rel = r.u64();
    let reference_rel = r.u64();
    TmnfRouteMetadata {
        lap_count,
        checkpoint_count,
        finish_count,
        reference_count,
        total_race_checkpoints,
        race_checkpoint_limit,
        flags,
        centerline_count,
        start_rel,
        checkpoints_rel,
        finish_rel,
        reference_rel,
    }
}

fn parse_route_initial_state(r: &mut Reader) -> TmnfRouteInitialState {
    let quat = read_quat(r);
    let rot = read_mat3(r);
    let pos = r.gm_vec3();
    let lin_vel = r.gm_vec3();
    let lin_vel_added = r.gm_vec3();
    let ang_vel = r.gm_vec3();
    let force = r.gm_vec3();
    let torque = r.gm_vec3();
    let inv_inertia_world = read_mat3(r);
    let not_tweaked_lin_vel = r.gm_vec3();
    TmnfRouteInitialState {
        quat,
        rot,
        pos,
        lin_vel,
        lin_vel_added,
        ang_vel,
        force,
        torque,
        inv_inertia_world,
        not_tweaked_lin_vel,
    }
}

fn parse_route_start(r: &mut Reader) -> TmnfRouteStart {
    let transform = r.gm_iso4();
    let initial_state = parse_route_initial_state(r);
    let block_index = r.u32();
    let waypoint_type = r.u32();
    let spawn = r.gm_iso4();
    TmnfRouteStart {
        transform,
        initial_state,
        block_index,
        waypoint_type,
        spawn,
    }
}

fn parse_route_trigger(r: &mut Reader) -> TmnfRouteTrigger {
    let race_index = r.u32();
    let block_index = r.u32();
    let waypoint_type = r.u32();
    let tree_flags = r.u32();
    let box_aligned = read_box_aligned(r);
    let transform = r.gm_iso4();
    let spawn = r.gm_iso4();
    let no_respawn = r.u32();
    let reserved = r.u32();
    TmnfRouteTrigger {
        race_index,
        block_index,
        waypoint_type,
        tree_flags,
        box_aligned,
        transform,
        spawn,
        no_respawn,
        reserved,
    }
}

fn parse_route_reference_point(r: &mut Reader) -> TmnfRouteReferencePoint {
    let position = r.gm_vec3();
    let arc_length = r.f32();
    let half_width = r.f32();
    let leg_index = r.u32();
    TmnfRouteReferencePoint {
        position,
        arc_length,
        half_width,
        leg_index,
    }
}

/* ---------------------------------------------------------------------------
 * Section validation / pointer resolution
 * ------------------------------------------------------------------------- */

fn section_bytes(section: &TmnfRouteSection) -> u64 {
    let bytes = (section.count as u64) * (section.stride as u64);
    if section.stride != 0 && bytes / section.stride as u64 != section.count as u64 {
        panic!("tmnf route: section size overflow");
    }
    bytes
}

fn validate_sections(header: &TmnfRouteHeader, mapping_size: usize) {
    let expected_stride: [u32; TMNF_ROUTE_SECTION_COUNT] = [
        size_of::<TmnfRouteMetadata>() as u32,
        size_of::<TmnfRouteStart>() as u32,
        size_of::<TmnfRouteTrigger>() as u32,
        size_of::<TmnfRouteTrigger>() as u32,
        size_of::<TmnfRouteReferencePoint>() as u32,
    ];

    for i in 0..TMNF_ROUTE_SECTION_COUNT {
        let section = &header.sections[i];
        if section.stride != expected_stride[i]
            || (section.count == 0 && i != TMNF_ROUTE_CHECKPOINTS)
            || (section.offset & 7) != 0
            || section.offset < size_of::<TmnfRouteHeader>() as u64
        {
            panic!("tmnf route: invalid section descriptor");
        }
        let bytes = section_bytes(section);
        if section.offset > mapping_size as u64
            || bytes > mapping_size as u64 - section.offset
        {
            panic!("tmnf route: section outside file");
        }
    }
    for i in 0..TMNF_ROUTE_SECTION_COUNT {
        let a0 = header.sections[i].offset;
        let a1 = a0 + section_bytes(&header.sections[i]);
        for j in i + 1..TMNF_ROUTE_SECTION_COUNT {
            let b0 = header.sections[j].offset;
            let b1 = b0 + section_bytes(&header.sections[j]);
            if a0 < b1 && b0 < a1 {
                panic!("tmnf route: overlapping sections");
            }
        }
    }
}

/// `resolve_array`: validates a relative offset against a section and returns
/// the byte offset from the file base (the C returned `base + relative`).
fn resolve_array(section: &TmnfRouteSection, relative: u64, count: u32) -> u64 {
    let bytes = (count as u64) * (section.stride as u64);
    let section_end = section.offset + section_bytes(section);
    if relative < section.offset
        || relative > section_end
        || bytes > section_end - relative
        || (relative - section.offset) % section.stride as u64 != 0
    {
        panic!("tmnf route: relative pointer outside target section");
    }
    relative
}

/* ---------------------------------------------------------------------------
 * Record validation (fix_and_validate)
 * ------------------------------------------------------------------------- */

fn finite_floats(values: &[f32]) -> bool {
    for v in values {
        if !v.is_finite() {
            return false;
        }
    }
    true
}

fn finite_vec3(v: &GmVec3) -> bool {
    finite_floats(&[v.x, v.y, v.z])
}

/// `finite_floats((const float *)iso, 12)`.
fn finite_iso4(iso: &GmIso4) -> bool {
    finite_floats(&iso.m) && finite_floats(&iso.t)
}

/// `finite_floats((const float *)&trigger->box, 6)`.
fn finite_box(box_aligned: &GmBoxAligned) -> bool {
    finite_vec3(&box_aligned.center) && finite_vec3(&box_aligned.half_extent)
}

/// `finite_floats((const float *)&state, 43)` — 43 floats.
fn finite_initial_state(state: &TmnfRouteInitialState) -> bool {
    finite_floats(&[state.quat.x, state.quat.y, state.quat.z, state.quat.w])
        && finite_floats(&state.rot.m)
        && finite_vec3(&state.pos)
        && finite_vec3(&state.lin_vel)
        && finite_vec3(&state.lin_vel_added)
        && finite_vec3(&state.ang_vel)
        && finite_vec3(&state.force)
        && finite_vec3(&state.torque)
        && finite_floats(&state.inv_inertia_world.m)
        && finite_vec3(&state.not_tweaked_lin_vel)
}

/// `finite_floats((const float *)&point, 5)` — 5 floats.
fn finite_reference_point(point: &TmnfRouteReferencePoint) -> bool {
    finite_vec3(&point.position)
        && point.arc_length.is_finite()
        && point.half_width.is_finite()
}

/// `memcmp(a, b, n) == 0` over floats: bit comparison, so `-0.0`/`+0.0`
/// (equal under `==` but different bytes) mismatch exactly as in the C.
fn f32_bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// A spawn isometry is a proper rotation (the game builds it from a block
/// placement and a block-info isometry) with a finite translation.
fn valid_spawn(spawn: &GmIso4) -> bool {
    if !finite_iso4(spawn) {
        return false;
    }
    for row in 0..3usize {
        let r = &spawn.m[row * 3..row * 3 + 3];
        let length = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if (length - 1.0f32).abs() > 1e-3f32 {
            return false;
        }
    }
    true
}

fn validate_trigger(trigger: &TmnfRouteTrigger, index: u32, waypoint_type: u32) {
    if trigger.race_index != index
        || trigger.waypoint_type != waypoint_type
        || trigger.tree_flags == 0
        || !finite_box(&trigger.box_aligned)
        || !finite_iso4(&trigger.transform)
        || !(trigger.box_aligned.half_extent.x > 0.0f32)
        || !(trigger.box_aligned.half_extent.y > 0.0f32)
        || !(trigger.box_aligned.half_extent.z > 0.0f32)
        || !valid_spawn(&trigger.spawn)
        || trigger.no_respawn > 1
        || trigger.reserved != 0
    {
        panic!("tmnf route: invalid route trigger");
    }
}

/// `fix_and_validate`: every metadata/record check of the C, in order, plus
/// the four relative-pointer resolutions. Returns the decoded records.
fn fix_and_validate(
    data: &[u8],
    header: &TmnfRouteHeader,
) -> (
    TmnfRouteMetadata,
    TmnfRouteStart,
    Vec<TmnfRouteTrigger>,
    Vec<TmnfRouteTrigger>,
    Vec<TmnfRouteReferencePoint>,
) {
    let mut r = Reader::at(data, header.sections[TMNF_ROUTE_METADATA].offset as usize);
    let metadata = parse_route_metadata(&mut r);
    if header.sections[TMNF_ROUTE_METADATA].count != 1
        || header.sections[TMNF_ROUTE_START].count != 1
        || header.sections[TMNF_ROUTE_FINISH].count != metadata.finish_count
        || metadata.lap_count == 0
        || metadata.finish_count == 0
        || metadata.finish_count as u64 + metadata.checkpoint_count as u64 > 64
        || metadata.checkpoint_count != header.sections[TMNF_ROUTE_CHECKPOINTS].count
        || metadata.centerline_count != header.sections[TMNF_ROUTE_REFERENCE].count
        || metadata.reference_count != metadata.checkpoint_count + 2
        || metadata.centerline_count < metadata.reference_count
        || (metadata.flags & !TMNF_ROUTE_MULTILAP) != 0
        || ((metadata.flags & TMNF_ROUTE_MULTILAP) != 0) != (metadata.lap_count > 1)
    {
        panic!("tmnf route: invalid route metadata");
    }

    /* The C fixes the four relative offsets up to native pointers here; this
     * port validates them (identical arithmetic) and parses at the offsets. */
    let start_rel = resolve_array(
        &header.sections[TMNF_ROUTE_START],
        metadata.start_rel,
        1,
    );
    let checkpoints_rel = resolve_array(
        &header.sections[TMNF_ROUTE_CHECKPOINTS],
        metadata.checkpoints_rel,
        metadata.checkpoint_count,
    );
    let finish_rel = resolve_array(
        &header.sections[TMNF_ROUTE_FINISH],
        metadata.finish_rel,
        metadata.finish_count,
    );
    let reference_rel = resolve_array(
        &header.sections[TMNF_ROUTE_REFERENCE],
        metadata.reference_rel,
        metadata.centerline_count,
    );

    let mut r = Reader::at(data, start_rel as usize);
    let start = parse_route_start(&mut r);

    let mut checkpoints = Vec::with_capacity(metadata.checkpoint_count as usize);
    {
        let mut r = Reader::at(data, checkpoints_rel as usize);
        for _ in 0..metadata.checkpoint_count {
            checkpoints.push(parse_route_trigger(&mut r));
        }
    }
    let mut finish = Vec::with_capacity(metadata.finish_count as usize);
    {
        let mut r = Reader::at(data, finish_rel as usize);
        for _ in 0..metadata.finish_count {
            finish.push(parse_route_trigger(&mut r));
        }
    }
    let mut centerline = Vec::with_capacity(metadata.centerline_count as usize);
    {
        let mut r = Reader::at(data, reference_rel as usize);
        for _ in 0..metadata.centerline_count {
            centerline.push(parse_route_reference_point(&mut r));
        }
    }

    if (start.waypoint_type != TMNF_ROUTE_WAYPOINT_START
        && start.waypoint_type != TMNF_ROUTE_WAYPOINT_START_FINISH)
        || !finite_iso4(&start.transform)
        || !finite_initial_state(&start.initial_state)
        || !f32_bits_equal(&start.transform.m, &start.initial_state.rot.m)
        || !f32_bits_equal(
            &start.transform.t,
            &[start.initial_state.pos.x, start.initial_state.pos.y, start.initial_state.pos.z],
        )
        || !valid_spawn(&start.spawn)
    {
        panic!("tmnf route: invalid route start");
    }
    for i in 0..metadata.checkpoint_count as usize {
        validate_trigger(&checkpoints[i], i as u32, TMNF_ROUTE_WAYPOINT_CHECKPOINT);
    }
    for i in 0..metadata.finish_count as usize {
        if finish[i].waypoint_type == TMNF_ROUTE_WAYPOINT_FINISH {
            validate_trigger(&finish[i], i as u32, TMNF_ROUTE_WAYPOINT_FINISH);
            if start.waypoint_type != TMNF_ROUTE_WAYPOINT_START {
                panic!("tmnf route: dedicated finish has a shared start");
            }
        } else {
            validate_trigger(&finish[i], i as u32, TMNF_ROUTE_WAYPOINT_START_FINISH);
            if metadata.finish_count != 1
                || start.waypoint_type != TMNF_ROUTE_WAYPOINT_START_FINISH
                || start.block_index != finish[i].block_index
                || metadata.lap_count <= 1
            {
                panic!("tmnf route: invalid shared start/finish");
            }
        }
    }
    if centerline[0].arc_length != 0.0f32
        || centerline[0].leg_index != 0
        || !f32_bits_equal(
            &[centerline[0].position.x, centerline[0].position.y, centerline[0].position.z],
            &start.transform.t,
        )
    {
        panic!("tmnf route: invalid reference-line origin");
    }
    for i in 0..metadata.centerline_count as usize {
        let point = &centerline[i];
        if !finite_reference_point(point)
            || !(point.half_width > 0.0f32)
            || point.leg_index > metadata.checkpoint_count
            || (i != 0
                && (point.leg_index < centerline[i - 1].leg_index
                    || point.leg_index > centerline[i - 1].leg_index + 1))
            || (i != 0 && !(point.arc_length > centerline[i - 1].arc_length))
        {
            panic!("tmnf route: invalid reference line");
        }
        if i != 0 {
            let dx = point.position.x - centerline[i - 1].position.x;
            let dy = point.position.y - centerline[i - 1].position.y;
            let dz = point.position.z - centerline[i - 1].position.z;
            let length = (dx * dx + dy * dy + dz * dz).sqrt();
            let arc_delta = point.arc_length - centerline[i - 1].arc_length;
            let tolerance = 0.001f32.max(length * 0.0001f32);
            if !(length > 0.0f32) || (arc_delta - length).abs() > tolerance {
                panic!("tmnf route: reference arc length does not match geometry");
            }
        }
    }
    if centerline[metadata.centerline_count as usize - 1].leg_index != metadata.checkpoint_count {
        panic!("tmnf route: reference line does not contain every route leg");
    }

    (metadata, start, checkpoints, finish, centerline)
}

/* ---------------------------------------------------------------------------
 * Projection BVH
 * ------------------------------------------------------------------------- */

fn build_projection_bvh(
    centerline: &[TmnfRouteReferencePoint],
    bvh: &mut [TmnfRouteBvhNode],
    node_count: &mut u32,
    height: &mut u32,
    first: u32,
    count: u32,
    depth: u32,
) -> u32 {
    if count == 0 || depth >= 64 {
        panic!("tmnf route: reference line exceeds BVH depth bound");
    }
    let node_index = *node_count;
    *node_count += 1;
    {
        let node = &mut bvh[node_index as usize];
        node.minimum = [f32::INFINITY; 3];
        node.maximum = [f32::NEG_INFINITY; 3];
    }
    for segment in first..first + count {
        let a = &centerline[segment as usize].position;
        let b = &centerline[(segment + 1) as usize].position;
        let pa = [a.x, a.y, a.z];
        let pb = [b.x, b.y, b.z];
        let node = &mut bvh[node_index as usize];
        for axis in 0..3usize {
            node.minimum[axis] = node.minimum[axis].min(pa[axis].min(pb[axis]));
            node.maximum[axis] = node.maximum[axis].max(pa[axis].max(pb[axis]));
        }
    }
    if depth + 1 > *height {
        *height = depth + 1;
    }
    if count == 1 {
        let node = &mut bvh[node_index as usize];
        node.left = u32::MAX;
        node.right = u32::MAX;
        node.segment = first;
        return node_index;
    }
    let left_count = count / 2;
    bvh[node_index as usize].segment = u32::MAX;
    let left =
        build_projection_bvh(centerline, bvh, node_count, height, first, left_count, depth + 1);
    bvh[node_index as usize].left = left;
    let right = build_projection_bvh(
        centerline,
        bvh,
        node_count,
        height,
        first + left_count,
        count - left_count,
        depth + 1,
    );
    bvh[node_index as usize].right = right;
    node_index
}

fn initialize_projection_bvh(
    metadata: &TmnfRouteMetadata,
    centerline: &[TmnfRouteReferencePoint],
) -> (Vec<TmnfRouteBvhNode>, u32, u32) {
    let segment_count = metadata.centerline_count - 1;
    if segment_count > (u32::MAX - 1) / 2 {
        panic!("tmnf route: reference line is too large for its BVH");
    }
    let node_count = segment_count * 2 - 1;
    let mut bvh = vec![TmnfRouteBvhNode::default(); node_count as usize];
    let mut count = 0u32;
    let mut height = 0u32;
    if build_projection_bvh(centerline, &mut bvh, &mut count, &mut height, 0, segment_count, 0)
        != 0
        || count != node_count
    {
        panic!("tmnf route: projection BVH construction failed");
    }
    (bvh, count, height)
}

/* ---------------------------------------------------------------------------
 * Loader
 * ------------------------------------------------------------------------- */

/// `TmnfRoute_Load`: reads and fully validates a `TMNFROU1` snapshot.
/// Panics exactly where the C aborts (see the module docs for the
/// `Option<&[u8; 32]>` hash deviation).
pub fn tmnf_route_load(path: &str, expected_track_sha256: Option<&[u8; 32]>) -> TmnfRoute {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => panic!("tmnf route: cannot open snapshot"),
    };
    if data.len() < size_of::<TmnfRouteHeader>() {
        panic!("tmnf route: invalid snapshot size");
    }
    let size = data.len();

    let header = parse_route_header(&data);
    if header.magic != *b"TMNFROU1"
        || header.version != TMNF_ROUTE_VERSION
        || header.endian != 0x12345678
        || header.header_size != size_of::<TmnfRouteHeader>() as u32
        || header.section_count != TMNF_ROUTE_SECTION_COUNT as u32
        || header.file_size != size as u64
    {
        panic!("tmnf route: invalid snapshot header");
    }
    if header.exe_sha256 != TMNF_21126_EXE_SHA256 {
        panic!("tmnf route: snapshot targets a different executable");
    }
    if let Some(expected) = expected_track_sha256 {
        if header.track_sha256 != *expected {
            panic!("tmnf route: snapshot is for a different track");
        }
    }
    validate_sections(&header, size);

    let payload_digest = sha256_digest(&data[header.header_size as usize..]);
    if payload_digest != header.payload_sha256 {
        panic!("tmnf route: payload SHA-256 mismatch");
    }

    let (metadata, start, checkpoints, finish, centerline) = fix_and_validate(&data, &header);
    let (projection_bvh, projection_bvh_count, projection_bvh_height) =
        initialize_projection_bvh(&metadata, &centerline);

    TmnfRoute {
        header,
        metadata,
        start,
        checkpoints,
        finish,
        centerline,
        projection_bvh,
        projection_bvh_count,
        projection_bvh_height,
    }
}

/* ---------------------------------------------------------------------------
 * Getters (TmnfRoute_Get*)
 * ------------------------------------------------------------------------- */

impl TmnfRoute {
    /// `TmnfRoute_GetStart`.
    pub fn get_start(&self) -> &TmnfRouteStart {
        &self.start
    }

    /// `TmnfRoute_GetCheckpointCount`.
    pub fn get_checkpoint_count(&self) -> u32 {
        self.metadata.checkpoint_count
    }

    /// `TmnfRoute_GetCheckpoint` (panics on out-of-range index, as the C).
    pub fn get_checkpoint(&self, index: u32) -> &TmnfRouteTrigger {
        if index >= self.metadata.checkpoint_count {
            panic!("tmnf route: checkpoint index out of range");
        }
        &self.checkpoints[index as usize]
    }

    /// `TmnfRoute_GetFinish` — the C returns the pointer to the first
    /// finish trigger (`finish_count` entries, guaranteed >= 1 by load).
    pub fn get_finish(&self) -> &[TmnfRouteTrigger] {
        &self.finish
    }

    /// `TmnfRoute_GetReferencePointCount`.
    pub fn get_reference_point_count(&self) -> u32 {
        if self.metadata.centerline_count < 2 {
            panic!("tmnf route: route has no centerline");
        }
        self.metadata.centerline_count
    }

    /// `TmnfRoute_GetReferencePoints`.
    pub fn get_reference_points(&self) -> &[TmnfRouteReferencePoint] {
        if self.metadata.centerline_count < 2 {
            panic!("tmnf route: route has no centerline");
        }
        &self.centerline
    }

    /// `TmnfRoute_GetReferenceLength`.
    pub fn get_reference_length(&self) -> f32 {
        if self.metadata.centerline_count < 2 {
            panic!("tmnf route: route has no centerline");
        }
        self.centerline[self.metadata.centerline_count as usize - 1].arc_length
    }

    /// `TmnfRoute_Project`: projects exactly onto the nearest finite 3D
    /// polyline segment through the immutable BVH. `lateral_offset` is the
    /// non-negative Euclidean distance, `half_width` is linearly
    /// interpolated, `segment_index` is the checkpoint leg,
    /// `centerline_segment_index` is the physical segment, and ties choose
    /// the first physical segment.
    ///
    /// (The C's NULL-BVH fallback to `project_in_memory_route` — the GPU
    /// device-copy lane — is unreachable here; see the module docs.)
    pub fn project(&self, world_position: &GmVec3) -> TmnfRouteProjection {
        if !finite_vec3(world_position) {
            panic!("tmnf route: cannot project a non-finite position");
        }
        let mut best_distance_sq = f32::INFINITY;
        let mut result = TmnfRouteProjection {
            arc_length: 0.0,
            lateral_offset: 0.0,
            half_width: 0.0,
            segment_index: u32::MAX,
            centerline_segment_index: u32::MAX,
            segments_tested: 0,
        };
        let mut stack = [0u32; 64];
        let mut stack_size = 1usize;
        stack[0] = 0;
        let position = [world_position.x, world_position.y, world_position.z];
        while stack_size != 0 {
            stack_size -= 1;
            let node = &self.projection_bvh[stack[stack_size] as usize];
            let mut node_distance_sq = 0.0f32;
            for axis in 0..3usize {
                let mut offset = 0.0f32;
                if position[axis] < node.minimum[axis] {
                    offset = node.minimum[axis] - position[axis];
                } else if position[axis] > node.maximum[axis] {
                    offset = position[axis] - node.maximum[axis];
                }
                node_distance_sq += offset * offset;
            }
            if node_distance_sq > best_distance_sq {
                continue;
            }
            if node.segment == u32::MAX {
                let left = &self.projection_bvh[node.left as usize];
                let right = &self.projection_bvh[node.right as usize];
                let mut left_distance_sq = 0.0f32;
                let mut right_distance_sq = 0.0f32;
                for axis in 0..3usize {
                    let mut left_offset = 0.0f32;
                    let mut right_offset = 0.0f32;
                    if position[axis] < left.minimum[axis] {
                        left_offset = left.minimum[axis] - position[axis];
                    } else if position[axis] > left.maximum[axis] {
                        left_offset = position[axis] - left.maximum[axis];
                    }
                    if position[axis] < right.minimum[axis] {
                        right_offset = right.minimum[axis] - position[axis];
                    } else if position[axis] > right.maximum[axis] {
                        right_offset = position[axis] - right.maximum[axis];
                    }
                    left_distance_sq += left_offset * left_offset;
                    right_distance_sq += right_offset * right_offset;
                }
                if stack_size + 2 > 64 {
                    panic!("tmnf route: projection BVH traversal stack overflow");
                }
                if left_distance_sq <= right_distance_sq {
                    stack[stack_size] = node.right;
                    stack_size += 1;
                    stack[stack_size] = node.left;
                    stack_size += 1;
                } else {
                    stack[stack_size] = node.left;
                    stack_size += 1;
                    stack[stack_size] = node.right;
                    stack_size += 1;
                }
                continue;
            }
            let i = node.segment;
            let a = &self.centerline[i as usize];
            let b = &self.centerline[(i + 1) as usize];
            let dx = b.position.x - a.position.x;
            let dy = b.position.y - a.position.y;
            let dz = b.position.z - a.position.z;
            let px = world_position.x - a.position.x;
            let py = world_position.y - a.position.y;
            let pz = world_position.z - a.position.z;
            let length_sq = dx * dx + dy * dy + dz * dz;
            let mut t = (px * dx + py * dy + pz * dz) / length_sq;
            if t < 0.0f32 {
                t = 0.0f32;
            } else if t > 1.0f32 {
                t = 1.0f32;
            }
            let ox = px - t * dx;
            let oy = py - t * dy;
            let oz = pz - t * dz;
            let distance_sq = ox * ox + oy * oy + oz * oz;
            result.segments_tested += 1;
            if distance_sq < best_distance_sq
                || (distance_sq == best_distance_sq && i < result.centerline_segment_index)
            {
                best_distance_sq = distance_sq;
                result.arc_length = a.arc_length + t * (b.arc_length - a.arc_length);
                result.half_width = a.half_width + t * (b.half_width - a.half_width);
                result.segment_index = a.leg_index;
                result.centerline_segment_index = i;
            }
        }
        if result.centerline_segment_index == u32::MAX {
            panic!("tmnf route: projection BVH returned no segment");
        }
        result.lateral_offset = best_distance_sq.sqrt();
        result
    }
}

/* ---------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    const ROUTE_PATH: &str = "/home/z/my-project/tmnf-physics/oracle/routes/A01-Race.tmnfroute";

    /// The Challenge.Gbx hash (oracle/tracks/manifest.txt), embedded in the
    /// route header as `track_sha256`.
    const TRACK_SHA256: [u8; 32] = [
        0xf0, 0xa8, 0x70, 0x80, 0x9b, 0xe9, 0x9d, 0xa2,
        0xcb, 0x36, 0xad, 0x5d, 0xf4, 0x3a, 0x2c, 0xf6,
        0x3d, 0x8f, 0x74, 0xfe, 0x4a, 0xc3, 0x47, 0x0e,
        0xca, 0xc6, 0x8b, 0x9e, 0x97, 0x62, 0x5d, 0xc3,
    ];

    #[test]
    fn load_a01_race_route() {
        let route = tmnf_route_load(ROUTE_PATH, Some(&TRACK_SHA256));

        // Header: layout + section table (from the fixture bytes).
        assert_eq!(route.header.magic, *b"TMNFROU1");
        assert_eq!(route.header.version, 3);
        assert_eq!(route.header.endian, 0x12345678);
        assert_eq!(route.header.header_size, 0xE0);
        assert_eq!(route.header.section_count, 5);
        assert_eq!(route.header.file_size, 27592);
        assert_eq!(route.header.exe_sha256, TMNF_21126_EXE_SHA256);
        assert_eq!(route.header.track_sha256, TRACK_SHA256);
        let sections: [(u64, u32, u32); 5] = [
            (0xE0, 1, 0x40),
            (0x120, 1, 0x114),
            (0x238, 2, 0x90),
            (0x358, 1, 0x90),
            (0x3E8, 1108, 0x18),
        ];
        for (i, &(offset, count, stride)) in sections.iter().enumerate() {
            assert_eq!(route.header.sections[i].offset, offset, "section {i} offset");
            assert_eq!(route.header.sections[i].count, count, "section {i} count");
            assert_eq!(route.header.sections[i].stride, stride, "section {i} stride");
        }

        // Metadata (the oracle log: checkpoints=2 finish=1 laps=1 total=3
        // limit=3 reference=4).
        assert_eq!(route.metadata.lap_count, 1);
        assert_eq!(route.metadata.checkpoint_count, 2);
        assert_eq!(route.metadata.finish_count, 1);
        assert_eq!(route.metadata.reference_count, 4);
        assert_eq!(route.metadata.total_race_checkpoints, 3);
        assert_eq!(route.metadata.race_checkpoint_limit, 3);
        assert_eq!(route.metadata.flags, 0);
        assert_eq!(route.metadata.centerline_count, 1108);
        // Relative offsets (kept on-disk; the C fixes them to base+offset).
        assert_eq!(route.metadata.start_rel, 0x120);
        assert_eq!(route.metadata.checkpoints_rel, 0x238);
        assert_eq!(route.metadata.finish_rel, 0x358);
        assert_eq!(route.metadata.reference_rel, 0x3E8);

        // Counts the task asks to report.
        assert_eq!(route.get_checkpoint_count(), 2);
        assert!(route.checkpoints.len() > 0);
        assert!(route.finish.len() > 0);
        assert_eq!(route.get_reference_point_count(), 1108);
        assert!(route.get_reference_points().len() > 0);

        // Start record.
        let start = route.get_start();
        assert_eq!(start.block_index, 43);
        assert_eq!(start.waypoint_type, TMNF_ROUTE_WAYPOINT_START);
        assert_eq!(start.transform.m, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0]);
        assert_eq!(start.transform.t, [171.1999969482422, 90.20999908447266, 688.0]);
        // Game quaternion, scalar-first in field .x (see gm.rs).
        assert_eq!(
            start.initial_state.quat,
            GmQuat {
                x: 0.70710677,
                y: 0.0,
                z: 0.70710677,
                w: 0.0
            }
        );
        assert_eq!(start.initial_state.rot.m, start.transform.m);
        assert_eq!(start.initial_state.pos.x, 171.1999969482422);
        assert_eq!(start.initial_state.pos.y, 90.20999908447266);
        assert_eq!(start.initial_state.pos.z, 688.0);
        assert_eq!(
            start.initial_state.inv_inertia_world.m,
            [1.2000000476837158, 0.0, 0.0, 0.0, 0.48000001907348633, 0.0, 0.0, 0.0, 0.48000001907348633]
        );
        assert_eq!(start.spawn, start.transform);

        // Checkpoints (from the oracle log).
        let cp0 = route.get_checkpoint(0);
        assert_eq!(cp0.race_index, 0);
        assert_eq!(cp0.block_index, 0);
        assert_eq!(cp0.waypoint_type, TMNF_ROUTE_WAYPOINT_CHECKPOINT);
        assert_eq!(cp0.tree_flags, 0x1e886);
        assert_eq!(cp0.no_respawn, 0);
        assert_eq!(cp0.box_aligned.half_extent.x, 14.179214477539062);
        assert_eq!(cp0.box_aligned.half_extent.y, 2.766376495361328);
        assert_eq!(cp0.box_aligned.half_extent.z, 0.2200002670288086);
        assert_eq!(cp0.transform.t, [992.0, 16.0, 320.0]);
        assert_eq!(cp0.spawn.t, [976.0, 24.0, 304.0]);
        let cp1 = route.get_checkpoint(1);
        assert_eq!(cp1.race_index, 1);
        assert_eq!(cp1.block_index, 134);
        assert_eq!(cp1.no_respawn, 1);
        assert_eq!(cp1.spawn.t, [496.0, 88.0, 176.0]);

        // Finish.
        let finish = route.get_finish();
        assert_eq!(finish.len(), 1);
        assert_eq!(finish[0].race_index, 0);
        assert_eq!(finish[0].block_index, 105);
        assert_eq!(finish[0].waypoint_type, TMNF_ROUTE_WAYPOINT_FINISH);
        assert_eq!(finish[0].no_respawn, 0);
        assert_eq!(finish[0].box_aligned.center.x, 16.5);
        assert_eq!(finish[0].box_aligned.center.y, 4.407114028930664);
        assert_eq!(finish[0].box_aligned.center.z, 11.999350547790527);
        assert_eq!(finish[0].box_aligned.half_extent.x, 13.5);
        assert_eq!(finish[0].box_aligned.half_extent.y, 3.407114028930664);
        assert_eq!(finish[0].box_aligned.half_extent.z, 0.20654010772705078);
        assert_eq!(finish[0].transform.t, [32.0, 112.0, 160.0]);
        assert_eq!(finish[0].spawn.t, [20.799999237060547, 113.69999694824219, 176.0]);

        // Centerline: 1108 dense points, first == start translation.
        let points = route.get_reference_points();
        assert_eq!(points[0].position.x, 171.1999969482422);
        assert_eq!(points[0].position.y, 90.20999908447266);
        assert_eq!(points[0].position.z, 688.0);
        assert_eq!(points[0].arc_length, 0.0);
        assert_eq!(points[0].half_width, 15.0);
        assert_eq!(points[0].leg_index, 0);
        assert_eq!(points[1107].arc_length, 2212.947265625);
        assert_eq!(points[1107].half_width, 15.0);
        assert_eq!(points[1107].leg_index, 2);
        assert_eq!(route.get_reference_length(), 2212.947265625);

        // Projection BVH: 1107 segments -> 2213 nodes, deepest leaf at
        // depth 11 (1107 -> 554 -> 277 -> ... -> 2 -> 1), so height 12.
        assert_eq!(route.projection_bvh_count, 2213);
        assert_eq!(route.projection_bvh.len(), 2213);
        assert_eq!(route.projection_bvh_height, 12);
        assert_eq!(route.projection_bvh[0].segment, u32::MAX); // root is internal
        let leaves = route.projection_bvh.iter().filter(|n| n.segment != u32::MAX).count();
        assert_eq!(leaves, 1107);
        // Leaf i carries centerline segment i.
        let mut segment_of_leaf = 0u32;
        for node in &route.projection_bvh {
            if node.segment != u32::MAX {
                assert_eq!(node.segment, segment_of_leaf);
                segment_of_leaf += 1;
            }
        }
        assert_eq!(segment_of_leaf, 1107);
    }

    #[test]
    fn load_a01_race_route_without_expected_sha() {
        let route = tmnf_route_load(ROUTE_PATH, None);
        assert_eq!(route.metadata.checkpoint_count, 2);
        assert_eq!(route.metadata.centerline_count, 1108);
    }

    #[test]
    fn route_projection_over_centerline() {
        let route = tmnf_route_load(ROUTE_PATH, None);

        // The start point projects to segment 0 at arc length 0.
        let start = GmVec3 {
            x: 171.1999969482422,
            y: 90.20999908447266,
            z: 688.0,
        };
        let p = route.project(&start);
        assert_eq!(p.centerline_segment_index, 0);
        assert_eq!(p.segment_index, 0);
        assert_eq!(p.arc_length, 0.0);
        assert_eq!(p.lateral_offset, 0.0);
        assert_eq!(p.half_width, 15.0);
        assert!(p.segments_tested > 0);

        // The final centerline point projects to the last segment.
        let last = route.centerline[1107].position;
        let p = route.project(&last);
        assert_eq!(p.centerline_segment_index, 1106);
        assert_eq!(p.segment_index, 2);
        assert_eq!(p.arc_length, 2212.947265625);
        assert_eq!(p.lateral_offset, 0.0);

        // Mid-route points project near their own arc length.
        for &k in &[200usize, 500, 900] {
            let p = route.project(&route.centerline[k].position);
            assert!(
                (p.arc_length - route.centerline[k].arc_length).abs() < 2.5,
                "k={k}: projected {} vs point {}",
                p.arc_length,
                route.centerline[k].arc_length
            );
            assert!(p.lateral_offset < 1.0);
        }

        // Deterministic: same input, same output.
        let a = route.project(&route.centerline[500].position);
        let b = route.project(&route.centerline[500].position);
        assert_eq!(a, b);

        // A point far above the track still resolves (nearest segment).
        let high = GmVec3 {
            x: route.centerline[500].position.x,
            y: route.centerline[500].position.y + 1000.0,
            z: route.centerline[500].position.z,
        };
        let p = route.project(&high);
        assert!(p.centerline_segment_index != u32::MAX);
        assert!(p.lateral_offset > 999.0);
    }

    #[test]
    #[should_panic(expected = "tmnf route: snapshot is for a different track")]
    fn load_rejects_wrong_track_sha() {
        let wrong = [0u8; 32];
        let _ = tmnf_route_load(ROUTE_PATH, Some(&wrong));
    }

    #[test]
    #[should_panic(expected = "tmnf route: cannot open snapshot")]
    fn load_rejects_missing_file() {
        let _ = tmnf_route_load("/nonexistent/A01.tmnfroute", None);
    }

    #[test]
    #[should_panic(expected = "tmnf route: payload SHA-256 mismatch")]
    fn load_rejects_corrupt_payload() {
        let data = fs::read(ROUTE_PATH).unwrap();
        let mut corrupt = data.clone();
        // Corrupt one centerline coordinate (reference section at 0x3E8).
        let offset = 0x3E8usize + 0x200 * 24 + 4;
        corrupt[offset] ^= 0xff;
        let path = std::env::temp_dir().join("tmnf_corrupt_payload.tmnfroute");
        fs::write(&path, &corrupt).unwrap();
        let path = path.to_str().unwrap();
        let _ = tmnf_route_load(path, None);
    }

    #[test]
    #[should_panic(expected = "tmnf route: invalid snapshot header")]
    fn load_rejects_bad_magic() {
        let data = fs::read(ROUTE_PATH).unwrap();
        let mut corrupt = data.clone();
        corrupt[3] = b'X';
        let path = std::env::temp_dir().join("tmnf_bad_magic.tmnfroute");
        fs::write(&path, &corrupt).unwrap();
        let path = path.to_str().unwrap();
        let _ = tmnf_route_load(path, None);
    }

    #[test]
    #[should_panic(expected = "tmnf route: checkpoint index out of range")]
    fn checkpoint_out_of_range_panics() {
        let route = tmnf_route_load(ROUTE_PATH, None);
        let _ = route.get_checkpoint(2);
    }

    #[test]
    #[should_panic(expected = "tmnf route: cannot project a non-finite position")]
    fn project_rejects_non_finite_position() {
        let route = tmnf_route_load(ROUTE_PATH, None);
        let _ = route.project(&GmVec3 {
            x: f32::NAN,
            y: 0.0,
            z: 0.0,
        });
    }
}
