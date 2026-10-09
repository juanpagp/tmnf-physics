//! `TMNFTRK1` static-track snapshot loader (`src/track.h` / `src/track.c`),
//! transliterated.
//!
//! The C loader mmaps the snapshot, fixes the on-disk relative offsets up to
//! native pointers in a private mapping, then mprotects it read-only. This
//! port reads the whole file into a `Vec<u8>` and builds **owned** structures
//! with indices where the C had pointers:
//!
//! * `TmnfTrackStaticEntry::surface_rel` (offset into the file) →
//!   [`HmsStaticCollisionEntry::surface`] = index into `TmnfTrack::surfaces`;
//! * `TmnfTrackSurface::mesh_rel` → the mesh is **cloned** into the surface's
//!   `CPlugSurface::geom` (`GmSurfGeom::Mesh`), because `CPlugSurface` owns
//!   its geometry in this port. The canonical meshes are also kept in
//!   `TmnfTrack::meshes` (field-for-field parity with the C struct, and what
//!   the static grid is built over). A01 clones ~17.7 MB of mesh data for
//!   1499 surfaces sharing 96 meshes.
//! * The shared `GmMap2` water map → an owned `Vec<u8>` of cells.
//!
//! Every validation, its order and its panic message mirror the C `fail()`
//! (which prints `tmnf track: <message>` and aborts). `TmnfTrack_Unload` is
//! plain `drop` here (the C's munmap/free tail has no failure mode worth a
//! panic), and the trailing `mprotect(PROT_READ)` is the C making its private
//! mapping immutable — owned data needs no equivalent.
//!
//! `TmnfTrack_BindStaticGroup(track, group)` (fills `group->is_static`,
//! `static_entry_count`, `static_entries`, `static_grid`) becomes
//! [`TmnfTrack::static_group`], the view a future `world.rs` binds into a
//! collision-manager zone group.
//!
//! # Static-grid deviation (documented, behavior-identical)
//!
//! The C interns the region-restricted mesh lists of **all** meshes into ONE
//! shared pool (`builder.mesh_pool`) purely to save memory; every
//! `TmnfMeshGrid` then points at that pool. `TmnfMeshGrid` in
//! [`crate::collision`] *owns* its pool, so this port gives each mesh its own
//! pool + intern table: the per-mesh lists, their contents and the cell →
//! list selection are byte-identical (pool slot numbers are internal), only
//! the cross-mesh storage sharing is dropped. `TmnfStaticGrid::mesh_pool`
//! (unused by the collision kernels, which read `TmnfMeshGrid::pool`) is left
//! empty for that reason. The track-level *node* lists (`grid.nodes`) are
//! interned globally exactly like the C.

use crate::collision::{
    GmBoxAligned, GmSurf, GmSurfGeom, GmSurfMesh, GmSurfMeshFace, GmSurfMeshNode, GM_SURF_MESH,
    HmsStaticCollisionEntry, TmnfMeshGrid, TmnfMeshGridLevel, TmnfMeshPoolSlot,
    TMNF_CELL_NODE_EMPTY, TMNF_CELL_NODE_INTERNAL, TMNF_MESH_GRID_LEVELS, TmnfSphereFaceEdges,
    TmnfStaticCellNode, TmnfStaticGrid,
};
use crate::gm::{GmIso4, GmVec3};
use crate::response::{AbsorbSink, CHmsResponseBody, CPlugSurfaceMaterialData};
use std::fs;
use std::mem::size_of;

/* ---------------------------------------------------------------------------
 * Constants (track.h)
 * ------------------------------------------------------------------------- */

pub const TMNF_TRACK_VERSION: u32 = 3;
pub const TMNF_TRACK_SECTION_COUNT: usize = 12;
pub const TMNF_TRACK_MATERIAL_COUNT: u32 = 31;

pub const TMNF_TRACK_ENTRIES: usize = 0;
pub const TMNF_TRACK_SURFACES: usize = 1;
pub const TMNF_TRACK_MESHES: usize = 2;
pub const TMNF_TRACK_VERTICES: usize = 3;
pub const TMNF_TRACK_FACES: usize = 4;
pub const TMNF_TRACK_NODES: usize = 5;
pub const TMNF_TRACK_MATERIAL_IDS: usize = 6;
pub const TMNF_TRACK_MATERIAL_DATA: usize = 7;
pub const TMNF_TRACK_COLLISION_PAIRS: usize = 8;
/// CHmsCorpus+0x18 location of static corpus id k at index k-1.
pub const TMNF_TRACK_CORPUS_ISOS: usize = 9;
/// CHmsZone+0x154 water map header and its width*height cells.
pub const TMNF_TRACK_WATER: usize = 10;
pub const TMNF_TRACK_WATER_CELLS: usize = 11;

/// GmMap2<unsigned char> water material (`TMNF_WATER_MATERIAL`).
pub const TMNF_WATER_MATERIAL: u32 = 13;

/// SHA-256 of the canonical TMNF 2.11.26 executable.
pub(crate) const TMNF_21126_EXE_SHA256: [u8; 32] = [
    0x38, 0x47, 0xcf, 0x9f, 0x20, 0xbf, 0xc6, 0x39,
    0x14, 0x45, 0x00, 0x60, 0xed, 0x52, 0x8c, 0x12,
    0x10, 0x4f, 0x74, 0x3d, 0x96, 0xad, 0x23, 0xd6,
    0xe7, 0x6a, 0xbd, 0x17, 0x8d, 0xe8, 0xc8, 0x4f,
];

/* ---------------------------------------------------------------------------
 * Byte-contract structs (on-disk layouts, parsed field-by-field)
 * ------------------------------------------------------------------------- */

/// 0x10 bytes. `TmnfTrackSection`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfTrackSection {
    pub offset: u64,
    pub count: u32,
    pub stride: u32,
}
const _: () = assert!(size_of::<TmnfTrackSection>() == 0x10);

/// 0x150 bytes. `TmnfTrackHeader`.
#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TmnfTrackHeader {
    pub magic: [u8; 8], /* "TMNFTRK1" */
    pub version: u32,
    pub endian: u32, /* 0x12345678 */
    pub header_size: u32, /* 0x150 */
    pub section_count: u32, /* 12 */
    pub file_size: u64,
    pub exe_sha256: [u8; 32],
    pub track_sha256: [u8; 32],
    pub sections: [TmnfTrackSection; TMNF_TRACK_SECTION_COUNT],
    pub payload_sha256: [u8; 32],
    pub reserved: [u8; 16],
}
const _: () = assert!(size_of::<TmnfTrackHeader>() == 0x150);

/// 0x60 bytes. `TmnfTrackStaticEntry` — on disk, `surface_rel` is an offset
/// from the file base; after the C loader's fixup it is the native
/// `CPlugSurface*` of `HmsStaticCollisionEntry`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfTrackStaticEntry {
    pub skip_count: u32,
    pub box_aligned: GmBoxAligned,
    pub iso: GmIso4,
    pub tree_flags: u32,
    pub surface_rel: u64,
    pub tree_id: u32,
    pub corpus_id: u32,
}
const _: () = assert!(size_of::<TmnfTrackStaticEntry>() == 0x60);

/// 0x18 bytes. `TmnfTrackSurface`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfTrackSurface {
    pub mesh_rel: u64,
    pub material_ids_rel: u64,
    pub material_count: u32,
    pub reserved: u32,
}
const _: () = assert!(size_of::<TmnfTrackSurface>() == 0x18);

/// 0x38 bytes. `TmnfTrackMesh`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfTrackMesh {
    pub base: GmSurf,
    pub vertex_count: u32,
    pub pad0c: u32,
    pub vertices_rel: u64,
    pub face_count: u32,
    pub pad1c: u32,
    pub faces_rel: u64,
    pub node_count: u32,
    pub pad2c: u32,
    pub nodes_rel: u64,
}
const _: () = assert!(size_of::<TmnfTrackMesh>() == 0x38);

/// 0x08 bytes. `TmnfTrackMaterialData` (≡ `CPlugSurfaceMaterialData`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfTrackMaterialData {
    pub friction: f32,
    pub restitution: f32,
}
const _: () = assert!(size_of::<TmnfTrackMaterialData>() == 0x08);

/// 0x14 bytes. `TmnfTrackCollisionPair`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfTrackCollisionPair {
    pub raw: [u32; 5],
}
const _: () = assert!(size_of::<TmnfTrackCollisionPair>() == 0x14);

// The C additionally _Static_asserts that the snapshot records are
// layout-identical to the *native* structs they overlay after pointer fixup
// (TmnfTrackStaticEntry == HmsStaticCollisionEntry incl. the surface offset,
// TmnfTrackSurface == CPlugSurface, TmnfTrackMesh == GmSurfMesh). Those four
// asserts are the mmap fixup's safety net and have no Rust counterpart: this
// port parses field-by-field into OWNED types whose "native" forms
// (HmsStaticCollisionEntry::surface: Option<usize>, CPlugSurface::geom:
// GmSurfGeom, GmSurfMesh::vertices: Vec<GmVec3>) deliberately redesign the
// pointer members, so their sizes are intentionally NOT equal. The on-disk
// size asserts above carry the real byte contract.

/// 0x28 bytes. `TmnfTrackWaterHeader`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfTrackWaterHeader {
    pub cell_x: f32,   /* map +0x00 */
    pub cell_z: f32,   /* map +0x04 */
    pub origin_x: f32, /* map +0x08 */
    pub origin_z: f32, /* map +0x0c */
    pub width: u32,    /* map +0x10 */
    pub height: u32,   /* map +0x14 */
    pub default_cell: u32, /* map +0x18, low byte */
    pub level: f32,    /* owner +0x178 */
    pub floor: f32,    /* owner +0x17c */
    pub reserved: u32,
}
const _: () = assert!(size_of::<TmnfTrackWaterHeader>() == 0x28);

/* ---------------------------------------------------------------------------
 * Owned runtime types
 * ------------------------------------------------------------------------- */

/// `TmnfTrackWater`: the CHmsZone+0x154 water map as the game holds it.
#[derive(Clone, Debug, Default)]
pub struct TmnfTrackWater {
    pub cell_x: f32,
    pub cell_z: f32,
    pub origin_x: f32,
    pub origin_z: f32,
    pub width: u32,
    pub height: u32,
    pub default_cell: u8,
    /// width * height flags, from the snapshot (the C pointed into the mmap).
    pub cells: Vec<u8>,
    /// Number of set cells.
    pub cell_count: u32,
    pub level: f32,
    pub floor: f32,
}

/// `TmnfTrack`: the loaded static collision snapshot. Field-for-field mirror
/// of the C struct; pointer members became owned data (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct TmnfTrack {
    pub header: TmnfTrackHeader,
    pub entries: Vec<HmsStaticCollisionEntry>,
    pub entry_count: u32,
    pub surfaces: Vec<CPlugSurfaceOwned>,
    pub surface_count: u32,
    pub meshes: Vec<GmSurfMesh>,
    pub mesh_count: u32,
    pub materials: Vec<CPlugSurfaceMaterialData>,
    pub collision_pairs: Vec<TmnfTrackCollisionPair>,
    pub collision_pair_count: u32,
    /// Indexed by `corpus_ref - 1`.
    pub corpus_isos: Vec<GmIso4>,
    pub corpus_count: u32,
    pub water: TmnfTrackWater,
    /// Region-restricted static tree copies per grid cell.
    pub grid: TmnfStaticGrid,
    pub grid_node_count: usize,
    pub grid_mesh_slot_count: usize,
    /// Response bodies of the static corpora, indexed by `corpus_ref`;
    /// `static_response_present` marks the referenced ones.
    pub static_response_bodies: Vec<CHmsResponseBody>,
    pub static_response_present: Vec<u8>,
    pub static_response_count: u32,
}

/// Alias documenting that a track surface is an owned `CPlugSurface`.
pub type CPlugSurfaceOwned = crate::collision::CPlugSurface;

/// What `TmnfTrack_BindStaticGroup` fills into a zone group in the C:
/// the group is marked static and handed the flattened static entries plus
/// the optional static grid. `world.rs` consumes this when binding a track
/// into a collision manager zone.
#[derive(Clone, Copy)]
pub struct StaticGroupData<'a> {
    /// C: `group->is_static = 1` (always set here).
    pub is_static: bool,
    /// C: `group->static_entry_count`.
    pub static_entry_count: u32,
    /// C: `group->static_entries` — indexed by entry, `surface` is an index
    /// into `TmnfTrack::surfaces`.
    pub static_entries: &'a [HmsStaticCollisionEntry],
    /// C: `group->static_grid` — `None` when the track has no bounded leaves
    /// (`grid.cell_count == 0`, the C passed NULL and the zone scans
    /// `static_entries` instead).
    pub static_grid: Option<&'a TmnfStaticGrid>,
}

impl TmnfTrack {
    /// `TmnfTrack_BindStaticGroup`: the static group view of this track.
    pub fn static_group(&self) -> StaticGroupData<'_> {
        StaticGroupData {
            is_static: true,
            static_entry_count: self.entry_count,
            static_entries: &self.entries,
            static_grid: if self.grid.cell_count != 0 {
                Some(&self.grid)
            } else {
                None
            },
        }
    }
}

/* ---------------------------------------------------------------------------
 * Reader + SHA-256 (shared with route.rs)
 * ------------------------------------------------------------------------- */

/// Little-endian field reader over the snapshot bytes (offset-based stand-in
/// for the C's struct overlays on the mmap).
#[derive(Clone, Copy)]
pub(crate) struct Reader<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub(crate) fn at(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos }
    }

    pub(crate) fn u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    pub(crate) fn u16(&mut self) -> u16 {
        let mut b = [0u8; 2];
        b.copy_from_slice(&self.data[self.pos..self.pos + 2]);
        self.pos += 2;
        u16::from_le_bytes(b)
    }

    pub(crate) fn u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.data[self.pos..self.pos + 4]);
        self.pos += 4;
        u32::from_le_bytes(b)
    }

    pub(crate) fn u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        u64::from_le_bytes(b)
    }

    pub(crate) fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }

    pub(crate) fn gm_vec3(&mut self) -> GmVec3 {
        GmVec3 {
            x: self.f32(),
            y: self.f32(),
            z: self.f32(),
        }
    }

    pub(crate) fn gm_iso4(&mut self) -> GmIso4 {
        let mut m = [0f32; 9];
        for v in &mut m {
            *v = self.f32();
        }
        let t = [self.f32(), self.f32(), self.f32()];
        GmIso4 { m, t }
    }

    pub(crate) fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.data[self.pos..self.pos + n];
        self.pos += n;
        v
    }
}

/// Compact SHA-256, transliterated from track.c (public-domain style).
/// Used to authenticate the immutable payload before it is decoded.
pub(crate) struct Sha256 {
    state: [u32; 8],
    byte_count: u64,
    block: [u8; 64],
    block_len: u32,
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn rotr32(value: u32, bits: u32) -> u32 {
    // C: (value >> bits) | (value << (32 - bits)). `wrapping_shl` reproduces
    // the x86 masked shift, so the expression is total; it is never called
    // with bits == 0 by the rounds below in either language.
    (value >> bits) | value.wrapping_shl(32 - bits)
}

fn load_be32(p: &[u8]) -> u32 {
    (p[0] as u32) << 24 | (p[1] as u32) << 16 | (p[2] as u32) << 8 | p[3] as u32
}

fn store_be32(p: &mut [u8], value: u32) {
    p[0] = (value >> 24) as u8;
    p[1] = (value >> 16) as u8;
    p[2] = (value >> 8) as u8;
    p[3] = value as u8;
}

fn sha256_transform(ctx: &mut Sha256, block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = load_be32(&block[i * 4..i * 4 + 4]);
    }
    for i in 16..64 {
        let s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        let s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = ctx.state[0];
    let mut b = ctx.state[1];
    let mut c = ctx.state[2];
    let mut d = ctx.state[3];
    let mut e = ctx.state[4];
    let mut f = ctx.state[5];
    let mut g = ctx.state[6];
    let mut h = ctx.state[7];
    for i in 0..64 {
        let s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        let choose = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(choose)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        let majority = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(majority);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    ctx.state[0] = ctx.state[0].wrapping_add(a);
    ctx.state[1] = ctx.state[1].wrapping_add(b);
    ctx.state[2] = ctx.state[2].wrapping_add(c);
    ctx.state[3] = ctx.state[3].wrapping_add(d);
    ctx.state[4] = ctx.state[4].wrapping_add(e);
    ctx.state[5] = ctx.state[5].wrapping_add(f);
    ctx.state[6] = ctx.state[6].wrapping_add(g);
    ctx.state[7] = ctx.state[7].wrapping_add(h);
}

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            byte_count: 0,
            block: [0u8; 64],
            block_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.byte_count += data.len() as u64;
        let mut bytes = data;
        while !bytes.is_empty() {
            let space = 64 - self.block_len as usize;
            let take = if bytes.len() < space {
                bytes.len()
            } else {
                space
            };
            self.block[self.block_len as usize..self.block_len as usize + take]
                .copy_from_slice(&bytes[..take]);
            self.block_len += take as u32;
            bytes = &bytes[take..];
            if self.block_len == 64 {
                let block = self.block;
                sha256_transform(self, &block);
                self.block_len = 0;
            }
        }
    }

    fn final_digest(&mut self, digest: &mut [u8; 32]) {
        let bit_count = self.byte_count.wrapping_mul(8);
        self.block[self.block_len as usize] = 0x80;
        self.block_len += 1;
        if self.block_len > 56 {
            for i in self.block_len as usize..64 {
                self.block[i] = 0;
            }
            let block = self.block;
            sha256_transform(self, &block);
            self.block_len = 0;
        }
        for i in self.block_len as usize..56 {
            self.block[i] = 0;
        }
        for i in 0..8u32 {
            self.block[(63 - i) as usize] = (bit_count >> (i * 8)) as u8;
        }
        let block = self.block;
        sha256_transform(self, &block);
        for i in 0..8 {
            store_be32(&mut digest[i * 4..i * 4 + 4], self.state[i]);
        }
    }
}

/// SHA-256 of a byte slice (the loader's one-shot use of `Sha256`).
pub(crate) fn sha256_digest(data: &[u8]) -> [u8; 32] {
    let mut ctx = Sha256::new();
    ctx.update(data);
    let mut digest = [0u8; 32];
    ctx.final_digest(&mut digest);
    digest
}

/* ---------------------------------------------------------------------------
 * Section validation / pointer resolution
 * ------------------------------------------------------------------------- */

fn section_bytes(section: &TmnfTrackSection) -> u64 {
    let bytes = (section.count as u64) * (section.stride as u64);
    if section.stride != 0 && bytes / section.stride as u64 != section.count as u64 {
        panic!("tmnf track: section size overflow");
    }
    bytes
}

fn validate_sections(header: &TmnfTrackHeader, mapping_size: usize) {
    let expected_stride: [u32; TMNF_TRACK_SECTION_COUNT] = [
        size_of::<TmnfTrackStaticEntry>() as u32,
        size_of::<TmnfTrackSurface>() as u32,
        size_of::<TmnfTrackMesh>() as u32,
        size_of::<GmVec3>() as u32,
        size_of::<GmSurfMeshFace>() as u32,
        size_of::<GmSurfMeshNode>() as u32,
        1,
        size_of::<TmnfTrackMaterialData>() as u32,
        size_of::<TmnfTrackCollisionPair>() as u32,
        size_of::<GmIso4>() as u32,
        size_of::<TmnfTrackWaterHeader>() as u32,
        1,
    ];

    for i in 0..TMNF_TRACK_SECTION_COUNT {
        let section = &header.sections[i];
        if section.stride != expected_stride[i]
            || section.count == 0
            || (section.offset & 7) != 0
            || section.offset < size_of::<TmnfTrackHeader>() as u64
        {
            panic!("tmnf track: invalid section descriptor");
        }
        let bytes = section_bytes(section);
        if section.offset > mapping_size as u64
            || bytes > mapping_size as u64 - section.offset
        {
            panic!("tmnf track: section outside file");
        }
    }
    for i in 0..TMNF_TRACK_SECTION_COUNT {
        let a0 = header.sections[i].offset;
        let a1 = a0 + section_bytes(&header.sections[i]);
        for j in i + 1..TMNF_TRACK_SECTION_COUNT {
            let b0 = header.sections[j].offset;
            let b1 = b0 + section_bytes(&header.sections[j]);
            if a0 < b1 && b0 < a1 {
                panic!("tmnf track: overlapping sections");
            }
        }
    }
}

/// `resolve_array`: validates a relative offset against a section and returns
/// the byte offset from the file base (the C returned `base + relative`).
fn resolve_array(section: &TmnfTrackSection, relative: u64, count: u32) -> u64 {
    let bytes = (count as u64) * (section.stride as u64);
    let section_end = section.offset + section_bytes(section);
    if relative < section.offset
        || relative > section_end
        || bytes > section_end - relative
        || (relative - section.offset) % section.stride as u64 != 0
    {
        panic!("tmnf track: relative pointer outside target section");
    }
    relative
}

/* ---------------------------------------------------------------------------
 * Parsers for the on-disk records
 * ------------------------------------------------------------------------- */

fn parse_track_header(data: &[u8]) -> TmnfTrackHeader {
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
    let mut sections = [TmnfTrackSection::default(); TMNF_TRACK_SECTION_COUNT];
    for s in &mut sections {
        s.offset = r.u64();
        s.count = r.u32();
        s.stride = r.u32();
    }
    let mut payload_sha256 = [0u8; 32];
    payload_sha256.copy_from_slice(r.bytes(32));
    let mut reserved = [0u8; 16];
    reserved.copy_from_slice(r.bytes(16));
    TmnfTrackHeader {
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

fn parse_track_mesh(data: &[u8], at: usize) -> TmnfTrackMesh {
    let mut r = Reader::at(data, at);
    let base = GmSurf {
        vtable: r.u32(),
        material_index: r.u16(),
        surf_type: r.u8(),
        reserved: r.u8(),
    };
    TmnfTrackMesh {
        base,
        vertex_count: r.u32(),
        pad0c: r.u32(),
        vertices_rel: r.u64(),
        face_count: r.u32(),
        pad1c: r.u32(),
        faces_rel: r.u64(),
        node_count: r.u32(),
        pad2c: r.u32(),
        nodes_rel: r.u64(),
    }
}

fn parse_track_surface(data: &[u8], at: usize) -> TmnfTrackSurface {
    let mut r = Reader::at(data, at);
    TmnfTrackSurface {
        mesh_rel: r.u64(),
        material_ids_rel: r.u64(),
        material_count: r.u32(),
        reserved: r.u32(),
    }
}

fn parse_track_static_entry(data: &[u8], at: usize) -> TmnfTrackStaticEntry {
    let mut r = Reader::at(data, at);
    let skip_count = r.u32();
    let center = r.gm_vec3();
    let half_extent = r.gm_vec3();
    let iso = r.gm_iso4();
    TmnfTrackStaticEntry {
        skip_count,
        box_aligned: GmBoxAligned {
            center,
            half_extent,
        },
        iso,
        tree_flags: r.u32(),
        surface_rel: r.u64(),
        tree_id: r.u32(),
        corpus_id: r.u32(),
    }
}

fn parse_mesh_face(data: &[u8], at: usize) -> GmSurfMeshFace {
    let mut r = Reader::at(data, at);
    let normal = r.gm_vec3();
    let reserved0c = r.u32();
    let vertex = [r.u32(), r.u32(), r.u32()];
    let material_index = r.u16();
    let reserved1e = r.u16();
    GmSurfMeshFace {
        normal,
        reserved0c,
        vertex,
        material_index,
        reserved1e,
    }
}

fn parse_mesh_node(data: &[u8], at: usize) -> GmSurfMeshNode {
    let mut r = Reader::at(data, at);
    let skip_count = r.u32();
    let center = r.gm_vec3();
    let half_extent = r.gm_vec3();
    GmSurfMeshNode {
        skip_count,
        box_aligned: GmBoxAligned {
            center,
            half_extent,
        },
        face_index: r.u32(),
    }
}

/* ---------------------------------------------------------------------------
 * Fixups (validation + owned construction, in the C's exact order)
 * ------------------------------------------------------------------------- */

/// `fix_meshes`: validate and decode every mesh, resolving its vertex/face/
/// node arrays and checking face vertices and octree nodes.
fn fix_meshes(data: &[u8], header: &TmnfTrackHeader) -> Vec<GmSurfMesh> {
    let mesh_section = &header.sections[TMNF_TRACK_MESHES];
    let mut meshes = Vec::with_capacity(mesh_section.count as usize);
    for i in 0..mesh_section.count {
        let raw = parse_track_mesh(
            data,
            (mesh_section.offset + i as u64 * size_of::<TmnfTrackMesh>() as u64) as usize,
        );
        if raw.base.surf_type != GM_SURF_MESH
            || raw.vertex_count == 0
            || raw.face_count == 0
            || raw.node_count == 0
        {
            panic!("tmnf track: invalid mesh header");
        }
        let vertices_rel = raw.vertices_rel;
        let faces_rel = raw.faces_rel;
        let nodes_rel = raw.nodes_rel;
        let vertices_off = resolve_array(
            &header.sections[TMNF_TRACK_VERTICES],
            vertices_rel,
            raw.vertex_count,
        ) as usize;
        let faces_off =
            resolve_array(&header.sections[TMNF_TRACK_FACES], faces_rel, raw.face_count) as usize;
        let nodes_off =
            resolve_array(&header.sections[TMNF_TRACK_NODES], nodes_rel, raw.node_count) as usize;

        let mut vertices = Vec::with_capacity(raw.vertex_count as usize);
        {
            let mut r = Reader::at(data, vertices_off);
            for _ in 0..raw.vertex_count {
                vertices.push(r.gm_vec3());
            }
        }
        let mut faces = Vec::with_capacity(raw.face_count as usize);
        {
            let mut r = Reader::at(data, faces_off);
            for _ in 0..raw.face_count {
                faces.push(parse_mesh_face(data, r.pos));
                r.pos += size_of::<GmSurfMeshFace>();
            }
        }
        let mut nodes = Vec::with_capacity(raw.node_count as usize);
        {
            let mut r = Reader::at(data, nodes_off);
            for _ in 0..raw.node_count {
                nodes.push(parse_mesh_node(data, r.pos));
                r.pos += size_of::<GmSurfMeshNode>();
            }
        }
        let mesh = GmSurfMesh {
            base: raw.base,
            vertex_count: raw.vertex_count,
            vertices,
            face_count: raw.face_count,
            faces,
            node_count: raw.node_count,
            nodes,
        };

        for f in 0..mesh.face_count {
            for v in 0..3 {
                if mesh.faces[f as usize].vertex[v] >= mesh.vertex_count {
                    panic!("tmnf track: mesh face vertex outside vertex array");
                }
            }
        }
        for n in 0..mesh.node_count {
            let node = &mesh.nodes[n as usize];
            if node.skip_count == 0
                || n.wrapping_add(node.skip_count) > mesh.node_count
                || (node.face_index != u32::MAX && node.face_index >= mesh.face_count)
            {
                panic!("tmnf track: invalid mesh octree node");
            }
        }
        meshes.push(mesh);
    }
    meshes
}

/// `fix_surfaces`: resolve each surface's mesh and material-remap array and
/// build the owned `CPlugSurface`s. Returns the surfaces plus the
/// surface-index → mesh-index map the grid builder needs (the C recovered it
/// from the pointer difference `mesh - track->meshes`).
fn fix_surfaces(
    data: &[u8],
    header: &TmnfTrackHeader,
    meshes: &[GmSurfMesh],
) -> (Vec<CPlugSurfaceOwned>, Vec<usize>) {
    let section = &header.sections[TMNF_TRACK_SURFACES];
    let mesh_section = &header.sections[TMNF_TRACK_MESHES];
    let mut surfaces = Vec::with_capacity(section.count as usize);
    let mut surface_mesh = Vec::with_capacity(section.count as usize);
    for i in 0..section.count {
        let raw =
            parse_track_surface(data, (section.offset + i as u64 * size_of::<TmnfTrackSurface>() as u64) as usize);
        if raw.material_count == 0 {
            panic!("tmnf track: surface has no material remap");
        }
        let mesh_rel = raw.mesh_rel;
        let material_ids_rel = raw.material_ids_rel;
        let mesh_off = resolve_array(&header.sections[TMNF_TRACK_MESHES], mesh_rel, 1);
        let material_ids_off = resolve_array(
            &header.sections[TMNF_TRACK_MATERIAL_IDS],
            material_ids_rel,
            raw.material_count,
        );

        let mesh_index =
            ((mesh_off - mesh_section.offset) / size_of::<TmnfTrackMesh>() as u64) as usize;
        let material_ids =
            data[material_ids_off as usize..material_ids_off as usize + raw.material_count as usize]
                .to_vec();

        let mesh = &meshes[mesh_index];
        for f in 0..mesh.face_count {
            if mesh.faces[f as usize].material_index as u32 >= raw.material_count {
                panic!("tmnf track: face material outside surface remap");
            }
        }
        // The C shared one GmSurfMesh between surfaces via pointers; here each
        // referencing surface owns a clone (identical bytes — see module docs).
        surfaces.push(CPlugSurfaceOwned {
            geom: GmSurfGeom::Mesh(mesh.clone()),
            material_ids,
            material_count: raw.material_count,
        });
        surface_mesh.push(mesh_index);
    }
    (surfaces, surface_mesh)
}

/// `fix_entries`: validate the flattened static entries and resolve their
/// surfaces (a surface index in this port).
fn fix_entries(data: &[u8], header: &TmnfTrackHeader) -> Vec<HmsStaticCollisionEntry> {
    let section = &header.sections[TMNF_TRACK_ENTRIES];
    let surfaces_section = &header.sections[TMNF_TRACK_SURFACES];
    let mut entries = Vec::with_capacity(section.count as usize);
    for i in 0..section.count {
        let raw = parse_track_static_entry(
            data,
            (section.offset + i as u64 * size_of::<TmnfTrackStaticEntry>() as u64) as usize,
        );
        if raw.skip_count == 0 || i.wrapping_add(raw.skip_count) > section.count {
            panic!("tmnf track: invalid flattened static entry");
        }
        let surface_rel = raw.surface_rel;
        let surface_off = resolve_array(&header.sections[TMNF_TRACK_SURFACES], surface_rel, 1);
        let surface_index =
            ((surface_off - surfaces_section.offset) / size_of::<TmnfTrackSurface>() as u64) as usize;
        entries.push(HmsStaticCollisionEntry {
            skip_count: raw.skip_count,
            box_aligned: raw.box_aligned,
            iso: raw.iso,
            tree_flags: raw.tree_flags,
            surface: Some(surface_index),
            tree_ref: raw.tree_id,
            corpus_ref: raw.corpus_id,
        });
    }
    entries
}

/// `derive_water`: validate and decode the CHmsZone water map.
fn derive_water(data: &[u8], header: &TmnfTrackHeader) -> TmnfTrackWater {
    let raw_at = header.sections[TMNF_TRACK_WATER].offset as usize;
    let mut r = Reader::at(data, raw_at);
    let cell_x = r.f32();
    let cell_z = r.f32();
    let origin_x = r.f32();
    let origin_z = r.f32();
    let width = r.u32();
    let height = r.u32();
    let default_cell = r.u32();
    let level = r.f32();
    let floor = r.f32();
    let reserved = r.u32();
    let cells_off = header.sections[TMNF_TRACK_WATER_CELLS].offset as usize;
    if header.sections[TMNF_TRACK_WATER].count != 1
        || cell_x != cell_z
        || !(cell_x > 0.0f32)
        || origin_x != 0.0f32
        || origin_z != 0.0f32
        || width == 0
        || width > 256
        || height == 0
        || height > 256
        || default_cell > 1
        || reserved != 0
        || !(floor <= level)
        || header.sections[TMNF_TRACK_WATER_CELLS].count != width.wrapping_mul(height)
    {
        panic!("tmnf track: invalid water map");
    }
    let cells = data[cells_off..cells_off + (width as usize * height as usize)].to_vec();
    let mut water = TmnfTrackWater {
        cell_x,
        cell_z,
        origin_x,
        origin_z,
        width,
        height,
        default_cell: default_cell as u8,
        cells,
        cell_count: 0,
        level,
        floor,
    };
    for i in 0..(width as usize * height as usize) {
        if water.cells[i] > 1 {
            panic!("tmnf track: water map cell is not a flag");
        }
        water.cell_count += water.cells[i] as u32;
    }
    water
}

/* ---------------------------------------------------------------------------
 * Static grid (region-restricted static tree copies)
 * ------------------------------------------------------------------------- */

const GRID_CELL: f64 = 16.0;
const GRID_EXPAND: f64 = 2.5;
const GRID_MESH_EXPAND: f64 = GRID_EXPAND + 1.0;
const GRID_MESH_REGION_MARGIN: f64 = 0.5;
const GRID_BOUNDS_LEAF_HALF: f64 = 64.0;
const GRID_MAX_CELLS: u64 = 1 << 20;
const GRID_ORTHONORMAL_TOLERANCE: f64 = 1e-5;
const MESH_GRID_CELL: [f64; TMNF_MESH_GRID_LEVELS] = [1.0, 2.0];
const MESH_GRID_EXPAND: [f64; TMNF_MESH_GRID_LEVELS] = [0.75, 2.25];
const MESH_GRID_MAX_CELLS: u64 = 1 << 16;

#[derive(Clone, Copy, Debug, Default)]
struct DBox {
    lo: [f64; 3],
    hi: [f64; 3],
}

/// Content-addressed store of variable-length lists inside a pool. Open
/// addressing, FNV-1a over the list bytes, power-of-two capacity, exactly the
/// C's `ListIndex`.
#[derive(Default)]
struct ListIndex {
    slots: Vec<u32>,  /* offsets into the owning pool, u32::MAX empty */
    lengths: Vec<u32>,
    count: usize,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv_bytes(hash: u64, bytes: &[u8]) -> u64 {
    let mut h = hash;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// A pool element whose in-memory bytes the intern table hashes and compares
/// (memcmp semantics — bit-level, so `-0.0` ≠ `0.0`, exactly like the C).
trait PoolElement: Copy {
    fn fnv_update(&self, hash: u64) -> u64;
    fn bytes_equal(&self, other: &Self) -> bool;
}

impl PoolElement for TmnfStaticCellNode {
    fn fnv_update(&self, hash: u64) -> u64 {
        // repr(C) bytes: center(3 f32), half_extent(3 f32), skip, entry_index.
        let mut h = hash;
        for f in [
            self.box_aligned.center.x,
            self.box_aligned.center.y,
            self.box_aligned.center.z,
            self.box_aligned.half_extent.x,
            self.box_aligned.half_extent.y,
            self.box_aligned.half_extent.z,
        ] {
            h = fnv_bytes(h, &f.to_bits().to_le_bytes());
        }
        h = fnv_bytes(h, &self.skip.to_le_bytes());
        h = fnv_bytes(h, &self.entry_index.to_le_bytes());
        h
    }

    fn bytes_equal(&self, other: &Self) -> bool {
        self.box_aligned.center.x.to_bits() == other.box_aligned.center.x.to_bits()
            && self.box_aligned.center.y.to_bits() == other.box_aligned.center.y.to_bits()
            && self.box_aligned.center.z.to_bits() == other.box_aligned.center.z.to_bits()
            && self.box_aligned.half_extent.x.to_bits()
                == other.box_aligned.half_extent.x.to_bits()
            && self.box_aligned.half_extent.y.to_bits()
                == other.box_aligned.half_extent.y.to_bits()
            && self.box_aligned.half_extent.z.to_bits()
                == other.box_aligned.half_extent.z.to_bits()
            && self.skip == other.skip
            && self.entry_index == other.entry_index
    }
}

impl PoolElement for TmnfMeshPoolSlot {
    fn fnv_update(&self, hash: u64) -> u64 {
        let h = fnv_bytes(hash, &self.word0.to_le_bytes());
        fnv_bytes(h, &self.word1.to_le_bytes())
    }

    fn bytes_equal(&self, other: &Self) -> bool {
        self.word0 == other.word0 && self.word1 == other.word1
    }
}

impl ListIndex {
    fn grow<T: PoolElement>(&mut self, pool: &Vec<T>) {
        let capacity = if self.slots.is_empty() {
            4096
        } else {
            self.slots.len() * 2
        };
        let mut slots = vec![u32::MAX; capacity];
        let mut lengths = vec![0u32; capacity];
        for i in 0..self.slots.len() {
            if self.slots[i] == u32::MAX {
                continue;
            }
            let start = self.slots[i] as usize;
            let length = self.lengths[i] as usize;
            let mut hash = FNV_OFFSET;
            for el in &pool[start..start + length] {
                hash = el.fnv_update(hash);
            }
            let mut at = (hash as usize) & (capacity - 1);
            while slots[at] != u32::MAX {
                at = (at + 1) & (capacity - 1);
            }
            slots[at] = self.slots[i];
            lengths[at] = self.lengths[i];
        }
        self.slots = slots;
        self.lengths = lengths;
    }

    /// The candidate list occupies `pool[start ..]`; returns the offset of an
    /// identical stored list (truncating the pool back to `start`), or keeps
    /// the new list and returns `start`.
    fn intern<T: PoolElement>(&mut self, pool: &mut Vec<T>, start: usize) -> u32 {
        let length = pool.len() - start;
        if self.count * 2 >= self.slots.len() {
            self.grow(pool);
        }
        let mut hash = FNV_OFFSET;
        for el in &pool[start..] {
            hash = el.fnv_update(hash);
        }
        let mut at = (hash as usize) & (self.slots.len() - 1);
        while self.slots[at] != u32::MAX {
            if self.lengths[at] as usize == length {
                let existing = self.slots[at] as usize;
                let identical = pool[existing..existing + length]
                    .iter()
                    .zip(pool[start..start + length].iter())
                    .all(|(a, b)| a.bytes_equal(b));
                if identical {
                    pool.truncate(start);
                    return self.slots[at];
                }
            }
            at = (at + 1) & (self.slots.len() - 1);
        }
        if start > u32::MAX as usize || length > u32::MAX as usize {
            panic!("tmnf track: static grid too large");
        }
        self.slots[at] = start as u32;
        self.lengths[at] = length as u32;
        self.count += 1;
        start as u32
    }
}

/// `box_overlaps` — exact arithmetic, in f64 as the C.
fn box_overlaps(box_: &GmBoxAligned, region: &DBox) -> bool {
    let c = [
        box_.center.x,
        box_.center.y,
        box_.center.z,
    ];
    let h = [
        box_.half_extent.x,
        box_.half_extent.y,
        box_.half_extent.z,
    ];
    for a in 0..3 {
        if !(((c[a] as f64) - (h[a] as f64) <= region.hi[a])
            && (region.lo[a] <= (c[a] as f64) + (h[a] as f64)))
        {
            return false;
        }
    }
    true
}

/// `box_contains` — with margin, in f64 as the C.
fn box_contains(box_: &GmBoxAligned, region: &DBox, margin: f64) -> bool {
    let c = [
        box_.center.x,
        box_.center.y,
        box_.center.z,
    ];
    let h = [
        box_.half_extent.x,
        box_.half_extent.y,
        box_.half_extent.z,
    ];
    for a in 0..3 {
        if !((((c[a] as f64) - (h[a] as f64)) + margin <= region.lo[a])
            && (region.hi[a] <= ((c[a] as f64) + (h[a] as f64)) - margin))
        {
            return false;
        }
    }
    true
}

fn box_reach(box_: &GmBoxAligned) -> f64 {
    let c = [
        box_.center.x,
        box_.center.y,
        box_.center.z,
    ];
    let h = [
        box_.half_extent.x,
        box_.half_extent.y,
        box_.half_extent.z,
    ];
    let mut reach = 0.0f64;
    for a in 0..3 {
        let value = (c[a] as f64).abs() + (h[a] as f64);
        if value > reach {
            reach = value;
        }
    }
    reach
}

fn rounding_margin(reach: f64) -> f64 {
    0.05 + 1e-6 * reach
}

/// Fixes list-local skip counts: a frame is (source subtree end, list index
/// of the node, or `usize::MAX` for a node that was dropped).
#[derive(Clone, Copy)]
struct SkipFrame {
    end: u32,
    list_index: usize,
}

fn close_frames<T>(
    frames: &mut [SkipFrame],
    depth: &mut u32,
    source_index: u32,
    list: &mut Vec<T>,
    skip_field: fn(&mut T) -> &mut u32,
) {
    while *depth > 0 && source_index >= frames[*depth as usize - 1].end {
        let frame = frames[*depth as usize - 1];
        *depth -= 1;
        if frame.list_index != usize::MAX {
            let total = list.len();
            *skip_field(&mut list[frame.list_index]) = (total - frame.list_index) as u32;
        }
    }
}

fn push_frame(frames: &mut [SkipFrame], depth: &mut u32, end: u32, list_index: usize) {
    if *depth >= 256 {
        panic!("tmnf track: collision tree deeper than 256");
    }
    frames[*depth as usize] = SkipFrame { end, list_index };
    *depth += 1;
}

fn iso_is_orthonormal(iso: &GmIso4) -> bool {
    for i in 0..3usize {
        for j in 0..3usize {
            let mut dot = 0.0f64;
            for k in 0..3usize {
                dot += (iso.m[k * 3 + i] as f64) * (iso.m[k * 3 + j] as f64);
            }
            if (dot - if i == j { 1.0 } else { 0.0 }).abs() > GRID_ORTHONORMAL_TOLERANCE {
                return false;
            }
        }
    }
    true
}

/// `build_mesh_list`: restricts a mesh's node array to a mesh-space region and
/// interns the list. Returns the header slot.
fn build_mesh_list(
    mesh_pool: &mut Vec<TmnfMeshPoolSlot>,
    mesh_lists: &mut ListIndex,
    mesh: &GmSurfMesh,
    region: &DBox,
    margin: f64,
) -> u32 {
    let start = mesh_pool.len();
    mesh_pool.push(TmnfMeshPoolSlot::default());

    let mut frames = [SkipFrame {
        end: 0,
        list_index: 0,
    }; 256];
    let mut depth = 0u32;
    let mut i = 0u32;
    while i < mesh.node_count {
        close_frames(&mut frames, &mut depth, i, mesh_pool, |s| &mut s.word1);
        let source = &mesh.nodes[i as usize];
        if !box_overlaps(&source.box_aligned, region) {
            i = i.wrapping_add(source.skip_count);
            continue;
        }
        let is_leaf = source.face_index != u32::MAX;
        let mut list_index = usize::MAX;
        if is_leaf || !box_contains(&source.box_aligned, region, margin) {
            mesh_pool.push(TmnfMeshPoolSlot {
                word0: i, /* entry.node */
                word1: 1, /* entry.skip */
            });
            list_index = mesh_pool.len() - 1;
        }
        if source.skip_count > 1 {
            push_frame(
                &mut frames,
                &mut depth,
                i.wrapping_add(source.skip_count),
                list_index,
            );
        }
        i += 1;
    }
    close_frames(
        &mut frames,
        &mut depth,
        u32::MAX,
        mesh_pool,
        |s| &mut s.word1,
    );

    let node_count = (mesh_pool.len() - start - 1) as u32;
    mesh_pool[start].word0 = node_count; /* header->node_count */
    mesh_lists.intern(mesh_pool, start)
}

/// `mesh_reaches_region`: true when some leaf of the mesh tree is reached by
/// the game's scan for a query box inside the region.
fn mesh_reaches_region(mesh: &GmSurfMesh, region: &DBox) -> bool {
    let mut i = 0u32;
    while i < mesh.node_count {
        let node = &mesh.nodes[i as usize];
        if !box_overlaps(&node.box_aligned, region) {
            i = i.wrapping_add(node.skip_count);
            continue;
        }
        if node.face_index != u32::MAX {
            return true;
        }
        i += 1;
    }
    false
}

/// `build_mesh_grid`: builds the per-level uniform grids over one mesh. Each
/// level's cell list region is the cell expanded by the level's query
/// half-extent budget.
fn build_mesh_grid(
    mesh_pools: &mut Vec<Vec<TmnfMeshPoolSlot>>,
    mesh_lists: &mut [ListIndex],
    mesh_index: usize,
    mesh: &GmSurfMesh,
    margin: f64,
) -> TmnfMeshGrid {
    let mut bounds = DBox {
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };
    for i in 0..mesh.node_count as usize {
        let c = [
            mesh.nodes[i].box_aligned.center.x,
            mesh.nodes[i].box_aligned.center.y,
            mesh.nodes[i].box_aligned.center.z,
        ];
        let h = [
            mesh.nodes[i].box_aligned.half_extent.x,
            mesh.nodes[i].box_aligned.half_extent.y,
            mesh.nodes[i].box_aligned.half_extent.z,
        ];
        for a in 0..3 {
            if (c[a] as f64) - (h[a] as f64) < bounds.lo[a] {
                bounds.lo[a] = (c[a] as f64) - (h[a] as f64);
            }
            if (c[a] as f64) + (h[a] as f64) > bounds.hi[a] {
                bounds.hi[a] = (c[a] as f64) + (h[a] as f64);
            }
        }
    }

    let mut grid = TmnfMeshGrid::default();
    if mesh.face_count != 0 {
        let mut edges = Vec::with_capacity(mesh.face_count as usize);
        for i in 0..mesh.face_count as usize {
            let face = &mesh.faces[i];
            let mut e = TmnfSphereFaceEdges {
                xyz: [[0.0f32; 4]; 3],
                edges: [Default::default(); 3],
                padding: [0.0f32; 2],
            };
            for v in 0..4usize {
                let vertex = mesh.vertices[face.vertex[v % 3] as usize];
                e.xyz[0][v] = vertex.x;
                e.xyz[1][v] = vertex.y;
                e.xyz[2][v] = vertex.z;
            }
            e.padding[0] = 0.0f32;
            e.padding[1] = 0.0f32;
            for ed in 0..3usize {
                e.edges[ed] = crate::collision::tmnf_sphere_mesh_edge_prepare(
                    &mesh.vertices[face.vertex[ed] as usize],
                    &mesh.vertices[face.vertex[(ed + 1) % 3] as usize],
                    &face.normal,
                );
            }
            edges.push(e);
        }
        grid.sphere_edges = edges;
    }
    for l in 0..TMNF_MESH_GRID_LEVELS {
        let expand = MESH_GRID_EXPAND[l];
        let mut cell_size = MESH_GRID_CELL[l];
        let mut dims = [0u32; 3];
        loop {
            let mut total: u64 = 1;
            for a in 0..3 {
                let extent = bounds.hi[a] - bounds.lo[a] + 2.0 * (expand + cell_size);
                dims[a] = (extent / cell_size).floor() as u32 + 1;
                total = total.wrapping_mul(dims[a] as u64);
            }
            if total <= MESH_GRID_MAX_CELLS {
                break;
            }
            cell_size *= 2.0;
        }
        let mut level = TmnfMeshGridLevel {
            origin: [0.0f64; 3],
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            max_half: expand - margin,
            dims,
            cell_count: 0,
            cells: Vec::new(),
        };
        let mut cell_count: u32 = 1;
        for a in 0..3 {
            level.origin[a] = bounds.lo[a] - expand - cell_size;
            level.dims[a] = dims[a];
            cell_count = cell_count.wrapping_mul(dims[a]);
        }
        level.cell_count = cell_count;
        let mut cells = Vec::with_capacity(cell_count as usize);
        for cell in 0..cell_count {
            let index = [
                cell % dims[0],
                (cell / dims[0]) % dims[1],
                cell / (dims[0] * dims[1]),
            ];
            let mut region = DBox::default();
            for a in 0..3 {
                region.lo[a] = level.origin[a] + index[a] as f64 * cell_size - expand;
                region.hi[a] = region.lo[a] + cell_size + 2.0 * expand;
            }
            let header = {
                let pool = &mut mesh_pools[mesh_index];
                let lists = &mut mesh_lists[mesh_index];
                build_mesh_list(pool, lists, mesh, &region, margin)
            };
            cells.push(header);
        }
        level.cells = cells;
        grid.levels[l] = level;
    }
    // The C pointed every mesh grid at the one shared pool; each grid here
    // owns its per-mesh pool (see the module docs — behavior-identical).
    grid.pool = mesh_pools[mesh_index].clone();
    grid
}

/// `entry_mesh_grid`: the mesh grid for a leaf entry, or `None` when its
/// surface is not a mesh or its placement is not a rotation.
#[allow(clippy::too_many_arguments)]
fn entry_mesh_grid(
    mesh_pools: &mut Vec<Vec<TmnfMeshPoolSlot>>,
    mesh_lists: &mut [ListIndex],
    mesh_grids: &mut [TmnfMeshGrid],
    mesh_grid_built: &mut [bool],
    mesh_margins: &[f64],
    track: &TmnfTrack,
    surface_mesh: &[usize],
    entry: &HmsStaticCollisionEntry,
) -> Option<usize> {
    let surface_index = match entry.surface {
        Some(i) => i,
        None => return None, /* C: entry->surface == NULL cannot happen post-fixup */
    };
    let surface = &track.surfaces[surface_index];
    let mesh = match &surface.geom {
        GmSurfGeom::Mesh(m) => m,
        _ => return None, /* C: surface->geom->type != GM_SURF_MESH */
    };
    if !iso_is_orthonormal(&entry.iso) {
        return None;
    }
    let m = surface_mesh[surface_index];
    if !mesh_grid_built[m] {
        let margin = mesh_margins[m];
        mesh_grids[m] = build_mesh_grid(mesh_pools, mesh_lists, m, mesh, margin);
        mesh_grid_built[m] = true;
    }
    Some(m)
}

/// `entry_reaches_region`: true when a query box inside `world_region` can
/// reach a leaf of the entry's mesh tree.
fn entry_reaches_region(
    track: &TmnfTrack,
    entry: &HmsStaticCollisionEntry,
    world_region: &DBox,
) -> bool {
    let surface_index = match entry.surface {
        Some(i) => i,
        None => return true,
    };
    let surface = &track.surfaces[surface_index];
    let mesh = match &surface.geom {
        GmSurfGeom::Mesh(m) => m,
        _ => return true,
    };
    if !iso_is_orthonormal(&entry.iso) {
        return true;
    }
    let iso = &entry.iso;
    let mut region = DBox {
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };
    for corner in 0..8u32 {
        let mut w = [0.0f64; 3];
        for a in 0..3usize {
            w[a] = if (corner >> a) & 1 != 0 {
                world_region.hi[a]
            } else {
                world_region.lo[a]
            };
            w[a] -= iso.t[a] as f64;
        }
        for a in 0..3usize {
            let m = (iso.m[0 * 3 + a] as f64) * w[0]
                + (iso.m[1 * 3 + a] as f64) * w[1]
                + (iso.m[2 * 3 + a] as f64) * w[2];
            if m < region.lo[a] {
                region.lo[a] = m;
            }
            if m > region.hi[a] {
                region.hi[a] = m;
            }
        }
    }
    for a in 0..3usize {
        region.lo[a] -= GRID_MESH_REGION_MARGIN;
        region.hi[a] += GRID_MESH_REGION_MARGIN;
    }
    mesh_reaches_region(mesh, &region)
}

/// `prune_empty_cell_nodes`: drops the leaves no query in the cell can hit
/// and the internal nodes left without a leaf below them.
fn prune_empty_cell_nodes(nodes: &mut Vec<TmnfStaticCellNode>, start: usize) {
    let count = (nodes.len() - start) as u32;
    let mut kept_below = vec![0u32; count as usize + 1];
    /* kept_below[j] = kept nodes in list[j..count) */
    kept_below[count as usize] = 0;
    let mut j = count;
    while j > 0 {
        j -= 1;
        let skip = nodes[start + j as usize].skip as usize;
        let keep;
        if nodes[start + j as usize].entry_index == TMNF_CELL_NODE_INTERNAL {
            keep = kept_below[j as usize + 1] - kept_below[j as usize + skip] > 0;
        } else {
            keep = nodes[start + j as usize].entry_index != TMNF_CELL_NODE_EMPTY;
        }
        kept_below[j as usize] = kept_below[j as usize + 1] + if keep { 1u32 } else { 0u32 };
    }
    let mut out = 0u32;
    for jj in 0..count {
        let j = jj as usize;
        if kept_below[j] == kept_below[j + 1] {
            continue;
        }
        let mut node = nodes[start + j];
        let skip = node.skip as usize;
        node.skip = kept_below[j] - kept_below[j + skip];
        nodes[start + out as usize] = node;
        out += 1;
    }
    nodes.truncate(start + out as usize);
}

/// `build_cell_list`: one grid cell's region-restricted copy of the static
/// tree, interned. Returns (offset, count) into the grid's node pool.
fn build_cell_list(
    nodes: &mut Vec<TmnfStaticCellNode>,
    node_lists: &mut ListIndex,
    track: &TmnfTrack,
    cell: &DBox,
    margin: f64,
) -> (u32, u32) {
    let mut region = DBox::default();
    let mut mesh_region = DBox::default();
    for a in 0..3usize {
        region.lo[a] = cell.lo[a] - GRID_EXPAND;
        region.hi[a] = cell.hi[a] + GRID_EXPAND;
        mesh_region.lo[a] = cell.lo[a] - GRID_MESH_EXPAND;
        mesh_region.hi[a] = cell.hi[a] + GRID_MESH_EXPAND;
    }

    let start = nodes.len();
    let mut frames = [SkipFrame {
        end: 0,
        list_index: 0,
    }; 256];
    let mut depth = 0u32;
    let mut i = 0u32;
    while i < track.entry_count {
        close_frames(&mut frames, &mut depth, i, nodes, |n| &mut n.skip);
        let entry = &track.entries[i as usize];
        if !box_overlaps(&entry.box_aligned, &region) {
            i = i.wrapping_add(entry.skip_count);
            continue;
        }
        let is_leaf = entry.surface.is_some() && (entry.tree_flags & 0x80u32) != 0;
        let mut list_index = usize::MAX;
        if is_leaf || !box_contains(&entry.box_aligned, &region, margin) {
            let entry_index = if is_leaf {
                if entry_reaches_region(track, entry, &mesh_region) {
                    i
                } else {
                    TMNF_CELL_NODE_EMPTY
                }
            } else {
                TMNF_CELL_NODE_INTERNAL
            };
            nodes.push(TmnfStaticCellNode {
                box_aligned: entry.box_aligned,
                skip: 1,
                entry_index,
            });
            list_index = nodes.len() - 1;
        }
        if entry.skip_count > 1 {
            push_frame(
                &mut frames,
                &mut depth,
                i.wrapping_add(entry.skip_count),
                list_index,
            );
        }
        i += 1;
    }
    close_frames(&mut frames, &mut depth, u32::MAX, nodes, |n| &mut n.skip);
    prune_empty_cell_nodes(nodes, start);
    let count = (nodes.len() - start) as u32;
    let offset = node_lists.intern(nodes, start);
    (offset, count)
}

/// `build_static_grid`.
fn build_static_grid(track: &mut TmnfTrack, surface_mesh: &[usize]) {
    let mut bounds = DBox {
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };
    let mut reach = 0.0f64;
    let mut bounded_leaves = 0u32;
    for entry in &track.entries {
        let value = box_reach(&entry.box_aligned);
        if value > reach {
            reach = value;
        }
        if entry.surface.is_none() || (entry.tree_flags & 0x80u32) == 0 {
            continue;
        }
        let c = [
            entry.box_aligned.center.x,
            entry.box_aligned.center.y,
            entry.box_aligned.center.z,
        ];
        let h = [
            entry.box_aligned.half_extent.x,
            entry.box_aligned.half_extent.y,
            entry.box_aligned.half_extent.z,
        ];
        if (h[0] as f64) > GRID_BOUNDS_LEAF_HALF
            || (h[1] as f64) > GRID_BOUNDS_LEAF_HALF
            || (h[2] as f64) > GRID_BOUNDS_LEAF_HALF
        {
            continue;
        }
        bounded_leaves += 1;
        for a in 0..3usize {
            if (c[a] as f64) - (h[a] as f64) < bounds.lo[a] {
                bounds.lo[a] = (c[a] as f64) - (h[a] as f64);
            }
            if (c[a] as f64) + (h[a] as f64) > bounds.hi[a] {
                bounds.hi[a] = (c[a] as f64) + (h[a] as f64);
            }
        }
    }
    if bounded_leaves == 0 {
        /* the C memsets the grid and returns; it stays default here */
        return;
    }
    if !reach.is_finite() {
        panic!("tmnf track: static collision tree has non-finite boxes");
    }

    let mut cell_size = GRID_CELL;
    let mut dims = [0u32; 3];
    loop {
        let mut total: u64 = 1;
        for a in 0..3usize {
            let extent = bounds.hi[a] - bounds.lo[a];
            dims[a] = (extent / cell_size).floor() as u32 + 1;
            total = total.wrapping_mul(dims[a] as u64);
        }
        if total <= GRID_MAX_CELLS {
            break;
        }
        cell_size *= 2.0;
    }

    let mut grid = TmnfStaticGrid {
        origin: [0.0f64; 3],
        cell_size,
        inv_cell_size: 1.0 / cell_size,
        expand: GRID_EXPAND,
        margin: rounding_margin(reach),
        dims,
        cell_count: dims[0].wrapping_mul(dims[1]).wrapping_mul(dims[2]),
        cell_offsets: Vec::new(),
        cell_counts: Vec::new(),
        nodes: Vec::new(),
        // The C's shared interned pool; per-mesh pools live in each
        // TmnfMeshGrid here (see the module docs), so this stays empty.
        mesh_pool: Vec::new(),
        mesh_grids: Vec::new(),
        entry_mesh_grids: Vec::new(),
        entry_inverse_isos: Vec::new(),
    };
    for a in 0..3usize {
        grid.origin[a] = bounds.lo[a];
        grid.dims[a] = dims[a];
    }

    let mesh_count = track.meshes.len();
    let mut nodes: Vec<TmnfStaticCellNode> = Vec::new();
    let mut node_lists = ListIndex::default();
    let mut mesh_pools: Vec<Vec<TmnfMeshPoolSlot>> = vec![Vec::new(); mesh_count];
    let mut mesh_lists: Vec<ListIndex> =
        (0..mesh_count).map(|_| ListIndex::default()).collect();
    let mut mesh_margins = vec![0.0f64; mesh_count];
    let mut mesh_grids: Vec<TmnfMeshGrid> = vec![TmnfMeshGrid::default(); mesh_count];
    let mut mesh_grid_built = vec![false; mesh_count];

    for m in 0..mesh_count {
        let mesh = &track.meshes[m];
        let mut mesh_reach = 0.0f64;
        for i in 0..mesh.node_count as usize {
            let value = box_reach(&mesh.nodes[i].box_aligned);
            if value > mesh_reach {
                mesh_reach = value;
            }
        }
        if !mesh_reach.is_finite() {
            panic!("tmnf track: mesh collision tree has non-finite boxes");
        }
        mesh_margins[m] = rounding_margin(mesh_reach);
    }

    let mut entry_mesh_grids: Vec<Option<usize>> = Vec::with_capacity(track.entries.len());
    let mut entry_inverse_isos: Vec<GmIso4> = Vec::with_capacity(track.entries.len());
    for entry in &track.entries {
        let mut inverse = GmIso4::default();
        inverse.set_inverse(&entry.iso);
        entry_inverse_isos.push(inverse);
        let mg = if entry.surface.is_some() && (entry.tree_flags & 0x80u32) != 0 {
            entry_mesh_grid(
                &mut mesh_pools,
                &mut mesh_lists,
                &mut mesh_grids,
                &mut mesh_grid_built,
                &mesh_margins,
                track,
                surface_mesh,
                entry,
            )
        } else {
            None
        };
        entry_mesh_grids.push(mg);
    }

    let mut cell_offsets = vec![0u32; grid.cell_count as usize];
    let mut cell_counts = vec![0u32; grid.cell_count as usize];
    for cell in 0..grid.cell_count {
        let index = [
            cell % dims[0],
            (cell / dims[0]) % dims[1],
            cell / (dims[0] * dims[1]),
        ];
        let mut cell_box = DBox::default();
        for a in 0..3usize {
            cell_box.lo[a] = grid.origin[a] + index[a] as f64 * cell_size;
            cell_box.hi[a] = cell_box.lo[a] + cell_size;
        }
        let (offset, count) =
            build_cell_list(&mut nodes, &mut node_lists, track, &cell_box, grid.margin);
        cell_offsets[cell as usize] = offset;
        cell_counts[cell as usize] = count;
    }

    let mut grid_mesh_slot_count = 0usize;
    for m in 0..mesh_count {
        grid_mesh_slot_count += mesh_pools[m].len();
    }
    grid.cell_offsets = cell_offsets;
    grid.cell_counts = cell_counts;
    grid.nodes = nodes;
    grid.mesh_grids = mesh_grids;
    grid.entry_mesh_grids = entry_mesh_grids;
    grid.entry_inverse_isos = entry_inverse_isos;
    track.grid_node_count = grid.nodes.len();
    track.grid_mesh_slot_count = grid_mesh_slot_count;
    track.grid = grid;
}

/* ---------------------------------------------------------------------------
 * Static response bodies
 * ------------------------------------------------------------------------- */

/// CHmsCorpus::GetLocation (0x005474A0) is corpus+0x18, not the flattened
/// tree iso of the entry, so the body iso is the corpus iso.
fn build_static_response_bodies(track: &mut TmnfTrack) {
    let mut maximum = 0u32;
    for i in 0..track.entries.len() {
        if maximum < track.entries[i].corpus_ref {
            maximum = track.entries[i].corpus_ref;
        }
    }
    track.static_response_count = maximum.wrapping_add(1);
    let count = track.static_response_count as usize;
    let mut bodies: Vec<CHmsResponseBody> = Vec::with_capacity(count);
    let mut present = vec![0u8; count];
    for _ in 0..count {
        // calloc-zeroed in the C; NULL dyna / no contact sink.
        bodies.push(CHmsResponseBody {
            corpus_ref: 0,
            classification_flags: 0,
            response_flags: 0,
            response_weight: 0.0,
            iso: GmIso4::default(),
            dyna: None,
            has_contact_sink: false,
            sink: AbsorbSink::None,
        });
    }
    for i in 0..track.entries.len() {
        let id = track.entries[i].corpus_ref;
        if id == 0 || present[id as usize] != 0 {
            continue;
        }
        let body = &mut bodies[id as usize];
        present[id as usize] = 1;
        body.corpus_ref = id;
        body.classification_flags = 0x11888101u32;
        body.response_flags = 0xfff18002u32;
        body.response_weight = 1.0f32;
        body.iso = track.corpus_isos[id as usize - 1];
    }
    track.static_response_bodies = bodies;
    track.static_response_present = present;
}

/* ---------------------------------------------------------------------------
 * TmnfTrack_Load
 * ------------------------------------------------------------------------- */

/// `TmnfTrack_Load`: loads and validates a `TMNFTRK1` snapshot.
///
/// `expected_track_sha256` is the SHA-256 of the source Challenge.Gbx (the
/// C dereferences the pointer unconditionally; `None` here skips that one
/// check). Panics on structural corruption exactly where the C aborts.
pub fn tmnf_track_load(path: &str, expected_track_sha256: Option<&[u8; 32]>) -> TmnfTrack {
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(_) => panic!("tmnf track: cannot open snapshot"),
    };
    if data.len() < size_of::<TmnfTrackHeader>() {
        panic!("tmnf track: invalid snapshot size");
    }
    let size = data.len();

    let header = parse_track_header(&data);
    if header.magic != *b"TMNFTRK1"
        || header.version != TMNF_TRACK_VERSION
        || header.endian != 0x12345678
        || header.header_size != size_of::<TmnfTrackHeader>() as u32
        || header.section_count != TMNF_TRACK_SECTION_COUNT as u32
        || header.file_size != size as u64
    {
        panic!("tmnf track: invalid snapshot header");
    }
    if header.exe_sha256 != TMNF_21126_EXE_SHA256 {
        panic!("tmnf track: snapshot targets a different executable");
    }
    if let Some(expected) = expected_track_sha256 {
        if header.track_sha256 != *expected {
            panic!("tmnf track: snapshot is for a different track");
        }
    }
    validate_sections(&header, size);
    if header.sections[TMNF_TRACK_MATERIAL_DATA].count != TMNF_TRACK_MATERIAL_COUNT {
        panic!("tmnf track: physical material table must contain 31 entries");
    }

    let payload_digest = sha256_digest(&data[header.header_size as usize..]);
    if payload_digest != header.payload_sha256 {
        panic!("tmnf track: payload SHA-256 mismatch");
    }

    let meshes = fix_meshes(&data, &header);
    let (surfaces, surface_mesh) = fix_surfaces(&data, &header, &meshes);
    let entries = fix_entries(&data, &header);

    let entries_section = &header.sections[TMNF_TRACK_ENTRIES];
    let surfaces_section = &header.sections[TMNF_TRACK_SURFACES];
    let meshes_section = &header.sections[TMNF_TRACK_MESHES];
    let materials_section = &header.sections[TMNF_TRACK_MATERIAL_DATA];
    let pairs_section = &header.sections[TMNF_TRACK_COLLISION_PAIRS];
    let corpus_section = &header.sections[TMNF_TRACK_CORPUS_ISOS];

    let mut materials = Vec::with_capacity(materials_section.count as usize);
    {
        let mut r = Reader::at(&data, materials_section.offset as usize);
        for _ in 0..materials_section.count {
            let friction = r.f32();
            let restitution = r.f32();
            materials.push(CPlugSurfaceMaterialData {
                friction,
                restitution,
            });
        }
    }
    let mut collision_pairs = Vec::with_capacity(pairs_section.count as usize);
    {
        let mut r = Reader::at(&data, pairs_section.offset as usize);
        for _ in 0..pairs_section.count {
            let raw = [r.u32(), r.u32(), r.u32(), r.u32(), r.u32()];
            collision_pairs.push(TmnfTrackCollisionPair { raw });
        }
    }
    let mut corpus_isos = Vec::with_capacity(corpus_section.count as usize);
    {
        let mut r = Reader::at(&data, corpus_section.offset as usize);
        for _ in 0..corpus_section.count {
            corpus_isos.push(r.gm_iso4());
        }
    }

    let mut track = TmnfTrack {
        entry_count: entries_section.count,
        surface_count: surfaces_section.count,
        mesh_count: meshes_section.count,
        collision_pair_count: pairs_section.count,
        corpus_count: corpus_section.count,
        header,
        entries,
        surfaces,
        meshes,
        materials,
        collision_pairs,
        corpus_isos,
        water: TmnfTrackWater::default(),
        grid: TmnfStaticGrid::default(),
        grid_node_count: 0,
        grid_mesh_slot_count: 0,
        static_response_bodies: Vec::new(),
        static_response_present: Vec::new(),
        static_response_count: 0,
    };

    for i in 0..track.entry_count as usize {
        if track.entries[i].corpus_ref > track.corpus_count {
            panic!("tmnf track: static entry references a missing corpus location");
        }
    }

    track.water = derive_water(&data, &track.header);
    build_static_grid(&mut track, &surface_mesh);
    build_static_response_bodies(&mut track);
    track
}

/* ---------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    fn hex32(s: &str) -> [u8; 32] {
        let bytes: Vec<u8> = (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect();
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    #[test]
    fn sha256_known_answers() {
        // FIPS 180 vectors, also proving the loader's inline SHA-256.
        assert_eq!(
            hex32("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            sha256_digest(b"")
        );
        assert_eq!(
            hex32("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            sha256_digest(b"abc")
        );
        assert_eq!(
            hex32("9a900403ac313ba27a1bc81f0932652b8020dac92c234d98fa0b06bf0040ecfd"),
            sha256_digest(b"qwertyuiop")
        );
    }

    #[test]
    fn load_a01_race_track() {
        // From oracle/tracks/manifest.txt: the Challenge.Gbx hash.
        let expected = hex32(
            "f0a870809be99da2cb36ad5df43a2cf63d8f74fe4ac3470ecac68b9e97625dc3",
        );
        let track = tmnf_track_load(
            "/home/z/my-project/tmnf-physics/oracle/tracks/A01-Race.tmnftrack",
            Some(&expected),
        );
        // Counts from oracle/tracks/A01-Race.tmnftrack.log.
        assert_eq!(track.entry_count, 2603);
        assert_eq!(track.surface_count, 1499);
        assert_eq!(track.mesh_count, 96);
        assert_eq!(track.meshes[0].vertex_count, 4);
        assert_eq!(track.meshes[0].face_count, 2);
        assert_eq!(track.meshes[0].node_count, 3);
        assert_eq!(track.materials.len(), 31);
        assert_eq!(track.collision_pair_count, 3);
        assert_eq!(track.corpus_count, 1432);

        // Materials: the reference table (Concrete: 1.0/0.5).
        assert_eq!(track.materials[0].friction, 1.0);
        assert_eq!(track.materials[0].restitution, 0.5);

        // Root entry: skip 2603, internal (tree_flags 0), surface 0, corpus 1.
        assert_eq!(track.entries[0].skip_count, 2603);
        assert_eq!(track.entries[0].tree_flags, 0);
        assert_eq!(track.entries[0].surface, Some(0));
        assert_eq!(track.entries[0].corpus_ref, 1);
        assert_eq!(track.entries[0].iso.t, [512.0, 41.010005950927734, 511.9999694824219]);

        // Face normals keep their bits (-0.0 must survive the parse).
        let mesh0 = &track.meshes[0];
        assert_eq!(mesh0.faces[0].normal.x.to_bits(), (-0.0f32).to_bits());
        assert_eq!(mesh0.faces[0].normal.y, 1.0);
        assert_eq!(mesh0.faces[0].vertex, [0, 1, 2]);
        assert_eq!(mesh0.nodes[0].skip_count, 3);
        assert_eq!(mesh0.nodes[0].face_index, u32::MAX);

        // Surface 0: single material id, remap {2 = Grass}.
        assert_eq!(track.surfaces[0].material_count, 1);
        assert_eq!(track.surfaces[0].material_ids, vec![2u8]);

        // Water: 32x32 cells of 32 m at level 8, floor -992, no set cells.
        assert_eq!(track.water.width, 32);
        assert_eq!(track.water.height, 32);
        assert_eq!(track.water.cell_x, 32.0);
        assert_eq!(track.water.level, 8.0);
        assert_eq!(track.water.floor, -992.0);
        assert_eq!(track.water.default_cell, 0);
        assert_eq!(track.water.cell_count, 0);
        assert_eq!(track.water.cells.len(), 1024);

        // Static grid: 65 x 12 x 65 cells of 16 m (Stadium bounds).
        assert_eq!(track.grid.dims, [65, 12, 65]);
        assert_eq!(track.grid.cell_count, 50700);
        assert_eq!(track.grid.cell_size, 16.0);
        assert_eq!(track.grid.cell_offsets.len(), 50700);
        assert!(track.grid_node_count > 0);
        assert!(track.grid_mesh_slot_count > 0);
        // Mesh grids were built (leaves reference all 96 meshes).
        assert_eq!(track.grid.mesh_grids.len(), 96);
        assert!(track
            .grid
            .entry_mesh_grids
            .iter()
            .filter(|m| m.is_some())
            .count() > 0);

        // Static response bodies: 1432 corpora, all 1499 leaf corpus refs set.
        assert_eq!(track.static_response_count, 1433);
        assert_eq!(track.static_response_bodies.len(), 1433);
        let set = track.static_response_present.iter().filter(|&&p| p != 0).count();
        assert_eq!(set, 1432);
        let body = &track.static_response_bodies[1];
        assert_eq!(body.corpus_ref, 1);
        assert_eq!(body.classification_flags, 0x11888101);
        assert_eq!(body.response_flags, 0xfff18002);
        assert_eq!(body.response_weight, 1.0);
        assert_eq!(body.iso.t, [0.0, 9.0, 0.0]);
        assert!(body.dyna.is_none());

        // static_group() mirrors BindStaticGroup.
        let group = track.static_group();
        assert!(group.is_static);
        assert_eq!(group.static_entry_count, 2603);
        assert_eq!(group.static_entries.len(), 2603);
        assert!(group.static_grid.is_some());
    }

    #[test]
    fn load_a01_race_track_without_expected_sha() {
        // None skips only the track-hash comparison (documented deviation).
        let track = tmnf_track_load(
            "/home/z/my-project/tmnf-physics/oracle/tracks/A01-Race.tmnftrack",
            None,
        );
        assert_eq!(track.entry_count, 2603);
    }

    #[test]
    #[should_panic(expected = "tmnf track: snapshot is for a different track")]
    fn load_rejects_wrong_track_sha() {
        let wrong = [0u8; 32];
        let _ = tmnf_track_load(
            "/home/z/my-project/tmnf-physics/oracle/tracks/A01-Race.tmnftrack",
            Some(&wrong),
        );
    }

    #[test]
    #[should_panic(expected = "tmnf track: cannot open snapshot")]
    fn load_rejects_missing_file() {
        let _ = tmnf_track_load("/nonexistent/A01.tmnftrack", None);
    }

    #[test]
    #[should_panic(expected = "tmnf track: payload SHA-256 mismatch")]
    fn load_rejects_corrupt_payload() {
        // Corrupt one payload byte (a vertex) in memory and load from it.
        let data = fs::read(
            "/home/z/my-project/tmnf-physics/oracle/tracks/A01-Race.tmnftrack",
        )
        .unwrap();
        let mut corrupt = data.clone();
        let vertex_off = 0x472f8usize + 4;
        corrupt[vertex_off] ^= 0xff;
        let path = std::env::temp_dir().join("tmnf_corrupt_payload.tmnftrack");
        fs::write(&path, &corrupt).unwrap();
        let _ = tmnf_track_load(path.to_str().unwrap(), None);
    }
}
