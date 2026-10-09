//! Collision types and narrowphase kernels, transliterated from
//! `src/collision.h` / `src/collision.c` (2.11.26 disassembly).
//!
//! Only the **scalar** paths are ported: the AVX2/AVX-512 packet kernel
//! (`collision_packet*`) and the in-file `TMNF_SSE` blocks are perf-only
//! accelerations proven round-identical to scalar in the golden replay gate.
//!
//! Pointer-linked graphs become owned data: `GmSurfMesh` owns its
//! vertices/faces/nodes; the polymorphic `GmSurf` vtable dispatch becomes
//! [`GmSurfGeom`]; mesh-list "pointers" into the accel pool become slices.

use crate::buffer::{CHmsCollisionBuffer, SHmsSphereBufferContact};
use crate::gm::*;
use std::mem::{offset_of, size_of};

/* ---------------------------------------------------------------------------
 * Magic FP constants (Rust has no hex-float literals — bits spelled out).
 * ------------------------------------------------------------------------- */

/// `0x1.b7cdfcp-34f` — the edge-length / normal-squared floor.
const TINY_SQUARED: f32 = f32::from_bits(0x2EB7_CDFC);
/// `0x1.4f8b58p-17f` — edge closest-point squared floor.
const EDGE_CLOSEST_SQUARED: f32 = f32::from_bits(0x374F_8B58);
/// The double multiplier of `GmVec3_IsNearlyEqual` (`0x1.4f8b588e368f1p-17`).
const NEARLY_EQUAL_SCALE: f64 = f64::from_bits(0x3EE4_F8B5_88E3_68F1);
/// Global 0x00D67530, cos(pi/6) — the sphere merge threshold (0x3f5db3d7).
const SPHERE_MERGE_THRESHOLD: f32 = f32::from_bits(0x3F5D_B3D7);

/* ---------------------------------------------------------------------------
 * Boxes
 * ------------------------------------------------------------------------- */

/// 0x18 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmBoxAligned {
    pub center: GmVec3,
    pub half_extent: GmVec3,
}
const _: () = assert!(size_of::<GmBoxAligned>() == 0x18);

impl GmBoxAligned {
    /// 24 raw little-endian bytes.
    pub fn as_bytes(&self) -> [u8; 24] {
        let mut b = [0u8; 24];
        b[0..12].copy_from_slice(&self.center.as_bytes());
        b[12..24].copy_from_slice(&self.half_extent.as_bytes());
        b
    }

    /// 0x00537530  Tests overlap of two center/half-extent AABBs (z, y, x
    /// order with early exits, as transcribed).
    pub fn test_inter(&self, other: &GmBoxAligned) -> bool {
        // Single-op F() widenings round identically to plain f32 arithmetic.
        let delta = other.center.z - self.center.z;
        let absolute_delta = delta.abs();
        let extent = other.half_extent.z + self.half_extent.z;
        if !(absolute_delta <= extent) {
            return false;
        }

        let delta = other.center.y - self.center.y;
        let absolute_delta = delta.abs();
        let extent = other.half_extent.y + self.half_extent.y;
        if !(absolute_delta <= extent) {
            return false;
        }

        let delta = other.center.x - self.center.x;
        let absolute_delta = delta.abs();
        let extent = other.half_extent.x + self.half_extent.x;
        absolute_delta <= extent
    }

    /// 0x008E5230  Transforms an AABB by an affine isometry.
    /// VALIDATED upstream: 714/714 golden records.
    pub fn set_mult(&mut self, source: &GmBoxAligned, iso: &GmIso4) {
        self.center.x = (((source.center.x * iso.m[0]) + (iso.m[1] * source.center.y))
            + (iso.m[2] * source.center.z))
            + iso.t[0];
        self.center.y = (((iso.m[4] * source.center.y) + (source.center.x * iso.m[3]))
            + (iso.m[5] * source.center.z))
            + iso.t[1];
        self.center.z = (((iso.m[7] * source.center.y) + (iso.m[6] * source.center.x))
            + (iso.m[8] * source.center.z))
            + iso.t[2];

        self.half_extent.x = ((iso.m[1].abs() * source.half_extent.y)
            + (iso.m[0].abs() * source.half_extent.x))
            + (iso.m[2].abs() * source.half_extent.z);
        self.half_extent.y = ((iso.m[4].abs() * source.half_extent.y)
            + (iso.m[3].abs() * source.half_extent.x))
            + (iso.m[5].abs() * source.half_extent.z);
        self.half_extent.z = ((iso.m[7].abs() * source.half_extent.y)
            + (iso.m[6].abs() * source.half_extent.x))
            + (iso.m[8].abs() * source.half_extent.z);
    }
}

/* ---------------------------------------------------------------------------
 * Surfaces
 * ------------------------------------------------------------------------- */

pub const GM_SURF_SPHERE: u8 = 0;
pub const GM_SURF_ELLIPSOID: u8 = 1;
pub const GM_SURF_PLANE: u8 = 2;
pub const GM_SURF_QUAD_HEIGHT: u8 = 3;
pub const GM_SURF_TRIANGLE_HEIGHT: u8 = 4;
pub const GM_SURF_POLYGON: u8 = 5;
pub const GM_SURF_BOX: u8 = 6;
pub const GM_SURF_MESH: u8 = 7;
pub const GM_SURF_CYLINDER: u8 = 8;

/// 0x08 bytes. The `vtable` word is carried for provenance only.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurf {
    pub vtable: u32,
    pub material_index: u16,
    pub surf_type: u8,
    pub reserved: u8,
}
const _: () = assert!(size_of::<GmSurf>() == 0x08);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfSphere {
    pub base: GmSurf,
    pub radius: f32,
}
const _: () = assert!(size_of::<GmSurfSphere>() == 0x0c);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfEllipsoid {
    pub base: GmSurf,
    pub radii: GmVec3,
}
const _: () = assert!(size_of::<GmSurfEllipsoid>() == 0x14);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfBox {
    pub base: GmSurf,
    pub center: GmVec3,
    pub half_extent: GmVec3,
}
const _: () = assert!(size_of::<GmSurfBox>() == 0x20);

/// 0x4c bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfPolygon {
    pub base: GmSurf,
    pub vertices: [GmVec3; 4],
    pub vertex_count: u8,
    pub reserved0: [u8; 3],
    pub normal: GmVec3,
    pub one_sided: u32,
}
const _: () = assert!(size_of::<GmSurfPolygon>() == 0x4c);

/// 0x20 bytes. `vertex[3]` are indices into the owning mesh's vertex array.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfMeshFace {
    pub normal: GmVec3,
    pub reserved0c: u32,
    pub vertex: [u32; 3],
    pub material_index: u16,
    pub reserved1e: u16,
}
const _: () = assert!(size_of::<GmSurfMeshFace>() == 0x20);

/// 0x20 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmSurfMeshNode {
    pub skip_count: u32,
    pub box_aligned: GmBoxAligned,
    pub face_index: u32,
}
const _: () = assert!(size_of::<GmSurfMeshNode>() == 0x20);

/// The mesh surface — owns its geometry (the C used pointers into the
/// mmap'd image; the Rust port owns the decoded arrays).
#[derive(Clone, Debug, Default)]
pub struct GmSurfMesh {
    pub base: GmSurf,
    pub vertex_count: u32,
    pub vertices: Vec<GmVec3>,
    pub face_count: u32,
    pub faces: Vec<GmSurfMeshFace>,
    pub node_count: u32,
    pub nodes: Vec<GmSurfMeshNode>,
}

/// The polymorphic `GmSurf*` family behind `CPlugSurface::geom`.
#[derive(Clone, Debug)]
pub enum GmSurfGeom {
    Sphere(GmSurfSphere),
    Ellipsoid(GmSurfEllipsoid),
    Box(GmSurfBox),
    Polygon(GmSurfPolygon),
    Mesh(GmSurfMesh),
}

impl GmSurfGeom {
    pub fn surf_type(&self) -> u8 {
        match self {
            GmSurfGeom::Sphere(_) => GM_SURF_SPHERE,
            GmSurfGeom::Ellipsoid(_) => GM_SURF_ELLIPSOID,
            GmSurfGeom::Box(_) => GM_SURF_BOX,
            GmSurfGeom::Polygon(_) => GM_SURF_POLYGON,
            GmSurfGeom::Mesh(_) => GM_SURF_MESH,
        }
    }

    pub fn base(&self) -> &GmSurf {
        match self {
            GmSurfGeom::Sphere(s) => &s.base,
            GmSurfGeom::Ellipsoid(s) => &s.base,
            GmSurfGeom::Box(s) => &s.base,
            GmSurfGeom::Polygon(s) => &s.base,
            GmSurfGeom::Mesh(s) => &s.base,
        }
    }

    pub fn as_sphere(&self) -> &GmSurfSphere {
        match self {
            GmSurfGeom::Sphere(s) => s,
            _ => panic!("tmnf: surface is not a sphere"),
        }
    }

    pub fn as_ellipsoid(&self) -> &GmSurfEllipsoid {
        match self {
            GmSurfGeom::Ellipsoid(s) => s,
            _ => panic!("tmnf: surface is not an ellipsoid"),
        }
    }

    pub fn as_box(&self) -> &GmSurfBox {
        match self {
            GmSurfGeom::Box(s) => s,
            _ => panic!("tmnf: surface is not a box"),
        }
    }

    pub fn as_mesh(&self) -> &GmSurfMesh {
        match self {
            GmSurfGeom::Mesh(s) => s,
            _ => panic!("tmnf: surface is not a mesh"),
        }
    }
}

/* ---------------------------------------------------------------------------
 * Sphere/mesh edge cache
 * ------------------------------------------------------------------------- */

/// Sphere/mesh edge directions and side planes depend only on the immutable
/// triangle. The exact scalar operations are shared by loading and fallback
/// queries, so caching does not change their rounding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfSphereMeshEdge {
    pub direction: GmVec3,
    pub side: GmVec3,
}

/// Per-face cache: SoA vertices `xyz[axis][lane]` (mesh space) + prepared
/// edges. Built by the track loader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TmnfSphereFaceEdges {
    pub xyz: [[f32; 4]; 3],
    pub edges: [TmnfSphereMeshEdge; 3],
    pub padding: [f32; 2],
}

pub fn tmnf_sphere_mesh_edge_prepare(
    vertex: &GmVec3,
    next: &GmVec3,
    normal: &GmVec3,
) -> TmnfSphereMeshEdge {
    let mut edge = GmVec3 {
        x: next.x - vertex.x,
        y: next.y - vertex.y,
        z: next.z - vertex.z,
    };
    let squared = ((edge.x * edge.x) + (edge.y * edge.y)) + (edge.z * edge.z);
    if TINY_SQUARED < squared {
        let length = squared.sqrt();
        let inverse = 1.0f32 / length;
        edge.x = inverse * edge.x;
        edge.y = edge.y * inverse;
        edge.z = inverse * edge.z;
    }
    let side = GmVec3 {
        x: (edge.y * normal.z) - (edge.z * normal.y),
        y: (normal.x * edge.z) - (normal.z * edge.x),
        z: (normal.y * edge.x) - (edge.y * normal.x),
    };
    TmnfSphereMeshEdge { direction: edge, side }
}

/* ---------------------------------------------------------------------------
 * Collision records
 * ------------------------------------------------------------------------- */

/// 0x38 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmCollision {
    pub separation: GmVec3,   /* 0x00 */
    pub normal: GmVec3,       /* 0x0c */
    pub position: GmVec3,     /* 0x18 */
    pub material1: u16,       /* 0x24 */
    pub material2: u16,       /* 0x26 */
    pub flags: u32,           /* 0x28 */
    pub face_normal: GmVec3,  /* 0x2c */
}
const _: () = assert!(size_of::<GmCollision>() == 0x38);
const _: () = assert!(offset_of!(GmCollision, normal) == 0x0c);
const _: () = assert!(offset_of!(GmCollision, position) == 0x18);
const _: () = assert!(offset_of!(GmCollision, flags) == 0x28);
const _: () = assert!(offset_of!(GmCollision, face_normal) == 0x2c);

/// 0x4c bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SHmsPhysicalCollision {
    pub corpus1: u32,
    pub tree1: u32,
    pub corpus2: u32,
    pub tree2: u32,
    pub collision: GmCollision,
    pub material: u32,
}
const _: () = assert!(size_of::<SHmsPhysicalCollision>() == 0x4c);
const _: () = assert!(offset_of!(SHmsPhysicalCollision, collision) == 0x10);
const _: () = assert!(offset_of!(SHmsPhysicalCollision, material) == 0x48);

/* ---------------------------------------------------------------------------
 * Mesh accel structures (grid + region-restricted lists)
 * ------------------------------------------------------------------------- */

/// A mesh list is a header slot (node count) followed by node_count entries
/// (node index + list-local skip count).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfMeshListEntry {
    pub node: u32,
    pub skip: u32,
}
const _: () = assert!(size_of::<TmnfMeshListEntry>() == 8);

/// The C union: a slot is either a list header (`node_count` in word 0) or an
/// entry (`node`/`skip`). Kept as two words with typed accessors.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfMeshPoolSlot {
    pub word0: u32,
    pub word1: u32,
}
const _: () = assert!(size_of::<TmnfMeshPoolSlot>() == 8);

impl TmnfMeshPoolSlot {
    pub fn node_count(&self) -> u32 {
        self.word0
    }
    pub fn entry(&self) -> TmnfMeshListEntry {
        TmnfMeshListEntry { node: self.word0, skip: self.word1 }
    }
}

pub const TMNF_MESH_GRID_LEVELS: usize = 2;

#[derive(Clone, Debug, Default)]
pub struct TmnfMeshGridLevel {
    pub origin: [f64; 3],
    pub cell_size: f64,
    pub inv_cell_size: f64,
    pub max_half: f64,
    pub dims: [u32; 3],
    pub cell_count: u32,
    /// Header slot per cell (index into `pool` of the owning grid).
    pub cells: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct TmnfMeshGrid {
    pub levels: [TmnfMeshGridLevel; TMNF_MESH_GRID_LEVELS],
    pub pool: Vec<TmnfMeshPoolSlot>,
    /// Per face, immutable.
    pub sphere_edges: Vec<TmnfSphereFaceEdges>,
}

/// Optional per-query acceleration for a static mesh.
#[derive(Clone, Copy)]
pub struct TmnfMeshQueryAccel<'a> {
    pub inverse_iso: Option<&'a GmIso4>,
    pub mesh_grid: Option<&'a TmnfMeshGrid>,
}

/// A located surface pair member.
#[derive(Clone, Copy)]
pub struct LocatedGmSurf<'a> {
    pub surf: &'a GmSurfGeom,
    pub iso: &'a GmIso4,
    pub is_located: u32,
    pub accel: Option<&'a TmnfMeshQueryAccel<'a>>,
}

/* ---------------------------------------------------------------------------
 * Static grid (region-restricted static tree copies)
 * ------------------------------------------------------------------------- */

pub const TMNF_CELL_NODE_INTERNAL: u32 = u32::MAX;
pub const TMNF_CELL_NODE_EMPTY: u32 = u32::MAX - 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfStaticCellNode {
    pub box_aligned: GmBoxAligned,
    pub skip: u32,
    pub entry_index: u32,
}
const _: () = assert!(size_of::<TmnfStaticCellNode>() == 32);

#[derive(Clone, Debug, Default)]
pub struct TmnfStaticGrid {
    pub origin: [f64; 3],
    pub cell_size: f64,
    pub inv_cell_size: f64,
    /// Query box half-extent budget, E.
    pub expand: f64,
    /// Float rounding guard, delta.
    pub margin: f64,
    pub dims: [u32; 3],
    pub cell_count: u32,
    pub cell_offsets: Vec<u32>,
    pub cell_counts: Vec<u32>,
    pub nodes: Vec<TmnfStaticCellNode>,
    pub mesh_pool: Vec<TmnfMeshPoolSlot>,
    /// One per track mesh.
    pub mesh_grids: Vec<TmnfMeshGrid>,
    /// Per static entry: grid index or None.
    pub entry_mesh_grids: Vec<Option<usize>>,
    /// One per static entry.
    pub entry_inverse_isos: Vec<GmIso4>,
}

/* ---------------------------------------------------------------------------
 * Plug surfaces and trees
 * ------------------------------------------------------------------------- */

#[derive(Clone, Debug)]
pub struct CPlugSurface {
    pub geom: GmSurfGeom,
    pub material_ids: Vec<u8>,
    pub material_count: u32,
}

/// A collision tree node. Children are arena indices in the owning
/// [`TreeArena`]; the game's `contact_buffer` lives beside the tree there.
#[derive(Clone, Debug)]
pub struct CPlugTree {
    pub object_ref: u32,
    pub flags: u32,
    pub box_aligned: GmBoxAligned,
    pub local_iso: GmIso4,
    pub surface: Option<CPlugSurface>,
    pub children: Vec<usize>,
}

/// Flat arena of the world's (dynamic) collision trees with their lazily
/// created per-tree merge buffers.
#[derive(Clone, Debug, Default)]
pub struct TreeArena {
    pub trees: Vec<CPlugTree>,
    pub contact_buffers: Vec<Option<SHmsSphereBufferContact>>,
}

impl TreeArena {
    pub fn push_tree(&mut self, tree: CPlugTree) -> usize {
        self.trees.push(tree);
        self.contact_buffers.push(None);
        self.trees.len() - 1
    }

    /// 0x00538100 path: lazily create and return the tree's merge buffer.
    pub fn contact_buffer_for(&mut self, tree: usize) -> &mut SHmsSphereBufferContact {
        if self.contact_buffers[tree].is_none() {
            self.contact_buffers[tree] = Some(SHmsSphereBufferContact::new());
        }
        self.contact_buffers[tree].as_mut().unwrap()
    }
}

/// One flattened entry of the static collision tree (the track's statics).
#[derive(Clone, Debug)]
pub struct HmsStaticCollisionEntry {
    pub skip_count: u32,
    pub box_aligned: GmBoxAligned,
    pub iso: GmIso4,
    pub tree_flags: u32,
    pub surface: Option<usize>, /* index into the track's surface array */
    pub tree_ref: u32,
    pub corpus_ref: u32,
}

/* ---------------------------------------------------------------------------
 * Iso helpers
 * ------------------------------------------------------------------------- */

fn iso_identity() -> GmIso4 {
    GmIso4::IDENTITY
}

fn iso_set_nuscale_trans(scale: &GmVec3, translation: &GmVec3) -> GmIso4 {
    let mut iso = GmIso4 {
        m: [0.0; 9],
        t: [translation.x, translation.y, translation.z],
    };
    iso.m[0] = scale.x;
    iso.m[4] = scale.y;
    iso.m[8] = scale.z;
    iso
}

fn iso_mult_inverse(iso: &mut GmIso4, other: &GmIso4) {
    let mut inverse = GmIso4::default();
    inverse.set_inverse(other);
    iso.mult(&inverse);
}

/// The world iso of a tree under a parent iso (flags bit 2 marks a local
/// transform).
pub fn tree_world_iso(tree: &CPlugTree, parent: &GmIso4) -> GmIso4 {
    if (tree.flags & 4u32) == 0 {
        return *parent;
    }
    let mut out = tree.local_iso;
    out.mult(parent);
    out
}

fn transform_direction(vector: &mut GmVec3, iso: &GmIso4) {
    let source = *vector;
    vector.set_mult_mat3(&source, &GmMat3 { m: iso.m });
}

fn relative_iso(first: &LocatedGmSurf, second: &LocatedGmSurf) -> GmIso4 {
    let mut relative = if first.is_located == 0 {
        iso_identity()
    } else {
        *first.iso
    };
    if second.is_located != 0 {
        match second.accel {
            Some(accel) if accel.inverse_iso.is_some() => {
                relative.mult(accel.inverse_iso.unwrap());
            }
            _ => {
                iso_mult_inverse(&mut relative, second.iso);
            }
        }
    }
    relative
}

/// A selected region-restricted node list: entries live at
/// `pool[start .. start + count]` of the owning grid's pool (the C returned a
/// pointer just past the header slot).
#[derive(Clone, Copy, Debug)]
pub struct MeshListNodeList {
    pub start: u32,
    pub count: u32,
}

/// Selects the region-restricted node list of the finest grid level whose
/// budget covers the query box, from the cell holding the box center.
/// `None` means the mesh's full node array. NaN coordinates fail every
/// comparison and take the full array, which rejects them too.
fn select_mesh_list<'a>(
    mesh: &GmSurfMesh,
    accel: Option<&'a TmnfMeshQueryAccel<'a>>,
    mesh_box: &GmBoxAligned,
) -> (Option<MeshListNodeList>, u32) {
    let full_count = mesh.node_count;
    let accel = match accel {
        Some(a) => a,
        None => return (None, full_count),
    };
    let grid = match accel.mesh_grid {
        Some(g) => g,
        None => return (None, full_count),
    };
    let c = [mesh_box.center.x, mesh_box.center.y, mesh_box.center.z];
    let h = [mesh_box.half_extent.x, mesh_box.half_extent.y, mesh_box.half_extent.z];
    for level in &grid.levels {
        let mut cell: u32 = 0;
        let mut a: i32 = 2;
        while a >= 0 {
            let a_u = a as usize;
            if !((h[a_u] as f64) <= level.max_half) {
                break;
            }
            let offset = (c[a_u] as f64) - level.origin[a_u];
            if !(offset >= 0.0 && offset < (level.dims[a_u] as f64) * level.cell_size) {
                break;
            }
            let index = (offset * level.inv_cell_size) as u32;
            if index >= level.dims[a_u] {
                break;
            }
            cell = cell * level.dims[a_u] + index;
            a -= 1;
        }
        if a >= 0 {
            continue;
        }
        let header = level.cells[cell as usize] as usize;
        let count = grid.pool[header].node_count();
        return (
            Some(MeshListNodeList { start: header as u32 + 1, count }),
            count,
        );
    }
    (None, full_count)
}

/* ---------------------------------------------------------------------------
 * Sphere buffer merge
 * ------------------------------------------------------------------------- */

fn gmvec3_is_nearly_equal(this: &GmVec3, other: &GmVec3) -> bool {
    let absolute = other.x.abs();
    let tolerance = (NEARLY_EQUAL_SCALE * (absolute as f64)) as f32;
    let lower = other.x - tolerance;
    let upper = other.x + tolerance;
    if !(lower <= this.x) || !(this.x <= upper) {
        return false;
    }

    let absolute = other.y.abs();
    let tolerance = (NEARLY_EQUAL_SCALE * (absolute as f64)) as f32;
    let lower = other.y - tolerance;
    let upper = other.y + tolerance;
    if !(lower <= this.y) || !(this.y <= upper) {
        return false;
    }

    let absolute = other.z.abs();
    let tolerance = (NEARLY_EQUAL_SCALE * (absolute as f64)) as f32;
    let lower = other.z - tolerance;
    let upper = other.z + tolerance;
    lower <= this.z && this.z <= upper
}

/// 0x00538100  Merges a per-sphere buffer into the destination.
/// UNVALIDATED upstream.
pub fn shms_sphere_buffer_contact_merge_and_add_to_collisions(
    this: &mut SHmsSphereBufferContact,
    destination: &mut CHmsCollisionBuffer,
) {
    let source_count = this.base.collisions.count() as usize;
    let destination_start = destination.collisions.count() as usize;

    for i in 0..source_count {
        let source = *this.base.collisions.at(i);
        if source.collision.flags != 0 {
            destination.collisions.data.push(source);
        }
    }

    let destination_end = destination.collisions.count() as usize;
    for i in 0..source_count {
        let source = *this.base.collisions.at(i);
        if source.collision.flags != 0 {
            continue;
        }

        let mut j = destination_start;
        while j < destination_end {
            let existing = &destination.collisions.data[j];
            if gmvec3_is_nearly_equal(
                &source.collision.face_normal,
                &existing.collision.face_normal,
            ) {
                break;
            }
            let dot = (((source.collision.normal.z * existing.collision.normal.z)
                + (source.collision.normal.x * existing.collision.normal.x))
                + (source.collision.normal.y * existing.collision.normal.y));
            if SPHERE_MERGE_THRESHOLD < dot {
                break;
            }
            j += 1;
        }
        if j == destination_end {
            destination.collisions.data.push(source);
        }
    }

    this.active = 0;
    this.base.collisions.data.clear();
}

/* ---------------------------------------------------------------------------
 * Sphere / sphere
 * ------------------------------------------------------------------------- */

/// 0x008F49D0  Computes sphere/sphere overlap and contact data.
/// UNVALIDATED upstream.
pub fn gm_collision_sphere_sphere(
    sphere1: &LocatedGmSurf,
    sphere2: &LocatedGmSurf,
    buffer: &mut CHmsCollisionBuffer,
) -> i32 {
    let shape1 = sphere1.surf.as_sphere();
    let shape2 = sphere2.surf.as_sphere();

    let mut dx = sphere2.iso.t[0] - sphere1.iso.t[0];
    let mut dy = sphere2.iso.t[1] - sphere1.iso.t[1];
    let mut dz = sphere2.iso.t[2] - sphere1.iso.t[2];
    let distance_squared = ((dx * dx) + (dy * dy)) + (dz * dz);
    /* Load-bearing f64: the radius sum and its square are computed in double
     * and compared in double (never stored to float). */
    let radius_sum_x87 = (shape2.radius as f64) + (shape1.radius as f64);
    let radius_squared = radius_sum_x87 * radius_sum_x87;
    if !((distance_squared as f64) < radius_squared) {
        return 0;
    }

    let distance = distance_squared.sqrt();
    if distance <= 1.0e-5f32 {
        let c = buffer.add_collision();
        c.normal.x = 0.0;
        c.normal.y = -1.0;
        c.normal.z = 0.0;
        c.separation.x = 0.0;
        c.separation.y = shape2.radius;
        c.separation.z = 0.0;
        c.position.x = sphere1.iso.t[0];
        c.position.y = sphere1.iso.t[1];
        c.position.z = sphere1.iso.t[2];
    } else {
        let inverse_distance = 1.0f32 / distance;
        dx = inverse_distance * dx;
        dy = dy * inverse_distance;
        dz = inverse_distance * dz;

        let c = buffer.add_collision();
        c.normal.x = -dx;
        c.normal.y = -dy;
        c.normal.z = -dz;

        let penetration = (shape2.radius + shape1.radius) - distance;
        c.separation.x = penetration * dx;
        c.separation.y = dy * penetration;
        c.separation.z = penetration * dz;

        let radius1 = shape1.radius;
        c.position.x = radius1 * dx;
        c.position.y = dy * radius1;
        c.position.z = radius1 * dz;
        c.position.x = sphere1.iso.t[0] + c.position.x;
        c.position.y = sphere1.iso.t[1] + c.position.y;
        c.position.z = sphere1.iso.t[2] + c.position.z;
    }
    let c = buffer.collisions.data.last_mut().unwrap();
    c.collision.material1 = shape1.base.material_index;
    c.collision.material2 = shape2.base.material_index;
    1
}

/* ---------------------------------------------------------------------------
 * Sphere / mesh
 * ------------------------------------------------------------------------- */

fn finish_mesh_collision(
    collision: &mut GmCollision,
    first: &GmSurf,
    face: &GmSurfMeshFace,
    position: &GmVec3,
    normal: &GmVec3,
    separation: &GmVec3,
    flags: u32,
) {
    collision.normal = *normal;
    collision.separation = *separation;
    collision.position = *position;
    collision.material1 = first.material_index;
    collision.material2 = face.material_index;
    collision.flags = flags;
    collision.face_normal = face.normal;
}

/// Sphere feature contact. dx/dy/dz is center - feature and distance is the
/// value the game uses as |center - feature| for this branch; the end-vertex
/// branch passes sqrt(sqrt(feature_squared)).
fn add_sphere_feature_collision(
    sphere: &GmSurfSphere,
    face: &GmSurfMeshFace,
    position: &GmVec3,
    dx: f32,
    dy: f32,
    dz: f32,
    distance: f32,
    buffer: &mut CHmsCollisionBuffer,
) {
    let inverse_distance = 1.0f32 / distance;
    let normal = GmVec3 {
        x: inverse_distance * dx,
        y: dy * inverse_distance,
        z: inverse_distance * dz,
    };
    let scale = (distance - sphere.radius) * inverse_distance;
    let radial_separation = GmVec3 {
        x: scale * dx,
        y: dy * scale,
        z: scale * dz,
    };
    let plane_separation = ((face.normal.x * radial_separation.x)
        + (radial_separation.y * face.normal.y))
        + (radial_separation.z * face.normal.z);
    let separation = GmVec3 {
        x: plane_separation * face.normal.x,
        y: plane_separation * face.normal.y,
        z: plane_separation * face.normal.z,
    };
    let collision = buffer.add_collision();
    finish_mesh_collision(
        collision, &sphere.base, face, position, &normal, &separation, 0,
    );
}

/// 0x008EA2D0  Computes sphere contacts against a triangle mesh.
///
/// Validated against the DesertA1 sphere/mesh trace. The three feature
/// branches differ in the game and are reproduced as-is:
///  - start vertex: reject when feature_squared > r*r or <= 1e-10,
///    distance = sqrt(feature_squared);
///  - end vertex: d = sqrt(feature_squared); reject when d > r*r or
///    d <= 1e-10, distance = sqrt(d) (the game square-roots twice);
///  - edge closest point: reject when feature_squared <= 1e-5, no radius
///    test, distance = sqrt(feature_squared).
pub fn gm_collision_sphere_mesh(
    sphere_located: &LocatedGmSurf,
    mesh_located: &LocatedGmSurf,
    buffer: &mut CHmsCollisionBuffer,
) -> i32 {
    /* 0x00D1A938: 1e-10, the edge-length and vertex-distance floor. */
    let sphere_tiny_squared = TINY_SQUARED;
    let sphere = sphere_located.surf.as_sphere();
    let mesh = mesh_located.surf.as_mesh();
    let cached_edges = match mesh_located.accel {
        Some(accel) if accel.mesh_grid.is_some() => {
            Some(&accel.mesh_grid.unwrap().sphere_edges)
        }
        _ => None,
    };
    let relative = relative_iso(sphere_located, mesh_located);

    let start = buffer.get_count();
    let sphere_box = GmBoxAligned {
        center: GmVec3::ZERO,
        half_extent: GmVec3 {
            x: sphere.radius,
            y: sphere.radius,
            z: sphere.radius,
        },
    };
    let mut transformed_box = GmBoxAligned::default();
    transformed_box.set_mult(&sphere_box, &relative);
    let center = transformed_box.center;
    let mut result = 0;

    let (list, node_count) = select_mesh_list(mesh, mesh_located.accel, &transformed_box);
    let list_pool = mesh_located.accel
        .and_then(|a| a.mesh_grid)
        .map(|g| &g.pool);
    let mut node_index: u32 = 0;
    while node_index < node_count {
        let (node_ref, skip_count): (usize, u32) = match (list, list_pool) {
            (Some(l), Some(pool)) => {
                let e = pool[(l.start + node_index) as usize].entry();
                (e.node as usize, e.skip)
            }
            _ => {
                let n = &mesh.nodes[node_index as usize];
                (node_index as usize, n.skip_count)
            }
        };
        let node = &mesh.nodes[node_ref];
        if !transformed_box.test_inter(&node.box_aligned) {
            node_index += skip_count;
            continue;
        }
        if node.face_index == u32::MAX {
            node_index += 1;
            continue;
        }

        let face = &mesh.faces[node.face_index as usize];
        let vertices = [
            mesh.vertices[face.vertex[0] as usize],
            mesh.vertices[face.vertex[1] as usize],
            mesh.vertices[face.vertex[2] as usize],
        ];
        let distance = (((center.y - vertices[0].y) * face.normal.y)
            + (face.normal.x * (center.x - vertices[0].x)))
            + ((center.z - vertices[0].z) * face.normal.z);
        if !(distance <= sphere.radius) || !(0.0f32 <= distance) {
            node_index += 1;
            continue;
        }

        let section_squared = (sphere.radius * sphere.radius) - (distance * distance);
        let section_radius = section_squared.sqrt();
        let negative_distance = -distance;
        let projected = GmVec3 {
            x: (face.normal.x * negative_distance) + center.x,
            y: (face.normal.y * negative_distance) + center.y,
            z: (negative_distance * face.normal.z) + center.z,
        };

        let mut emitted = 0;
        for edge_index in 0..3usize {
            let next_index = if edge_index == 2 { 0 } else { edge_index + 1 };
            let vertex = &vertices[edge_index];
            let next = &vertices[next_index];
            let prepared = match cached_edges {
                Some(cache) => cache[node.face_index as usize].edges[edge_index],
                None => tmnf_sphere_mesh_edge_prepare(vertex, next, &face.normal),
            };
            let edge = prepared.direction;
            let side = prepared.side;
            let from_vertex = GmVec3 {
                x: projected.x - vertex.x,
                y: projected.y - vertex.y,
                z: projected.z - vertex.z,
            };
            let side_distance = ((from_vertex.x * side.x) + (from_vertex.y * side.y))
                + (from_vertex.z * side.z);
            if section_radius < side_distance {
                emitted = -1;
                break;
            }
            if !(0.0f32 < side_distance) {
                continue;
            }

            let along_start = ((from_vertex.x * edge.x) + (from_vertex.y * edge.y))
                + (from_vertex.z * edge.z);
            let feature: &GmVec3;
            let mut closest = GmVec3::ZERO;
            let mut is_edge = false;
            let mut is_end_vertex = false;
            if 0.0f32 <= along_start {
                let from_next = GmVec3 {
                    x: projected.x - next.x,
                    y: projected.y - next.y,
                    z: projected.z - next.z,
                };
                let along_end = ((from_next.x * edge.x) + (from_next.y * edge.y))
                    + (from_next.z * edge.z);
                if along_end <= 0.0f32 {
                    let mv = -side_distance;
                    closest.x = (mv * side.x) + projected.x;
                    closest.y = projected.y + (side.y * mv);
                    closest.z = (mv * side.z) + projected.z;
                    feature = &closest;
                    is_edge = true;
                } else {
                    feature = next;
                    is_end_vertex = true;
                }
            } else {
                feature = vertex;
            }

            let dx = center.x - feature.x;
            let dy = center.y - feature.y;
            let dz = center.z - feature.z;
            let feature_squared =
                ((dx * dx) + (dy * dy)) + (dz * dz);
            let feature_distance;
            if is_edge {
                if feature_squared <= 1.0e-5f32 {
                    emitted = -1;
                    break;
                }
                feature_distance = feature_squared.sqrt();
            } else {
                let radius_squared = sphere.radius * sphere.radius;
                let tested = if is_end_vertex {
                    feature_squared.sqrt()
                } else {
                    feature_squared
                };
                if !(tested <= radius_squared) || tested <= sphere_tiny_squared {
                    emitted = -1;
                    break;
                }
                feature_distance = tested.sqrt();
            }
            add_sphere_feature_collision(
                sphere, face, feature, dx, dy, dz, feature_distance, buffer,
            );
            result = 1;
            emitted = 1;
            break;
        }

        if emitted == 0 && 0.0f32 < distance {
            let depth = distance - sphere.radius;
            let separation = GmVec3 {
                x: depth * face.normal.x,
                y: depth * face.normal.y,
                z: depth * face.normal.z,
            };
            let collision = buffer.add_collision();
            finish_mesh_collision(
                collision, &sphere.base, face, &projected, &face.normal,
                &separation, 1,
            );
            result = 1;
        }
        node_index += 1;
    }

    if result != 0 {
        let end = buffer.get_count();
        for i in start..end {
            let c = buffer.get_collision(i as usize);
            transform_direction(&mut c.normal, mesh_located.iso);
            transform_direction(&mut c.separation, mesh_located.iso);
            c.position.mult_iso4(mesh_located.iso);
        }
    }
    result
}

/* ---------------------------------------------------------------------------
 * Ellipsoid / mesh
 * ------------------------------------------------------------------------- */

fn collision_dot3(a: &GmVec3, b: &GmVec3) -> f32 {
    ((a.x * b.x) + (a.y * b.y)) + (a.z * b.z)
}

fn collision_length_squared(value: &GmVec3) -> f32 {
    collision_dot3(value, value)
}

fn make_ellipsoid_inverse(
    relative: &GmIso4,
    radii: &GmVec3,
) -> (GmIso4, GmIso4) {
    /* (scaled_inverse, inverse) */
    let mut inverse = GmIso4::default();
    inverse.set_inverse(relative);
    let mut scaled_inverse = inverse;

    let inverse_x = 1.0f32 / radii.x;
    scaled_inverse.m[0] = inverse_x * scaled_inverse.m[0];
    scaled_inverse.m[1] = scaled_inverse.m[1] * inverse_x;
    scaled_inverse.m[2] = inverse_x * scaled_inverse.m[2];
    scaled_inverse.t[0] = scaled_inverse.t[0] * inverse_x;

    let inverse_y = 1.0f32 / radii.y;
    scaled_inverse.m[3] = inverse_y * scaled_inverse.m[3];
    scaled_inverse.m[4] = scaled_inverse.m[4] * inverse_y;
    scaled_inverse.m[5] = inverse_y * scaled_inverse.m[5];
    scaled_inverse.t[1] = scaled_inverse.t[1] * inverse_y;

    let inverse_z = 1.0f32 / radii.z;
    scaled_inverse.m[6] = inverse_z * scaled_inverse.m[6];
    scaled_inverse.m[7] = scaled_inverse.m[7] * inverse_z;
    scaled_inverse.m[8] = inverse_z * scaled_inverse.m[8];
    scaled_inverse.t[2] = scaled_inverse.t[2] * inverse_z;
    (scaled_inverse, inverse)
}

fn add_ellipsoid_feature_collision(
    ellipsoid: &GmSurfEllipsoid,
    face: &GmSurfMeshFace,
    point: &GmVec3,
    distance: f32,
    face_normal: &GmVec3,
    buffer: &mut CHmsCollisionBuffer,
) {
    let inverse_distance = 1.0f32 / distance;
    let mut radial = GmVec3 {
        x: -point.x,
        y: -point.y,
        z: -point.z,
    };
    let normal = GmVec3 {
        x: inverse_distance * radial.x,
        y: radial.y * inverse_distance,
        z: radial.z * inverse_distance,
    };
    let scale = (distance - 1.0f32) * inverse_distance;
    radial.x = scale * radial.x;
    radial.y = radial.y * scale;
    radial.z = scale * radial.z;
    let plane_scale = collision_dot3(face_normal, &radial);
    let separation = GmVec3 {
        x: plane_scale * face_normal.x,
        y: face_normal.y * plane_scale,
        z: plane_scale * face_normal.z,
    };

    let collision = buffer.add_collision();
    finish_mesh_collision(
        collision, &ellipsoid.base, face, point, &normal, &separation, 0,
    );
    collision.face_normal = *face_normal;
}

/// The two output isos the game rebuilds for every emitting face; built once
/// on first emission and reused within one query.
#[derive(Clone, Copy, Default)]
struct EllipsoidOutputIsos {
    position_iso: GmIso4,
    normal_iso: GmIso4,
    ready: bool,
}

fn prepare_ellipsoid_output_isos(
    isos: &mut EllipsoidOutputIsos,
    ellipsoid: &GmSurfEllipsoid,
    inverse_relative: &GmIso4,
    mesh_iso: &GmIso4,
) {
    if isos.ready {
        return;
    }
    let radii = &ellipsoid.radii;
    let inverse_x = 1.0f32 / radii.x;
    let inverse_y = 1.0f32 / radii.y;
    let inverse_z = 1.0f32 / radii.z;
    let zero = GmVec3::ZERO;

    let mut position_iso = iso_set_nuscale_trans(radii, &zero);
    iso_mult_inverse(&mut position_iso, inverse_relative);
    position_iso.mult(mesh_iso);

    let inverse_radii = GmVec3 {
        x: inverse_x,
        y: inverse_y,
        z: inverse_z,
    };
    let mut normal_iso = iso_set_nuscale_trans(&inverse_radii, &zero);
    iso_mult_inverse(&mut normal_iso, inverse_relative);
    normal_iso.mult(mesh_iso);

    isos.position_iso = position_iso;
    isos.normal_iso = normal_iso;
    isos.ready = true;
}

fn transform_ellipsoid_collisions(
    buffer: &mut CHmsCollisionBuffer,
    start: u32,
    isos: &EllipsoidOutputIsos,
) {
    let position_iso = isos.position_iso;
    let normal_iso = isos.normal_iso;
    let end = buffer.get_count();
    for i in start..end {
        let c = buffer.get_collision(i as usize);
        c.position.mult_iso4(&position_iso);
        transform_direction(&mut c.normal, &normal_iso);
        let normal_squared = collision_length_squared(&c.normal);
        if TINY_SQUARED < normal_squared {
            let length = normal_squared.sqrt();
            let inverse_length = 1.0f32 / length;
            c.normal.x = inverse_length * c.normal.x;
            c.normal.y = inverse_length * c.normal.y;
            c.normal.z = inverse_length * c.normal.z;
        }
        transform_direction(&mut c.separation, &position_iso);
    }
}

/// One ellipsoid/mesh query's fixed state.
struct EllipsoidMeshQuery<'a> {
    ellipsoid: &'a GmSurfEllipsoid,
    mesh: &'a GmSurfMesh,
    mesh_iso: &'a GmIso4,
    inverse_relative: GmIso4,
    scaled_inverse: GmIso4,
    mesh_box: GmBoxAligned,
    list: Option<MeshListNodeList>,
    node_count: u32,
    pool: Option<&'a Vec<TmnfMeshPoolSlot>>,
    output_isos: EllipsoidOutputIsos,
}

fn ellipsoid_mesh_begin<'a>(
    ellipsoid_located: &'a LocatedGmSurf<'a>,
    mesh_located: &'a LocatedGmSurf<'a>,
) -> EllipsoidMeshQuery<'a> {
    let ellipsoid = ellipsoid_located.surf.as_ellipsoid();
    let mesh = mesh_located.surf.as_mesh();
    let mesh_iso = mesh_located.iso;
    let relative = relative_iso(ellipsoid_located, mesh_located);

    let source_box = GmBoxAligned {
        center: GmVec3::ZERO,
        half_extent: ellipsoid.radii,
    };
    let mut mesh_box = GmBoxAligned::default();
    mesh_box.set_mult(&source_box, &relative);
    let (scaled_inverse, inverse_relative) =
        make_ellipsoid_inverse(&relative, &ellipsoid.radii);
    let (list, node_count) = select_mesh_list(mesh, mesh_located.accel, &mesh_box);
    let pool = mesh_located.accel
        .and_then(|a| a.mesh_grid)
        .map(|g| &g.pool);
    EllipsoidMeshQuery {
        ellipsoid,
        mesh,
        mesh_iso,
        inverse_relative,
        scaled_inverse,
        mesh_box,
        list,
        node_count,
        pool,
        output_isos: EllipsoidOutputIsos::default(),
    }
}

/// Tests one leaf whose box passed and appends its contact, if any.
fn ellipsoid_mesh_face(
    q: &mut EllipsoidMeshQuery,
    node: &GmSurfMeshNode,
    buffer: &mut CHmsCollisionBuffer,
) -> i32 {
    let tiny_squared = TINY_SQUARED;
    let edge_closest_squared = EDGE_CLOSEST_SQUARED;
    let ellipsoid = q.ellipsoid;
    let mesh = q.mesh;
    let scaled_inverse = q.scaled_inverse;
    let face = &mesh.faces[node.face_index as usize];
    let mut vertices = [GmVec3::ZERO; 3];
    for i in 0..3 {
        vertices[i] = mesh.vertices[face.vertex[i] as usize];
        vertices[i].mult_iso4(&scaled_inverse);
    }

    let edge01 = GmVec3 {
        x: vertices[1].x - vertices[0].x,
        y: vertices[1].y - vertices[0].y,
        z: vertices[1].z - vertices[0].z,
    };
    let edge02 = GmVec3 {
        x: vertices[2].x - vertices[0].x,
        y: vertices[2].y - vertices[0].y,
        z: vertices[2].z - vertices[0].z,
    };
    let mut face_normal = GmVec3 {
        x: (edge01.y * edge02.z) - (edge01.z * edge02.y),
        y: (edge02.x * edge01.z) - (edge01.x * edge02.z),
        z: (edge02.y * edge01.x) - (edge01.y * edge02.x),
    };
    let normal_squared = collision_length_squared(&face_normal);
    if !(tiny_squared < normal_squared) {
        return 0;
    }
    let normal_length = normal_squared.sqrt();
    let inverse_normal_length = 1.0f32 / normal_length;
    face_normal.x = inverse_normal_length * face_normal.x;
    face_normal.y = face_normal.y * inverse_normal_length;
    face_normal.z = inverse_normal_length * face_normal.z;

    let collision_start = buffer.get_count();
    let negative_vertex0 = GmVec3 {
        x: -vertices[0].x,
        y: -vertices[0].y,
        z: -vertices[0].z,
    };
    let plane_distance = collision_dot3(&negative_vertex0, &face_normal);
    if 1.0f32 < plane_distance || plane_distance < 0.0f32 {
        return 0;
    }

    let radius_squared = 1.0f32 * 1.0f32;
    let section_squared = radius_squared - (plane_distance * plane_distance);
    let section_radius = section_squared.sqrt();
    let negative_distance = -plane_distance;
    let projected = GmVec3 {
        x: (face_normal.x * negative_distance) + 0.0f32,
        y: (face_normal.y * negative_distance) + 0.0f32,
        z: (face_normal.z * negative_distance) + 0.0f32,
    };

    let mut reject_face = false;
    let mut emitted = false;
    for edge_index in 0..3usize {
        let next_index = if edge_index == 2 { 0 } else { edge_index + 1 };
        let start = &vertices[edge_index];
        let end = &vertices[next_index];
        let mut edge = GmVec3 {
            x: end.x - start.x,
            y: end.y - start.y,
            z: end.z - start.z,
        };
        let edge_squared = collision_length_squared(&edge);
        if tiny_squared < edge_squared {
            let edge_length = edge_squared.sqrt();
            let inverse_edge_length = 1.0f32 / edge_length;
            edge.x = inverse_edge_length * edge.x;
            edge.y = edge.y * inverse_edge_length;
            edge.z = inverse_edge_length * edge.z;
        }

        let side = GmVec3 {
            x: (edge.y * face_normal.z) - (edge.z * face_normal.y),
            y: (edge.z * face_normal.x) - (face_normal.z * edge.x),
            z: (face_normal.y * edge.x) - (edge.y * face_normal.x),
        };
        let from_start = GmVec3 {
            x: projected.x - start.x,
            y: projected.y - start.y,
            z: projected.z - start.z,
        };
        let side_distance = collision_dot3(&from_start, &side);
        if section_radius < side_distance {
            reject_face = true;
            break;
        }
        if !(0.0f32 < side_distance) {
            continue;
        }

        let along_start = collision_dot3(&from_start, &edge);
        let feature: &GmVec3;
        let mut closest = GmVec3::ZERO;
        let mut is_edge = false;
        let mut is_end_vertex = false;
        if 0.0f32 <= along_start {
            let from_end = GmVec3 {
                x: projected.x - end.x,
                y: projected.y - end.y,
                z: projected.z - end.z,
            };
            let along_end = collision_dot3(&from_end, &edge);
            if along_end <= 0.0f32 {
                let mv = -side_distance;
                closest.x = (mv * side.x) + projected.x;
                closest.y = projected.y + (side.y * mv);
                closest.z = (mv * side.z) + projected.z;
                feature = &closest;
                is_edge = true;
            } else {
                feature = end;
                is_end_vertex = true;
            }
        } else {
            feature = start;
        }

        let feature_squared = collision_length_squared(feature);
        if is_edge {
            if feature_squared <= edge_closest_squared {
                reject_face = true;
                break;
            }
        } else if 1.0f32 < feature_squared || feature_squared <= tiny_squared {
            reject_face = true;
            break;
        }

        let mut feature_distance = feature_squared.sqrt();
        if is_end_vertex {
            /* Retain the game's double square root for the end vertex. */
            feature_distance = feature_distance.sqrt();
        }
        add_ellipsoid_feature_collision(
            ellipsoid, face, feature, feature_distance, &face_normal, buffer,
        );
        emitted = true;
        break;
    }

    if !reject_face && !emitted && 0.0f32 < plane_distance {
        let depth = plane_distance - 1.0f32;
        let separation = GmVec3 {
            x: depth * face_normal.x,
            y: face_normal.y * depth,
            z: depth * face_normal.z,
        };
        let collision = buffer.add_collision();
        finish_mesh_collision(
            collision, &ellipsoid.base, face, &projected, &face_normal,
            &separation, 1,
        );
        collision.face_normal = face_normal;
        emitted = true;
    }

    if emitted {
        prepare_ellipsoid_output_isos(
            &mut q.output_isos, ellipsoid, &q.inverse_relative, q.mesh_iso,
        );
        transform_ellipsoid_collisions(buffer, collision_start, &q.output_isos);
    }
    emitted as i32
}

/// 0x008EADC0  Computes ellipsoid contacts against a triangle mesh.
/// VALIDATED upstream: 128/128 graph-complete golden records.
pub fn gm_collision_ellipsoid_mesh(
    ellipsoid_located: &LocatedGmSurf,
    mesh_located: &LocatedGmSurf,
    buffer: &mut CHmsCollisionBuffer,
) -> i32 {
    let mut q = ellipsoid_mesh_begin(ellipsoid_located, mesh_located);

    let mut result = 0;
    let mut node_index: u32 = 0;
    while node_index < q.node_count {
        let (node_ref, skip_count): (usize, u32) = match (q.list, q.pool) {
            (Some(l), Some(pool)) => {
                let e = pool[(l.start + node_index) as usize].entry();
                (e.node as usize, e.skip)
            }
            _ => {
                let n = &q.mesh.nodes[node_index as usize];
                (node_index as usize, n.skip_count)
            }
        };
        // Copy the node: the query struct is borrowed mutably by the kernel.
        let node = q.mesh.nodes[node_ref];
        if !q.mesh_box.test_inter(&node.box_aligned) {
            node_index += skip_count;
            continue;
        }
        if node.face_index != u32::MAX {
            result |= ellipsoid_mesh_face(&mut q, &node, buffer);
        }
        node_index += 1;
    }
    result
}

/* ---------------------------------------------------------------------------
 * Box / mesh
 * ------------------------------------------------------------------------- */

fn min3(a: f32, b: f32, c: f32) -> f32 {
    let mut result = a;
    if b < result {
        result = b;
    }
    if c < result {
        result = c;
    }
    result
}

fn max3(a: f32, b: f32, c: f32) -> f32 {
    let mut result = a;
    if result < b {
        result = b;
    }
    if result < c {
        result = c;
    }
    result
}

fn axis_interval_overlap(a: f32, b: f32, center: f32, radius: f32) -> bool {
    let mut minimum = a;
    let mut maximum = a;
    if b < minimum {
        minimum = b;
    }
    if maximum < b {
        maximum = b;
    }
    minimum <= radius + center && center - radius <= maximum
}

fn box_triangle_overlap(box_: &GmSurfBox, vertices: &[GmVec3; 3]) -> bool {
    let x0 = vertices[0].x - box_.center.x;
    let x1 = vertices[1].x - box_.center.x;
    let x2 = vertices[2].x - box_.center.x;
    if !(min3(x0, x1, x2) <= box_.half_extent.x)
        || !(-box_.half_extent.x <= max3(x0, x1, x2))
    {
        return false;
    }

    let y0 = vertices[0].y - box_.center.y;
    let y1 = vertices[1].y - box_.center.y;
    let y2 = vertices[2].y - box_.center.y;
    if !(min3(y0, y1, y2) <= box_.half_extent.y)
        || !(-box_.half_extent.y <= max3(y0, y1, y2))
    {
        return false;
    }

    let z0 = vertices[0].z - box_.center.z;
    let z1 = vertices[1].z - box_.center.z;
    let z2 = vertices[2].z - box_.center.z;
    if !(min3(z0, z1, z2) <= box_.half_extent.z)
        || !(-box_.half_extent.z <= max3(z0, z1, z2))
    {
        return false;
    }

    let edge0 = GmVec3 {
        x: x1 - x0,
        y: y1 - y0,
        z: z1 - z0,
    };
    let edge1 = GmVec3 {
        x: x2 - x1,
        y: y2 - y1,
        z: z2 - z1,
    };
    let triangle_normal = GmVec3 {
        x: (edge0.y * edge1.z) - (edge0.z * edge1.y),
        y: (edge1.x * edge0.z) - (edge0.x * edge1.z),
        z: (edge1.y * edge0.x) - (edge1.x * edge0.y),
    };
    let plane = -(((triangle_normal.x * x0) + (triangle_normal.y * y0))
        + (triangle_normal.z * z0));
    let mut positive = GmVec3::ZERO;
    let mut negative = GmVec3::ZERO;
    let extents = [
        box_.half_extent.x,
        box_.half_extent.y,
        box_.half_extent.z,
    ];
    let normal = [
        triangle_normal.x,
        triangle_normal.y,
        triangle_normal.z,
    ];
    let positive_values = [&mut positive.x, &mut positive.y, &mut positive.z];
    let negative_values = [&mut negative.x, &mut negative.y, &mut negative.z];
    for axis in 0..3 {
        if normal[axis] <= 0.0f32 {
            *positive_values[axis] = extents[axis];
            *negative_values[axis] = -extents[axis];
        } else {
            *positive_values[axis] = -extents[axis];
            *negative_values[axis] = extents[axis];
        }
    }
    let positive_plane = (((positive.x * triangle_normal.x)
        + (positive.y * triangle_normal.y))
        + (positive.z * triangle_normal.z))
        + plane;
    let negative_plane = (((negative.x * triangle_normal.x)
        + (negative.y * triangle_normal.y))
        + (negative.z * triangle_normal.z))
        + plane;
    if !(positive_plane <= 0.0f32) || !(0.0f32 <= negative_plane) {
        return false;
    }

    let edges = [
        edge0,
        edge1,
        GmVec3 {
            x: x0 - x2,
            y: y0 - y2,
            z: z0 - z2,
        },
    ];
    let centered = [
        GmVec3 { x: x0, y: y0, z: z0 },
        GmVec3 { x: x1, y: y1, z: z1 },
        GmVec3 { x: x2, y: y2, z: z2 },
    ];

    for edge_index in 0..3usize {
        let edge = &edges[edge_index];
        let base = &centered[edge_index];
        let opposite = &centered[(edge_index + 2) % 3];
        let mut projections = [0.0f32; 2];
        let mut radius;

        projections[0] = (edge.z * base.y) - (edge.y * base.z);
        projections[1] = (edge.z * opposite.y) - (edge.y * opposite.z);
        radius = (box_.half_extent.y * edge.z.abs())
            + (box_.half_extent.z * edge.y.abs());
        if !axis_interval_overlap(projections[0], projections[1], 0.0, radius) {
            return false;
        }

        projections[0] = (edge.x * base.z) - (edge.z * base.x);
        projections[1] = (edge.x * opposite.z) - (edge.z * opposite.x);
        radius = (box_.half_extent.z * edge.x.abs())
            + (box_.half_extent.x * edge.z.abs());
        if !axis_interval_overlap(projections[0], projections[1], 0.0, radius) {
            return false;
        }

        projections[0] = (edge.y * base.x) - (edge.x * base.y);
        projections[1] = (edge.y * opposite.x) - (edge.x * opposite.y);
        radius = (box_.half_extent.y * edge.x.abs())
            + (box_.half_extent.x * edge.y.abs());
        if !axis_interval_overlap(projections[0], projections[1], 0.0, radius) {
            return false;
        }
    }
    true
}

/// 0x008F5200  Tests an oriented box against a triangle mesh.
/// UNVALIDATED upstream.
pub fn gm_collision_box_mesh(
    box_located: &LocatedGmSurf,
    mesh_located: &LocatedGmSurf,
    buffer: &mut CHmsCollisionBuffer,
) -> i32 {
    let box_ = box_located.surf.as_box();
    let mesh = mesh_located.surf.as_mesh();
    let relative = relative_iso(box_located, mesh_located);

    let source_box = GmBoxAligned {
        center: box_.center,
        half_extent: box_.half_extent,
    };
    let mut mesh_box = GmBoxAligned::default();
    mesh_box.set_mult(&source_box, &relative);

    let mut node_index: u32 = 0;
    while node_index < mesh.node_count {
        let node = &mesh.nodes[node_index as usize];
        if !mesh_box.test_inter(&node.box_aligned) {
            node_index += node.skip_count;
            continue;
        }
        if node.face_index == u32::MAX {
            node_index += 1;
            continue;
        }

        let face = &mesh.faces[node.face_index as usize];
        let mut vertices = [GmVec3::ZERO; 3];
        for i in 0..3 {
            let source = mesh.vertices[face.vertex[i] as usize];
            vertices[i].x = source.x - relative.t[0];
            vertices[i].y = source.y - relative.t[1];
            vertices[i].z = source.z - relative.t[2];
            vertices[i].mult_transpose(&GmMat3 { m: relative.m });
        }
        if !box_triangle_overlap(box_, &vertices) {
            node_index += 1;
            continue;
        }

        let c = buffer.add_collision();
        c.separation.x = 0.0;
        c.separation.y = 0.0;
        c.separation.z = 0.0;
        let mut position = mesh.vertices[face.vertex[0] as usize];
        position.mult_iso4(mesh_located.iso);
        c.position = position;
        c.normal = face.normal;
        transform_direction(&mut c.normal, mesh_located.iso);
        c.material1 = box_.base.material_index;
        c.material2 = face.material_index;
        return 1;
    }
    0
}

/* ---------------------------------------------------------------------------
 * Dispatch
 * ------------------------------------------------------------------------- */

pub type GmCollisionHandler =
    fn(&LocatedGmSurf, &LocatedGmSurf, &mut CHmsCollisionBuffer) -> i32;

/// The shape dispatch table (`CollisionRuntime`). Only the three mesh
/// handlers are populated, exactly as `CollisionRuntime_Init` does.
#[derive(Clone, Copy)]
pub struct CollisionShapeDispatch {
    pub sphere_box: Option<GmCollisionHandler>,
    pub sphere_ellipsoid: Option<GmCollisionHandler>,
    pub sphere_polygon: Option<GmCollisionHandler>,
    pub ellipsoid_polygon: Option<GmCollisionHandler>,
    pub sphere_mesh: Option<GmCollisionHandler>,
    pub ellipsoid_mesh: Option<GmCollisionHandler>,
    pub box_box: Option<GmCollisionHandler>,
    pub box_mesh: Option<GmCollisionHandler>,
    pub mesh_mesh: Option<GmCollisionHandler>,
}

impl Default for CollisionShapeDispatch {
    fn default() -> Self {
        CollisionShapeDispatch {
            sphere_box: None,
            sphere_ellipsoid: None,
            sphere_polygon: None,
            ellipsoid_polygon: None,
            sphere_mesh: Some(gm_collision_sphere_mesh),
            ellipsoid_mesh: Some(gm_collision_ellipsoid_mesh),
            box_box: None,
            box_mesh: Some(gm_collision_box_mesh),
            mesh_mesh: None,
        }
    }
}

fn resolved_handler(
    type1: u8,
    type2: u8,
    runtime: &CollisionShapeDispatch,
) -> Option<GmCollisionHandler> {
    if type1 == GM_SURF_SPHERE && type2 == GM_SURF_BOX {
        runtime.sphere_box
    } else if type1 == GM_SURF_SPHERE && type2 == GM_SURF_ELLIPSOID {
        runtime.sphere_ellipsoid
    } else if type1 == GM_SURF_SPHERE && type2 == GM_SURF_POLYGON {
        runtime.sphere_polygon
    } else if type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_POLYGON {
        runtime.ellipsoid_polygon
    } else if type1 == GM_SURF_SPHERE && type2 == GM_SURF_MESH {
        runtime.sphere_mesh
    } else if type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_MESH {
        runtime.ellipsoid_mesh
    } else if type1 == GM_SURF_BOX && type2 == GM_SURF_BOX {
        runtime.box_box
    } else if type1 == GM_SURF_BOX && type2 == GM_SURF_MESH {
        runtime.box_mesh
    } else if type1 == GM_SURF_MESH && type2 == GM_SURF_MESH {
        runtime.mesh_mesh
    } else {
        None
    }
}

fn is_resolved_pair(type1: u8, type2: u8) -> bool {
    (type1 == GM_SURF_SPHERE && type2 == GM_SURF_BOX)
        || (type1 == GM_SURF_SPHERE && type2 == GM_SURF_ELLIPSOID)
        || (type1 == GM_SURF_SPHERE && type2 == GM_SURF_POLYGON)
        || (type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_POLYGON)
        || (type1 == GM_SURF_SPHERE && type2 == GM_SURF_MESH)
        || (type1 == GM_SURF_ELLIPSOID && type2 == GM_SURF_MESH)
        || (type1 == GM_SURF_BOX && type2 == GM_SURF_BOX)
        || (type1 == GM_SURF_BOX && type2 == GM_SURF_MESH)
        || (type1 == GM_SURF_MESH && type2 == GM_SURF_MESH)
}

pub fn gm_collision_neg(collision: &mut GmCollision) {
    collision.normal.x = -collision.normal.x;
    collision.normal.y = -collision.normal.y;
    collision.normal.z = -collision.normal.z;
    let material = collision.material1;
    collision.material1 = collision.material2;
    collision.separation.x = -collision.separation.x;
    collision.material2 = material;
    collision.separation.y = -collision.separation.y;
    collision.separation.z = -collision.separation.z;
    collision.face_normal.x = -collision.face_normal.x;
    collision.face_normal.y = -collision.face_normal.y;
    collision.face_normal.z = -collision.face_normal.z;
}

/// 0x008E8890  Dispatches a geometry pair and fixes reversed output.
/// UNVALIDATED upstream.
pub fn gm_surf_compute_collision(
    surf1: &LocatedGmSurf,
    surf2: &LocatedGmSurf,
    buffer: &mut CHmsCollisionBuffer,
    runtime: &CollisionShapeDispatch,
) -> i32 {
    let first: &LocatedGmSurf;
    let second: &LocatedGmSurf;
    let mut reversed = false;
    if surf1.surf.surf_type() > surf2.surf.surf_type() {
        first = surf2;
        second = surf1;
        reversed = true;
    } else {
        first = surf1;
        second = surf2;
    }

    let type1 = first.surf.surf_type();
    let type2 = second.surf.surf_type();
    let start = buffer.get_count();
    let result;
    if type1 == GM_SURF_SPHERE && type2 == GM_SURF_SPHERE {
        result = gm_collision_sphere_sphere(first, second, buffer);
    } else if !is_resolved_pair(type1, type2) {
        result = 0;
    } else {
        let handler = match resolved_handler(type1, type2, runtime) {
            Some(h) => h,
            None => panic!("tmnf: no collision handler for shape pair"),
        };
        result = handler(first, second, buffer);
    }
    if result == 0 || !reversed {
        return result;
    }

    let end = buffer.get_count();
    for i in start..end {
        gm_collision_neg(buffer.get_collision(i as usize));
    }
    1
}

/// 0x00537150  Computes surface collision and remaps material indices.
/// UNVALIDATED upstream.
pub fn compute_surface_collision(
    surface1: &CPlugSurface,
    iso1: &GmIso4,
    surface2: &CPlugSurface,
    iso2: &GmIso4,
    accel2: Option<&TmnfMeshQueryAccel>,
    buffer: &mut CHmsCollisionBuffer,
    runtime: &CollisionShapeDispatch,
) -> i32 {
    let start = buffer.get_count();
    let surf1 = LocatedGmSurf {
        surf: &surface1.geom,
        iso: iso1,
        is_located: 1,
        accel: None,
    };
    let surf2 = LocatedGmSurf {
        surf: &surface2.geom,
        iso: iso2,
        is_located: 1,
        accel: accel2,
    };
    if gm_surf_compute_collision(&surf1, &surf2, buffer, runtime) == 0 {
        return 0;
    }

    let end = buffer.get_count();
    for i in start..end {
        let collision = buffer.get_collision(i as usize);
        if collision.material1 as usize >= surface1.material_ids.len()
            || collision.material2 as usize >= surface2.material_ids.len()
        {
            panic!("tmnf: collision material index out of range");
        }
        collision.material1 =
            surface1.material_ids[collision.material1 as usize] as u16;
        collision.material2 =
            surface2.material_ids[collision.material2 as usize] as u16;
    }
    1
}

/// 0x00537150 public form (no accel).
pub fn cplug_surface_compute_collision(
    surface1: &CPlugSurface,
    iso1: &GmIso4,
    surface2: &CPlugSurface,
    iso2: &GmIso4,
    buffer: &mut CHmsCollisionBuffer,
    runtime: &CollisionShapeDispatch,
) -> i32 {
    compute_surface_collision(surface1, iso1, surface2, iso2, None, buffer, runtime)
}

/* ---------------------------------------------------------------------------
 * Static grid locate
 * ------------------------------------------------------------------------- */

/// A query is served from the grid when its box lies inside the region its
/// cell's list was built for and the surface is a sphere or ellipsoid no
/// larger than the region expansion.
pub fn static_grid_locate(
    grid: &TmnfStaticGrid,
    box_: &GmBoxAligned,
    surface: &CPlugSurface,
) -> Option<u32> {
    let shape_radii = match &surface.geom {
        GmSurfGeom::Ellipsoid(e) => e.radii,
        GmSurfGeom::Sphere(s) => {
            GmVec3 { x: s.radius, y: s.radius, z: s.radius }
        }
        _ => return None,
    };
    let radii = [shape_radii.x, shape_radii.y, shape_radii.z];
    let c = [box_.center.x, box_.center.y, box_.center.z];
    let h = [box_.half_extent.x, box_.half_extent.y, box_.half_extent.z];
    let inner = grid.expand - grid.margin;
    let mut index = [0u32; 3];
    for a in 0..3usize {
        if !((radii[a] as f64) <= grid.expand) {
            return None;
        }
        let offset = (c[a] as f64) - grid.origin[a];
        if !(offset >= 0.0 && offset < (grid.dims[a] as f64) * grid.cell_size) {
            return None;
        }
        index[a] = (offset * grid.inv_cell_size) as u32;
        if index[a] >= grid.dims[a] {
            return None;
        }
        let lo = grid.origin[a] + (index[a] as f64) * grid.cell_size;
        let hi = lo + grid.cell_size;
        if !(((c[a] as f64) - (h[a] as f64)) >= lo - inner
            && ((c[a] as f64) + (h[a] as f64)) <= hi + inner)
        {
            return None;
        }
    }
    Some((index[2] * grid.dims[1] + index[1]) * grid.dims[0] + index[0])
}
