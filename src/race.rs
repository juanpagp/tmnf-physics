//! TMNF race progression over an immutable [`TmnfRoute`] snapshot
//! (`src/race.h` / `src/race.c`), transliterated.
//!
//! Rules follow the game as measured against six TMX world-record replays:
//! a checkpoint counts once per lap in any order; the finish counts only when
//! every checkpoint of the lap has been taken; a trigger fires when a car
//! collision ellipsoid touches the trigger volume, computed with the game's
//! own narrowphase (ellipsoid vs the trigger box's twelve face triangles)
//! tested where the game tests it — in every collision detection pass of the
//! physics step, against the predicted pre-response transform.
//!
//! Trigger contact mirrors 0x0053A660 `ComputeCollisionTree2RootOnly` with
//! the trigger as tree two: world-aligned root AABB overlap, per-child AABB
//! overlap, then 0x008EADC0 `GmCollision_Ellipsoid_Mesh` against the twelve
//! box faces — reached here through
//! [`crate::collision::cplug_surface_compute_collision`] (the port of
//! 0x00537150, whose shape dispatch installs `gm_collision_ellipsoid_mesh`).
//! The C's local `compose_with_parent`/`tree_world_iso` statics are the
//! identical code as [`crate::collision::tree_world_iso`] (self = self *
//! parent via `GmMat3_Mult` + `GmVec3_Mult_Iso4`), so the shared function is
//! used.
//!
//! The C's explicit-stack traversal replaces the game's recursion so the same
//! code runs on a GPU thread; only the OR of the per-pair results matters, so
//! the visiting order is free — it is kept identical anyway.
//!
//! # Documented deviations (all structural, none arithmetic)
//!
//! * The C's NULL argument checks (`route == NULL`, `state == NULL`, …) are
//!   vacuous under Rust references and dropped; every other validation and
//!   its panic message is kept verbatim (`tmnf race: <message>`).
//! * `car_tree` is an index into a `&[CPlugTree]` arena (the port's
//!   `CPlugTree::children` are arena indices); the C's `CPlugSurface::geom
//!   != NULL` guard is structurally impossible — an owned surface always has
//!   geometry.
//! * The `__CUDA_ARCH__` fixed-capacity contact buffer is skipped: the host
//!   path (`CHmsCollisionBuffer_Init`/`Destroy`) is ported (a plain owned
//!   buffer).

use crate::buffer::CHmsCollisionBuffer;
use crate::collision::{
    cplug_surface_compute_collision, tree_world_iso, CollisionShapeDispatch, CPlugSurface,
    CPlugTree, GmBoxAligned, GmSurf, GmSurfGeom, GmSurfMesh, GmSurfMeshFace, GmSurfMeshNode,
    GM_SURF_MESH,
};
use crate::gm::{GmIso4, GmVec3};
use crate::route::{
    TmnfRoute, TmnfRouteProjection, TmnfRouteReferencePoint, TmnfRouteTrigger,
    TMNF_ROUTE_MULTILAP,
};
use crate::surface::{TMNF_SURFACE_GRASS, TMNF_SURFACE_WET_GRASS};

/* ---------------------------------------------------------------------------
 * Constants (race.h)
 * ------------------------------------------------------------------------- */

pub const TMNF_RACE_TICK_MS: u32 = 10;
pub const TMNF_RACE_PROJECTION_WINDOW_SEGMENTS: u32 = 32;
pub const TMNF_RACE_TELEPORT_DISTANCE_METERS: u32 = 32;
pub const TMNF_RACE_MAX_CHECKPOINTS: u32 = 63;
pub const TMNF_RACE_FINISH_CONTACT_BIT: u32 = 63;

/// `TMNF_MATERIAL_GRASS` = `TMNF_SURFACE_GRASS`.
pub const TMNF_MATERIAL_GRASS: u32 = TMNF_SURFACE_GRASS;
/// `TMNF_MATERIAL_WET_GRASS` = `TMNF_SURFACE_WET_GRASS`.
pub const TMNF_MATERIAL_WET_GRASS: u32 = TMNF_SURFACE_WET_GRASS;

/*
 * Route corridor. The car is inside when its horizontal (XZ) distance from
 * the projected centerline point is at most max(WIDTH_FACTOR * half_width,
 * WIDTH_FLOOR) and its height relative to that point lies in
 * [-BELOW, ABOVE]. Bounds come from every committed game capture: record
 * lines reach 24.6 m off the centerline (E01 wall ride, 2.9 half-widths on
 * an 8.3 m sample and 20 half-widths where the route width degenerates to
 * 1.0 m), 17.6 m above (E01) and 12.7 m below (E01, B04) while grounded; the
 * A01 record's final jump flies 39.2 m above the road it clears. A04's pool
 * glide runs 22 to 30 m below the raised route and is outside.
 */
pub const TMNF_RACE_CORRIDOR_WIDTH_FACTOR: f32 = 3.0;
pub const TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS: f32 = 28.0;
pub const TMNF_RACE_CORRIDOR_ABOVE_METERS: f32 = 44.0;
pub const TMNF_RACE_CORRIDOR_BELOW_METERS: f32 = 16.0;

/* ---------------------------------------------------------------------------
 * State (plain structs — race.h carries no size contracts)
 * ------------------------------------------------------------------------- */

/// `TmnfRaceState`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRaceState {
    /// bit i: checkpoint i taken this lap.
    pub visited_checkpoints: u64,
    /// low bits: checkpoints; finish i uses bit 63-i.
    pub trigger_contacts: u64,
    pub visited_count: u32,
    pub completed_laps: u32,
    pub passed_race_checkpoints: u32,
    pub elapsed_ticks: u32,
    pub finish_time_ms: u32,
    pub projection_segment: u32,
    pub centerline_segment: u32,
    pub off_track_ticks: u32,
    pub stuck_ticks: u32,
    pub arc_length: f32,
    /// 3D distance to the centerline.
    pub lateral_offset: f32,
    pub half_width: f32,
    /// XZ distance to the projected point.
    pub corridor_lateral: f32,
    /// car y minus projected point y.
    pub corridor_vertical: f32,
    pub unwrapped_progress: f32,
    pub best_progress: f32,
    pub previous_progress: f32,
    pub previous_car_transform: GmIso4,
    /// CTrackManiaPlayerInfo+0x274: where a respawn places the car. The last
    /// accepted checkpoint's spawn unless that block has no_respawn set
    /// (0x0047C330 OnCheckpoint keeps the previous one); the start spawn
    /// after a lap (0x00480820 OnFinishLine) and at reset (0x004831F0
    /// ResetPlayer).
    pub respawn_location: GmIso4,
    pub finished: u8,
    /// progress frozen this tick.
    pub outside_corridor: u8,
    /// CTrackManiaPlayerInfo+0x2e8: a respawnable checkpoint was passed.
    /// Without it 0x00472700 SmallRespawn restarts the race instead of
    /// respawning.
    pub respawn_available: u8,
    pub reserved: u8,
}

/// `TmnfRaceStepResult`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TmnfRaceStepResult {
    pub checkpoint_accepted: u8,
    pub checkpoint_repeated: u8,
    pub lap_completed: u8,
    pub finished: u8,
    pub checkpoint_index: u32,
    pub race_time_ms: u32,
}

/* ---------------------------------------------------------------------------
 * Route validation
 * ------------------------------------------------------------------------- */

fn validate_route(route: &TmnfRoute) {
    let metadata = &route.metadata;
    /* checkpoint_count may be zero: tracks like A10-Acrobatic have only a
     * start and a finish. The visited set, the finish gate, and the
     * total-checkpoint arithmetic below are all well defined for zero. */
    if metadata.lap_count == 0
        || metadata.finish_count == 0
        || metadata.finish_count as u64 + metadata.checkpoint_count as u64 > 64
        || metadata.checkpoint_count > TMNF_RACE_MAX_CHECKPOINTS
        || metadata.reference_count != metadata.checkpoint_count + 2
        || metadata.centerline_count < 2
    {
        panic!("tmnf race: route metadata is inconsistent");
    }
    if ((metadata.flags & TMNF_ROUTE_MULTILAP) != 0) != (metadata.lap_count > 1) {
        panic!("tmnf race: route multilap flag is inconsistent");
    }
    let expected = (metadata.checkpoint_count as u64 + 1) * metadata.lap_count as u64;
    if expected > u32::MAX as u64 || metadata.total_race_checkpoints != expected as u32 {
        panic!("tmnf race: route total checkpoint count is inconsistent");
    }
}

/* ---------------------------------------------------------------------------
 * Trigger volume as the game sees it: a twelve-face box mesh
 * ------------------------------------------------------------------------- */

/// 0x5f * (low + high) etc. — the face's AABB from its three vertices
/// (`face_box`).
fn face_box(a: &GmVec3, b: &GmVec3, c: &GmVec3) -> GmBoxAligned {
    let mut low = *a;
    let mut high = *a;
    let points = [b, c];
    for p in &points {
        if p.x < low.x {
            low.x = p.x;
        }
        if p.y < low.y {
            low.y = p.y;
        }
        if p.z < low.z {
            low.z = p.z;
        }
        if p.x > high.x {
            high.x = p.x;
        }
        if p.y > high.y {
            high.y = p.y;
        }
        if p.z > high.z {
            high.z = p.z;
        }
    }
    GmBoxAligned {
        center: GmVec3 {
            x: 0.5f32 * (low.x + high.x),
            y: 0.5f32 * (low.y + high.y),
            z: 0.5f32 * (low.z + high.z),
        },
        half_extent: GmVec3 {
            x: 0.5f32 * (high.x - low.x),
            y: 0.5f32 * (high.y - low.y),
            z: 0.5f32 * (high.z - low.z),
        },
    }
}

/// `add_face`: one box-face triangle, wound so its geometric normal points
/// along `outward`, with its AABB node (`skip_count` 1).
fn add_face(
    faces: &mut [GmSurfMeshFace; 12],
    nodes: &mut [GmSurfMeshNode; 12],
    vertices: &[GmVec3; 8],
    index: usize,
    v0: u32,
    v1: u32,
    v2: u32,
    outward: &GmVec3,
) {
    let a = &vertices[v0 as usize];
    let b = &vertices[v1 as usize];
    let c = &vertices[v2 as usize];
    let e1 = GmVec3 {
        x: b.x - a.x,
        y: b.y - a.y,
        z: b.z - a.z,
    };
    let e2 = GmVec3 {
        x: c.x - a.x,
        y: c.y - a.y,
        z: c.z - a.z,
    };
    let n = GmVec3 {
        x: e1.y * e2.z - e1.z * e2.y,
        y: e1.z * e2.x - e1.x * e2.z,
        z: e1.x * e2.y - e1.y * e2.x,
    };
    let (v1, v2) = if n.x * outward.x + n.y * outward.y + n.z * outward.z < 0.0f32 {
        (v2, v1)
    } else {
        (v1, v2)
    };
    let face = &mut faces[index];
    *face = GmSurfMeshFace::default(); /* memset(face, 0, sizeof(*face)) */
    face.normal = *outward;
    face.vertex = [v0, v1, v2];
    face.material_index = 0;
    let node = &mut nodes[index];
    node.skip_count = 1;
    node.box_aligned = face_box(
        &vertices[v0 as usize],
        &vertices[v1 as usize],
        &vertices[v2 as usize],
    );
    node.face_index = index as u32;
}

/// `build_trigger_mesh`: the trigger box as a `CPlugSurface` owning a
/// twelve-triangle mesh (the C's `TriggerMesh` pointed into itself).
fn build_trigger_mesh(box_aligned: &GmBoxAligned) -> CPlugSurface {
    if !(box_aligned.half_extent.x > 0.0f32)
        || !(box_aligned.half_extent.y > 0.0f32)
        || !(box_aligned.half_extent.z > 0.0f32)
    {
        panic!("tmnf race: trigger box has a non-positive extent");
    }
    let mut vertices = [GmVec3::ZERO; 8];
    for i in 0..8u32 {
        vertices[i as usize].x = box_aligned.center.x
            + if (i & 1u32) != 0 {
                box_aligned.half_extent.x
            } else {
                -box_aligned.half_extent.x
            };
        vertices[i as usize].y = box_aligned.center.y
            + if (i & 2u32) != 0 {
                box_aligned.half_extent.y
            } else {
                -box_aligned.half_extent.y
            };
        vertices[i as usize].z = box_aligned.center.z
            + if (i & 4u32) != 0 {
                box_aligned.half_extent.z
            } else {
                -box_aligned.half_extent.z
            };
    }
    let mut faces = [GmSurfMeshFace::default(); 12];
    let mut nodes = [GmSurfMeshNode::default(); 12];
    let mut face = 0usize;
    for axis in 0..3u32 {
        let bit = 1u32 << axis;
        for side in 0..2u32 {
            let mut outward = GmVec3::ZERO;
            let sign = if side != 0 { 1.0f32 } else { -1.0f32 };
            if axis == 0 {
                outward.x = sign;
            } else if axis == 1 {
                outward.y = sign;
            } else {
                outward.z = sign;
            }
            let mut corners = [0u32; 4];
            let mut count = 0usize;
            for v in 0..8u32 {
                if ((v & bit) != 0) == (side != 0) {
                    corners[count] = v;
                    count += 1;
                }
            }
            /* corners are in increasing index order: (00, 01, 10, 11) over
             * the two remaining axes, so 0-1-3 and 0-3-2 tile the quad. */
            add_face(&mut faces, &mut nodes, &vertices, face, corners[0], corners[1], corners[3], &outward);
            face += 1;
            add_face(&mut faces, &mut nodes, &vertices, face, corners[0], corners[3], corners[2], &outward);
            face += 1;
        }
    }
    CPlugSurface {
        geom: GmSurfGeom::Mesh(GmSurfMesh {
            base: GmSurf {
                vtable: 0,
                material_index: 0,
                surf_type: GM_SURF_MESH,
                reserved: 0,
            },
            vertex_count: 8,
            vertices: vertices.to_vec(),
            face_count: 12,
            faces: faces.to_vec(),
            node_count: 12,
            nodes: nodes.to_vec(),
        }),
        material_ids: vec![0u8],
        material_count: 1,
    }
}

/* ---------------------------------------------------------------------------
 * Car subtrees against the trigger (ComputeCollisionTree2RootOnly)
 * ------------------------------------------------------------------------- */

const TRIGGER_TREE_STACK_CAPACITY: usize = 16;

/// `car_tree_contact`: the car tree (a root with leaf children) against the
/// trigger surface. `car_tree` indexes `car_trees`.
fn car_tree_contact(
    car_trees: &[CPlugTree],
    car_tree: usize,
    car_world_transform: &GmIso4,
    trigger_world_box: &GmBoxAligned,
    mesh: &CPlugSurface,
    trigger_iso: &GmIso4,
    buffer: &mut CHmsCollisionBuffer,
    runtime: &CollisionShapeDispatch,
) -> bool {
    let mut stack: [(usize, GmIso4); TRIGGER_TREE_STACK_CAPACITY] =
        [(0usize, GmIso4::IDENTITY); TRIGGER_TREE_STACK_CAPACITY];
    let mut depth = 0usize;
    let mut contact = false;

    let root = &car_trees[car_tree];
    let car_world = tree_world_iso(root, car_world_transform);
    if let Some(surface) = &root.surface {
        contact |= cplug_surface_compute_collision(
            surface,
            &car_world,
            mesh,
            trigger_iso,
            buffer,
            runtime,
        ) != 0;
    }
    if root.children.len() > TRIGGER_TREE_STACK_CAPACITY {
        panic!("tmnf race: car collision tree is wider than the trigger stack");
    }
    let mut i = root.children.len();
    while i > 0 {
        i -= 1;
        stack[depth] = (root.children[i], car_world);
        depth += 1;
    }
    while depth != 0 {
        depth -= 1;
        let frame = stack[depth];
        let tree = &car_trees[frame.0];
        if (tree.flags & 0x80u32) == 0 {
            continue;
        }
        let mut box_aligned = GmBoxAligned::default();
        box_aligned.set_mult(&tree.box_aligned, &frame.1);
        if !box_aligned.test_inter(trigger_world_box) {
            continue;
        }
        let world = tree_world_iso(tree, &frame.1);
        if let Some(surface) = &tree.surface {
            contact |= cplug_surface_compute_collision(
                surface,
                &world,
                mesh,
                trigger_iso,
                buffer,
                runtime,
            ) != 0;
        }
        if depth + tree.children.len() > TRIGGER_TREE_STACK_CAPACITY {
            panic!("tmnf race: car collision tree is deeper than the trigger stack");
        }
        let mut i = tree.children.len();
        while i > 0 {
            i -= 1;
            stack[depth] = (tree.children[i], world);
            depth += 1;
        }
    }
    contact
}

/// `TmnfRace_TriggerContact`: game-faithful trigger contact. The game
/// registers a waypoint through the collision manager: the waypoint mobil
/// has collision enabled and a contact sink (0x0047CBA0
/// `CTrackManiaRaceTriggerAbsorbContact::AbsorbContact`). The test therefore
/// mirrors 0x0053A8F0 `CHmsCollisionManager_SZone::ComputeCollision` for the
/// car tree against a trigger tree whose surface is the captured root box.
///
/// `car_tree` indexes `car_trees` (the port's tree arena).
pub fn tmnf_race_trigger_contact(
    trigger: &TmnfRouteTrigger,
    car_trees: &[CPlugTree],
    car_tree: usize,
    car_world_transform: &GmIso4,
) -> bool {
    let root = &car_trees[car_tree];
    if (root.flags & 0x80u32) == 0 {
        panic!("tmnf race: car collision tree has collisions disabled");
    }
    if (trigger.tree_flags & 0x80u32) == 0 {
        panic!("tmnf race: trigger tree has collisions disabled");
    }

    let mut car_box = GmBoxAligned::default();
    car_box.set_mult(&root.box_aligned, car_world_transform);
    let mut trigger_box = GmBoxAligned::default();
    trigger_box.set_mult(&trigger.box_aligned, &trigger.transform);
    if !car_box.test_inter(&trigger_box) {
        return false;
    }

    let mesh = build_trigger_mesh(&trigger.box_aligned);
    let runtime = CollisionShapeDispatch::default();
    /* The contacts themselves are discarded; only their existence counts. */
    let mut buffer = CHmsCollisionBuffer::new();
    let contact = car_tree_contact(
        car_trees,
        car_tree,
        car_world_transform,
        &trigger_box,
        &mesh,
        &trigger.transform,
        &mut buffer,
        &runtime,
    );
    /* CHmsCollisionBuffer_Destroy(&buffer) — plain drop. */
    contact
}

/// `TmnfRace_TriggerContactMask`: every trigger of the route against the car
/// tree at one transform — bit i for checkpoint i,
/// `TMNF_RACE_FINISH_CONTACT_BIT` for the finish. The physics step calls this
/// from each collision detection pass with the predicted pre-response iso.
pub fn tmnf_race_trigger_contact_mask(
    route: &TmnfRoute,
    car_trees: &[CPlugTree],
    car_tree: usize,
    car_world_transform: &GmIso4,
) -> u64 {
    validate_route(route);
    let mut mask = 0u64;
    for i in 0..route.metadata.checkpoint_count {
        if tmnf_race_trigger_contact(
            &route.checkpoints[i as usize],
            car_trees,
            car_tree,
            car_world_transform,
        ) {
            mask |= 1u64 << i;
        }
    }
    /* Each alternative owns an edge bit: merging them would miss entering a
     * second finish while still touching a previously rejected first one. */
    for i in 0..route.metadata.finish_count {
        if tmnf_race_trigger_contact(
            &route.finish[i as usize],
            car_trees,
            car_tree,
            car_world_transform,
        ) {
            mask |= 1u64 << (TMNF_RACE_FINISH_CONTACT_BIT - i);
        }
    }
    mask
}

/* ---------------------------------------------------------------------------
 * Dense projection
 * ------------------------------------------------------------------------- */

/// Projection plus the offset vector from the projected centerline point to
/// the car, split into the horizontal (XZ) distance and the signed height
/// the corridor rule tests.
struct DenseProjection {
    projection: TmnfRouteProjection,
    lateral_horizontal: f32,
    vertical: f32,
}

/// `project_segment`: (projection, distance_sq, offset) — the C used
/// out-parameters.
fn project_segment(
    route: &TmnfRoute,
    position: &GmVec3,
    segment: u32,
) -> (TmnfRouteProjection, f32, GmVec3) {
    let points: &[TmnfRouteReferencePoint] = route.get_reference_points();
    let a = &points[segment as usize];
    let b = &points[(segment + 1) as usize];
    let dx = b.position.x - a.position.x;
    let dy = b.position.y - a.position.y;
    let dz = b.position.z - a.position.z;
    let px = position.x - a.position.x;
    let py = position.y - a.position.y;
    let pz = position.z - a.position.z;
    let length_sq = dx * dx + dy * dy + dz * dz;
    if !(length_sq > 0.0f32) {
        panic!("tmnf race: route centerline contains a degenerate segment");
    }
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
    let offset = GmVec3 {
        x: ox,
        y: oy,
        z: oz,
    };
    /* The C's designated initializer zeroes lateral_offset. */
    let projection = TmnfRouteProjection {
        arc_length: a.arc_length + t * (b.arc_length - a.arc_length),
        lateral_offset: 0.0,
        half_width: a.half_width + t * (b.half_width - a.half_width),
        segment_index: a.leg_index,
        centerline_segment_index: segment,
        segments_tested: 1,
    };
    (projection, distance_sq, offset)
}

fn dense_from_offset(projection: TmnfRouteProjection, offset: &GmVec3) -> DenseProjection {
    DenseProjection {
        projection,
        lateral_horizontal: (offset.x * offset.x + offset.z * offset.z).sqrt(),
        vertical: offset.y,
    }
}

/// `project_full`: full-route search (reset, teleport).
/// `TmnfRoute_Project` selects the segment; the offset vector is recomputed
/// on that segment.
fn project_full(route: &TmnfRoute, position: &GmVec3) -> DenseProjection {
    let projection = route.project(position);
    let (_, _, offset) =
        project_segment(route, position, projection.centerline_segment_index);
    dense_from_offset(projection, &offset)
}

/// `project_local`: the 65-segment dense window around the cursor.
fn project_local(route: &TmnfRoute, state: &TmnfRaceState, position: &GmVec3) -> DenseProjection {
    let segment_count = route.get_reference_point_count() - 1;
    if state.centerline_segment >= segment_count {
        panic!("tmnf race: dense projection cursor is out of range");
    }
    let minimum = if state.centerline_segment > TMNF_RACE_PROJECTION_WINDOW_SEGMENTS {
        state.centerline_segment - TMNF_RACE_PROJECTION_WINDOW_SEGMENTS
    } else {
        0
    };
    let mut maximum = state.centerline_segment + TMNF_RACE_PROJECTION_WINDOW_SEGMENTS;
    if maximum >= segment_count {
        maximum = segment_count - 1;
    }

    let (mut best, mut best_distance_sq, mut best_offset) =
        project_segment(route, position, minimum);
    for segment in minimum + 1..=maximum {
        let (candidate, candidate_distance_sq, candidate_offset) =
            project_segment(route, position, segment);
        if candidate_distance_sq < best_distance_sq {
            best = candidate;
            best_distance_sq = candidate_distance_sq;
            best_offset = candidate_offset;
        }
    }
    best.lateral_offset = best_distance_sq.sqrt();
    best.segments_tested = maximum - minimum + 1;
    dense_from_offset(best, &best_offset)
}

/// `project_dense`.
fn project_dense(route: &TmnfRoute, state: &TmnfRaceState, position: &GmVec3) -> DenseProjection {
    /*
     * A 65-segment dense window (cursor +/- 32) keeps ordinary motion on its
     * current road branch at self-intersections. More than 32 metres in one
     * 10 ms race tick is a restore/teleport, so reacquire with one full dense
     * search. A spatially close move to overlapping geometry remains local.
     */
    if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() {
        panic!("tmnf race: cannot project a non-finite position");
    }
    let dx = position.x - state.previous_car_transform.t[0];
    let dy = position.y - state.previous_car_transform.t[1];
    let dz = position.z - state.previous_car_transform.t[2];
    let displacement_sq = dx * dx + dy * dy + dz * dz;
    let teleport_distance = TMNF_RACE_TELEPORT_DISTANCE_METERS as f32;
    if displacement_sq > teleport_distance * teleport_distance {
        return project_full(route, position);
    }
    project_local(route, state, position)
}

/// `apply_projection`.
fn apply_projection(state: &mut TmnfRaceState, dense: &DenseProjection) {
    state.arc_length = dense.projection.arc_length;
    state.lateral_offset = dense.projection.lateral_offset;
    state.half_width = dense.projection.half_width;
    state.projection_segment = dense.projection.segment_index;
    state.centerline_segment = dense.projection.centerline_segment_index;
    state.corridor_lateral = dense.lateral_horizontal;
    state.corridor_vertical = dense.vertical;
    let mut allowed = TMNF_RACE_CORRIDOR_WIDTH_FACTOR * dense.projection.half_width;
    if allowed < TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS {
        allowed = TMNF_RACE_CORRIDOR_WIDTH_FLOOR_METERS;
    }
    state.outside_corridor = (dense.lateral_horizontal > allowed
        || dense.vertical > TMNF_RACE_CORRIDOR_ABOVE_METERS
        || dense.vertical < -TMNF_RACE_CORRIDOR_BELOW_METERS) as u8;
}

/* ---------------------------------------------------------------------------
 * Public race API
 * ------------------------------------------------------------------------- */

/// `TmnfRace_IsGroundPlaneMaterial`: the Stadium ground plane
/// (StadiumGrass terrain: Grass, WetGrass). Every Stadium block surface
/// reports another id; other collections have their own terrain material.
pub fn tmnf_race_is_ground_plane_material(material_id: i32) -> bool {
    material_id == TMNF_MATERIAL_GRASS as i32 || material_id == TMNF_MATERIAL_WET_GRASS as i32
}

/// `TmnfRace_Reset`. A fresh player has no contact history (0x004831F0
/// ResetPlayer clears the checkpoint flags); the first physics step's
/// detection passes decide the first contacts.
pub fn tmnf_race_reset(route: &TmnfRoute, state: &mut TmnfRaceState, car_world_transform: &GmIso4) {
    validate_route(route);
    *state = TmnfRaceState::default(); /* memset(state, 0, sizeof(*state)) */
    state.previous_car_transform = *car_world_transform;
    state.respawn_location = route.start.spawn;
    let position = GmVec3 {
        x: car_world_transform.t[0],
        y: car_world_transform.t[1],
        z: car_world_transform.t[2],
    };
    let dense = project_full(route, &position);
    apply_projection(state, &dense);
    state.unwrapped_progress = dense.projection.arc_length;
    state.best_progress = dense.projection.arc_length;
    state.previous_progress = dense.projection.arc_length;
}

/// `TmnfRace_UpdateOffTrack`: off-track counter. Counts ticks on which at
/// least one wheel touches the ground and every touching wheel is on the
/// ground plane. Any wheel on a track surface resets the counter. Airborne
/// ticks hold it, so bouncing on the grass cannot evade the rule. Returns
/// true on the tick the count reaches `grace_ticks`.
pub fn tmnf_race_update_off_track(
    state: &mut TmnfRaceState,
    grace_ticks: u32,
    wheels_in_contact: u32,
    wheels_on_ground_plane: u32,
) -> bool {
    if grace_ticks == 0 || wheels_on_ground_plane > wheels_in_contact {
        panic!("tmnf race: invalid off-track state");
    }
    if wheels_in_contact == 0 {
        return false;
    }
    if wheels_on_ground_plane != wheels_in_contact {
        state.off_track_ticks = 0;
        return false;
    }
    if state.off_track_ticks < grace_ticks {
        state.off_track_ticks += 1;
    }
    state.off_track_ticks == grace_ticks
}

/// `TmnfRace_Step`: advances race bookkeeping by one canonical 10 ms physics
/// tick. `trigger_contacts` is the tick's OR of
/// [`tmnf_race_trigger_contact_mask`] over the step's detection passes; a
/// trigger event is a contact edge (contact this tick, none on the previous
/// tick). `car_world_transform` is the post-step state and only feeds the
/// progress projection, which is the environment's, not the game's.
pub fn tmnf_race_step(
    route: &TmnfRoute,
    state: &mut TmnfRaceState,
    trigger_contacts: u64,
    car_world_transform: &GmIso4,
) -> TmnfRaceStepResult {
    validate_route(route);
    if state.finished != 0 {
        panic!("tmnf race: cannot step a finished race");
    }

    let mut result = TmnfRaceStepResult::default();
    let checkpoint_count = route.metadata.checkpoint_count;
    state.elapsed_ticks += 1;

    /* The game's per-checkpoint flags (0x0047E400 InternalOnCheckpoint)
     * make every contact after the first of a lap inert; the contact edge
     * against the previous tick's passes is the same rule. */
    let entered = trigger_contacts & !state.trigger_contacts;
    state.trigger_contacts = trigger_contacts;

    for i in 0..checkpoint_count {
        let bit = 1u64 << i;
        if (entered & bit) == 0 {
            continue;
        }
        if (state.visited_checkpoints & bit) != 0 {
            result.checkpoint_repeated = 1;
            continue;
        }
        state.visited_checkpoints |= bit;
        state.visited_count += 1;
        state.passed_race_checkpoints += 1;
        result.checkpoint_accepted = 1;
        result.checkpoint_index = i;
        /* 0x0047C330 OnCheckpoint: a block whose info has +0x120 set keeps
         * the previous spawn and does not arm the respawn. */
        if route.checkpoints[i as usize].no_respawn == 0 {
            state.respawn_location = route.checkpoints[i as usize].spawn;
            state.respawn_available = 1;
        }
    }

    let entered_finish = (entered & (u64::MAX << (64 - route.metadata.finish_count))) != 0;
    let mut began_new_lap = false;
    if entered_finish && state.visited_count == checkpoint_count {
        state.passed_race_checkpoints += 1;
        state.completed_laps += 1;
        result.lap_completed = 1;
        if state.completed_laps == route.metadata.lap_count {
            if state.passed_race_checkpoints != route.metadata.total_race_checkpoints {
                panic!("tmnf race: finished with the wrong checkpoint total");
            }
            state.finished = 1;
            /* u32 multiply wraps in the C; wrapping_mul keeps that exact. */
            state.finish_time_ms = state.elapsed_ticks.wrapping_mul(TMNF_RACE_TICK_MS);
            result.finished = 1;
            result.race_time_ms = state.finish_time_ms;
        } else {
            let origin: &[TmnfRouteReferencePoint] = route.get_reference_points();
            state.visited_checkpoints = 0;
            state.visited_count = 0;
            /* 0x00480820 OnFinishLine -> 0x0047E400 InternalOnCheckpoint
             * with a null spawn: +0x274 = +0x244, the start spawn. */
            state.respawn_location = route.start.spawn;
            state.projection_segment = origin[0].leg_index;
            state.centerline_segment = 0;
            state.arc_length = 0.0;
            state.lateral_offset = 0.0;
            state.half_width = origin[0].half_width;
            state.corridor_lateral = 0.0;
            state.corridor_vertical = 0.0;
            state.outside_corridor = 0;
            began_new_lap = true;
        }
    }

    if state.finished != 0 {
        let point_count = route.get_reference_point_count();
        let points: &[TmnfRouteReferencePoint] = route.get_reference_points();
        state.arc_length = route.get_reference_length();
        state.lateral_offset = 0.0;
        state.half_width = points[point_count as usize - 1].half_width;
        state.centerline_segment = point_count - 2;
        state.corridor_lateral = 0.0;
        state.corridor_vertical = 0.0;
        state.outside_corridor = 0;
    } else if !began_new_lap {
        let position = GmVec3 {
            x: car_world_transform.t[0],
            y: car_world_transform.t[1],
            z: car_world_transform.t[2],
        };
        let dense = project_dense(route, state, &position);
        apply_projection(state, &dense);
    }
    /*
     * Progress is credited only inside the corridor. Outside it the value
     * freezes: it neither advances (A04's pool glide collected the last
     * 90 m of the route from 18 m below it) nor reverses, so the
     * progress-only stuck rule ends an excursion that does not come back
     * and the potential pays only for corridor progress. A finish contact
     * is the game's own rule and always anchors the new lap.
     */
    state.previous_progress = state.unwrapped_progress;
    if state.finished != 0 || began_new_lap {
        state.unwrapped_progress =
            state.completed_laps as f32 * route.get_reference_length();
    } else if state.outside_corridor == 0 {
        state.unwrapped_progress = state.completed_laps as f32 * route.get_reference_length()
            + state.arc_length;
    }
    if state.unwrapped_progress > state.best_progress {
        state.best_progress = state.unwrapped_progress;
    }
    state.previous_car_transform = *car_world_transform;
    result
}

/// `TmnfRace_RespawnLocation`: the spawn a respawn press applies (0x00472700
/// `CTrackManiaRace1P::SmallRespawn` -> 0x0047C0D0 RespawnPlayer ->
/// 0x0047BF00 RespawnPlayerVehicle), or `None` when no respawnable
/// checkpoint has been passed: the game then restarts the race
/// (`CGameRace::SetStatus(3)`) rather than moving the car.
pub fn tmnf_race_respawn_location(state: &TmnfRaceState) -> Option<&GmIso4> {
    if state.finished != 0 {
        panic!("tmnf race: cannot respawn a finished race");
    }
    if state.respawn_available != 0 {
        Some(&state.respawn_location)
    } else {
        None
    }
}

/* ---------------------------------------------------------------------------
 * Tests
 * ------------------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{GmSurfEllipsoid, GM_SURF_ELLIPSOID, TreeArena};
    use crate::route::tmnf_route_load;

    const ROUTE_PATH: &str = "/home/z/my-project/tmnf-physics/oracle/routes/A01-Race.tmnfroute";

    fn load_route() -> TmnfRoute {
        tmnf_route_load(ROUTE_PATH, None)
    }

    #[test]
    fn race_reset_and_full_lap() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;

        tmnf_race_reset(&route, &mut state, &car);
        assert_eq!(state.previous_car_transform, car);
        assert_eq!(state.respawn_location, route.start.spawn);
        assert_eq!(state.respawn_available, 0);
        assert_eq!(state.visited_count, 0);
        assert_eq!(state.elapsed_ticks, 0);
        assert_eq!(state.arc_length, 0.0);
        assert_eq!(state.lateral_offset, 0.0);
        assert_eq!(state.centerline_segment, 0);
        assert_eq!(state.projection_segment, 0);
        assert_eq!(state.half_width, 15.0);
        assert_eq!(state.corridor_lateral, 0.0);
        assert_eq!(state.corridor_vertical, 0.0);
        assert_eq!(state.outside_corridor, 0);
        assert_eq!(state.unwrapped_progress, 0.0);
        assert_eq!(state.best_progress, 0.0);
        assert_eq!(state.previous_progress, 0.0);

        // Tick 1: no contacts.
        let r = tmnf_race_step(&route, &mut state, 0, &car);
        assert_eq!(r, TmnfRaceStepResult::default());
        assert_eq!(state.elapsed_ticks, 1);
        assert_eq!(state.visited_count, 0);
        assert_eq!(state.unwrapped_progress, 0.0);

        // Tick 2: checkpoint 0 contact edge.
        let r = tmnf_race_step(&route, &mut state, 1, &car);
        assert_eq!(r.checkpoint_accepted, 1);
        assert_eq!(r.checkpoint_repeated, 0);
        assert_eq!(r.lap_completed, 0);
        assert_eq!(r.finished, 0);
        assert_eq!(r.checkpoint_index, 0);
        assert_eq!(state.visited_count, 1);
        assert_eq!(state.visited_checkpoints, 1);
        assert_eq!(state.passed_race_checkpoints, 1);
        assert_eq!(state.respawn_available, 1);
        assert_eq!(state.respawn_location, route.checkpoints[0].spawn);
        assert_eq!(state.completed_laps, 0);

        // Tick 3: contact held — no new edge, nothing happens.
        let r = tmnf_race_step(&route, &mut state, 1, &car);
        assert_eq!(r, TmnfRaceStepResult::default());
        assert_eq!(state.visited_count, 1);

        // Tick 4: contact released.
        let r = tmnf_race_step(&route, &mut state, 0, &car);
        assert_eq!(r, TmnfRaceStepResult::default());

        // Tick 5: re-touch — repeated, not accepted.
        let r = tmnf_race_step(&route, &mut state, 1, &car);
        assert_eq!(r.checkpoint_accepted, 0);
        assert_eq!(r.checkpoint_repeated, 1);
        assert_eq!(state.visited_count, 1);
        assert_eq!(state.passed_race_checkpoints, 1);

        // Tick 6: checkpoint 1 (no_respawn set: spawn kept).
        let r = tmnf_race_step(&route, &mut state, 2, &car);
        assert_eq!(r.checkpoint_accepted, 1);
        assert_eq!(r.checkpoint_repeated, 0);
        assert_eq!(r.checkpoint_index, 1);
        assert_eq!(state.visited_count, 2);
        assert_eq!(state.visited_checkpoints, 3);
        assert_eq!(state.passed_race_checkpoints, 2);
        assert_eq!(state.respawn_location, route.checkpoints[0].spawn);
        assert_eq!(state.respawn_available, 1);
        assert_eq!(
            tmnf_race_respawn_location(&state),
            Some(&route.checkpoints[0].spawn)
        );

        // Tick 7: finish contact (bit 63) completes the lap and the race.
        let r = tmnf_race_step(&route, &mut state, 1u64 << 63, &car);
        assert_eq!(r.lap_completed, 1);
        assert_eq!(r.finished, 1);
        assert_eq!(r.race_time_ms, 70);
        assert_eq!(state.finished, 1);
        assert_eq!(state.finish_time_ms, 70);
        assert_eq!(state.completed_laps, 1);
        assert_eq!(state.passed_race_checkpoints, 3);
        assert_eq!(state.arc_length, 2212.947265625);
        assert_eq!(state.lateral_offset, 0.0);
        assert_eq!(state.half_width, 15.0);
        assert_eq!(state.centerline_segment, 1106);
        assert_eq!(state.corridor_lateral, 0.0);
        assert_eq!(state.corridor_vertical, 0.0);
        assert_eq!(state.outside_corridor, 0);
        assert_eq!(state.unwrapped_progress, 2212.947265625);
        assert_eq!(state.best_progress, 2212.947265625);
        assert_eq!(state.previous_progress, 0.0);
    }

    #[test]
    fn race_finishes_in_a_single_contact_tick() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;
        tmnf_race_reset(&route, &mut state, &car);
        // Both checkpoints and the finish in one tick (all edges at once).
        let r = tmnf_race_step(&route, &mut state, 3 | (1u64 << 63), &car);
        assert_eq!(r.checkpoint_accepted, 1);
        assert_eq!(r.lap_completed, 1);
        assert_eq!(r.finished, 1);
        assert_eq!(r.race_time_ms, 10);
        assert_eq!(state.finish_time_ms, 10);
        assert_eq!(state.passed_race_checkpoints, 3);
        assert_eq!(state.unwrapped_progress, 2212.947265625);
    }

    #[test]
    fn race_finish_requires_all_checkpoints() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;
        tmnf_race_reset(&route, &mut state, &car);
        // Finish contact with only checkpoint 0 taken: no lap.
        let _ = tmnf_race_step(&route, &mut state, 1, &car);
        let r = tmnf_race_step(&route, &mut state, 1 | (1u64 << 63), &car);
        assert_eq!(r.lap_completed, 0);
        assert_eq!(r.finished, 0);
        assert_eq!(state.completed_laps, 0);
        assert_eq!(state.finished, 0);
        // The finish edge is consumed by trigger_contacts; taking checkpoint
        // 1 afterwards does not retroactively complete the lap.
        let _ = tmnf_race_step(&route, &mut state, 3, &car);
        assert_eq!(state.completed_laps, 0);
    }

    #[test]
    fn race_corridor_freezes_progress_outside() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;
        tmnf_race_reset(&route, &mut state, &car);

        // A teleport (> 32 m in one tick) to far above the track: still
        // projects (full search), but far outside the corridor.
        let high = GmIso4 {
            m: car.m,
            t: [car.t[0] + 300.0, car.t[1] + 400.0, car.t[2] + 300.0],
        };
        let _ = tmnf_race_step(&route, &mut state, 0, &high);
        assert_eq!(state.outside_corridor, 1);
        assert!(state.corridor_vertical > TMNF_RACE_CORRIDOR_ABOVE_METERS);
        // Progress froze at the pre-excursion value.
        assert_eq!(state.unwrapped_progress, 0.0);
        assert_eq!(state.best_progress, 0.0);

        // Back on the start line (another teleport, full search reacquires).
        let r = tmnf_race_step(&route, &mut state, 0, &car);
        assert_eq!(r, TmnfRaceStepResult::default());
        assert_eq!(state.outside_corridor, 0);
        assert_eq!(state.unwrapped_progress, 0.0);
    }

    #[test]
    fn off_track_counter() {
        let mut state = TmnfRaceState::default();
        // 4 wheels, all on the ground plane, grace 5.
        for tick in 1..=4u32 {
            assert!(!tmnf_race_update_off_track(&mut state, 5, 4, 4));
            assert_eq!(state.off_track_ticks, tick);
        }
        assert!(tmnf_race_update_off_track(&mut state, 5, 4, 4));
        assert_eq!(state.off_track_ticks, 5);
        // Stays armed while the count is held at grace.
        assert!(tmnf_race_update_off_track(&mut state, 5, 4, 4));
        // Any wheel on a track surface resets the counter.
        assert!(!tmnf_race_update_off_track(&mut state, 5, 4, 3));
        assert_eq!(state.off_track_ticks, 0);
        // Airborne ticks hold the counter (bouncing on grass cannot evade).
        assert!(!tmnf_race_update_off_track(&mut state, 5, 0, 0));
        assert_eq!(state.off_track_ticks, 0);
        // ...so the full grace is needed again.
        for _ in 1..=4 {
            assert!(!tmnf_race_update_off_track(&mut state, 5, 4, 4));
        }
        assert!(tmnf_race_update_off_track(&mut state, 5, 4, 4));
    }

    #[test]
    #[should_panic(expected = "tmnf race: invalid off-track state")]
    fn off_track_rejects_zero_grace() {
        let mut state = TmnfRaceState::default();
        let _ = tmnf_race_update_off_track(&mut state, 0, 4, 4);
    }

    #[test]
    #[should_panic(expected = "tmnf race: invalid off-track state")]
    fn off_track_rejects_inverted_wheel_counts() {
        let mut state = TmnfRaceState::default();
        let _ = tmnf_race_update_off_track(&mut state, 5, 2, 3);
    }

    #[test]
    fn ground_plane_materials() {
        assert!(tmnf_race_is_ground_plane_material(2)); // Grass
        assert!(tmnf_race_is_ground_plane_material(20)); // WetGrass
        assert!(!tmnf_race_is_ground_plane_material(13)); // Water
        assert!(!tmnf_race_is_ground_plane_material(0)); // Concrete
        assert_eq!(TMNF_MATERIAL_GRASS, 2);
        assert_eq!(TMNF_MATERIAL_WET_GRASS, 20);
    }

    /// A minimal car collision tree: a single root carrying an ellipsoid
    /// surface (the game's car tree is a root with leaf children; a
    /// childless root exercises the same root-level path).
    fn probe_tree(radii: GmVec3) -> (TreeArena, usize) {
        let mut arena = TreeArena::default();
        let root = arena.push_tree(CPlugTree {
            object_ref: 0,
            flags: 0x80,
            box_aligned: GmBoxAligned {
                center: GmVec3::ZERO,
                half_extent: radii,
            },
            local_iso: GmIso4::IDENTITY,
            surface: Some(CPlugSurface {
                geom: GmSurfGeom::Ellipsoid(GmSurfEllipsoid {
                    base: GmSurf {
                        vtable: 0,
                        material_index: 0,
                        surf_type: GM_SURF_ELLIPSOID,
                        reserved: 0,
                    },
                    radii,
                }),
                material_ids: vec![0u8],
                material_count: 1,
            }),
            children: Vec::new(),
        });
        (arena, root)
    }

    #[test]
    fn trigger_contact_at_finish_wall() {
        let route = load_route();
        let finish = &route.finish[0];

        // The trigger's world box (oracle log: center 20.0006485,
        // 116.407112, 176.5 — a thin wall across the road).
        let mut world_box = GmBoxAligned::default();
        world_box.set_mult(&finish.box_aligned, &finish.transform);
        assert!((world_box.center.x - 20.0006485).abs() < 1e-3);
        assert!((world_box.center.y - 116.407112).abs() < 1e-3);
        assert!((world_box.center.z - 176.5).abs() < 1e-3);
        assert!((world_box.half_extent.x - 0.2065401).abs() < 1e-4);
        assert!((world_box.half_extent.y - 3.407114).abs() < 1e-4);
        assert!((world_box.half_extent.z - 13.5).abs() < 1e-4);

        // An ellipsoid touching the wall's +x face from outside (penetrating
        // by a quarter radius), offset off the face quad's diagonal.
        let radii = GmVec3 {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        let car = GmIso4 {
            m: GmIso4::IDENTITY.m,
            t: [
                world_box.center.x + world_box.half_extent.x + 0.25,
                world_box.center.y + 1.0,
                world_box.center.z + 3.0,
            ],
        };
        let (arena, root) = probe_tree(radii);
        assert!(tmnf_race_trigger_contact(finish, &arena.trees, root, &car));

        // Same shape far away: no contact (root AABBs disjoint).
        let far = GmIso4 {
            m: GmIso4::IDENTITY.m,
            t: [
                world_box.center.x + 100.0,
                world_box.center.y,
                world_box.center.z,
            ],
        };
        let (arena2, root2) = probe_tree(radii);
        assert!(!tmnf_race_trigger_contact(finish, &arena2.trees, root2, &far));

        // The whole route against the car: only the finish bit fires.
        let (arena3, root3) = probe_tree(radii);
        let mask = tmnf_race_trigger_contact_mask(&route, &arena3.trees, root3, &car);
        assert_eq!(mask, 1u64 << 63);

        // A car placed at the far probe touches nothing.
        let (arena4, root4) = probe_tree(radii);
        let mask = tmnf_race_trigger_contact_mask(&route, &arena4.trees, root4, &far);
        assert_eq!(mask, 0);
    }

    #[test]
    fn trigger_contact_at_checkpoint() {
        let route = load_route();
        let cp0 = &route.checkpoints[0];
        let mut world_box = GmBoxAligned::default();
        world_box.set_mult(&cp0.box_aligned, &cp0.transform);
        // Oracle log: center 976, 26.6984291, 304.121582.
        assert!((world_box.center.x - 976.0).abs() < 1e-3);
        assert!((world_box.center.y - 26.6984291).abs() < 1e-3);
        assert!((world_box.center.z - 304.121582).abs() < 1e-3);

        let radii = GmVec3 {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        };
        let car = GmIso4 {
            m: GmIso4::IDENTITY.m,
            t: [
                world_box.center.x,
                world_box.center.y + world_box.half_extent.y + 0.25,
                world_box.center.z,
            ],
        };
        let (arena, root) = probe_tree(radii);
        assert!(tmnf_race_trigger_contact(cp0, &arena.trees, root, &car));
        // In the mask this is checkpoint bit 0 only.
        let mask = tmnf_race_trigger_contact_mask(&route, &arena.trees, root, &car);
        assert_eq!(mask, 1);
    }

    #[test]
    #[should_panic(expected = "tmnf race: car collision tree has collisions disabled")]
    fn trigger_contact_requires_collision_enabled_car_tree() {
        let route = load_route();
        let mut arena = TreeArena::default();
        let root = arena.push_tree(CPlugTree {
            object_ref: 0,
            flags: 0,
            box_aligned: GmBoxAligned::default(),
            local_iso: GmIso4::IDENTITY,
            surface: None,
            children: Vec::new(),
        });
        let _ = tmnf_race_trigger_contact(
            &route.finish[0],
            &arena.trees,
            root,
            &GmIso4::IDENTITY,
        );
    }

    #[test]
    #[should_panic(expected = "tmnf race: cannot step a finished race")]
    fn step_after_finish_panics() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;
        tmnf_race_reset(&route, &mut state, &car);
        let _ = tmnf_race_step(&route, &mut state, 3 | (1u64 << 63), &car);
        assert_eq!(state.finished, 1);
        let _ = tmnf_race_step(&route, &mut state, 0, &car);
    }

    #[test]
    #[should_panic(expected = "tmnf race: cannot respawn a finished race")]
    fn respawn_after_finish_panics() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        let car = route.start.spawn;
        tmnf_race_reset(&route, &mut state, &car);
        let _ = tmnf_race_step(&route, &mut state, 3 | (1u64 << 63), &car);
        let _ = tmnf_race_respawn_location(&state);
    }

    #[test]
    fn respawn_location_is_none_before_any_checkpoint() {
        let route = load_route();
        let mut state = TmnfRaceState::default();
        tmnf_race_reset(&route, &mut state, &route.start.spawn);
        assert_eq!(tmnf_race_respawn_location(&state), None);
    }
}
