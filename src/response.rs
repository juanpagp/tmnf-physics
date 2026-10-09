//! Collision response (`src/collision_response.c`), transliterated.
//!
//! The C used function-pointer resolvers (`resolve_body` / `resolve_material`
//! with a `void *user`) to decouple the response from the world. Here the
//! same seam is a trait ([`ResponseCtx`]) implemented by the world; the
//! FP kernels are free functions transliterated exactly.
//!
//! `CHmsPhysicalContact` carries `corpus_ref` ids instead of body pointers.

use crate::buffer::{shms_physical_collision_compare, CFastBufferShmsPhysicalCollision};
use crate::collision::{GmCollision, SHmsPhysicalCollision};
use crate::dyna::CHmsDyna;
use crate::gm::*;

const RESPONSE_EPSILON: f32 = 1.0e-5;
const CENTRAL_IMPULSE_FLAG: u32 = 0x1000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CPlugSurfaceMaterialData {
    pub friction: f32,
    pub restitution: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CHmsResponseMaterial {
    pub category: u32,
    pub response_mode: u32,
    pub side_enabled: [u32; 2],
}

/// What a body's contact sink dispatches to (the C stored an
/// `AbsorbContact` fn pointer + user data). `None` = no sink.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorbSink {
    None,
    VehicleBody,
    VehicleWheel(u32),
}

/// A response body. `dyna` is an index into the world's dyna arena (the
/// vehicle's), or `None` for static bodies.
#[derive(Clone, Copy, Debug)]
pub struct CHmsResponseBody {
    pub corpus_ref: u32,
    pub classification_flags: u32,
    pub response_flags: u32,
    pub response_weight: f32,
    pub iso: GmIso4,
    pub dyna: Option<usize>,
    pub has_contact_sink: bool,
    pub sink: AbsorbSink,
}

impl CHmsResponseBody {
    #[inline]
    pub fn local_contact_mode(&self) -> u32 {
        (self.classification_flags >> 17) & 3
    }

    #[inline]
    pub fn rank(&self) -> u32 {
        (self.classification_flags >> 11) & 3
    }

    #[inline]
    pub fn category(&self) -> u32 {
        (self.classification_flags >> 13) & 0xf
    }
}

/// One absorbed contact (pointers replaced by corpus refs).
#[derive(Clone, Debug, Default)]
pub struct CHmsPhysicalContact {
    pub body_corpus_ref: u32,
    pub tree_ref: u32,
    pub surface_material: u16,
    pub normal: GmVec3,
    pub position: GmVec3,
    pub relative_speed: GmVec3,
    pub replacement: GmVec3,
    pub accepted: u32,
    pub other_body_corpus_ref: Option<u32>,
    pub other_tree_ref: u32,
    pub other_surface_material: u16,
}

/// The seam the world implements (the C's resolver function pointers).
pub trait ResponseCtx {
    /// Resolve a corpus ref to its body (panics if unknown, like the C).
    fn body(&self, corpus_ref: u32) -> &CHmsResponseBody;
    /// The body's world iso (static iso, or the dyna live state's
    /// rot+pos overlay), by value.
    fn body_iso(&self, corpus_ref: u32) -> GmIso4;
    /// 0x00533DD0 inlined with its PC=24 schedule (f32 per-op, unlike the
    /// dyna method's f64 intermediates).
    fn get_speed(&self, corpus_ref: u32, position: &GmVec3) -> GmVec3;
    /// `CHmsDyna_AddReplacement` routed to the body's dyna (no-op static).
    fn add_replacement(&mut self, corpus_ref: u32, d: &GmVec3);
    /// `solve_body_impulse` routed to the body's dyna (no-op static).
    fn solve_body_impulse(&mut self, corpus_ref: u32, collision: &GmCollision,
                          speed: &GmVec3, friction_product: f32, restitution: f32);
    /// The body's contact sink (no-op when absent).
    fn absorb_contact(&mut self, corpus_ref: u32, contact: &mut CHmsPhysicalContact);
    /// The response materials table (resolve_material).
    fn response_material(&self, material_ref: u32) -> &CHmsResponseMaterial;
    /// The 31-entry surface material data.
    fn surface_materials(&self) -> &[CPlugSurfaceMaterialData];
    /// The collision buffer being responded to (sorted in place).
    fn collisions(&mut self) -> &mut CFastBufferShmsPhysicalCollision;
}

/// 0x00533DD0, inlined with its PC=24 operation schedule (per-op f32 —
/// deliberately different from `CHmsDyna::get_speed`'s f64 intermediates).
pub fn response_get_speed(
    body: &CHmsResponseBody,
    dyna: Option<&CHmsDyna>,
    position: &GmVec3,
) -> GmVec3 {
    let dyna = match dyna {
        None => return GmVec3::ZERO,
        Some(d) => d,
    };
    if dyna.mode == 2 {
        return GmVec3::ZERO;
    }
    let state = dyna.live();
    let mut speed = state.lin_vel;
    if dyna.mode == 1 {
        let mut center_of_mass = GmVec3::ZERO;
        center_of_mass.set_mult_iso4(&dyna.params.com_offset, &state.rot_pos_iso());
        let radius_x = position.x - center_of_mass.x;
        let radius_y = position.y - center_of_mass.y;
        let radius_z = position.z - center_of_mass.z;
        let cross_x = (radius_z * state.ang_vel.y) - (radius_y * state.ang_vel.z);
        let cross_y = (radius_x * state.ang_vel.z) - (radius_z * state.ang_vel.x);
        let cross_z = (radius_y * state.ang_vel.x) - (radius_x * state.ang_vel.y);
        speed.x = speed.x + cross_x;
        speed.y = speed.y + cross_y;
        speed.z = speed.z + cross_z;
    }
    speed
}

fn length_squared(vector: &GmVec3) -> f32 {
    ((vector.x * vector.x) + (vector.y * vector.y)) + (vector.z * vector.z)
}

fn dot_collision_normal(normal: &GmVec3, vector: &GmVec3) -> f32 {
    (normal.z * vector.z) + ((normal.x * vector.x) + (normal.y * vector.y))
}

fn cross(a: &GmVec3, b: &GmVec3) -> GmVec3 {
    GmVec3 {
        x: (a.y * b.z) - (a.z * b.y),
        y: (a.z * b.x) - (a.x * b.z),
        z: (a.x * b.y) - (a.y * b.x),
    }
}

/// 0x0087D9A0  Combines two material restitution coefficients.
/// UNVALIDATED upstream.
pub fn restitution_coef_with(
    this: &CPlugSurfaceMaterialData,
    other: &CPlugSurfaceMaterialData,
) -> f32 {
    if this.restitution <= 0.0f32 {
        if 0.0f32 < other.restitution {
            return this.restitution;
        }
        return other.restitution + this.restitution;
    }
    if 0.0f32 < other.restitution {
        return other.restitution * this.restitution;
    }
    other.restitution
}

/// `make_contact` (transcribed): builds a contact for the body when its side
/// is enabled and it has a sink.
pub fn make_contact(
    body: &CHmsResponseBody,
    body_iso: &GmIso4,
    tree_ref: u32,
    surface_material_index: u16,
    other_body: &CHmsResponseBody,
    other_tree_ref: u32,
    other_surface_material: u16,
    collision: &GmCollision,
    side_enabled: u32,
) -> Option<CHmsPhysicalContact> {
    if side_enabled == 0 || body.has_contact_sink == false {
        return None;
    }
    let mut contact = CHmsPhysicalContact {
        body_corpus_ref: body.corpus_ref,
        tree_ref,
        surface_material: surface_material_index,
        other_body_corpus_ref: Some(other_body.corpus_ref),
        other_tree_ref,
        other_surface_material,
        ..Default::default()
    };
    if body.local_contact_mode() > 1 {
        let mut normal = collision.normal;
        normal.mult_transpose(&GmMat3 { m: body_iso.m });
        contact.normal = normal;
        let mut position = GmVec3 {
            x: collision.position.x - body_iso.t[0],
            y: collision.position.y - body_iso.t[1],
            z: collision.position.z - body_iso.t[2],
        };
        position.mult_transpose(&GmMat3 { m: body_iso.m });
        contact.position = position;
    }
    Some(contact)
}

/// `solve_body_impulse` (transcribed). The dyna is mutated for impulses.
pub fn solve_body_impulse(
    body: &CHmsResponseBody,
    dyna: Option<&mut CHmsDyna>,
    collision: &GmCollision,
    speed: &GmVec3,
    friction_product: f32,
    restitution: f32,
) {
    let mut dyna = match dyna {
        None => return,
        Some(d) => d,
    };

    let normal_speed = dot_collision_normal(&collision.normal, speed);
    let projection = GmVec3 {
        x: collision.normal.x * normal_speed,
        y: collision.normal.y * normal_speed,
        z: normal_speed * collision.normal.z,
    };
    let mut tangent = GmVec3 {
        x: speed.x - projection.x,
        y: speed.y - projection.y,
        z: speed.z - projection.z,
    };

    let projection_length = length_squared(&projection).sqrt();
    let tangent_length = length_squared(&tangent).sqrt();
    let tangent_limit = projection_length * friction_product;
    if tangent_limit < tangent_length {
        let scale = tangent_limit / tangent_length;
        tangent.x = scale * tangent.x;
        tangent.y = tangent.y * scale;
        tangent.z = scale * tangent.z;
    }

    let mut impulse_direction = GmVec3 {
        x: -(tangent.x + projection.x),
        y: -(tangent.y + projection.y),
        z: -(tangent.z + projection.z),
    };
    let speed_length = length_squared(&impulse_direction).sqrt();
    if !(RESPONSE_EPSILON < speed_length) {
        return;
    }

    let inverse_speed = 1.0f32 / speed_length;
    impulse_direction.x = inverse_speed * impulse_direction.x;
    impulse_direction.y = impulse_direction.y * inverse_speed;
    impulse_direction.z = inverse_speed * impulse_direction.z;

    let numerator = (restitution + 1.0f32) * speed_length;
    let mut denominator = 1.0f32 / dyna.params.mass;

    if dyna.mode == 1 && (body.response_flags & CENTRAL_IMPULSE_FLAG) == 0 {
        let state = dyna.live();
        let mut center_of_mass = GmVec3::ZERO;
        center_of_mass.set_mult_iso4(&dyna.params.com_offset, &state.rot_pos_iso());
        let radius = GmVec3 {
            x: collision.position.x - center_of_mass.x,
            y: collision.position.y - center_of_mass.y,
            z: collision.position.z - center_of_mass.z,
        };
        let radius_cross_normal = cross(&radius, &impulse_direction);
        let mut inertia_cross = GmVec3::ZERO;
        inertia_cross.set_mult_mat3(&radius_cross_normal, &state.inv_inertia_world);
        let angular = cross(&inertia_cross, &radius);
        let rotational = ((angular.x * impulse_direction.x)
            + (angular.y * impulse_direction.y))
            + (angular.z * impulse_direction.z);
        denominator = rotational + denominator;
    }

    let scale = numerator / denominator;
    let impulse = GmVec3 {
        x: impulse_direction.x * scale,
        y: impulse_direction.y * scale,
        z: scale * impulse_direction.z,
    };
    if (body.response_flags & CENTRAL_IMPULSE_FLAG) == 0 {
        dyna.add_impulse_at(&impulse, &collision.position);
    } else {
        dyna.add_impulse(&impulse);
    }
}

/// `compute_replacements` (transcribed).
pub fn compute_replacements(
    body1: &CHmsResponseBody,
    body2: &CHmsResponseBody,
    normal: &GmVec3,
) -> (GmVec3, GmVec3) {
    let rank1 = body1.rank();
    let rank2 = body2.rank();
    if rank2 < rank1 {
        return (
            GmVec3 { x: -normal.x, y: -normal.y, z: -normal.z },
            GmVec3::ZERO,
        );
    }
    if rank1 < rank2 {
        return (GmVec3::ZERO, *normal);
    }

    let inverse_total = 1.0f32 / (body2.response_weight + body1.response_weight);
    let scale1 = -body2.response_weight * inverse_total;
    let replacement1 = GmVec3 {
        x: normal.x * scale1,
        y: scale1 * normal.y,
        z: scale1 * normal.z,
    };
    let scale2 = body1.response_weight * inverse_total;
    let replacement2 = GmVec3 {
        x: normal.x * scale2,
        y: scale2 * normal.y,
        z: scale2 * normal.z,
    };
    (replacement1, replacement2)
}

/// `prepare_solve_contact` (transcribed). Returns the replacement in world
/// space; `relative_speed` is updated in place as the C does.
#[allow(clippy::too_many_arguments)]
pub fn prepare_solve_contact(
    ctx: &mut dyn ResponseCtx,
    contact: &mut CHmsPhysicalContact,
    replacement: &GmVec3,
    relative_speed: &mut GmVec3,
    first_body: bool,
) -> GmVec3 {
    contact.accepted = 1;
    let body_iso = ctx.body_iso(contact.body_corpus_ref);
    let rotation = GmMat3 { m: body_iso.m };
    let mut contact_replacement = *replacement;
    contact_replacement.mult_transpose(&rotation);
    contact.replacement = contact_replacement;
    if first_body {
        contact.relative_speed = GmVec3 {
            x: -relative_speed.x,
            y: -relative_speed.y,
            z: -relative_speed.z,
        };
    } else {
        contact.relative_speed = *relative_speed;
    }
    contact.relative_speed.mult_transpose(&rotation);
    ctx.absorb_contact(contact.body_corpus_ref, contact);
    contact.relative_speed.mult_mat3(&rotation);
    if first_body {
        relative_speed.x = -contact.relative_speed.x;
        relative_speed.y = -contact.relative_speed.y;
        relative_speed.z = -contact.relative_speed.z;
    }
    let mut replacement_world = GmVec3::ZERO;
    replacement_world.set_mult_mat3(&contact.replacement, &rotation);
    replacement_world
}

/// 0x00548BF0  Resolves replacement, contact callbacks, friction, and
/// impulse. VALIDATED upstream: 2,000 graph-complete A01 records.
/// (The record is mutated in place — `GmCollision_Neg` before body2's
/// impulse — exactly as the C does inside the buffer.)
pub fn solve_impulse(
    ctx: &mut dyn ResponseCtx,
    physical: &mut SHmsPhysicalCollision,
    contact1: &mut Option<CHmsPhysicalContact>,
    contact2: &mut Option<CHmsPhysicalContact>,
) {
    let body1 = *ctx.body(physical.corpus1);
    let body2 = *ctx.body(physical.corpus2);
    let surface_materials = ctx.surface_materials();
    let surface1 = surface_materials[physical.collision.material1 as usize];
    let surface2 = surface_materials[physical.collision.material2 as usize];
    let restitution = restitution_coef_with(&surface1, &surface2);

    let (replacement1, replacement2) =
        compute_replacements(&body1, &body2, &physical.collision.separation);

    let speed1 = ctx.get_speed(physical.corpus1, &physical.collision.position);
    let speed2 = ctx.get_speed(physical.corpus2, &physical.collision.position);
    let mut relative_speed = GmVec3 {
        x: speed2.x - speed1.x,
        y: speed2.y - speed1.y,
        z: speed2.z - speed1.z,
    };

    let mut replacement1_world = replacement1;
    if let Some(contact1) = contact1.as_mut() {
        replacement1_world = prepare_solve_contact(
            ctx, contact1, &replacement1, &mut relative_speed, true,
        );
    }
    ctx.add_replacement(physical.corpus1, &replacement1_world);
    let mut replacement2_world = replacement2;
    if let Some(contact2) = contact2.as_mut() {
        replacement2_world = prepare_solve_contact(
            ctx, contact2, &replacement2, &mut relative_speed, false,
        );
    }
    ctx.add_replacement(physical.corpus2, &replacement2_world);

    if contact1.as_ref().map(|c| c.accepted == 0).unwrap_or(false)
        || contact2.as_ref().map(|c| c.accepted == 0).unwrap_or(false)
    {
        return;
    }

    let friction_product = surface1.friction * surface2.friction;
    ctx.solve_body_impulse(physical.corpus1, &physical.collision, &speed1,
                           friction_product, restitution);
    // GmCollision_Neg in place, as the C does.
    crate::collision::gm_collision_neg(&mut physical.collision);
    ctx.solve_body_impulse(physical.corpus2, &physical.collision, &speed2,
                           friction_product, restitution);
}

/// `fill_external_contact_speed` (transcribed).
pub fn fill_external_contact_speed(
    ctx: &mut dyn ResponseCtx,
    contact: &mut CHmsPhysicalContact,
    relative_speed: &GmVec3,
    first_body: bool,
) {
    let body = *ctx.body(contact.body_corpus_ref);
    if body.local_contact_mode() > 1 {
        if first_body {
            contact.relative_speed = GmVec3 {
                x: -relative_speed.x,
                y: -relative_speed.y,
                z: -relative_speed.z,
            };
        } else {
            contact.relative_speed = *relative_speed;
        }
        let body_iso = ctx.body_iso(contact.body_corpus_ref);
        contact.relative_speed.mult_transpose(&GmMat3 { m: body_iso.m });
    }
    ctx.absorb_contact(contact.body_corpus_ref, contact);
}

/// 0x005497C0  Sorts and dispatches all detected physical collisions.
/// VALIDATED upstream: 2,000 graph-complete A01 records.
pub fn compute_collision_response(ctx: &mut dyn ResponseCtx) {

    {
        let collisions = ctx.collisions();
        collisions.qsort(shms_physical_collision_compare);
    }

    let count = ctx.collisions().count() as usize;
    for i in 0..count {
        let mut physical = *ctx.collisions().at(i);
        let body1 = *ctx.body(physical.corpus1);
        let body2 = *ctx.body(physical.corpus2);
        let material = *ctx.response_material(physical.material);

        let side1 = (body1.category() != material.category) as u32;
        let iso1 = ctx.body_iso(physical.corpus1);
        let iso2 = ctx.body_iso(physical.corpus2);
        let mut contact1 = make_contact(
            &body1,
            &iso1,
            physical.tree1,
            physical.collision.material1,
            &body2,
            physical.tree2,
            physical.collision.material2,
            &physical.collision,
            material.side_enabled[side1 as usize],
        );
        let mut contact2 = make_contact(
            &body2,
            &iso2,
            physical.tree2,
            physical.collision.material2,
            &body1,
            // NB: the C passes `collision->tree2` here as well — a quirk of
            // the transcription, preserved verbatim.
            physical.tree2,
            physical.collision.material1,
            &physical.collision,
            material.side_enabled[(1 - side1) as usize],
        );

        if material.response_mode != 0 {
            solve_impulse(ctx, &mut physical, &mut contact1, &mut contact2);
            continue;
        }

        let speed1 = ctx.get_speed(physical.corpus1, &physical.collision.position);
        let speed2 = ctx.get_speed(physical.corpus2, &physical.collision.position);
        let relative_speed = GmVec3 {
            x: speed2.x - speed1.x,
            y: speed2.y - speed1.y,
            z: speed2.z - speed1.z,
        };
        if let Some(contact1) = contact1.as_mut() {
            fill_external_contact_speed(ctx, contact1, &relative_speed, true);
        }
        if let Some(contact2) = contact2.as_mut() {
            fill_external_contact_speed(ctx, contact2, &relative_speed, false);
        }
    }
}
