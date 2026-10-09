//! `CHmsDyna` integration core, transliterated from the 2.11.26 disassembly
//! (`src/hms_dyna.c`, `src/hms_state.h`).
//!
//! [`CHmsStateDyna`] is the 180-byte integrated rigid-body state — pure float
//! data, byte-identical to the game, and the heart of the golden record.
//! [`CHmsDynaParams`] is the byte-identical params block (mass, body inverse
//! inertia, drags, substep length, force-field scale, COM offset) that the
//! vehicle image supplies.
//!
//! `CHmsDyna` itself is *not* byte-mirrored (the C port already rebuilt it for
//! 64-bit); this Rust version owns its state slots by value where the game used
//! pointers (`stateB`, `liveState`), with accessors preserving the semantics.
//!
//! **Quaternion convention** (decoded and cross-verified three ways — see
//! `gm::GmQuat`): the dyna state's `GmQuat` bytes hold the rotation as
//! **(w, x, y, z), scalar first**, in the fields *named* (x, y, z, w).

use crate::gm::*;
use std::mem::{offset_of, size_of};

/* FP constants preserving the original float32 values. */
const DAT_00CDB690: f32 = 9.999999439624929e-11; /* ~1e-10 magnitude epsilon */
const DAT_00CDB67C: f32 = 0.009999999776482582; /* 0.01, max replacement/step */

/// 0xB4 = 180 bytes. `CHmsStateDyna`, the integrated rigid-body state.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CHmsStateDyna {
    pub quat: GmQuat,           /* 0x00  orientation (bytes: w,x,y,z scalar-first) */
    pub rot: GmMat3,            /* 0x10  rotation matrix (from quat) */
    pub pos: GmVec3,            /* 0x34  position, integrated by linVel+linVelAdd */
    pub lin_vel: GmVec3,        /* 0x40  linear velocity, integrated by force */
    pub lin_vel_added: GmVec3,  /* 0x4C  added linear speed accumulator (zeroed) */
    pub ang_vel: GmVec3,        /* 0x58  angular velocity */
    pub force: GmVec3,          /* 0x64  accumulated force */
    pub torque: GmVec3,         /* 0x70  accumulated torque */
    pub inv_inertia_world: GmMat3, /* 0x7C  R * Ibody^-1 * R^T */
    pub tail: [f32; 5],         /* 0xA0  5 dwords, purpose TBD — keep the bytes! */
}
const _: () = assert!(size_of::<CHmsStateDyna>() == 180);

const _: () = assert!(offset_of!(CHmsStateDyna, quat) == 0x00);
const _: () = assert!(offset_of!(CHmsStateDyna, rot) == 0x10);
const _: () = assert!(offset_of!(CHmsStateDyna, pos) == 0x34);
const _: () = assert!(offset_of!(CHmsStateDyna, lin_vel) == 0x40);
const _: () = assert!(offset_of!(CHmsStateDyna, lin_vel_added) == 0x4C);
const _: () = assert!(offset_of!(CHmsStateDyna, ang_vel) == 0x58);
const _: () = assert!(offset_of!(CHmsStateDyna, force) == 0x64);
const _: () = assert!(offset_of!(CHmsStateDyna, torque) == 0x70);
const _: () = assert!(offset_of!(CHmsStateDyna, inv_inertia_world) == 0x7C);

impl Default for CHmsStateDyna {
    fn default() -> Self {
        // Zeroed bytes, identity rotation, identity quaternion — a reasonable
        // base; loaders overwrite everything they care about. The quaternion
        // bytes are (w,x,y,z) scalar-first, so identity is .x = 1.
        CHmsStateDyna {
            quat: GmQuat { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
            rot: GmMat3::IDENTITY,
            pos: GmVec3::ZERO,
            lin_vel: GmVec3::ZERO,
            lin_vel_added: GmVec3::ZERO,
            ang_vel: GmVec3::ZERO,
            force: GmVec3::ZERO,
            torque: GmVec3::ZERO,
            inv_inertia_world: GmMat3::IDENTITY,
            tail: [0.0; 5],
        }
    }
}

impl CHmsStateDyna {
    /// The game treats `rot` (0x10) + `pos` (0x34) — 48 contiguous bytes — as a
    /// `GmIso4` { m, t }. This builds that overlay without unsafe reinterprets.
    pub fn rot_pos_iso(&self) -> GmIso4 {
        GmIso4 {
            m: self.rot.m,
            t: [self.pos.x, self.pos.y, self.pos.z],
        }
    }

    /// Raw little-endian bytes of the state (for the golden record / diffs).
    pub fn to_bytes(&self) -> [u8; 180] {
        let mut b = [0u8; 180];
        let mut o = 0usize;
        for f in self.quat_words() {
            b[o..o + 4].copy_from_slice(&f.to_le_bytes());
            o += 4;
        }
        for f in &self.rot.m {
            b[o..o + 4].copy_from_slice(&f.to_le_bytes());
            o += 4;
        }
        for v in [&self.pos, &self.lin_vel, &self.lin_vel_added, &self.ang_vel,
                  &self.force, &self.torque] {
            for f in [v.x, v.y, v.z] {
                b[o..o + 4].copy_from_slice(&f.to_le_bytes());
                o += 4;
            }
        }
        for f in &self.inv_inertia_world.m {
            b[o..o + 4].copy_from_slice(&f.to_le_bytes());
            o += 4;
        }
        for f in &self.tail {
            b[o..o + 4].copy_from_slice(&f.to_le_bytes());
            o += 4;
        }
        b
    }

    fn quat_words(&self) -> [f32; 4] {
        [self.quat.x, self.quat.y, self.quat.z, self.quat.w]
    }
}

/// `CHmsDynaParams` (68 bytes): the block pointed to by `CHmsDyna+0x108` in
/// the game object. Byte-identical.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CHmsDynaParams {
    pub mass: f32,             /* 0x00 */
    pub inv_inertia_body: GmMat3, /* 0x04 */
    pub drag_linear: f32,      /* 0x28 */
    pub drag_angular: f32,     /* 0x2C */
    pub substep_len: f32,      /* 0x30 */
    pub force_field_scale: f32, /* 0x34 */
    pub com_offset: GmVec3,    /* 0x38 */
}
const _: () = assert!(size_of::<CHmsDynaParams>() == 68);
const _: () = assert!(offset_of!(CHmsDynaParams, inv_inertia_body) == 0x04);
const _: () = assert!(offset_of!(CHmsDynaParams, drag_linear) == 0x28);
const _: () = assert!(offset_of!(CHmsDynaParams, drag_angular) == 0x2C);
const _: () = assert!(offset_of!(CHmsDynaParams, substep_len) == 0x30);
const _: () = assert!(offset_of!(CHmsDynaParams, force_field_scale) == 0x34);
const _: () = assert!(offset_of!(CHmsDynaParams, com_offset) == 0x38);

/// `CHmsDyna`: rigid-body integration and state management. The C port's
/// 64-bit-native layout, further Rustified: the `stateB` / `liveState` pointer
/// slots are owned state slots here.
#[derive(Clone, Debug)]
pub struct CHmsDyna {
    pub clamp_angular: i32,
    pub max_angular_speed: f32,
    pub params: CHmsDynaParams,
    pub temp_state: CHmsStateDyna,
    state_b: CHmsStateDyna,
    live_state: CHmsStateDyna,
    pub replacement_buf: Vec<GmVec3>,
    pub dirty_flag: i32,
    pub mode: i32,
}

impl CHmsDyna {
    pub fn new(params: CHmsDynaParams, mode: i32, clamp_angular: i32,
               max_angular_speed: f32) -> Self {
        CHmsDyna {
            clamp_angular,
            max_angular_speed,
            params,
            temp_state: CHmsStateDyna::default(),
            state_b: CHmsStateDyna::default(),
            live_state: CHmsStateDyna::default(),
            replacement_buf: Vec::new(),
            dirty_flag: 0,
            mode,
        }
    }

    pub fn live(&self) -> &CHmsStateDyna { &self.live_state }
    pub fn live_mut(&mut self) -> &mut CHmsStateDyna { &mut self.live_state }
    pub fn state_b(&self) -> &CHmsStateDyna { &self.state_b }
    pub fn state_b_mut(&mut self) -> &mut CHmsStateDyna { &mut self.state_b }
    pub fn set_live(&mut self, s: CHmsStateDyna) { self.live_state = s; }
    pub fn set_state_b(&mut self, s: CHmsStateDyna) { self.state_b = s; }

    #[inline]
    fn length_sq(x: f32, y: f32, z: f32) -> f32 {
        ((y * y) + (x * x)) + (z * z)
    }

    /// 0x00532D40  temp = *live
    pub fn copy_state_to_temp(&mut self) {
        self.temp_state = self.live_state;
    }

    /// 0x00532D60  *stateB = temp
    pub fn copy_temp_to_state(&mut self) {
        self.state_b = self.temp_state;
    }

    /// 0x00532D20  *stateB = *live
    pub fn validate_dynamic_state(&mut self) {
        self.state_b = self.live_state;
    }

    /// 0x00533EE0
    pub fn get_linear_speed(&self) -> GmVec3 { self.live_state.lin_vel }

    /// 0x00533F30
    pub fn get_angular_speed(&self) -> GmVec3 { self.live_state.ang_vel }

    /// 0x00533A30
    pub fn set_force(&mut self, f: &GmVec3) { self.live_state.force = *f; }

    /// 0x00533A50
    pub fn set_torque(&mut self, t: &GmVec3) { self.live_state.torque = *t; }

    /// 0x00533B80. UNVALIDATED upstream.
    pub fn add_force(&mut self, force: &GmVec3) {
        self.live_state.force.x = self.live_state.force.x + force.x;
        self.live_state.force.y = force.y + self.live_state.force.y;
        self.live_state.force.z = force.z + self.live_state.force.z;
    }

    /// 0x00533BB0.
    pub fn get_force(&self) -> GmVec3 { self.live_state.force }

    /// 0x00533BE0. UNVALIDATED upstream.
    pub fn add_torque(&mut self, torque: &GmVec3) {
        self.live_state.torque.x = self.live_state.torque.x + torque.x;
        self.live_state.torque.y = torque.y + self.live_state.torque.y;
        self.live_state.torque.z = torque.z + self.live_state.torque.z;
    }

    /// 0x00533EC0.
    pub fn set_linear_speed(&mut self, speed: &GmVec3) {
        self.live_state.lin_vel = *speed;
    }

    /// 0x00533F10.
    pub fn set_angular_speed(&mut self, speed: &GmVec3) {
        self.live_state.ang_vel = *speed;
    }

    /// 0x00533A70. Adds a world-space force and its moment about the
    /// world-space centre of mass. UNVALIDATED upstream.
    pub fn add_force_at(&mut self, force: &GmVec3, world_point: &GmVec3) {
        self.add_force(force);

        let mut world_com = GmVec3::ZERO;
        world_com.set_mult_iso4(&self.params.com_offset, &self.live_state.rot_pos_iso());
        let lever = GmVec3 {
            x: world_point.x - world_com.x,
            y: world_point.y - world_com.y,
            z: world_point.z - world_com.z,
        };
        let torque = GmVec3 {
            x: (lever.y * force.z) - (lever.z * force.y),
            y: (force.x * lever.z) - (lever.x * force.z),
            z: (force.y * lever.x) - (lever.y * force.x),
        };
        self.add_torque(&torque);
    }

    fn local_to_world_vector(&self, local: &GmVec3) -> GmVec3 {
        let mut world = GmVec3::ZERO;
        world.set_mult_mat3(local, &self.live_state.rot);
        world
    }

    /// 0x00533F60. UNVALIDATED upstream.
    pub fn add_local_force_at(&mut self, force: &GmVec3, local_point: &GmVec3) {
        let world_force = self.local_to_world_vector(force);
        let mut world_point = GmVec3::ZERO;
        world_point.set_mult_iso4(local_point, &self.live_state.rot_pos_iso());
        self.add_force_at(&world_force, &world_point);
    }

    /// 0x00534030. UNVALIDATED upstream.
    pub fn set_local_force(&mut self, force: &GmVec3) {
        let world = self.local_to_world_vector(force);
        self.set_force(&world);
    }

    /// 0x005340B0. UNVALIDATED upstream.
    pub fn add_local_force(&mut self, force: &GmVec3) {
        let world = self.local_to_world_vector(force);
        self.add_force(&world);
    }

    /// 0x00534120. UNVALIDATED upstream.
    pub fn get_local_force(&self) -> GmVec3 {
        let mut force = self.get_force();
        force.mult_transpose(&self.live_state.rot);
        force
    }

    /// 0x00534140. UNVALIDATED upstream.
    pub fn set_local_torque(&mut self, torque: &GmVec3) {
        let world = self.local_to_world_vector(torque);
        self.set_torque(&world);
    }

    /// 0x005341C0. UNVALIDATED upstream.
    pub fn add_local_torque(&mut self, torque: &GmVec3) {
        let world = self.local_to_world_vector(torque);
        self.add_torque(&world);
    }

    /// 0x00534300. UNVALIDATED upstream.
    pub fn add_local_impulse(&mut self, impulse: &GmVec3) {
        let world = self.local_to_world_vector(impulse);
        self.add_impulse(&world);
    }

    /// 0x00534370. UNVALIDATED upstream.
    pub fn set_local_linear_speed(&mut self, speed: &GmVec3) {
        let world = self.local_to_world_vector(speed);
        self.set_linear_speed(&world);
    }

    /// 0x005343E0. UNVALIDATED upstream.
    pub fn get_local_linear_speed(&self) -> GmVec3 {
        let mut speed = self.get_linear_speed();
        speed.mult_transpose(&self.live_state.rot);
        speed
    }

    /// 0x00534400. UNVALIDATED upstream.
    pub fn set_local_angular_speed(&mut self, speed: &GmVec3) {
        let world = self.local_to_world_vector(speed);
        self.set_angular_speed(&world);
    }

    /// 0x00534470. UNVALIDATED upstream.
    pub fn get_local_angular_speed(&self) -> GmVec3 {
        let mut speed = self.get_angular_speed();
        speed.mult_transpose(&self.live_state.rot);
        speed
    }

    /// 0x005334E0  live.pos += d
    pub fn apply_replacement(&mut self, d: &GmVec3) {
        self.live_state.pos.x = self.live_state.pos.x + d.x;
        self.live_state.pos.y = d.y + self.live_state.pos.y;
        self.live_state.pos.z = d.z + self.live_state.pos.z;
    }

    /// 0x00533DD0  velocity at world point `at` = linVel + angVel x (at - worldCOM).
    ///
    /// The angular term is a **load-bearing f64 single-rounding expression**:
    /// the whole term is computed in double and rounded once at the store.
    pub fn get_speed(&self, at: &GmVec3) -> GmVec3 {
        if self.mode == 2 {
            return GmVec3::ZERO;
        }
        let s = &self.live_state;
        let mut out = s.lin_vel;
        if self.mode == 1 {
            let mut com_world = GmVec3::ZERO;
            com_world.set_mult_iso4(&self.params.com_offset, &s.rot_pos_iso());
            let rx = (at.x as f64) - (com_world.x as f64);
            let ry = (at.y as f64) - (com_world.y as f64);
            let rz = (at.z as f64) - (com_world.z as f64);
            out.x = ((out.x as f64)
                + ((s.ang_vel.y as f64) * rz - ry * (s.ang_vel.z as f64)))
                as f32;
            out.y = ((out.y as f64)
                + ((s.ang_vel.z as f64) * rx - rz * (s.ang_vel.x as f64)))
                as f32;
            out.z = ((out.z as f64)
                + ((s.ang_vel.x as f64) * ry - rx * (s.ang_vel.y as f64)))
                as f32;
        }
        out
    }

    /// 0x00533D50  central linear impulse: linVel += J/mass
    pub fn add_impulse(&mut self, j: &GmVec3) {
        if self.mode == 2 {
            return;
        }
        if self.dirty_flag == 0 {
            self.dirty_flag = 1;
        }
        let inv_mass = 1.0f32 / self.params.mass;
        let s = &mut self.live_state;
        s.lin_vel.x = (inv_mass * j.x) + s.lin_vel.x;
        s.lin_vel.y = s.lin_vel.y + (j.y * inv_mass);
        s.lin_vel.z = (inv_mass * j.z) + s.lin_vel.z;
    }

    /// 0x00533C10  impulse J at world point `at`: linear + angular
    pub fn add_impulse_at(&mut self, j: &GmVec3, at: &GmVec3) {
        if self.mode == 2 {
            return;
        }
        if self.dirty_flag == 0 {
            self.dirty_flag = 1;
        }
        let inv_mass = 1.0f32 / self.params.mass;
        {
            let s = &mut self.live_state;
            s.lin_vel.x = s.lin_vel.x + (inv_mass * j.x);
            s.lin_vel.y = s.lin_vel.y + (j.y * inv_mass);
            s.lin_vel.z = (inv_mass * j.z) + s.lin_vel.z;
        }
        if self.mode == 1 {
            let s = &self.live_state;
            let mut com_world = GmVec3::ZERO;
            com_world.set_mult_iso4(&self.params.com_offset, &s.rot_pos_iso());
            let rx = at.x - com_world.x;
            let ry = at.y - com_world.y;
            let rz = at.z - com_world.z;
            let mut tau = GmVec3 {
                x: (ry * j.z) - (rz * j.y),
                y: (rz * j.x) - (rx * j.z),
                z: (rx * j.y) - (j.x * ry),
            };
            tau.mult_mat3(&s.inv_inertia_world);
            let s = &mut self.live_state;
            // Single-op F() adds: identical to f32 addition.
            s.ang_vel.x = s.ang_vel.x + tau.x;
            s.ang_vel.y = s.ang_vel.y + tau.y;
            s.ang_vel.z = tau.z + s.ang_vel.z;
        }
    }

    /// 0x00533510  one integration substep: out = integrate(in, dt)
    pub fn integrate_step(&self, input: &CHmsStateDyna, out: &mut CHmsStateDyna,
                          dt: f32) {
        if self.mode == 2 {
            *out = *input;
            return;
        }

        let inv_mass = 1.0f32 / self.params.mass;
        let force_x = input.force.x * inv_mass;
        let force_y = input.force.y * inv_mass;
        let force_z = inv_mass * input.force.z;
        let pos_dx = input.lin_vel.x * dt;
        let pos_dy = input.lin_vel.y * dt;
        let pos_dz = input.lin_vel.z * dt;

        out.pos.x = pos_dx + input.pos.x;
        out.pos.y = input.pos.y + pos_dy;
        out.pos.z = input.pos.z + pos_dz;
        let added_dx = input.lin_vel_added.x * dt;
        let added_dy = input.lin_vel_added.y * dt;
        let added_dz = input.lin_vel_added.z * dt;
        out.pos.x = out.pos.x + added_dx;
        out.pos.y = out.pos.y + added_dy;
        out.pos.z = out.pos.z + added_dz;
        out.lin_vel_added.z = 0.0;
        out.lin_vel_added.y = 0.0;
        out.lin_vel_added.x = 0.0;
        out.lin_vel.x = input.lin_vel.x + (force_x * dt);
        out.lin_vel.y = input.lin_vel.y + (force_y * dt);
        out.lin_vel.z = input.lin_vel.z + (dt * force_z);

        if self.mode != 0 {
            /* Iinv_world * torque */
            let mut ang_acc = GmVec3::ZERO;
            ang_acc.set_mult_mat3(&input.torque, &input.inv_inertia_world);

            let (wx, wy, wz) =
                (input.ang_vel.x, input.ang_vel.y, input.ang_vel.z);
            if Self::length_sq(wx, wy, wz) <= DAT_00CDB690 {
                out.rot.m = input.rot.m;
                out.quat = input.quat;
            } else {
                /* Quaternion bytes are (w,x,y,z) scalar-first in fields named
                 * (x,y,z,w): field .x = w, .y = x, .z = y, .w = z. Transcribed
                 * verbatim from the C, which reads the fields as q[0..3]. */
                let qx = input.quat.x; /* w */
                let qy = input.quat.y; /* x */
                let qz = input.quat.z; /* y */
                let qw = input.quat.w; /* z */
                let dqx = (((-wx * qy) - (wy * qz)) - (qw * wz)) * 0.5f32;
                let dqy = (((qx * wx) + (wy * qw)) - (qz * wz)) * 0.5f32;
                let dqz = (((wy * qx) - (qw * wx)) + (wz * qy)) * 0.5f32;
                let dqw = (((qz * wx) - (wy * qy)) + (qx * wz)) * 0.5f32;
                out.quat = input.quat;
                out.quat.x = (dqx * dt) + out.quat.x;
                out.quat.y = (dqy * dt) + out.quat.y;
                out.quat.z = (dqz * dt) + out.quat.z;
                out.quat.w = (dt * dqw) + out.quat.w;
                out.quat.normalize();
                /* GmMat3_SetFromQuat takes its arguments as (w,x,y,z): the C
                 * passes (quat.x, quat.y, quat.z, quat.w) — the byte order. */
                out.rot.set_from_quat(out.quat.x, out.quat.y, out.quat.z,
                                      out.quat.w);
                let mut com_old = GmVec3::ZERO;
                com_old.set_mult_mat3(&self.params.com_offset, &input.rot);
                let mut com_new = GmVec3::ZERO;
                com_new.set_mult_mat3(&self.params.com_offset, &out.rot);
                let com_dx = com_new.x - com_old.x;
                let com_dy = com_new.y - com_old.y;
                let com_dz = com_new.z - com_old.z;
                out.pos.x = out.pos.x - com_dx;
                out.pos.y = out.pos.y - com_dy;
                out.pos.z = out.pos.z - com_dz;
            }
            out.ang_vel.x = input.ang_vel.x + (dt * ang_acc.x);
            out.ang_vel.y = input.ang_vel.y + (ang_acc.y * dt);
            out.ang_vel.z = input.ang_vel.z + (dt * ang_acc.z);

            if self.clamp_angular != 0 {
                let mx = self.max_angular_speed;
                let nsq = Self::length_sq(out.ang_vel.x, out.ang_vel.y,
                                          out.ang_vel.z);
                if (mx * mx) < nsq {
                    let s = nsq.sqrt();
                    let ratio = mx / s;
                    out.ang_vel.x = out.ang_vel.x * ratio;
                    out.ang_vel.y = ratio * out.ang_vel.y;
                    out.ang_vel.z = ratio * out.ang_vel.z;
                }
            }

            out.inv_inertia_world.set_transpose(&out.rot);
            /* GmMat3_Mult(self, B): self = self * B, against params blocks. */
            let mut iw = GmMat3 { m: out.inv_inertia_world.m };
            iw.mult(&self.params.inv_inertia_body);
            iw.mult(&GmMat3 { m: out.rot.m });
            out.inv_inertia_world = iw;
            return;
        }

        out.rot.m = input.rot.m;
    }

    /// 0x00535D00  integrate live state one step, then clear the replacement
    /// buffer. UNVALIDATED upstream.
    pub fn do_pre_collision_dynamic(&mut self, dt: f32) {
        let local = self.live_state;
        // `out` starts as a copy of live: the unwritten fields (force, torque,
        // tail) must retain live's values, exactly as the C's aliasing does.
        let mut out = self.live_state;
        self.integrate_step(&local, &mut out, dt);
        self.live_state = out;
        self.replacement_buf.clear();
    }

    /// 0x00535FC0  append a penetration-correction vector and mark dirty.
    /// UNVALIDATED upstream.
    pub fn add_replacement(&mut self, d: &GmVec3) {
        if self.dirty_flag == 0 {
            self.dirty_flag = 1;
        }
        self.replacement_buf.push(*d);
    }

    /// 0x00535D50  reduce accumulated replacement vectors into one clamped
    /// vector. UNVALIDATED upstream; transcribed from the decompiler order.
    pub fn compute_synthetized_replacement(&self) -> GmVec3 {
        let b = &self.replacement_buf;
        if b.is_empty() {
            return GmVec3::ZERO;
        }
        let (mut ax, mut ay, mut az) = (b[0].x, b[0].y, b[0].z);
        for p in &b[1..] {
            let (px, py, pz) = (p.x, p.y, p.z);
            let mut dot = ((ax * px) + (ay * py)) + (az * pz);
            let mag = Self::length_sq(ax, ay, az);
            if dot > 0.0f32 && mag > DAT_00CDB690 {
                if mag < dot {
                    dot = mag;
                }
                dot = dot / mag;
                ax = ax - (dot * ax);
                ay = ay - (dot * ay);
                az = az - (dot * az);
            }
            ax = px + ax;
            ay = ay + py;
            az = az + pz;
        }
        let mag = Self::length_sq(ax, ay, az);
        if mag <= DAT_00CDB67C * DAT_00CDB67C {
            return GmVec3::ZERO;
        }
        let s = mag.sqrt();
        let inv = 1.0f32 / s;
        let nx = ax * inv;
        let ny = ay * inv;
        let nz = az * inv;
        GmVec3 {
            x: ax - (DAT_00CDB67C * nx),
            y: ay - (DAT_00CDB67C * ny),
            z: az - (DAT_00CDB67C * nz),
        }
    }

    /// 0x00536400  apply the synthesized replacement to the live position.
    /// UNVALIDATED upstream.
    pub fn do_post_collision_dynamic(&mut self) {
        let repl = self.compute_synthetized_replacement();
        self.apply_replacement(&repl);
    }
}

impl CHmsStateDyna {
    /// Reads the 180-byte game state (little-endian, field order as above).
    pub fn from_bytes(b: &[u8; 180]) -> Self {
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let vec = |o: usize| GmVec3 { x: f(o), y: f(o + 4), z: f(o + 8) };
        let mat = |o: usize| {
            let mut m = [0.0f32; 9];
            for (i, item) in m.iter_mut().enumerate() {
                *item = f(o + 4 * i);
            }
            GmMat3 { m }
        };
        CHmsStateDyna {
            quat: GmQuat { x: f(0x00), y: f(0x04), z: f(0x08), w: f(0x0c) },
            rot: mat(0x10),
            pos: vec(0x34),
            lin_vel: vec(0x40),
            lin_vel_added: vec(0x4c),
            ang_vel: vec(0x58),
            force: vec(0x64),
            torque: vec(0x70),
            inv_inertia_world: mat(0x7c),
            tail: [f(0xa0), f(0xa4), f(0xa8), f(0xac), f(0xb0)],
        }
    }
}

impl CHmsDynaParams {
    /// Reads the 0x5C-byte params block.
    pub fn from_bytes(b: &[u8]) -> Self {
        let f = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let mut m = [0.0f32; 9];
        for (i, item) in m.iter_mut().enumerate() {
            *item = f(0x04 + 4 * i);
        }
        CHmsDynaParams {
            mass: f(0x00),
            inv_inertia_body: GmMat3 { m },
            drag_linear: f(0x28),
            drag_angular: f(0x2c),
            substep_len: f(0x30),
            force_field_scale: f(0x34),
            com_offset: GmVec3 { x: f(0x38), y: f(0x3c), z: f(0x40) },
        }
    }

    /// Writes the 0x5C-byte params block.
    pub fn to_bytes(&self) -> [u8; 0x5c] {
        let mut b = [0u8; 0x5c];
        b[0..4].copy_from_slice(&self.mass.to_le_bytes());
        for i in 0..9 {
            b[4 + 4 * i..8 + 4 * i]
                .copy_from_slice(&self.inv_inertia_body.m[i].to_le_bytes());
        }
        b[0x28..0x2c].copy_from_slice(&self.drag_linear.to_le_bytes());
        b[0x2c..0x30].copy_from_slice(&self.drag_angular.to_le_bytes());
        b[0x30..0x34].copy_from_slice(&self.substep_len.to_le_bytes());
        b[0x34..0x38].copy_from_slice(&self.force_field_scale.to_le_bytes());
        b[0x38..0x44].copy_from_slice(&self.com_offset.as_bytes());
        b
    }
}
