//! TMNF geometry primitives, transliterated from the 2.11.26 disassembly
//! (`src/gm.h` / `src/gm.c` in the C transcription). Layouts mirror the game's
//! memory exactly — these types alias captured state snapshots byte-for-byte,
//! so every one is `#[repr(C)]` with a compile-time size assertion.
//!
//! # The quaternion trap
//!
//! `GmQuat` stores `(x, y, z, w)` **in that field order** (the bytes), but
//! `GmQuat_SetFromMat3` — the function the dyna state uses — writes the
//! **scalar first**: `q[0] = w`, `q[1..4] = x,y,z`. The field names are the
//! C transcription's own; the array order is what the bytes say. This port
//! keeps the raw layout and converts only at the public API boundary.

use crate::fp::{x87_atan2, x87_fmod};
use std::mem::size_of;

/// 12 bytes. `GmVec3`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
const _: () = assert!(size_of::<GmVec3>() == 12);

/// 16 bytes. Quaternion bytes `(x, y, z, w)` in field order — see the module
/// docs for the scalar-first trap.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmQuat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
const _: () = assert!(size_of::<GmQuat>() == 16);

/// 36 bytes, row-major: rows `{m[0..2]}, {m[3..5]}, {m[6..8]}`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmMat3 {
    pub m: [f32; 9],
}
const _: () = assert!(size_of::<GmMat3>() == 36);

/// 48 bytes: row-major 3x3 `m[0..8]` then translation `t[0..2]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GmIso4 {
    pub m: [f32; 9],
    pub t: [f32; 3],
}
const _: () = assert!(size_of::<GmIso4>() == 48);

/// `x87_dot3_left(a0,b0,a1,b1,a2,b2)` = `((a0*b0 + a1*b1) + a2*b2)`.
/// Kept as a named helper so the groupings below stay visually identical to
/// the transcription.
#[inline(always)]
pub fn dot3_left(a0: f32, b0: f32, a1: f32, b1: f32, a2: f32, b2: f32) -> f32 {
    ((a0 * b0) + (a1 * b1)) + (a2 * b2)
}

impl GmVec3 {
    pub const ZERO: GmVec3 = GmVec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// 0x0045BBA0  `out = A * v + A.t`  (A is [`GmIso4`])
    pub fn set_mult_iso4(&mut self, v: &GmVec3, a: &GmIso4) {
        self.x = dot3_left(a.m[1], v.y, v.x, a.m[0], a.m[2], v.z) + a.t[0];
        self.y = dot3_left(a.m[3], v.x, a.m[4], v.y, a.m[5], v.z) + a.t[1];
        self.z = dot3_left(a.m[6], v.x, a.m[7], v.y, a.m[8], v.z) + a.t[2];
    }

    /// 0x0045BCE0  `out = M * v`  (M is [`GmMat3`], no translation)
    pub fn set_mult_mat3(&mut self, v: &GmVec3, m: &GmMat3) {
        self.x = dot3_left(m.m[1], v.y, m.m[0], v.x, m.m[2], v.z);
        self.y = dot3_left(m.m[3], v.x, m.m[4], v.y, m.m[5], v.z);
        self.z = dot3_left(m.m[6], v.x, m.m[7], v.y, m.m[8], v.z);
    }

    /// 0x0045BC60  `self = A * self + A.t`
    pub fn mult_iso4(&mut self, a: &GmIso4) {
        let (x, y, z) = (self.x, self.y, self.z);
        self.x = ((a.m[2] * z) + ((a.m[0] * x) + (a.m[1] * y))) + a.t[0];
        self.y = dot3_left(a.m[3], x, a.m[4], y, a.m[5], z) + a.t[1];
        self.z = ((z * a.m[8]) + ((y * a.m[7]) + (x * a.m[6]))) + a.t[2];
    }

    /// 0x0045BD40  `self = M^T * self`
    pub fn mult_transpose(&mut self, m: &GmMat3) {
        let (x, y, z) = (self.x, self.y, self.z);
        self.x = (m.m[6] * z) + ((m.m[0] * x) + (m.m[3] * y));
        self.y = dot3_left(m.m[1], x, m.m[4], y, m.m[7], z);
        self.z = (z * m.m[8]) + ((y * m.m[5]) + (x * m.m[2]));
    }

    /// 0x0045BCE0 (in-place form)  `self = M * self`
    pub fn mult_mat3(&mut self, m: &GmMat3) {
        let (x, y, z) = (self.x, self.y, self.z);
        let out_y = dot3_left(m.m[3], x, m.m[4], y, m.m[5], z);
        let out_z = dot3_left(m.m[6], x, m.m[7], y, m.m[8], z);
        self.x = dot3_left(m.m[1], y, m.m[0], x, m.m[2], z);
        self.y = out_y;
        self.z = out_z;
    }

    pub fn as_slice(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn as_bytes(&self) -> [u8; 12] {
        let mut b = [0u8; 12];
        b[0..4].copy_from_slice(&self.x.to_le_bytes());
        b[4..8].copy_from_slice(&self.y.to_le_bytes());
        b[8..12].copy_from_slice(&self.z.to_le_bytes());
        b
    }
}

impl GmMat3 {
    pub const IDENTITY: GmMat3 = GmMat3 {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    pub fn from_iso4_rotation(iso: &GmIso4) -> GmMat3 {
        GmMat3 { m: iso.m }
    }

    /// 0x008E0B10  build rotation matrix from a quaternion.
    ///
    /// **Decoded convention (verified against `GmQuat_SetFromMat3` and
    /// `IntegrateStep`'s quaternion derivative):** the four arguments are the
    /// quaternion in the game's byte order **(w, x, y, z), scalar first** —
    /// the C transcription names them `(qx, qy, qz, qw)` but its formula is
    /// the standard one for `(a, b, c, d) = (w, x, y, z)`. Call sites pass
    /// `(quat.x, quat.y, quat.z, quat.w)`, i.e. the raw field/byte order.
    pub fn set_from_quat(&mut self, a: f32, b: f32, c: f32, d: f32) {
        let two_y = ((b as f64) * 2.0) as f32;
        let two_z = ((c as f64) * 2.0) as f32;
        let two_w = ((d as f64) * 2.0) as f32;
        let xy2 = a * two_y;
        let xz2 = a * two_z;
        let xw2 = a * two_w;
        let yy2 = two_y * b;
        let yz2 = b * two_z;
        let yw2 = b * two_w;
        let zz2 = two_z * c;
        let zw2 = c * two_w;
        let ww2 = two_w * d;

        self.m[0] = (1.0f32 - zz2) - ww2;
        self.m[3] = yz2 + xw2;
        self.m[6] = yw2 - xz2;
        self.m[1] = yz2 - xw2;
        self.m[4] = (1.0f32 - yy2) - ww2;
        self.m[7] = zw2 + xy2;
        self.m[2] = yw2 + xz2;
        self.m[5] = zw2 - xy2;
        self.m[8] = (1.0f32 - yy2) - zz2;
    }

    /// 0x008E0C60  `self = src^T`
    pub fn set_transpose(&mut self, src: &GmMat3) {
        self.m[0] = src.m[0];
        self.m[4] = src.m[4];
        self.m[8] = src.m[8];
        self.m[1] = src.m[3];
        self.m[3] = src.m[1];
        self.m[2] = src.m[6];
        self.m[6] = src.m[2];
        self.m[5] = src.m[7];
        self.m[7] = src.m[5];
    }

    /// 0x008E09F0  `self = self * B`
    pub fn mult(&mut self, b: &GmMat3) {
        let a = self.m;
        self.m[0] = (b.m[2] * a[6]) + ((b.m[1] * a[3]) + (b.m[0] * a[0]));
        self.m[1] = dot3_left(b.m[1], a[4], b.m[0], a[1], b.m[2], a[7]);
        self.m[2] = dot3_left(b.m[0], a[2], b.m[1], a[5], b.m[2], a[8]);
        self.m[3] = dot3_left(b.m[4], a[3], a[0], b.m[3], b.m[5], a[6]);
        self.m[4] = dot3_left(b.m[4], a[4], b.m[3], a[1], b.m[5], a[7]);
        self.m[5] = dot3_left(b.m[4], a[5], b.m[3], a[2], b.m[5], a[8]);
        self.m[6] = (a[6] * b.m[8]) + ((a[0] * b.m[6]) + (a[3] * b.m[7]));
        self.m[7] = dot3_left(a[1], b.m[6], a[4], b.m[7], b.m[8], a[7]);
        self.m[8] = dot3_left(b.m[6], a[2], b.m[7], a[5], b.m[8], a[8]);
    }

    /// 0x008E08A0  `self = A`, then [`mult`](Self::mult)`(B)`.
    pub fn set_mult(&mut self, a: &GmMat3, b: &GmMat3) {
        self.m = a.m;
        self.mult(b);
    }

    /// 0x008E06B0  `self[i][j] = sum_k M[k][i] * self[k][j]`  (`self = M^T * self`)
    pub fn mult_transpose(&mut self, m: &GmMat3) {
        let s = self.m;
        for i in 0..3 {
            for j in 0..3 {
                self.m[i * 3 + j] =
                    dot3_left(m.m[i], s[j], m.m[3 + i], s[3 + j], m.m[6 + i], s[6 + j]);
            }
        }
    }
}

impl GmIso4 {
    pub const IDENTITY: GmIso4 = GmIso4 {
        m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        t: [0.0, 0.0, 0.0],
    };

    /// The iso's rotation block as a bare [`GmMat3`] (copy).
    pub fn rotation(&self) -> GmMat3 {
        GmMat3 { m: self.m }
    }

    /// `this = this * parent`, translation included: rotation as
    /// `GmMat3_Mult`, then translation = `parent * translation + parent.t`.
    pub fn mult(&mut self, parent: &GmIso4) {
        let mut translation = GmVec3 {
            x: self.t[0],
            y: self.t[1],
            z: self.t[2],
        };
        let mut rot = GmMat3 { m: self.m };
        rot.mult(&GmMat3 { m: parent.m });
        self.m = rot.m;
        translation.mult_iso4(parent);
        self.t[0] = translation.x;
        self.t[1] = translation.y;
        self.t[2] = translation.z;
    }

    /// 0x008E2570  `self = source^-1` (UNVALIDATED upstream, kept verbatim).
    pub fn set_inverse(&mut self, source: &GmIso4) {
        let mut rot = GmMat3 { m: self.m };
        rot.set_transpose(&GmMat3 { m: source.m });
        self.m = rot.m;
        let mut translation = GmVec3 {
            x: -source.t[0],
            y: -source.t[1],
            z: -source.t[2],
        };
        translation.mult_mat3(&GmMat3 { m: self.m });
        self.t[0] = translation.x;
        self.t[1] = translation.y;
        self.t[2] = translation.z;
    }
}

impl GmQuat {
    /// 0x008E34C0 — quaternion from a rotation matrix; **scalar first, as the
    /// dyna state stores it** (see module docs). Shepperd's method with the
    /// game's axis successor table.
    pub fn set_from_mat3(&mut self, m: &GmMat3) {
        const NEXT: [usize; 3] = [1, 2, 0]; /* DAT_00d1a86c */
        let m = &m.m;
        // q[0..4] array over the byte layout (x,y,z,w fields in order); the
        // game writes the scalar into q[0].
        let mut q = [0f32; 4];
        let trace = (m[0] + m[4]) + m[8];
        if 0.0f32 < trace {
            let s = (((trace as f64) + 1.0) as f32).sqrt();
            let f = (0.5 / (s as f64)) as f32;
            q[0] = ((s as f64) * 0.5) as f32;
            q[1] = (m[7] - m[5]) * f;
            q[2] = (m[2] - m[6]) * f;
            q[3] = (m[3] - m[1]) * f;
        } else {
            let mut i = if m[0] < m[4] { 1 } else { 0 };
            if m[i * 4] < m[8] {
                i = 2;
            }
            let j = NEXT[i];
            let k = NEXT[j];
            let s = ((((m[i * 4] - (m[k * 4] + m[j * 4])) as f64) + 1.0) as f32)
                .sqrt();
            let f = (0.5 / (s as f64)) as f32;
            q[i + 1] = ((s as f64) * 0.5) as f32;
            q[0] = (m[3 * k + j] - m[3 * j + k]) * f;
            q[j + 1] = (m[3 * i + j] + m[3 * j + i]) * f;
            q[k + 1] = (m[3 * i + k] + m[3 * k + i]) * f;
        }
        self.x = q[0];
        self.y = q[1];
        self.z = q[2];
        self.w = q[3];
    }

    /// 0x008E3120  normalize quaternion in place.
    pub fn normalize(&mut self) {
        let sumsq = (((self.y * self.y) + (self.x * self.x)) + (self.z * self.z))
            + (self.w * self.w);
        let s = sumsq.sqrt();
        let inv = 1.0f32 / s;
        self.x = self.x * inv;
        self.y = self.y * inv;
        self.z = self.z * inv;
        self.w = inv * self.w;
    }
}

/// 0x008E7FD0.
pub fn gmfunc_is_a_number(value: f32) -> bool {
    !value.is_nan()
}

/// 0x00457540. Uses the float sign bit directly, including signed zero/NaN.
pub fn gmfunc_sign(value: f32) -> f32 {
    let bits = value.to_bits();
    if (bits & 0x8000_0000u32) != 0 {
        -1.0f32
    } else {
        1.0f32
    }
}

/// 0x009C1C40 `__CIacos` (dispatcher takes the x87 path in the live image).
pub fn gmfunc_acos(value: f32) -> f32 {
    let product = (1.0f32 + value) * (1.0f32 - value);
    let root = product.sqrt();
    x87_atan2(root, value)
}

/// 0x004575B0. The clamp precedes the `__CIasin` dispatcher.
pub fn gmfunc_asin_safe(value: f32) -> f32 {
    if value < -0.999999f32 {
        return -1.5707964f32;
    }
    if value > 0.999999f32 {
        return 1.5707964f32;
    }
    let product = (1.0f32 + value) * (1.0f32 - value);
    let root = product.sqrt();
    x87_atan2(value, root)
}

/// 0x00575440.
pub fn gmfunc_mod(value: f32, lower: f32, upper: f32) -> f32 {
    if lower < value && value < upper {
        return value;
    }
    let period = upper - lower;
    let relative = value - lower;
    let mut result = x87_fmod(relative, period);
    if result < 0.0f32 {
        result = result + period;
    }
    result + lower
}

/// Helper behind `GmVec3_GetAngle` (0x008E7DC0 is its only caller upstream).
pub(crate) fn normalize_vec3_if_nonzero(value: &mut GmVec3) {
    let length_sq =
        ((value.y * value.y) + (value.x * value.x)) + (value.z * value.z);
    if length_sq > 9.999999439624929e-11f32 {
        let length = length_sq.sqrt();
        let inverse = 1.0f32 / length;
        value.x = inverse * value.x;
        value.y = value.y * inverse;
        value.z = value.z * inverse;
    }
}

/// 0x008E7DC0. Signed angle around the global Y axis. UNVALIDATED upstream.
pub fn gmvec3_get_angle(a: &GmVec3, b: &GmVec3) -> f32 {
    let dot = ((a.x * b.x) + (a.y * b.y)) + (a.z * b.z);
    let guarded_dot = ((dot as f64) * 0.9999900000002526f64) as f32;
    let mut angle = gmfunc_acos(guarded_dot);
    if angle > 9.999999747378752e-06f32 {
        let mut an = *a;
        let mut bn = *b;
        normalize_vec3_if_nonzero(&mut an);
        normalize_vec3_if_nonzero(&mut bn);

        let cross_z = (bn.y * an.x) - (an.y * bn.x);
        let cross_y = (an.z * bn.x) - (an.x * bn.z);
        let cross_x = (bn.z * an.y) - (an.z * bn.y);
        let sign_value = ((cross_z * 0.0f32) + cross_y) + (cross_x * 0.0f32);
        if sign_value < 0.0f32 {
            angle = -angle;
        }
    }
    angle
}
