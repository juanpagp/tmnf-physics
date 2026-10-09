//! Floating-point policy for the bit-exact TMNF physics port.
//!
//! **The model (corrected from golden traces):** the live game ran its x87 FPU
//! with precision control = 24 bits (Direct3D reprogrammed the control word
//! before physics ran). Under PC=24 every x87 arithmetic result is rounded to
//! a 24-bit significand — i.e. **bit-identical to plain IEEE-754 binary32
//! arithmetic** (the wider x87 exponent range is never reached at car-physics
//! magnitudes).
//!
//! Therefore the C transcription's `x87_mul`/`x87_add`/`x87_sub`/`x87_div`
//! helper macros collapse into ordinary Rust `f32` operators: Rust guarantees
//! per-operation IEEE round-to-nearest-even, never contracts `a*b+c` into FMA,
//! and never reassociates. `x87_sqrt` is `f32::sqrt`, `x87_rcp` is
//! `1.0/x` (never `recip()`, which may be approximate).
//!
//! What does *not* collapse — the places where the transcription spells out
//! non-default semantics, preserved here as real functions:
//!
//! 1. [`ftol`] — the game's `_ftol`/FISTP: truncation toward zero with
//!    `0x80000000` (integer indefinite) for NaN/out-of-range. Rust's `as i32`
//!    *saturates*, which is different, so it is spelled out.
//! 2. [`u32_to_x87_float`] — the game's `fild`-based unsigned-to-float
//!    conversion (signed load + conditional `+2^32`).
//! 3. **f64 single-rounding expressions** — where the transcription computes a
//!    multi-operator expression in `double` and rounds once, the Rust port
//!    keeps an explicit `f64` block with one `as f32` at the end. These are
//!    load-bearing: per-op f32 arithmetic is *different arithmetic*.
//! 4. **Transcendentals** — the game called the CRT's double `sin/cos/exp/
//!    atan2` and rounded to f32. `((x as f64).sin()) as f32` calls the same
//!    system libm on glibc targets. Never use `f32::sin` (a different,
//!    single-precision algorithm).

/// `_ftol` / FISTP semantics: truncation toward zero, `i32::MIN`
/// (0x80000000, the integer indefinite) for NaN and out-of-range values.
/// Rust's `as i32` saturates instead — never use it directly on game values.
#[inline]
pub fn ftol(x: f64) -> i32 {
    if x > -2147483649.0 && x < 2147483648.0 {
        x as i32
    } else {
        i32::MIN
    }
}

/// The game's unsigned-int-to-float conversion: `fild` (signed) plus a
/// conditional `+2^32`, each step rounded to f32. Do **not** replace with
/// `v as f32` without proof (the two-step add rounds differently on some
/// inputs).
#[inline]
pub fn u32_to_x87_float(v: u32) -> f32 {
    // NOTE: Rust has no hex-float literals (0x1p32f32 is a syntax error), so
    // exact constants are spelled with from_bits. 0x1p32 == 0x4F800000.
    const TWO_POW_32: f32 = f32::from_bits(0x4F80_0000);
    let signed = v as i32;
    let mut r = signed as f32;
    if signed < 0 {
        r = r + TWO_POW_32;
    }
    r
}

/// `x87_sin`: the CRT's double `sin`, rounded to f32.
#[inline]
pub fn x87_sin(x: f32) -> f32 {
    ((x as f64).sin()) as f32
}

/// `x87_cos`: the CRT's double `cos`, rounded to f32.
#[inline]
pub fn x87_cos(x: f32) -> f32 {
    ((x as f64).cos()) as f32
}

/// `x87_exp`: the CRT's double `exp`, rounded to f32.
#[inline]
pub fn x87_exp(x: f32) -> f32 {
    ((x as f64).exp()) as f32
}

/// `x87_atan2`: the CRT's double `atan2`, rounded to f32.
#[inline]
pub fn x87_atan2(y: f32, x: f32) -> f32 {
    ((y as f64).atan2(x as f64)) as f32
}

/// `x87_fmod`: the CRT's double `fmod`, rounded to f32 (see `GmFunc_Mod`).
#[inline]
pub fn x87_fmod(value: f32, period: f32) -> f32 {
    ((value as f64) % (period as f64)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftol_matches_fistp() {
        assert_eq!(ftol(0.9), 0);
        assert_eq!(ftol(-0.9), 0); // truncation toward zero, not floor
        assert_eq!(ftol(-2.7), -2);
        assert_eq!(ftol(2147483647.5), 2147483647);
        assert_eq!(ftol(-2147483648.0), -2147483648);
        assert_eq!(ftol(2147483648.0), i32::MIN); // out of range
        assert_eq!(ftol(f64::NAN), i32::MIN);
        assert_eq!(ftol(f64::INFINITY), i32::MIN);
    }

    #[test]
    fn u32_to_x87_float_matches_two_step() {
        // The two-step form is what the game did; the identity vs `v as f32`
        // is a separate question and not assumed anywhere.
        for &v in &[
            0u32,
            1,
            0x7fffffff,
            0x80000000,
            0x80000001,
            0xffffffff,
            123456789,
        ] {
            let signed = v as i32;
            let mut r = signed as f32;
            if signed < 0 {
                r = r + f32::from_bits(0x4F80_0000);
            }
            assert_eq!(u32_to_x87_float(v), r);
        }
    }
}
