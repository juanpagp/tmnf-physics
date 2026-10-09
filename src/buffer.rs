//! `CFastBuffer` of `SHmsPhysicalCollision` — the collision contact buffer
//! (`src/fastbuffer.c`). Append-order semantics matter: the response iterates
//! in insertion order, and `qsort` is the game's **MSVC quicksort
//! transcription** (0x00547DE0) — an *unstable* sort whose exact swap
//! sequence determines the order of equal elements, so it is ported verbatim
//! over indices.

use crate::collision::{GmCollision, SHmsPhysicalCollision};

/// `SHmsPhysicalCollision` comparison for response processing (0x00547C80).
/// UNVALIDATED upstream; the field order and `<=`/`<` pattern are the
/// transcription's own.
pub fn shms_physical_collision_compare(
    a: &SHmsPhysicalCollision,
    b: &SHmsPhysicalCollision,
) -> i32 {
    // Per field: if !(b <= a) return 1; if b < a return -1; else continue.
    // (NaN fails `b <= a` and returns 1, exactly like the C.)
    fn ord(b: f32, a: f32) -> i32 {
        if !(b <= a) {
            1
        } else if b < a {
            -1
        } else {
            0
        }
    }
    let o = ord(b.collision.position.x, a.collision.position.x);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.position.y, a.collision.position.y);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.position.z, a.collision.position.z);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.normal.x, a.collision.normal.x);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.normal.y, a.collision.normal.y);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.normal.z, a.collision.normal.z);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.separation.x, a.collision.separation.x);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.separation.y, a.collision.separation.y);
    if o != 0 {
        return o;
    }
    let o = ord(b.collision.separation.z, a.collision.separation.z);
    if o != 0 {
        return o;
    }

    if a.collision.flags == 0 && b.collision.flags != 0 {
        return -1;
    }
    1
}

/// The game's collision record buffer. Capacity growth is a plain `Vec`
/// (the 1.5x policy never affects results, only allocation).
#[derive(Clone, Debug, Default)]
pub struct CFastBufferShmsPhysicalCollision {
    pub data: Vec<SHmsPhysicalCollision>,
}

impl CFastBufferShmsPhysicalCollision {
    /// 0x00536DA0. Returns the indexed 0x4c-byte collision record.
    pub fn at(&self, index: usize) -> &SHmsPhysicalCollision {
        &self.data[index]
    }

    pub fn at_mut(&mut self, index: usize) -> &mut SHmsPhysicalCollision {
        &mut self.data[index]
    }

    pub fn count(&self) -> u32 {
        self.data.len() as u32
    }

    /// 0x00537400. Appends one collision record and returns it by index
    /// (borrow rules: fetch the record, write it, repeat).
    pub fn add_new_elem(&mut self) -> usize {
        self.data.push(SHmsPhysicalCollision::default());
        self.data.len() - 1
    }

    /// 0x00547DE0 — MSVC quicksort, transcribed. Indices stand in for the
    /// C's byte pointers; the algorithm, pivots, swaps and iteration order
    /// are identical, so the resulting permutation (including equal-element
    /// order) is identical.
    pub fn qsort(
        &mut self,
        compare: fn(&SHmsPhysicalCollision, &SHmsPhysicalCollision) -> i32,
    ) {
        if self.data.len() < 2 {
            return;
        }

        let mut lo_stack = [0usize; 30];
        let mut hi_stack = [0usize; 30];
        let mut stack_top: i32 = 0;

        let mut lo = 0usize;
        let mut hi = self.data.len() - 1;

        loop {
            let mut count = (hi - lo + 1) as u32;
            while count > 8 {
                let mid = lo + (count >> 1) as usize;

                if compare(&self.data[lo], &self.data[mid]) > 0 {
                    self.data.swap(lo, mid);
                }
                if compare(&self.data[lo], &self.data[hi]) > 0 {
                    self.data.swap(lo, hi);
                }
                if compare(&self.data[mid], &self.data[hi]) > 0 {
                    self.data.swap(mid, hi);
                }

                let mut left = lo;
                let mut right = hi;
                let mut mid = mid;
                loop {
                    if left < mid {
                        loop {
                            left += 1;
                            if mid <= left {
                                break;
                            }
                            if compare(&self.data[left], &self.data[mid]) >= 1 {
                                break;
                            }
                        }
                    }
                    if mid <= left {
                        loop {
                            left += 1;
                            if hi < left {
                                break;
                            }
                            if compare(&self.data[left], &self.data[mid]) >= 1 {
                                break;
                            }
                        }
                    }

                    loop {
                        right -= 1;
                        if right <= mid {
                            break;
                        }
                        if compare(&self.data[right], &self.data[mid]) <= 0 {
                            break;
                        }
                    }

                    if left > right {
                        break;
                    }
                    self.data.swap(left, right);
                    if mid == right {
                        mid = left;
                    }
                }

                right += 1;
                if mid < right {
                    loop {
                        right -= 1;
                        if right <= mid {
                            break;
                        }
                        if compare(&self.data[right], &self.data[mid]) != 0 {
                            break;
                        }
                    }
                    if mid < right {
                        // goto partitioned
                        if right - lo < hi - left {
                            if left < hi {
                                lo_stack[stack_top as usize] = left;
                                hi_stack[stack_top as usize] = hi;
                                stack_top += 1;
                            }
                            hi = right;
                            if right <= lo {
                                break;
                            }
                        } else {
                            if lo < right {
                                lo_stack[stack_top as usize] = lo;
                                hi_stack[stack_top as usize] = right;
                                stack_top += 1;
                            }
                            lo = left;
                            if hi <= left {
                                break;
                            }
                        }
                        count = (hi - lo + 1) as u32;
                        continue;
                    }
                }
                if mid <= right {
                    loop {
                        right -= 1;
                        if right <= lo {
                            break;
                        }
                        if compare(&self.data[right], &self.data[mid]) != 0 {
                            break;
                        }
                    }
                }

                // partitioned:
                if right - lo < hi - left {
                    if left < hi {
                        lo_stack[stack_top as usize] = left;
                        hi_stack[stack_top as usize] = hi;
                        stack_top += 1;
                    }
                    hi = right;
                    if right <= lo {
                        break;
                    }
                } else {
                    if lo < right {
                        lo_stack[stack_top as usize] = lo;
                        hi_stack[stack_top as usize] = right;
                        stack_top += 1;
                    }
                    lo = left;
                    if hi <= left {
                        break;
                    }
                }
                count = (hi - lo + 1) as u32;
            }

            if lo < hi {
                // shortsort: selection sort, transcribed.
                let mut hi_s = hi;
                while lo < hi_s {
                    let mut max = lo;
                    for scan in lo + 1..=hi_s {
                        if compare(&self.data[scan], &self.data[max]) > 0 {
                            max = scan;
                        }
                    }
                    self.data.swap(max, hi_s);
                    hi_s -= 1;
                }
            }
            stack_top -= 1;
            if stack_top < 0 {
                return;
            }
            lo = lo_stack[stack_top as usize];
            hi = hi_stack[stack_top as usize];
        }
    }
}

/// `CHmsCollisionBuffer` (0x00538090): a collision record buffer with the
/// game's initial capacity of 50 (capacity itself is not observable).
#[derive(Clone, Debug, Default)]
pub struct CHmsCollisionBuffer {
    pub collisions: CFastBufferShmsPhysicalCollision,
}

impl CHmsCollisionBuffer {
    pub fn new() -> Self {
        CHmsCollisionBuffer {
            collisions: CFastBufferShmsPhysicalCollision {
                data: Vec::with_capacity(0x32),
            },
        }
    }

    pub fn add_collision(&mut self) -> &mut GmCollision {
        let idx = self.collisions.add_new_elem();
        &mut self.collisions.data[idx].collision
    }

    pub fn get_collision(&mut self, index: usize) -> &mut GmCollision {
        &mut self.collisions.data[index].collision
    }

    pub fn get_count(&self) -> u32 {
        self.collisions.count()
    }
}

/// `SHmsSphereBufferContact` (0x00539880): a mergeable per-sphere buffer.
#[derive(Clone, Debug, Default)]
pub struct SHmsSphereBufferContact {
    pub base: CHmsCollisionBuffer,
    pub active: u32,
}

impl SHmsSphereBufferContact {
    pub fn new() -> Self {
        SHmsSphereBufferContact {
            base: CHmsCollisionBuffer::new(),
            active: 0,
        }
    }
}
