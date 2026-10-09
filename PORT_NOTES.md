# Rust Port Notes — recovery anchor

Crate: `/home/z/my-project/tmnf-physics-rs` (package `tmnf-physics`).
C source: `/home/z/my-project/tmnf-physics` (commit 476a1c2).
Gate binaries (C oracle): `/home/z/my-project/scripts/replay_tick{,_masked}`.
Fixtures: `oracle/tracks/A01-Race.tmnftrack`, `oracle/vehicles/A01-Stadium.tmnfvehicle`,
`oracle/routes/A01-Race.tmnfroute`, `oracle/results/a01_mixed{,_inputs}.bin`,
`a01_wall_contact{,_inputs}.bin`, `local/game-mask.bin`.
Gate: replay_tick --route ROUTE TRACK VEHICLE RESULTS input_file INPUTS →
1200/1200 (a01_mixed) + 1400/1400 (a01_wall_contact) ticks byte-exact,
1,668-byte record per tick (race_time 4B + dyna 172B + scene_mobil 180B + 4×328B wheels).

## FP cheat sheet (CRITICAL — never deviate)

- x87 PC=24 == plain IEEE f32. `x87_mul/add/sub/div` → ordinary Rust operators.
  NO contraction, NO reassociation in Rust — guaranteed by the language.
- `x87_sqrt` → `.sqrt()`. `x87_rcp` → `1.0/x` (never `.recip()`).
- `ftol(x)` → `fp::ftol` (trunc + i32::MIN out-of-range; Rust `as` saturates!).
- `u32_to_x87_float` → `fp::u32_to_x87_float` (signed fild + conditional +2^32).
- Transcendentals: `((x as f64).sin()) as f32` etc. (system libm). Never f32::sin.
- Multi-op `double` expressions in C (e.g. `F(a) + F(b)*F(c)`) → f64 block,
  ONE `as f32` at the end. Single-op `F(a) op F(b)` → plain f32 op.
  Comparisons `F(a) <= F(b)` → plain f32 compare.
- **Rust has NO hex-float literals.** `0x1.b7cdfcp-34f` → `f32::from_bits(0x...)`
  or decimal. f32::from_bits is safest for magic constants.
- Rust never enables FTZ/DAZ; `debug` == `release` bitwise.
- Transliterate, don't refactor: preserve operand order, grouping (a*b + a*c
  must NOT become a*(b+c)), and the double-sqrt bug where present.

## Decoded conventions

- **Game quaternion = (w, x, y, z) scalar-first**, stored in GmQuat fields named
  (x,y,z,w): field .x holds w. Verified 3 ways (SetFromMat3 trace branch,
  SetFromQuat formula match, IntegrateStep dq derivation). Identity = .x=1.
  Public API must convert at the boundary.
- CHmsStateDyna (180B): quat@0x00, rot@0x10, pos@0x34, linVel@0x40,
  linVelAdded@0x4C, angVel@0x58, force@0x64, torque@0x70, invInertiaWorld@0x7C,
  tail[5]@0xA0 (keep bytes!). rot+pos 48B overlay == GmIso4 { m: rot.m, t: pos }.
- CHmsDynaParams (68B): mass@0, invInertiaBody@4, dragLinear@0x28,
  dragAngular@0x2C, substepLen@0x30, forceFieldScale@0x34, comOffset@0x38.
- MSVC quicksort in fastbuffer.c: port verbatim (unstable, swap order matters).

## 🏆 GOLDEN GATE PASSED (2025 session)

a01_mixed 1200/1200 + a01_wall_contact 1400/1400 ticks byte-exact. 89/89 tests.
Divergence bugs fixed: dirtyFlag decode, shape offset 0x6c, tree-ref wiring,
model6 (gas*braking+rev*braking*brake)×accel grouping, finish_integration
(rollover_axis sync), wheel iso t at +36 (not +48), committed-rot for
relative_rotz_axis. tick_time advances by tick_ms (ms, not ticks).

## Ported (✅) / TODO — STATUS @ world/image phase

- ✅ fp.rs, gm.rs, dyna.rs, buffer.rs (MSVC qsort verbatim), collision.rs
     (scalar narrowphase + grid + dispatch), response.rs (kernels + ctx trait),
     track.rs (2135 L, real-fixture tested: 2603 entries/1499 surfaces/96 meshes/
     65674 faces/88035 nodes, payload SHA validated), route.rs (1292 L),
     race.rs (1254 L), surface.rs.
- ✅ vehicle/: mod.rs (types contract + C→Rust mapping table), base.rs (1569 L,
     input mapper 0x004FE500), curve.rs (2177 L), aux.rs (1909 L), contact.rs
     (mine, ~700 L), compute.rs (mine, orchestrator), model6.rs (mine, full
     ComputeForcesModel6 + donut + material6).
- ⬜ world.rs (world.c 2288: corpora/groups/zone + TMNFM6G1 decode + respawn +
     hms_item glue) — ME, NEXT
- ⬜ physics.rs (physics.c 222, PhysicsStep2) — ME
- ⬜ api/vehicle_image + car_tuning + data lane (THE GAP FIX) — ME
- ⬜ sim.rs facade (tmnf_sim.c 1080) + record.rs (replay_tick.c) — ME
- ⬜ examples + README + gate test — ME
- Legacy models 3/4/5: panic stubs (friction_model != 5). Dead code not ported.

## Key API facts (for the world/sim port)

- VehicleCtx { vehicle: &mut Vehicle, tuning: &VehicleTuning, dyna: &mut CHmsDyna }
  (src/vehicle/mod.rs has the full C→Rust field map).
- compute::cscene_vehicle_car_compute_forces(ctx, &mut ComputeExternal, dt);
  ComputeExternal { ext: &Model6External, water: &TmnfTrackWater,
  post_force: &mut dyn FnMut(&mut VehicleCtx), contact_token: &dyn Fn(u32)->u32,
  body_iso: &dyn Fn(u32)->GmIso4, fake_contacts_active, water_forces_active }.
- Model6External { model_iso_source: GmIso4, body_reference_position: GmVec3 }.
- contact::absorb_contact(ctx, contact, resolve_body_iso) — the wheel sink.
- CHmsItem ops → direct ctx.dyna local-speed/force methods (single corpus).
- response::compute_collision_response(ctx: &mut dyn ResponseCtx) — the world
  implements ResponseCtx (body/body_iso/get_speed/add_replacement/
  solve_body_impulse/absorb_contact/response_material/surface_materials/
  collisions). Response bodies: CHmsResponseBody { corpus_ref,
  classification_flags, response_flags, response_weight, iso, dyna: Option<usize>,
  has_contact_sink, sink: AbsorbSink {None|VehicleBody|VehicleWheel(idx)} }.
- collision::compute_surface_collision(surface1, iso1, surface2, iso2, accel2,
  buffer, &CollisionShapeDispatch::default()).
- Vehicle tuning wiring: tuning.wheel_tree_refs / body_tree_refs map tree
  object_refs → wheels/body (needed by absorb_contact dispatch).
- Wheel contact_body: Option<u32> = corpus ref of contacted static body.
- Tick u32 arithmetic uses wrapping_sub (matches C wrap, debug==release).

## Design decisions

- Single crate, modules mirror C files. `#![forbid(unsafe_code)]`.
- Pointer-linked graphs (track/route images) → owned Vec + u32 indices,
  same traversal order. CHmsDyna owns state_b/live_state by value.
- Loaders read whole files (no mmap needed); images are little-endian.
- Data lane:     + serde_json (network to crates.io works). JSON shapes per
  docs/FORMATS.md; floats via serde default (shortest round-trip decimal is
  bit-exact for f32).
- Fixtures referenced by path (repo), NOT vendored (licensing: derived game
  data, never redistribute).
- `tmnf_warp.h` host identity helpers → plain Rust (max of one value).

## Gotchas found so far

- Rust hex-float literals don't exist (see cheat sheet).
- `cargo` needs `source $HOME/.cargo/env` (in ~/.bashrc now).
- Don't run cargo build in the same message as file edits (races).
