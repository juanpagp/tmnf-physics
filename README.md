# tmnf-physics

**TrackMania Nations Forever (2.11.26) car physics, as a standalone, data-driven Rust library — byte-exact.**

This is a full port of [`juanpagp/tmnf-physics`](https://github.com/juanpagp/tmnf-physics)
(upstream: [`adonis-singh/TMNF-C`](https://github.com/adonis-singh/TMNF-C/)) — itself a
reverse-engineering of the game's physics from the 2.11.26 disassembly. The port preserves
simulation behavior **down to the byte**, proven by replaying captured game runs and comparing
every 10 ms tick against the reference records:

```
a01_mixed:        1200/1200 ticks byte-exact
a01_wall_contact: 1400/1400 ticks byte-exact
```

No game installation, no asset dumping, no ML/AI training harness. A track file, a car image,
a runtime mask — and your game loop.

## Quick start

```rust
use tmnf_physics::sim::{Sim, SimOptions, TmnfInput};

let mut sim = Sim::load(
    "A01-Race.tmnftrack",
    "A01-Stadium.tmnfvehicle",
    &SimOptions {
        fake_contact_mask: Some("game-mask.bin".into()),
        route: Some("A01-Race.tmnfroute".into()),
        pin_provenance: true,
    },
);

let mut playing = true;
while playing {
    let input = TmnfInput::digital(1, 0, 0, 0);      // accelerate only
    let state = sim.step(&input);                     // one 10 ms tick
    playing = state.race_time_ms < 30_000;
    // render_car(&state): position, quaternion, per-wheel contact, rpm, gear
}
```

`step()` is a fixed 10 ms tick by construction (`PhysicsStep2`). Drive it from an accumulator
in your game loop; never scale dt.

## What's in the box

| Module | Contents |
|---|---|
| `fp` | The floating-point policy: x87 PC=24 == IEEE binary32, `ftol` (FISTP), `u32_to_x87_float`, libm-parity transcendentals |
| `gm` | Game math with the disassembly's exact operand groupings (incl. the scalar-first `(w,x,y,z)` quaternion convention, decoded and verified three ways) |
| `dyna` | The 180-byte rigid-body state + integrator (`IntegrateStep`, impulses, replacement buffers) |
| `collision` | Ellipsoid/sphere-vs-mesh narrowphase (with the game's faithfully-kept double-square-root bug), static grid traversal, the game's MSVC quicksort |
| `response` | The impulse solver + contact sink dispatch |
| `vehicle` | The full model-6 Stadium force pipeline: input mapper (0x004FE500), engine/gearbox, suspension, tire forces, turbo/roulette, air control, water, burnout, donut |
| `world` | Corpora, collision zone, the 100 Hz tick (`PhysicsStep2`), respawn, game-state export |
| `track` / `route` / `race` | The `TMNFTRK1` / `TMNFROU1` image loaders (SHA-256 validated) and the race layer (checkpoints, laps, off-track) |
| `sim` | The facade: `Sim::step`, `VehicleState`, snapshots-by-copy determinism |
| `data` | **The authoring lane** (below) |

## The gap that's now closed: authoring cars from data

The fork's own honest roadmap item #1 was that car **tuning** was data (132 scalars, 23 curves,
4 gear tables) but car **structure** — wheel geometry, body ellipsoids, mass/inertia, spawn
state, per-surface tire tables — was an opaque captured image. This port makes it data:

```rust
use tmnf_physics::data::{structure_from_image, image_with_structure};
use tmnf_physics::world::VehicleImage;

let image = VehicleImage::read(&std::fs::read("A01-Stadium.tmnfvehicle")?)?;
let mut car = structure_from_image(&image);        // plain data

car.wheels[0].offset_from_vehicle.y -= 0.1;        // slam it lower
car.body.inverse_inertia.m[4] *= 0.5;              // stiffer in pitch
for node in &mut car.collision_nodes { /* ... */ } // fatter tires, lower body

let image = image_with_structure(image, &car);     // still a valid image
let sim = Sim::from_image(&image, track, mask, &options);
```

Guarantees, all tested against the shipped Stadium capture
(`tests/data_lane.rs`):

* `VehicleImage::write(VehicleImage::read(b))` is **byte-identical** to `b`.
* `image_with_structure(img, structure_from_image(img))` is **byte-identical** — the structure
  authoring path is lossless; the same holds for tuning.
* Structure and tuning serialize to JSON and back without loss.
* Authored changes provably change the physics (fatter wheel ellipsoids, different inertia,
  softer suspension all diverge — the equality above is not vacuous).

The exported `reference_car_structure.json` / `reference_car_tuning.json` (see
`examples/authored_car.rs`) are complete, readable templates to start new cars from.

## Determinism

* **Bit-exact by replay**: 2,600 golden ticks match the captured game records byte-for-byte.
* **Bit-exact by construction**: plain IEEE binary32 per-op semantics (Rust never contracts or
  reassociates FP), the FP-contract hazards of the C build are non-issues in Rust, tick
  arithmetic uses wrapping ops so debug == release.
* Two fresh sims fed the same inputs produce **bit-identical** trajectories (asserted in
  `examples/drive.rs`) — lockstep/rollback netcode and branch-a-tree-of-futures bots are free.
* The only cross-platform caveat is the system libm behind `sin/cos/exp/atan2`; on the same
  platform family the port is identical to the C oracle by construction. For
  cross-platform byte-identical replays, vendor one transcendental implementation for all
  targets and re-run the gate.

## The FP contract in one paragraph

The live game ran its x87 FPU at precision control = 24 bits, which is bit-identical to plain
IEEE binary32 — so the transcription's `x87_*` helper wall collapses into ordinary Rust
operators. What the port preserves by hand instead: operation trees and operand order (never
"clean up" `a*b + a*c`), the handful of deliberate f64-single-rounding expressions
(`GetSpeed`, the collision grid), `ftol` truncation semantics, `u32_to_x87_float`'s two-step
conversion, and the double square root for end-vertex contacts — a bug in the original game,
kept on purpose. See `src/fp.rs` and `PORT_NOTES.md`.

## Examples

```bash
export TMNF_REPO=/path/to/tmnf-physics   # the C repo with the fixtures

cargo run --release --example drive        # the game loop: telemetry CSV + determinism proof
cargo run --release --example authored_car # export a car to JSON, author a variant, drive it
cargo test --release                       # 98 tests incl. the 2,600-tick golden gate
```

## Layout

```
src/
├── fp.rs          # the FP policy
├── gm.rs          # game math (exact groupings)
├── dyna.rs        # rigid-body state + integrator
├── buffer.rs      # collision records + MSVC qsort
├── collision.rs   # narrowphase + grids + dispatch
├── response.rs    # impulse solver
├── vehicle/       # base, curve, aux, contact, compute, model6
├── world.rs       # zone, tick, TMNFM6G1 decode, image read/write
├── track.rs       # TMNFTRK1 loader
├── route.rs       # TMNFROU1 loader + projection
├── race.rs        # checkpoints, laps, off-track
├── surface.rs     # the 31 physical materials
├── data.rs        # car structure/tuning authoring (the gap fix)
└── sim.rs         # the public facade
```

## Fixtures and licensing

The golden fixtures (`oracle/`, `local/game-mask.bin`) are **derived game data** shipped in the
C repository for validation; keep them private and honor the upstream repo's
never-redistribute note. Tests locate them via `TMNF_REPO` (default `../tmnf-physics`).

## What is deliberately not ported

* Legacy car models 3/4/5 (`friction_model != 5`): the A01 oracle exercises model 6; the port
  panics loudly instead of silently guessing.
* The AVX2/AVX-512 packet kernel, the CUDA/RL vec-env, the oracle capture rigs: perf-only or
  research scaffolding, all proven unnecessary for byte-exactness.
* The dead regional re-ports (`vehicle_model6_{burnout,drive,lateral,longitudinal,wheels}.c`).

Porting notes, the divergence-debugging log, and the C↔Rust field maps live in `PORT_NOTES.md`.
