//! # tmnf-physics
//!
//! TrackMania Nations Forever (2.11.26) car physics, transliterated from the
//! C transcription in `juanpagp/tmnf-physics` (upstream `adonis-singh/TMNF-C`)
//! with **byte-exact** behavior as the contract.
//!
//! Layering (same seam discipline as the C fork):
//!
//! * [`fp`] — the floating-point policy (x87 PC=24 == IEEE binary32).
//! * [`gm`] — game math primitives with the game's exact operand groupings.
//! * `dyna` — the 180-byte rigid-body state and integrator (`CHmsDyna`).
//! * `collision` — ellipsoid-vs-mesh narrowphase, static grid, response.
//! * `vehicle` — the model-6 (Stadium) force pipeline.
//! * `world` — corpora, collision groups, the vehicle image (`TMNFM6G1`).
//! * `track` / `route` / `race` — `TMNFTRK1` / `TMNFROU1` images and race logic.
//! * `sim` — the public facade (`Sim::step`, one 10 ms tick).
//! * `data` — the authoring lane: car tuning, car **structure**, tracks,
//!   routes (JSON in, internal images out).
//!
//! Porting rule: **transliterate, don't refactor.** Operation trees, operand
//! order and per-operation precision are the physics. See `PORT_NOTES.md`.

#![forbid(unsafe_code)]

pub mod fp;
pub mod gm;
pub mod dyna;
pub mod buffer;
pub mod collision;
pub mod response;
pub mod track;
pub mod route;
pub mod race;
pub mod surface;
pub mod vehicle;
pub mod world;
pub mod data;
pub mod sim;
