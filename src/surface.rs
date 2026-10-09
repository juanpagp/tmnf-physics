//! CPlugSurface physical material ids (`src/surface_material.h`), transliterated.
//!
//! The 31 ids are `EPlugSurfaceMaterialId`, shared by every environment; the
//! names are the reflection table at 0x00D05500 (31 `char*` entries). The
//! per-id friction/restitution pairs live at 0x00D6EEC0 and are carried by the
//! track snapshot's material-data section (31 entries, required).

/// `TMNF_SURFACE_CONCRETE`
pub const TMNF_SURFACE_CONCRETE: u32 = 0;
/// `TMNF_SURFACE_PAVEMENT`
pub const TMNF_SURFACE_PAVEMENT: u32 = 1;
/// `TMNF_SURFACE_GRASS`
pub const TMNF_SURFACE_GRASS: u32 = 2;
/// `TMNF_SURFACE_ICE`
pub const TMNF_SURFACE_ICE: u32 = 3;
/// `TMNF_SURFACE_METAL`
pub const TMNF_SURFACE_METAL: u32 = 4;
/// `TMNF_SURFACE_SAND`
pub const TMNF_SURFACE_SAND: u32 = 5;
/// `TMNF_SURFACE_DIRT`
pub const TMNF_SURFACE_DIRT: u32 = 6;
/// `TMNF_SURFACE_TURBO`
pub const TMNF_SURFACE_TURBO: u32 = 7;
/// `TMNF_SURFACE_DIRT_ROAD`
pub const TMNF_SURFACE_DIRT_ROAD: u32 = 8;
/// `TMNF_SURFACE_RUBBER`
pub const TMNF_SURFACE_RUBBER: u32 = 9;
/// `TMNF_SURFACE_SLIDING_RUBBER`
pub const TMNF_SURFACE_SLIDING_RUBBER: u32 = 10;
/// `TMNF_SURFACE_TEST`
pub const TMNF_SURFACE_TEST: u32 = 11;
/// `TMNF_SURFACE_ROCK`
pub const TMNF_SURFACE_ROCK: u32 = 12;
/// `TMNF_SURFACE_WATER`
pub const TMNF_SURFACE_WATER: u32 = 13;
/// `TMNF_SURFACE_WOOD`
pub const TMNF_SURFACE_WOOD: u32 = 14;
/// `TMNF_SURFACE_DANGER`
pub const TMNF_SURFACE_DANGER: u32 = 15;
/// `TMNF_SURFACE_ASPHALT`
pub const TMNF_SURFACE_ASPHALT: u32 = 16;
/// `TMNF_SURFACE_WET_DIRT_ROAD`
pub const TMNF_SURFACE_WET_DIRT_ROAD: u32 = 17;
/// `TMNF_SURFACE_WET_ASPHALT`
pub const TMNF_SURFACE_WET_ASPHALT: u32 = 18;
/// `TMNF_SURFACE_WET_PAVEMENT`
pub const TMNF_SURFACE_WET_PAVEMENT: u32 = 19;
/// `TMNF_SURFACE_WET_GRASS`
pub const TMNF_SURFACE_WET_GRASS: u32 = 20;
/// `TMNF_SURFACE_SNOW`
pub const TMNF_SURFACE_SNOW: u32 = 21;
/// `TMNF_SURFACE_RESONANT_METAL`
pub const TMNF_SURFACE_RESONANT_METAL: u32 = 22;
/// `TMNF_SURFACE_GOLF_BALL`
pub const TMNF_SURFACE_GOLF_BALL: u32 = 23;
/// `TMNF_SURFACE_GOLF_WALL`
pub const TMNF_SURFACE_GOLF_WALL: u32 = 24;
/// `TMNF_SURFACE_GOLF_GROUND`
pub const TMNF_SURFACE_GOLF_GROUND: u32 = 25;
/// `TMNF_SURFACE_TURBO2`
pub const TMNF_SURFACE_TURBO2: u32 = 26;
/// `TMNF_SURFACE_BUMPER`
pub const TMNF_SURFACE_BUMPER: u32 = 27;
/// `TMNF_SURFACE_NOT_COLLIDABLE`
pub const TMNF_SURFACE_NOT_COLLIDABLE: u32 = 28;
/// `TMNF_SURFACE_FREE_WHEELING`
pub const TMNF_SURFACE_FREE_WHEELING: u32 = 29;
/// `TMNF_SURFACE_TURBO_ROULETTE`
pub const TMNF_SURFACE_TURBO_ROULETTE: u32 = 30;
/// `TMNF_SURFACE_MATERIAL_COUNT`
pub const TMNF_SURFACE_MATERIAL_COUNT: u32 = 31;

/// Upper bound on material-manager entries a vehicle snapshot may carry
/// (`TMNF_MAX_GROUND_MATERIALS`). Stadium has 13; the other collections are
/// read from their snapshots.
pub const TMNF_MAX_GROUND_MATERIALS: u32 = 32;

/// The 31 reflection names at 0x00D05500.
pub const TMNF_SURFACE_MATERIAL_NAMES: [&str; TMNF_SURFACE_MATERIAL_COUNT as usize] = [
    "Concrete",
    "Pavement",
    "Grass",
    "Ice",
    "Metal",
    "Sand",
    "Dirt",
    "Turbo",
    "DirtRoad",
    "Rubber",
    "SlidingRubber",
    "Test",
    "Rock",
    "Water",
    "Wood",
    "Danger",
    "Asphalt",
    "WetDirtRoad",
    "WetAsphalt",
    "WetPavement",
    "WetGrass",
    "Snow",
    "ResonantMetal",
    "GolfBall",
    "GolfWall",
    "GolfGround",
    "Turbo2",
    "Bumper",
    "NotCollidable",
    "FreeWheeling",
    "TurboRoulette",
];

/// `TmnfSurfaceMaterial_Name`: the reflection name of a material id.
/// Out-of-range ids abort in the C (`abort()`); here they panic.
pub fn tmnf_surface_material_name(material_id: u32) -> &'static str {
    if material_id >= TMNF_SURFACE_MATERIAL_COUNT {
        panic!("tmnf: surface material id out of range");
    }
    TMNF_SURFACE_MATERIAL_NAMES[material_id as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_table_shape() {
        assert_eq!(TMNF_SURFACE_MATERIAL_COUNT, 31);
        assert_eq!(TMNF_SURFACE_MATERIAL_NAMES.len(), 31);
        assert_eq!(tmnf_surface_material_name(0), "Concrete");
        assert_eq!(tmnf_surface_material_name(2), "Grass");
        assert_eq!(tmnf_surface_material_name(13), "Water");
        assert_eq!(tmnf_surface_material_name(20), "WetGrass");
        assert_eq!(tmnf_surface_material_name(30), "TurboRoulette");
    }

    #[test]
    #[should_panic]
    fn material_name_out_of_range_panics() {
        let _ = tmnf_surface_material_name(31);
    }
}
