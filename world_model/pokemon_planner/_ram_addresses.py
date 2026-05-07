"""Pokemon Red RAM address constants.

Sourced from the existing v2/baselines memory maps (PWhiddy's repo) and
pret/pokered ROM disassembly. Addresses target Game Boy WRAM ($C000-$DFFF).

This module is INTERNAL — outside callers use pokemon_planner.env.read_state(),
which returns a GameState. Isolating addresses here keeps the extractor readable.

References:
- pret/pokered: https://github.com/pret/pokered
- datacrystal RAM map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
- v2/red_gym_env_v2.py inlines many of these as 0xD... constants
"""

# ---- Position ----
MAP_ID = 0xD35E
PLAYER_Y = 0xD361
PLAYER_X = 0xD362

# ---- Party ----
PARTY_SIZE = 0xD163
PARTY_SPECIES_LIST = 0xD164          # 6 bytes; 0xFF = empty
PARTY_STRUCTS_START = 0xD16B         # 6 contiguous 44-byte structs
PARTY_STRUCT_SIZE = 44

# Offsets within a single 44-byte party struct
PARTY_OFFSET_SPECIES = 0
PARTY_OFFSET_HP_CUR = 1              # 2 bytes BE
PARTY_OFFSET_BOX_LEVEL = 3           # 1 byte (storage); use OFFSET_LEVEL for current
PARTY_OFFSET_STATUS = 4
PARTY_OFFSET_TYPE1 = 5
PARTY_OFFSET_TYPE2 = 6
PARTY_OFFSET_CATCH_RATE = 7
PARTY_OFFSET_MOVES = 8               # 4 bytes
PARTY_OFFSET_OT_ID = 12              # 2 bytes
PARTY_OFFSET_EXP = 14                # 3 bytes BE
PARTY_OFFSET_HP_EV = 17              # 2 bytes BE
PARTY_OFFSET_ATK_EV = 19
PARTY_OFFSET_DEF_EV = 21
PARTY_OFFSET_SPD_EV = 23
PARTY_OFFSET_SPC_EV = 25
PARTY_OFFSET_IVS = 27                # 2 bytes packed
PARTY_OFFSET_PP = 29                 # 4 bytes
PARTY_OFFSET_LEVEL = 33
PARTY_OFFSET_HP_MAX = 34             # 2 bytes BE
PARTY_OFFSET_ATK = 36
PARTY_OFFSET_DEF = 38
PARTY_OFFSET_SPD = 40
PARTY_OFFSET_SPC = 42

# ---- Bag ----
BAG_COUNT = 0xD31D
BAG_ITEMS_START = 0xD31E             # pairs of (item_id, qty), 20 max, terminated 0xFF
BAG_MAX_SLOTS = 20

# ---- Money & time ----
MONEY = 0xD347                       # 3 BCD bytes BE
TIME_PLAYED_HOURS = 0xDA40
TIME_PLAYED_MINUTES = 0xDA42
TIME_PLAYED_SECONDS = 0xDA43
TIME_PLAYED_FRAMES = 0xDA44

# ---- Progress ----
BADGES = 0xD356                      # 8-bit mask
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E             # inclusive
EVENT_FLAGS_LEN = EVENT_FLAGS_END - EVENT_FLAGS_START + 1  # 312 bytes

# ---- Battle ----
IS_IN_BATTLE = 0xD057                # 0 = no, 1 = wild, 2 = trainer
BATTLE_OPP_SPECIES = 0xCFE5
BATTLE_OPP_LEVEL = 0xCFF3
BATTLE_OPP_HP = 0xCFE6               # 2 bytes BE
BATTLE_TURN = 0xCCD5

# ---- Tile / dialogue / menu ----
CURRENT_TILESET = 0xD367
TEXT_BOX_FLAG = 0xCD60               # rough indicator of dialogue state
MENU_FLAGS = 0xCC26                  # cursor / menu state
