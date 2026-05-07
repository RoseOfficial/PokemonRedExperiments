"""Smoke tests for RAM address constants — sanity checks, not full coverage."""
from pokemon_planner import _ram_addresses as ram


def test_addresses_in_wram_range():
    """Address-shaped constants we read should be in WRAM ($C000-$DFFF).

    Filters out small-int constants (sizes, offsets, counts) by requiring
    the value look like a 16-bit address (>= 0x100). Real Game Boy addresses
    are >= $C000 anyway; anything under 0x100 is a count/offset, not an addr.
    """
    for name in dir(ram):
        if name.startswith("_") or name.upper() != name:
            continue
        value = getattr(ram, name)
        if not isinstance(value, int):
            continue
        if value < 0x8000:
            continue   # size / offset / count / length, not an address
        assert 0xC000 <= value <= 0xDFFF, f"{name} = {hex(value)} not in WRAM"


def test_known_canonical_addresses():
    """Cross-check against well-documented Pokemon Red RAM map."""
    # These are widely-documented; if they drift we have a regression
    assert ram.PLAYER_X == 0xD362
    assert ram.PLAYER_Y == 0xD361
    assert ram.MAP_ID == 0xD35E
    assert ram.PARTY_SIZE == 0xD163
    assert ram.PARTY_SPECIES_LIST == 0xD164  # 6 bytes starting here
    assert ram.MONEY == 0xD347                # 3 BCD bytes
    assert ram.BADGES == 0xD356


def test_event_flag_range():
    """Event flag region covers $D747-$D87E per pret/pokered."""
    assert ram.EVENT_FLAGS_START == 0xD747
    assert ram.EVENT_FLAGS_END == 0xD87E
    assert ram.EVENT_FLAGS_END - ram.EVENT_FLAGS_START + 1 == 0x138  # 312 bytes


def test_party_struct_offsets():
    """Per-Pokemon party struct is 44 bytes per pret docs."""
    assert ram.PARTY_STRUCT_SIZE == 44
    # Offsets within a single party member's 44-byte struct
    assert ram.PARTY_OFFSET_SPECIES == 0
    assert ram.PARTY_OFFSET_HP_CUR == 1   # 2 bytes BE
    assert ram.PARTY_OFFSET_LEVEL == 33
    assert ram.PARTY_OFFSET_STATUS == 4
    assert ram.PARTY_OFFSET_HP_MAX == 34  # 2 bytes BE
    assert ram.PARTY_OFFSET_MOVES == 8    # 4 bytes
