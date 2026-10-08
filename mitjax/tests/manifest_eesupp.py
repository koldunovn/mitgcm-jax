"""Manifest fragment of the tiles, exchanges, global sums and MDS readers (plan Task 7b), of the cg2d rule's
integration with them (Task 7c) and of the pkg/exch2 W2 set-up (owner: lane B). Tier-1 budget (2026-10-01): one compact test per module in tier 1,
the rest in tier 1x."""

MANIFEST = {
    "mitjax/tests/test_global_sum.py": "tier1",
    "mitjax/tests/test_exchange.py": "tier1",
    "mitjax/tests/test_mds.py": "tier1",
    "mitjax/tests/test_exchange_controls.py": "tier1x",
    "mitjax/tests/test_sharded_exchange.py": "tier1x",
    "mitjax/tests/test_cg2d_rule_sharded.py": "tier1x",
    # pkg/exch2 W2 topology set-up (lane B, 4th session): print-out, topology vs headers/maps, grid from it
    "mitjax/tests/test_w2_setup.py": "tier1x",
    # the grid at P=4 (and P=3, padded) == P=1 bitwise: host build once, TileSharding.put_tree (lane B)
    "mitjax/tests/test_grid_sharded.py": "tier1x",
    # exch2 C-grid vector exchanges as sums sa1*A1 + sa2*A2 (EXCH2_PUT_RX2): tables, zu/zv probes, P=4, transpose
    "mitjax/tests/test_exch2_vector.py": "tier1x",
}
