from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "assets" / "fresh_xrc" / "field"
ROBOT = ROOT / "assets" / "fresh_xrc" / "robot"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_source_hashes_are_pristine_xrc_v20_2b():
    manifest = load(FIELD / "manifest.json")
    assert manifest["source_sha256"] == {
        "level74": "48be3206000d3dbfbd4c66a54d00907910f38eaf66f52199716f2047cbed7502",
        "level75": "06b076efcec330e480995608cce0845dcc8a4809dabfe017dd145a47e1e2c341",
        "level76": "a5f8bc6f6620c128339366b1a515edec0e7a62e059f90c06446840e1aabf0346",
    }


def test_complete_field_inventory():
    manifest = load(FIELD / "manifest.json")
    assert manifest["visual_instances"] == 1275
    assert manifest["render_triangles"] == 120703
    assert manifest["colliders"] == 701
    assert manifest["rigidbodies"] == 496
    assert manifest["joints"] == 2
    assert manifest["visual_categories"]["hub"] >= 300
    assert manifest["visual_categories"]["trench"] >= 300
    assert manifest["visual_categories"]["bump"] == 16


def test_hubs_have_four_exits_each_and_scorers():
    colliders = load(FIELD / "colliders.json")
    exits = [item for item in colliders if "/Physics/bs_out" in item["path"] and item["trigger"]]
    scorers = [item for item in colliders if item["path"].endswith(("redscorer", "bluescorer"))]
    assert len(exits) == 8
    assert len(scorers) == 2
    assert all(item["trigger"] for item in scorers)


def test_four_bumps_have_eight_physical_ramp_surfaces():
    colliders = load(FIELD / "colliders.json")
    ramps = [item for item in colliders if item["path"].startswith("Ramp") and item["type"] == "BoxCollider"]
    assert len(ramps) == 8
    assert len({item["path"].split("/")[0] for item in ramps}) == 4


def test_xrc_has_456_physical_fuel_bodies():
    colliders = load(FIELD / "colliders.json")
    bodies = load(FIELD / "rigidbodies.json")
    fuel_colliders = [item for item in colliders if item["level"] in {"level75", "level76"} and "fuel" in item["path"].lower() and item["type"] == "SphereCollider"]
    fuel_bodies = [item for item in bodies if item["level"] in {"level75", "level76"} and "fuel" in item["path"].lower()]
    assert len(fuel_colliders) == len(fuel_bodies) == 456
    assert all(abs(item["radius"] - 0.076) < 1e-5 for item in fuel_colliders)
    assert all(abs(item["mass"] - 0.08) < 1e-5 for item in fuel_bodies)


def test_exact_legacy_prefab_component_counts():
    provenance = load(ROBOT / "provenance.json")
    counts = provenance["counts"]
    assert provenance["root_name"] == "Legacy Robot"
    assert counts["descendant_game_objects_including_root"] == 490
    assert counts["components_by_type"]["Rigidbody"] == 23
    assert counts["components_by_type"]["ConfigurableJoint"] == 11
    assert counts["components_by_type"]["BoxCollider"] == 46
    assert counts["components_by_type"]["CapsuleCollider"] == 18
    assert counts["components_by_type"]["SphereCollider"] == 6
