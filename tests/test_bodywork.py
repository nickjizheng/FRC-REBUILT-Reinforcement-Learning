"""Phase C2: containment-panel selection matches the C1 audit.

The USD panel geometry is built in Isaac (RobotArticulationBuilder.build);
here we lock the *selection* logic — which colliders become translucent panels
— to the audited ground truth without launching Isaac.
"""
from __future__ import annotations

import json
from pathlib import Path

from xrc_rebuilt.robot_model import (
    CONTAINMENT_SUBTREES,
    containment_collider_paths,
    is_containment_collider_path,
)

ROBOT = Path(__file__).resolve().parents[1] / "assets" / "fresh_xrc" / "robot"


def _components():
    return json.loads((ROBOT / "components.json").read_text(encoding="utf-8"))


def test_is_containment_predicate():
    assert is_containment_collider_path(
        "Legacy Robot/Body[0]/Physics[7]/IntakeHopper[5]/intake (1)[12]"
    )
    assert is_containment_collider_path("Legacy Robot/Body[0]/Physics[7]/Body[6]/Cube[2]")
    # trigger/other subtrees are not containment
    assert not is_containment_collider_path("Legacy Robot/Body[0]/Physics[7]/keepClear[2]")
    assert not is_containment_collider_path(
        "Legacy Robot/Body[0]/Visuals[6]/MainBody[3]/Intake[4]"
    )


def test_twenty_solid_containment_colliders():
    """Matches docs/legacy_BODYWORK_AUDIT.md: 9 IntakeHopper + 11 Body = 20."""
    paths = containment_collider_paths(_components())
    assert len(paths) == 20
    intake = [p for p in paths if CONTAINMENT_SUBTREES[0] in p]
    body = [p for p in paths if CONTAINMENT_SUBTREES[1] in p]
    assert len(intake) == 9
    assert len(body) == 11


def test_containment_colliders_are_never_triggers():
    components = _components()
    by_path = {}
    for c in components:
        if c["source"]["type"].endswith("Collider"):
            by_path.setdefault(c["owner_path"], []).append(c)
    for path in containment_collider_paths(components):
        for col in by_path[path]:
            if is_containment_collider_path(path):
                assert col["values"].get("m_IsTrigger") in (False, None)
