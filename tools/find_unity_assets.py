"""Find named Unity objects in the pristine xRC player data."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import UnityPy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--pattern", default=r"legacy|Legacy Robot")
    args = parser.parse_args()
    pattern = re.compile(args.pattern, re.IGNORECASE)
    found: list[dict[str, object]] = []
    for path in args.paths:
        env = UnityPy.load(str(path))
        for obj in env.objects:
            try:
                value = obj.read()
                name = str(getattr(value, "m_Name", ""))
                tree_text = ""
                if obj.type.name in {"MonoBehaviour", "TextAsset"}:
                    try:
                        tree_text = json.dumps(obj.read_typetree(), default=str)
                    except Exception:
                        tree_text = str(getattr(value, "m_Script", ""))
                if pattern.search(name) or pattern.search(tree_text):
                    found.append(
                        {
                            "file": str(path),
                            "path_id": obj.path_id,
                            "type": obj.type.name,
                            "name": name,
                            "text_excerpt": next(
                                (tree_text[max(0, match.start() - 120): match.end() + 240] for match in [pattern.search(tree_text)] if match),
                                "",
                            ),
                        }
                    )
            except Exception:
                continue
    print(json.dumps(found, indent=2))


if __name__ == "__main__":
    main()
