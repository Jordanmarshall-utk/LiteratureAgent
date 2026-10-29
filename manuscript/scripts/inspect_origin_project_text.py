"""Read text-object properties from a saved Origin project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("graph")
    parser.add_argument("--close", action="store_true")
    args = parser.parse_args()
    import originpro as op
    if getattr(op, "oext", False):
        op.set_show(False)
    op.open(str(args.project.resolve()), readonly=True, asksave=False)
    graph = op.find_graph(args.graph)
    if graph is None:
        raise RuntimeError(f"Graph not found: {args.graph}")
    layer = graph[0]
    result: dict[str, object] = {}
    for name in ("legend", *(f"laLabel{index}" for index in range(1, 21)), "xb"):
        label = layer.label(name)
        if label is None:
            continue
        result[name] = {
            "text": label.text,
            "background": label.get_int("background"),
            "whiteout": label.get_int("whiteout"),
            "font": label.get_int("font"),
            "fsize": label.get_float("fsize"),
            "attach": label.get_int("attach"),
            "anchor": label.get_int("anchor"),
            "x": label.get_float("x"),
            "y": label.get_float("y"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if getattr(op, "oext", False) and args.close:
        op.exit()


if __name__ == "__main__":
    main()
