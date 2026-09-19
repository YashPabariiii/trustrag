"""Grafana dashboards are provisioned from git, so a typo in one is a broken
panel nobody notices until an incident. Check what provisioning cannot."""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DASHBOARDS = sorted((ROOT / "grafana/dashboards").glob("*.json"))
DATASOURCE_UID = "trustrag-prometheus"

# Every metric family the application actually defines.
metrics_source = (ROOT / "app/core/metrics.py").read_text()
metrics_source += (ROOT / "app/core/system_metrics.py").read_text()
DEFINED = set(re.findall(r'"(trustrag_[a-z_]+)"', metrics_source))

failures = []
uids = set()

print(f"{len(DEFINED)} metric families defined in the application\n")

for path in DASHBOARDS:
    try:
        body = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[FAIL] {path.name}: invalid JSON — {exc}")
        failures.append(path.name)
        continue

    panels = body.get("panels", [])
    real = [p for p in panels if p["type"] != "row"]
    print(f"[PASS] {path.name}: {body['title']!r}, {len(real)} panels")

    if body["uid"] in uids:
        print(f"  [FAIL] duplicate dashboard uid {body['uid']}")
        failures.append(path.name)
    uids.add(body["uid"])

    for panel in real:
        if panel["type"] == "text":
            continue
        if (panel.get("datasource") or {}).get("uid") != DATASOURCE_UID:
            print(f"  [FAIL] panel {panel['title']!r} has no {DATASOURCE_UID} datasource")
            failures.append(f"{path.name}:{panel['title']}")
        if not panel.get("targets"):
            print(f"  [FAIL] panel {panel['title']!r} has no query")
            failures.append(f"{path.name}:{panel['title']}")

        for target in panel.get("targets", []):
            for used in re.findall(r"trustrag_[a-z_]+", target["expr"]):
                # Prometheus suffixes: a histogram query names _bucket/_sum/_count.
                base = re.sub(r"_(bucket|sum|count)$", "", used)
                if base not in DEFINED and used not in DEFINED:
                    print(f"  [FAIL] panel {panel['title']!r} queries undefined {used}")
                    failures.append(f"{path.name}:{used}")

print("\n" + "=" * 62)
if failures:
    print(f"FAILED: {len(set(failures))} — {', '.join(sorted(set(failures)))}")
    sys.exit(1)
print(f"{len(DASHBOARDS)} dashboards valid; every queried metric exists.")
