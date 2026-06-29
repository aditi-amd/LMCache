#!/usr/bin/env python3
"""
gpu_map.py — correct rocm-smi card  <->  HIP device-index mapping on MI300/MI355.

WHY: rocm-smi enumerates GPUs as card0..cardN in PCI order, but the HIP runtime
(and therefore CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES) enumerates by ASCENDING
KFD node id. The two orderings differ on this box, so passing a rocm-smi card index
into CUDA_VISIBLE_DEVICES silently targets the WRONG physical GPU -> contention.

Bridge: KFD node -> PCI bus (sysfs location_id) -> rocm-smi card (PCI Bus, --json).
HIP index = rank of the card's KFD node in ascending node-id order.

Commands:
  table                         human-readable HIP|card|node|bus|VRAM%|status
  cards2hip  0,1                HIP indices for physical card0,card1   -> e.g. "1,3"
  hip2cards  2,3               physical cards for HIP 2,3             -> e.g. "2,1"
  free       N                  first N FREE HIP indices (VRAM < threshold)
  assertfree-hip  2,3          exit 0 if those HIP map to FREE cards, else exit 1
  assertfree-cards 1,2         exit 0 if those physical cards are FREE, else exit 1

Env: GPU_FREE_PCT (default 5) = max VRAM% for "free".
"""
import json, os, subprocess, sys, glob

FREE_PCT = float(os.environ.get("GPU_FREE_PCT", "5"))
KFD = "/sys/class/kfd/kfd/topology/nodes"


def _kfd_node_to_bus():
    """node_id(int) -> pci bus (int) for GPU nodes (gpu_id != 0)."""
    out = {}
    for nd in glob.glob(f"{KFD}/*/"):
        try:
            gid = open(os.path.join(nd, "gpu_id")).read().strip()
        except OSError:
            continue
        if not gid or gid == "0":
            continue
        node = int(os.path.basename(nd.rstrip("/")))
        loc = None
        for line in open(os.path.join(nd, "properties")):
            if line.startswith("location_id"):
                loc = int(line.split()[1])
                break
        if loc is None:
            continue
        bus = (loc >> 8) & 0xFF
        out[node] = bus
    return out


def _rocm_cards():
    """card_index(int) -> dict(bus:int, used:int, total:int)."""
    raw = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--showbus", "--json"],
        capture_output=True, text=True,
    ).stdout
    data = json.loads(raw)
    cards = {}
    for k, v in data.items():
        if not k.startswith("card"):
            continue
        idx = int(k[4:])
        busstr = v.get("PCI Bus", "")  # e.g. 0000:05:00.0
        try:
            bus = int(busstr.split(":")[1], 16)
        except (IndexError, ValueError):
            continue
        used = int(v.get("VRAM Total Used Memory (B)", 0))
        total = int(v.get("VRAM Total Memory (B)", 1))
        cards[idx] = {"bus": bus, "used": used, "total": total}
    return cards


def build():
    """Return list of rows sorted by HIP index: dict(hip,card,node,bus,used,total,pct,free)."""
    node_bus = _kfd_node_to_bus()
    bus_node = {b: n for n, b in node_bus.items()}
    cards = _rocm_cards()
    rows = []
    for cidx, c in cards.items():
        node = bus_node.get(c["bus"])
        if node is None:
            continue
        pct = 100.0 * c["used"] / c["total"] if c["total"] else 0.0
        rows.append({"card": cidx, "node": node, "bus": c["bus"],
                     "used": c["used"], "total": c["total"], "pct": pct,
                     "free": pct < FREE_PCT})
    rows.sort(key=lambda r: r["node"])          # ascending KFD node == HIP order
    for hip, r in enumerate(rows):
        r["hip"] = hip
    return rows


def _by_hip(rows):
    return {r["hip"]: r for r in rows}


def _by_card(rows):
    return {r["card"]: r for r in rows}


def cmd_table(rows):
    print(f"{'HIP':<4}{'card':<7}{'node':<6}{'bus':<6}{'VRAM%':<8}status")
    print("-" * 38)
    for r in rows:
        print(f"{r['hip']:<4}card{r['card']:<3}{r['node']:<6}0x{r['bus']:02x}  "
              f"{r['pct']:<8.1f}{'FREE' if r['free'] else 'BUSY'}")
    free_hip = ",".join(str(r["hip"]) for r in rows if r["free"])
    print(f"\nFREE_HIP: {free_hip}")
    print("GPU_MAP: " + " ".join(f"hip{r['hip']}=card{r['card']}" for r in rows))


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    cmd = sys.argv[1]
    rows = build()
    bh, bc = _by_hip(rows), _by_card(rows)

    if cmd == "table":
        cmd_table(rows)

    elif cmd == "cards2hip":
        cards = [int(x) for x in sys.argv[2].split(",")]
        miss = [c for c in cards if c not in bc]
        if miss:
            print(f"ERROR: unknown card(s): {miss}", file=sys.stderr); sys.exit(1)
        print(",".join(str(bc[c]["hip"]) for c in cards))

    elif cmd == "hip2cards":
        hips = [int(x) for x in sys.argv[2].split(",")]
        miss = [h for h in hips if h not in bh]
        if miss:
            print(f"ERROR: unknown hip(s): {miss}", file=sys.stderr); sys.exit(1)
        print(",".join(str(bh[h]["card"]) for h in hips))

    elif cmd == "free":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        freehip = [r["hip"] for r in rows if r["free"]]
        if len(freehip) < n:
            print(f"ERROR: only {len(freehip)} free GPUs, need {n}: "
                  f"free HIP={freehip}", file=sys.stderr); sys.exit(1)
        print(",".join(str(h) for h in freehip[:n]))

    elif cmd in ("assertfree-hip", "assertfree-cards"):
        ids = [int(x) for x in sys.argv[2].split(",")]
        sel = ([bh.get(i) for i in ids] if cmd.endswith("hip")
               else [bc.get(i) for i in ids])
        bad = []
        for i, r in zip(ids, sel):
            if r is None:
                print(f"ERROR: unknown {cmd.split('-')[1]} id {i}", file=sys.stderr)
                sys.exit(1)
            if not r["free"]:
                bad.append(f"hip{r['hip']}=card{r['card']} ({r['pct']:.1f}% used)")
        if bad:
            print("BUSY (another job present): " + "; ".join(bad), file=sys.stderr)
            sys.exit(1)
        print("OK: all free")

    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
