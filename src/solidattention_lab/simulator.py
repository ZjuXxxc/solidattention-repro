"""A visible model of SolidAttention's per-layer DAG.

This is intentionally a scheduler model, not an LLM. Durations represent the
resource conflicts that we want to reason about: NVMe reads, host staging,
PCIe copies, GPU projections/selection/attention, and asynchronous writeback.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .trace import Event, write_trace


@dataclass
class Resource:
    name: str
    ready_us: float = 0.0

    def reserve(self, earliest: float, duration: float) -> tuple[float, float]:
        start = max(earliest, self.ready_us)
        self.ready_us = start + duration
        return start, duration


class Schedule:
    def __init__(self) -> None:
        self.resources = {name: Resource(name) for name in
                          ("SSD read", "DRAM staging", "PCIe H2D", "GPU compute", "PCIe D2H", "SSD write")}
        self.events: list[Event] = []

    def task(self, name: str, resource: str, earliest: float, duration: float,
             layer: int, **args: object) -> float:
        start, dur = self.resources[resource].reserve(earliest, duration)
        self.events.append(Event(name, "solidattention", start, dur, 1, resource,
                                 {"layer": layer, **args}))
        return start + dur


def build_schedule(layers: int, block_kib: int, selected_blocks: int,
                   prefetch: bool, overlap: bool) -> tuple[list[Event], dict]:
    s = Schedule()
    layer_ready = 0.0
    previous_selected = list(range(selected_blocks))
    # Larger requests approach peak bandwidth; 4 KiB random reads pay heavily.
    io_efficiency = min(1.0, max(0.12, block_kib / 32.0))
    read_us = (selected_blocks * block_kib / io_efficiency) * 0.45
    copy_us = selected_blocks * block_kib * 0.10

    for layer in range(layers):
        q_done = s.task("Q projection", "GPU compute", layer_ready, 90, layer)
        kv_done = s.task("interleaved KV projection", "GPU compute", q_done, 120, layer,
                         layout="K0,V0,K1,V1,...")
        select_done = s.task("representative dot + top-k", "GPU compute", q_done, 55, layer,
                             blocks=selected_blocks)

        io_start = layer_ready if (prefetch and overlap) else select_done
        disk_done = s.task("prefetch KV blocks", "SSD read", io_start, read_us, layer,
                           block_kib=block_kib, predicted=previous_selected)
        staged = s.task("SSD completion / DRAM buffer", "DRAM staging", disk_done, 18, layer,
                        bytes=selected_blocks * block_kib * 1024)
        h2d_done = s.task("DRAM → VRAM", "PCIe H2D", staged, copy_us, layer,
                          bytes=selected_blocks * block_kib * 1024)

        # Every fourth layer demonstrates a speculative miss and overwrite.
        miss_done = select_done
        if prefetch and layer % 4 == 3:
            miss_disk = s.task("load one missing block", "SSD read", select_done,
                               block_kib * 0.45 / io_efficiency, layer, overwrite=True)
            miss_stage = s.task("missing block → DRAM", "DRAM staging", miss_disk, 8, layer)
            miss_done = s.task("overwrite wrong VRAM slot", "PCIe H2D", miss_stage,
                               block_kib * 0.10, layer)

        attention_ready = max(kv_done, select_done, h2d_done, miss_done)
        attention_done = s.task("sparse attention", "GPU compute", attention_ready, 145, layer,
                                order_independent=True)

        # New KV writeback is non-critical and may overlap the next layer.
        d2h_start = attention_done if not overlap else kv_done
        d2h_done = s.task("new KV: VRAM → DRAM", "PCIe D2H", d2h_start, 15, layer)
        s.task("buffered KV append (32 KiB)", "SSD write", d2h_done, 35, layer)
        layer_ready = attention_done

    makespan = max(resource.ready_us for resource in s.resources.values())
    metadata = {"layers": layers, "block_kib": block_kib, "selected_blocks": selected_blocks,
                "prefetch": prefetch, "overlap": overlap, "makespan_us": makespan,
                "note": "Scaled educational scheduler; not paper performance data"}
    return s.events, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--block-kib", type=int, default=32)
    parser.add_argument("--selected-blocks", type=int, default=8)
    parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace", default="artifacts/trace.json")
    args = parser.parse_args()
    events, metadata = build_schedule(args.layers, args.block_kib, args.selected_blocks,
                                      args.prefetch, args.overlap)
    write_trace(args.trace, events, metadata)
    print(json.dumps(metadata, indent=2))
    print(f"trace: {Path(args.trace).resolve()}")


if __name__ == "__main__":
    main()
