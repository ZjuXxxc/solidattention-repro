"""Render the trace as dependency-free HTML/SVG."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

COLORS = {"SSD read": "#ef8354", "DRAM staging": "#4f5d75", "PCIe H2D": "#43aa8b",
          "GPU compute": "#577590", "PCIe D2H": "#90be6d", "SSD write": "#f9c74f"}


def render(trace_path: Path, output: Path) -> None:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data["traceEvents"]
    lanes = list(COLORS)
    end = max(event["ts"] + event["dur"] for event in events)
    width, left, row_h = 1400, 170, 54
    scale = (width - left - 30) / end
    svg = [f'<svg viewBox="0 0 {width} {80 + row_h * len(lanes)}" xmlns="http://www.w3.org/2000/svg">']
    svg.append('<style>text{font:13px sans-serif}.label{font-weight:600}.task:hover{opacity:.72}</style>')
    for index, lane in enumerate(lanes):
        y = 50 + index * row_h
        svg.append(f'<text class="label" x="8" y="{y+20}">{html.escape(lane)}</text>')
        svg.append(f'<line x1="{left}" y1="{y+27}" x2="{width-20}" y2="{y+27}" stroke="#ddd"/>')
        for event in (item for item in events if item["tid"] == lane):
            x, w = left + event["ts"] * scale, max(2, event["dur"] * scale)
            title = html.escape(f'{event["name"]} | start={event["ts"]:.1f}us dur={event["dur"]:.1f}us | {event["args"]}')
            svg.append(f'<g class="task"><title>{title}</title><rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="25" rx="3" fill="{COLORS[lane]}"/><text x="{x+3:.1f}" y="{y+17}" clip-path="none">{html.escape(event["name"])}</text></g>')
    svg.append('</svg>')
    meta = html.escape(json.dumps(data.get("metadata", {}), indent=2))
    document = f'''<!doctype html><meta charset="utf-8"><title>SolidAttention schedule</title>
<style>body{{font-family:system-ui;margin:24px;color:#20232a}}pre{{background:#f5f6f8;padding:14px}}.hint{{color:#555}}</style>
<h1>SolidAttention SSD → DRAM → VRAM schedule</h1>
<p class="hint">Hover a bar for timing, layer, byte count, and prediction details. Horizontal overlap means concurrent resource use; gaps mean idle/stall time.</p>
{''.join(svg)}<h2>Run metadata</h2><pre>{meta}</pre>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.trace, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
