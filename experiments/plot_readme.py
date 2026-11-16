"""Generate the README benchmark figure from the published experiment JSON."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCALE = 2
BACKGROUND = "#f3f6fa"
INK = "#142137"
MUTED = "#516078"
BASELINE = "#7990af"
SHRF = "#1b65d3"


def result(name):
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def font(size, bold=False):
    candidates = ("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for name in candidates:
        try:
            return ImageFont.truetype(name, size * SCALE)
        except OSError:
            continue
    return ImageFont.load_default(size=size * SCALE)


def panel(draw, left, kicker, title, subtitle, entries, summary):
    right = left + 350

    def box(x0, y0, x1, y1, radius, fill):
        draw.rounded_rectangle(tuple(int(v*SCALE) for v in (x0,y0,x1,y1)),
                               radius=radius*SCALE, fill=fill)

    def text(x, y, value, size, bold=False, color=INK, anchor="lt"):
        draw.text((x*SCALE,y*SCALE), value, font=font(size,bold), fill=color, anchor=anchor)

    box(left, 90, right, 352, 13, "white")
    text(left+21, 111, kicker, 12, True, SHRF)
    text(left+21, 136, title, 23, True)
    text(left+21, 169, subtitle, 13, False, MUTED)
    biggest = max(value for _, value, _ in entries)
    for index, (label, value, display) in enumerate(entries):
        y = 212+48*index
        text(left+21, y, label, 14, True)
        text(right-21, y, display, 14, True, anchor="rt")
        box(left+21, y+21, right-21, y+31, 5, "#e5ebf3")
        length = 308*value/biggest
        box(left+21, y+21, left+21+length, y+31, min(5,int(length/2)), (BASELINE,SHRF)[index])
    text(left+21, 324, summary, 13, True, MUTED)


def main():
    memory = result("complexity")
    reference = next(row for row in memory["contexts"] if row["tokens"] == 128_000)
    gqa = reference["gqa8"]["kv_bytes"] / 1e9
    compact = memory["contextual_local"]["shrf"]["hot_fp32_bytes"] / 1e9
    old = next(row["tokens_s"] for row in result("prefill_v1")["rows"] if row["length"] == 4096)
    new = next(row["tokens_s"] for row in result("prefill")["rows"] if row["length"] == 4096)
    recall = result("needle_blocks")["runs"]

    def recall_entry(key, label):
        row = recall[key]["accuracy_512"]
        targets = row["targets"]
        hits = round(row["accuracy"]*targets)
        return label, hits/targets, f"{hits}/{targets}"

    image = Image.new("RGB", (1140*SCALE, 380*SCALE), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((24*SCALE, 26*SCALE), "SHRF benchmark snapshot",
              font=font(31, True), fill=INK, anchor="lt")
    panel(draw, 24, "MODELED", "Resident memory", "128k tokens · d=4096 · 32 layers",
          (("GQA-8", gqa, f"{gqa:.2f} GB"), ("Compact SHRF", compact, f"{compact:.3f} GB")),
          f"{gqa/compact:.1f}× ratio · cold archive excluded")
    panel(draw, 395, "MEASURED", "GPU prefill", "4,096 tokens · d=512 · GTX 1650",
          (("Previous SHRF",old,f"{old:,.0f} tok/s"), ("Current SHRF",new,f"{new:,.0f} tok/s")),
          f"{new/old:.2f}× throughput · one layer")
    panel(draw, 766, "SYNTHETIC", "Long-range recall", "512 tokens · random payloads",
          (recall_entry("local_only","Local filter only"), recall_entry("shrf_gated","Gated SHRF")),
          "128 fresh episodes per model")
    output = ROOT / "docs" / "assets" / "benchmarks.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((1140, 380), Image.Resampling.LANCZOS).save(output, optimize=True)


if __name__ == "__main__":
    main()
