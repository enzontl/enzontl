"""Photo -> ASCII rows -> typing-animation SVG (hd-portrait.svg)."""
import sys, base64, io
import numpy as np, cv2
from PIL import Image
from rembg import remove

RAMP = " .`:-=+*cs#%@"
COLS, FONT, CHAR_W = 100, 12.9, 7.74
LINE_H = CHAR_W / 0.48
GAMMA, CLIP = 1.25, 4.0
CROP = (45, 70, 415, 395)  # (left, top, right, bottom) in source pixels
FLOOR = 0.18  # minimum ink inside the subject, so pale subjects keep a silhouette
EDGE = 0.8  # weight of the edge layer; helps pale subjects

# ears and forehead, in CROP coordinates; GrabCut alone clips them off
HEAD = [(37, 60), (95, 72), (165, 64), (240, 66), (285, 66), (334, 31), (322, 112), (290, 104),
        (100, 104), (58, 104)]


def poly_mask(shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(HEAD, np.int32)], 1)
    return cv2.GaussianBlur(m.astype(float), (0, 0), 1.5)


def grabcut(rgb):
    h, w = rgb.shape[:2]
    m = np.zeros((h, w), np.uint8)
    rect = (int(w * .03), int(h * .02), int(w * .94), int(h * .96))
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m, rect, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_RECT)
    a = np.where((m == 1) | (m == 3), 1.0, 0.0)
    return cv2.GaussianBlur(a, (0, 0), 1.5)


def rows_from(path):
    img = Image.open(path).convert("RGB")
    if CROP: img = img.crop(CROP)
    cut = remove(img)                       # RGBA, background transparent
    alpha = np.array(cut)[:, :, 3] / 255.0
    rgb = np.array(img)
    if alpha.mean() < 0.25:                 # rembg latched onto an accessory (glasses): GrabCut the whole subject
        alpha = np.maximum(grabcut(rgb), poly_mask(rgb.shape[:2]))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 9, 60, 60)
    g = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(8, 8)).apply(g)
    inside = g[alpha > 0.5]
    lo, hi = np.percentile(inside, (2, 98))
    t = np.clip((g - lo) / max(hi - lo, 1), 0, 1)       # tone stretched over the subject only
    blur = cv2.GaussianBlur(g, (0, 0), 2.2).astype(float)
    dog = np.clip((blur - cv2.GaussianBlur(g, (0, 0), 0.8)) / 14.0, 0, 1)
    t = np.clip(t - EDGE * dog, 0, 1)
    h, w = t.shape
    rows = max(1, int(COLS * (h / w) * 0.48))

    def to_rows(density):                    # density 0..1 -> text rows
        small = cv2.resize(density, (COLS, rows), interpolation=cv2.INTER_AREA)
        idx = (small.clip(0, 1) * (len(RAMP) - 1)).round().astype(int)
        return ["".join(RAMP[i] for i in r).rstrip() for r in idx]

    neg = alpha * (FLOOR + (1 - FLOOR) * (1 - t ** GAMMA))   # light theme: shadows are ink
    pos = alpha * (0.04 + 0.96 * t ** 0.9)                   # dark theme: light is ink
    return to_rows(neg), to_rows(pos)

def font_b64(ttf, chars):
    import subprocess, tempfile, os
    out = tempfile.mktemp(suffix=".woff2")
    subprocess.run(["pyftsubset", ttf, f"--text={chars}", "--flavor=woff2",
                    "--layout-features=", "--no-hinting", f"--output-file={out}"], check=True)
    return base64.b64encode(open(out, "rb").read()).decode()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build(layers, ttf):
    neg, pos = layers
    n = len(neg)
    W, H = COLS * CHAR_W, n * LINE_H + 8
    chars = "".join(sorted(set("".join(neg + pos)) | {" "}))
    b64 = font_b64(ttf, chars)
    layered = neg is not pos                 # one layer per theme, or a single shared one
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.2f} {H:.2f}" role="img" aria-label="ASCII portrait of a cat wearing glasses">',
         f'<defs><style>@font-face{{font-family:"M";src:url(data:font/woff2;base64,{b64}) format("woff2")}}'
         f'text{{font-family:"M",monospace;font-size:{FONT}px;fill:#24292f;white-space:pre}}.k{{fill:#24292f}}.p{{display:none}}'
         f'@media(prefers-color-scheme:dark){{text,.k{{fill:#e6edf3}}.p{{display:inline}}.n{{display:none}}}}</style>']
    for i in range(n):
        o.append(f'<clipPath id="c{i}"><rect x="0" y="{i*LINE_H:.2f}" width="0" height="{LINE_H+2:.2f}">'
                 f'<animate attributeName="width" from="0" to="{W:.2f}" dur="0.9s" begin="{i*0.09:.2f}s" fill="freeze"/></rect></clipPath>')
    o.append("</defs>")
    for i in range(n):
        y = (i + 1) * LINE_H - 3
        for cls, rows in ((("n", neg), ("p", pos)) if layered else (("", neg),)):
            attr = f' class="{cls}"' if cls else ""
            o.append(f'<text{attr} x="0" y="{y:.2f}" xml:space="preserve" clip-path="url(#c{i})">{esc(rows[i])}</text>')
        o.append(f'<rect width="{CHAR_W:.2f}" height="{LINE_H-2:.2f}" x="0" y="{i*LINE_H+1:.2f}" class="k" opacity="0">'
                 f'<set attributeName="opacity" to="0.9" begin="{i*0.09:.2f}s"/>'
                 f'<animate attributeName="x" from="0" to="{W:.2f}" dur="0.9s" begin="{i*0.09:.2f}s" fill="freeze"/>'
                 f'<set attributeName="opacity" to="0" begin="{i*0.09+0.9:.2f}s"/></rect>')
    o.append("</svg>")
    return "\n".join(o)

if __name__ == "__main__":
    photo, ttf, out = sys.argv[1:4]
    layers = rows_from(photo)
    open(out.replace(".svg", ".txt"), "w").write("\n".join(layers[1]))
    open(out, "w").write(build(layers, ttf))
    print(len(layers[0]), "rows")
