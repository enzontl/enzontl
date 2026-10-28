"""Photo -> ASCII rows -> typing-animation SVG (hd-portrait.svg)."""
import sys, base64, io
import numpy as np, cv2
from PIL import Image
from rembg import remove

RAMP = " .`:-=+*cs#%@"
COLS, FONT, CHAR_W = 90, 12.9, 7.74
LINE_H = CHAR_W / 0.48
GAMMA, CLIP = 1.25, 4.0
CROP = (55, 100, 405, 375)  # (left, top, right, bottom) in source pixels
FLOOR = 0.26  # minimum ink inside the subject, so pale subjects keep a silhouette
EDGE = 0.55  # weight of the edge layer; helps pale subjects

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
        alpha = grabcut(rgb)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 9, 60, 60)
    g = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(8, 8)).apply(g)
    v = (g / 255.0) ** GAMMA
    blur = cv2.GaussianBlur(g, (0, 0), 2.2).astype(float)
    dog = np.clip((blur - cv2.GaussianBlur(g, (0, 0), 0.8)) / 14.0, 0, 1)
    v = np.clip(v - EDGE * dog, 0, 1)
    ink = FLOOR + (1 - FLOOR) * (1 - v)
    v = 1 - ink * alpha                     # outside subject -> white (blank)
    h, w = v.shape
    rows = max(1, int(COLS * (h / w) * 0.48))
    small = cv2.resize(v, (COLS, rows), interpolation=cv2.INTER_AREA)
    idx = ((1 - small) * (len(RAMP) - 1)).round().astype(int)
    return ["".join(RAMP[i] for i in r).rstrip() for r in idx]

def font_b64(ttf, chars):
    import subprocess, tempfile, os
    out = tempfile.mktemp(suffix=".woff2")
    subprocess.run(["pyftsubset", ttf, f"--text={chars}", "--flavor=woff2",
                    "--layout-features=", "--no-hinting", f"--output-file={out}"], check=True)
    return base64.b64encode(open(out, "rb").read()).decode()

def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build(rows, ttf, fill="#e6e6e6"):
    W, H = COLS * CHAR_W, len(rows) * LINE_H + 8
    b64 = font_b64(ttf, RAMP)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.2f} {H:.2f}" role="img" aria-label="ASCII portrait">',
         f'<defs><style>@font-face{{font-family:"M";src:url(data:font/woff2;base64,{b64}) format("woff2")}}'
         f'text{{font-family:"M",monospace;font-size:{FONT}px;fill:#24292f;white-space:pre}}.k{{fill:#24292f}}'
         f'@media(prefers-color-scheme:dark){{text,.k{{fill:#e6edf3}}}}</style>']
    for i in range(len(rows)):
        o.append(f'<clipPath id="c{i}"><rect x="0" y="{i*LINE_H:.2f}" width="0" height="{LINE_H+2:.2f}">'
                 f'<animate attributeName="width" from="0" to="{W:.2f}" dur="0.9s" begin="{i*0.09:.2f}s" fill="freeze"/></rect></clipPath>')
    o.append("</defs>")
    for i, r in enumerate(rows):
        y = (i + 1) * LINE_H - 3
        o.append(f'<text x="0" y="{y:.2f}" xml:space="preserve" clip-path="url(#c{i})">{esc(r)}</text>')
        o.append(f'<rect width="{CHAR_W:.2f}" height="{LINE_H-2:.2f}" x="0" y="{i*LINE_H+1:.2f}" class="k" opacity="0">'
                 f'<set attributeName="opacity" to="0.9" begin="{i*0.09:.2f}s"/>'
                 f'<animate attributeName="x" from="0" to="{W:.2f}" dur="0.9s" begin="{i*0.09:.2f}s" fill="freeze"/>'
                 f'<set attributeName="opacity" to="0" begin="{i*0.09+0.9:.2f}s"/></rect>')
    o.append("</svg>")
    return "\n".join(o)

if __name__ == "__main__":
    photo, ttf, out = sys.argv[1:4]
    rows = rows_from(photo)
    open(out.replace(".svg", ".txt"), "w").write("\n".join(rows))
    open(out, "w").write(build(rows, ttf))
    print(len(rows), "rows")
