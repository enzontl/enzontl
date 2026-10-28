"""Photo -> line drawing -> ASCII by glyph matching -> typing-animation SVG (hd-portrait.svg).

Detected edges and dark areas give the glasses and eyes. The head outline, ears, nose, mouth
and whiskers are too low-contrast in the source photo, so they are traced by hand below
(source pixels).
"""
import sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from make_portrait import build

COLS = 100
CROP = (60, 92, 460, 367)            # left, top, right, bottom in source pixels
CELL_W, CELL_H = 12, 25              # glyph cell at matching resolution (0.48 aspect)
GLYPHS = " .,'`-_~:;|/\\()<>^v=+*oO#@"

HEAD = [(115, 215), (117, 255), (128, 295), (150, 328), (200, 349), (250, 354), (300, 347),
        (343, 325), (360, 290), (367, 250), (370, 212)]
EAR_L = [(110, 148), (80, 110), (140, 128)]
EAR_R = [(330, 125), (373, 102), (379, 148)]
FOREHEAD = [(140, 128), (190, 122), (250, 121), (300, 122), (330, 125)]
INNER_EARS = [[(92, 121), (118, 139)], [(365, 113), (346, 131)]]
MOUTH = [[(247, 240), (247, 272)], [(214, 284), (231, 289), (247, 278), (263, 289), (281, 284)]]
NOSE = [(222, 211), (235, 206), (247, 205), (259, 206), (272, 211), (262, 226), (250, 240), (244, 240),
        (232, 226), (222, 211)]
WHISKERS = [[(206, 250), (150, 247), (90, 252), (40, 262)], [(206, 258), (140, 264), (80, 280), (40, 296)],
            [(209, 266), (150, 287), (100, 312), (60, 340)],
            [(288, 248), (345, 243), (405, 248), (455, 258)], [(288, 256), (355, 262), (415, 278), (455, 294)],
            [(285, 264), (345, 285), (395, 310), (440, 338)]]
EYES = [((175, 185), 22), ((312, 180), 22)]
SPECULAR = 5                         # catchlight radius, up and right of each pupil centre
NOSE_TIP = (247, 222)                # fur texture leans away from here
FUR_BASE, FUR_GAIN = 0.10, 0.55     # chance of a fur mark: base everywhere in the head, plus shade
GLASSES = (100, 138, 382, 238)       # frame box: drawn from its centreline, not its two edges


def skeleton(mask):
    sk = np.zeros_like(mask)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    m = mask.copy()
    while m.any():
        er = cv2.erode(m, k)
        sk |= m & ~cv2.dilate(er, k)
        m = er
    return sk


def smooth(pts, n=80):
    p = np.array(pts, float)
    if len(p) < 3:
        return p
    t = np.linspace(0, 1, len(p))
    tt = np.linspace(0, 1, n)
    return np.stack([np.interp(tt, t, p[:, 0]), np.interp(tt, t, p[:, 1])], 1)


def line_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # detected edges, kept inside the head
    big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    e = cv2.Canny(cv2.GaussianBlur(cv2.bilateralFilter(big, 9, 40, 40), (0, 0), 2.0), 35, 90)
    e = cv2.resize(e, (w, h), interpolation=cv2.INTER_AREA) > 40
    head = np.zeros((h, w), np.uint8)
    cv2.fillPoly(head, [np.array(EAR_L + FOREHEAD + EAR_R + HEAD[::-1], np.int32)], 1)
    gx0, gy0, gx1, gy1 = GLASSES
    e[gy0:gy1, gx0:gx1] = False
    e[:140, :] = False                         # the TV behind the ears
    keep = np.zeros((h, w), bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(e.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        comp = lab == i
        if (comp & (head > 0)).sum() > 0.6 * area and area > 6:
            keep |= comp

    S = 3                                      # matching resolution: one source px -> 3 px
    canvas = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(canvas)
    ys, xs = np.nonzero(keep)
    for x, y in zip(xs, ys):
        d.ellipse([x * S - 1, y * S - 1, x * S + 1, y * S + 1], fill=255)

    # glasses: centreline of the dark frame, at 3x so the skeleton stays smooth
    frame = np.zeros((h, w), np.uint8)
    frame[gy0:gy1, gx0:gx1] = g[gy0:gy1, gx0:gx1] < 60
    for (cx, cy), r in EYES:                 # pupils are dark too; they are filled separately
        cv2.circle(frame, (cx, cy), r, 0, -1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    big_frame = cv2.resize(frame * 255, (w * S, h * S), interpolation=cv2.INTER_LINEAR) > 127
    sk = skeleton(big_frame.astype(np.uint8))
    sk = cv2.dilate(sk, np.ones((3, 3), np.uint8))

    def stroke(pts, width=3):
        q = [(x * S, y * S) for x, y in smooth(pts)]
        d.line(q, fill=255, width=width, joint="curve")

    stroke(HEAD); stroke(EAR_L); stroke(EAR_R); stroke(FOREHEAD)
    for s in INNER_EARS:
        stroke(s, 3)
    for s in MOUTH:
        stroke(s, 2)
    for s in WHISKERS:
        stroke(s, 2)
    d.line([(x * S, y * S) for x, y in NOSE], fill=255, width=3, joint="curve")

    # pupils: solid, so the eyes read from a distance
    for (cx, cy), r in EYES:
        yy, xx = np.ogrid[:h, :w]
        pupil = ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r) & (g < 42)
        pupil = cv2.morphologyEx(pupil.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        canvas_np = np.array(canvas)
        canvas_np[cv2.resize(pupil, (w * S, h * S), interpolation=cv2.INTER_NEAREST) > 0] = 255
        canvas = Image.fromarray(canvas_np)
        d = ImageDraw.Draw(canvas)
        hx, hy = (cx + 7) * S, (cy - 7) * S
        d.ellipse([hx - SPECULAR * S, hy - SPECULAR * S, hx + SPECULAR * S, hy + SPECULAR * S], fill=0)

    canvas_np = np.array(canvas)
    canvas_np[sk > 0] = 255
    canvas = Image.fromarray(canvas_np)
    # shade for the fur texture: darker fur inside the head, glasses left bare
    inside = g[head > 0]
    lo, hi = np.percentile(inside, (5, 95))
    shade = np.clip((hi - cv2.GaussianBlur(g, (0, 0), 3).astype(float)) / max(hi - lo, 1), 0, 1) * head
    for (cx, cy), r in EYES:                   # keep the lenses clean around the pupils
        cv2.circle(shade, (cx, cy), r + 18, 0, -1)

    l, t, r, b = CROP
    return np.array(canvas.crop((l * S, t * S, r * S, b * S))), shade[t:b, l:r]


def glyph_bank(ttf):
    f = ImageFont.truetype(ttf, 20)
    bank = []
    for ch in GLYPHS:
        im = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(im).text((0, 1), ch, font=f, fill=255)
        bank.append(cv2.GaussianBlur(np.array(im, float) / 255, (0, 0), 1.6))
    return np.stack(bank)


def fur(v, r, c, nr, nc, rng):
    """A light mark for a bare cell inside the head, following fur that grows out from the nose."""
    if v <= 0 or rng.random() > FUR_BASE + FUR_GAIN * v:
        return " "
    if v > 0.75 and rng.random() < 0.5:
        return ":" if rng.random() < 0.6 else ";"
    dx, dy = c - nc, (r - nr) * 2.1          # rows are about twice as tall as columns
    if abs(dx) > 2.2 * abs(dy):              # cheeks and whisker pads: fur runs sideways
        return "-" if rng.random() < 0.2 else ("." if dy > 0 else "`")
    if abs(dy) > 2.2 * abs(dx):              # forehead and chin: fur runs up and down
        return "'" if dy < 0 else ","
    if (dx > 0) == (dy > 0):                 # down-right or up-left
        return "\\" if rng.random() < 0.12 else ("," if dy > 0 else "`")
    return "/" if rng.random() < 0.12 else ("." if dy > 0 else "'")


def to_rows(lines, shade, ttf):
    bank = glyph_bank(ttf)
    lines = cv2.resize(lines, (COLS * CELL_W, int(round(lines.shape[0] * COLS * CELL_W / lines.shape[1]))))
    rows = lines.shape[0] // CELL_H
    src = cv2.GaussianBlur(lines.astype(float) / 255, (0, 0), 1.6)
    grid = [[" "] * COLS for _ in range(rows)]
    for r in range(rows):
        for c in range(COLS):
            cell = src[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W]
            if cell.max() < 0.12:
                continue
            err = ((bank - cell) ** 2).sum(axis=(1, 2))
            err[0] *= 1.15                     # nudge faint cells toward a glyph rather than blank
            grid[r][c] = GLYPHS[int(err.argmin())]

    # fur texture in bare cells, kept one cell clear of every line so outlines stay crisp
    sh = cv2.resize(shade, (COLS, rows), interpolation=cv2.INTER_AREA)
    l, t, rr, b = CROP
    nc = int((NOSE_TIP[0] - l) / (rr - l) * COLS)
    nr = int((NOSE_TIP[1] - t) / (b - t) * rows)
    drawn = np.array([[ch != " " for ch in row] for row in grid])
    near = cv2.dilate(drawn.astype(np.uint8), np.ones((1, 3), np.uint8)) > 0
    rng = np.random.default_rng(7)             # fixed seed: same texture on every run
    for r in range(rows):
        for c in range(COLS):
            if not near[r, c]:
                grid[r][c] = fur(sh[r, c], r, c, nr, nc, rng)
    out = ["".join(row).rstrip() for row in grid]
    while out and not out[-1]:
        out.pop()
    return out


if __name__ == "__main__":
    photo, ttf, out = sys.argv[1:4]
    rows = to_rows(*line_image(photo), ttf)
    open(out.replace(".svg", ".txt"), "w").write("\n".join(rows))
    open(out, "w").write(build((rows, rows), ttf))
    print(len(rows), "rows")
