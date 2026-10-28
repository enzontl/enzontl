"""Photo -> line drawing -> ASCII by glyph matching -> typing-animation SVG (hd-portrait.svg).

Detected edges give the glasses, eyes and whiskers. The head outline, ears and mouth are
too low-contrast in the source photo, so they are traced by hand below (source pixels).
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
MOUTH = [[(246, 242), (247, 272)], [(214, 284), (231, 289), (247, 278), (263, 289), (281, 284)]]
EYES = [((175, 185), 22), ((312, 180), 22)]
WHISKER_BAND = (222, 395)            # y range where edges outside the head are kept
CLOTH = (0, 215, 118, 280)           # crumpled sheet left of the head: not whiskers
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

    # detected edges, kept inside the head or as whiskers in the band beside it
    big = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    e = cv2.Canny(cv2.GaussianBlur(cv2.bilateralFilter(big, 9, 40, 40), (0, 0), 2.0), 35, 90)
    e = cv2.resize(e, (w, h), interpolation=cv2.INTER_AREA) > 40
    head = np.zeros((h, w), np.uint8)
    cv2.fillPoly(head, [np.array(EAR_L + FOREHEAD + EAR_R + HEAD[::-1], np.int32)], 1)
    gx0, gy0, gx1, gy1 = GLASSES
    e[gy0:gy1, gx0:gx1] = False
    cx0, cy0, cx1, cy1 = CLOTH
    e[cy0:cy1, cx0:cx1] = False
    keep = np.zeros((h, w), bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(e.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        comp = lab == i
        if (comp & (head > 0)).sum() > 0.6 * area:
            keep |= comp
        elif WHISKER_BAND[0] < y and y + bh < WHISKER_BAND[1] and bw > 3.5 * bh and bw > 22:
            keep |= comp                       # long, flat strokes: whiskers, not door frames

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
    for s in INNER_EARS + MOUTH:
        stroke(s, 3)

    # nose: the pink patch, found on the Lab a* channel
    a = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 1].astype(float)
    nose = np.zeros((h, w), np.uint8)
    roi = a[195:250, 210:285]
    nose[195:250, 210:285] = (roi > np.percentile(roi, 72)).astype(np.uint8)
    nose = cv2.morphologyEx(nose, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(nose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cs:
        c = max(cs, key=cv2.contourArea)[:, 0, :]
        d.line([(x * S, y * S) for x, y in np.vstack([c, c[:1]])], fill=255, width=3)

    # pupils: solid, so the eyes read from a distance
    for (cx, cy), r in EYES:
        yy, xx = np.ogrid[:h, :w]
        pupil = ((xx - cx) ** 2 + (yy - cy) ** 2 < r * r) & (g < 42)
        pupil = cv2.morphologyEx(pupil.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        canvas_np = np.array(canvas)
        canvas_np[cv2.resize(pupil, (w * S, h * S), interpolation=cv2.INTER_NEAREST) > 0] = 255
        canvas = Image.fromarray(canvas_np)
        d = ImageDraw.Draw(canvas)

    canvas_np = np.array(canvas)
    canvas_np[sk > 0] = 255
    canvas = Image.fromarray(canvas_np)
    l, t, r, b = CROP
    return np.array(canvas.crop((l * S, t * S, r * S, b * S)))


def glyph_bank(ttf):
    f = ImageFont.truetype(ttf, 20)
    bank = []
    for ch in GLYPHS:
        im = Image.new("L", (CELL_W, CELL_H), 0)
        ImageDraw.Draw(im).text((0, 1), ch, font=f, fill=255)
        bank.append(cv2.GaussianBlur(np.array(im, float) / 255, (0, 0), 1.6))
    return np.stack(bank)


def to_rows(lines, ttf):
    bank = glyph_bank(ttf)
    lines = cv2.resize(lines, (COLS * CELL_W, int(round(lines.shape[0] * COLS * CELL_W / lines.shape[1]))))
    rows = lines.shape[0] // CELL_H
    src = cv2.GaussianBlur(lines.astype(float) / 255, (0, 0), 1.6)
    out = []
    for r in range(rows):
        line = ""
        for c in range(COLS):
            cell = src[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W]
            if cell.max() < 0.12:
                line += " "
                continue
            err = ((bank - cell) ** 2).sum(axis=(1, 2))
            err[0] *= 1.15                     # nudge faint cells toward a glyph rather than blank
            line += GLYPHS[int(err.argmin())]
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return out


if __name__ == "__main__":
    photo, ttf, out = sys.argv[1:4]
    rows = to_rows(line_image(photo), ttf)
    open(out.replace(".svg", ".txt"), "w").write("\n".join(rows))
    open(out, "w").write(build((rows, rows), ttf))
    print(len(rows), "rows")
