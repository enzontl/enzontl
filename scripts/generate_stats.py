"""Draw the profile graphics from the GitHub GraphQL API. Standard library only.

Writes stats.svg, streak.svg, langs.svg, year.svg and the h-*.svg section headings.
Reads GITHUB_TOKEN and GH_LOGIN from the environment.
"""
import base64, json, os, sys, urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGIN = os.environ["GH_LOGIN"]
TOKEN = os.environ["GITHUB_TOKEN"]
W = 800
RAMP = "-=+*#%@"
HEADINGS = ["about", "stats", "projects", "stack", "contact"]

QUERY = """
query($login:String!,$from:DateTime!,$to:DateTime!){
  user(login:$login){
    contributionsCollection(from:$from,to:$to){
      contributionCalendar{ totalContributions weeks{ contributionDays{ date contributionCount } } }
    }
    repositories(first:100, ownerAffiliations:OWNER, privacy:PUBLIC, isFork:false){
      nodes{ languages(first:10, orderBy:{field:SIZE, direction:DESC}){ edges{ size node{ name } } } }
    }
  }
}"""


def fetch():
    # whole UTC days, so two runs on the same day bucket identically
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=364)
    variables = {"login": LOGIN, "from": f"{start}T00:00:00Z", "to": f"{today}T23:59:59Z"}
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": QUERY, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {TOKEN}", "Content-Type": "application/json", "User-Agent": "profile-stats"},
    )
    body = json.load(urllib.request.urlopen(req, timeout=60))
    if "errors" in body:
        sys.exit(json.dumps(body["errors"]))
    return body["data"]["user"], start, today


def b64(name):
    return base64.b64encode((ROOT / "assets/fonts" / name).read_bytes()).decode()


FONT_CSS = None


def css():
    global FONT_CSS
    if FONT_CSS is None:
        FONT_CSS = (
            '@font-face{font-family:"M";font-weight:400;src:url(data:font/woff2;base64,%s) format("woff2")}'
            '@font-face{font-family:"M";font-weight:700;src:url(data:font/woff2;base64,%s) format("woff2")}'
            ':root{--ink:#24292f;--dim:#6e7781;--rule:#d0d7de;--acc:#0969da}'
            '@media(prefers-color-scheme:dark){:root{--ink:#e6edf3;--dim:#7d8590;--rule:#30363d;--acc:#58a6ff}}'
            'text{font-family:"M",monospace;fill:var(--ink)}.d{fill:var(--dim)}.b{font-weight:700}'
        ) % (b64("jbm-regular.woff2"), b64("jbm-bold.woff2"))
    return FONT_CSS


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{label}">'
            f"<style>{css()}</style>{body}</svg>")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_date(d):
    return d.strftime("%b %d").replace(" 0", " ")


def write(name, text):
    (ROOT / name).write_text(text)


def streaks(days, today):
    counts = {d: c for d, c in days}
    longest = run = 0
    best_end = None
    for d, c in days:
        run = run + 1 if c else 0
        if run > longest:
            longest, best_end = run, d
    cur, cur_end = 0, None
    d = today if counts.get(today) else today - timedelta(days=1)  # today may not have a commit yet
    while counts.get(d):
        cur += 1
        cur_end = cur_end or d
        d -= timedelta(days=1)
    return (cur, cur_end), (longest, best_end)


def render_stats(days, weekly, total):
    h = 190
    mx = max(weekly) or 1
    pts = [(300 + i * 480 / (len(weekly) - 1), 150 - (v / mx) * 100) for i, v in enumerate(weekly)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"300,150 {line} 780,150"
    best = max(days, key=lambda x: x[1])
    body = (
        f'<text x="0" y="92" font-size="76" class="b">{total:,}</text>'
        f'<text x="2" y="122" font-size="14" class="d">contributions · last 12 months</text>'
        f'<text x="2" y="150" font-size="12" class="d">busiest day · {best[1]} on {fmt_date(best[0])}</text>'
        f'<polygon points="{area}" fill="var(--acc)" opacity="0.14"/>'
        f'<polyline points="{line}" fill="none" stroke="var(--acc)" stroke-width="1.6" stroke-linejoin="round"/>'
        f'<line x1="300" y1="150" x2="780" y2="150" stroke="var(--rule)"/>'
        f'<text x="300" y="172" font-size="11" class="d">{fmt_date(days[0][0])}</text>'
        f'<text x="780" y="172" font-size="11" class="d" text-anchor="end">{fmt_date(days[-1][0])}</text>'
        f'<text x="540" y="40" font-size="11" class="d" text-anchor="middle">weekly total</text>'
    )
    write("stats.svg", svg(W, h, body, f"{total} contributions in the last 12 months"))


def render_streak(cur, longest):
    def block(x, big, label, rng):
        return (f'<text x="{x}" y="78" font-size="64" class="b">{big}</text>'
                f'<text x="{x+2}" y="106" font-size="14" class="d">{label}</text>'
                f'<text x="{x+2}" y="128" font-size="12" class="d">{rng}</text>')

    def rng(n, end):
        return f"{fmt_date(end - timedelta(days=n-1))} → {fmt_date(end)}" if n else "—"

    body = (block(0, cur[0], "current streak · days", rng(*cur)) +
            f'<line x1="400" y1="20" x2="400" y2="130" stroke="var(--rule)"/>' +
            block(430, longest[0], "longest streak · days", rng(*longest)))
    write("streak.svg", svg(W, 150, body, f"current streak {cur[0]} days, longest {longest[0]} days"))


def render_langs(repos):
    size, count = {}, {}
    for r in repos:
        seen = set()
        for e in r["languages"]["edges"]:
            n = e["node"]["name"]
            size[n] = size.get(n, 0) + e["size"]
            if n not in seen:
                count[n] = count.get(n, 0) + 1
                seen.add(n)
    tot = sum(size.values()) or 1
    top = sorted(size, key=size.get, reverse=True)[:6]
    h = 30 + 30 * len(top)
    body = ""
    for i, n in enumerate(top):
        y = 24 + i * 30
        pct = size[n] / tot * 100
        body += (f'<text x="0" y="{y+11}" font-size="14">{esc(n)}</text>'
                 f'<rect x="170" y="{y}" width="440" height="14" fill="var(--rule)" opacity="0.5"/>'
                 f'<rect x="170" y="{y}" width="{440*pct/100:.1f}" height="14" fill="var(--acc)"/>'
                 f'<text x="626" y="{y+11}" font-size="13">{pct:.1f}%</text>'
                 f'<text x="720" y="{y+11}" font-size="12" class="d">{count[n]} repo{"s" if count[n] > 1 else ""}</text>')
    write("langs.svg", svg(W, h, body, "top languages: " + ", ".join(top)))


def render_year(days):
    CW, LH = 14.4, 22
    vals = sorted(c for _, c in days if c)
    q = [vals[int(len(vals) * f)] for f in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9)] if vals else []

    def ch(c):
        if not c:
            return "."
        return RAMP[sum(c > t for t in q)] if q else RAMP[0]

    first = days[0][0]
    pad = (first.weekday() + 1) % 7  # weeks start on Sunday
    cells = [None] * pad + days
    rows = [[] for _ in range(7)]
    for i, d in enumerate(cells):
        rows[i % 7].append(d)
    body = ""
    for r, row in enumerate(rows):
        for c, d in enumerate(row):
            if d is None:
                continue
            dim = '' if d[1] else ' class="d" opacity="0.45"'
            body += f'<text x="{c*CW:.1f}" y="{20+r*LH}" font-size="{CW/0.6:.1f}"{dim}>{ch(d[1])}</text>'
    wcols = max(len(r) for r in rows)
    w = round(wcols * CW)
    write("year.svg", svg(w, 20 + 7 * LH, body, "one character per day, last 12 months"))


def render_headings():
    for name in HEADINGS:
        w = W
        body = (f'<text x="0" y="22" font-size="16" class="b">{name}</text>'
                f'<line x1="{len(name)*9.6+14:.0f}" y1="17" x2="{w}" y2="17" stroke="var(--rule)"/>')
        write(f"h-{name}.svg", svg(w, 32, body, name))


def main():
    user, start, today = fetch()
    weeks = user["contributionsCollection"]["contributionCalendar"]["weeks"]
    days = [(date.fromisoformat(d["date"]), d["contributionCount"]) for w in weeks for d in w["contributionDays"]]
    days = [x for x in days if start <= x[0] <= today]
    weekly = [sum(c for _, c in days[i:i + 7]) for i in range(0, len(days), 7)]
    total = sum(c for _, c in days)
    cur, longest = streaks(days, today)
    render_stats(days, weekly, total)
    render_streak(cur, longest)
    render_langs(user["repositories"]["nodes"])
    render_year(days)
    render_headings()
    print(f"{total} contributions, streak {cur[0]}/{longest[0]}")


if __name__ == "__main__":
    main()
