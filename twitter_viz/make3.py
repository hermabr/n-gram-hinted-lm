"""Third sketch: context -> training data (CPU) -> hint fed into the layers of the model (GPU), bars beside the stack."""
from make2 import HAND, HERE, LM, THEMES, VOTES, Canvas


GAIN_PX = {"hint": 176}   # fixed block heights; others are drawn from their votes


def block(c, x, cy, w, h, label, fs=29):
    th = c.th
    c.rect(x, cy - h / 2, w, h, th["chip"], th["ink"], 2.5, rx=12)
    c.text(x + w / 2, cy + fs * 0.36 * c.f["scale"], label, fs, th["ink"], "middle", "bold")


def draw(theme, ctx_in=False, badge=False, plus=False, fan=False, striped=False, ctx=("this", "is", "a"), words=None,
         ctx_fs=38, line_fs=30, gpu_x=920, fan_dx=160, bar_gap=16):
    word = lambda n: (words or {}).get(n, n)   # shown label for the hint / trick / cool slots
    th = THEMES[theme]
    c = Canvas(th)
    c.solid = "tint"
    ink, muted = th["ink"], th["muted"]
    col = {"hint": th["hint"], "trick": th["trick"]}
    tint = {"hint": th["hint_bg"], "trick": th["trick_bg"]}
    hand = th["font"] is HAND
    fg = th["panel"] if th["panel"] != "none" else th["bg"]

    # context
    # with fan, the context, the middle matched line and the hint line all sit at the height where the hint fans out
    fy = 70 + (400 if plus else 475)
    x, cy, fs = 40, fy if fan else 460, ctx_fs
    ctx_l = x
    for tok in ctx:
        x += c.chip(x, cy - fs * 0.875, tok, fs) + 12
    ctx_r = x - 12
    ctx_top = cy - fs * 0.875

    # training data (CPU)
    bx, by, bw, bh = (352, fy - 214, 462, 345) if fan else (352, 265, 440, 380)
    pres = [p + " " + " ".join(ctx) + " " for p in ["… so", "… but,", "… and"]]
    if fan:   # grow to fit a longer context, keeping the default layout as is
        bx = max(bx, ctx_r + 76)
        bw = max(bw, 32 + max(c.tw(p, line_fs) + c.tw(word(n), line_fs, "bold") for p, n in zip(pres, ["hint", "trick"] * 2))
                 + 4 + 0.6 * line_fs + 90 + 42)
    c.rect(bx, by, bw, bh, th["panel"], th["panel_stroke"], th["panel_sw"], rx=22)
    if fan:   # tighter header, so the embedding line can pass above the box
        c.cpu(bx + 22, by + 20, 76, ink, fg)
        c.text(bx + 114, by + 70, "Training data", 36, ink, weight="bold")
    else:
        c.cpu(bx + 22, by + 26, 84, ink, fg)
        c.text(bx + 124, by + 79, "Training data", 36, ink, weight="bold")
    c.arrow(f"M{ctx_r + 12},{cy} L{bx - 14},{cy}")
    ends = []
    for i, nxt in enumerate(["hint", "trick", "hint"]):
        y = by + (160 if fan else 178) + i * 64
        pre, ch = pres[i], round(46 * line_fs / 30)
        c.text(bx + 32, y, pre, line_fs, muted)
        tx = bx + 32 + c.tw(pre, line_fs) + 4
        nw = c.chip(tx, y - 10 - ch / 2, word(nxt), line_fs, stroke=col[nxt], weight="bold", sw=3, pad=0.45 if hand else 0.3, h=ch,
                    fill=tint[nxt])
        ends.append((tx + nw, y - 10, nxt))

    # model (GPU)
    lx, ly, lw, lh = (gpu_x, 70, 1570 - gpu_x, 760) if fan else (975, 70, 595, 760)
    c.rect(lx, ly, lw, lh, th["panel"], th["panel_stroke"], th["panel_sw"], rx=22)
    c.gpu(lx + 28, ly + 32, 60, ink, fg)
    sx, sw_, bh_ = lx + (fan_dx if fan else 92), 230, 68
    ys = {"embedding": ly + (150 if fan else 175), "transformer 1": ly + 325, "transformer 2": ly + 475, "LM head": ly + 625}
    for name, yc in ys.items():
        block(c, sx, yc, sw_, bh_, name.split(" ")[0] if name.startswith("transformer") else name)
    yv = list(ys.values())
    mid = sx + sw_ / 2
    adds = []   # where the hint goes in
    for a, b in zip(yv, yv[1:]):
        if plus:
            m = (a + b) / 2
            c.add(f'<line x1="{mid}" y1="{a + bh_ / 2}" x2="{mid}" y2="{m - 15}" stroke="{ink}" stroke-width="3"/>')
            c.add(f'<circle cx="{mid}" cy="{m}" r="15" fill="{fg}" stroke="{ink}" stroke-width="3"/>')
            c.add(f'<path d="M{mid - 8},{m} H{mid + 8} M{mid},{m - 8} V{m + 8}" stroke="{ink}" stroke-width="3"/>')
            c.arrow(f"M{mid},{m + 15} L{mid},{b - bh_ / 2 - 6}", sw=3)
            adds.append((m, mid - 15))
        else:
            c.arrow(f"M{mid},{a + bh_ / 2 + 4} L{mid},{b - bh_ / 2 - 6}", sw=3)
    if not plus:
        adds = [(y, sx) for y in yv[1:]]

    # the matched next tokens merge into one hint, which goes into the stack
    if fan:   # merge inside the CPU box, one line across, fan out inside the GPU box
        mx, my = bx + bw - 42, ends[1][1]
        for ex, ey, n in ends:
            c.add(f'<path d="M{ex + 6},{ey} C{ex + 50},{ey} {mx - 50},{my} {mx},{my}" fill="none" stroke="{col[n]}" '
                  f'stroke-width="3.5" stroke-linecap="round"/>')
        fx = lx + 36
        assert my == fy == adds[1][0]
        paths = [f"M{mx},{my} H{fx}"] + [f"M{fx},{fy} C{fx + 50},{fy} {sx - 70},{y} {sx - 10},{y} H{x2 - 8}"   # level before the stack
                                         for y, x2 in adds]
        if striped:   # keep both colours: pink line with orange dashes
            c.add(f'<path d="{paths[0]}" fill="none" stroke="{col["hint"]}" stroke-width="3.5"/>')
            for d in paths[1:]:
                c.arrow(d, col["hint"], sw=3.5)
            for d in paths:
                c.add(f'<path d="{d}" fill="none" stroke="{col["trick"]}" stroke-width="3.5" stroke-dasharray="22 22"/>')
        else:
            c.add(f'<path d="{paths[0]}" fill="none" stroke="{ink}" stroke-width="3.5"/>')
            for d in paths[1:]:
                c.arrow(d, sw=3.5)
    else:
        mx, my = (890, by - 82) if badge else (884, by + 60)   # the badge needs more room than the gap between boxes
        if badge:
            bw2, bh2 = 150, 116
            c.rect(mx - bw2 / 2, my - bh2 / 2, bw2, bh2, th["panel"] if th["panel"] != "none" else th["bg"], ink, 2.5, rx=14)
            for j, (n, v) in enumerate(VOTES.items()):
                yy = my - 25 + j * 52
                w = c.chip(mx - bw2 / 2 + 14, yy - 20, n, 25, stroke=col[n], weight="bold", sw=2.5, pad=0.3, h=40, fill=tint[n])
                c.text(mx - bw2 / 2 + 14 + w + 8, yy + 9, f"×{v}", 25, ink, weight="bold")
            inx, outx = mx - bw2 / 2, mx + bw2 / 2
        else:
            c.add(f'<circle cx="{mx}" cy="{my}" r="7" fill="{ink}"/>')
            inx = outx = mx
        for ex, ey, n in ends:
            c.add(f'<path d="M{ex + 6},{ey} C{ex + 90},{ey} {inx - 90},{my} {inx},{my}" fill="none" stroke="{col[n]}" '
                  f'stroke-width="3.5" stroke-linecap="round"/>')
        busx, r = lx + 50, 18
        last_y, last_x = adds[-1]
        c.arrow(f"M{outx},{my} H{busx - r} Q{busx},{my} {busx},{my + r} V{last_y - r} Q{busx},{last_y} {busx + r},{last_y} "
                f"H{last_x - 8}", sw=3.5)
        for y, x2 in adds[:-1]:
            c.arrow(f"M{busx},{y} H{x2 - 8}", sw=3.5)
        c.text(mx, my - (72 if badge else 22), "hint", 26, ink, "middle", "bold")

    # context also goes into the embedding
    if ctx_in:
        a = (ctx_l + ctx_r) / 2
        top = 150
        if fan:   # straight up, one bend, straight into the embedding
            ye, r = ys["embedding"], 40
            c.arrow(f"M{a},{ctx_top - 12} V{ye + r} Q{a},{ye} {a + r},{ye} H{sx - 8}", sw=3.5)
        else:
            c.arrow(f"M{a},{ctx_top - 12} C{a},{top} {a + 40},{top} {a + 160},{top} H{lx - 120} "
                    f"C{lx - 40},{top} {lx - 40},{ys['embedding']} {lx + 20},{ys['embedding']} H{sx - 8}", sw=3.5)

    # legend + bars
    label = "gain from hints"
    lt = lx + lw - 28 - c.tw(label, 24)   # right-aligned with a 28 px margin
    c.hatched(lt - 86, ly + 44, 30, 30, col["hint"])
    c.hatched(lt - 50, ly + 44, 30, 30, col["trick"])
    c.text(lt, ly + 68, label, 24, ink)
    w3, g3, U, V, G = (54, bar_gap, 480, 110, 6) if fan else (58, 18, 480, 110, 6)
    x0, base = sx + (258 if fan else 270), ys["LM head"] + bh_ / 2
    for i, n in enumerate(["hint", "cool", "trick"]):
        xx = x0 + i * (w3 + g3)
        h = LM[n] * U
        if th["lm_stroke"] != "none":
            c.rect(xx + 1.25, base - h + 1.25, w3 - 2.5, h - 2.5, th["lm_fill"], th["lm_stroke"], 2.5, rx=5)
        else:
            c.rect(xx, base - h, w3, h, th["lm_fill"], rx=6)
        if VOTES.get(n):   # one block per token
            gh = GAIN_PX.get(n, VOTES[n] * V - G)
            c.hatched(xx, base - h - G - gh, w3, gh, col[n])
        c.text(xx + w3 / 2, base + 38, word(n), 27, ink, "middle", "bold")
    c.add(f'<line x1="{x0 - 10}" y1="{base}" x2="{x0 + 3 * w3 + 2 * g3 + 10}" y2="{base}" stroke="{ink}" stroke-width="2.5"/>')
    return c.svg()


VARIANTS = [("f1", "blog", {}),
            ("f2", "blog", dict(ctx_in=True)),
            ("f3", "blog", dict(badge=True)),
            ("f4", "blog", dict(ctx_in=True, plus=True)),
            ("f5", "hand", {}),
            ("g1", "blog", dict(ctx_in=True, fan=True)),
            ("g2", "blog", dict(ctx_in=True, plus=True, fan=True)),
            ("g3", "blog", dict(ctx_in=True, plus=True, fan=True, striped=True)),
            ("h1", "blog", dict(ctx_in=True, plus=True, fan=True, striped=True, ctx=("hill", "climbing", "is"),
                                words=dict(hint="fun", trick="hard", cool="solved"), ctx_fs=32, line_fs=26, gpu_x=935,
                                fan_dx=124, bar_gap=30))]
if __name__ == "__main__":
    for name, theme, kw in VARIANTS:
        (HERE / f"retrieval_hints_{name}.svg").write_text(draw(theme, **kw))
    print("ok")
