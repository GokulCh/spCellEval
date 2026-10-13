"""A plain-text twin of every figure: panel titles, axis labels, the plotted numbers (heatmap cells, bar heights, line / scatter
points, annotations, legend entries). Written next to each figure as ``<name>.txt`` so a figure can be read, diffed or pasted
into a chat without looking at the image."""
from __future__ import annotations

import numpy as np

MAX_CELLS, MAX_POINTS = 70, 120


def _f(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4g}"


def _ticks(labels, n):
    labs = [t.get_text() for t in labels]
    return labs if len(labs) == n and any(labs) else [str(i) for i in range(n)]


def describe(fig, title: str = "") -> str:
    out = [f"FIGURE {title}".rstrip()]
    for k, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        head = f"[panel {k}] title={ax.get_title()!r} x_label={ax.get_xlabel()!r} y_label={ax.get_ylabel()!r}"
        body: list[str] = []
        for im in ax.images:                                              # heatmaps
            a = np.asarray(im.get_array(), float)
            if a.ndim != 2:
                continue
            xt, yt = _ticks(ax.get_xticklabels(), a.shape[1]), _ticks(ax.get_yticklabels(), a.shape[0])
            r, c = min(a.shape[0], MAX_CELLS), min(a.shape[1], MAX_CELLS)
            body.append("heatmap (rows = y ticks, columns = x ticks)" + (f"; first {r}x{c} of {a.shape[0]}x{a.shape[1]} shown" if (r, c) != a.shape else ""))
            body.append("\t" + "\t".join(xt[:c]))
            body += [yt[i] + "\t" + "\t".join(_f(a[i, j]) for j in range(c)) for i in range(r)]
        for cont in ax.containers:                                        # bars (error-bar containers have no patches: their points come from ax.lines)
            if not hasattr(cont, "patches"):
                continue
            lab = cont.get_label()
            horiz = getattr(cont, "orientation", None) == "horizontal"
            ticks = _ticks(ax.get_yticklabels() if horiz else ax.get_xticklabels(), len(ax.get_yticks() if horiz else ax.get_xticks()))
            pos = np.asarray(ax.get_yticks() if horiz else ax.get_xticks(), float)
            rows = []
            for p in list(cont.patches)[:MAX_POINTS]:
                c = p.get_y() + p.get_height() / 2 if horiz else p.get_x() + p.get_width() / 2
                name = ticks[int(np.argmin(abs(pos - c)))] if len(pos) else _f(c)
                rows.append(f"{name}={_f(p.get_width() if horiz else p.get_height())}")
            if rows:
                body.append(f"bars{'' if lab.startswith('_') else ' ' + repr(lab)}: " + "; ".join(rows))
        for ln in ax.lines[:40]:                                          # lines
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if len(x) > 1:
                lab = ln.get_label()
                body.append(f"line{'' if lab.startswith('_') else ' ' + repr(lab)}: " + "; ".join(f"({_f(a)}, {_f(b)})" for a, b in zip(x[:MAX_POINTS], y[:MAX_POINTS])))
        for col in ax.collections:                                        # scatter (colour bars are QuadMesh: skipped)
            if type(col).__name__ != "PathCollection":
                continue
            off = np.asarray(col.get_offsets(), float)
            if off.ndim != 2 or off.shape[1] != 2 or not len(off):
                continue
            lab = col.get_label()
            if len(off) <= MAX_POINTS:
                body.append(f"points{'' if lab.startswith('_') else ' ' + repr(lab)}: " + "; ".join(f"({_f(a)}, {_f(b)})" for a, b in off))
            else:
                body.append(f"points{'' if lab.startswith('_') else ' ' + repr(lab)}: n={len(off)} x=[{_f(off[:, 0].min())}, {_f(off[:, 0].max())}] "
                            f"y=[{_f(off[:, 1].min())}, {_f(off[:, 1].max())}] (too many to list)")
        labels = [t.get_text() for t in ax.texts if t.get_text().strip()]
        if labels and not ax.images:                                      # point / bar annotations
            body.append("annotations: " + "; ".join(f"{t.get_text()} @({_f(t.get_position()[0])}, {_f(t.get_position()[1])})" for t in ax.texts[:MAX_POINTS] if t.get_text().strip()))
        leg = ax.get_legend()
        if leg is not None:
            body.append("legend: " + ", ".join(t.get_text() for t in leg.get_texts()))
        if body or ax.get_title():
            out += [head, *("  " + b for b in body)]
    return "\n".join(out) + "\n"
