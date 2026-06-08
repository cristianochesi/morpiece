#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morpiece_trie_explorer.py  —  interactive explorer for MorPiece tries
=====================================================================

Companion tool for MorPiece >= 1.4.3.  It loads the *complete* (un-pruned)
trie snapshot produced by

    tok.train(corpus, save_complete_tries="tokenizer/complete_tries.json")

and lets you browse every registered word/morpheme path (the nodes marked
with ``IDX``) one at a time.

The "Word:" panel is an INTERACTIVE TREE
----------------------------------------
* The chosen word is drawn as a straight horizontal spine of character
  nodes (the root-trie path of that word).
* **Every node is clickable.**  Clicking a node expands it: all of its
  immediate daughter nodes in the trie appear, hanging below.  Those
  daughters are clickable too, so the tree can be explored to any depth —
  well past the end of the word itself (revealing every continuation).
  Clicking an expanded node collapses it again.
* Each node reports its **branching factor** — the number of daughter
  nodes it has in the trie — on a small pill below it:
      - amber pill "1 single"  -> exactly one daughter (a deterministic,
        non-branching continuation); the edge to it is drawn thick amber.
      - blue  pill "n"         -> n >= 2 daughters (a real branch point).
      - faint "leaf"           -> no daughters.
  The "+" / "-" on the pill shows whether the node is collapsed/expanded.

Frequencies & pruning
---------------------
* Each node's raw ``##`` frequency is printed non-invasively below it.
* Node fill colour encodes its fate under the live ``min_frequency``
  slider: blue = kept, a red scale = pruned (deep red at freq 1, fading to
  pale pink at freq == min_freq), grey = no count.
* On the spine, morphological splits and the ``min_frequency`` cut are
  drawn exactly as before.

Usage
-----
    python morpiece_trie_explorer.py complete_tries.json
    python morpiece_trie_explorer.py            # opens a file picker

Pure-stdlib (tkinter only); no third-party dependencies.  The data-logic
functions at the top are import-safe and tkinter-free, so they can be
unit-tested headlessly.
"""

import argparse
import json
import os
import sys


# =============================================================================
# Data logic  —  no tkinter, safe to import / unit-test headlessly
# =============================================================================

# Keys that are structural metadata rather than real trie edges.
META_KEYS = {"##", "IDX"}
# Top-level keys of `roots` that are not ordinary letter subtries.
SKIP_TOP_KEYS = {"[RSX]", "++"}
# Everything that must never be treated as a letter-child edge.
STRUCT_KEYS = META_KEYS | SKIP_TOP_KEYS


def load_tries(path):
    """
    Load a complete-tries JSON file written by MorPiece.save_complete_tries.

    Returns a dict with keys: meta, roots, infls, suffix_stems.
    Accepts both the v1.4.3 wrapper ({"format","meta","roots",...}) and a
    bare trie dict (treated as `roots` with empty meta).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "roots" in data:
        roots = data["roots"]
        meta = data.get("meta", {}) or {}
        infls = data.get("infls", {}) or {}
        suffix_stems = data.get("suffix_stems", {}) or {}
    else:
        roots = data
        meta, infls, suffix_stems = {}, {}, {}

    suffix_stems = {k: set(v) for k, v in suffix_stems.items()}
    return {"meta": meta, "roots": roots, "infls": infls,
            "suffix_stems": suffix_stems}


def node_at_path(trie, path):
    """
    Walk `trie` along `path` (a string) and return the node dict reached,
    or None if the path cannot be resolved.
    """
    node = trie
    for ch in path:
        if isinstance(node, dict) and ch in node and isinstance(node[ch], dict):
            node = node[ch]
        else:
            return None
    return node


def child_chars(node):
    """Sorted list of real (letter) child edges of a trie node."""
    if not isinstance(node, dict):
        return []
    return sorted(k for k, v in node.items()
                  if k not in STRUCT_KEYS and isinstance(v, dict))


def collect_words(trie, skip_keys=SKIP_TOP_KEYS):
    """
    Walk a trie iteratively and return a sorted list of every word whose
    path terminates on a node carrying an 'IDX' marker.
    """
    words = []
    stack = []
    for k, v in trie.items():
        if k in skip_keys or k in META_KEYS:
            continue
        if isinstance(v, dict):
            stack.append((v, k))

    while stack:
        node, path = stack.pop()
        if "IDX" in node:
            words.append(path)
        for k, v in node.items():
            if k in META_KEYS:
                continue
            if isinstance(v, dict):
                stack.append((v, path + k))

    return sorted(set(words))


def path_info(trie, word):
    """
    Resolve `word` through `trie`; return a list of per-character records:
        char, freq (## count or None), has_idx, children (branching factor),
        broken (True once the path can no longer be resolved).
    """
    info = []
    node = trie
    broken = False
    for ch in word:
        if (not broken) and isinstance(node, dict) and ch in node \
                and isinstance(node[ch], dict):
            node = node[ch]
            info.append({
                "char": ch,
                "freq": node.get("##"),
                "has_idx": "IDX" in node,
                "children": len(child_chars(node)),
                "broken": False,
            })
        else:
            broken = True
            info.append({"char": ch, "freq": None, "has_idx": False,
                         "children": 0, "broken": True})
    return info


def splits_for_word(word, suffix_stems, min_suffix_stems):
    """
    Every stem|suffix decomposition of `word` that MorPiece proposed during
    training, sorted by split index.  Each dict: index, stem, suffix,
    n_stems, survives (n_stems >= min_suffix_stems).
    """
    splits = []
    for suffix, stems in suffix_stems.items():
        if not suffix or len(suffix) >= len(word):
            continue
        if not word.endswith(suffix):
            continue
        stem = word[: len(word) - len(suffix)]
        if stem not in stems:
            continue
        n = len(stems)
        splits.append({
            "index": len(stem), "stem": stem, "suffix": suffix,
            "n_stems": n, "survives": n >= min_suffix_stems,
        })
    splits.sort(key=lambda s: s["index"])
    return splits


def first_pruned_index(info, min_freq):
    """
    Index of the first node on the path that the min_frequency cut removes,
    or None if the whole path survives.
    """
    for i, rec in enumerate(info):
        f = rec["freq"]
        if f is None:
            return i
        if f <= min_freq:
            return i
    return None


def freq_color(freq, min_freq):
    """
    Map a node frequency to a hex fill colour:
        None              -> light grey  (#dddddd)
        > min_freq        -> calm blue   (#b9d4ec)  [kept]
        1..min_freq       -> red scale, deep red at 1, pale pink at min_freq
    """
    if freq is None:
        return "#dddddd"
    if freq > min_freq:
        return "#b9d4ec"

    deep = (176, 18, 24)
    pale = (255, 212, 212)
    if min_freq <= 1:
        t = 0.0
    else:
        t = (freq - 1) / (min_freq - 1)
        t = max(0.0, min(1.0, t))
    rgb = tuple(round(d + (p - d) * t) for d, p in zip(deep, pale))
    return "#%02x%02x%02x" % rgb


def plan_branches(trie, word, expanded):
    """
    Pure layout pass for the expandable part of the tree (tkinter-free, so
    it is unit-testable).

    `word`     : the spine word.
    `expanded` : set of node paths (strings) the user has opened.

    A branch root is any non-spine daughter of an *expanded* spine node;
    branch nodes recurse the same way.  Returns:
        positions : dict  branch_path -> (depth, row)
                    depth = len(path) - 1   (word[0] is depth 0)
                    row   = float vertical slot, 0 at the top branch row
        n_rows    : total number of leaf rows used
    Spine nodes are NOT included (their layout is fixed by the GUI).
    """
    positions = {}
    counter = [0]

    def layout(path):
        node = node_at_path(trie, path)
        depth = len(path) - 1
        vis = child_chars(node) if (node is not None and path in expanded) else []
        if not vis:
            row = counter[0]
            counter[0] += 1
        else:
            rows = [layout(path + c) for c in vis]
            row = sum(rows) / len(rows)
        positions[path] = (depth, row)
        return row

    n = len(word)
    for i in range(n):
        spath = word[: i + 1]
        node = node_at_path(trie, spath)
        if node is None or spath not in expanded:
            continue
        spine_child = word[i + 1] if i + 1 < n else None
        for c in child_chars(node):
            if c == spine_child:
                continue
            layout(spath + c)

    return positions, counter[0]


# =============================================================================
# GUI  —  imported lazily so the logic above stays headless-testable
# =============================================================================

def _build_gui():
    """Import tkinter and build the explorer class; returns a run() callable."""
    import tkinter as tk
    from tkinter import ttk, filedialog

    # ----- canvas / layout constants -----------------------------------------
    NODE_R = 23
    NODE_DX = 96
    DY_BRANCH = 96
    X0 = 70
    Y_SPINE = 156
    GOLD = "#d4a017"
    SEL_RING = "#2563eb"
    AMBER = "#e6b34a"
    AMBER_DK = "#b9842a"
    BLUE_PILL = "#bcd3e6"
    BLUE_PILL_DK = "#7da6c4"
    DISPLAY_CAP = 4000

    WS_GLYPH = {" ": "\u2423", "\n": "\u23ce", "\t": "\u21e5"}

    def glyph(ch):
        return WS_GLYPH.get(ch, ch)

    class TrieExplorer(tk.Tk):
        def __init__(self, bundle, source_path):
            super().__init__()
            self.title("MorPiece Trie Explorer")
            self.geometry("1240x760")
            self.minsize(960, 600)

            self.bundle = bundle
            self.roots = bundle["roots"]
            self.suffix_stems = bundle["suffix_stems"]
            self.meta = bundle["meta"]
            self.source_path = source_path

            self.min_suffix_stems = int(self.meta.get("min_suffix_stems", 3))
            default_minfreq = int(self.meta.get("min_frequency", 10))

            self._root_words = collect_words(self.roots, SKIP_TOP_KEYS)
            plus = self.roots.get("++", {})
            self._plus_words = collect_words(plus, ()) if isinstance(plus, dict) else []

            self.mode_var = tk.StringVar(value="roots")
            self.minfreq_var = tk.IntVar(value=max(1, default_minfreq))
            self.search_var = tk.StringVar(value="")

            self._all_words = []
            self._shown_words = []
            self.word = None              # current spine word
            self.expanded = set()         # node paths the user has opened
            self.selected = None          # last clicked node path
            self._nodes_xy = {}           # path -> (x, y) for hit-testing

            self._build_widgets()
            self._refresh_word_list()
            self.status.set(
                f"Loaded {os.path.basename(source_path)} — "
                f"{len(self._root_words)} root words, "
                f"{len(self._plus_words)} ++ suffixes.  "
                f"Pick a word, then click any node to expand its daughters."
            )

        # -- widget construction ----------------------------------------------
        def _build_widgets(self):
            top = ttk.Frame(self, padding=(10, 8))
            top.pack(side="top", fill="x")

            ttk.Label(top, text="View:").pack(side="left")
            ttk.Radiobutton(top, text="root words", value="roots",
                            variable=self.mode_var,
                            command=self._on_mode_change).pack(side="left")
            ttk.Radiobutton(top, text="++ suffixes", value="plus",
                            variable=self.mode_var,
                            command=self._on_mode_change).pack(side="left",
                                                               padx=(0, 16))

            ttk.Label(top, text="min_frequency:").pack(side="left")
            self.scale = tk.Scale(top, from_=1, to=60, orient="horizontal",
                                  variable=self.minfreq_var, length=230,
                                  command=lambda _e: self._on_minfreq_change())
            self.scale.pack(side="left", padx=(2, 12))

            ttk.Button(top, text="Collapse all",
                       command=self._collapse_all).pack(side="left", padx=(0, 12))

            meta = self.meta
            ttk.Label(
                top,
                text=(f"snapshot v{meta.get('version', '?')}   "
                      f"min_suffix_stems={self.min_suffix_stems}   "
                      f"type_based={meta.get('type_based', '?')}"),
                foreground="#666",
            ).pack(side="left")

            body = ttk.Frame(self)
            body.pack(side="top", fill="both", expand=True)

            # left: word list
            left = ttk.Frame(body, padding=(10, 6))
            left.pack(side="left", fill="y")
            ttk.Label(left, text="Filter:").pack(anchor="w")
            ttk.Entry(left, textvariable=self.search_var, width=26).pack(
                anchor="w", pady=(0, 4))
            self.search_var.trace_add("write",
                                      lambda *_: self._refresh_word_list())
            lf = ttk.Frame(left)
            lf.pack(fill="both", expand=True)
            self.listbox = tk.Listbox(lf, width=28, activestyle="none",
                                      font=("TkFixedFont", 11))
            sb = ttk.Scrollbar(lf, orient="vertical",
                               command=self.listbox.yview)
            self.listbox.configure(yscrollcommand=sb.set)
            self.listbox.pack(side="left", fill="both", expand=True)
            sb.pack(side="left", fill="y")
            self.listbox.bind("<<ListboxSelect>>", self._on_select_word)
            self.count_label = ttk.Label(left, text="", foreground="#666")
            self.count_label.pack(anchor="w", pady=(4, 0))

            # right: canvas + legend + detail
            right = ttk.Frame(body, padding=(6, 6))
            right.pack(side="left", fill="both", expand=True)

            cw = ttk.Frame(right)
            cw.pack(side="top", fill="both", expand=True)
            self.canvas = tk.Canvas(cw, bg="white", highlightthickness=0)
            hsb = ttk.Scrollbar(cw, orient="horizontal",
                                command=self.canvas.xview)
            vsb = ttk.Scrollbar(cw, orient="vertical",
                                command=self.canvas.yview)
            self.canvas.configure(xscrollcommand=hsb.set,
                                  yscrollcommand=vsb.set)
            self.canvas.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            cw.rowconfigure(0, weight=1)
            cw.columnconfigure(0, weight=1)
            self.canvas.bind("<Button-1>", self._on_canvas_click)

            self._build_legend(right)

            self.detail = tk.Text(right, height=11, wrap="word",
                                  font=("TkFixedFont", 10), background="#fafafa")
            self.detail.pack(side="top", fill="x", pady=(6, 0))
            self.detail.configure(state="disabled")

            self.status = tk.StringVar(value="")
            ttk.Label(self, textvariable=self.status, relief="sunken",
                      anchor="w", foreground="#444").pack(side="bottom",
                                                          fill="x")

        def _build_legend(self, parent):
            lg = tk.Canvas(parent, height=36, bg="white", highlightthickness=0)
            lg.pack(side="top", fill="x", pady=(4, 0))
            x = 8

            def swatch(fill, outline, label):
                nonlocal x
                lg.create_rectangle(x, 10, x + 18, 26, fill=fill,
                                    outline=outline)
                lg.create_text(x + 23, 18, anchor="w", text=label,
                               font=("TkDefaultFont", 8), fill="#555")
                x += 23 + len(label) * 5.6 + 16

            swatch("#b9d4ec", "#7da6c4", "freq > min_freq (kept)")
            swatch("#b01218", "#7a0c10", "pruned (freq -> 1)")
            swatch(AMBER, AMBER_DK, "1 daughter (single / deterministic)")
            swatch(BLUE_PILL, BLUE_PILL_DK, "n daughters (branch point)")
            lg.create_oval(x, 9, x + 18, 27, fill="white", outline=GOLD,
                           width=3)
            lg.create_text(x + 23, 18, anchor="w", text="IDX",
                           font=("TkDefaultFont", 8), fill="#555")

        # -- event handlers ---------------------------------------------------
        def _on_mode_change(self):
            self.search_var.set("")
            self.word = None
            self.expanded.clear()
            self.selected = None
            self._refresh_word_list()
            self.canvas.delete("all")
            self._set_detail("")

        def _on_minfreq_change(self):
            if self.word is not None:
                self._draw()

        def _collapse_all(self):
            self.expanded.clear()
            self.selected = None
            if self.word is not None:
                self._draw()

        def _refresh_word_list(self):
            self._all_words = (self._root_words
                               if self.mode_var.get() == "roots"
                               else self._plus_words)
            q = self.search_var.get().strip().lower()
            shown = ([w for w in self._all_words if q in w.lower()]
                     if q else list(self._all_words))
            self._shown_words = shown[:DISPLAY_CAP]
            self.listbox.delete(0, "end")
            for w in self._shown_words:
                self.listbox.insert("end", repr(w) if w.strip() != w else w)
            total = len(self._all_words)
            if len(shown) > DISPLAY_CAP:
                self.count_label.config(
                    text=f"{len(shown)} matches (first {DISPLAY_CAP} shown) "
                         f"/ {total}")
            else:
                self.count_label.config(text=f"{len(shown)} shown / {total}")

        def _on_select_word(self, _event):
            sel = self.listbox.curselection()
            if not sel:
                return
            self.word = self._shown_words[sel[0]]
            self.expanded.clear()
            self.selected = None
            self._draw()

        def _on_canvas_click(self, event):
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            hit = None
            for path, (nx, ny) in self._nodes_xy.items():
                if (x - nx) ** 2 + (y - ny) ** 2 <= (NODE_R + 3) ** 2:
                    hit = path
                    break
            if hit is None:
                return
            self.selected = hit
            node = self._active_node(hit)
            n_children = len(child_chars(node)) if node is not None else 0
            if n_children > 0:
                if hit in self.expanded:
                    self.expanded.discard(hit)
                else:
                    self.expanded.add(hit)
            self._draw()

        # -- trie access ------------------------------------------------------
        def _active_trie(self):
            if self.mode_var.get() == "roots":
                return self.roots
            return self.roots.get("++", {})

        def _active_node(self, path):
            return node_at_path(self._active_trie(), path)

        # -- core drawing -----------------------------------------------------
        def _draw(self):
            c = self.canvas
            c.delete("all")
            self._nodes_xy = {}
            word = self.word
            if not word:
                return
            min_freq = max(1, int(self.minfreq_var.get()))
            trie = self._active_trie()

            info = path_info(trie, word)
            n = len(info)
            splits = (splits_for_word(word, self.suffix_stems,
                                      self.min_suffix_stems)
                      if self.mode_var.get() == "roots" else [])
            cut = first_pruned_index(info, min_freq)

            branch_pos, n_rows = plan_branches(trie, word, self.expanded)

            # canvas extent
            max_depth = n - 1
            for (depth, _row) in branch_pos.values():
                max_depth = max(max_depth, depth)
            width = X0 * 2 + max_depth * NODE_DX
            height = Y_SPINE + DY_BRANCH * (n_rows + 1) + 70
            c.configure(scrollregion=(0, 0, max(width, 640),
                                      max(height, 380)))

            def spine_xy(i):
                return X0 + i * NODE_DX, Y_SPINE

            def branch_xy(path):
                depth, row = branch_pos[path]
                return X0 + depth * NODE_DX, Y_SPINE + DY_BRANCH * (row + 1)

            # ---- spine background bits (pruned wash, splits, cut) -----------
            if cut is not None:
                x0 = spine_xy(cut)[0] - NODE_R - NODE_DX * 0.30
                x1 = spine_xy(n - 1)[0] + NODE_R + 16
                c.create_rectangle(x0, Y_SPINE - NODE_R - 48,
                                   x1, Y_SPINE + NODE_R + 22,
                                   fill="#fdeced", outline="")
                c.create_text(x1 - 6, Y_SPINE - NODE_R - 40, anchor="ne",
                              text="pruned by min_frequency", fill="#b01218",
                              font=("TkDefaultFont", 8, "italic"))

            for depth, sp in enumerate(splits):
                idx = sp["index"]
                if idx < 1 or idx > n:
                    continue
                if idx < n:
                    xline = (spine_xy(idx - 1)[0] + spine_xy(idx)[0]) / 2
                else:
                    xline = spine_xy(n - 1)[0] + NODE_DX / 2
                col = "#1b9e3e" if sp["survives"] else "#e07b1a"
                c.create_line(xline, Y_SPINE - NODE_R - 30 - depth * 18,
                              xline, Y_SPINE + NODE_R + 12,
                              fill=col, width=2, dash=(5, 3))
                verdict = "++kept" if sp["survives"] else "++pruned"
                c.create_text(xline + 5, Y_SPINE - NODE_R - 32 - depth * 18,
                              anchor="w",
                              text=(f"{sp['stem']} | ++{sp['suffix']}  "
                                    f"[{sp['n_stems']} stems, {verdict}]"),
                              fill=col, font=("TkDefaultFont", 8))

            if cut is not None:
                xc = spine_xy(cut)[0] - NODE_R - NODE_DX * 0.30
                c.create_line(xc, Y_SPINE - NODE_R - 10,
                              xc, Y_SPINE + NODE_R + 10,
                              fill="#b01218", width=2, dash=(3, 2))
                c.create_text(xc, Y_SPINE - NODE_R - 14, anchor="s",
                              text=f"\u2702 min_freq = {min_freq}",
                              fill="#b01218",
                              font=("TkDefaultFont", 8, "bold"))

            # ---- edges (drawn before nodes) ---------------------------------
            # spine edges
            for i in range(n - 1):
                x0, y0 = spine_xy(i)
                x1, y1 = spine_xy(i + 1)
                self._edge(c, x0, y0, x1, y1, info[i]["children"])
            # branch edges (parent -> child)
            for path in branch_pos:
                parent = path[:-1]
                cx, cy = branch_xy(path)
                if len(parent) >= 1 and parent == word[:len(parent)] \
                        and len(parent) <= n:
                    px, py = spine_xy(len(parent) - 1)
                    pkids = info[len(parent) - 1]["children"]
                elif parent in branch_pos:
                    px, py = branch_xy(parent)
                    pnode = node_at_path(trie, parent)
                    pkids = len(child_chars(pnode))
                else:
                    continue
                self._edge(c, px, py, cx, cy, pkids)

            # ---- nodes ------------------------------------------------------
            # spine
            for i, rec in enumerate(info):
                x, y = spine_xy(i)
                self._node(c, x, y, rec["char"], rec["freq"], rec["has_idx"],
                           rec["children"], word[: i + 1], min_freq,
                           rec["broken"], i)
            # branches
            for path in branch_pos:
                node = node_at_path(trie, path)
                x, y = branch_xy(path)
                freq = node.get("##") if isinstance(node, dict) else None
                has_idx = isinstance(node, dict) and "IDX" in node
                kids = len(child_chars(node))
                self._node(c, x, y, path[-1], freq, has_idx, kids, path,
                           min_freq, node is None, len(path) - 1)

            # ---- title ------------------------------------------------------
            disp = repr(word) if word.strip() != word else word
            c.create_text(X0 - NODE_R, 28, anchor="w",
                          text=f"Word:  {disp}",
                          font=("TkDefaultFont", 13, "bold"), fill="#222")
            c.create_text(X0 - NODE_R, 48, anchor="w",
                          text="click any node to expand / collapse its "
                               "daughters",
                          font=("TkDefaultFont", 8, "italic"), fill="#999")

            self._describe(word, info, splits, cut, min_freq)
            kept = ("fully retained" if cut is None
                    else f"cut at depth {cut}")
            sel = f"   |   selected: {self.selected!r}" if self.selected else ""
            self.status.set(f"{disp}: {n} spine nodes, "
                            f"{len(branch_pos)} expanded daughter(s), "
                            f"min_freq={min_freq} \u2192 {kept}{sel}")

        def _edge(self, c, x0, y0, x1, y1, parent_kids):
            """Connector; amber+thick when the parent has a single daughter."""
            if parent_kids == 1:
                c.create_line(x0, y0, x1, y1, fill=AMBER_DK, width=3)
            else:
                c.create_line(x0, y0, x1, y1, fill="#c2c2c2", width=2)

        def _node(self, c, x, y, char, freq, has_idx, n_children, path,
                  min_freq, broken, depth):
            """Draw one trie node with its frequency and branching pill."""
            self._nodes_xy[path] = (x, y)

            # selection ring
            if path == self.selected:
                c.create_oval(x - NODE_R - 5, y - NODE_R - 5,
                              x + NODE_R + 5, y + NODE_R + 5,
                              outline=SEL_RING, width=3)

            fill = freq_color(freq, min_freq)
            outline = GOLD if has_idx else ("#bbbbbb" if broken else "#888888")
            ow = 4 if has_idx else 1
            c.create_oval(x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R,
                          fill=fill, outline=outline, width=ow)
            c.create_text(x, y, text=glyph(char),
                          font=("TkDefaultFont", 15, "bold"), fill="#222")
            # depth index above
            c.create_text(x, y - NODE_R - 11, text=str(depth),
                          font=("TkDefaultFont", 7), fill="#bcbcbc")
            # frequency below (non-invasive)
            ftext = "\u2014" if freq is None else str(freq)
            c.create_text(x, y + NODE_R + 11, text=ftext,
                          font=("TkDefaultFont", 9), fill="#777777")

            # branching pill
            by = y + NODE_R + 28
            if broken:
                return
            if n_children == 0:
                c.create_text(x, by, text="leaf",
                              font=("TkDefaultFont", 8, "italic"),
                              fill="#bbbbbb")
                return
            opened = path in self.expanded
            sign = "\u2212" if opened else "+"
            if n_children == 1:
                pill, pdk, label = AMBER, AMBER_DK, f"{sign} 1 single"
            else:
                pill, pdk, label = (BLUE_PILL, BLUE_PILL_DK,
                                    f"{sign} {n_children}")
            half = 5 + len(label) * 3.7
            c.create_rectangle(x - half, by - 9, x + half, by + 9,
                               fill=pill, outline=pdk)
            c.create_text(x, by, text=label,
                          font=("TkDefaultFont", 8, "bold"), fill="#2a2a2a")

        # -- detail panel -----------------------------------------------------
        def _set_detail(self, text):
            self.detail.configure(state="normal")
            self.detail.delete("1.0", "end")
            self.detail.insert("1.0", text)
            self.detail.configure(state="disabled")

        def _describe(self, word, info, splits, cut, min_freq):
            lines = []
            disp = repr(word) if word.strip() != word else word
            lines.append(f"WORD  {disp}   ({len(word)} chars)")
            if cut is None:
                lines.append(f"min_frequency = {min_freq}: whole spine "
                             f"survives.")
            else:
                lines.append(f"min_frequency = {min_freq}: cut at depth "
                              f"{cut}; longest surviving prefix "
                              f"{word[:cut]!r}.")
            if splits:
                lines.append("splits: " + "; ".join(
                    f"{s['stem']}|++{s['suffix']} "
                    f"({'kept' if s['survives'] else 'pruned'})"
                    for s in splits))

            lines.append("")
            trie = self._active_trie()
            if self.selected is None:
                lines.append("Click a node to inspect its daughters here.")
            else:
                sp = self.selected
                node = node_at_path(trie, sp)
                lines.append(f"NODE  {sp!r}   depth {len(sp) - 1}")
                if node is None:
                    lines.append("  (this character is not in the trie — "
                                  "broken path, no daughters)")
                else:
                    freq = node.get("##")
                    kids = child_chars(node)
                    lines.append(
                        f"  freq(##)={freq}   "
                        f"IDX={'yes' if 'IDX' in node else 'no'}   "
                        f"branching factor = {len(kids)}")
                    if len(kids) == 0:
                        lines.append("  leaf node — no daughters.")
                    elif len(kids) == 1:
                        ch = kids[0]
                        kn = node[ch]
                        lines.append(
                            f"  SINGLE daughter (deterministic continuation):")
                        lines.append(
                            f"    {(sp + ch)!r}  freq={kn.get('##')}  "
                            f"IDX={'yes' if 'IDX' in kn else 'no'}  "
                            f"children={len(child_chars(kn))}")
                    else:
                        lines.append(f"  {len(kids)} daughters:")
                        for ch in kids:
                            kn = node[ch]
                            lines.append(
                                f"    {(sp + ch)!r:14}  freq={kn.get('##')}  "
                                f"IDX={'yes' if 'IDX' in kn else 'no'}  "
                                f"children={len(child_chars(kn))}")
                    op = "expanded" if sp in self.expanded else "collapsed"
                    lines.append(f"  ({op} — click the node to toggle)")
            self._set_detail("\n".join(lines))

    def run(bundle, source_path):
        app = TrieExplorer(bundle, source_path)
        app.mainloop()

    return run


# =============================================================================
# Entry point
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Interactive explorer for MorPiece complete-trie "
                    "snapshots (save_complete_tries output).")
    parser.add_argument("json_file", nargs="?", default=None,
                        help="complete_tries.json written by MorPiece "
                             ">=1.4.3. If omitted, a file picker opens.")
    args = parser.parse_args(argv)

    path = args.json_file
    if path is None:
        import tkinter as tk
        from tkinter import filedialog
        picker_root = tk.Tk()
        picker_root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a MorPiece complete_tries.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        picker_root.destroy()
        if not path:
            print("No file selected — exiting.")
            return 1

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    bundle = load_tries(path)
    if not bundle["roots"]:
        print("The file contains no 'roots' trie — nothing to explore.",
              file=sys.stderr)
        return 1

    run = _build_gui()
    run(bundle, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
