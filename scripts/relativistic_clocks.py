#!/usr/bin/env python3
"""
relativistic_clocks.py — the companion animation for the "Relativistic Clocks"
article.  Four beats, one continuous scene, in the site's dark palette:

  1. Three clocks tick at different rates — a GPS satellite really gains
     +38 microseconds a day (the honest physics).
  2. A causal chain (send -> receive) between a fast-clock node and a slow-clock
     node: sorting by wall-clock timestamp puts the two events in the WRONG
     order; a vector clock gets it right.
  3. A replicated key-value store: Last-Write-Wins compares the dilated
     timestamps and silently DROPS the newer write; vector / hybrid logical
     clocks keep it.
  4. Takeaway card: same physics, different ruler.

Matches assets/css/site.css (dark theme) and the house style of
rocking_hemisphere.py: no LaTeX (labels are Text, so it renders on a bare
install), 16:9, silent.  The article carries the words.

Render (low-q while iterating, then high-q for the site):

    manim -ql --format mp4 scripts/relativistic_clocks.py RelativisticClocks
    manim -qh --format mp4 scripts/relativistic_clocks.py RelativisticClocks

Requires: manim (Community Edition) + ffmpeg.
"""

from functools import partial

import numpy as np
from manim import (
    Scene, VGroup, VMobject, Text, Line, DashedLine, Arrow, Dot,
    Rectangle, RoundedRectangle, ValueTracker, always_redraw, config,
    FadeIn, FadeOut, Write, Create, Transform, Indicate, Flash,
    ORIGIN, UP, DOWN, LEFT, RIGHT, PI,
)

FONT = "Helvetica Neue"
Label = partial(Text, font=FONT)

# --- site tokens (assets/css/site.css, dark theme) -----------------------
BG          = "#0d0c0a"   # --canvas-bg
FG          = "#e8e3d8"   # --fg
FG_SOFT     = "#b7b1a3"   # --fg-soft
FG_FAINT    = "#7c766a"   # --fg-faint
ACCENT      = "#8fb4ff"   # --accent
ACCENT_SOFT = "#263452"   # --accent-soft
LOSS        = "#e0563a"   # warm red: dropped write / wrong order
OK          = "#83c167"   # green: correct

config.background_color = BG


class RelativisticClocks(Scene):
    def construct(self):
        self.beat_title()
        self.beat_three_clocks()
        self.beat_ordering()
        self.beat_kvstore()
        self.beat_takeaway()

    # ----------------------------------------------------------------- title
    def beat_title(self):
        title = Label("Relativistic Clocks", font_size=58, color=FG)
        sub = Label("when time dilation breaks a distributed system",
                    font_size=26, color=FG_SOFT)
        sub.next_to(title, DOWN, buff=0.35)
        g = VGroup(title, sub).move_to(ORIGIN)
        self.play(Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=0.2 * UP), run_time=1.0)
        self.wait(1.2)
        self.play(FadeOut(g), run_time=0.8)

    # --------------------------------------------------- beat 1: three clocks
    def beat_three_clocks(self):
        head = Label("Every clock ticks at its own rate", font_size=34, color=FG)
        head.to_edge(UP, buff=0.7)
        self.play(Write(head), run_time=1.0)

        t = ValueTracker(0.0)
        specs = [
            ("GROUND",    1.00, FG_SOFT),
            ("SATELLITE", 1.35, ACCENT),   # exaggerated: higher, weaker gravity -> fast
            ("FAST SHIP", 0.55, LOSS),     # exaggerated: near light-speed -> slow
        ]
        tags = VGroup(*[Label(n, font_size=26, color=FG_FAINT) for n, _, _ in specs])
        tags.arrange(DOWN, buff=0.85, aligned_edge=LEFT).move_to(0.2 * DOWN + 1.4 * LEFT)

        readouts = VGroup()
        for (name, rate, col), tag in zip(specs, tags):
            # always_redraw re-anchors next to a FIXED tag every frame, so the
            # readouts count up in place instead of collapsing to the origin.
            r = always_redraw(
                lambda tag=tag, rate=rate, col=col:
                Label(f"{rate * t.get_value():5.2f} s", font_size=32, color=col)
                .next_to(tag, RIGHT, buff=1.5))
            readouts.add(r)

        self.play(LaggedFadeIn(tags))
        self.add(readouts)
        self.play(t.animate.set_value(6.0), run_time=3.4, rate_func=lambda x: x)
        for r in readouts:
            r.clear_updaters()

        note = Label("A real GPS satellite gains  +38 microseconds / day",
                     font_size=24, color=ACCENT)
        note.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(note, shift=0.2 * UP), run_time=1.0)
        self.wait(1.3)
        self.play(FadeOut(VGroup(head, tags, readouts, note)), run_time=0.8)

    # ------------------------------------------------ beat 2: ordering breaks
    def beat_ordering(self):
        head = Label("Which event happened first?", font_size=34, color=FG)
        head.to_edge(UP, buff=0.7)
        self.play(Write(head), run_time=1.0)

        # two worldlines
        y_top, y_bot = 2.4, -2.4
        fast_x, slow_x = -3.6, 3.6
        fast_line = Line([fast_x, y_bot, 0], [fast_x, y_top, 0], color=LOSS, stroke_width=3)
        slow_line = Line([slow_x, y_bot, 0], [slow_x, y_top, 0], color=ACCENT, stroke_width=3)
        fast_tag = Label("FAST clock", font_size=22, color=LOSS).next_to(fast_line, UP, buff=0.2)
        slow_tag = Label("SLOW clock", font_size=22, color=ACCENT).next_to(slow_line, UP, buff=0.2)
        self.play(Create(fast_line), Create(slow_line),
                  FadeIn(fast_tag), FadeIn(slow_tag), run_time=1.0)

        # the send event (high up on FAST, big wall time) and receive (low on SLOW)
        send_pt = np.array([fast_x, 1.2, 0])
        recv_pt = np.array([slow_x, -1.0, 0])
        send_dot = Dot(send_pt, radius=0.09, color=FG)
        recv_dot = Dot(recv_pt, radius=0.09, color=FG)
        msg = Arrow(send_pt, recv_pt, buff=0.12, color=FG, stroke_width=4,
                    max_tip_length_to_length_ratio=0.12)
        send_lbl = Label("send  ·  wall 3.0", font_size=22, color=FG).next_to(send_dot, LEFT, buff=0.25)
        recv_lbl = Label("receive  ·  wall 1.5", font_size=22, color=FG).next_to(recv_dot, RIGHT, buff=0.25)

        self.play(FadeIn(send_dot), Write(send_lbl), run_time=0.8)
        self.play(Create(msg), run_time=1.0)
        self.play(FadeIn(recv_dot), Write(recv_lbl), run_time=0.8)
        self.wait(0.4)

        wrong = Label("sort by timestamp:  send is AFTER receive   ✗",
                      font_size=26, color=LOSS).to_edge(DOWN, buff=1.3)
        right = Label("vector clock:  send is BEFORE receive   ✓",
                      font_size=26, color=OK).next_to(wrong, DOWN, buff=0.35)
        self.play(FadeIn(wrong, shift=0.2 * UP), run_time=1.0)
        self.play(Indicate(wrong, color=LOSS, scale_factor=1.06), run_time=0.9)
        self.wait(0.5)
        self.play(FadeIn(right, shift=0.2 * UP), run_time=1.0)
        self.wait(1.3)
        self.play(FadeOut(VGroup(head, fast_line, slow_line, fast_tag, slow_tag,
                                 send_dot, recv_dot, msg, send_lbl, recv_lbl,
                                 wrong, right)), run_time=0.8)

    # -------------------------------------------------- beat 3: data loss
    def beat_kvstore(self):
        head = Label("Last-Write-Wins loses data", font_size=34, color=FG)
        head.to_edge(UP, buff=0.7)
        self.play(Write(head), run_time=1.0)

        def write_card(title, value, ts, col):
            box = RoundedRectangle(corner_radius=0.12, width=4.2, height=1.5,
                                   stroke_color=col, fill_color=ACCENT_SOFT, fill_opacity=0.25)
            t1 = Label(title, font_size=22, color=col)
            t2 = Label(f'x = "{value}"', font_size=26, color=FG)
            t3 = Label(f"wall {ts}", font_size=20, color=FG_FAINT)
            txt = VGroup(t1, t2, t3).arrange(DOWN, buff=0.14)
            return VGroup(box, txt)

        w1 = write_card("W1  (fast clock)", "v1", "3.0", FG_SOFT)
        w2 = write_card("W2  (slow clock, newer)", "v2", "1.5", ACCENT)
        cards = VGroup(w1, w2).arrange(RIGHT, buff=1.6).move_to(0.6 * UP)
        cause = Label("W2 happened AFTER W1  (it read v1 first)", font_size=22, color=FG_SOFT)
        cause.next_to(cards, DOWN, buff=0.5)
        self.play(FadeIn(w1, shift=0.2 * UP), run_time=0.8)
        self.play(FadeIn(w2, shift=0.2 * UP), run_time=0.8)
        self.play(Write(cause), run_time=1.0)
        self.wait(0.6)

        verdict = Label("bigger timestamp wins  →  keeps v1", font_size=26, color=LOSS)
        verdict.to_edge(DOWN, buff=1.2)
        self.play(FadeIn(verdict), run_time=0.9)
        # W2 (the correct, newer write) is silently dropped
        self.play(Indicate(w2, color=LOSS, scale_factor=1.05), run_time=0.7)
        self.play(w2.animate.set_opacity(0.12), run_time=0.9)
        lost = Label("update lost — no error, no log", font_size=22, color=LOSS)
        lost.next_to(verdict, DOWN, buff=0.3)
        self.play(FadeIn(lost), run_time=0.8)
        self.wait(1.4)
        self.play(FadeOut(VGroup(head, cards, cause, verdict, lost)), run_time=0.8)

    # ------------------------------------------------------- beat 4: takeaway
    def beat_takeaway(self):
        head = Label("Same physics.  Different ruler.", font_size=40, color=FG)
        head.move_to(1.7 * UP)
        self.play(Write(head), run_time=1.4)

        rows = [
            ("wall clock (timestamp)", "loses data", LOSS, "✗"),
            ("Lamport clock", "keeps order", OK, "✓"),
            ("vector clock", "keeps order + finds concurrency", OK, "✓"),
            ("hybrid logical clock", "keeps order, stays readable", OK, "✓"),
        ]
        table = VGroup()
        for name, verdict, col, mark in rows:
            m = Label(mark, font_size=26, color=col)
            n = Label(name, font_size=24, color=FG)
            v = Label(verdict, font_size=22, color=FG_SOFT)
            table.add(VGroup(m, n, v).arrange(RIGHT, buff=0.4))
        table.arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(0.9 * DOWN)
        self.play(LaggedFadeIn(table))
        self.wait(2.0)
        self.play(FadeOut(VGroup(head, table)), run_time=1.0)


def LaggedFadeIn(group, lag=0.15):
    """Small helper: stagger a FadeIn over a VGroup's children."""
    from manim import LaggedStart
    return LaggedStart(*[FadeIn(m, shift=0.15 * UP) for m in group],
                       lag_ratio=lag, run_time=1.4)
