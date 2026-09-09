#!/usr/bin/env python3
"""
rocking_hemisphere.py — the rocking-hemisphere setup as a live animation.

Replaces the static SVG figure in articles/rotational-mechanics-higher-
dimensions.html.  A uniform solid hemisphere rests flat-face up on a table,
balancing on its dome, and rocks without slipping.  The animation makes three
facts from the article visible in motion:

  * the sphere's centre O rides along a perfectly LEVEL line — it stays a
    height R above the table for every tilt, because the contact point is
    always the point of the dome directly below O;
  * the centre of mass, a distance b = 3R/8 below O, swings on a shallow arc
    and sits LOWEST when the hemisphere is upright — that dip is the potential
    well that makes upright the stable balance point;
  * the weight Mg hangs straight down from the centre of mass, and the tilt
    angle theta opens at O between the vertical and the body's axis.

Rolling without slipping: turning the body by theta rolls its centre O a
horizontal distance R*theta, so O = (R*theta, R) and the contact point is
(R*theta, 0).  The centre of mass is O + b*(-sin theta, -cos theta).

Render (matches the site's dark canvas, 16:9):

    manim -qh --format mp4 scripts/rocking_hemisphere.py RockingHemisphere

Requires: manim (Community Edition) + ffmpeg.  No LaTeX — labels are Text so
the scene renders on a bare install.
"""

from functools import partial

import numpy as np
from manim import (
    Scene, VMobject, VGroup, Line, DashedLine, Arc, Dot, Text, Arrow,
    ValueTracker, config,
    ORIGIN, UP, DOWN, LEFT, RIGHT, PI,
)

FONT = "Helvetica Neue"          # a clean sans the site would use; even kerning
Label = partial(Text, font=FONT)

# --- site tokens (assets/css/site.css, dark theme) -----------------------
BG          = "#0d0c0a"   # --canvas-bg
FG          = "#e8e3d8"   # --fg   (table, primary ink)
FG_SOFT     = "#b7b1a3"   # --fg-soft (labels)
FG_FAINT    = "#7c766a"   # --fg-faint (guides)
ACCENT      = "#8fb4ff"   # --accent (dome outline)
ACCENT_SOFT = "#263452"   # --accent-soft (dome fill)
CM_COLOR    = "#e0563a"   # the warm centre-of-mass dot (hsl(8,74%,55%))

# --- geometry (manim units) ----------------------------------------------
R       = 2.4                 # sphere radius
B       = 3.0 / 8.0 * R       # centre of mass below O:  3R/8 = 0.9
Y_TABLE = -1.4                # table height (keeps the figure vertically centred)
O_Y     = Y_TABLE + R         # O rides on this level line

# --- motion ---------------------------------------------------------------
AMP      = 24 * PI / 180.0    # rock amplitude (radians)
PERIOD   = 2.5                # seconds per full back-and-forth
N_LOOPS  = 2                  # whole periods -> seamless loop
DURATION = PERIOD * N_LOOPS

config.background_color = BG


def half_disk(radius):
    """A filled half-disk: lower semicircle (the dome) closed by its diameter
    (the flat face).  Centre of the sphere sits at the origin."""
    arc = Arc(radius=radius, start_angle=PI, angle=PI, arc_center=ORIGIN)
    dome = VMobject()
    dome.set_points(arc.get_points())
    dome.add_line_to(arc.get_start())          # close along the flat diameter
    dome.set_fill(ACCENT_SOFT, opacity=0.92)
    dome.set_stroke(ACCENT, width=3.5)
    return dome


def theta_of(t):
    return AMP * np.sin(2 * PI * t / PERIOD)


class RockingHemisphere(Scene):
    def construct(self):
        t = ValueTracker(0.0)

        # No-slip: a CCW body turn of th rolls the centre O the *other* way,
        # O_x = -R*th (the material point at the contact has zero velocity).
        def O_pos(th):
            return np.array([-R * th, O_Y, 0.0])

        def contact_pos(th):
            return np.array([-R * th, Y_TABLE, 0.0])

        # CM sits a distance b down the body's symmetry axis; a CCW turn th
        # rotates the downward pole (0,-1) to (sin th, -cos th) -- toward the
        # dome's bulge.  Gravity through it then restores toward upright, which
        # is exactly why the rock is stable.
        def cm_pos(th):
            o = O_pos(th)
            return o + B * np.array([np.sin(th), -np.cos(th), 0.0])

        # --- static backdrop -------------------------------------------------
        table = Line([-6.4, Y_TABLE, 0], [6.4, Y_TABLE, 0],
                     color=FG, stroke_width=3)
        self.add(table)

        # --- moving body -----------------------------------------------------
        base = half_disk(R)                    # upright, O at origin
        dome = base.copy()

        O_dot = Dot(radius=0.055, color=FG)
        cm_dot = Dot(radius=0.075, color=CM_COLOR).set_stroke(BG, width=2.5)
        contact_dot = Dot(radius=0.05, color=FG)

        axis = Line(ORIGIN, ORIGIN, color=FG_FAINT, stroke_width=1.5)   # O->CM
        drop = DashedLine(ORIGIN, ORIGIN, color=FG_FAINT, stroke_width=1.5,
                          dash_length=0.1)                              # O->contact
        mg = Arrow(ORIGIN, ORIGIN, color=FG, stroke_width=3,
                   buff=0, max_tip_length_to_length_ratio=0.35)

        # only the labels that carry the physics: O, the 3R/8 offset, and Mg
        O_lbl  = Label("O", font_size=24, color=FG)
        b_lbl  = Label("3R/8", font_size=20, color=FG_SOFT)
        mg_lbl = Label("Mg", font_size=22, color=FG)

        moving = VGroup(dome, O_dot, cm_dot, contact_dot, axis, drop, mg,
                        O_lbl, b_lbl, mg_lbl)

        def relayout(_m):
            th = theta_of(t.get_value())
            o = O_pos(th)
            cm = cm_pos(th)
            ct = contact_pos(th)

            dome.become(base.copy().rotate(th, about_point=ORIGIN).shift(o))
            O_dot.move_to(o)
            cm_dot.move_to(cm)
            contact_dot.move_to(ct)

            axis.put_start_and_end_on(o, cm)
            drop.put_start_and_end_on(o, ct)
            mg.put_start_and_end_on(cm, cm + DOWN * 1.25)

            O_lbl.next_to(o, UP, buff=0.1)
            # "3R/8" rides the axis toward the CM, pushed to the outer side so
            # it stays clear of the vertical plumb line as the body leans.
            side = 1.0 if th >= 0 else -1.0
            b_lbl.move_to(o + 0.6 * (cm - o) + np.array([0.42 * side, 0.0, 0]))
            mg_lbl.next_to(mg.get_end(), RIGHT, buff=0.12)

        relayout(None)
        moving.add_updater(relayout)
        self.add(moving)

        self.play(t.animate.set_value(DURATION), run_time=DURATION,
                  rate_func=lambda x: x)
        moving.remove_updater(relayout)
