"""THE GOAL IS ON THE BOARD -- the relation sense, on synthetic boards.

Pinned here (after adversarial review 2026-09-05):
  * a same-shape static/dynamic pair is discovered, its mismatch is the
    value, progress fires exactly when it falls and regress when it rises
  * a fragment of the target that is also a pair does not outrank the
    whole picture
  * a blob painted IN PLACE (a canvas) is never "the object he controls"
    -- his older self organ called the cd82 canvas him
  * the controlled object is the class that TRANSLATES; a maze cell he
    passes through restores and is never a canvas; paint under him stays
    and IS a canvas once he moves off
  * the picture inside a two-cell frame is the content box
  * a marker near the controlled object: progress when he approaches;
    a marker that moves is dropped; a full-width line is never a marker
  * a relation that stops improving is set aside; a confirmed one never
  * a clear confirms only a candidate that CAME NEARER its goal this
    life (a constant marker never wins) and never revokes anything; a
    wrong confirmation is revoked by sitting satisfied without a clear
  * persistence round trip, including the controlled object
"""
import unittest
import numpy as np

from seagi.world import relsense
from seagi.world.relsense import RelSense, segment, STATIC_AFTER, CTRL_AFTER

W = 64
BG = 5


def board():
    return np.full((W, W), BG, dtype=np.int16)


def paint(G, r0, c0, r1, c1, col):
    G[r0:r1 + 1, c0:c1 + 1] = col


def picture(G, r0, c0):
    """a 10x10 target: colour 15 upper-left triangle, 12 elsewhere"""
    for r in range(10):
        for c in range(10):
            G[r0 + r, c0 + c] = 15 if r + c < 9 else 12


def move(G, r0, c0, r1, c1, dr, dc, col):
    """translate a solid block"""
    paint(G, r0, c0, r1, c1, BG)
    paint(G, r0 + dr, c0 + dc, r1 + dr, c1 + dc, col)
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


class _Base(unittest.TestCase):
    def feed(self, rs, G, new_life=False, line=None):
        bg, objs = segment(G, line)
        return rs.observe(G, bg, objs, None, new_life=new_life, line=line)

    def settle(self, rs, G, n=STATIC_AFTER + 1):
        for _ in range(n):
            self.feed(rs, G)


class TestEquality(_Base):

    def make(self):
        G = board()
        paint(G, 0, 0, 17, 17, 4)          # a frame, 2-cell border
        paint(G, 2, 2, 15, 15, BG)
        picture(G, 3, 3)                   # target inside it
        paint(G, 34, 27, 43, 36, 0)        # blank canvas
        return G

    def test_pair_found_and_progress_tracks_mismatch(self):
        rs = RelSense(); G = self.make()
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        self.assertEqual(rs.pairs(), [])
        for r in range(10):
            for c in range(10):
                if r + c < 9:
                    G[34 + r, 27 + c] = 15
        o = self.feed(rs, G)
        self.assertEqual(len(rs.pairs()), 1)
        self.assertEqual(rs.pairs()[0], ((34, 27, 43, 36), (3, 3, 12, 12)))
        self.assertEqual(o["n"], 1)
        for r in range(10):
            for c in range(10):
                if r + c >= 9:
                    G[34 + r, 27 + c] = 12
        o = self.feed(rs, G)
        self.assertGreater(o["progress"], 0.0)
        self.assertEqual(o["regress"], 0.0)
        self.assertEqual(o["view"][1], 0.0)
        G[40, 30] = 9
        o = self.feed(rs, G)
        self.assertEqual(o["progress"], 0.0)
        self.assertGreater(o["regress"], 0.0)

    def test_a_canvas_painted_in_place_is_not_the_controlled_object(self):
        rs = RelSense(); G = self.make()
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        # the blank canvas shrinks as it is painted, several steps
        for k in range(CTRL_AFTER + 2):
            G[34 + k, 27:37] = 15
            self.feed(rs, G)
        self.assertIsNone(rs.ctrl, "a blob changing in place does not translate")
        self.assertEqual(len(rs.pairs()), 1, "and the canvas is still a canvas")

    def test_a_fragment_does_not_outrank_the_picture(self):
        rs = RelSense(); G = self.make()
        paint(G, 20, 30, 22, 33, 15)
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        G[34, 27] = 15
        G[21, 31] = 12
        self.feed(rs, G)
        self.assertEqual(rs._pursued[1], (34, 27, 43, 36))

    def test_content_box_excludes_the_frame_colour(self):
        G = self.make()
        bg, objs = segment(G)
        self.assertIn((3, 3, 12, 12), relsense.propose_regions(G, bg, objs))


class TestControlledObject(_Base):

    def walk(self, rs, G, box, steps, col=9):
        for dr, dc in steps:
            box = move(G, *box, dr, dc, col)
            self.feed(rs, G)
        return box

    def test_translation_makes_the_controlled_object(self):
        rs = RelSense(); G = board()
        paint(G, 10, 10, 12, 12, 9)
        self.feed(rs, G, new_life=True)
        self.assertIsNone(rs.ctrl)
        self.walk(rs, G, (10, 10, 12, 12), [(0, 3)] * CTRL_AFTER)
        self.assertEqual(rs.ctrl, (9, 9))
        self.assertIsNotNone(rs.self_box)

    def test_a_ring_and_its_centre_are_one_body(self):
        rs = RelSense(); G = board()
        paint(G, 10, 10, 12, 12, 9); G[11, 11] = 4      # a ring with a centre pixel
        paint(G, 30, 30, 32, 32, 0)                     # a static cell on his way
        paint(G, 50, 50, 52, 52, 14)
        self.feed(rs, G, new_life=True)
        box = (10, 10, 12, 12)
        for _ in range(CTRL_AFTER + 1):
            paint(G, *box, BG); box = (box[0], box[1] + 3, box[2], box[3] + 3)
            paint(G, *box, 9); G[box[0] + 1, box[1] + 1] = 4
            self.feed(rs, G)
        self.assertEqual(rs.ctrl, (9, 8), "the tie goes to the larger part")
        self.assertIn((4, 1), rs.body, "the centre pixel is attached")
        self.assertEqual(rs.self_box, box)

    def test_passage_restores_paint_stays(self):
        rs = RelSense(); G = board()
        paint(G, 10, 10, 12, 12, 9)         # him
        paint(G, 10, 20, 12, 22, 0)         # a cell he will pass through
        paint(G, 30, 30, 32, 32, 3)         # a static 3x3 elsewhere
        paint(G, 50, 50, 52, 52, 3)         # another
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        box = self.walk(rs, G, (10, 10, 12, 12), [(0, 3)] * 3)   # he is now at cols 19-21: over the cell
        self.assertEqual(rs.ctrl, (9, 9))
        # the cell under him reads as him; not judged
        G[10:13, 20:23] = BG; G[10:13, 19:22] = 9
        self.feed(rs, G)
        # he leaves; the cell is restored
        box = move(G, 10, 19, 12, 21, 0, 8, 9); paint(G, 10, 20, 12, 22, 0)
        self.feed(rs, G)
        self.assertFalse(rs.regs[(10, 20, 12, 22)][1], "passage is not a change")
        # he stamps paint where he stands, then moves off: that region IS dynamic
        paint(G, 30, 30, 32, 32, 9)          # he lands on the static block (colour 9 over it)
        G[10:13, 27:30] = BG
        self.feed(rs, G)
        paint(G, 30, 30, 32, 32, 7)          # moved off, leaving paint (colour 7) behind
        paint(G, 30, 40, 32, 42, 9)
        self.feed(rs, G)
        move(G, 30, 40, 32, 42, 0, 5, 9)     # one more step away: judged now
        self.feed(rs, G)
        self.assertTrue(rs.regs[(30, 30, 32, 32)][1], "paint that stays is a change")


class TestNear(_Base):

    def make(self):
        G = board()
        paint(G, 10, 10, 12, 12, 9)        # him
        paint(G, 50, 50, 52, 52, 14)       # the goal cell
        paint(G, 63, 0, 63, 63, 7)         # a full-width line
        return G

    def learn_him(self, rs, G):
        box = (10, 10, 12, 12)
        for _ in range(CTRL_AFTER):
            box = move(G, *box, 0, 1, 9); self.feed(rs, G)
        for _ in range(CTRL_AFTER):
            box = move(G, *box, 0, -1, 9); self.feed(rs, G)
        return box

    def test_marker_and_progress_on_approach(self):
        rs = RelSense(); G = self.make()
        self.feed(rs, G, new_life=True)
        self.learn_him(rs, G)
        # a new life: markers are derived with him known
        self.feed(rs, G, new_life=True)
        self.assertEqual(rs.ctrl, (9, 9))
        self.assertIn(("near", (14, 9)), rs.v0)
        self.assertNotIn(("near", (7, 64)), rs.v0, "a full-width line is not a place")
        move(G, 10, 10, 12, 12, 3, 3, 9)
        o = self.feed(rs, G)
        self.assertGreater(o["progress"], 0.0)
        move(G, 13, 13, 15, 15, -3, -3, 9)
        o = self.feed(rs, G)
        self.assertGreater(o["regress"], 0.0)
        self.assertEqual(rs.context(G), ((14, 9), 1, 1))

    def test_a_marker_that_moves_is_dropped(self):
        rs = RelSense(); G = self.make()
        paint(G, 10, 14, 10, 15, 0)        # a 2-cell cap that moves with him
        self.feed(rs, G, new_life=True)
        box = (10, 10, 12, 12); cap = (10, 14, 10, 15)
        for _ in range(CTRL_AFTER + 1):
            box = move(G, *box, 1, 0, 9); cap = move(G, *cap, 1, 0, 0)
            self.feed(rs, G)
        self.feed(rs, G, new_life=True)
        self.assertNotIn((0, 2), rs.markers)

    def test_a_stuck_relation_is_set_aside(self):
        rs = RelSense(); G = self.make()
        paint(G, 10, 20, 12, 22, 3)        # a nearer marker
        self.feed(rs, G, new_life=True)
        self.learn_him(rs, G)
        self.feed(rs, G, new_life=True)
        self.assertEqual(rs._pursued, ("near", (14, 9)), "the biggest job first: the far goal")
        v = int(rs.last[("near", (14, 9))])
        for _ in range(2 * v + 6):         # he never gets nearer to it
            self.feed(rs, G)
        self.assertEqual(rs._pursued, ("near", (3, 9)), "set aside; the next relation")


class TestConfirm(_Base):

    def test_clear_confirms_at_life_minimum_and_persists(self):
        rs = RelSense(); G = TestEquality().make()
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        G[34, 27] = 15
        self.feed(rs, G)
        picture(G, 34, 27)
        rid = rs.confirm(G)
        self.assertEqual(rid, ("eq", (34, 27, 43, 36), (3, 3, 12, 12)))
        d = rs.to_dict()
        self.assertIn("eq:34,27,43,36:3,3,12,12", d["confirmed"])
        rs2 = RelSense()
        self.assertGreater(rs2.from_dict(d), 0)
        self.assertIn(rid, rs2.confirmed)
        G2 = TestEquality().make()
        self.feed(rs2, G2, new_life=True)
        G2[34, 27] = 15
        o = self.feed(rs2, G2)
        self.assertTrue(o["view"][3], "the pursued relation is the confirmed one")

    def test_a_spoil_then_fix_route_still_confirms_the_picture(self):
        # measured 2026-09-06: one of the five cd82 L1 clears came by
        # this route, and an earlier "must be at its life minimum" rule
        # revoked the correct pair and confirmed a marker instead
        rs = RelSense(); G = TestEquality().make()
        paint(G, 50, 50, 52, 52, 14)              # a marker: constant all life
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        picture(G, 34, 27); G[43, 36] = 0         # nearly complete: the minimum
        self.feed(rs, G)
        G[40, 30] = 9; G[41, 31] = 9              # spoiled, then cleared anyway
        self.feed(rs, G)
        rid = rs.confirm(G)
        self.assertEqual(rid, ("eq", (34, 27, 43, 36), (3, 3, 12, 12)))
        self.assertEqual(rs.revoked, 0, "a clear is evidence FOR, never against")

    def test_a_constant_candidate_is_never_confirmed(self):
        rs = RelSense(); G = TestEquality().make()
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        # nothing moved toward any goal this life
        rid = rs.confirm(G)
        self.assertIsNone(rid)
        self.assertEqual(rs.confirmed, set())

    def test_a_satisfied_confirmed_relation_without_a_clear_is_revoked(self):
        rs = RelSense(); G = TestEquality().make()
        self.feed(rs, G, new_life=True)
        self.settle(rs, G)
        G[34, 27] = 15
        self.feed(rs, G)
        rid = ("eq", (34, 27, 43, 36), (3, 3, 12, 12))
        rs.confirmed.add(rid); rs._pursued = rid
        picture(G, 34, 27)                         # satisfied ...
        v0 = rs.v0[rid]
        for _ in range(int(2 * v0) + 7):           # ... and nothing happens
            self.feed(rs, G)
        self.assertNotIn(rid, rs.confirmed)

    def test_malformed_persistence_is_ignored(self):
        rs = RelSense()
        for bad in (None, 3, [], {"static": "x"}, {"confirmed": ["nonsense"]}, {"dynamic": [[1, 2]]}, {"ctrl": [1]}):
            rs.from_dict(bad)
        self.assertEqual(rs.confirmed, set())


if __name__ == "__main__":
    unittest.main()
