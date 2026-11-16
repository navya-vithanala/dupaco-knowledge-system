"""Regression tests for defects found in review.

The website snapshots are not in this repository (they must not be
redistributed), so these tests load them from the participant kit. Point
DUPACO_SNAPSHOTS at the kit's unpacked `snapshots/` folder:

    DUPACO_SNAPSHOTS=/path/to/kit/snapshots python -m unittest discover tests

Without it every test is skipped. Standard library only.
"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import solution  # noqa: E402

SNAPS = os.environ.get("DUPACO_SNAPSHOTS")


class _Page:
    def __init__(self, url, title, markdown, fetched_at):
        self.url, self.title, self.markdown, self.fetched_at = url, title, markdown, fetched_at


class _Snapshot:
    def __init__(self, snapshot_id, taken_on, pages):
        self.snapshot_id, self.taken_on, self.pages = snapshot_id, taken_on, pages


def _load(root):
    snaps = []
    for d in sorted(p for p in Path(root).iterdir() if (p / "manifest.json").exists()):
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        pages = [_Page(p["url"], p.get("title", ""),
                       (d / p["file"]).read_text(encoding="utf-8"),
                       p.get("fetched_at", m["taken_on"])) for p in m["pages"]]
        snaps.append(_Snapshot(m["snapshot_id"], m["taken_on"], pages))
    return sorted(snaps, key=lambda s: s.taken_on)


@unittest.skipUnless(SNAPS, "set DUPACO_SNAPSHOTS to the kit's snapshots folder")
class Regressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ks = solution.build()
        for s in _load(SNAPS):
            cls.ks.ingest(s)

    def ask(self, q, as_of=None):
        return self.ks.answer(q, as_of=as_of)

    def assertAnswers(self, q, *expected, as_of=None, absent=()):
        a = self.ask(q, as_of)
        self.assertTrue(a["confident"], f"{q!r} refused: {a['text']}")
        for e in expected:
            self.assertIn(e, a["text"], q)
        for x in absent:
            self.assertNotIn(x, a["text"], q)

    def assertRefuses(self, q, as_of=None):
        a = self.ask(q, as_of)
        self.assertFalse(a["confident"], f"{q!r} answered: {a['text']}")

    # -- dates ------------------------------------------------------------
    def test_full_date_in_question_is_not_an_ambiguous_month(self):
        # "May 20, 2026" was refused as an ambiguous bare month
        self.assertAnswers("How many branches did Dupaco have as of May 20, 2026?", "22")

    def test_iso_date_in_question(self):
        self.assertAnswers("What was the 30-year mortgage rate on 2026-06-10?", "6.125%",
                           absent=("7.13%",))

    def test_timestamp_as_of(self):
        self.assertAnswers("What was the 10-month fixed jumbo special certificate rate?",
                           "3.90%", as_of="2026-05-31T00:00:00", absent=("4.00%",))

    def test_bare_month_is_still_ambiguous(self):
        self.assertRefuses("What was the 30-year mortgage rate in May?")

    # -- terms ------------------------------------------------------------
    def test_spelled_out_term(self):
        self.assertAnswers("What is the current rate on a thirty-year mortgage?", "7.13")
        self.assertAnswers("APY for the seventeen-month special CD?", "4.10%")

    def test_years_become_months_for_month_based_loans(self):
        # "5-year" auto loan was answered with the 48-month rate
        self.assertAnswers("Rate for a 5-year auto loan on a 2025 vehicle?", "60-month", "5.39%",
                           absent=("5.19%",))
        self.assertAnswers("personal loan rate for 3 years", "10.49%")

    def test_model_year_with_car(self):
        self.assertAnswers("What is the APR on a 60 month car loan for a 2025 car?", "5.39%")

    def test_nonexistent_term_still_refused(self):
        self.assertRefuses("What is the APY on a 13-month certificate?")

    # -- branches ---------------------------------------------------------
    def test_branch_count_phrasings(self):
        for q in ("How many branch locations does Dupaco operate?",
                  "How many locations does Dupaco have?",
                  "Number of Dupaco branches?"):
            self.assertAnswers(q, "22", absent=("24 branch",))

    def test_office_address(self):
        self.assertAnswers("Where is the Galena office located?", "11375 Oldenburg", "61036")

    def test_unknown_branch_refused(self):
        self.assertRefuses("Address of the Chicago branch?")

    def test_hours_phrasings_and_no_boilerplate(self):
        a = self.ask("When are branches open?")
        self.assertTrue(a["confident"])
        self.assertIn("8:30", a["text"])
        self.assertNotIn("ABOUT DUPACO", a["text"])

    def test_drive_up_only(self):
        for q in ("What are the drive-up hours?", "What time does the drive-thru open?"):
            self.assertAnswers(q, "8 a.m.", "6 p.m.", absent=("Lobby",))

    # -- prose fallback ---------------------------------------------------
    def test_how_many_without_a_number_is_not_answered_from_prose(self):
        a = self.ask("How many members does Dupaco have?")
        if a["confident"]:
            self.assertRegex(a["text"], r"\d")

    def test_blog_post_about_another_org_not_quoted(self):
        a = self.ask("Who is Dupaco's CEO?")
        self.assertNotIn("their current CEO retiring", a["text"])

    # -- conflicts --------------------------------------------------------
    def test_conflicts(self):
        cs = self.ks.conflicts()
        by_topic = {c.topic: c for c in cs}
        # same number, different formatting, is not a conflict
        for c in cs:
            nums = {v.replace(" APR", "").replace(" APY", "") for v in c.values}
            self.assertGreater(len(nums), 1, c.topic)
            # every topic names a product, not a bare balance tier
            self.assertNotRegex(c.topic, r"^\$", c.topic)
        count = by_topic["Number of Dupaco branch office locations"]
        self.assertEqual(count.resolution, "22")
        auto60 = next(c for c in cs if c.topic.startswith("Auto loan APR, 60-month"))
        self.assertIn("5.34% APR", auto60.values)
        self.assertEqual(auto60.resolution, "5.39% APR")
        # a newer promo page must not beat an older rate page
        chk = next(c for c in cs if c.topic.startswith("Checking $5,000") and c.topic.endswith("APY"))
        self.assertEqual(chk.resolution, "2.50%")
        # two cards sharing a row label on one page are not a conflict
        self.assertFalse(any(c.topic.startswith("Variable Interest Rate") for c in cs))


if __name__ == "__main__":
    unittest.main()
