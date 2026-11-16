# BankGPT / interface.ai — Knowledge System over Dupaco Snapshots

## What I built

A deterministic, standard-library-only Python system that answers questions
about Dupaco Community Credit Union using only the five supplied website
snapshots (12 May, 19 May, 9 June, 27 August, 28 September 2026). It makes
no network calls, uses no model or API, and needs no API keys.

The entry point is `solution.py`. `build()` returns an object with three
methods, exactly as the contract requires:

- `ingest(snapshot)` — called once per snapshot, oldest first.
- `answer(question, as_of=None)` — returns text, sources, a confidence flag,
  and a conflict note.
- `conflicts()` — returns the disagreements found across snapshots, with a
  resolution and a reason for each.

## How it works

**Ingest extracts structured facts, it does not memorise answers.** Every
snapshot's pages are parsed into facts — rate tables, the branch directory,
hours, and dated events (branch closures, award announcements, the routing
number, contact details, eligibility wording) — each tagged with the
snapshot it came from, the page URL, and the verbatim text it was taken
from. No answer value is written into the code: the routing number, the
Home$tart amount, the Hy-Vee closure date, the Forbes honour count, the
branch count and the join URL are all pulled out of the pages at ingest
time, so a re-crawl with new values is picked up automatically. The
extraction *rules* are hand-written, though, and some are aimed at specific
pages (the Hy-Vee closure release, the Home$tart release, the Forbes
releases, `/contact`, `/about/join`) and at the site's four Visa cards. A
re-crawl of the same site works; a redesigned site would need new rules.

**Authority decides which fact wins.** Facts are ranked by where they came
from: the rate hub first, then product rate pages, then other dupaco.com
pages, then calculators, promo landing pages, blogs and press releases,
then PDFs. A newer effective date breaks ties. This matters because stale
values live inside the newest snapshot too — an old mortgage rate on a
guidance page, a promo APY on a landing page — and the wrong page will
happily answer with the wrong number. Rate pages always beat other pages,
even when the other page is newer: in the September snapshot the rate
pages renamed a checking tier to "1st Rate Checking" (2.50% APY) while
three promo pages still show the old row at 3.00%, and a newer promo must
not win that. A promo or calculator value is used only when no rate page
ever carried the fact.

**Releases are dated by their own dateline.** Old newsroom releases stay on
the site with the numbers they were written with: every snapshot carries
releases saying 19, 20, 22, 23 and 24 branches. The branch count comes from
the release with the newest publication date (May 5, 2026: 22), and an
as-of question only sees releases published by that date.

**Answers are looked up, not generated.** A question is routed by intent
(link, routing number, branch, branch count, hours, eligibility, and so
on) to a structured lookup that filters facts by the product, term, tier,
balance and attribute actually asked for. APY, dividend rate, APR,
minimum balance, opening deposit and fees are treated as different
attributes — a question about a minimum is never answered with a rate.
Where the question names a product or branch, the match is a hard filter:
if the named thing does not exist, the answer says so instead of
substituting a similar product or another branch's address.

**When the evidence is not there, it says so.** The prose fallback only
answers when the retrieved passage genuinely covers the question; below
that relevance threshold it refuses with "I don't know - this isn't listed
on the website." Questions about dates before the earliest snapshot
refuse, as do questions about unpublished rates (adjustable-rate and
construction loans) and illustrative examples presented as if they were
published rates. A refused answer carries no numbers beyond those in the
question itself.

**As-of questions use the right snapshot.** A date in the question (or the
`as_of` argument) slices the facts to the newest snapshot taken on or
before that date, so a May question gets May's rates and a September
question gets September's. This is how the system handles values that
changed between snapshots, including the 17-month certificate special
that only exists in the later snapshots.

**Conflicts are reported, not hidden.** `conflicts()` lists the
disagreements within and across snapshots — the branch count (19 to 24 in
older releases, 22 in the newest), the Forbes honour (fourth, then fifth),
auto-loan rates where promo and checking pages still show last year's
model-year band, stale mortgage rates on guidance pages, and rates that
changed between snapshots — each with the competing values, a source for
each, the value the system treats as current, and a reason generated from
the evidence (which page, which snapshot, what it outranks). To avoid false
conflicts: values are compared as numbers ("12.50% - 19.50%" and
"12.50% - 19.50% APR" are the same), personal and business products are
kept apart, two rows sharing a label on one page are treated as two
products, and rows whose product cannot be identified are skipped. The
list is capped at 40 entries.

**Every factual claim is cited.** Each source carries the snapshot, the
page URL, and the verbatim quote the answer rests on. Citations are
checked against the snapshot contents; an answer never cites a page that
does not contain the supporting text.

## Testing

- Public practice set (40 questions): **0.969** overall — accuracy 0.969,
  consistency 1.0, speed 1.0 (p95 0.01s, ingest ~4s), on Python 3.10 and
  3.14. The one question below 1.0 (dup-005) has an empty expected-answer
  list, which the scorer grades 0 unless its optional LLM judge is on; the
  answer itself reads back every certificate rate, as the item asks.
- A review pass with reworded questions found and fixed: spelled-out
  numbers ("thirty-year mortgage"), terms given in years for loans
  published in months ("5-year auto loan" → the 60-month row), full dates
  inside the question ("as of May 20, 2026" was wrongly treated as an
  ambiguous month), broader phrasings for the branch count, office
  addresses and opening hours, drive-up-only hours answers, and blog
  passages about other organisations being quoted as facts about Dupaco.
- Beyond the public set I tested with 61 additional questions (unseen,
  deliberately misleading, and historical), then 30 fresh unseen questions
  and 22 adversarial ones covering wrong product names, similar branch
  names, nonexistent certificate terms, missing dates and misleading
  examples. The system refused unsupported questions instead of guessing
  in those tests, and every citation was verified against the snapshots.
- I also ran an independent overfitting check with 65 further questions
  written without knowledge of the implementation. It showed the honest
  limit of this design: the system is reliable on the question styles it
  was built around, and degrades on unfamiliar phrasing — unusual date
  formats, synonyms, negation variants and typos in product names can
  still produce a wrong answer or a refusal where an answer exists. The
  refusal behaviour is strongest for nonexistent products, nonexistent
  branches and pre-evidence dates. This is a rule-based system, and its
  coverage ends where its rules' wording patterns end; that is the
  trade-off for being fully deterministic and explainable.

## Running it

```bash
python harness/run_eval.py --submission <folder> --eval eval/public_eval.jsonl
```

from the kit directory, with `<folder>` pointing at the directory
containing `solution.py`. No installation step is needed beyond Python 3
itself.

Regression tests for the defects fixed in review live in
`tests/test_regressions.py` (standard-library `unittest`). The snapshots are
not in this repository, so point the tests at the kit's unpacked folder;
without it they skip:

```bash
DUPACO_SNAPSHOTS=<kit>/snapshots python -m unittest discover tests
```

## API keys

None. The system uses only the Python standard library (`re`,
`dataclasses`, `datetime`) and reads only the supplied snapshots. There
is nothing to configure and no secret to supply.

## What is in this repository

- `solution.py` — the whole system.
- `NOTES.md` — this file.
- `requirements.txt` — empty; standard library only.
- `tests/test_regressions.py` — regression tests (need the kit's snapshots).

The snapshots themselves are not included: they are the assignment's
supplied data for a real credit union's website and should not be
redistributed.
