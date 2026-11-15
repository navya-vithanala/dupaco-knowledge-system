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
from. Nothing about Dupaco is written into the code as a fixed value: the
routing number, the Home$tart amount, the Hy-Vee closure date, the Forbes
honour count and the join URL are all pulled out of the pages at ingest
time, so the same code would work on a re-crawl.

**Authority decides which fact wins.** Facts are ranked by where they came
from: the rate hub first, then product rate pages, then other dupaco.com
pages, then calculators, promo landing pages, blogs and press releases,
then PDFs. A newer effective date breaks ties. This matters because stale
values live inside the newest snapshot too — an old mortgage rate on a
guidance page, a promo APY on a landing page — and the wrong page will
happily answer with the wrong number.

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
disagreements across snapshots — the branch count (24 in older releases,
22 in the newest), the Forbes honour (third, fourth, then fifth), model-year
ranges on auto loans, and rate disagreements — each with the competing
values, their sources, and which value the system treats as current and
why.

**Every factual claim is cited.** Each source carries the snapshot, the
page URL, and the verbatim quote the answer rests on. Citations are
checked against the snapshot contents; an answer never cites a page that
does not contain the supporting text.

## Testing

- Public practice set (40 questions): **0.969** overall — accuracy 0.969,
  consistency 1.0, speed 1.0 (p95 0.08s, ingest ~6s). The one question
  below 1.0 is a scorer quirk on an empty expected-answer list, not a
  wrong answer.
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

## API keys

None. The system uses only the Python standard library (`re`,
`dataclasses`, `datetime`) and reads only the supplied snapshots. There
is nothing to configure and no secret to supply.

## What is in this repository

- `solution.py` — the whole system.
- `NOTES.md` — this file.
- `requirements.txt` — empty; standard library only.

The snapshots themselves are not included: they are the assignment's
supplied data for a real credit union's website and should not be
redistributed.
