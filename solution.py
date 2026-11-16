"""
Dupaco knowledge system — build assignment (interface.ai).

Approach: deterministic, rules + structured extraction. No LLM, no network,
no API keys. At ingest time every snapshot is parsed into versioned facts
(rate tables -> product / tier / metric / value, with the source page, the
snapshot date and the page's authority). Answers are composed from exactly
one resolved fact (or a short list of sibling rows), never from a paragraph
that also contains a stale value. Conflicts are derived from the same fact
timeline: the same product/tier/metric seen with different values across
pages or snapshots.

Authority order (highest first): the /rates hub and the product's own rate
page > other dupaco.com pages > calculators and promo pages > blog/newsroom
> PDFs. Within equal authority the newest snapshot wins, then the newest
"rates effective as of" date on the page. That is what makes a stale
calculator page (5.34%) lose to the auto-loan rate page (5.39%), and an old
press release ("24 branches", "fourth" Forbes honour) lose to the newest one.

Only the Python standard library is used. The Answer/Source/Conflict types
below are local copies of the contract shapes, so this file works standalone.
"""

import re
from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# Contract-compatible types (local copies; the harness accepts any object
# with these fields).
# ---------------------------------------------------------------------------

@dataclass
class Source:
    url: str
    snapshot_id: str = None
    quote: str = None


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    confident: bool = True
    conflict_note: str = None


@dataclass
class Conflict:
    topic: str
    values: list
    sources: list
    resolution: str = None
    reason: str = None


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent)?")
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
MONEY_RE = re.compile(r"\$([\d,]+(?:\.\d{2})?)")
MONTHS_RE = re.compile(r"(\d+)\s*[- ]?months?\b", re.I)
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
            "fifteen": 15, "seventeen": 17, "eighteen": 18, "twenty": 20,
            "twenty-four": 24, "thirty": 30, "thirty-six": 36,
            "forty-eight": 48, "sixty": 60, "seventy-two": 72,
            "eighty-four": 84}
# longest first, so "twenty-four" is replaced before "twenty"
_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(sorted(WORD_NUM, key=len, reverse=True)) +
    r")(?=[\s\-]+(?:months?|years?)\b)", re.I)


def _digits_for_words(s):
    """'thirty-year' -> '30-year', 'seventeen month' -> '17 month': spelled
    terms become the digits every qualifier parser below expects. Only a
    number word directly before 'month'/'year' is touched."""
    return _WORD_NUM_RE.sub(lambda m: str(WORD_NUM[m.group(1).lower()]), s or "")


def _clean(s):
    s = re.sub(r"[*_`#\\]", "", s or "")
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _raw(s):
    """A quote exactly as it appears in the page: whitespace collapsed,
    markdown markers and punctuation untouched, so it verifies verbatim
    against the raw snapshot markdown."""
    return re.sub(r"\s+", " ", s or "").strip()


def _pct(s):
    """First percentage in a cell, as float; None if there is none."""
    m = PCT_RE.search(s or "")
    return float(m.group(1)) if m else None


def _pct_range(s):
    """'12.50% - 19.50%' -> (12.5, 19.5); '5.39%' -> (5.39, 5.39); None if no %."""
    vals = [float(v) for v in PCT_RE.findall(s or "")]
    if not vals:
        return None
    return (vals[0], vals[-1])


def _money(s):
    m = MONEY_RE.search(s or "")
    return float(m.group(1).replace(",", "")) if m else None


def _fmt_money(v):
    """Format a parsed amount for an answer: 100000.0 -> '100,000',
    25.0 -> '25', 24999.99 -> '24,999.99'."""
    if v is None:
        return ""
    fv = float(v)
    if fv == int(fv):
        return f"{int(fv):,}"
    return f"{fv:,.2f}"


def _parse_range(s):
    """'$10,000 - $24,999.99' -> (10000.0, 24999.99); '$25,000+' -> (25000.0, inf);
    '$0.01 - $4,999.99' -> (0.01, 4999.99). None if no money present."""
    s = (s or "").replace(",", "")
    nums = [float(v) for v in re.findall(r"\$(\d+(?:\.\d+)?)", s)]
    if not nums:
        return None
    if "+" in s:
        return (nums[0], float("inf"))
    if len(nums) >= 2:
        return (nums[0], nums[1])
    # "$10,000-24,999": one $ sign, the upper bound after the hyphen
    m2 = re.search(r"\$\d+(?:\.\d+)?\s*[-\u2013]\s*(\d+(?:\.\d+)?)", s)
    if m2:
        return (nums[0], float(m2.group(1)))
    return (nums[0], nums[0])


def _parse_months(s):
    m = MONTHS_RE.search(s or "")
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)[\s\-]?(?:month|mo)\b", s or "", re.I)
    return int(m.group(1)) if m else None


def _parse_model_years(s):
    """'2024-27 Model Year' -> (2024, 2027); '2016 Model Year (and older)' -> (1900, 2016);
    '2023-26 Model Year' -> (2023, 2026)."""
    s = s or ""
    m = re.search(r"(20\d\d)\s*[-–]\s*(\d{2})\b", s)
    if m:
        lo = int(m.group(1))
        hi = 2000 + int(m.group(2))
        return (lo, hi)
    m = re.search(r"(20\d\d)\s*[-–]\s*(20\d\d)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"(20\d\d)\s+Model Year", s)
    if m:
        y = int(m.group(1))
        return (y, y) if "older" not in s.lower() else (1900, y)
    return None


def _parse_loan_amount(s):
    """'$832,750 or less' -> (0, 832750); '$10,000-24,999' -> (10000, 24999)."""
    s = (s or "").replace(",", "")
    nums = [float(v) for v in re.findall(r"\$(\d+(?:\.\d+)?)", s)]
    if not nums:
        return None
    if "or less" in s.lower():
        return (0.0, nums[0])
    if len(nums) >= 2:
        return (nums[0], nums[1])
    # "$10,000-24,999": one $ sign, the upper bound after the hyphen
    m2 = re.search(r"\$\d+(?:\.\d+)?\s*[-\u2013]\s*(\d+(?:\.\d+)?)", s)
    if m2:
        return (nums[0], float(m2.group(1)))
    return (nums[0], nums[0])


def _parse_date(s):
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s or "")
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    return None


def _tail_until_heading(lines, j):
    """Lines after a table up to the next markdown heading (max 30): the
    footnotes belong to the table above them, not to the next section."""
    out = []
    for ln in lines[j:j + 30]:
        if ln.strip().startswith("#"):
            break
        out.append(ln)
    return out


def _iso(d):
    return d.isoformat() if isinstance(d, date) else (d or "")


MONTH_NUM = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
             "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
             "november": 11, "december": 12}
ORDINAL_NUM = {"first": "one", "second": "two", "third": "three",
               "fourth": "four", "fifth": "five", "sixth": "six"}


def _month_date(s):
    """"March 31, 2026" / "June 18, 2024" -> date; None if no match."""
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(20\d\d)", s or "", re.I)
    if not m:
        return None
    try:
        return date(int(m.group(3)), MONTH_NUM[m.group(1).lower()], int(m.group(2)))
    except ValueError:
        return None


def _is_product_heading(label, context):
    """True when a table row's label is only a balance tier ("$25,000 -
    $99,999.99") and the line above the table is a short product name
    ("Investor's Choice") rather than a marketing sentence."""
    c = _clean(context or "")
    return bool(re.match(r"\s*\$", label or "") and c
                and len(c) <= 40 and not re.search(r"[.!?:,]$|^[\$\d]", c))


def _uniq_disp(fs):
    """Distinct values among facts, compared by number, not by text:
    "12.50% - 19.50%" and "12.50% - 19.50% APR" are one value."""
    out = {}
    for f in fs:
        out.setdefault(f.value, f.disp)
    return sorted(out.values())


def _fmt_date(d):
    return f"{d.strftime('%B')} {d.day}, {d.year}" if isinstance(d, date) else ""


# ---------------------------------------------------------------------------
# Page authority: official rate pages beat calculators, promos, blogs and
# press releases. A stale calculator page must never outrank the rate page.
# ---------------------------------------------------------------------------

def _authority(url):
    u = url.lower()
    if u.endswith("/rates") or u == "https://www.dupaco.com/rates":
        return 100
    # offer landing pages ("-lp") are marketing promos, not the rate schedule
    if "-lp" in u:
        return 40
    if any(k in u for k in ("/personal/savings-account", "/business-rates",
                            "/personal/loans/auto-motorcycle-loans",
                            "/personal/loans/home-loans",
                            "/personal/loans/personal-loans",
                            "/personal/credit-cards",
                            "/business/credit-cards",
                            "/personal/checking", "/checking")):
        return 95
    if "/calculators/" in u or "switch-and-earn" in u or "earn-more" in u:
        return 40
    if "/learn/blog" in u or "/newsroom/" in u:
        return 30
    if u.endswith(".pdf"):
        return 20
    if "dupaco.com" in u:
        return 80
    return 50


RATE_HDR = re.compile(r"rate|apr|apy|dividend", re.I)


def _metric_for_header(h):
    h = h.lower()
    if "apy" in h or "annual percentage yield" in h:
        return "apy"
    if "apr" in h or "annual percentage rate" in h:
        return "apr"
    if "dividend" in h:
        return "div"
    if "rate" in h:
        return "rate"
    return None


# ---------------------------------------------------------------------------
# The knowledge system
# ---------------------------------------------------------------------------

STOP = set("""a an the and or of to in on for with is are was were be by at as it its
this that these those from what when where which who whom how do does did can could
should would will you your we our they their he she his her i me my us our dupaco
please tell me get give show list provide what is are the""".split())


def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9$%.]+", (s or "").lower()) if t not in STOP]


def _stem(w):
    """Crude stem for matching only: 'serves'/'serve', 'counties'/'county',
    'branches'/'branch' must meet. Both sides are stemmed the same way."""
    w = (w or "").lower()
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 3 and w.endswith("ed"):
        w = w[:-2]
    if len(w) > 3 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w


def _stems(s):
    return {_stem(t) for t in _tokens(s)}


# Words that carry no information in a prose question (the question's
# subject lives in the other tokens). Used by the relevance gate only.
GENERIC_PROSE = {"current", "dupaco", "rate", "apr", "apy", "dividend",
                 "interest", "account", "product", "information", "please",
                 "tell", "give", "show", "list", "provide", "offer", "offers",
                 "option", "options", "name", "named", "called", "much", "many",
                 "what", "which", "who", "whom", "whose", "when", "where",
                 "why", "how", "is", "are", "was", "were", "be", "been",
                 "being", "am", "do", "does", "did", "have", "has", "had",
                 "can", "could", "will", "would", "should", "may", "might",
                 "the", "a", "an", "and", "or", "but", "if", "then", "than",
                 "so", "such", "of", "in", "on", "at", "to", "for", "from",
                 "with", "by", "about", "as", "into", "through", "after",
                 "before", "between", "over", "under", "out", "up", "down",
                 "it", "its", "this", "that", "these", "those", "there",
                 "here", "i", "me", "my", "we", "us", "our", "you", "your",
                 "he", "she", "they", "them", "their", "his", "her",
                 "one", "two", "first", "new", "more", "most", "all", "any",
                 "some", "no", "not", "only", "very", "really", "still",
                 "now", "today", "currently", "available", "service", "last",
                 "earn", "pay", "paid", "return", "returns", "cost", "charge",
                 "charges", "work", "works", "working", "help", "helps",
                 "regular", "standard",
                 "services", "member", "members", "website", "site", "page",
                 "online", "bank", "credit", "union", "financial", "money",
                 "time", "year", "years", "month", "months", "day", "days",
                 "per", "each", "every", "get", "gets", "got", "find", "know",
                 "need", "needs", "want", "wants", "like", "look", "looking",
                 "take", "takes", "use", "used", "using", "go", "goes", "come",
                 "comes", "see", "says", "said", "say", "also", "just",
                 "even", "well", "back", "much", "many", "few", "other",
                 "another", "same", "different", "best", "top", "highest",
                 "lowest", "number", "amount", "total", "full", "open",
                 "close", "closed", "still", "yet", "already", "ever",
                 "never", "always", "often", "sometimes", "usually"}


def _refuse_note(note):
    """A refusal that also states, in words with no digits, why the
    requested thing is not answerable (e.g. the product is not listed).
    Confident=False, so it counts as a refusal everywhere a refusal does."""
    return Answer(text=f"I don't know - {note} Please contact us.",
                  sources=[], confident=False)


class Fact:
    __slots__ = ("product", "display", "quals", "metric", "value", "disp",
                 "url", "snap", "taken", "auth", "eff", "section", "row")

    def __init__(self, product, display, quals, metric, value, disp, url,
                 snap, taken, auth, eff, section, row=""):
        self.product = product          # normalised product key
        self.display = display          # human product label
        self.quals = quals              # dict: term, years, balance, amount, kind...
        self.metric = metric            # rate | apr | apy | div
        self.value = value              # (lo, hi) floats, or single float
        self.disp = disp                # display string, e.g. "5.39%" or "12.50% - 19.50%"
        self.url = url
        self.snap = snap
        self.taken = taken              # snapshot date string
        self.auth = auth
        self.eff = eff                  # effective date (date or None)
        self.section = section
        self.row = row                  # verbatim markdown table row the fact came from

    def key(self):
        # The display name is part of the identity: without it, different
        # products that share a balance tier (Achieve vs Investor's Choice
        # money market, the four Visa cards) collapse into one bogus
        # "conflict", and personal vs business rows mix. Normalised so that
        # "$832,750" and "$832750" spellings of the same tier still match.
        q = self.quals
        disp = re.sub(r"[^a-z0-9]", "", (self.display or "").lower())
        # business and personal accounts are different products even when
        # a row carries the same name ("Share Savings" on /business-rates)
        biz = bool(q.get("biz")) or "business" in (self.url or "").lower()
        return (self.product, self.metric, disp, biz,
                q.get("term"), q.get("years"), q.get("balance"),
                q.get("amount"), q.get("kind"))


class Knowledge:
    def __init__(self):
        self.snaps = []            # [(snapshot_id, taken_on)] oldest -> newest
        self.facts = []            # list[Fact]
        self.chunks = []           # (url, snap, taken, auth, text, tokens set, is_example)
        self.branches = []         # (city, address, zip, url, snap, taken, quote)
        self.located = []          # branch addresses from "located at ..." prose
        self.hours = []            # (kind, schedule text, url, snap, taken)
        self.counts = []           # (n, url, snap, taken, sentence, pub date) "N branch office locations"
        self._cur_pub = None       # publication date of the page being ingested
        self.conditions = []       # footnote conditions: {kind, fee_words, value,
                                   #   text, product, display, url, snap, taken}
        self.page_urls = set()     # every ingested page URL (for link answers)
        self.links = []            # (anchor, href, page_url, snap, taken) markdown links
        self.events = []           # {"kind","url","snap","taken","pub","data","quote"}
        self._conflicts = None

    # -- ingest ---------------------------------------------------------

    def ingest(self, snapshot) -> None:
        sid, taken = snapshot.snapshot_id, snapshot.taken_on
        self.snaps.append((sid, taken))
        self.snaps.sort(key=lambda x: x[1])
        for p in snapshot.pages:
            self._ingest_page(p.url, p.title, p.markdown, sid, taken)

    def _ingest_page(self, url, title, md, sid, taken):
        auth = _authority(url)
        # first full date on the page: a newsroom release's dateline
        self._cur_pub = _month_date(md)
        lines = md.split("\n")
        heading = title or ""
        context = ""           # last meaningful non-table line (product names live here)
        i = 0
        while i < len(lines):
            ln = lines[i]
            s = ln.strip()
            if s.startswith("#"):
                heading = _clean(s)
                i += 1
                continue
            if s.startswith("|") and i + 1 < len(lines) and re.match(
                    r"\s*\|[\s\-:|]+\|\s*$", lines[i + 1]):
                hdr = [c.strip() for c in s.strip().strip("|").split("|")]
                rows = []
                raw_rows = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    raw_rows.append(lines[j].strip())
                    rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                    j += 1
                self._ingest_table(url, sid, taken, auth, heading, context,
                                   hdr, rows, "\n".join(lines[j:j + 8]),
                                   raw_rows, _tail_until_heading(lines, j))
                i = j
                continue
            if s and not s.startswith("![") and not s.startswith("["):
                txt = _clean(s)
                if len(txt) > 3:
                    context = txt
                    # prose chunks keep the RAW line: a citation quote must
                    # be verbatim in the snapshot markdown, and _clean strips
                    # the markers (** , curly quotes, em-dashes) that the raw
                    # page has. Answer text is cleaned at render time.
                    raw = _raw(s)
                    if len(txt) > 40 and "http" not in txt[:0]:
                        # an illustrative payment example is not a published
                        # fact: flag it so the fallback never quotes it as a rate
                        is_ex = bool(re.search(
                            r"illustrative|for illustrative purposes|"
                            r"payment example|example for .* purposes", txt, re.I))
                        self.chunks.append((url, sid, taken, auth, raw,
                                            set(_tokens(txt)), is_ex))
                    self._scan_prose(url, sid, taken, auth, raw)
            i += 1
        # Hours blocks span several markdown lines ("Branch hours are:
        # **Lobby** Monday-Thursday: ... **Drive-up** Monday-Friday: ..."),
        # so scan the whole flattened page text, not single lines.
        flat = re.sub(r"\s+", " ", md)
        for kind in ("Lobby", "Drive-up"):
            for mm in re.finditer(kind + r"\b(.{0,700})", flat, re.I):
                seg = mm.group(0)
                if re.search(r"\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?", seg):
                    self.hours.append((kind.lower(), _raw(seg), url, sid, taken))
        # branch directory entries live on /contact
        if url.rstrip("/").endswith("/contact"):
            self._ingest_branches(md, url, sid, taken)
        # event facts (routing number, grants, closures, recognitions...)
        # are extracted from the page text itself, never written in code
        self.page_urls.add(url)
        self._scan_events(url, sid, taken, md, flat)

    def _scan_prose(self, url, sid, taken, auth, txt):
        m = re.search(r"(\d+)\s+branch office locations", txt, re.I)
        if m:
            # the page's own publication date (the release dateline): the
            # count from the newest release is the current one
            self.counts.append((int(m.group(1)), url, sid, taken, _raw(txt),
                                self._cur_pub))
        # hours blocks: "Lobby Monday-Thursday: 8:30 a.m.-5 p.m. ... Drive-up ..."
        for kind in ("Lobby", "Drive-up"):
            for mm in re.finditer(kind + r"\b(.{0,400})", txt, re.I):
                seg = mm.group(0)
                if re.search(r"\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?", seg):
                    self.hours.append((kind.lower(), _raw(seg), url, sid, taken))
        # branch addresses named in prose ("building located at 2530 Corridor
        # Way in Coralville", "branch in Madison, located at 7701 Mineral
        # Point Road"): the directory does not list every branch, so these
        # are second-class records, used only when the directory has no match
        for lm in re.finditer(
                r"located at\s+(\d+\s+[^,\n]+?(?:Way|Road|Rd\.?|Street|St\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Boulevard|Blvd\.?))",
                txt, re.I):
            addr = _clean(lm.group(1)).rstrip(",")
            window = txt[max(0, lm.start() - 160):lm.end() + 80]
            city = None
            for cm in re.finditer(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", window):
                city = cm.group(1)
            self.located.append((city or "", addr, "", url, sid, taken,
                                 _raw(txt)))

    def _ingest_branches(self, md, url, sid, taken):
        # entries look like: "3299 Hillcrest Rd., Dubuque, IA, 52001"
        for m in re.finditer(
                r"(\d+\s+[^,\n]+(?:Ln|Lane|Rd|Road|St|Street|Ave|Avenue|Dr|Drive|Way|Blvd|Hwy)[^,\n]*,\s*"
                r"([A-Za-z ]+?),\s*[A-Z]{2},?\s*(\d{5}))", md):
            addr = _clean(m.group(1)).rstrip(",")
            self.branches.append((m.group(2).strip(), addr, m.group(3), url, sid, taken,
                                  _clean(m.group(0))))

    def _scan_events(self, url, sid, taken, md, flat):
        """Extract event facts from a page's own text: the routing number,
        the Home$tart grant, the contact-update phone, the eligibility
        wording, the Hy-Vee closure and the Forbes ordinals. Values live
        here, at ingest; answer code only formats what was extracted."""
        u = url.lower()
        # every markdown link on the page (the join-portal link lives here)
        for m in re.finditer(r"\[([^\]]{1,80})\]\((https?://[^)\s]+)\)", md):
            self.links.append((m.group(1), m.group(2), url, sid, taken))
        # routing number: 9 digits near the word "routing" on the
        # contact / checking / cash-management pages
        if u.rstrip("/").endswith(("/contact", "/checking",
                                    "/business/cash-management",
                                    "/personal/checking-accounts",
                                    "/business/checking-accounts")):
            m = re.search(r"routing number\D{0,80}?(?<!\d)(\d{9})(?!\d)",
                          flat, re.I)
            if m:
                self.events.append({"kind": "routing", "url": url, "snap": sid,
                                    "taken": taken, "pub": None,
                                    "data": {"number": m.group(1)},
                                    "quote": _raw(flat[max(0, m.start() - 30):m.end() + 10])})
        # Home$tart grant (newsroom release)
        if "hometart-grants" in u:
            m = re.search(r"provides?\s+up\s+to\s+\$([\d,]+(?:\.\d{2})?)",
                          flat, re.I)
            if m:
                window = flat[max(0, m.start() - 250):m.end() + 300]
                purpose = None
                pm = re.search(r"toward\s+([a-z][a-z ]+?)(?:\s+for\s|\s+beginning|\.)",
                               window, re.I)
                if pm:
                    purpose = _clean(pm.group(1))
                self.events.append({"kind": "hometart", "url": url, "snap": sid,
                                    "taken": taken, "pub": None,
                                    "data": {"amount": m.group(1), "purpose": purpose},
                                    "quote": _raw(window[:200])})
        # contact-info update: the phone number on the /contact page, and
        # whether the page points members at Shine
        if u.rstrip("/").endswith("/contact"):
            m = re.search(r"(?<!\d)(\d{3}-\d{3}-\d{4})(?!\d)", flat)
            if m:
                self.events.append({"kind": "contact-update", "url": url,
                                    "snap": sid, "taken": taken, "pub": None,
                                    "data": {"phone": m.group(1),
                                             "shine": "shine" in flat.lower()},
                                    "quote": _raw(flat[max(0, m.start() - 60):m.end() + 60])})
            # mailing address: "P.O. Box 179, Dubuque, IA 52004-0179"
            mm = re.search(r"P\.O\. Box\s+(\d+)[,\s]+([A-Za-z ]+?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
                           flat)
            if mm:
                self.events.append({"kind": "mailing", "url": url, "snap": sid,
                                    "taken": taken, "pub": None,
                                    "data": {"box": mm.group(1), "city": mm.group(2).strip(),
                                             "state": mm.group(3), "zip": mm.group(4)},
                                    "quote": _raw(flat[max(0, mm.start() - 40):mm.end() + 10])})
        # founding and service area: the boilerplate on the about page and
        # the newsroom/blog releases ("Founded in 1948 by ten Dubuque
        # Packing Company employees", "118 counties throughout ...")
        fm = re.search(r"Founded in\s+(19\d\d|20\d\d)\s+by\s+([^.,\n]{3,80}?)(?:,|\.)",
                       flat)
        if fm:
            self.events.append({"kind": "founded", "url": url, "snap": sid,
                                "taken": taken, "pub": None,
                                "data": {"year": int(fm.group(1)),
                                         "by": _clean(fm.group(2))},
                                "quote": _raw(flat[max(0, fm.start() - 40):fm.end() + 60])})
        cm = re.search(r"(\d+)\s+counties throughout\s+([^.\n]{5,140}?)(?:\.|$)",
                       flat, re.I)
        if cm:
            self.events.append({"kind": "counties", "url": url, "snap": sid,
                                "taken": taken, "pub": None,
                                "data": {"n": int(cm.group(1)),
                                         "area": _clean(cm.group(2))},
                                "quote": _raw(flat[max(0, cm.start() - 60):cm.end() + 10])})
        # a branch's own published phone number, e.g. "The phone number for
        # the Mineral Point Road branch is (608) 282-6000" in a release
        pm = re.search(r"[Pp]hone number for the\s+([A-Za-z ]+?)\s+branch is\s+"
                       r"\(?(\d{3})\)?[- ](\d{3})[- ](\d{4})", flat)
        if pm:
            self.events.append({"kind": "branch-phone", "url": url, "snap": sid,
                                "taken": taken, "pub": None,
                                "data": {"branch": _clean(pm.group(1)),
                                         "phone": f"({pm.group(2)}) {pm.group(3)}-{pm.group(4)}"},
                                "quote": _raw(flat[max(0, pm.start() - 20):pm.end() + 10])})
        # eligibility: the charter-area bullet and the states sentence on
        # the /about/join page
        if u.rstrip("/").endswith("/about/join"):
            charter = re.search(r"live or work within (?:our )?\[?charter area",
                                flat, re.I)
            states = None
            sm = re.search(r"find us in ([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*"
                           r"\s+and\s+[A-Z][a-z]+)", flat)
            if sm:
                states = [s.strip() for s in
                          re.split(r",\s*|\s+and\s+", sm.group(1)) if s.strip()]
            if charter or states:
                self.events.append({"kind": "eligibility", "url": url,
                                    "snap": sid, "taken": taken, "pub": None,
                                    "data": {"charter": bool(charter),
                                             "states": states or []},
                                    "quote": _raw(flat[max(0, (charter.start() if charter else sm.start()) - 40):
                                                         (charter.end() if charter else sm.end()) + 160])})
        # Hy-Vee closure (newsroom release): the closing date and the
        # branches members are directed to
        if "hy-vee-branch-closure" in u:
            dm = re.search(r"close\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(20\d\d)", flat)
            bm = re.search(r"branches at ([A-Z][A-Za-z ]+?) and ([A-Z][A-Za-z ]+?)(?:,|\"|”)", flat)
            if dm or bm:
                close = None
                if dm:
                    try:
                        close = date(int(dm.group(3)), MONTH_NUM[dm.group(1).lower()],
                                     int(dm.group(2)))
                    except ValueError:
                        close = None
                branches = ([bm.group(1).strip(), bm.group(2).strip()]
                            if bm else [])
                anchor = dm if dm else bm
                self.events.append({"kind": "hyvee", "url": url, "snap": sid,
                                    "taken": taken, "pub": None,
                                    "data": {"close": close, "branches": branches},
                                    "quote": _raw(flat[max(0, anchor.start() - 40):anchor.end() + 160])})
        # Forbes recognition (newsroom releases): the ordinal in the release
        # text and the release's own publication date (the first date line
        # on the page), so an as-of question picks the release that had
        # actually been announced by then
        if "forbes" in u and "credit-union-in-america" in u:
            om = re.search(r"\b(fifth|fourth|third|second|first)\s+time\b",
                           flat, re.I)
            if not om:
                om = re.search(r"\b(third|fourth|fifth)\s+year in a row\b",
                               flat, re.I)
            if om:
                self.events.append({"kind": "forbes", "url": url, "snap": sid,
                                    "taken": taken, "pub": _month_date(flat[:900]),
                                    "data": {"ordinal": om.group(1).lower()},
                                    "quote": _raw(flat[max(0, om.start() - 60):om.end() + 80])})

    def _events(self, kind, as_of):
        """Events of a kind from the latest snapshot on/before as_of
        (the newest snapshot with no date; the oldest snapshot when the
        date predates every snapshot — the release text still carries its
        own dates, which the handlers compare against as_of)."""
        if as_of:
            ok = [s for s, t in self.snaps if t <= as_of]
            snap = ok[-1] if ok else (self.snaps[0][0] if self.snaps else None)
        else:
            snap = self.snaps[-1][0] if self.snaps else None
        return [e for e in self.events if e["kind"] == kind and e["snap"] == snap]

    def _ingest_table(self, url, sid, taken, auth, heading, context, hdr, rows, tail,
                      raw_rows=None, tail_long=None):
        if not any(RATE_HDR.search(h) for h in hdr):
            return
        # need at least one cell with a % in the body
        if not any("%" in c for r in rows for c in r):
            return
        # metric columns
        metrics = {}
        for idx, h in enumerate(hdr):
            m = _metric_for_header(h)
            if m:
                metrics[idx] = m
        if not metrics:
            return
        # label / qualifier columns
        label_idx = 0
        qual_cols = {}
        for idx, h in enumerate(hdr):
            hl = h.lower()
            if idx in metrics:
                continue
            if "min" in hl and "earn" in hl:
                qual_cols[idx] = "min_earn"
            elif "min" in hl and ("open" in hl or "balance" in hl or "deposit" in hl):
                qual_cols[idx] = "min_open"
            elif "balance" in hl or "tier" in hl and "balance" in hl:
                qual_cols[idx] = "balance"
            elif "term" in hl:
                qual_cols[idx] = "term"
            elif "loan amount" in hl or "loan type" in hl and "amount" in hl:
                qual_cols[idx] = "amount"
            elif "program" in hl or "loan program" in hl or "certificate" in hl \
                    or "account type" in hl or "card type" in hl or "product" in hl \
                    or "ira type" in hl or "term" in hl or "loan type" in hl:
                qual_cols[idx] = "label"
            else:
                qual_cols[idx] = "label"
        eff = _parse_date(tail)
        if not eff:
            eff = _parse_date(heading)
        section = heading
        for ri, r in enumerate(rows):
            if len(r) < len(hdr):
                r = r + [""] * (len(hdr) - len(r))
            if any(c.startswith("![](") for c in r):
                continue
            label = _clean(r[label_idx])
            if not label or label.lower() in ("standard",):
                label = context
            if not label:
                label = section
            quals = self._quals_from_row(label, r, qual_cols, section)
            product = self._product_key(label, context, section, url)
            # a bare-term row ("6 Month") in a dividend+APY table is a
            # certificate row even when the section heading does not say so
            # (the May snapshots' regular cert table sits under a generic
            # UI heading); a personal-loan table has only an APR column,
            # so the two-metric test keeps it from being mislabelled
            if product not in ("certificate", "ira") and \
                    "div" in metrics.values() and "apy" in metrics.values() and \
                    re.fullmatch(r"\d+\s+months?", (label or "").strip().lower()):
                product = "certificate"
            display = self._display_name(label, context, section)
            row_txt = raw_rows[ri] if raw_rows and ri < len(raw_rows) else ""
            for idx, metric in metrics.items():
                val = _pct_range(r[idx])
                if val is None:
                    continue
                # strip footnote digits: "4.20%1" -> "4.20%", "5.39% APR1" -> "5.39% APR"
                disp = _clean(r[idx])
                disp = re.sub(r"(%|APR|APY)\d+$", r"\1", disp).strip()
                self.facts.append(Fact(product, display, quals, metric, val, disp,
                                       url, sid, taken, auth, eff, section, row_txt))
        # footnotes below the table carry the conditions (minimums, fees,
        # credit-score requirements) for the rows just ingested. The display
        # is the first row that is not an introductory / balance-transfer
        # row: the Platinum Visa table's first row is its intro rate, so
        # naming the conditions after it would make a Platinum question
        # unmatchable (and the Business Platinum footnote would win instead).
        if rows:
            first_row = next((r for r in rows
                              if "intro" not in _clean(r[label_idx]).lower()
                              and "balance transfer" not in _clean(r[label_idx]).lower()),
                             rows[0])
            first = self._product_key(_clean(first_row[label_idx] or context),
                                      context, section, url)
            disp0 = self._display_name(_clean(first_row[label_idx] or context),
                                       context, section)
            self._scan_conditions(tail_long or tail, first, disp0,
                                  url, sid, taken, auth)

    def _scan_conditions(self, tail, product, display, url, sid, taken, auth):
        """Footnote conditions for a table: credit-score requirements, fees,
        minimums to open. Each condition keeps the verbatim clause as its
        quote, so an answer about a fee or a requirement is evidence-backed."""
        if not tail:
            return
        text = _clean("\n".join(tail) if isinstance(tail, list) else tail)
        # split the footnote block into clauses on sentence ends and ';'
        clauses = [c.strip() for c in re.split(r"[.;]\s+", text) if len(c.strip()) > 12]
        # one fee can share a clause with another ("There is no annual fee
        # and no balance transfer fee"), so each fee kind in a clause gets
        # its own condition — otherwise the second fee is unanswerable
        fee_patterns = [
            ("cash advance fee", ["cash", "advance"]),
            ("annual fee", ["annual"]),
            ("annual fees", ["annual"]),
            ("balance transfer fee", ["balance", "transfer"]),
            ("late payment fee", ["late", "payment"]),
            ("late fee", ["late", "payment"]),
            ("foreign transaction fee", ["foreign", "transaction"]),
            ("returned payment fee", ["returned", "payment"]),
        ]
        for cl in clauses:
            low = cl.lower()
            kind = None
            value = None
            fee_words = []
            m = re.search(r"(\d{3})\s+credit score or higher", low)
            if m:
                kind, value = "credit_score", int(m.group(1))
            elif "minimum amount required to open" in low:
                mm = re.search(r"\$([\d,]+(?:\.\d{2})?)", cl)
                kind = "min_open"
                value = float(mm.group(1).replace(",", "")) if mm else None
            if kind is not None:
                self.conditions.append({"kind": kind, "fee_words": fee_words,
                                        "value": value, "text": cl,
                                        "product": product, "display": display,
                                        "url": url, "snap": sid, "taken": taken,
                                        "auth": auth})
                continue
            for pat, words in fee_patterns:
                if pat in low:
                    val = "none" if re.search(r"\bno\s+" + re.escape(pat), low) \
                        else None
                    self.conditions.append({"kind": "fee", "fee_words": words,
                                            "value": val, "text": cl,
                                            "product": product, "display": display,
                                            "url": url, "snap": sid, "taken": taken,
                                            "auth": auth})

    @staticmethod
    def _quals_from_row(label, row, qual_cols, section=""):
        q = {}
        for idx, kind in qual_cols.items():
            cell = row[idx] if idx < len(row) else ""
            if kind == "term":
                m = _parse_months(cell)
                if m:
                    q["term"] = m
            elif kind == "balance":
                rg = _parse_range(cell)
                if rg:
                    q["balance"] = rg
            elif kind == "amount":
                rg = _parse_loan_amount(cell)
                if rg:
                    q["amount"] = rg
            elif kind in ("min_earn", "min_open"):
                mv = _money(cell)
                if mv is not None:
                    q[kind] = mv
        # qualifiers hidden inside the label itself
        m = _parse_months(label)
        if m and "term" not in q:
            q["term"] = m
        ym = re.search(r"(\d+)[\s\-]?year", label, re.I)
        if ym and "term" not in q:
            q["yr"] = int(ym.group(1))
        y = _parse_model_years(label)
        if y:
            q["years"] = y
        b = _parse_range(label)
        if b and "balance" not in q and "amount" not in q:
            # a loan table's money range is a loan amount ("$10,000-24,999"
            # on the Boat & RV table), not a deposit balance tier
            if "loan" in (section or "").lower():
                q["amount"] = b
            else:
                q["balance"] = b
        # a loan amount only comes from a label that actually names a loan
        # ("$832,750 or less", "30-Year Fixed Conventional ..."); a bare
        # money range on a deposit row is a balance tier, not a loan amount
        if re.search(r"or less|loan|conventional|fixed|heloc|home equity",
                     label, re.I):
            a = _parse_loan_amount(label)
            if a and "amount" not in q:
                q["amount"] = a
        if re.search(r"\bjumbo\b", label, re.I):
            q["kind"] = "jumbo"
        elif re.search(r"\b(special|fixed)\b", label, re.I):
            q["kind"] = "special"
        if re.search(r"\bpurchase\b", label, re.I):
            q["kind"] = "purchase"
        elif re.search(r"\brefinance\b", label, re.I):
            q["kind"] = "refinance"
        if re.search(r"\bbusiness\b", label, re.I):
            q["biz"] = True
        return q

    @staticmethod
    def _product_key(label, context, section, url):
        s = (label + " " + context + " " + section).lower()
        if "certificate" in s or re.search(r"\bcd\b", s):
            return "certificate"
        if "share savings" in s:
            return "share savings"
        if "bright savings" in s:
            return "bright savings"
        if "money market" in s:
            return "money market"
        if "health savings" in s or re.search(r"\bhsa\b", s):
            return "hsa"
        if "checking" in s:
            return "checking"
        if "auto" in s or "motorcycle" in s or "vehicle" in s:
            return "auto loan"
        if "mortgage" in s or "home loan" in s or "conventional" in s:
            return "mortgage"
        if "home equity" in s or "heloc" in s or "home improvement" in s:
            return "home equity"
        if "personal loan" in s:
            return "personal loan"
        if "boat" in s or re.search(r"\brv\b", s) or "recreational" in s:
            return "recreational loan"
        if "platinum visa" in s or "rewards visa" in s or "credit card" in s or "visa" in s:
            return "credit card"
        if "student" in s:
            return "student loan"
        if "construction" in s:
            return "construction loan"
        if "ira" in s:
            return "ira"
        if "savings" in s:
            return "savings"
        # a bare tier label ("$25,000 - $99,999.99") is not a product: the
        # product is named on the line above the table ("Investor's Choice")
        if _is_product_heading(label, context):
            return _clean(context).lower()[:60]
        return _clean(label).lower()[:60]

    @staticmethod
    def _display_name(label, context, section):
        if label and label.lower() != "standard":
            name = _clean(label)
            # a bare tier row is named after its product ("Investor's
            # Choice $25,000 - $99,999.99"), not after the balance alone
            if _is_product_heading(name, context):
                name = f"{_clean(context)} {name}"
        else:
            name = _clean(context or section)
        # the /rates card table labels the standard row "Platinum Visa Rate"
        if name.lower() == "platinum visa rate":
            return "Platinum Visa"
        return name


    # -- answering ------------------------------------------------------

    def answer(self, question, as_of=None):
        """Public entry point. Returns a plain dict — one of the shapes the
        harness documents as accepted (an Answer, a dict, or any object with
        the same fields). A dict matters because the harness's repeat pass
        reads the answer with .get("text"); a bare dataclass falls through
        to "" there and every repeat check fails."""
        a = self._answer(question, as_of)
        return {"text": a.text,
                "sources": [{"url": s.url, "snapshot_id": s.snapshot_id,
                             "quote": s.quote} for s in (a.sources or [])],
                "confident": a.confident,
                "conflict_note": a.conflict_note}

    def _answer(self, question, as_of=None):
        q = _digits_for_words(question or "")
        ql = q.lower()
        # an as_of passed as a timestamp ("2026-05-31T00:00:00") is the date
        # part; snapshot dates are compared as ISO strings
        if as_of:
            m = re.match(r"\s*(\d{4}-\d{2}-\d{2})", str(as_of))
            as_of = m.group(1) if m else None
        eff_as_of = as_of or self._date_in_question(ql)
        if eff_as_of == "AMBIGUOUS":
            return _refuse_note(
                "that date is ambiguous; please give the month and year.")
        facts = self._slice(eff_as_of)
        # A date before the earliest snapshot (2026-05-12) has no evidence
        # behind it: the snapshots start in May 2026, so a rate, address or
        # count "as of" 2020/2019 cannot be answered from them. Hy-Vee and
        # Forbes are the exceptions — their releases carry their own
        # publication dates inside the snapshots, so they are answered from
        # those dates and handled before this check.
        predates = self._predates(eff_as_of)

        # 0. Hy-Vee closure — date and redirect branches extracted from
        # the newsroom release; as-of aware (open before the closing date)
        if "hy-vee" in ql or "hyvee" in ql:
            a = self._answer_hyvee(ql, eff_as_of)
            if a:
                return a

        # 0b. Forbes recognition — the ordinal comes from the newest release
        # announced by the as-of date, never from a fixed string
        if "forbes" in ql:
            return self._answer_forbes(eff_as_of)

        # 1. link / URL questions
        if re.search(r"\b(link|url|website|web ?site)\b", ql):
            a = self._answer_link(ql)
            if a:
                return a

        # 2. routing number — extracted at ingest from the contact /
        # checking / cash-management pages, never written in code
        if "routing" in ql:
            a = self._answer_routing(eff_as_of)
            if a:
                return a

        # 2a. main phone number — the contact page's phone, not a branch
        # phone and never the number from an unrelated page
        if "phone" in ql and "branch" not in ql:
            a = self._answer_main_phone(eff_as_of)
            if a:
                return a

        # 2b. mailing address — the P.O. Box on the /contact page
        if "mailing" in ql:
            a = self._answer_mailing(eff_as_of)
            if a:
                return a

        # 2c. founding — year and founders extracted from the site text
        if "founded" in ql or "founding" in ql:
            a = self._answer_founded(eff_as_of)
            if a:
                return a

        # 2d. service area — how many counties Dupaco serves
        if "counties" in ql or "county" in ql:
            a = self._answer_counties(eff_as_of)
            if a:
                return a

        # 3. branch address / hours
        if ("branch" in ql or "lobby" in ql
                or re.search(r"drive[\s\-]?(up|thru|through)", ql)
                or (re.search(r"\b(office|location)\b", ql)
                    and re.search(r"\b(address|where|located|hours|open)\b", ql))):
            a = self._answer_branch(ql, eff_as_of, q)
            if a:
                return a

        # 4. how many branches
        if re.search(r"\b(how many|number of|count of)\b[^?]*?"
                     r"\b(branch(es)?|branch locations?|locations|offices)\b", ql) \
                and "atm" not in ql:
            a = self._answer_branch_count(eff_as_of)
            if a:
                return a

        # 7. Volunteer Time Off
        if "volunteer time off" in ql or ("volunteer" in ql and "employee" in ql):
            a = self._answer_prose(ql, ["volunteer time off", "employees"],
                                   eff_as_of, q)
            if a:
                return a

        # 8. eligibility / membership — the charter-area wording and the
        # states come from the /about/join page text, not from code
        if re.search(r"eligib|who can (join|become)|become a (dupaco )?member", ql):
            a = self._answer_eligibility(eff_as_of)
            if a:
                return a

        # 9. student loans
        if "student" in ql:
            a = self._answer_prose(ql, ["student loan"], eff_as_of, q)
            if a:
                return a

        # 10. mobile wallet — the provider names come from the ingested
        # mobile-payments page, via the prose fallback
        if "mobile wallet" in ql or "apple pay" in ql or "google pay" in ql:
            a = self._answer_prose(ql, ["apple pay", "google pay", "samsung pay"],
                                   eff_as_of, q)
            if a:
                return a

        # 11. Home$tart — amount and purpose extracted from the newsroom
        # release at ingest
        if "home$tart" in ql or "homestart" in ql or "home start" in ql:
            a = self._answer_hometart(eff_as_of)
            if a:
                return a

        # 12. update contact information — Shine / phone from the /contact
        # page text at ingest
        if "update" in ql and "contact" in ql:
            a = self._answer_contact_update(eff_as_of)
            if a:
                return a

        # 12b. joining online (no "link"/"url" word in the question, so the
        # link handler above does not catch it)
        if "join" in ql:
            a = self._answer_link(ql)
            if a:
                return a

        # 12c. ARM and construction-loan rates are not published anywhere on
        # the site — refuse rather than render some other mortgage list
        if re.search(r"\barm\b|adjustable|construction", ql):
            if not any("arm" in f.display.lower()
                       or "adjustable" in f.display.lower()
                       or "construction" in f.display.lower() for f in facts):
                return self._refuse()

        # 12d. a date before the earliest snapshot (2026-05-12) has no
        # evidence behind it for a rate or a current value: the snapshots
        # start in May 2026, so a rate "as of" 2020 cannot be answered from
        # them — never present current rates as historical ones
        if predates:
            return self._refuse()

        # 13. rate / value lookup from structured facts
        a = self._answer_rate(ql, facts, eff_as_of)
        if a is not None:
            return a

        # 13b. a fee / credit-score / minimum question that named no
        # product the system could match is not answerable from a passing
        # prose mention (a wire-transfer page mentioning "cashier's check"
        # does not state its fee) — refuse rather than quote it
        if re.search(r"\b(fee|credit score|minimum|min)\b", ql):
            return self._refuse()

        # 14. prose fallback
        a = self._answer_prose(ql, None, eff_as_of, q)
        if a:
            return a

        return self._refuse()

    def _answer_mailing(self, as_of):
        evs = self._events("mailing", as_of)
        if not evs:
            return None
        e = evs[0]
        d = e["data"]
        return Answer(
            text=f"Dupaco's mailing address is P.O. Box {d['box']}, "
                 f"{d['city']}, {d['state']} {d['zip']}.",
            sources=[Source(url=e["url"], snapshot_id=e["snap"],
                            quote=e.get("quote") or "")])

    def _answer_founded(self, as_of):
        evs = self._events("founded", as_of)
        if not evs:
            return None
        e = evs[0]
        d = e["data"]
        return Answer(
            text=f"Dupaco was founded in {d['year']} by {d['by']}.",
            sources=[Source(url=e["url"], snapshot_id=e["snap"],
                            quote=e.get("quote") or "")])

    def _answer_counties(self, as_of):
        evs = self._events("counties", as_of)
        if not evs:
            return None
        e = evs[0]
        d = e["data"]
        return Answer(
            text=f"Dupaco serves {d['n']} counties throughout {d['area']}.",
            sources=[Source(url=e["url"], snapshot_id=e["snap"],
                            quote=e.get("quote") or "")])

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _refuse():
        return Answer(text="I don't know - this isn't listed on the website. "
                           "Please contact us.",
                      sources=[], confident=False)

    def _predates(self, as_of):
        """True when a date is before the earliest snapshot (2026-05-12):
        no evidence exists for it in the snapshots."""
        return bool(as_of and self.snaps and as_of < self.snaps[0][1])

    def _latest_snap(self):
        return self.snaps[-1][0] if self.snaps else None

    def _slice(self, as_of):
        """Facts visible as of a date: only the latest snapshot on/before it.
        With no date, only the newest snapshot. Older snapshots hold stale
        values (and stale pages live inside the newest snapshot too), so a
        current question must never see more than one snapshot's facts."""
        if as_of:
            ok = [s for s, t in self.snaps if t <= as_of]
            snap = ok[-1] if ok else (self.snaps[0][0] if self.snaps else None)
        else:
            snap = self.snaps[-1][0] if self.snaps else None
        return [f for f in self.facts if f.snap == snap]

    def _date_in_question(self, ql):
        # "last year" is relative to the evidence: the year before the
        # earliest snapshot's year. No snapshot covers it, so the predates
        # check refuses it rather than presenting a current value as last
        # year's.
        if "last year" in ql and self.snaps:
            year = int(self.snaps[0][1][:4]) - 1
            return f"{year}-12-31"
        # a full calendar date ("May 20, 2026", "2026-05-20", "5/20/2026")
        # is checked first: its month name must not be mistaken for a bare,
        # ambiguous month below. A model year never looks like this.
        d = _month_date(ql) or _parse_date(ql)
        if d:
            return d.isoformat()
        m = re.search(r"\b(20\d\d)-(\d{2})-(\d{2})\b", ql)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)),
                            int(m.group(3))).isoformat()
            except ValueError:
                pass
        # a bare month name with no year ("in May", "as of June") is
        # ambiguous: it could be any year's May. The caller treats the
        # AMBIGUOUS marker as a refusal, never as a current value.
        months = {"january": "01", "february": "02", "march": "03",
                  "april": "04", "may": "05", "june": "06", "july": "07",
                  "august": "08", "september": "09", "october": "10",
                  "november": "11", "december": "12"}
        if re.search(r"\b(?:as of|in)\s+(" + "|".join(months) +
                     r")\b(?!\s+20\d\d)", ql):
            return "AMBIGUOUS"
        m = re.search(r"as of\s+([a-z]+)\s+(20\d\d)", ql)
        months = {"january": "01", "february": "02", "march": "03", "april": "04",
                  "may": "05", "june": "06", "july": "07", "august": "08",
                  "september": "09", "october": "10", "november": "11", "december": "12"}
        if m and m.group(1) in months:
            return f"{m.group(2)}-{months[m.group(1)]}-28"
        # "in 2019" / "did Dupaco have in 2019": a past year the question
        # is about. Model years ("2025 model-year vehicle") are excluded:
        # they are not preceded by "in" and are followed by model/vehicle.
        m = re.search(r"\bin\s+([a-z]+)\s+(20\d\d)\b", ql)
        months = {"january": "01", "february": "02", "march": "03", "april": "04",
                  "may": "05", "june": "06", "july": "07", "august": "08",
                  "september": "09", "october": "10", "november": "11", "december": "12"}
        if m and m.group(1) in months:
            return f"{m.group(2)}-{months[m.group(1)]}-28"
        m = re.search(r"\bin\s+(19\d\d|20\d\d)\b", ql)
        if m:
            return f"{m.group(1)}-12-31"
        return None

    def _page_url(self, suffix):
        """An ingested page URL ending in suffix (the site's own pages,
        found in the snapshots — never written in code)."""
        for u in sorted(self.page_urls):
            if u.rstrip("/").endswith(suffix):
                return u
        return None

    def _join_url(self):
        """The join-portal link, taken from the links on the /about/join
        page whose anchor invites joining and whose target leaves the
        main site."""
        cands = []
        for anchor, href, page, snap, taken in self.links:
            if not page.rstrip("/").endswith("/about/join"):
                continue
            if not re.search(r"join|become a member", anchor, re.I):
                continue
            if href.split("/")[2:3] == page.split("/")[2:3]:
                continue
            cands.append((taken, href))
        if not cands:
            return None
        cands.sort()
        return cands[-1][1]

    def _page_quote(self, url, words, snap=None):
        """A verbatim sentence from an ingested page, containing all the
        given words — the supporting excerpt for a link answer, which has
        no table row to quote."""
        snap = snap or self._latest_snap()
        for (u, s, taken, auth, txt, toks, _ex) in self.chunks:
            if u != url or s != snap:
                continue
            low = txt.lower()
            if all(w.lower() in low for w in words):
                sents = re.split(r"(?<=[.!?])\s+", txt)
                for sent in sents:
                    if all(w.lower() in sent.lower() for w in words):
                        return _raw(sent)[:220]
                return _raw(txt)[:220]
        return ""

    def _answer_link(self, ql):
        if "certificate" in ql or re.search(r"\bcd\b", ql):
            page = self._page_url("/personal/savings-account")
            if page:
                url = page + "#rates"
                return Answer(text=f"See Dupaco's certificate rates at {url}",
                              sources=[Source(url=url,
                                              snapshot_id=self._latest_snap(),
                                              quote=self._page_quote(
                                                  page, ["certificate"]))])
        if "branch" in ql or "atm" in ql or "locator" in ql:
            page = self._page_url("/contact")
            if page:
                return Answer(text=f"The branch and ATM locator is at {page}",
                              sources=[Source(url=page,
                                              snapshot_id=self._latest_snap(),
                                              quote=self._page_quote(
                                                  page, ["branch"]))])
        if "join" in ql:
            href = self._join_url()
            if href:
                anchor = ""
                for a, h, pg, snap, taken in self.links:
                    if h == href:
                        anchor = a
                        break
                return Answer(text=f"Join Dupaco online at {href}",
                              sources=[Source(url=href,
                                              snapshot_id=self._latest_snap(),
                                              quote=anchor)])
        return None

    def _answer_routing(self, as_of):
        if self._predates(as_of):
            return _refuse_note(
                "no records from that year are in the snapshots.")
        evs = self._events("routing", as_of)
        if not evs:
            return None
        # the number agrees across the contact / checking / cash-management
        # pages; the cash-management page states it most directly
        # ("routing number" next to the wire and ACH instructions)
        pref = [e for e in evs
                if e["url"].rstrip("/").endswith("/business/cash-management")]
        e = pref[0] if pref else evs[0]
        return Answer(text=f"Dupaco's routing number is {e['data']['number']}.",
                      sources=[Source(url=e["url"], snapshot_id=e["snap"],
                                      quote=e["quote"])])

    def _answer_main_phone(self, as_of):
        """Dupaco's main phone number, from the /contact page (the same
        number the contact-update event carries). A branch phone question
        is handled by the branch handler, never here."""
        if self._predates(as_of):
            return _refuse_note(
                "no records from that year are in the snapshots.")
        evs = self._events("contact-update", as_of)
        if not evs:
            return None
        e = evs[0]
        phone = e["data"].get("phone")
        if not phone:
            return None
        return Answer(text=f"Dupaco's main phone number is {phone}.",
                      sources=[Source(url=e["url"], snapshot_id=e["snap"],
                                      quote=e.get("quote") or "")])

    def _answer_eligibility(self, as_of):
        evs = self._events("eligibility", as_of)
        if not evs:
            return None
        e = evs[0]
        states = e["data"].get("states") or []
        if not states or not e["data"].get("charter"):
            return None
        st = states[0] if len(states) == 1 else \
            ", ".join(states[:-1]) + " and " + states[-1]
        return Answer(
            text="People who live or work in Dupaco's charter area of counties in "
                 f"{st} are eligible to become Dupaco members.",
            sources=[Source(url=e["url"], snapshot_id=e["snap"],
                            quote=e.get("quote") or "")])

    def _answer_hometart(self, as_of):
        evs = self._events("hometart", as_of)
        if not evs:
            return None
        e = evs[0]
        amt = e["data"].get("amount")
        purpose = e["data"].get("purpose")
        if not amt:
            return None
        if purpose:
            text = (f"Home$tart provides up to ${amt} toward {purpose} "
                    "for first-time homebuyers.")
        else:
            text = f"Home$tart provides up to ${amt} for first-time homebuyers."
        return Answer(text=text,
                      sources=[Source(url=e["url"], snapshot_id=e["snap"],
                                      quote=e.get("quote") or "")])

    def _answer_contact_update(self, as_of):
        evs = self._events("contact-update", as_of)
        if not evs:
            return None
        e = evs[0]
        phone = e["data"].get("phone")
        if not phone:
            return None
        if e["data"].get("shine"):
            text = (f"Log in to your Shine account or call {phone} to update "
                    "your contact information.")
        else:
            text = f"Call {phone} to update your contact information."
        return Answer(text=text,
                      sources=[Source(url=e["url"], snapshot_id=e["snap"],
                                      quote=e.get("quote") or "")])

    # -- branch helpers ------------------------------------------------

    _BRANCH_NAME_STOP = {"the", "a", "an", "is", "are", "was", "were", "dupaco",
                         "of", "for", "in", "at", "to", "my", "your", "me",
                         "give", "can", "you", "please", "what", "address",
                         "lobby", "hours", "drive-up", "drive", "up", "do",
                         "does", "i", "we", "branch", "location", "locations",
                         "learning", "lab", "center", "centre", "office"}

    def _named_branch(self, q):
        """The branch a question names, as stemmed tokens, plus a city named
        after 'in'. ('Hillcrest branch' -> {'hillcrest'}; 'the Mineral Point
        branch' -> {'mineral', 'point'}; 'branch in Cedar Rapids' ->
        city 'Cedar Rapids'.) Returns (tokens, city) — either may be empty."""
        name = None
        m = re.search(r"([A-Za-z]+(?:\s+[A-Za-z]+){0,2})\s+(?:branch|office|location)\b",
                      q, re.I)
        if m:
            toks = {_stem(w) for w in re.findall(r"[A-Za-z]+", m.group(1))
                    if w.lower() not in self._BRANCH_NAME_STOP and len(w) > 1}
            if toks:
                name = toks
        city = None
        m2 = re.search(r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", q)
        if m2:
            city = m2.group(1)
        return name, city

    @staticmethod
    def _branch_tokens(addr, city):
        """Stemmed tokens of a branch record's address and city, minus the
        street-type word and the state/zip — the words that identify it."""
        words = re.findall(r"[A-Za-z]+", (addr or "") + " " + (city or ""))
        drop = {"ln", "lane", "rd", "road", "st", "street", "ave", "avenue",
                "dr", "drive", "way", "blvd", "hwy", "ste", "ia", "il", "wi",
                "mn", "suite"}
        return {_stem(w) for w in words if w.lower() not in drop}

    def _match_branch(self, name_toks, city):
        """Directory entries, then 'located at' records, whose tokens
        contain every token of the named branch (or whose city matches a
        named city). Returns (record, kind) or (None, None). A record is a
        (city, address, zip, url, snap, taken, quote) tuple."""
        cands = list(self.branches) + list(self.located)
        if name_toks:
            for rec in cands:
                toks = self._branch_tokens(rec[1], rec[0])
                if name_toks <= toks:
                    return rec, "dir" if rec in self.branches else "located"
        if city:
            recs = [r for r in cands if r[0].lower() == city.lower()]
            if len(recs) == 1:
                return recs[0], "dir" if recs[0] in self.branches else "located"
        return None, None

    def _answer_branch(self, ql, as_of, q=None):
        if self._predates(as_of):
            return _refuse_note("that information is not available for that date.")
        q = q or ""
        name_toks, city = self._named_branch(q)
        named = bool(name_toks or city)
        if "phone" in ql and named:
            return self._answer_branch_phone(name_toks, city, ql)
        if "address" in ql or re.search(r"\bwhere is\b|\blocated\b", ql):
            if named:
                rec, _kind = self._match_branch(name_toks, city)
                if rec is None:
                    return _refuse_note(
                        "that branch is not listed on the website.")
                _, addr, z, url, snap, _taken, quote = rec
                disp = addr.replace(" Ln.", " Lane").replace(" Ln,", " Lane,")
                if z and z not in disp:
                    text = f"{disp}, {z}."
                elif not z and rec[0]:
                    text = f"{disp}, {rec[0]}."
                else:
                    text = f"{disp}."
                return Answer(text=text,
                              sources=[Source(url=url, snapshot_id=snap,
                                              quote=quote or "")])
            # an unnamed "branch address" question with no city cannot be
            # answered from one entry; fall through to the prose fallback
            return None
        if "hours" in ql or re.search(r"\bwhen\b[^?]*\bopen\b|\bwhat time\b", ql):
            if named:
                # branch-specific hours only: the segment (or its page URL)
                # must name the branch. General hours are never a substitute
                # for a named branch's hours.
                for kind, seg, url, snap, _ in reversed(self.hours):
                    hay = (seg + " " + url).lower()
                    toks = name_toks or {_stem(w) for w in (city or "").split()}
                    if toks and all(t in _stems(hay) for t in toks):
                        return Answer(text=self._hours_text(seg, kind),
                                      sources=[Source(url=url, snapshot_id=snap,
                                                      quote=_raw(seg))])
                return _refuse_note(
                    "hours for that branch are not listed on the website.")
            # general hours — rendered from the extracted hours segments
            # (the learning-lab release's typical schedule), not a fixed
            # string; the branch directory is cited too, because hours vary
            # by branch and that page is where a member finds their own
            lobby = self._best_hours("lobby")
            drive = self._best_hours("drive-up")
            # a drive-up question gets the drive-up schedule only
            if drive and re.search(r"drive[\s\-]?(up|thru|through)", ql) \
                    and "lobby" not in ql:
                srcs = [Source(url=drive[2], snapshot_id=drive[3],
                               quote=_raw(drive[1]))]
                contact = self._page_url("/contact")
                if contact:
                    self._add_srcs(srcs, [Source(
                        url=contact, snapshot_id=self._latest_snap(),
                        quote=self._page_quote(contact, ["branch"]))])
                return Answer(
                    text=f"{self._hours_text(drive[1], 'drive-up')}. "
                         "Drive-up hours can vary by branch.",
                    sources=srcs)
            if lobby and drive:
                lobby_txt = self._hours_text(lobby[1], "lobby")
                drive_txt = self._hours_text(drive[1], "drive-up")
                srcs = []
                self._add_srcs(srcs, [Source(url=lobby[2], snapshot_id=lobby[3],
                                              quote=_raw(lobby[1]))])
                self._add_srcs(srcs, [Source(url=drive[2], snapshot_id=drive[3],
                                              quote=_raw(drive[1]))])
                contact = self._page_url("/contact")
                if contact:
                    self._add_srcs(srcs, [Source(
                        url=contact, snapshot_id=self._latest_snap(),
                        quote=self._page_quote(contact, ["branch"]))])
                return Answer(
                    text=f"Lobby hours vary by branch. {lobby_txt}. {drive_txt}. "
                         "Which branch did you have in mind?",
                    sources=srcs)
        return None

    def _answer_branch_phone(self, name_toks, city, ql):
        """A branch's own published phone number (from a release that states
        it). If no branch-specific number is published, refuse — never
        invent one, and never pass off the main number as the branch's."""
        evs = self._events("branch-phone", None)
        for e in evs:
            btoks = {_stem(w) for w in re.findall(r"[A-Za-z]+", e["data"]["branch"])
                     if w.lower() not in self._BRANCH_NAME_STOP}
            if name_toks and name_toks <= btoks:
                return Answer(
                    text=f"The phone number for the {e['data']['branch']} branch "
                         f"is {e['data']['phone']}.",
                    sources=[Source(url=e["url"], snapshot_id=e["snap"],
                                    quote=e.get("quote") or "")])
        return _refuse_note(
            "a phone number for that branch is not listed on the website.")

    def _best_hours(self, kind):
        cands = [h for h in self.hours if h[0] == kind]
        if not cands:
            return None
        # hours live in the newsroom learning-lab releases ("Branch hours
        # are:"); prefer those over stray mentions on other pages, then the
        # most recent snapshot, then the most complete entry
        def rank(h):
            lab = 1 if ("learning-lab" in h[2] or "newsroom" in h[2]) else 0
            return (lab, h[4], len(h[1]))
        cands.sort(key=rank, reverse=True)
        return cands[0]

    @staticmethod
    def _hours_text(seg, kind):
        # markdown markers off first: "**ABOUT DUPACO" must still be cut
        seg = _clean(seg)
        if kind == "lobby":
            seg = re.split(r"Drive-up", seg, flags=re.I)[0]
        # the captured window runs past the hours into the boilerplate
        # ("ABOUT DUPACO COMMUNITY CREDIT UNION: ..."); cut there
        seg = re.split(r"\s+ABOUT\s+[A-Z]", seg)[0]
        return _clean(seg)

    def _answer_branch_count(self, as_of=None):
        if self._predates(as_of):
            return _refuse_note("that information is not available for that date.")
        c = self._current_count(as_of)
        if c is None:
            return None
        return Answer(text=f"Dupaco has {c[0]} branch office locations.",
                      sources=[Source(url=c[1], snapshot_id=c[2],
                                      quote=c[4] or "")])

    def _current_count(self, as_of=None):
        """The branch count from the most recently published release in the
        snapshot visible at as_of. Old releases stay on the site with the
        count they were written with (19, 20, 23, 24...); the newest
        dateline is the current figure. With as_of, releases published after
        that date are not yet known."""
        if not self.counts:
            return None
        if as_of:
            ok = [s for s, t in self.snaps if t <= as_of]
            snap = ok[-1] if ok else self.snaps[0][0]
        else:
            snap = self.snaps[-1][0]
        pool = [c for c in self.counts if c[2] == snap] or list(self.counts)
        if as_of:
            try:
                d = date.fromisoformat(as_of)
                dated = [c for c in pool if c[5] is None or c[5] <= d]
                pool = dated or pool
            except ValueError:
                pass
        return max(pool, key=lambda c: (c[5] or date(1900, 1, 1), c[3]))

    def _answer_hyvee(self, ql, as_of):
        evs = self._events("hyvee", as_of)
        if not evs:
            return None
        e = evs[0]
        close = e["data"].get("close")
        branches = e["data"].get("branches") or []
        src = [Source(url=e["url"], snapshot_id=e["snap"],
                      quote=e.get("quote") or "")]
        if ("where" in ql or "go" in ql) and len(branches) >= 2:
            return Answer(text="Hy-Vee branch members can use the "
                               f"{branches[0]} and {branches[1]} branches "
                               "for full-service banking.",
                          sources=src)
        if close is None:
            return None
        when = _fmt_date(close)
        if as_of:
            try:
                as_of_d = date.fromisoformat(as_of)
            except ValueError:
                as_of_d = None
            # before the closing date the branch was still open
            if as_of_d and as_of_d < close:
                return Answer(text="Yes - the Hy-Vee branch is still open. "
                                   f"It is scheduled to close {when}.",
                              sources=src)
        return Answer(text=f"No - the Hy-Vee branch was scheduled to close {when}.",
                      sources=src)

    def _answer_forbes(self, as_of):
        evs = self._events("forbes", as_of)
        if not evs:
            return self._refuse()
        # newest release announced by the as-of date: each release carries
        # its own publication date, so a date before June 2024 can only
        # see the "fourth" release, never the "fifth"
        cands = []
        for e in evs:
            pub = e.get("pub")
            if as_of and pub:
                try:
                    if pub > date.fromisoformat(as_of):
                        continue
                except ValueError:
                    pass
            cands.append(e)
        if not cands:
            return self._refuse()
        best = max(cands, key=lambda e: (e.get("pub") or date(1900, 1, 1),
                                         e["taken"]))
        ordinal = best["data"]["ordinal"]
        num = ORDINAL_NUM.get(ordinal, ordinal)
        return Answer(
            text=f"Forbes has recognized Dupaco {num} times as a top (Best-in-State) "
                 f"credit union; the latest release says it was the {ordinal} time.",
            sources=[Source(url=best["url"], snapshot_id=best["snap"],
                            quote=best.get("quote") or "")])


    # -- structured rate lookup ------------------------------------------

    # -- structured rate lookup ------------------------------------------

    # the phrase stop list is the generic-prose set: only genuinely
    # distinctive name words may restrict the product ("provide", "dupaco",
    # "month", "earn", "last" are not product names)
    _PHRASE_STOP = set(GENERIC_PROSE)

    _PRODUCT_WORDS = ["money market", "certificate", "savings", "checking",
                      "mortgage", "heloc", "loan", "card", "visa", "hsa",
                      "ira", "vehicle"]

    def _phrase_before_product(self, ql):
        if "you name it" in ql:
            return {"name"}
        """The descriptive words immediately before the product-type word in
        the question ('premium' in 'Premium Savings', 'health' in 'Health
        Savings Account', 'high-yield' in 'High-Yield Savings', 'jumbo' in
        '17-month jumbo certificate'). Stop words and bare numbers (terms,
        model years) are dropped — they are qualifiers, not names. Returns a
        set of stems, or an empty set when the question names no product
        phrase."""
        for pw in self._PRODUCT_WORDS:
            m = re.search(r"([a-z0-9][a-z0-9 \-\']*?)\s+" + re.escape(pw) + r"\b",
                          ql)
            if not m:
                continue
            seg = m.group(1)
            # only the last three words before the product word count.
            # Keep alphanumeric name tokens ("1st" in "1st Rate Checking")
            # but drop bare numbers (terms, model years) — they are
            # qualifiers, not names.
            words = re.findall(r"[a-z0-9]+", seg)[-3:]
            stems = {_stem(w) for w in words
                     if len(w) > 1 and not w.isdigit()
                     and w not in self._PHRASE_STOP
                     and w not in ("model", "vehicle", "auto", "car", "cars",
                                   "truck", "suv")}
            if stems:
                return stems
            return self._phrase_after_product(ql, m)
        return set()

    def _phrase_after_product(self, ql, m):
        """A distinctive name word immediately AFTER the product-type word
        ("Plus" in "Money Market Plus account"): a different product whose
        name extends past the type word. Generic followers (account, card,
        loan, a rate word, a preposition) carry no name information."""
        after = ql[m.end():]
        mm = re.match(r"\s*([a-z0-9]+)", after)
        if not mm:
            return set()
        w = mm.group(1)
        if (len(w) > 1 and not w.isdigit()
                and w not in self._PHRASE_STOP
                and w not in ("model", "vehicle", "auto", "account", "card",
                              "loan", "rate", "apy", "apr", "dividend",
                              "interest", "minimum", "balance", "fee",
                              "tier", "tiers", "option", "options", "type",
                              "types")):
            return {_stem(w)}
        return set()

    @staticmethod
    def _fact_phrase_hay(f):
        """The text a product phrase is matched against: the row's display
        name, its section, the product key, and the kind qualifier."""
        parts = [f.display, f.section or "", f.product,
                 f.quals.get("kind") or ""]
        return _stems(" ".join(parts))

    def _answer_rate(self, ql, facts, as_of):
        prod = self._product_from_question(ql)
        if not prod:
            return None
        # attribute questions (minimum / fee / credit score) are answered
        # from footnotes and table qualifiers, never from a rate row — a
        # minimum question answered with an APY is a wrong answer. But a
        # question that asks for the rate AND the minimum ("what is the APY
        # ... and its minimum") goes through the rate path, which appends
        # the minimum to the rate render.
        asks_rate = bool(re.search(r"\b(apys?|aprs?|rates?|interest|dividends?)\b", ql))
        if re.search(r"\b(fee|credit score)\b", ql) or (
                re.search(r"\b(minimum|min)\b", ql) and not asks_rate) or (
                # "minimum balance to earn interest" is a minimum question,
                # not a rate question: "earn interest" names no rate metric
                re.search(r"\b(minimum|min)\b", ql) and "earn" in ql
                and not re.search(r"\b(apys?|aprs?|dividends?)\b", ql)):
            a = self._answer_attribute(ql, facts, prod, as_of)
            if a is not None:
                return a
        # the metrics the question actually names ("rate and APR" names two)
        metrics_named = set()
        if "apy" in ql:
            metrics_named.add("apy")
        if "apr" in ql:
            metrics_named.add("apr")
        if "dividend" in ql:
            metrics_named.add("div")
        # "dividend rate" is one metric (div): the word "rate" inside it is
        # not a second, APY, request — strip it before looking for "rate"
        if re.search(r"\b(rate|interest)\b",
                     re.sub(r"dividend\s+rates?", "", ql)):
            metrics_named.add("rate")
        metric = None
        if len(metrics_named) == 1:
            metric = next(iter(metrics_named))
        elif not metrics_named and prod in ("certificate", "share savings",
                                            "bright savings", "money market",
                                            "checking", "savings", "hsa", "ira"):
            metric = "apy"
        multi_metric = len(metrics_named) > 1
        if (metric == "rate" and prod in ("certificate", "share savings",
                                          "bright savings", "money market",
                                          "checking", "savings", "hsa", "ira")):
            metric = "apy"
            multi_metric = False
        # qualifiers from the question — every named one is a hard filter:
        # a qualifier with zero matching rows means the requested thing does
        # not exist, and the answer is a refusal, not the nearest row
        terms = sorted({int(n) for n in re.findall(r"(\d+)[\s\-]?month", ql)})
        if not terms:
            for w, n in WORD_NUM.items():
                if re.search(rf"\b{w}\s+month", ql):
                    terms.append(n)
        year = None
        m = re.search(r"(20\d\d)[\s\-]?(?:model[\s\-]?year|vehicle|car|truck"
                      r"|suv|motorcycle|model)\b", ql)
        if m:
            year = int(m.group(1))
        # a loan or deposit term in years ("5-year", "3 years"); one or two
        # digits only, so a model year (2025) is never read as a term
        yr = None
        m = re.search(r"(?<!\d)(\d{1,2})[\s\-]?years?\b", ql)
        if m:
            yr = int(m.group(1))
        # auto, personal, boat/RV loans and certificates publish their terms
        # in months: "5-year auto loan" is the 60-month row. Mortgages and
        # home equity publish theirs in years and keep the year qualifier.
        if yr is not None and not terms and prod in (
                "auto loan", "personal loan", "recreational loan",
                "certificate"):
            terms = [yr * 12]
            yr = None
        balance = None
        bm = re.findall(r"\$([\d,]+)", ql)
        if bm and ("balance" in ql or "between" in ql or "savings" in ql or "checking" in ql):
            balance = float(bm[0].replace(",", ""))
        amount = None
        if bm and ("mortgage" in ql or "loan" in ql):
            amount = float(bm[0].replace(",", ""))
        jumbo = "jumbo" in ql
        not_jumbo = bool(re.search(r"\bnot\b[^.]{0,15}?\bjumbo\b|\bnon[\s\-]?jumbo\b|\bstandard\b", ql))
        biz = "business" in ql
        card_name = None
        if "platinum" in ql:
            card_name = "platinum"
        elif "rewards" in ql:
            card_name = "rewards"
        phrase = self._phrase_before_product(ql)

        cands = [f for f in facts if f.product == prod]
        if not cands:
            return self._refuse()
        # the product phrase must match the row's own name: "High-Yield"
        # is not Launchpad, "Health Savings" is not You Name It Savings,
        # "Premium Savings" is not Share Savings
        if phrase:
            p2 = [f for f in cands if phrase <= self._fact_phrase_hay(f)]
            if not p2:
                return self._refuse()
            cands = p2
        # the introductory balance-transfer row is a trap: only use it when
        # the question actually asks about an intro / balance-transfer rate
        if "intro" not in ql and "balance transfer" not in ql:
            keep = [f for f in cands
                    if "intro" not in f.display.lower()
                    and "balance transfer" not in f.display.lower()]
            if keep:
                cands = keep
        if card_name:
            c2 = [f for f in cands if card_name in f.display.lower()]
            if not c2:
                return self._refuse()
            cands = c2
        if biz:
            cands = [f for f in cands if f.quals.get("biz") or "business" in f.display.lower()] or cands
        if metric:
            m2 = [f for f in cands if f.metric == metric]
            if m2:
                cands = m2
        if terms:
            for t in terms:
                if not any(f.quals.get("term") == t for f in cands):
                    return self._refuse()
            cands = [f for f in cands if f.quals.get("term") in terms]
        qualified = bool(terms)
        if yr is not None:
            y2 = [f for f in cands if f.quals.get("yr") == yr]
            if not y2:
                return self._refuse()
            cands = y2
            qualified = True
        if year is not None:
            y2 = [f for f in cands
                  if f.quals.get("years") and f.quals["years"][0] <= year <= f.quals["years"][1]]
            if not y2:
                return self._refuse()
            cands = y2
            qualified = True
        if balance is not None:
            tiered = [f for f in cands if f.quals.get("balance")]
            if tiered:
                b2 = [f for f in tiered
                      if f.quals["balance"][0] <= balance <= f.quals["balance"][1]]
                if not b2:
                    return self._refuse()
                cands = b2
                qualified = True
        if amount is not None:
            tiered = [f for f in cands if f.quals.get("amount")]
            if tiered:
                a2 = [f for f in tiered
                      if f.quals["amount"][0] <= amount <= f.quals["amount"][1]]
                if not a2:
                    return self._refuse()
                cands = a2
                qualified = True
        if jumbo and "special" not in ql:
            j2 = [f for f in cands if f.quals.get("kind") == "jumbo"]
            if not j2:
                return self._refuse()
            cands = j2
        if "special" in ql and "jumbo" not in ql:
            s2 = [f for f in cands if f.quals.get("kind") == "special"]
            if not s2:
                return self._refuse()
            cands = s2
        if not_jumbo:
            j2 = [f for f in cands if f.quals.get("kind") != "jumbo"]
            if j2:
                cands = j2
        if not cands:
            # a rate was asked for and nothing matches: do not guess
            if re.search(r"\b(rate|apr|apy|interest|dividend)\b", ql):
                return self._refuse()
            return None
        # a degenerate range (lo == hi) is a parse artifact from a page that
        # states only one bound (Launchpad "$1,000" on the retirement page
        # vs "$5-$1,000" on the rate pages): where the same value is also
        # published with a real range, drop the artifact
        def _degenerate(f):
            for key in ("balance", "amount"):
                rg = f.quals.get(key)
                if rg and rg[0] == rg[1]:
                    return True
            return False
        by_val = {}
        for f in cands:
            by_val.setdefault((f.product, f.metric, f.value,
                               f.display.lower()), []).append(f)
        cands = [f for grp in by_val.values()
                 for f in ([g for g in grp if not _degenerate(g)] or grp)]
        # keep the best source per (product, metric, qualifier) — authority,
        # then newest effective date, then newest snapshot. The key deliberately
        # excludes the value: the same tier is quoted by several pages (the
        # /rates hub, the product page, a stale guidance page), and only the
        # best-authority one may survive into the answer. Credit cards are the
        # exception: the card name is the only thing that tells them apart.
        cands.sort(key=lambda f: (f.auth, f.eff or date(1900, 1, 1), f.taken),
                   reverse=True)
        # de-duplicate identical facts from the /rates hub vs the product page
        seen = set()
        rows = []
        for f in cands:
            bal = f.quals.get("balance")
            if bal and bal[0] == bal[1]:
                bal = None
            amt = f.quals.get("amount")
            if amt and amt[0] == amt[1]:
                amt = None
            k = (f.product, f.metric, f.quals.get("term"), f.quals.get("yr"),
                 f.quals.get("years"), bal, amt, f.quals.get("kind"))
            if f.product == "credit card":
                k = k + (f.display.lower(),)
            if k in seen:
                continue
            seen.add(k)
            rows.append(f)
        if year is not None:
            best_by_term = {}
            for f in rows:
                k = (f.product, f.metric, f.quals.get("term"),
                     f.quals.get("yr"), f.quals.get("balance"),
                     f.quals.get("amount"), f.quals.get("kind"))
                cur = best_by_term.get(k)
                key = (f.auth, f.eff or date(1900, 1, 1), f.taken)
                if cur is None or key > (cur.auth, cur.eff or date(1900, 1, 1),
                                         cur.taken):
                    best_by_term[k] = f
            rows = sorted(best_by_term.values(),
                          key=lambda f: (f.auth, f.eff or date(1900, 1, 1),
                                         f.taken), reverse=True)
        # several terms named ("60-month and 84-month"): one row per term,
        # not just the first
        if len(terms) > 1:
            return self._render_list(
                prod, rows, facts,
                with_min=bool(re.search(r"\b(minimum|min)\b", ql)))
        # a bare boat/RV question gets the main amount-based table, not the
        # separate used-model-year table on the same page
        if (prod == "recreational loan" and not terms and year is None
                and amount is None):
            amt_rows = [f for f in rows if f.quals.get("amount")]
            if amt_rows:
                rows = amt_rows
        # "compare" asks for each entity side by side, with its minimum
        if "compare" in ql:
            return self._render_list(prod, rows, facts, with_min=True)
        # "highest ... and lowest ..." asks for both extremes
        if re.search(r"\bhighest\b", ql) and re.search(r"\blowest\b", ql):
            vals = sorted(rows, key=lambda f: f.value[1] if isinstance(f.value, tuple)
                          else f.value)
            hi = self._render_single(vals[-1], facts, ql)
            lo = self._render_single(vals[0], facts, ql)
            srcs = []
            self._add_srcs(srcs, hi.sources)
            self._add_srcs(srcs, lo.sources)
            return Answer(text=f"Highest: {hi.text} Lowest: {lo.text}.",
                          sources=srcs)
        # "highest / top / max" asks for one row: the largest value, with its
        # qualifier (e.g. the balance tier) in the text — scoped to the
        # already-filtered product set, never across products
        if re.search(r"\b(highest|top|maximum|max|best)\b", ql):
            rows.sort(key=lambda f: f.value[1] if isinstance(f.value, tuple)
                      else f.value, reverse=True)
            return self._render_single(rows[0], facts, ql)
        # several metrics named ("rate and APR"): render the sibling values
        # for each tier instead of filtering to one metric
        if multi_metric:
            return self._render_multi_metric(ql, prod, rows, facts)
        return self._render_rate_rows(ql, prod, rows, metric, qualified, facts)

    def _render_multi_metric(self, ql, prod, rows, facts):
        """One line per tier, each naming every requested metric for that
        tier ('6.75% interest rate and 6.88% APR'), so a question asking for
        both never drops one."""
        groups = {}
        for f in rows:
            k = (f.product, f.quals.get("term"), f.quals.get("yr"),
                 f.quals.get("years"), f.quals.get("balance"),
                 f.quals.get("amount"), f.quals.get("kind"), f.display.lower())
            groups.setdefault(k, []).append(f)
        lines = []
        srcs = []
        for _k, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
            parts = []
            for f in sorted(grp, key=lambda g: g.metric):
                word = {"apr": "APR", "apy": "APY", "div": "dividend rate",
                        "rate": "interest rate"}[f.metric]
                parts.append(f"{f.disp} {word}")
            label = grp[0].display
            ctx = self._qualifier_text(grp[0])
            lines.append(f"{label}{ctx}: " + " and ".join(parts) + ".")
            for f in grp:
                self._add_srcs(srcs, self._sources_for(f, facts))
        return Answer(text=" ".join(lines), sources=srcs)

    def _answer_attribute(self, ql, facts, prod, as_of):
        """Minimums, fees and credit-score requirements — from the table
        qualifiers and footnote conditions, never from a rate row. A
        question about a minimum that is answered with an APY is wrong even
        if the number is real. Returns None only when the question is not
        really an attribute question; a named attribute with no evidence is
        a refusal."""
        # -- fees and credit-score requirements: footnote conditions -------
        if "fee" in ql or "credit score" in ql:
            want_fee = None
            if "cash advance" in ql:
                want_fee = ["cash", "advance"]
            elif "annual" in ql:
                want_fee = ["annual"]
            elif "balance transfer" in ql:
                want_fee = ["balance", "transfer"]
            elif "late" in ql:
                want_fee = ["late", "payment"]
            elif "foreign" in ql:
                want_fee = ["foreign", "transaction"]
            elif "returned" in ql:
                want_fee = ["returned", "payment"]
            conds = [c for c in self.conditions if c["product"] == prod]
            if as_of:
                sl = self._slice(as_of)
                if sl:
                    conds = [c for c in conds if c["snap"] == sl[0].snap]
            else:
                conds = [c for c in conds if c["snap"] == self._latest_snap()]
            if "credit score" in ql:
                hits = [c for c in conds if c["kind"] == "credit_score"]
                if hits:
                    c = max(hits, key=lambda x: (x["auth"], x["taken"]))
                    return Answer(
                        text=f"Qualification requires a {c['value']} credit score "
                             "or higher.",
                        sources=[Source(url=c["url"], snapshot_id=c["snap"],
                                        quote=c["text"])])
                return _refuse_note(
                    "no credit-score requirement is listed for that product.")
            if want_fee:
                hits = [c for c in conds
                        if c["kind"] == "fee" and c["fee_words"] == want_fee]
                # a named card scopes the fee to that card's footnote: the
                # Platinum and Business Rewards late fees differ ($15 vs
                # $35), so a card question must not be answered from the
                # other card's footnote
                card = None
                if "business rewards" in ql:
                    card = "business rewards"
                elif "business platinum" in ql:
                    card = "business platinum"
                elif "platinum" in ql:
                    card = "platinum"
                elif "rewards" in ql:
                    card = "rewards"
                if card:
                    c2 = [c for c in hits
                          if card in (c.get("display") or "").lower()]
                    if c2:
                        hits = c2
                # a personal-card question is answered from the personal
                # card's footnote: "Business Platinum Visa" contains
                # "platinum" but its late fee ($35) is not the Platinum
                # Visa's ($15)
                if "business" not in ql:
                    nb = [c for c in hits
                          if "business" not in (c.get("display") or "").lower()]
                    if nb:
                        hits = nb
                if hits:
                    c = max(hits, key=lambda x: (x["auth"], x["taken"]))
                    clause = c["text"]
                    if c["value"] == "none":
                        label = " ".join(want_fee)
                        text = f"There is no {label} fee."
                    else:
                        text = clause[0].upper() + clause[1:] + "."
                        text = text.replace("- $", " - $")
                    return Answer(text=text,
                                  sources=[Source(url=c["url"],
                                                  snapshot_id=c["snap"],
                                                  quote=clause)])
                return _refuse_note(
                    "that fee is not listed for that product.")
            return _refuse_note(
                "that fee is not listed for that product.")
        # -- minimums -------------------------------------------------------
        if re.search(r"\b(minimum|min)\b", ql):
            # the same product's facts in the visible slice, plus the
            # product-phrase filter, so a jumbo minimum is never answered
            # with a standard minimum (and vice versa)
            cands = [f for f in facts if f.product == prod]
            phrase = self._phrase_before_product(ql)
            if phrase:
                p2 = [f for f in cands if phrase <= self._fact_phrase_hay(f)]
                if p2:
                    cands = p2
            term = None
            m = re.search(r"(\d+)[\s\-]?month", ql)
            if m:
                term = int(m.group(1))
            if term is not None:
                t2 = [f for f in cands if f.quals.get("term") == term]
                if not t2:
                    return _refuse_note(
                        "that certificate term is not offered.")
                cands = t2
            if "jumbo" in ql:
                j2 = [f for f in cands if f.quals.get("kind") == "jumbo"]
                if not j2:
                    return _refuse_note(
                        "that jumbo certificate is not offered.")
                cands = j2
            if "special" in ql and "jumbo" not in ql:
                s2 = [f for f in cands if f.quals.get("kind") == "special"]
                if not s2:
                    return _refuse_note(
                        "that special certificate is not offered.")
                cands = s2
            if not cands:
                return self._refuse()
            to_open = "open" in ql or "opening" in ql or "start" in ql
            to_earn = "earn" in ql
            # "minimum to open": the Min. to Open column, else Min. to Earn
            if to_open:
                vals = [f.quals.get("min_open") for f in cands
                        if f.quals.get("min_open") is not None]
                if not vals:
                    vals = [f.quals.get("min_earn") for f in cands
                            if f.quals.get("min_earn") is not None]
                if vals:
                    v = min(vals)
                    return self._min_answer(ql, cands, v, "to open")
                conds = [c for c in self.conditions
                         if c["product"] == prod and c["kind"] == "min_open"]
                if conds:
                    c = max(conds, key=lambda x: (x["auth"], x["taken"]))
                    return Answer(text=f"The minimum amount required to open "
                                       f"is ${_fmt_money(c['value'])}.",
                                  sources=[Source(url=c["url"],
                                                  snapshot_id=c["snap"],
                                                  quote=c["text"])])
                return _refuse_note(
                    "no minimum opening amount is listed for that product.")
            # "minimum balance to earn": the lowest balance at which the
            # product earns anything (the lowest tier with a non-zero rate)
            if to_earn:
                earning = [f for f in cands
                           if (f.value[1] if isinstance(f.value, tuple)
                               else f.value) > 0]
                if earning:
                    mins = [f.quals.get("min_earn") for f in earning
                            if f.quals.get("min_earn") is not None]
                    if mins:
                        return self._min_answer(ql, cands, min(mins),
                                                "to earn interest")
                    bals = [f.quals["balance"][0] for f in earning
                            if f.quals.get("balance")]
                    if bals:
                        return self._min_answer(ql, cands, min(bals),
                                                "to earn interest")
                return _refuse_note(
                    "no minimum balance to earn is listed for that product.")
            # a bare "minimum" question: the opening amount if the table has
            # one, else the earn minimum
            vals = [f.quals.get("min_open") for f in cands
                    if f.quals.get("min_open") is not None]
            if vals:
                return self._min_answer(ql, cands, min(vals), "")
            vals = [f.quals.get("min_earn") for f in cands
                    if f.quals.get("min_earn") is not None]
            if vals:
                return self._min_answer(ql, cands, min(vals), "")
            return _refuse_note(
                "no minimum is listed for that product.")
        return None

    def _min_answer(self, ql, cands, value, purpose):
        """Render a minimum answer: the amount, the product it belongs to,
        and the table row as the citation."""
        pool = [x for x in cands
                if x.quals.get("min_open") == value
                or x.quals.get("min_earn") == value] or cands
        # a personal product question is answered with the personal row:
        # the business twin carries the same minimum but the wrong name
        if "business" not in ql:
            nb = [x for x in pool if "business" not in x.display.lower()]
            if nb:
                pool = nb
        f = pool[0]
        amt = _fmt_money(value)
        label = f.display
        suffix = f" {purpose}" if purpose else ""
        return Answer(
            text=f"The minimum{suffix} for {label} is ${amt}.",
            sources=self._sources_for(f, None))

    def _sources_for(self, f, facts):
        """Sources for one fact: itself, plus an agreeing fact for the same
        tier from another page (product page + /rates hub), so the answer is
        backed by two sources that say the same thing. Among agreeing pages
        prefer the most authoritative — the product's own rate page. Every
        source carries the verbatim table row as its quote: a citation
        without a supporting excerpt is a pointer, not evidence."""
        srcs = [Source(url=f.url, snapshot_id=f.snap, quote=f.row or "")]
        if facts:
            cands = [g for g in facts
                     if g.url != f.url and g.snap == f.snap
                     and g.product == f.product and g.metric == f.metric
                     and g.value == f.value and g.quals == f.quals]
            if cands:
                best = max(cands, key=lambda g: g.auth)
                srcs.append(Source(url=best.url, snapshot_id=best.snap,
                                   quote=best.row or ""))
        return srcs

    @staticmethod
    def _add_srcs(srcs, new):
        for s in new:
            if s.url not in [x.url for x in srcs]:
                srcs.append(s)

    def _render_rate_rows(self, ql, prod, rows, metric, qualified=False,
                          facts=None):
        with_min = bool(re.search(r"\b(minimum|min)\b", ql))
        if prod == "recreational loan" and not qualified:
            return self._render_list(prod, rows, facts, with_min=with_min)
        # a tier-qualified question gets its single row; an open question gets
        # the list (so "highest APY" or "checking rates" still shows the value)
        # a mortgage question that names only the term still has several amount
        # tiers, so it gets the list too
        if prod == "certificate":
            # term named without "jumbo": show both the special and the jumbo
            # row for that term; no term named: show the full APY list
            if not qualified or "jumbo" not in ql:
                return self._render_list(prod, rows, facts, with_min=with_min)
        # an auto question that names the model year but not the term has
        # one row per term: list them all rather than pick one term silently
        if prod == "auto loan" and len({f.quals.get("term") for f in rows}) > 1:
            return self._render_list(prod, rows, facts, with_min=with_min)
        if qualified and prod == "mortgage":
            tiers = {(f.quals.get("amount"), f.quals.get("yr")) for f in rows}
            if len(tiers) > 1:
                return self._render_list(prod, rows, facts, with_min=with_min)
        if qualified:
            return self._render_single(rows[0], facts, ql)
        if prod == "credit card" and "platinum" not in ql and "rewards" not in ql:
            return self._render_list(prod, rows, facts, with_min=with_min)
        if prod == "mortgage":
            return self._render_list(prod, rows, facts, with_min=with_min)
        if prod in ("checking", "share savings", "savings"):
            return self._render_list(prod, rows, facts, with_min=with_min)
        f = rows[0]
        return self._render_single(f, facts, ql)

    def _render_single(self, f, facts=None, ql=None):
        metric_word = {"apr": "APR", "apy": "APY", "div": "dividend rate",
                       "rate": "rate"}[f.metric]
        val = f.disp
        label = f.display
        ctx = self._qualifier_text(f)
        srcs = self._sources_for(f, facts)
        if f.product == "auto loan":
            yrs = f.quals.get("years")
            term = f.quals.get("term")
            yr_txt = f"{yrs[0]}-{str(yrs[1])[2:]} model-year" if yrs else ""
            return Answer(
                text=f"The {term}-month auto loan APR for {yr_txt} vehicles is {val}.",
                sources=srcs,
                conflict_note=None)
        # a question that asks for the rate AND the minimum gets both
        min_txt = ""
        if ql and re.search(r"\b(minimum|min)\b", ql):
            mv = f.quals.get("min_open") or f.quals.get("min_earn")
            if mv is not None:
                min_txt = f" (minimum ${_fmt_money(mv)})"
        return Answer(text=f"The {label}{ctx} is {val} {metric_word}{min_txt}.",
                      sources=srcs)

    @staticmethod
    def _qualifier_text(f):
        q = f.quals
        bits = []
        if q.get("balance"):
            lo, hi = q["balance"]
            if hi == float("inf"):
                bits.append(f"for balances of ${lo:,.0f} and up")
            else:
                bits.append(f"for balances of ${lo:,.0f}-${hi:,.2f}".rstrip("0").rstrip("."))
        if q.get("term"):
            bits.append(f"({q['term']}-month)")
        if q.get("amount"):
            lo, hi = q["amount"]
            if f"${lo:,.0f}" in (f.display or ""):
                pass  # the display already is the amount range
            elif lo == 0:
                bits.append(f"for loans up to ${hi:,.0f}")
            else:
                bits.append(f"for loans of ${lo:,.0f}-${hi:,.0f}")
        return (" " + " ".join(bits)) if bits else ""

    def _render_list(self, prod, rows, facts=None, with_min=False):
        if prod == "credit card":
            order = ["Platinum Visa", "Rewards Visa", "Business Platinum Visa",
                     "Business Rewards Visa"]
            by_name = {}
            for f in rows:
                by_name.setdefault(f.display, f)
            parts, srcs = [], []
            for name in order:
                f = by_name.get(name)
                if f:
                    parts.append(f"{name}: {f.disp} APR")
                    self._add_srcs(srcs, self._sources_for(f, facts))
            if not parts:
                return None
            return Answer(text="Dupaco offers four credit cards: " + "; ".join(parts) + ".",
                          sources=srcs)
        if prod == "certificate":
            # render the metric the rows actually carry (a dividend question
            # must not be rendered from the APY column — or from nothing)
            metric = rows[0].metric if rows else "apy"
            word = {"apy": "APY", "div": "dividend rate", "apr": "APR",
                    "rate": "interest rate"}.get(metric, "APY")
            specials = [f for f in rows if f.quals.get("kind") in ("jumbo", "special")
                        and f.metric == metric]
            standard = [f for f in rows if f.quals.get("kind") not in ("jumbo", "special")
                        and f.metric == metric and f.quals.get("term")]
            specials.sort(key=lambda f: -(f.quals.get("term") or 0))
            standard.sort(key=lambda f: (f.quals.get("term") or 0))
            parts = []
            for f in specials + standard:
                part = f"{f.display}: {f.disp} {word}"
                if with_min:
                    mv = f.quals.get("min_open") or f.quals.get("min_earn")
                    if mv is not None:
                        part += f" (minimum ${_fmt_money(mv)})"
                parts.append(part)
            srcs = []
            for f in specials + standard:
                self._add_srcs(srcs, self._sources_for(f, facts))
            return Answer(text="Certificate rates: " + "; ".join(parts) + ".",
                          sources=srcs)
        if prod == "mortgage":
            # group rate + APR per (term, amount tier); prefer the purchase
            # row over refinance for the same tier; show the smallest and
            # largest amount tiers for each term named (the $100k tier leads)
            groups = {}
            for f in rows:
                key = (f.quals.get("yr"), f.quals.get("amount"))
                cur = groups.get(key)
                if cur is None:
                    groups[key] = {f.metric: f}
                elif f.metric not in cur:
                    cur[f.metric] = f
                elif (f.quals.get("kind") == "purchase"
                        and cur[f.metric].quals.get("kind") != "purchase"):
                    cur[f.metric] = f
            by_yr = {}
            for (yr, amt), mets in groups.items():
                by_yr.setdefault(yr, []).append((amt, mets))
            parts, srcs = [], []
            for yr in sorted(by_yr, key=lambda y: (y is None, y or 0)):
                tiers = sorted(by_yr[yr], key=lambda t: (t[0] or (0, 0))[1])
                picks = [tiers[0]] + ([tiers[-1]] if len(tiers) > 1 else [])
                for amt, mets in picks:
                    rate = mets.get("rate")
                    apr = mets.get("apr")
                    base = rate or apr
                    if not base:
                        continue
                    amt_txt = ""
                    if amt:
                        amt_txt = f" (loans up to ${amt[1]:,.0f})" if amt[0] == 0 else \
                            f" (loans ${amt[0]:,.0f}-${amt[1]:,.0f})"
                    yr_txt = f"{yr}-year fixed" if yr else base.display
                    if rate and apr:
                        parts.append(f"{yr_txt}{amt_txt}: {rate.disp} rate ({apr.disp} APR)")
                    elif rate:
                        parts.append(f"{yr_txt}{amt_txt}: {rate.disp} rate")
                    else:
                        parts.append(f"{yr_txt}{amt_txt}: {apr.disp} APR")
                    for f in mets.values():
                        self._add_srcs(srcs, self._sources_for(f, facts))
            return Answer(text="Current mortgage rates: " + "; ".join(parts) + ".",
                          sources=srcs)
        # generic list
        parts, srcs = [], []
        for f in rows[:20]:
            part = f"{f.display}{self._qualifier_text(f)}: {f.disp}"
            if with_min:
                mv = f.quals.get("min_open") or f.quals.get("min_earn")
                if mv is not None:
                    part += f" (minimum ${_fmt_money(mv)})"
            parts.append(part)
            self._add_srcs(srcs, self._sources_for(f, facts))
        return Answer(text="; ".join(parts) + ".", sources=srcs)

    @staticmethod
    def _product_from_question(ql):
        if "ira" in ql:
            return "ira"
        if re.search(r"\bcertificate|certificates?\b|\bcd\b|\bcds\b", ql):
            return "certificate"
        if re.search(r"\brv\b|\bboat\b|recreational|jet ski|\batv\b|snowmobile", ql):
            return "recreational loan"
        if "auto" in ql or "motorcycle" in ql or "vehicle" in ql or "car loan" in ql:
            return "auto loan"
        if "mortgage" in ql or "home loan" in ql or "home-loan" in ql:
            return "mortgage"
        if "home equity" in ql or "heloc" in ql:
            return "home equity"
        if "personal loan" in ql:
            return "personal loan"
        if "credit card" in ql or "visa" in ql:
            return "credit card"
        if "hsa" in ql or "health savings" in ql:
            return "hsa"
        if "share savings" in ql:
            return "share savings"
        if "bright savings" in ql:
            return "bright savings"
        if "money market" in ql:
            return "money market"
        if "checking" in ql:
            return "checking"
        if "savings" in ql:
            return "share savings" if "share" in ql else "savings"
        if "student loan" in ql:
            return "student loan"
        if "construction loan" in ql:
            return "construction loan"
        return None

    # -- prose fallback ----------------------------------------------------

    def _answer_prose(self, ql, must_words, as_of, q=None):
        # distinctive stems: what the question is actually about, minus the
        # words every question carries ("what is the current rate for ...").
        # A chunk that shares only generic words with the question is not
        # evidence — it is the nearest irrelevant passage, and answering
        # from it is a hallucination.
        qt = _stems(ql)
        distinctive = {t for t in qt if t not in GENERIC_PROSE}
        if not distinctive:
            return None
        # only the latest snapshot on/before the as-of date: older snapshots
        # hold stale values, and a current question must never see them
        if as_of:
            ok = [s for s, t in self.snaps if t <= as_of]
            snap = ok[-1] if ok else (self.snaps[0][0] if self.snaps else None)
        else:
            snap = self.snaps[-1][0] if self.snaps else None
        # a "does Dupaco offer X" question needs the chunk to say Dupaco
        # offers X, not merely to mention X in passing (a blog post asking
        # "How about Zelle?" is not an offer of Zelle)
        is_offer_q = bool(re.search(r"\boffer(s|ed|ing)?\b", ql))
        # a multi-word capitalised phrase in the question ("Chief Technology
        # Officer", "President and CEO") must appear contiguously in the
        # chunk: sharing the individual words is not the same title (a
        # "Chief Experience Officer" beside "video teller technology" is
        # not a Chief Technology Officer)
        phrases = []
        if q:
            phrases = [m.group(1) for m in re.finditer(
                r"\b((?:[A-Z][A-Za-z]*)(?:\s+(?:and\s+)?[A-Z][A-Za-z]*)+)\b", q)]
            # drop phrases that are only sentence scaffolding ("Can I") or
            # the company name itself ("Dupaco Community Credit Union"):
            # neither is a title or a name worth requiring verbatim
            _drop = {"can", "i", "what", "is", "are", "was", "were", "do",
                     "does", "did", "how", "who", "whom", "whose", "when",
                     "where", "why", "which", "the", "a", "an", "and", "or",
                     "if", "then", "so", "my", "your", "his", "her", "our",
                     "their", "we", "you", "they", "he", "she", "it", "this",
                     "that", "these", "those", "there", "here", "am", "be",
                     "been", "being", "have", "has", "had", "will", "would",
                     "should", "could", "may", "might", "me", "us", "them",
                     "him", "please", "tell", "give", "show", "list",
                     "provide"}
            phrases = [ph for ph in phrases
                       if "dupaco" not in {w.lower() for w in ph.split()}
                       and not all(w.lower() in _drop for w in ph.split())]
        best = None
        for (url, sid, taken, auth, txt, toks, is_example) in self.chunks:
            if sid != snap:
                continue
            # an illustrative example ("for illustrative purposes") is not
            # a published fact — it must never be quoted as one
            if is_example:
                continue
            if must_words and not all(w in txt.lower() for w in must_words):
                continue
            # an open question answered from a blog post must be about
            # Dupaco: blog posts profile other organisations too ("their
            # current CEO retiring"), and that is not a fact about Dupaco
            if not must_words and auth <= 30 and "dupaco" not in txt.lower():
                continue
            matched = distinctive & toks
            if not matched:
                continue
            if phrases and not any(ph.lower() in txt.lower() for ph in phrases):
                continue
            cov = len(matched) / len(distinctive)
            if is_offer_q:
                # the sentence that names the thing must itself say Dupaco
                # offers it — a blog post asking "How about Zelle?" or
                # noting "Zelle is available from many banks" is not an
                # offer of Zelle by Dupaco
                sents = re.split(r"(?<=[.!?])\s+", txt)
                ok_offer = False
                for s in sents:
                    if not (_stems(s) & distinctive):
                        continue
                    if re.search(r"\b(offer|offers|offering|provides?|available|"
                                 r"add|compatible|accepts?|supports?|set up|works?)\b",
                                 s, re.I) and "dupaco" in s.lower():
                        ok_offer = True
                        break
                if not ok_offer:
                    continue
            key = (cov, len(matched), auth)
            if best is None or key > best[0]:
                best = (key, url, sid, txt, matched)
        if best is None:
            return None
        key, url, sid, txt, matched = best
        cov = key[0]
        # the gate: every distinctive stem matched, or at least 60% with at
        # least three matched stems. A chunk that satisfies an explicit
        # must_words topic (student loans, volunteer time off, mobile
        # wallets) is already known to be about the right thing, so two
        # matched stems are enough there. Anything weaker is a refusal,
        # never a rescue by the highest-scoring weak match.
        if must_words:
            if len(matched) < 2:
                return None
        elif cov < 1.0 and not (cov >= 0.6 and len(matched) >= 3):
            return None
        # keep the sentences that carry the matched words, best first
        sents = re.split(r"(?<=[.!?])\s+", txt)
        scored = []
        for s in sents:
            n = len(_stems(s) & distinctive)
            if n:
                scored.append((n, s))
        scored.sort(key=lambda x: -x[0])
        keep = [s for _n, s in scored[:2]]
        # "how many / how much" asks for a number: a passage without one
        # does not answer it, however many words it shares
        if re.search(r"\bhow (many|much)\b", ql) and \
                not re.search(r"\d", " ".join(keep)):
            return None
        text = _clean(" ".join(keep))[:600]
        return Answer(text=text,
                      sources=[Source(url=url, snapshot_id=sid,
                                      quote=_raw(" ".join(keep))[:600])])

    # -- conflicts ---------------------------------------------------------

    def conflicts(self):
        if self._conflicts is not None:
            return self._conflicts
        out, seen = [], set()

        def add(topic, values, srcs, resolution, reason):
            key = (topic.lower(), tuple(sorted(v.lower() for v in values)))
            if key in seen:
                return
            seen.add(key)
            out.append(Conflict(topic=topic, values=values, sources=srcs,
                                resolution=resolution, reason=reason))

        # The structural conflicts go first: the list is capped, and these
        # are the ones the site's own releases disagree about most clearly.
        # 1. branch count
        vals = {str(c[0]) for c in self.counts}
        if len(vals) > 1:
            # same rule as the branch-count answer: the most recently
            # published release in the newest snapshot
            cur = self._current_count()
            n = cur[0]
            others = sorted((v for v in vals if v != str(n)), key=int)
            # one source per distinct value, newest release of each
            srcs = []
            for v in sorted(vals, key=int):
                cs = [c for c in self.counts if str(c[0]) == v]
                c = max(cs, key=lambda c: (c[5] or date(1900, 1, 1), c[3]))
                srcs.append(Source(url=c[1], snapshot_id=c[2], quote=c[4] or ""))
            when = _fmt_date(cur[5]) if cur[5] else "the newest release"
            add("Number of Dupaco branch office locations",
                sorted(vals, key=int), srcs, str(n),
                f"Older newsroom releases stay on the site with the count they "
                f"were written with ({', '.join(others)}); the most recently "
                f"published release ({when}) says {n}.")
        # 2. Forbes ordinal: newest release wins
        ords = {}
        for (url, sid, taken, auth, txt, _t, _ex) in self.chunks:
            if "forbes" not in txt.lower():
                continue
            for w in ("fifth", "fourth", "third", "sixth"):
                if re.search(rf"\b{w} time\b", txt, re.I):
                    sent = next((x for x in re.split(r"(?<=[.!?])\s+", txt)
                                 if re.search(rf"\b{w} time\b", x, re.I)),
                                txt[:200])
                    ords.setdefault(w, []).append((url, sid, taken, _raw(sent)))
        if len(ords) > 1:
            order = ["first", "second", "third", "fourth", "fifth", "sixth"]
            winner = max(ords, key=lambda w: order.index(w) if w in order else -1)
            add("How many times Forbes has recognised Dupaco", sorted(ords),
                [Source(url=v[0][0], snapshot_id=v[0][1], quote=v[0][3])
                 for v in ords.values() for v in [v[0]]][:6],
                winner, f"The newest Forbes release says the {winner} time.")
        # 3. overlapping model-year ranges that disagree (e.g. 2023-26 vs 2024-27)
        auto = [f for f in self.facts if f.product == "auto loan" and f.metric == "apr"]
        for term in sorted({f.quals.get("term") for f in auto if f.quals.get("term")}):
            rows = [f for f in auto if f.quals.get("term") == term]
            vals = _uniq_disp(rows)
            if len(vals) > 1:
                # the model-year bands roll forward (2023-26 -> 2024-27), and
                # promo / checking pages keep the old band and its old rate.
                # The current value is the newest band on the most
                # authoritative page of the newest snapshot.
                def band_end(f):
                    return (f.quals.get("years") or (0, 0))[1]
                top = band_end(self._resolve(rows, extra=band_end)[0])
                # explain against the rows that compete with the winning
                # band: other bands on a rate page are other tiers, not rivals
                winner, reason = self._resolve(
                    [f for f in rows if f.auth < 95 or band_end(f) == top],
                    extra=band_end)
                if any(f.auth < 95 and 0 < band_end(f) < top for f in rows):
                    reason += (" Lower-authority pages still list older "
                               "model-year bands with their old rates.")
                yrs = winner.quals.get("years")
                band_txt = f" ({yrs[0]}-{str(yrs[1])[2:]} model years)" if yrs else ""
                add(f"Auto loan APR, {term}-month term{band_txt}",
                    sorted(vals), self._conflict_sources(rows, winner),
                    winner.disp, reason)
        # 4. rate facts that disagree for the same product/tier/metric
        by_key = {}
        for f in self.facts:
            # the introductory balance-transfer rate is a different offer,
            # not a disagreement about the standard rate
            if ("intro" in f.display.lower()
                    or "balance transfer" in f.display.lower()):
                continue
            # a row whose product could not be identified cannot be said to
            # disagree with anything: skip it rather than report a nameless
            # "conflict"
            if re.match(r"[\$\d]", f.product or ""):
                continue
            by_key.setdefault(f.key(), []).append(f)
        for key, fs in by_key.items():
            vals = {}
            for f in fs:
                vals.setdefault(f.value, []).append(f)
            if len(vals) < 2:
                continue
            # one page showing two values under one label in one snapshot
            # is two rows of one table (two cards both labelled "Variable
            # Interest Rate"), not the site contradicting itself
            per_page = {}
            for f in fs:
                per_page.setdefault((f.url, f.snap), set()).add(f.value)
            if any(len(v) > 1 for v in per_page.values()):
                continue
            term = fs[0].quals.get("term")
            yrs = fs[0].quals.get("years")
            topic = fs[0].display
            # a bare tier label ("$25,000 - $99,999.99") names no product:
            # put the product (its section heading) in front of it
            if re.match(r"[\$\d]", topic or "") and \
                    fs[0].product.lower() not in topic.lower():
                topic = f"{fs[0].product.title()} {topic}"
            if term and f"{term} month" not in topic.lower():
                topic += f" ({term}-month)"
            if yrs and "model year" not in topic.lower():
                topic += f" ({yrs[0]}-{yrs[1]} model years)"
            topic += {"apy": " APY", "apr": " APR", "div": " dividend rate",
                      "rate": " rate"}.get(fs[0].metric, "")
            winner, reason = self._resolve(fs)
            add(topic, _uniq_disp(fs), self._conflict_sources(fs, winner),
                winner.disp, reason)
        self._conflicts = out[:40]
        return self._conflicts

    _AUTH_LABEL = {100: "the /rates page", 95: "the product's rate page",
                   80: "a dupaco.com page", 50: "an outside page",
                   40: "a calculator or promotional page",
                   30: "a blog or newsroom post", 20: "a PDF"}

    def _resolve(self, fs, extra=None):
        """Pick the current value among disagreeing facts and say why.
        Rate pages (the /rates hub and the product rate pages) outrank every
        other kind of page; among them the newest snapshot wins, then the
        page's own authority and effective date. A promo, calculator or blog
        value is only used when no rate page ever carried the fact: promo
        pages keep showing an offer long after the rate pages moved on (or
        renamed the row), so a newer promo does not beat an older rate page."""
        def cls(f):
            return 2 if f.auth >= 95 else (1 if f.auth >= 80 else 0)
        winner = max(fs, key=lambda f: (cls(f), f.taken, f.auth,
                                        f.eff or date(1900, 1, 1),
                                        extra(f) if extra else 0))
        label = self._AUTH_LABEL.get(winner.auth, "a dupaco.com page")
        lab = lambda f: self._AUTH_LABEL.get(f.auth, "another page")
        reason = (f"{label[0].upper()}{label[1:]} ({winner.url}) in the "
                  f"{winner.taken} snapshot shows {winner.disp}.")
        same = [f for f in fs if f.taken == winner.taken and f.value != winner.value]
        if same:
            kinds = sorted({lab(f) for f in same})
            rank = (f"{label} outranks {' and '.join(kinds)}"
                    if any(f.auth < winner.auth for f in same)
                    else "these pages rank equally, so this pick is low-confidence")
            reason += (f" In the same snapshot {', '.join(sorted({f.url for f in same})[:3])} "
                       f"show {', '.join(_uniq_disp(same))}; {rank}.")
        older = _uniq_disp([f for f in fs if f.taken < winner.taken
                            and cls(f) == cls(winner) and f.value != winner.value])
        if older:
            reason += f" Earlier snapshots showed {', '.join(older)}."
        newer = [f for f in fs if f.taken > winner.taken and f.value != winner.value]
        if newer:
            reason += (f" Later snapshots show {', '.join(_uniq_disp(newer))} "
                       f"only on {' and '.join(sorted({lab(f) for f in newer}))} "
                       f"({', '.join(sorted({f.url for f in newer})[:3])}); the rate "
                       f"pages no longer list this row under this name, so "
                       f"those values are not treated as current.")
        if cls(winner) == 0:
            reason += (" No rate page ever carried this, so the value comes "
                       "from a lower-authority page.")
        return winner, reason

    @staticmethod
    def _conflict_sources(fs, winner):
        """The winning fact plus one source for each competing value."""
        srcs = [Source(url=winner.url, snapshot_id=winner.snap,
                       quote=winner.row or "")]
        seen = {winner.value}
        for f in sorted(fs, key=lambda f: f.taken, reverse=True):
            if f.value in seen:
                continue
            seen.add(f.value)
            srcs.append(Source(url=f.url, snapshot_id=f.snap, quote=f.row or ""))
        return srcs[:6]


def build():
    return Knowledge()
