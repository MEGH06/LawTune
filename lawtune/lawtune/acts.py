"""The canonical registry of Indian statutes.

One source of truth, used by two very different consumers:

  * `guardrails.py` range-checks the model's citations against `max_provision` -- a
    "Section 812 IPC" is provably fabricated because the IPC stops at 511.
  * `kg/extract.py` resolves "S. 138 of the NI Act" in a judgment into a Provision node
    hanging off the right Act node.

`max_provision` is None where the Act's numbering is irregular or I could not verify the
ceiling. None means "cannot range-check", not "unlimited" -- the guardrail skips those
rather than guessing, because a false accusation of hallucination is its own bug.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Act(NamedTuple):
    key: str
    name: str
    year: int | None
    max_provision: int | None      # highest section/article number, None if unverified
    category: str
    patterns: list[str]            # regex alternatives that name this Act in prose


# Ordered most-specific-first: BNSS must be tried before BNS, or "B.N.S.S." matches "BNS".
ACTS: list[Act] = [
    Act("constitution", "Constitution of India", 1950, 395, "constitutional",
        [r"constitution of india", r"\bconstitution\b", r"\bconstitutional\b"]),

    # --- criminal
    Act("ipc", "Indian Penal Code, 1860", 1860, 511, "criminal",
        [r"indian penal code", r"\bi\.?p\.?c\.?\b", r"\bpenal code\b"]),
    Act("bnss", "Bharatiya Nagarik Suraksha Sanhita, 2023", 2023, 531, "criminal",
        [r"bharatiya nagarik suraksha", r"\bb\.?n\.?s\.?s\.?\b"]),
    Act("bns", "Bharatiya Nyaya Sanhita, 2023", 2023, 358, "criminal",
        [r"bharatiya nyaya sanhita", r"\bb\.?n\.?s\.?\b"]),
    Act("bsa", "Bharatiya Sakshya Adhiniyam, 2023", 2023, 170, "evidence",
        [r"bharatiya sakshya", r"\bb\.?s\.?a\.?\b"]),
    Act("crpc", "Code of Criminal Procedure, 1973", 1973, 484, "procedure",
        [r"code of criminal procedure", r"\bcr\.?\s?p\.?\s?c\.?\b"]),
    Act("evidence", "Indian Evidence Act, 1872", 1872, 167, "evidence",
        [r"indian evidence act", r"\bevidence act\b"]),
    Act("ndps", "NDPS Act, 1985", 1985, 83, "criminal",
        [r"narcotic drugs and psychotropic", r"\bn\.?d\.?p\.?s\.?\b"]),
    Act("pocso", "POCSO Act, 2012", 2012, 46, "criminal",
        [r"protection of children from sexual offences", r"\bpocso\b"]),
    Act("corruption", "Prevention of Corruption Act, 1988", 1988, 31, "criminal",
        [r"prevention of corruption act"]),
    Act("uapa", "Unlawful Activities (Prevention) Act, 1967", 1967, 53, "criminal",
        [r"unlawful activities.{0,20}prevention", r"\buapa\b"]),
    Act("pmla", "Prevention of Money Laundering Act, 2002", 2002, 75, "criminal",
        [r"prevention of money.?laundering", r"\bpmla\b"]),
    Act("sc_st_act", "SC and ST (Prevention of Atrocities) Act, 1989", 1989, 23, "criminal",
        [r"prevention of atrocities", r"atrocities act"]),
    Act("jj_act", "Juvenile Justice (Care and Protection) Act, 2015", 2015, 112, "criminal",
        [r"juvenile justice"]),
    Act("dv_act", "Protection of Women from Domestic Violence Act, 2005", 2005, 37, "family",
        [r"domestic violence act", r"protection of women from domestic"]),

    # --- civil / private
    Act("cpc", "Code of Civil Procedure, 1908", 1908, 158, "procedure",
        [r"code of civil procedure", r"\bc\.?\s?p\.?\s?c\.?\b"]),
    Act("contract", "Indian Contract Act, 1872", 1872, 238, "contract",
        [r"indian contract act", r"\bcontract act\b"]),
    Act("specific_relief", "Specific Relief Act, 1963", 1963, 44, "contract",
        [r"specific relief act"]),
    Act("tpa", "Transfer of Property Act, 1882", 1882, 137, "property",
        [r"transfer of property act", r"\bt\.?p\.? act\b"]),
    Act("registration", "Registration Act, 1908", 1908, 91, "property",
        [r"registration act"]),
    Act("limitation", "Limitation Act, 1963", 1963, 32, "procedure",
        [r"limitation act"]),
    Act("easements", "Indian Easements Act, 1882", 1882, 64, "property",
        [r"easements act"]),
    Act("partnership", "Indian Partnership Act, 1932", 1932, 74, "commercial",
        [r"partnership act"]),
    Act("sale_of_goods", "Sale of Goods Act, 1930", 1930, 66, "commercial",
        [r"sale of goods act"]),
    Act("ni_act", "Negotiable Instruments Act, 1881", 1881, 147, "commercial",
        [r"negotiable instruments", r"\bn\.?i\.? act\b"]),

    # --- family / succession
    Act("hma", "Hindu Marriage Act, 1955", 1955, 30, "family",
        [r"hindu marriage act", r"\bh\.?m\.? act\b"]),
    Act("hsa", "Hindu Succession Act, 1956", 1956, 31, "family",
        [r"hindu succession act"]),
    Act("hama", "Hindu Adoptions and Maintenance Act, 1956", 1956, 30, "family",
        [r"hindu adoptions and maintenance"]),
    Act("hminor", "Hindu Minority and Guardianship Act, 1956", 1956, 13, "family",
        [r"hindu minority and guardianship"]),
    Act("sma", "Special Marriage Act, 1954", 1954, 51, "family",
        [r"special marriage act"]),
    Act("indian_succession", "Indian Succession Act, 1925", 1925, 391, "family",
        [r"indian succession act"]),
    Act("guardians_wards", "Guardians and Wards Act, 1890", 1890, 53, "family",
        [r"guardians and wards"]),

    # --- regulatory / economic
    Act("income_tax", "Income Tax Act, 1961", 1961, None, "tax",
        [r"income.?tax act", r"\bi\.?t\.? act\b"]),
    Act("gst", "Central Goods and Services Tax Act, 2017", 2017, 174, "tax",
        [r"goods and services tax act", r"\bc?gst act\b"]),
    Act("customs", "Customs Act, 1962", 1962, 161, "tax",
        [r"customs act"]),
    Act("excise", "Central Excise Act, 1944", 1944, None, "tax",
        [r"central excise act"]),
    Act("companies", "Companies Act, 2013", 2013, 470, "corporate",
        [r"companies act"]),
    Act("ibc", "Insolvency and Bankruptcy Code, 2016", 2016, 255, "corporate",
        [r"insolvency and bankruptcy code", r"\bi\.?b\.?c\.?\b"]),
    Act("sarfaesi", "SARFAESI Act, 2002", 2002, 42, "banking",
        [r"securitisation and reconstruction", r"\bsarfaesi\b"]),
    Act("rdb", "Recovery of Debts and Bankruptcy Act, 1993", 1993, 37, "banking",
        [r"recovery of debts"]),
    Act("sebi", "SEBI Act, 1992", 1992, 35, "securities",
        [r"securities and exchange board of india act", r"\bsebi act\b"]),
    Act("competition", "Competition Act, 2002", 2002, 66, "competition",
        [r"competition act"]),
    Act("arbitration", "Arbitration and Conciliation Act, 1996", 1996, 87, "arbitration",
        [r"arbitration and conciliation", r"\barbitration act\b"]),
    Act("consumer", "Consumer Protection Act, 2019", 2019, 107, "consumer",
        [r"consumer protection act"]),
    Act("it_act", "Information Technology Act, 2000", 2000, 90, "technology",
        [r"information technology act", r"\bi\.?t\.? act, 2000\b"]),
    Act("copyright", "Copyright Act, 1957", 1957, 79, "ip",
        [r"copyright act"]),
    Act("trademarks", "Trade Marks Act, 1999", 1999, 159, "ip",
        [r"trade marks act", r"trademarks act"]),
    Act("patents", "Patents Act, 1970", 1970, 163, "ip",
        [r"patents act"]),
    Act("designs", "Designs Act, 2000", 2000, 48, "ip",
        [r"designs act"]),

    # --- labour / service
    Act("ida", "Industrial Disputes Act, 1947", 1947, 40, "labour",
        [r"industrial disputes act", r"\bi\.?d\.? act\b"]),
    Act("gratuity", "Payment of Gratuity Act, 1972", 1972, 15, "labour",
        [r"payment of gratuity"]),
    Act("epf", "Employees' Provident Funds Act, 1952", 1952, 22, "labour",
        [r"provident funds? (and )?misc", r"\be\.?p\.?f\.? act\b"]),
    Act("esi", "Employees' State Insurance Act, 1948", 1948, 100, "labour",
        [r"employees.? state insurance", r"\be\.?s\.?i\.? act\b"]),
    Act("minimum_wages", "Minimum Wages Act, 1948", 1948, 31, "labour",
        [r"minimum wages act"]),
    Act("contract_labour", "Contract Labour (R&A) Act, 1970", 1970, 35, "labour",
        [r"contract labour"]),
    Act("workmen_comp", "Employees' Compensation Act, 1923", 1923, 36, "labour",
        [r"workmen.?s compensation", r"employees.? compensation act"]),

    # --- public / other
    Act("mv_act", "Motor Vehicles Act, 1988", 1988, 217, "motor",
        [r"motor vehicles act", r"\bm\.?v\.? act\b"]),
    Act("rti", "Right to Information Act, 2005", 2005, 31, "public",
        [r"right to information act", r"\brti act\b"]),
    Act("rte", "Right to Education Act, 2009", 2009, 38, "education",
        [r"right of children to free and compulsory", r"\brte act\b"]),
    Act("epa", "Environment (Protection) Act, 1986", 1986, 26, "environment",
        [r"environment.{0,3}protection.{0,3} act"]),
    Act("ngt_act", "National Green Tribunal Act, 2010", 2010, 38, "environment",
        [r"national green tribunal act"]),
    Act("forest_conservation", "Forest (Conservation) Act, 1980", 1980, 5, "environment",
        [r"forest.{0,3}conservation.{0,3} act"]),
    Act("wildlife", "Wild Life (Protection) Act, 1972", 1972, 66, "environment",
        [r"wild ?life.{0,3}protection"]),
    Act("water_act", "Water (Prevention and Control of Pollution) Act, 1974", 1974, 64,
        "environment", [r"water.{0,40}pollution.{0,3} act"]),
    Act("air_act", "Air (Prevention and Control of Pollution) Act, 1981", 1981, 54,
        "environment", [r"air.{0,40}pollution.{0,3} act"]),
    Act("land_acq_2013", "Right to Fair Compensation ... Act, 2013", 2013, 114, "property",
        [r"right to fair compensation and transparency", r"\blarr act\b"]),
    Act("land_acq_1894", "Land Acquisition Act, 1894", 1894, 55, "property",
        [r"land acquisition act, 1894", r"land acquisition act"]),
    Act("rpa_1951", "Representation of the People Act, 1951", 1951, 171, "election",
        [r"representation of the people act"]),
    Act("contempt_act", "Contempt of Courts Act, 1971", 1971, 24, "procedure",
        [r"contempt of courts act"]),
    Act("general_clauses", "General Clauses Act, 1897", 1897, 31, "procedure",
        [r"general clauses act"]),
    Act("aadhaar", "Aadhaar Act, 2016", 2016, 59, "technology",
        [r"aadhaar.{0,40}act"]),
    Act("dpdp", "Digital Personal Data Protection Act, 2023", 2023, 44, "technology",
        [r"digital personal data protection", r"\bdpdp\b"]),
]

BY_KEY: dict[str, Act] = {a.key: a for a in ACTS}

# Compiled once. Order matters -- see the BNSS/BNS note above.
COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile("|".join(a.patterns), re.IGNORECASE), a.key) for a in ACTS
]

# Provision limits in the shape guardrails.py wants: {key: (max, display name)}
LIMITS: dict[str, tuple[int, str]] = {
    a.key: (a.max_provision, a.name) for a in ACTS if a.max_provision is not None
}


def identify_act(text: str) -> str | None:
    """Which Act does this fragment name? Returns an act key, or None.

    Registry order decides ties, so this is only safe on a fragment naming ONE Act.
    When a window may name several, use `nearest_act` -- see the note there.
    """
    for pattern, key in COMPILED:
        if pattern.search(text):
            return key
    return None


def find_acts(text: str) -> list[tuple[int, int, str]]:
    """Every Act mention with its position: [(start, end, key), ...] sorted by start."""
    hits: list[tuple[int, int, str]] = []
    for pattern, key in COMPILED:
        for m in pattern.finditer(text):
            hits.append((m.start(), m.end(), key))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    return hits


def nearest_act(text: str, start: int, end: int, right_bias: float = 1.6,
                max_distance: float = float("inf")) -> str | None:
    """The Act named closest to the span [start, end).

    `identify_act` returns whichever Act sits earliest in the registry, which is wrong
    whenever a window names more than one. Concretely: in "Section 32 of the Indian
    Evidence Act, and on Section 498A of the Indian Penal Code", it attributed BOTH to
    the Act that happened to come first in ACTS -- and since a stray "Constitution" a
    line later outranked the IPC, "Section 498A" was then dropped for exceeding
    Article 395. Position, not registry order, decides.

    Rightward mentions are preferred because Indian drafting is "Section X of the Y
    Act"; a leftward match pays a distance penalty rather than being excluded, so
    "under the Evidence Act, Section 32 provides" still resolves.
    """
    best_key, best_dist = None, float("inf")
    for a_start, a_end, key in find_acts(text):
        if a_start >= end:
            dist = float(a_start - end)
        elif a_end <= start:
            dist = (start - a_end) * right_bias
        else:
            dist = 0.0                       # the span sits inside the Act's own name
        if dist < best_dist:
            best_key, best_dist = key, dist
    # A statute named far away is not the statute being cited -- a judgment about a
    # State sales-tax Act we do not carry would otherwise borrow whatever central Act
    # happened to appear a paragraph later.
    return best_key if best_dist <= max_distance else None


def all_act_keys() -> list[str]:
    return [a.key for a in ACTS]
