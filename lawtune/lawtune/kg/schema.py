"""Knowledge-graph schema and the legal taxonomy behind it.

Design note on scope: most legal KG projects pick one vertical -- matrimonial disputes,
or consumer, or tax -- because a narrow ontology is easy. This schema is deliberately
general: nodes and edges are the ones every Indian judgment has (a court, a bench, dates,
provisions applied, precedents cited, a disposal), and the subject matter is a *label* on
the judgment rather than a hardcoded structure. That way one graph covers criminal,
constitutional, tax, service, IP, environment, arbitration and the rest without a schema
change per area.

Kuzu is the store: embedded (no server, no Docker, no JVM), Cypher, MIT licence, one
`pip install`. Free in the sense that matters -- nothing to run, nothing to pay for, and
the whole database is a directory you can delete.
"""
from __future__ import annotations

# --------------------------------------------------------------------- node tables
# Kuzu requires a declared schema. Keys are table names; values are DDL bodies.
NODE_TABLES: dict[str, str] = {
    # A judgment or order. `cid` is our stable internal id (from the ecourts case id).
    "Judgment": """
        cid STRING PRIMARY KEY,
        title STRING,
        petitioner STRING,
        respondent STRING,
        citation STRING,
        neutral_citation STRING,
        decision_date DATE,
        year INT64,
        disposal STRING,
        court STRING,
        bench_size INT64,
        languages STRING,
        source_path STRING,
        n_chars INT64,
        has_text BOOLEAN
    """,
    "Judge": """
        name STRING PRIMARY KEY,
        n_judgments INT64
    """,
    "Court": """
        name STRING PRIMARY KEY,
        level STRING
    """,
    # A statute: IPC, CrPC, the Constitution, the Income Tax Act, ...
    "Act": """
        key STRING PRIMARY KEY,
        name STRING,
        year INT64,
        max_provision INT64,
        category STRING
    """,
    # A single Section / Article / Rule within an Act.
    "Provision": """
        pid STRING PRIMARY KEY,
        act_key STRING,
        kind STRING,
        number INT64,
        suffix STRING,
        label STRING
    """,
    # Subject matter. See AREAS below -- this is what makes the graph cover everything.
    "Area": """
        key STRING PRIMARY KEY,
        name STRING,
        parent STRING
    """,
    # Named doctrines and legal concepts ("basic structure", "res judicata", ...).
    "Doctrine": """
        key STRING PRIMARY KEY,
        name STRING
    """,
    # A retrievable chunk of judgment text. Bridges the graph to the FAISS index:
    # the vector search returns chunk ids, and this table walks them back to Judgments.
    "Chunk": """
        chunk_id STRING PRIMARY KEY,
        cid STRING,
        ordinal INT64,
        n_chars INT64,
        vector_row INT64
    """,
}

# --------------------------------------------------------------------- rel tables
# (name, FROM, TO, extra properties)
REL_TABLES: list[tuple[str, str, str, str]] = [
    ("DECIDED_BY",   "Judgment", "Judge",    "authored BOOLEAN"),
    ("IN_COURT",     "Judgment", "Court",    ""),
    ("CITES",        "Judgment", "Judgment", "raw STRING, treatment STRING"),
    ("INTERPRETS",   "Judgment", "Provision", "mentions INT64"),
    ("UNDER_ACT",    "Judgment", "Act",      "mentions INT64"),
    ("PART_OF",      "Provision", "Act",     ""),
    ("ABOUT",        "Judgment", "Area",     "score DOUBLE"),
    ("INVOKES",      "Judgment", "Doctrine", "mentions INT64"),
    ("HAS_CHUNK",    "Judgment", "Chunk",    ""),
]

# --------------------------------------------------------------------- taxonomy
# Every substantive area of Indian law, each with the surface terms that identify it.
# Matching is keyword-based and therefore free, deterministic and inspectable -- no LLM
# call per judgment, which at ~100k judgments is the difference between minutes and money.
AREAS: dict[str, tuple[str, str, list[str]]] = {
    # key: (display name, parent, trigger terms)
    "constitutional": ("Constitutional Law", "public", [
        "fundamental right", "article 14", "article 19", "article 21", "article 32",
        "basic structure", "writ petition", "ultra vires", "constitutional validity",
        "directive principle", "article 226", "judicial review"]),
    "criminal": ("Criminal Law", "public", [
        "indian penal code", "ipc", "bharatiya nyaya sanhita", "accused", "conviction",
        "acquittal", "prosecution", "offence", "murder", "culpable homicide",
        "sentence of imprisonment", "life sentence", "quantum of sentence",
        "criminal appeal", "charge sheet"]),
    "criminal_procedure": ("Criminal Procedure", "procedure", [
        "cr.p.c", "crpc", "code of criminal procedure", "bharatiya nagarik suraksha",
        "bail", "anticipatory bail", "remand", "fir", "investigation", "cognizance",
        "quashing", "section 482"]),
    "evidence": ("Law of Evidence", "procedure", [
        "evidence act", "bharatiya sakshya", "admissibility", "hostile witness",
        "dying declaration", "circumstantial evidence", "burden of proof",
        "expert evidence", "confession"]),
    "civil_procedure": ("Civil Procedure", "procedure", [
        "code of civil procedure", "c.p.c", "cpc", "order vii rule", "decree",
        "execution petition", "res judicata", "temporary injunction", "plaint"]),
    "contract": ("Contract Law", "private", [
        "indian contract act", "breach of contract", "specific relief",
        "novation", "quantum meruit", "lawful consideration",
        "want of consideration", "liquidated damages", "damages for breach",
        "agreement to sell", "concluded contract", "privity of contract"]),
    "property": ("Property Law", "private", [
        "transfer of property", "sale deed", "easement", "mortgage", "partition",
        "adverse possession", "benami", "title deed", "clear title",
        "title to the property", "khata", "possession of the suit",
        "settled possession", "specific performance"]),
    "land_revenue": ("Land & Revenue", "private", [
        "land acquisition", "revenue record", "ceiling act", "tenancy", "mutation",
        "khasra", "patta", "compensation for land"]),
    "family": ("Family Law", "private", [
        "hindu marriage act", "divorce", "maintenance", "custody", "guardianship",
        "matrimonial", "cruelty", "desertion", "restitution of conjugal",
        "special marriage act", "muslim personal law", "talaq", "adoption",
        "domestic violence", "section 125"]),
    "succession": ("Succession & Inheritance", "private", [
        "hindu succession", "testamentary", "probate", "coparcener",
        "intestate", "legal heir", "letters of administration", "last will",
        "will and testament", "execution of the will", "bequest", "legatee",
        "succession certificate"]),
    "tort": ("Tort & Negligence", "private", [
        "negligence", "vicarious liability", "nuisance", "defamation",
        "medical negligence", "duty of care", "tortious"]),
    "motor_accident": ("Motor Accident Claims", "private", [
        "motor vehicles act", "mact", "claims tribunal", "compensation for accident",
        "insurance company", "permanent disability", "rash and negligent driving"]),
    "consumer": ("Consumer Protection", "private", [
        "consumer protection", "deficiency in service", "unfair trade practice",
        "national commission", "ncdrc", "complainant consumer"]),
    "labour": ("Labour & Industrial", "regulatory", [
        "industrial disputes", "workman", "retrenchment", "gratuity",
        "provident fund", "trade union", "esi act", "minimum wages",
        "contract labour", "reinstatement"]),
    "service": ("Service Law", "public", [
        "departmental enquiry", "disciplinary proceedings", "promotion", "seniority",
        "pension", "compassionate appointment", "government servant",
        "central administrative tribunal",
        "recruitment rules", "regularisation"]),
    "tax_direct": ("Direct Tax", "regulatory", [
        "income tax act", "assessment year", "assessing officer", "itat",
        "capital gains", "tds", "reassessment", "section 148", "deduction under section"]),
    "tax_indirect": ("Indirect Tax & GST", "regulatory", [
        "goods and services tax", "gst", "cenvat", "excise", "customs act",
        "service tax", "input tax credit", "cestat", "valuation of goods"]),
    "company": ("Company & Corporate", "regulatory", [
        "companies act", "oppression and mismanagement", "nclt", "nclat",
        "share transfer", "director", "winding up", "scheme of amalgamation"]),
    "insolvency": ("Insolvency & Bankruptcy", "regulatory", [
        "insolvency and bankruptcy code", "ibc", "corporate insolvency resolution",
        "moratorium", "resolution professional", "committee of creditors",
        "financial creditor", "operational creditor", "liquidation"]),
    "banking": ("Banking & Finance", "regulatory", [
        "sarfaesi", "recovery of debts", "drt", "negotiable instruments",
        "section 138", "cheque dishonour", "guarantor", "secured creditor"]),
    "arbitration": ("Arbitration & ADR", "procedure", [
        "arbitration and conciliation", "arbitral award", "section 34",
        "section 11", "arbitrator", "seat of arbitration", "mediation"]),
    "ip": ("Intellectual Property", "regulatory", [
        "trade marks act", "copyright", "patent", "passing off", "infringement",
        "design act", "geographical indication", "prior use"]),
    "environment": ("Environmental Law", "public", [
        "environment protection act", "ngt", "national green tribunal", "pollution",
        "forest conservation", "coastal regulation", "polluter pays",
        "precautionary principle", "wildlife protection"]),
    "cyber": ("Cyber & IT", "regulatory", [
        "information technology act", "section 66", "cyber", "data protection",
        "intermediary liability", "electronic record", "digital evidence"]),
    "election": ("Election Law", "public", [
        "representation of the people act", "election petition", "returned candidate",
        "corrupt practice", "nomination paper", "election commission"]),
    "human_rights": ("Human Rights & Civil Liberties", "public", [
        "human rights", "custodial", "torture", "preventive detention",
        "habeas corpus", "personal liberty", "manual scavenging", "bonded labour"]),
    "sc_st_reservation": ("Reservation & Social Justice", "public", [
        "scheduled caste", "scheduled tribe", "other backward", "reservation",
        "creamy layer", "atrocities act", "caste certificate"]),
    "education": ("Education Law", "regulatory", [
        "right to education", "university", "medical college", "admission to the course",
        "admission process", "counselling for admission",
        "neet", "affiliation", "minority institution", "ugc"]),
    "juvenile": ("Juvenile & Child Law", "public", [
        "juvenile justice", "pocso", "child welfare committee", "minor victim",
        "child in conflict with law"]),
    "narcotics": ("Narcotics", "public", [
        "ndps", "narcotic drugs", "psychotropic", "commercial quantity",
        "section 37", "contraband"]),
    "competition": ("Competition Law", "regulatory", [
        "competition act", "cci", "abuse of dominance", "anti-competitive",
        "combination", "cartel"]),
    "securities": ("Securities & Capital Markets", "regulatory", [
        "sebi", "securities and exchange board", "insider trading", "sat",
        "listing agreement", "collective investment"]),
    "administrative": ("Administrative Law", "public", [
        "natural justice", "audi alteram partem", "arbitrary", "delegated legislation",
        "administrative action", "legitimate expectation", "mala fide"]),
    "contempt": ("Contempt & Court Administration", "procedure", [
        "contempt of court", "wilful disobedience", "apology", "court's order"]),
    "public_interest": ("Public Interest Litigation", "public", [
        "public interest litigation", "pil", "locus standi", "continuing mandamus"]),
}

PARENTS = {
    "public": "Public Law",
    "private": "Private Law",
    "regulatory": "Regulatory & Economic Law",
    "procedure": "Procedural Law",
}

# --------------------------------------------------------------------- doctrines
DOCTRINES: dict[str, list[str]] = {
    "basic_structure": ["basic structure"],
    "res_judicata": ["res judicata"],
    "stare_decisis": ["stare decisis", "binding precedent"],
    "natural_justice": ["natural justice", "audi alteram partem"],
    "doctrine_of_severability": ["severability"],
    "doctrine_of_eclipse": ["doctrine of eclipse"],
    "legitimate_expectation": ["legitimate expectation"],
    "proportionality": ["doctrine of proportionality", "proportionality test"],
    "promissory_estoppel": ["promissory estoppel"],
    "lifting_corporate_veil": ["lifting the corporate veil", "piercing the veil"],
    "polluter_pays": ["polluter pays"],
    "precautionary_principle": ["precautionary principle"],
    "public_trust": ["public trust doctrine"],
    "parens_patriae": ["parens patriae"],
    "colourable_legislation": ["colourable legislation"],
    "pith_and_substance": ["pith and substance"],
    "harmonious_construction": ["harmonious construction"],
    "ejusdem_generis": ["ejusdem generis"],
    "beyond_reasonable_doubt": ["beyond reasonable doubt"],
    "benefit_of_doubt": ["benefit of doubt"],
    "last_seen_theory": ["last seen together", "last seen theory"],
    "sole_testimony": ["sole testimony"],
    "doctrine_of_merger": ["doctrine of merger"],
    "lis_pendens": ["lis pendens"],
    "caveat_emptor": ["caveat emptor"],
    "force_majeure": ["force majeure"],
    "double_jeopardy": ["double jeopardy"],
    "locus_standi": ["locus standi"],
}


def ddl() -> list[str]:
    """Full CREATE statements, in dependency order."""
    stmts = [f"CREATE NODE TABLE IF NOT EXISTS {name}({body.strip()})"
             for name, body in NODE_TABLES.items()]
    for name, src, dst, props in REL_TABLES:
        tail = f", {props}" if props else ""
        stmts.append(f"CREATE REL TABLE IF NOT EXISTS {name}(FROM {src} TO {dst}{tail})")
    return stmts
