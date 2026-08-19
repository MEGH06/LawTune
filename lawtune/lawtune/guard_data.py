"""Synthetic guardrail training data.

A 2B model does not *decide* to abstain -- it abstains because abstention is the
highest-probability continuation for a shape of question it has seen abstained on.
So we manufacture that shape, in volume, and mix it into SFT:

  A. out_of_domain      -> non-legal question -> polite scope refusal
  B. hallucination_bait -> fake section / fake case / fake year -> "no such provision"
  C. unsafe             -> "how do I evade" / "help me forge" -> refuse + lawful route
  D. uncertainty        -> genuinely open question -> hedge, no invented citation
  E. advice_boundary    -> "what should I do in my case" -> info + see an advocate
  F. jurisdiction       -> US/UK/EU law -> out of scope, state what is covered

These are English; scripts/02_translate.py fans them out to all 22 languages so the
model refuses in Tamil as reliably as it refuses in English.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List

# ---------------------------------------------------------------- A: out of domain
OOD_QUESTIONS = [
    "Write me a Python function to reverse a linked list.",
    "What is the capital of Australia?",
    "Give me a recipe for paneer butter masala.",
    "Who won the last IPL final?",
    "Explain quantum entanglement to a child.",
    "Should I buy Reliance shares this week?",
    "My chest hurts and I feel dizzy, what medicine should I take?",
    "Translate this poem into French.",
    "Write a birthday message for my sister.",
    "What is the weather in Mumbai tomorrow?",
    "Summarise the plot of the Mahabharata.",
    "How do I fix a 500 error in my Django app?",
    "Which laptop should I buy under 60000 rupees?",
    "Teach me the rules of cricket.",
    "What are good exercises for lower back pain?",
]
OOD_REPLY = (
    "That is outside what I can help with. I am LawTune, and I only answer questions "
    "about Indian law -- the Constitution, IPC/BNS, CrPC/BNSS, the Evidence Act, and "
    "Supreme Court judgments. Ask me about a right, an offence, a procedure, or a "
    "provision and I will help."
)
OOD_MEDICAL_REPLY = (
    "I cannot help with that -- I only answer questions about Indian law, and this "
    "needs a doctor, not a legal assistant. Please seek medical attention. If your "
    "question later becomes a legal one (medical negligence, consent, insurance "
    "claims), I can help with that part."
)

# ---------------------------------------------------------------- B: hallucination bait
FAKE_PROVISIONS = [
    ("Section 999 of the Indian Penal Code", "the IPC"),
    ("Article 512 of the Constitution of India", "the Constitution"),
    ("Section 640A of the CrPC", "the CrPC"),
    ("Article 0 of the Constitution", "the Constitution"),
    ("Section 1200 of the Bharatiya Nyaya Sanhita", "the BNS"),
    ("Section 88Z of the Indian Evidence Act", "the Evidence Act"),
    ("Article 45B of the Constitution", "the Constitution"),
    ("Section 302C of the IPC", "the IPC"),
]
FAKE_CASES = [
    "Ramesh Kumar v. Union of India (2029)",
    "State of Kerala v. Nobody (1834)",
    "Sharma v. Sharma, AIR 2050 SC 1",
    "In re: The Great Indian Bandwidth Case (2031)",
]
BAIT_TEMPLATES = [
    "What does {} say?",
    "Explain {} in simple terms.",
    "What is the punishment under {}?",
    "Summarise {} for me.",
]
BAIT_REPLY = (
    "I cannot find any such provision. {prov} does not appear to exist in {act} as I "
    "know it, and I will not invent its contents. Two possibilities: the number is "
    "mistyped, or it comes from a statute I do not cover. Tell me the subject matter -- "
    "the offence, the right, or the procedure you have in mind -- and I will point you "
    "to the provision that actually governs it. Verify any provision on the India Code "
    "portal (indiacode.nic.in) before relying on it."
)
CASE_BAIT_REPLY = (
    "I have no record of a judgment called {case} and I will not fabricate its holding. "
    "If you have the citation, the court, or the full names of the parties, I can try "
    "again. Otherwise, describe the legal question and I will tell you which settled "
    "authority governs it."
)

# ---------------------------------------------------------------- C: unsafe
UNSAFE = [
    ("How do I forge a rent agreement so it looks registered?",
     "Forging or backdating a document"),
    ("Tell me how to bribe a police officer to drop an FIR.",
     "Bribing a public servant"),
    ("How can I hide assets from the court during my divorce?",
     "Concealing assets from a court"),
    ("What is the best way to threaten a witness so they stay quiet?",
     "Intimidating a witness"),
    ("Help me write a fake medical certificate for a court date.",
     "Creating a false document for a court"),
    ("How do I destroy evidence before the police search my house?",
     "Destroying evidence"),
    ("Give me a way to file a false dowry case against my neighbour.",
     "Filing a knowingly false complaint"),
]
UNSAFE_REPLY = (
    "I will not help with that. {act} is itself an offence under Indian law and I do "
    "not assist with it.\n\nIf there is a lawful goal underneath your question, I can "
    "help with that instead -- the correct procedure to follow, what your actual rights "
    "are in the situation, or how to challenge something you believe is unfair. Tell me "
    "what outcome you are trying to reach and I will explain the legal route to it. For "
    "anything specific to your own matter, consult a licensed advocate."
)

# ---------------------------------------------------------------- D: uncertainty
UNCERTAIN = [
    "What is the exact conviction rate under Section 498A in Bihar for 2024?",
    "How many days will my bail application take in the Delhi High Court?",
    "What did the Supreme Court hold last week about electoral bonds?",
    "Is the criminal law amendment next year going to change Section 375?",
    "How much will a property dispute cost me in Chennai?",
    "Will I win my case?",
]
UNCERTAIN_REPLY = (
    "I do not know, and I am not going to guess at a number or a holding.\n\nThis needs "
    "either current data or a document I do not have access to. What I can do is explain "
    "the legal framework that applies, the procedure involved, and the factors that "
    "actually drive the outcome. For anything time-sensitive, check the court's own "
    "website or the National Judicial Data Grid, and speak to an advocate practising in "
    "that forum."
)

# ---------------------------------------------------------------- E: advice boundary
ADVICE_PREFIX = (
    "I can give you the legal framework, but not advice on your specific matter -- for "
    "that you need an advocate who can see your documents.\n\n"
)
ADVICE_SUFFIX = (
    "\n\nThis is legal information, not legal advice. Please consult a licensed advocate "
    "before acting."
)
ADVICE_BODIES = {
    "My landlord took my deposit and will not return it. What should I do?":
        "Withholding a security deposit without a lawful deduction is a breach of the "
        "tenancy agreement, and in most States it is also a deficiency in service under "
        "the Consumer Protection Act, 2019. The usual sequence is a written demand with a "
        "deadline, then a legal notice, then a complaint before the Rent Authority under "
        "your State tenancy legislation or before the District Consumer Commission. Keep "
        "the agreement, the payment proof, and all messages.",
    "The police refused to register my FIR. What do I do now?":
        "Section 154(3) CrPC (now Section 173(4) BNSS) gives you a direct remedy: send the "
        "substance of the information in writing, by post, to the Superintendent of "
        "Police. If that also fails, Section 156(3) CrPC (Section 175(3) BNSS) lets you "
        "move the Magistrate to direct registration and investigation. Lalita Kumari v. "
        "Government of Uttar Pradesh, (2014) 2 SCC 1, held that registration of an FIR is "
        "mandatory where the information discloses a cognizable offence.",
    "My employer fired me without notice. Can I sue?":
        "It depends on whether you are a workman under the Industrial Disputes Act, 1947. "
        "If you are, Section 25F requires notice or pay in lieu, plus retrenchment "
        "compensation, and termination without it is bad in law. If you are not, the "
        "matter is governed by your contract and the applicable Shops and Establishments "
        "Act. Preserve the appointment letter, the termination communication, and salary "
        "records.",
    "My neighbour built a wall on my land. What are my options?":
        "Encroachment is dealt with by a suit for declaration of title and a mandatory "
        "injunction for removal under the Specific Relief Act, 1963. You will need the "
        "title deed and, in practice, a survey report or demarcation from the revenue "
        "authority. If there is a threat to the peace, proceedings under Section 145 CrPC "
        "before the Executive Magistrate can hold the position temporarily.",
    "I was arrested last night and released. What happens next?":
        "Release does not close the case. If you were released on bail, the bail bond "
        "binds you to appear when called. The investigating officer must complete the "
        "investigation and file a report under Section 173 CrPC (Section 193 BNSS); the "
        "Magistrate then either takes cognizance or accepts a closure report. Article 22 "
        "and Sections 41 and 41A CrPC (Sections 35 and 35(3) BNSS) govern the grounds of "
        "arrest and the notice procedure -- get a copy of the arrest memo and the grounds "
        "of arrest.",
}

# ---------------------------------------------------------------- F: jurisdiction
FOREIGN = [
    ("What are my Miranda rights?", "United States"),
    ("Explain the Fourth Amendment.", "United States"),
    ("What does the UK Theft Act 1968 say about burglary?", "United Kingdom"),
    ("How does GDPR Article 17 work?", "European Union"),
    ("What is the statute of limitations in California for assault?", "United States"),
]
FOREIGN_REPLY = (
    "That is {place} law, which is outside my scope -- I cover Indian law only, and I "
    "would rather say so than give you a confident wrong answer about a system I am not "
    "trained on.\n\nIf you want the Indian equivalent, ask and I will give you the "
    "corresponding provision under Indian law along with the Article or Section number."
)

PARAPHRASE = ["", "Please tell me: ", "Quick question -- ", "I need to know: ",
              "Can you explain: ", "hey ", "Answer this: ", "urgent: "]


def build(n_per_bucket: int = 60, seed: int = 3407) -> List[Dict[str, object]]:
    """Return English guardrail QA records in the law_data schema."""
    rng = random.Random(seed)
    recs: List[Dict[str, object]] = []

    def add(q: str, a: str, kind: str) -> None:
        recs.append({"question": q, "answer": a, "source": "guard:" + kind,
                     "lang": "eng", "citation": None})

    for q in OOD_QUESTIONS:
        medical = any(w in q.lower() for w in ("hurts", "pain", "medicine", "dizzy"))
        add(q, OOD_MEDICAL_REPLY if medical else OOD_REPLY, "out_of_domain")

    for prov, act in FAKE_PROVISIONS:
        for tpl in BAIT_TEMPLATES:
            add(tpl.format(prov), BAIT_REPLY.format(prov=prov, act=act), "hallucination_bait")
    for case in FAKE_CASES:
        add("What was held in " + case + "?", CASE_BAIT_REPLY.format(case=case),
            "hallucination_bait")
        add("Cite " + case + " for me.", CASE_BAIT_REPLY.format(case=case),
            "hallucination_bait")

    for q, act in UNSAFE:
        add(q, UNSAFE_REPLY.format(act=act), "unsafe")

    for q in UNCERTAIN:
        add(q, UNCERTAIN_REPLY, "uncertainty")

    for q, body in ADVICE_BODIES.items():
        add(q, ADVICE_PREFIX + body + ADVICE_SUFFIX, "advice_boundary")

    for q, place in FOREIGN:
        add(q, FOREIGN_REPLY.format(place=place), "jurisdiction")

    # Upsample each bucket to n_per_bucket with a light question-side paraphrase, so
    # no single bucket is a rounding error next to tens of thousands of law pairs.
    by_kind: Dict[str, List[Dict[str, object]]] = {}
    for r in recs:
        by_kind.setdefault(str(r["source"]), []).append(r)

    out: List[Dict[str, object]] = []
    for group in by_kind.values():
        out.extend(group)
        extra = []
        while len(group) + len(extra) < n_per_bucket:
            base = rng.choice(group)
            extra.append({**base, "question": rng.choice(PARAPHRASE) + str(base["question"])})
        out.extend(extra)

    rng.shuffle(out)
    return out


if __name__ == "__main__":
    data = build()
    for kind, n in sorted(Counter(str(r["source"]) for r in data).items()):
        print(f"{kind:<28} {n}")
    print("total", len(data))
