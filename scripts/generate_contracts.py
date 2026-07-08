"""Generate a small synthetic corpus of contracts + an eval set.

Produces realistic-looking (but entirely fictional) agreements so the repo is
self-contained and runnable without shipping any real/proprietary documents.
"""
from __future__ import annotations

import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "contracts")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_cases.json")

PARTIES = [
    ("Acme Analytics LLC", "Northwind Retail Inc."),
    ("BluePeak Software Pvt. Ltd.", "Sterling Foods Co."),
    ("Helios Energy Partners", "Vertex Manufacturing GmbH"),
    ("Cobalt Cloud Services", "Meridian Bank N.A."),
]

TEMPLATE = """MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is entered into as of {date}
by and between {provider} ("Provider") and {client} ("Client").

1. DEFINITIONS
1.1 "Services" means the {service} services described in each Statement of Work.
1.2 "Confidential Information" means non-public information disclosed by either party.

2. TERM
2.1 This Agreement commences on the Effective Date and continues for an initial
term of {term} months, renewing automatically for successive twelve (12) month
periods unless either party gives written notice of non-renewal.

3. FEES AND PAYMENT
3.1 Client shall pay Provider the fees set out in the applicable Statement of Work.
3.2 Invoices are payable within {net} days of the invoice date. Late amounts
accrue interest at {interest}% per month.

4. TERMINATION
4.1 Either party may terminate this Agreement for convenience upon {notice} days'
prior written notice.
4.2 Either party may terminate immediately if the other party commits a material
breach that remains uncured for {cure} days after written notice of the breach.

5. CONFIDENTIALITY
5.1 Each party shall protect the other's Confidential Information using at least
reasonable care and shall not disclose it for a period of {conf} years.

6. LIMITATION OF LIABILITY
6.1 Except for breaches of confidentiality or indemnification obligations, each
party's aggregate liability shall not exceed the fees paid in the {liab} months
preceding the claim.

7. INDEMNIFICATION
7.1 Provider shall indemnify Client against third-party claims that the Services
infringe any intellectual property right.

8. GOVERNING LAW
8.1 This Agreement is governed by the laws of {law}, without regard to conflict
of law principles. Disputes shall be resolved by binding arbitration in {venue}.

9. DATA PROTECTION
9.1 Provider shall process personal data only per Client's instructions and shall
notify Client of any personal data breach within {breach} hours.
"""

SERVICES = ["data analytics", "cloud hosting", "software development", "managed IT"]
LAWS = ["the State of Delaware, USA", "England and Wales", "the Republic of India"]
VENUES = ["Wilmington, Delaware", "London", "Mumbai"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(7)
    cases = []
    for i, (provider, client) in enumerate(PARTIES, start=1):
        notice = random.choice([30, 60, 90])
        cure = random.choice([15, 30, 45])
        net = random.choice([15, 30, 45])
        text = TEMPLATE.format(
            date=f"2025-0{i}-15",
            provider=provider,
            client=client,
            service=random.choice(SERVICES),
            term=random.choice([12, 24, 36]),
            net=net,
            interest=random.choice([1, 1.5, 2]),
            notice=notice,
            cure=cure,
            conf=random.choice([3, 5]),
            liab=random.choice([6, 12]),
            law=random.choice(LAWS),
            venue=random.choice(VENUES),
            breach=random.choice([24, 48, 72]),
        )
        doc_id = f"msa_{i:02d}"
        with open(os.path.join(OUT_DIR, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        cases.append(
            {
                "question": f"What is the termination notice period in {provider}'s agreement?",
                "expected_doc_id": doc_id,
                "expected_keywords": [str(notice)],
            }
        )
        cases.append(
            {
                "question": f"How many days to cure a material breach under {doc_id}?",
                "expected_doc_id": doc_id,
                "expected_keywords": [str(cure)],
            }
        )

    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    print(f"Wrote {len(PARTIES)} contracts to {OUT_DIR} and {len(cases)} eval cases.")


if __name__ == "__main__":
    main()
