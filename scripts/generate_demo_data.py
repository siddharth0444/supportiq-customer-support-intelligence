"""Generate deterministic synthetic ticket records for local development."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "sample_tickets.csv"
EXAMPLES = {
    "billing": [
        "I was charged twice for my monthly subscription.",
        "Please explain the invoice on my latest payment.",
        "I need a refund for an incorrect charge.",
        "My card payment failed but money was deducted.",
    ],
    "technical": [
        "The mobile app crashes whenever I open settings.",
        "I cannot upload a document from the dashboard.",
        "The page keeps loading and never displays my account.",
        "The latest update broke the notification feature.",
    ],
    "account": [
        "I am locked out and cannot reset my password.",
        "Please help me update the email on my account.",
        "I need to enable two-factor authentication.",
        "My account verification link has expired.",
    ],
    "shipping": [
        "My order has not arrived by the expected delivery date.",
        "The tracking page has not updated for three days.",
        "The package was delivered to the wrong address.",
        "Can you tell me when my replacement will ship?",
    ],
    "product": [
        "The product arrived damaged and does not turn on.",
        "I would like to know whether this feature is available.",
        "The size and description do not match the listing.",
        "I am disappointed with the quality of this item.",
    ],
}
PREFIXES = ["Hello support, ", "Hi team, ", "I need assistance: ", "Please help. ", ""]
SUFFIXES = [" Thank you.", " This is urgent.", " Please advise.", " I appreciate your help.", ""]


def main(rows_per_category: int = 40, seed: int = 42) -> None:
    rng = random.Random(seed)
    rows = []
    for category, examples in EXAMPLES.items():
        for index in range(rows_per_category):
            text = rng.choice(PREFIXES) + rng.choice(examples) + rng.choice(SUFFIXES)
            rows.append({
                "ticket_id": f"DEMO-{category[:3].upper()}-{index + 1:03d}",
                "text": text,
                "category": category,
                "source": "synthetic_demo",
            })
    rng.shuffle(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {len(rows)} synthetic tickets to {OUTPUT}")


if __name__ == "__main__":
    main()
