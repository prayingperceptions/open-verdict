"""THE DEMO: review 300 agent outputs in ONE review() call.

Run:  python examples/review_300.py

It generates 300 fake agent outputs — clean answers, obvious failures
(tracebacks, empty strings, TODOs, leaked "null"), and nasty edge cases
(duplicated texts, truncated output, missing required terms) — then
judges all 300 in a single pass with typed verdicts + confidence scores.

Cost: $0.00. API calls: 0.
"""

from __future__ import annotations

import random
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from verdict import review  # noqa: E402

random.seed(42)

RUBRIC = """Review agent-written customer support replies.
MUST: apologi
MUST: refund
MUST NOT: policy
"""

GOOD_BODIES = [
    "We sincerely apologize for the delay with your order. I've issued a full refund to your card, "
    "which should appear within 3-5 business days. Again, I apologize for the inconvenience.",
    "I apologize for the mix-up — that's entirely on us. Your refund has been processed and a "
    "confirmation email is on its way. Please accept my apology for the trouble.",
    "Apologies for the late shipment. I've refunded the delivery charge and escalated your ticket "
    "so this doesn't happen again. Thank you for your patience.",
    "I'm sorry you received the wrong item. A replacement is on its way and I've refunded the "
    "original charge in full. My apologies for the hassle.",
    "Please accept my apology for the billing error. A full refund was issued this morning and "
    "you'll see it post shortly. I apologize again for the confusion.",
    "I apologize for the long wait on support. Your refund is confirmed — reference #RFD-88112 — "
    "and I've credited your account for the shipping cost as well.",
    "Apologies — the duplicate charge was our mistake. I've reversed it, so the refund should "
    "land within 2 business days. Sorry for the scare.",
    "I'm sorry the promo code didn't apply. I've manually applied the discount and issued a "
    "partial refund for the difference. Apologies for the extra step.",
]

FAILURES = [
    "Traceback (most recent call last):\n  File \"agent.py\", line 42, in reply\nRuntimeError: null response",
    "",
    "   \n\t  ",
    "TODO: write the actual reply later",
    "Error: failed to generate response (undefined)",
    "null",
    "Your request violates company policy and cannot be processed. Per policy, no exceptions.",
    "We regret to inform you that per our policy no refund can be offered at this time.",
    "ok",
    "no",
    "Sorry. (see internal docs for actual answer...)",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Refund TBD.",
    "I cannot help with that. This is outside my training data xxx placeholder.",
    "FAILED: agent timed out waiting for tools",
]


def build_batch(n: int = 300) -> list[dict]:
    items = []
    for i in range(n):
        roll = random.random()
        if roll < 0.55:
            # good output, with occasional term sabotage (edge case);
            # each gets a unique ticket reference so identical good bodies
            # don't trip the duplicate detector — real agent outputs vary.
            body = random.choice(GOOD_BODIES)
            if random.random() < 0.15:
                body = body.replace("refund", "credit")  # drops a MUST term
            body += f" [ticket #{random.randint(10000, 99999)}]"
            items.append({"id": f"agent-{i:03d}", "text": body})
        elif roll < 0.80:
            items.append({"id": f"agent-{i:03d}", "text": random.choice(FAILURES)})
        else:
            # edge cases
            kind = random.random()
            if kind < 0.4:
                # duplicated text across the batch (stuck agent)
                items.append({"id": f"agent-{i:03d}", "text": GOOD_BODIES[0]})
            elif kind < 0.6:
                # truncated mid-sentence
                items.append(
                    {
                        "id": f"agent-{i:03d}",
                        "text": "We sincerely apologize for the delay and I have processed your refund wh...",
                    }
                )
            elif kind < 0.8:
                # missing MUST term entirely
                items.append(
                    {
                        "id": f"agent-{i:03d}",
                        "text": "Your order has been noted. Someone will contact you eventually.",
                    }
                )
            else:
                # whisper-short
                items.append({"id": f"agent-{i:03d}", "text": "done"})
    return items


def main() -> None:
    items = build_batch(300)
    print(f"Reviewing {len(items)} agent outputs in ONE review() call...\n")

    start = time.perf_counter()
    verdicts = review(items, RUBRIC, backend="heuristic")
    elapsed = time.perf_counter() - start

    counts = {"pass": 0, "fail": 0, "retry": 0}
    for v in verdicts:
        counts[v["verdict"]] += 1

    per_sec = len(verdicts) / elapsed if elapsed > 0 else float("inf")
    bar = lambda n: "█" * int(40 * n / len(verdicts))

    print(f"⏱  wall-clock: {elapsed:.3f}s  ({per_sec:,.0f} verdicts/sec)")
    print(f"💰 cost: $0.00   🌐 api calls: 0   📦 dependencies: 0\n")
    print("VERDICT DISTRIBUTION")
    print(f"  ✅ pass   {counts['pass']:>3}  {bar(counts['pass'])}")
    print(f"  ❌ fail   {counts['fail']:>3}  {bar(counts['fail'])}")
    print(f"  🔁 retry  {counts['retry']:>3}  {bar(counts['retry'])}")

    print("\nSAMPLE FAILS (what a serial LLM review would charge you per-token to find):")
    shown = 0
    for v in verdicts:
        if v["verdict"] == "fail" and shown < 5:
            print(f"  [{v['id']}] conf={v['confidence']:.2f} — {v['reason']}")
            shown += 1

    print("\nSAMPLE RETRIES:")
    shown = 0
    for v in verdicts:
        if v["verdict"] == "retry" and shown < 3:
            print(f"  [{v['id']}] conf={v['confidence']:.2f} — {v['reason']}")
            shown += 1

    print(
        f"\n{len(verdicts)} verdicts. One pass. No queue. No invoice. "
        "That's the whole pitch."
    )


if __name__ == "__main__":
    main()
