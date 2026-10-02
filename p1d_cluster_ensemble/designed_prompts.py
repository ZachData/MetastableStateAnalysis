"""
p1d_cluster_ensemble/designed_prompts.py — three prompts with known content,
for the "second first check" of unit 1 (`design-1d.md` "Unit 1: move the
text", added after `/challenge-pr` on #129, finding 7) and unit 4's
designed-content control.

FROZEN 2026-10-02, BEFORE ANY FORWARD PASS ON THEM
--------------------------------------------------
This file is committed on its own, ahead of the runner that reads it, so
git shows the texts, their labels and the pass rule predate every model
output on them. Only the tokenizer has seen them: words were swapped for
single-token ones before freezing (`crimson`, `scarlet`, `beige`, `maroon`,
`turquoise`, `magenta`, `indigo`, `lavender`, `amber`, `ivory`, `teal`,
`owl` are 2+ tokens and were replaced; names and role nouns likewise).
Nothing below is revised after a run. These are new text, not v2: whether
they join a battery is the user's (`design-1d.md` "Unit 4").

The three prompts (`design-1d.md` "Unit 4"):

- ``designed_category_list``: 72 words, 24 each of animals, colours and
  body parts, comma-separated after one sentence. The order is one draw of
  ``numpy.random.default_rng(0).permutation(72)`` over the lists below
  (animals, colours, body parts, in that order), pasted literally, so a
  word's category is not a function of its position (a strict rotation
  would make category = position mod 3).
- ``designed_prose_code``: five prose paragraphs and four Python blocks,
  alternating, on one topic (reading and counting words in a file). Labels
  are the spans: ``prose`` / ``code``.
- ``designed_entities``: a short narrative in which three named people
  recur under several names: ``teacher`` (Miss Anna Klein), ``sailor`` (Captain
  Peter Marsh), ``surgeon`` (Doctor Hans Weber). Only the name, title and
  role words are labelled; every one is a single token. After deduplication
  (T3) each entity is the first occurrence of each of its strings.

THE READOUT RULE (placed, written before any run)
-------------------------------------------------
Inputs are unit 1's: the passage cloud after T1–T3 with passage offset 0
excluded, level-set HDBSCAN (`admit.layer_groups`), every layer L1–24, both
frames, ``min_cluster_size`` 2 (primary) and 4. Preambles: all three
continuations (`wiki_paragraph`, `hdbscan_code`, `latex_monograph`); none of
them is a designed prompt's own.

1. **Content group.** A P = 0 group is a content group for label ``c`` if at
   least ``CONTENT_MIN`` (3) of its kept members carry ``c`` and they are at
   least ``CONTENT_PURITY`` (0.75) of its labelled members. Placed.
2. **Classified.** Only content groups that are stable at P = 0 (median
   subsample Jaccard >= 0.5, unit 1's noise-floor rule) are classified, by
   unit 1's classes (moves / preamble-dependent / opening-bound /
   context-bound).
3. **Pass.** At step143000, centred frame, ``min_cluster_size`` 2, pooled
   over the three prompts and L1–24: at least ``PASS_SHARE`` (0.5) of the
   classified content groups **move**. Each prompt's share is reported
   beside it, and a prompt with no classified content group is reported as
   such (the prediction "groups that hold one category / entity" failed
   there, which is not the readout failing). No classified content group in
   any prompt is a failure: "moves" has then not been shown to fire.
4. **Step 0** is run at P = 0 only, as the baseline the design names ("at
   step 0, not"): its count of content groups is reported beside
   step143000's. Chance purity is not small with two or three labels (a
   3-member group is pure by chance about 1/9 of the time with three
   balanced labels), so step 0's count is what step143000's is read
   against.

If the check fails, v1's trained cells are not read until that is
understood (`design-1d.md`).

Tier 1: exploratory, unregistered.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Sequence, Tuple

#: Rule 1, placed.
CONTENT_MIN = 3
CONTENT_PURITY = 0.75
#: Rule 3, placed.
PASS_SHARE = 0.5

ANIMALS = ("dog cat horse cow pig sheep goat mouse rabbit wolf fox bear lion tiger deer "
           "duck frog goose eagle snake whale monkey camel donkey").split()
COLOURS = ("red blue green yellow purple brown black white grey pink orange violet navy "
           "gold silver tan cyan olive cream coral rust bronze charcoal rose").split()
BODY = ("arm leg nose ear knee elbow finger toe ankle wrist chin neck shoulder lip tooth "
        "thumb hip heel cheek jaw skull spine rib belly").split()

_LIST = ("A class of children was asked to sort these words into three groups before the "
         "bell rang: violet, whale, sheep, elbow, frog, rib, wrist, black, rabbit, bronze, "
         "coral, rust, pig, jaw, toe, orange, yellow, cream, snake, donkey, fox, cheek, horse, "
         "belly, skull, tooth, bear, goose, rose, purple, chin, eagle, tan, cow, cat, navy, "
         "red, monkey, dog, goat, camel, green, gold, spine, knee, nose, shoulder, heel, duck, "
         "leg, grey, blue, charcoal, deer, wolf, silver, hip, ear, white, lip, cyan, arm, "
         "tiger, lion, mouse, thumb, neck, finger, brown, olive, ankle, pink.")

_PROSE = (
    "Reading a file line by line is one of the first things a new programmer learns to do. "
    "The idea is simple: open the file, look at each line in turn, and close the file when "
    "you are finished. In Python the with statement takes care of the closing for you.",
    "Often you want to keep only some of the lines. Suppose every line holds a name and an "
    "age separated by a comma, and you care only about people older than thirty. You can "
    "split each line and turn the second part into a number before comparing it.",
    "Counting is another common task. A dictionary maps each word to the number of times it "
    "has been seen so far, and its get method supplies a default when a word appears for "
    "the first time.",
    "Finally, it helps to sort the results so that the most frequent words come first. The "
    "sorted function accepts a key, and asking for reverse order puts the largest counts at "
    "the top of the list.",
    "With these few pieces, a short script can already answer real questions about a text, "
    "such as which words an author leans on most and which ones appear only once.",
)
_CODE = (
    'with open("notes.txt") as handle:\n'
    '    for line in handle:\n'
    '        print(line.strip())',
    'adults = []\n'
    'with open("people.csv") as handle:\n'
    '    for row in handle:\n'
    '        name, age = row.split(",")\n'
    '        if int(age) > 30:\n'
    '            adults.append(name)',
    'counts = {}\n'
    'for word in text.split():\n'
    '    counts[word] = counts.get(word, 0) + 1',
    'ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)\n'
    'for word, n in ranked[:10]:\n'
    '    print(f"{word}: {n}")',
)
_SEP = "\n\n"


def _prose_code() -> Tuple[str, List[Tuple[int, int, str]]]:
    parts = [(_PROSE[0], "prose")]
    for c, p in zip(_CODE, _PROSE[1:]):
        parts += [(c, "code"), (p, "prose")]
    text, spans = "", []
    for i, (s, lab) in enumerate(parts):
        if i:
            text += _SEP
        spans.append((len(text), len(text) + len(s), lab))
        text += s
    return text, spans


_ENTITIES_TEXT = (
    "Three people kept the small harbour town running through the long winter. Anna Klein, "
    "the teacher, opened the schoolhouse every morning at seven. Captain Peter Marsh, an old "
    "sailor, ran the only boat that still crossed the bay when the ice allowed it. Doctor Hans "
    "Weber, the town's surgeon, lived above the pharmacy and was woken at all hours.\n\n"
    "In January the storm cut the road. Miss Klein moved her pupils into the church hall, where "
    "the stove was larger, and taught them arithmetic with dried beans. Marsh refused to sit "
    "idle; the captain mended nets on the quay and listened to the radio for news of the "
    "supply ship. Weber walked from farm to farm, because the doctor could not drive on the "
    "frozen hill.\n\n"
    "When a fisherman's son broke his arm on the ice, it was Anna who carried him to the "
    "pharmacy, and Hans who set the bone while the teacher held the lamp. Peter took the "
    "boat out the next morning to fetch plaster from the island, and the sailor came back "
    "with the plaster and a sack of flour.\n\n"
    "By March the road was open again. The surgeon wrote in his diary that the town had "
    "managed because Klein, Marsh and he had each done one thing well. The teacher said it "
    "was the beans; the captain said it was the boat; the doctor said nothing and went to "
    "bed."
)
ENTITY_WORDS = {"teacher": ("Anna", "Miss", "Klein", "teacher"),
                "sailor": ("Captain", "captain", "Peter", "Marsh", "sailor"),
                "surgeon": ("Doctor", "doctor", "Hans", "Weber", "surgeon")}


def _word_spans(text: str, words: Dict[str, Sequence[str]]) -> List[Tuple[int, int, str]]:
    spans = []
    for lab, ws in words.items():
        for w in ws:
            spans += [(m.start(), m.end(), lab) for m in re.finditer(rf"\b{re.escape(w)}\b", text)]
    return sorted(spans)


def prompts() -> Dict[str, Tuple[str, List[Tuple[int, int, str]]]]:
    """``{key: (text, [(char_start, char_end, label), ...])}``, in a fixed order."""
    pc_text, pc_spans = _prose_code()
    cats = {"animal": ANIMALS, "colour": COLOURS, "body": BODY}
    return {"designed_category_list": (_LIST, _word_spans(_LIST, cats)),
            "designed_prose_code": (pc_text, pc_spans),
            "designed_entities": (_ENTITIES_TEXT, _word_spans(_ENTITIES_TEXT, ENTITY_WORDS))}


KEYS = tuple(prompts())


def token_labels(text: str, spans: Sequence[Tuple[int, int, str]], tokenizer) -> Tuple[List[int], List]:
    """
    Token ids of ``text`` (no special tokens) and each token's label: the
    label of the span its non-space characters fall in, else ``None``. A
    token that straddles two labels refuses.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = []
    for a, b in enc["offset_mapping"]:
        while a < b and text[a].isspace():
            a += 1
        hit = {lab for s, e, lab in spans if a < e and s < b} if a < b else set()
        if len(hit) > 1:
            raise ValueError(f"token at chars {a}:{b} straddles labels {sorted(hit)}")
        labels.append(hit.pop() if hit else None)
    return list(enc["input_ids"]), labels


def designed_hash() -> str:
    """Short hash of the texts and spans, recorded in every run that reads them."""
    h = hashlib.sha256()
    for k, (t, s) in prompts().items():
        h.update(k.encode() + b"\0" + t.encode() + b"\0" + repr(s).encode() + b"\0")
    return h.hexdigest()[:12]
