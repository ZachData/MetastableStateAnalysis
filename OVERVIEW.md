<!-- OVERVIEW.md -->
# What this project is, and where it stands

This is the plain-English companion to `PROJECT.md`, which is the file to read if you
need to actually run something. This one is for understanding what's going on and why,
without the cross-references and machine-checkable detail. It's a snapshot as of
September 2026 — the technical files are the ones that stay current day to day.

## The starting idea

There's a mathematical theory (Geshkovski et al., "A Mathematical Perspective on
Transformers") that models what happens inside a transformer language model in a
particular way: think of every token in a sentence, at every layer, as a particle
sitting on the surface of a high-dimensional sphere. As the sentence passes through the
network's layers, attention moves these particles around — pulling some together,
pushing others apart. The theory proves that if you let this process run long enough
(more layers than any real network has), every particle eventually collapses onto the
exact same point. All the tokens become indistinguishable. That's a strange prediction,
because it's obviously not what real language models do — they clearly keep track of
different tokens meaning different things all the way through.

The paper's authors noticed something else on the way to that collapse, in their toy
version of the model: the particles don't move straight to one point. They pass through
stretches where they sit in a handful of separate clusters and stay there for a while —
plateaus — before eventually merging further. They call these "metastable states," and
they explicitly say they can't explain why this happens; it's an open question, backed
by small-scale numerical experiments, not a theorem.

This project's original question was simple: does that plateau behavior — metastability
— actually show up in real, trained language models, at realistic sizes, with all the
architectural complexity (multi-head attention, feed-forward layers, learned weights)
the simplified theory leaves out? The answer, from the first phase of work, was yes —
consistently, across several different trained model families. That was surprising
enough to be worth a write-up on its own (referred to internally as "Blog 1": the
finding that trained transformers actively *resist* the collapse their own architecture
mathematically predicts, rather than sliding toward it).

Everything since has been chasing that result. The project's identity by now is really:
take token representations as physical particles moving under forces, seriously, and see
how far that framing goes.

## The discipline this project insists on

A project like this lives or dies on whether its findings hold up, and geometry research
on neural networks is notoriously easy to fool yourself with — patterns that look
meaningful can be artifacts of how you measured, or of the specific model you happened to
look at, or just noise you didn't control for. So a large fraction of the actual
engineering effort here goes into *not believing things too early*.

Concretely, that means: every prediction gets written down — in plain language, with a
stated way it could turn out to be wrong — before the code that tests it exists. Findings
aren't allowed to be quietly revised after the fact; if a prediction needs updating, the
update gets added as a dated note rather than editing the original. And there's a formal
statistical layer (borrowed from a 2025 paper on sequential hypothesis testing) whose
whole job is to convert "does this look real?" into an actual, calibrated number, so that
"we found something" comes with a stated rate of how often that kind of claim would be
wrong by chance, and pre-registration means that if you keep looking at more results after
the fact you're not silently allowed to inflate that confidence.

Almost every substantial piece of code in this repository has a companion piece whose
entire job is to try to break it — feed it a case with a known right answer and check it
gets there, or feed it deliberately meaningless data and check it correctly reports
"nothing here" instead of finding a false pattern. Several real bugs have been caught this
way over the project's life, including at least one this month that would have silently
corrupted a months-long analysis with no visible symptom.

## The journey so far, briefly

The work moved through several stages before landing on what it's doing now:

- **Does the metastability itself show up in trained models?** Yes — tested across
  several different model architectures, and the plateau pattern reliably appeared.
- **What does the geometry inside attention actually look like — is it "pulling
  together" or "pushing apart," and is it real, straight-line movement or is some of it
  rotation?** This phase found that attention's effect on particles splits cleanly into
  those two flavors — an attractive channel and a repulsive channel — and separately, that
  a lot of what looks like movement is actually rotation around a fixed point rather than
  straight-line translation, which needed its own careful accounting to avoid being
  misread as much bigger structural change than it actually is.
- **A run of ideas that mostly didn't pan out or got parked.** Several follow-on
  attempts — using sparse ("dictionary") decompositions of the internal activity, looking
  for the same structure in a different model family, steering the geometry directly —
  were tried, tested honestly, and in most cases came back null or were shelved rather
  than forced to show something. That's treated as useful information, not as failure:
  a couple of write-ups exist specifically to record "we checked and it wasn't there,"
  which closes off wrong turns for good instead of leaving them to be re-tried later.
- **Moving from a handful of separately-trained models to one model's training history.**
  The project switched to Pythia-410M specifically because Pythia publishes dozens of
  checkpoints of the *same* model at different points during its training — meaning, for
  the first time, the geometry could be watched forming over time, rather than only
  compared at the finish line across different architectures.

That last shift is what set up the current work.

## What's being tested right now

The current phase asks a more ambitious question than anything before it: can this
particle-and-forces language actually *explain* things that mechanistic interpretability
research already knows about — using its own vocabulary, not by translating into English
descriptions after the fact?

The test case is a well-known, well-studied piece of transformer behavior called an
**induction head** — actually a two-part circuit, discovered by other researchers, that
lets a language model complete repeated patterns. If a sentence contains "... Alice went
to the store ... Alice", the model can predict "went" next, because it has learned to
notice "the token *before* the previous occurrence of the current token" and copy
whatever came after it. That's a genuinely useful trick and a big part of how these
models handle context, and it's known to appear at a fairly specific point during
training, not from the very start.

The particle-language restatement of that circuit is a **two-stage relay**: one part of
the network "tags" a token during an early layer (an attractive pull between adjacent
positions), and a second part of the network, later on, finds anything carrying that tag
and copies its context forward. If the particle framework is doing real explanatory work
and not just redescribing things, then this relay pattern — measured purely
geometrically, from the forces particles exert on each other — should start showing up
in the model's internals at *the same point in training* that the model's actual,
externally-observable behavior (successfully completing repeated patterns) starts
showing up. If the two show up at wildly different times, or the geometric pattern is
already there from the very beginning regardless of training, that would be evidence the
geometric account isn't really capturing what "induction head" means.

Answering that well took two separate measurements, tracked across nineteen points in
the model's training:

1. **The geometric side.** How often does the two-stage relay pattern actually occur, per
   attention head, at each point in training — counted directly from the internal forces,
   with no reference to what the model outputs.
2. **The behavioral side.** How strongly does the model actually *attend* to the
   pattern-completing token when it should, at each of those same training points — this
   is the standard, externally-observable measure interpretability researchers already use
   to say "this head does induction."

Both of those had to be built and measured for real, on real model checkpoints, rather
than estimated or assumed.

## Why a plain count wasn't enough, and what had to be built to fix it

An early pass just counting the relay pattern ran into an immediate problem: a huge
share of the count turned out to be explained by nothing more interesting than how many
repeatable positions a given piece of text happens to contain — some example texts just
have vastly more opportunities to look like a repeated pattern than others, entirely
independent of whether the model has "learned" anything. Measured directly: how often a
piece of text's own repeated-token count predicted the raw geometric count, essentially
the entire variance was explained that way. A rising raw count over training could easily
just mean nothing more than "the network processes text slightly differently as it
trains" rather than "a real circuit is forming."

So a proper control had to be built: for every checkpoint, take the model's real internal
activity and randomly reshuffle *which particles* each attention head's forces are
actually pointing at, while keeping everything else about that head — how many
connections it makes, and the full character of the forces it exerts — exactly as
measured. That gives a "what would this count look like by pure chance, given how this
text is structured and how active this head is" baseline, separately for every head at
every point in training. The real geometric count minus that chance baseline is the part
that isn't explainable by coincidence — and that's the number worth comparing to the
model's actual behavior.

Building that control rigorously — and building the matching real-behavior measurement,
which needed its own careful handling so the two measurements weren't secretly reading
the same underlying number twice — was most of the recent work. It also surfaced a subtle
bug during construction: an early, faster version of the reshuffling step had a mix-up
that would have silently mixed up different pieces of text with each other, corrupting
the whole comparison with no visible error. A test built specifically to guard against
exactly that kind of mistake caught it immediately, before it touched the real analysis.

## Where that left things

With both real measurements and the chance-level control finally all in place and run
end to end on the real training checkpoints: the raw geometric pattern does sit well
above its chance baseline at every point where the model is actively forming this
capability — several times the level pure coincidence would produce, which on its own is
a reassuring sign that something real is being measured and not noise.

But the actual test — do the two curves (the geometric pattern's above-chance excess, and
the model's real behavioral capability) turn on at the *same point* in training, more
than you'd expect from randomly pairing them up — came back **inconclusive**. Not
disproven, not confirmed: the evidence isn't strong enough either way to call it, at
least not yet, and not with the amount of computational budget spent checking so far
(the check is itself a Monte Carlo estimate, and a version with roughly twice the
sampling effort was queued to see whether the number moves, before being deliberately
paused to write this down instead of leaving a multi-hour job unattended).

That's a legitimate, if unglamorous, scientific outcome. It doesn't mean the particle
framing is wrong — the geometric structure is clearly real and clearly forms during
training, well above chance. It means this particular test, at this level of precision,
can't yet say whether it's forming in *lockstep* with the specific behavior it's meant to
explain, which is a narrower and harder claim than "the structure exists at all." Nothing
about this result has been entered into the project's permanent findings ledger — that's
a separate, deliberate decision that hasn't been made yet, on purpose, because printing a
number isn't the same thing as deciding it's worth standing behind.
