<!-- trackC_writeup/lit-C4pc.md -->
# C4′ (c), the replicate pair: literature scan before its design freezes (2026-11-17)

Trigger (`CLAUDE.md` "Literature scans"): C4′ (c) opens for its design (STATE Blocked 165 (a),
the user's, 2026-11-17), and its readings may be registered. Scope: (1) how much of a run's
variance comes from the data order with the init held fixed; (2) how far induction's date moves
between runs; (3) whether anyone has replicated a warmup-removed run. Three web searches
(2026-11-17); no full text read, so every outside row is [S], a search engine's summary only.
Not repeated: `lit-C4.md` (Qu & Liu, Olsson et al., Kalra & Barkeshli), `lit-C4p.md` (Singh et
al., 2408.11029, 2503.12811, 2505.15353, 2509.26643).

## 1. Data order with the init fixed

| work | what [S] | bearing |
|---|---|---|
| Dodge et al. 2002.06305 | BERT fine-tuning on GLUE with the init seed and the data-order seed varied separately: the two contribute about equally to run-to-run variance; some orders are consistently better | the only separation of the two seeds found, and it is fine-tuning (the init seed sets ~2k head weights), not pretraining from scratch. It says an order-only replicate need not be a small perturbation |
| a Pythia-14M retrain (search summary, source not identified) | the same Pile data in another order on other hardware is "not the same run" | no number; a reminder only |

No pretraining study that varies the order at a fixed init was found. PolyPythias (C6 U0's
seeds 1–9) vary init and order together, so their spread (70m: V 256 or 512, the centred date
512–1000, induction 1000 on all ten) bounds an order-only spread from above only if the two
sources add, which nothing here shows.

## 2. Induction's date between runs

| work | what [S] | bearing |
|---|---|---|
| 2511.16893, "Predicting the emergence of induction heads in LM pretraining" | batch size and context length set the emergence point; six configurations rerun on two more seeds give tight intervals; a fitted law predicts the point regardless of seed | at C4's batch and context (both fixed in (c)), the lean is a small spread of induction's date on A0′; item 9 is a prediction about F′ relative to A0′, so a small spread makes it sharper |
| Aoyama et al. (2505-era, "Predicting the formation of induction heads") | the frequency and reliability of repeated bigrams in the data drive formation | the data's statistics, not their order, by this account; an order-only replicate keeps the corpus (D2 (a)), so it predicts little movement |
| 2510.12071 (influence dynamics, stagewise data attribution) | data influence changes by stage | none for (c) |

## 3. A warmup-removed run, replicated

| work | what [S] | bearing |
|---|---|---|
| Kosson et al. 2410.23922 ("Analyzing & reducing the need for LR warmup in GPT training"; named in `lit-C4.md` as not returned, returned now) | warmup's role is to limit the size of the early updates; a modified optimizer makes it largely unnecessary | if F's transient is the early updates' size, it is the optimizer's at that rate, not the data's, so item 7's transient should replicate under another order. (c) holds the optimizer fixed, so it cannot test this, only predict from it |
| 2510.03164 (a theory of warmup with LM ablations) | 10–20 % linear warmup best; without warmup, much worse at the peak rate | one seed per setting in the summary; F trained without diverging at 1e-3 (C4 U2) |
| AAAI 2026 (ojs 40284) | standard warmup makes training sensitive to seeds and hyperparameters | a motive for a method, not a controlled replicate |

**Verdict:** no replicate of a warmup-removed pretraining run, and no order-only variance study
in pretraining, was found. (c) would be the first of either on a Pythia model that this scan
can see. Nothing found predicts the order-only spread of C2's counts' dates.

## What the scan changes in the design

1. **The order-only spread is not assumed small** (Dodge et al.). The replicate's A0′ is read as
   a measurement first (its dates against A0's), and every registered lean is stated against
   A0′, never against A0, so that a large order spread breaks no prediction by itself.
2. **D2 recommends a permutation of C4's own rows** over fresh rows: it holds the corpus fixed
   and varies only the order, which is the variable 2511.16893 and Aoyama et al. point at as
   *not* setting induction's date. A fresh-rows replicate would vary both.
3. **Item 7's transient is predicted to replicate** (Kosson et al.'s early-update account). The
   design records this as the lean's reason, not as evidence.

## Parked

1. **Kosson et al.'s full text** (why: if it dates a no-warmup transient in GPT training, item 7
   has an outside reference; cost: one reading; could change: item 7's wording in `paper/` §7,
   not the reading). Read before U3's reading, or dropped at the close-out with why.
