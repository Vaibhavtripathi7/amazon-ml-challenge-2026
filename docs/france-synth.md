# France synthetic supervision — self-contained plan (2026-09-27 17:05). Facts, research, design, gates.

## Why (measured)
- France is unseen in train; we have no French labels. The cross-encoder trained on US/India does NOT transfer to an
  unseen country: US-only model on India scores 0.8909 alone vs 0.9755 for our judge (Cell 6).
- The leaderboard gap is test-only (dev 0.981 vs LB 0.957); France is a large share of it under the most likely split.

## Research
- Sudowoodo (ICDE 2023, arXiv 2207.04122): same-entity pairs made by rule-based corruptions of real records train a
  matcher without labels (up to +16% F1).
- Ditto (arXiv 2004.00584): augmentation operators; which ones help is dataset-specific → take ours from our data.
- Domain adaptation of rankers (DUQGen 2404.02489; hard-negative remining 2501.14434): positives generated from
  target-domain items, hard negatives mined from the target domain.

## Design
- **Positives:** corrupted copies of real French S1 records, with the generator's operators at the rates measured on
  73,531 real US/India true pairs (below), using France's own filler vocabulary.
- **Hard negatives, exact by construction:** S1 is deduplicated (official README), so two different French S1 are
  different businesses. Negative = a corrupted copy of S1 A paired with A's real French sibling B (same door, same
  street, or shared name words) — the decoy pattern seen in France.
- **Training:** continue the trained cross-encoder on synthetic French pairs mixed with real US/India pairs (no
  forgetting), then re-score France's uncertain test pairs only.

## Operators measured on real true pairs (India / US)
suffix dropped 29% / 18% · suffix changed 19% / 2% · suffix added 3% / 8% · ALL CAPS 27% / 13% · all lower 23% / 8% ·
core word replaced 16% / 19% (53% of replacements are a generic filler: center/services/service/partners; 42% are
phonetic garbles: knsltimg=consulting, phainans=finance, menejmemt=management) · typo 7% / 9% · word added 12% / 8% ·
word dropped 1% / 8% · accent added 5% / 7% · reorder 1% / 3% · junk prefix 2% / 3% · domain form 5% / 7% ·
alias (formerly/dba) 1% / 2% · address: fewer parts 47% / 12%, ALL CAPS 16% / 48%, first number digit changed 6% / 8%,
No./#/H.No prefix 9% / 4%, empty 4% / 5%, parts reordered 3% / 3%.
Decoys differ: their replacement words are specific real words (enterprises, trading, global, silver, mumbai), only
6% filler words.

## France vocabulary (test records vs French S1, label-free)
Filler/added: participations, holding, groupe, snc, (France), com; aliases dba/formerly/doing/known; homoglyph 5arl;
dotted legal forms S.A.R.L.; "et". Address: departments instead of regions (Nord, Gironde, Loire-Atlantique,
Pas-de-Calais), R./Rue, bis→B, N°, ALL CAPS, St for Saint.

## Gates (set before any result)
1. **Gate 1 (realism, labelled):** apply the SAME generator to India S1; continue the US-only model (Cell 6, India
   0.8909 alone) on synthetic India + real US; score REAL India held-out pairs. Pass = CE-alone India clearly above
   0.8909 (≥ +0.02) AND the blend ≥ the judge (0.9755).
2. **Gate 2 (no harm):** after continuing the main model with synthetic France, US/India held-out must not drop
   (blend within −0.001 of 0.9838 / 0.9904).
3. Only if both pass: re-score France's uncertain test pairs → upload 4. Otherwise upload 4 = stricter variant.

## Schedule (deadline midnight)
generator + local test by 17:45 · gate 1 on GPU 17:45–18:10 · v4 + transformer 18:10–19:40 (main path) ·
France continuation + gate 2 + France re-score 19:40–20:30 · upload 4 by 21:00 · zip 21:00–22:00.
