# Data analysis — 2026 Business Entity Resolution

_2026-09-25, first pass (hour 0–1). Facts only; what to do about them is the team's call._

## 1. The match structure

| Fact | Number | Plain meaning |
| --- | --- | --- |
| S1 businesses in train | 2,206,821 | The "reference list" we must fill in |
| S1 with **no** match (singletons) | **5.6%** (123k) | Predicting nothing is right for these, and scores 1.0 |
| Matches per S1 | median 3, 90% ≤ 6, max 11 | Most businesses appear several times in S2/S3 |
| Can one S2/S3 record belong to two S1 businesses? | **Never** (7.64M matched IDs, all unique) | Each S2/S3 record has at most one owner |
| S2/S3 train records that match **nothing** | about **2.7M of 10.3M (26%)** | Distractors: records whose business isn't in S1 at all |
| Matches split S2 / S3 | 48% / 52% | Both sources matter equally |
| S1 with ≥ 1 S2 match / ≥ 1 S3 match | 87% / 88% | |

Distribution of matches per S1: 0 → 123k · 1 → 119k · 2 → 375k · 3 → 531k · 4 → 484k · 5 → 322k · 6 → 165k · 7 → 64k · 8+ → 23k.

## 2. Countries: test ≠ train

| | US | India | France |
| --- | --- | --- | --- |
| Train (all sources) | 60% | 40% | — |
| **Test** S1 | 38% | **47%** | **15%** (259k S1 rows) |

- In 69,504 checked matched pairs, **the match always had the same country as S1 (100%)**.
- **France isn't in train at all**, yet it's 15% of the test businesses we're scored on.
- India's share rises from 40% to 47%.

## 3. How different are true matches? (from 69,504 matched pairs)

- Exact same name (ignoring case): **only 10.8%**
- Exact same address (ignoring case): **only 7.2%**

Real examples of **one** business across sources:

```
S1: Espinoza & Davis Industrials LLC | 5425 152nd Avenue, Ramsey, MN
S2: Espinoza & LLC Industrials Davis | 5425 152ND AVENUE, RAMSEY, MN          ← word order
S2: Espinoza & Davis Industrials L.L.C. | 5425 152ND AVENUE, RAMSEY, MN       ← suffix format
S2: Espinoza & Davis Industrials Llc | MN, RAMSEY, 5425 152ND AVENUE          ← address reordered
S3: Espinoza & Davis Industrials LLC | 1425 152nd Ave, Anoka CITY, Minnesota  ← different number + county
S3: Onyxveo formerly Espinoza & Davis Industrials LLC | 5425 152nd Avenue ... ← "formerly"
```
```
S1: A Strategic Environmental | 2 Muldowney Circle, Unit Apartment B, Poughkeepsie, NY
S2: A SOBTRNATMGIC ENVIRONMENTAL | 2 MULDOWNEY CIR, POUGHKEEPSIE, NY   ← garbled word
S3: A Environmental (Strategic) | 2 Muldowney Cir, # Apartment B, ...   ← brackets, reorder
S3: The Á Strategic Environmental | ...                                  ← added "The", accent
```
```
S1: Lotus Marketing Private Limited | G.S.Residency, Site No.8, ..., Bangalore, Karnataka
S2: ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | G.S.RESIDENCY, ..., BANGALORE, ಕರ್ನಾಟಕ      ← name in Kannada script
S2: Lotus Marketing ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | ...                                   ← half English, half Kannada
S3: Lotus Marketing Private Ltd | H.no 3-568 G.s.residency, ..., Bengaluru, ... ← Ltd, Bengaluru
```

Other noise seen: junk prefixes (`***`, `...`, `--`), empty address (≈3% of S2/S3), `Saint` for `St`, `UP`/`MH` for state names, state written in local script.

## 4. What makes wrong merges easy

- **39.6% of S1 businesses share their (cleaned) name with at least one *other* S1 business.** Top repeats: "primary care group" (253 different businesses), "ear nose throat group" (251), "pediatric group" (222).
- So **a name alone can't decide a match**. Address (and other fields) has to separate them.

## 5. Scripts and text

| | Names in non-Latin script (test S2) |
| --- | --- |
| India | **23.6%** (Devanagari, Kannada, …) |
| US | 0% |
| France | 0% |

- S1 is always in Latin script (0% Devanagari in S1).
- Median lengths: name ≈ 25 characters, address ≈ 40–50 characters.

## 6. France (test only): what it looks like

```
S1: Shidokan Ecole SARL          | 17 Rue de Valmy, Lille, Hauts-de-France
S1: Pique Ecole EURL             | 20 BD Jules Verne, Nantes, Pays de la Loire
S2: Nantes Compagnie SCI         | 6 RUE GUILLAUME GROOTAERS, NANTES
S2: SASU Soutien Àmis (France)   | Saint-Nazaire, Loire-Atlantique, 28 R. PIERRE WALDECK-ROUSSEAU
S2: Ets Art SASU                 | # 47 RUE CHARLES MARIONNEAU, BORDEAUX, Nouvelle-Aquitaine
```

French legal suffixes (SARL, EURL, SAS, SASU, SCI), street words (Rue / R. / BD), `N°`, regions and departments, accents.

## 7. Size and our hardware

- Train 12.5M records, test 11.7M records. **Test S2+S3 alone is about 10M records** to search through for 1.7M S1 businesses.
- Loading one ~500 MB source file in pandas peaks at about 2–2.5 GB. **This laptop has 5.7 GB RAM and no GPU.**
