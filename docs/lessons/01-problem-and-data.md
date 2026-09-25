# Lesson 1 (2026) — The problem and the data, in plain words

**The big idea in one line:** we have one clean list of businesses and two messy lists. For each business in the clean list, we must find every messy entry that is secretly the same business, without ever wrongly accusing a stranger.

---

## 1. The story: three phone books

Imagine three phone books of the same city, made by three different companies:

- **Book 1 (Source 1)** is **neat**: every business appears once, spelled properly. This is our **reference list**.
- **Books 2 and 3 (Sources 2 and 3)** are **messy**: typed by hurried clerks. The same shop may appear 3 or 4 times with different spellings. Some entries are for shops that aren't in Book 1 at all.

**Our job:** for every business in Book 1, write down every entry in Books 2 and 3 that is the same shop.

Nothing links the books: there's no shared ID number. We only have what's written in each entry, **name, address, country**, and we must judge "same shop or not?" from those alone.

> **Tech words:** this job is called **Entity Resolution (ER)** or **record linkage**. Each entry is a **record**. The real-world shop behind it is the **entity**. Book 1 is the **reference source**.

---

## 2. What one "answer" looks like

Here's one real business from Book 1 and its true matches from Books 2 and 3:

```
Book 1:  Espinoza & Davis Industrials LLC | 5425 152nd Avenue, Ramsey, MN

Book 2:  Espinoza & LLC Industrials Davis | 5425 152ND AVENUE, RAMSEY, MN          ← words shuffled
Book 2:  Espinoza & Davis Industrials L.L.C. | 5425 152ND AVENUE, RAMSEY, MN       ← dots in LLC
Book 2:  Espinoza & Davis Industrials Llc | MN, RAMSEY, 5425 152ND AVENUE          ← address back-to-front
Book 3:  LLC Espinoza & Davis Industrials | Anoka CITY, 1425 152nd Ave, Minnesota  ← wrong house number, county name
Book 3:  Onyxveo formerly Espinoza & Davis Industrials LLC | 5425 152nd Avenue ... ← "formerly" (old name!)
```

All five are the **same business**. Our answer row would be:

```
S1-xxxx    S2-a,S2-b,S2-c,S3-d,S3-e
```

**By the numbers (train):** a business usually has **3** matches (90% have 6 or fewer, the most is 11). Matches come half from Book 2 and half from Book 3.

---

## 3. Three kinds of businesses you'll meet

### (a) The normal case: 94% of businesses have 1–11 matches
Like Espinoza above.

### (b) The loner: 5.6% have **no** match at all
The right answer is an **empty list**. For these, saying *nothing* earns a **perfect 1.0**, and naming even one match earns **0**.

```
Book 1:  Holy Synagogue | Sapulpa, OK, 2204 Park Street        ← has NO true match

Book 2:  holy synagogue co | WA, LYNNWOOD, 6716 200TH ST        ← same name, different town
Book 2:  Holy Synagogue Corporation | 1609 8TH ST, MASON CITY, IA   ← belongs to a DIFFERENT Holy Synagogue
Book 2:  *** Holy Synagogue | 716 BOLWING CAVE RD, SEVIERVILLE, TN  ← yet another one
```

The names look right, but **none of these is our synagogue**. Different towns, different streets.

### (c) Namesakes: different businesses with the same name
**40% of businesses in Book 1 share their name with at least one other business.** "Primary Care Group" alone appears 253 times.

```
Book 1:  Behavioral Health Partners Inc | 185 Locust View Way, Troy, OH
Book 1:  Behavioral Health Partners     | 1908 Caroline Street, Town Of Campbell, WI
Book 1:  Behavioral Health Partners Inc | 1331 Pleasant View Drive, Des Moines, IA
```

Three different businesses. Only the **address** tells their entries apart:

```
Book 2:  BEHAVIORAL HEALTH-PARTNERS  | #185 LOCUST VIEW WAY, TROY, OH           → belongs to the Troy one
Book 2:  BEHAVIORAL HEALTH PARTNERS  | 1908-1912 CAROLINE ST, TOWN OF CAMPBELL, WI → belongs to the Wisconsin one
```

**Lesson:** a name alone can never decide. **The address is the tie-breaker.**

### Also: the strangers (not a business type, but they matter)
About **26% of entries in Books 2 and 3 match nobody** in Book 1. They're traps: some look very similar to a real Book 1 business.

**One more rule of the world:** each entry in Books 2 and 3 belongs to **at most one** Book 1 business. Never two.

---

## 4. Why is it hard? Five kinds of mess

| Mess | Real example | What a human sees instantly |
| --- | --- | --- |
| **Typos and garbling** | `Christ [Chape1]` · `A SOBTRNATMGIC ENVIRONMENTAL` | "Chape1" is "Chapel"; the second is a mangled "Strategic" |
| **Shuffled words and extras** | `LLC Espinoza & Davis Industrials` · `Ectosyn dba Christ Chapel` · `Mr Prabhav Business Services` | Same core name, with extra words around it |
| **Different script** | `ಲೋಟಸ್ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` (Kannada) = "Lotus Marketing Private Limited" | Only readable if you know the script. **24% of India's Book 2 names are in a local script** |
| **Address written differently** | `2100 Cameron Drive, Unit APARTMENT G` vs `# Apartment G, Dundalk, 210 Cameron Drive` | Same place, reordered, with a digit dropped |
| **Missing pieces** | `Siobhan's Bike Shop | (no address)` · `CAMERON DRIVE, DUNDALK, MD` (no number) | Less to go on, so more doubt |

Even the answer key has surprises: `Christ Center` is a true match for `Christ Chapel` at the same address. A name can change a lot while the address stays put.

**By the numbers:** only **11%** of true matches have exactly the same name, and only **7%** exactly the same address. So "exact match" finds very little. That's why our first try scored low.

---

## 5. How we're marked (F0.5), in plain words

For each Book 1 business, the judge asks two questions:

- **"Of the entries you named, how many were right?"** → **precision** (are you *careful*?)
- **"Of the true entries, how many did you find?"** → **recall** (are you *thorough*?)

F0.5 combines both, but **cares twice as much about being careful**. Think of a detective:

- **Arresting an innocent person** (naming a wrong match) is **very bad**.
- **Letting a guilty person slip away** (missing a match) is **bad, but less bad**.

Then it **averages over every Book 1 business equally**. A business with 11 matches counts the same as a loner.

README example: we name 3 entries, 2 are right, and we found both true ones. Precision 2/3, recall 1, score **0.714**.

> **Tech words:** **precision**, **recall**, **F-beta** (β = 0.5 means precision-heavy), **macro-average** (average per business, not per pair). A wrong match is a **false positive** (a "false merge"); a missed one is a **false negative**.

---

## 6. How any solution works: shortlist, then judge

There are about **10 million** entries in Books 2 and 3, and **1.7 million** businesses in Book 1. Comparing every pair would be about 17 trillion comparisons, which is impossible.

So every solution has two steps, like a police investigation:

1. **Shortlist (the round-up):** for each business, quickly gather a small group of *possible* suspects (same country, similar-ish name or address). It must be fast, and it must **not miss the real ones**, because **anyone not rounded up can never be matched later**.
2. **Judge (the interrogation):** look carefully at each suspect on the shortlist and decide "same business or not", with a confidence.

```
Book 1 business ──► shortlist ~20–50 suspects ──► judge each ──► keep the confident ones ──► answer
```

> **Tech words:** the shortlist is **blocking** or **candidate generation**. The judge is a **pairwise classifier**. How many true matches survive the shortlist is **blocking recall**, and it's the ceiling for everything after.

---

## 7. What our first submission did, and where it failed

**Our rule:** shortlist only entries whose cleaned name is *exactly* the same; then keep them only if the house number is also the same.

**Score: 0.56 on the leaderboard** (0.554 on our own train check; they agree, so our self-check is trustworthy).

| Measure | Result | Meaning |
| --- | --- | --- |
| When we said "match", were we right? | **93%** of the time | We're careful. Good |
| How many true matches did we find? | **only 34%** | We're far from thorough. **This is where the points go** |
| True matches that never even got shortlisted | **51%** | The "exact same name" round-up is far too strict (typos, scripts, extra words) |
| Shortlisted true matches thrown away by the house-number rule | **31%** | E.g. `2100` vs `210`, or `HN 567 797` read as house number 567 |
| Businesses where we named **nothing** but should have | **30.5%** | Each of these scores **0** |

**In one line: we're a careful detective who rounds up far too few suspects.** The top of the leaderboard is 0.986, so a much better round-up and judge are clearly possible.

---

## 8. France: the country we've never seen

The test set has **France** (15% of test businesses). Training has **none**. It looks like this:

```
Shidokan Ecole SARL       | 17 Rue de Valmy, Lille, Hauts-de-France
Ets Art SASU              | # 47 RUE CHARLES MARIONNEAU, BORDEAUX, Nouvelle-Aquitaine
SASU Soutien Àmis (France)| Saint-Nazaire, Loire-Atlantique, 28 R. PIERRE WALDECK-ROUSSEAU
```

New legal words (SARL, SAS, SASU, EURL, SCI), new street words (Rue, R., BD), accents, and the house number *before* the street. **No score will tell us how we do on France.** Only human eyes can check it.

---

## 9. Where your eyes come in

Once the new shortlist and judge exist, you'll get lists like:

- **Wrong merges:** "we said these are the same, but they're not." *Why did we fall for it?*
- **Misses:** "these were the same, but we didn't find or accept them." *What made it hard?*
- **France decisions:** "here's what we matched for France." *Do these look right to you?*

For each one, write down **what you notice**, in your own words. Patterns you see there, which the model doesn't, are the input only your team can give.

**Warm-up now (5 minutes):** re-read the examples in sections 3 and 4. For each, ask yourself: *what exactly told me these are (or aren't) the same business?* Write your answers in the notes below.

## Your notes / ideas

_Write what you notice here._

---

### Word list

| Word | Simple meaning |
| --- | --- |
| Entity resolution / record linkage | Finding entries that are the same real thing |
| Record / entity | One written entry / the real business behind it |
| Reference source (Source 1) | The neat list; each business once |
| Singleton | A business with no match; the right answer is empty |
| Distractor | An entry that matches nobody in the reference list |
| Namesake / collision | Different businesses with the same name |
| Precision / recall | Careful (few wrong) / thorough (few missed) |
| F0.5 | Score that cares twice as much about careful as thorough |
| Macro-average | Average per business, each counts equally |
| False merge / false positive | Saying "same" when it's not |
| Miss / false negative | Saying nothing when it was the same |
| Blocking / shortlist / candidates | Quickly gathering possible matches |
| Blocking recall | Share of true matches that make the shortlist (the ceiling) |
| Pairwise classifier | The model that judges each (business, entry) pair |
| Transliteration | Converting a script (e.g. Kannada) into Latin letters |
