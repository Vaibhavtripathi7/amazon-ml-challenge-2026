# Test vs train — side by side

_Built by `scripts/test_analysis.py` from every row of both splits. Facts only; what they mean is the team's call._

- **Train** columns use the ground truth. **Test** columns have no labels: where a row needs an owner, it uses **our submitted predictions** (`matching_results-v3.tsv`, v2) and says so.
- "Name key" = our cleaned, word-sorted business name (`parse_name`), so `Davis & Espinoza LLC` = `Espinoza Davis L.L.C.`. "House" = the parsed house number.
- **Bold** = test differs from train by more than 20% (relative).

## 1. Size

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| S1 businesses | 1,323,633 | **663,106** | 883,188 | 809,986 | — | 259,452 |
| S2 records | 3,016,817 | **1,871,330** | 2,017,799 | 2,312,565 | — | 703,378 |
| S3 records | 3,170,056 | **1,945,701** | 2,115,547 | 2,405,000 | — | 731,615 |
| S2+S3 records per S1 | 4.67 | **5.76** | 4.68 | **5.82** | — | 5.53 |

## 2. Group sizes: records attached to each S1

Train = **truth**. Test = **our v2 predictions** (not truth). A gap here is either a real data difference or a v2 mistake — this table alone can't tell which.

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| mean records per S1 | 3.46 | 3.40 | 3.46 | 3.28 | — | 3.61 |
| S1 with 0 (empty list) | 5.6% | 5.6% | 5.6% | 6.0% | — | 4.0% |
| S1 with 1 | 5.4% | 6.4% | 5.4% | **8.6%** | — | 6.9% |
| S1 with 2–3 | 41.1% | 41.8% | 41.0% | 42.1% | — | 37.9% |
| S1 with 4–5 | 36.5% | 35.4% | 36.6% | 33.2% | — | 37.2% |
| S1 with 6+ | 11.4% | 10.8% | 11.5% | 10.1% | — | 14.0% |
| records attached to no S1 | 26.0% | **40.9%** | 26.0% | **43.8%** | — | 34.7% |

## 3. What the text looks like (S2+S3 records)

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| no address | 3.6% | 2.9% | 3.0% | **2.4%** | — | 3.0% |
| address has a house number | 89.9% | 91.8% | 80.7% | 82.4% | — | 92.8% |
| name in an Indian script | 0.0% | 0.0% | 18.2% | 18.4% | — | 0.0% |
| name looks like a domain/handle | 5.1% | 4.2% | 4.0% | 3.2% | — | 4.0% |
| name has formerly / dba / fka | 1.6% | 1.3% | 1.0% | **0.8%** | — | 0.8% |
| name has brackets | 3.5% | 3.4% | 7.4% | 7.4% | — | 8.6% |
| name starts with junk (<<, ***, --) | 2.5% | 2.4% | 1.8% | 1.6% | — | 3.9% |
| name ALL CAPS | 12.1% | 11.6% | 8.6% | 8.2% | — | 13.1% |
| median name length | 23 | 24 | 27 | 28 | — | 20 |
| median address length | 35 | 35 | 64 | 64 | — | 40 |

## 4. What the text looks like (S1)

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| no address | 0.0% | 0.0% | 0.0% | 0.0% | — | 0.0% |
| address has a house number | 99.9% | 99.9% | 81.8% | 81.8% | — | 99.5% |
| name starts with junk | 0.1% | 0.1% | 0.1% | 0.1% | — | 3.8% |
| name ALL CAPS | 0.0% | 0.0% | 0.0% | 0.0% | — | 0.0% |
| median name length | 22 | 22 | 27 | 27 | — | 19 |
| median address length | 34 | 34 | 76 | 76 | — | 48 |

## 5. Namesakes: how many S1 share a name key

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| S1 whose name key is shared with another S1 | 47.9% | 40.1% | 54.4% | 53.7% | — | 53.1% |
| S1 whose name key is shared by 10+ S1 | 19.2% | 16.8% | 33.4% | 33.2% | — | 25.2% |
| records whose name key is in NO S1 | 43.5% | 49.5% | 45.1% | 51.9% | — | 36.3% |
| records whose name key is in exactly 1 S1 | 28.1% | 29.1% | 23.5% | 20.4% | — | 27.6% |
| records whose name key is in 2–9 S1 | 16.8% | **12.1%** | 11.6% | 9.7% | — | 18.6% |
| records whose name key is in 10+ S1 | 11.6% | **9.2%** | 19.8% | 18.0% | — | 17.5% |
| records with name key AND house equal to some S1 | 35.8% | 29.0% | 27.8% | 22.6% | — | 43.0% |

## 6. Records around one S1 (S1 with a name key no other S1 has)

If test kept every record but dropped some S1, a different pattern shows up here and in section 5 ("name key in NO S1").

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| mean records with the same name key | 2.52 | 2.80 | 2.41 | 2.57 | — | 3.26 |
| … 0 records | 8.1% | **6.3%** | 9.8% | 8.8% | — | 4.6% |
| … 1–3 records | 68.5% | 63.8% | 68.4% | 66.0% | — | 55.0% |
| … 4–6 records | 22.1% | **27.9%** | 20.8% | 23.8% | — | 36.5% |
| … 7+ records | 1.3% | **2.0%** | 1.0% | **1.5%** | — | 4.0% |

## 7. How reliable is a name-key hit? (train = truth, test = our v2)

Train: how often the truth agrees with the simple name-key link. Test: how often **v2** attached such a record. Large train-vs-test gaps are where v2 behaves differently on test than truth does on train.

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| record with a 1-S1 name key is attached to THAT S1 | 84.8% | 76.8% | 91.2% | 86.8% | — | 82.2% |
| record with a name key in NO S1 is attached to some S1 | 58.4% | **41.1%** | 54.9% | **35.1%** | — | 38.7% |
| record with name key + house hit is attached to some S1 | 99.7% | 99.7% | 98.2% | 94.6% | — | 96.6% |
| record with a name key in 10+ S1 is attached to some S1 | 84.2% | 72.3% | 84.1% | **65.2%** | — | 75.1% |

## 8. Places

| | US train | US test | India train | India test | France train | France test |
| --- | --- | --- | --- | --- | --- | --- |
| record city appears among S1 cities | 83.6% | 84.0% | 87.4% | 87.5% | — | 91.2% |
| S1 per city (mean over cities) | 46.41 | **31.65** | 2.52 | 2.49 | — | 48.63 |
| distinct S1 cities | 28,248 | **20,751** | 350,669 | 325,458 | — | 5,335 |

## 9. Top states (share of S1)

**US**: tx 10.0% (train 10.1%) · ny 7.6% (train 7.7%) · nc 7.2% (train 7.1%) · oh 6.3% (train 6.4%) · il 5.7% (train 5.7%) · tn 4.5% (train 4.5%) · va 4.5% (train 4.5%) · az 4.4% (train 4.3%) · ma 4.3% (train 4.3%) · wa 3.7% (train 3.7%) · in 3.7% (train 3.7%) · md 3.0% (train 3.1%)

**India**: maharashtra 21.7% (train 21.7%) · delhi 13.6% (train 13.7%) · uttar pradesh 8.3% (train 8.3%) · karnataka 8.0% (train 7.9%) · tamil nadu 7.2% (train 7.2%) · gujarat 6.5% (train 6.5%) · west bengal 6.5% (train 6.4%) · telangana 6.3% (train 6.3%) · haryana 4.1% (train 4.1%) · kerala 4.0% (train 4.0%) · rajasthan 3.6% (train 3.6%) · bihar 2.6% (train 2.6%)

**France**: hauts de france 39.1% (train —) · nouvelle aquitaine 32.8% (train —) · pays de la loire 28.0% (train —)

