# Business Entity Resolution

Our solution for the Amazon ML Challenge 2026. Three sources describe the same businesses with no shared IDs, and for
every business in the reference list (Source 1) we have to find all of its records in the other two sources. Names
and addresses are noisy on purpose: typos, missing words, reordered addresses, legal suffixes that come and go, names
written in Indian scripts, and a third country (France) that only appears in the test set.

Final public leaderboard score: **0.9738** macro F0.5. Our first full pipeline scored 0.946.

## How it works

```
 Source 2/3 records                                 Source 1 businesses
        |                                                  |
        +------------------ clean + parse -----------------+     stage 0: names, addresses, house numbers,
                                  |                                       Indian-script words -> English
                         candidate search                        stage 1: 8 exact keys + TF-IDF on the name
                                  |                                       and on name + address
                LightGBM ranker, keeps the top 3 per record               ranker prunes to about 6 per business
                                  |
                    LightGBM judge (about 60 features)           stage 2: one probability per pair
                                  |
             only pairs with judge probability 0.01 to 0.99
                                  v
                   xlm-roberta-base cross-encoder                stage 3: reads both records together,
                                  |                                       blended with the judge
            each record -> its best business, if the score clears that country's threshold
```

The expensive model only runs where it is needed. The two LightGBM stages are cheap enough for 10.7M test pairs and are
already sure about most of them. The transformer re-scores only the uncertain ones. On held-out data that raised India
from 0.9752 to 0.9838 and the US from 0.9873 to 0.9904. India's 0.9838 is close to 0.9843, the best its candidate set
allows.

## What made the difference

- **Validation that looks like the test set.** Our validation score was 0.975 while the leaderboard said 0.946. Test
  has about twice as many look-alike records per business as train. Validation copies with the same density predicted
  what the leaderboard did, including that stricter thresholds win.
- **Reading the data before modelling it.** We compared 73,531 true pairs to see how records get corrupted (dropped
  suffixes, swapped words, phonetic misspellings, changed house numbers) and built the parser around that.
- **Care with France.** There is no French training data, so we measured how each model carries over to a country it
  never saw (train on the US, test on India). The transformer carried over badly (0.8909 against 0.9755 for LightGBM),
  so for France the blend leans on LightGBM and on the French address rules.

## Results

| version | change | public leaderboard |
|---|---|---|
| rules baseline | exact name key + house number | 0.56 |
| v2 | LightGBM ranker + judge | 0.946 |
| v3 | Indian-script word list, website-name splitting, name-frequency features, deeper searches | 0.953 |
| v3, stricter thresholds | France 0.7 instead of 0.4 | 0.957 |
| v3 + transformer | cross-encoder on the uncertain pairs | 0.9708 |
| thresholds up | India 0.85, US 0.95, France 0.85 | 0.9731 |
| final | India 0.90, US 0.97, France 0.90 | 0.9738 |

What did not help, with numbers, is in the [write-up](docs/writeup.md): linking records to each other, one model per
country, self-training, a character n-gram search, a pretrained embedding search, and synthetic French training pairs.

## Running it

The competition data is not included. With the dataset folder (`train/` and `test/`):

```bash
uv sync                      # Python 3.12, CPU dependencies
uv run pytest                # 58 tests

# stages 0-2 on CPU, about 6 hours on a 4-core Kaggle machine; a restart skips finished steps
uv run python scripts/run.py --data <dataset dir> --out run --threads 4

# stage 3 needs a GPU (one T4 is enough) and two more packages
uv pip install torch transformers
export PYTHONPATH=src
python scripts/ce_rerank.py build --pairs run/dev_pairs.parquet --data <dataset dir> --out ce/train.parquet --easy-frac 0.3
python scripts/ce_rerank.py train --in ce/train.parquet --model-dir ce/model --epochs 2 --bs 32 --max-len 160 --max-pairs 600000
python scripts/ce_rerank.py train --in ce/train.parquet --model-dir ce/model_us --train-countries US --epochs 1 --max-pairs 150000
python scripts/ce_rerank.py eval --in ce/train.parquet --model-dir ce/model_us --all-pairs
python scripts/ce_rerank.py eval --in ce/train.parquet --model-dir ce/model --lo 0.01 --hi 0.99 --tag _b
python scripts/ce_rerank.py apply --test-pairs run/output/test_pairs.parquet --data <dataset dir> --model-dir ce/model \
    --tag _b --unseen-blend ce/model_us/blend.json --unseen-band-only --unseen-threshold 0.7 --out ce/out

# final files: output/matching_results.tsv and output/candidate_pairs.tsv
python scripts/final_thresholds.py --pairs ce/out/test_pairs_blend.parquet --data <dataset dir> --out output \
    --thr India=0.90 US=0.97 France=0.90
```

Training saves checkpoints and test scoring saves its work in chunks, so a dropped GPU session continues where it
stopped. `scripts/mistakes.py` breaks a validation run down into where the score is lost, with examples.

## Layout

```
src/amlc/      cleaning and parsing (norm.py, clean.py), candidate search (keys.py, blocking.py, stage.py),
               judge features (judge_features.py), metric (metric.py), learned word lists (resources/)
scripts/       run.py (stages 0-2), prepare.py, judge.py, ce_rerank.py (stage 3), final_thresholds.py,
               mistakes.py, synth_pairs.py, and the two scripts that build the word lists
tests/         tests for the parser, candidate search, features and metric
docs/          writeup.md: methodology, experiments and numbers
```

## Team

Team Pacman: Vaibhav Tripathi (lead), Bazaru Lokesh Ram Chand, S J Krithi.

## License

MIT, see [LICENSE](LICENSE).
