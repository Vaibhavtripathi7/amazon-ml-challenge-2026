# Lesson 1 — How we get scored (SMAPE)

**The big idea in one line:** before building anything, understand exactly how the judges mark your answers. The way you're marked decides how you should guess.

- Code: [`src/amlc/metrics.py`](../../src/amlc/metrics.py)
- Tests: [`tests/test_metrics.py`](../../tests/test_metrics.py) — run `uv run pytest`
- Experiment: [`experiments/01_smape_demo.py`](../../experiments/01_smape_demo.py) — run `uv run python experiments/01_smape_demo.py`

---

## 1. The game we're playing

Imagine a friend shows you a product (a chocolate, a shampoo, a TV) and you have to guess its price. Your friend knows the real price and gives you a mark for how close you were.

Now, what does "close" mean?

- You guess a **₹50 chocolate** is ₹60. You're off by ₹10. That's a big mistake for a chocolate: 20% off.
- You guess a **₹50,000 TV** is ₹50,010. You're also off by ₹10. But for a TV that's a tiny mistake: almost perfect.

So the same ₹10 mistake can be terrible or tiny. A fair judge should look at **how wrong you were compared to the price**, not just the rupee difference.

> **Tech word:** marking by "percent wrong" instead of "rupees wrong" is called a **percentage error** (or **relative error**).

---

## 2. SMAPE: the judge's rule

SMAPE is the rule the 2025 challenge used to mark every guess. For each product:

1. Take how far off you were: `|guess − real|`. The `| |` bars mean "ignore minus signs", so off by −10 and off by +10 both count as 10.
2. Divide by the **average of the real price and your guess**.
3. Turn it into a percent.

Then average that percent over all products. That's your score. **Lower is better.**

```
SMAPE = 100/n · Σ |F − A| / ((|A| + |F|) / 2)
```

Reading the formula like a sentence: *"for each of the n products, take the miss, divide by the average of real (A) and forecast (F), add them all up, divide by n, times 100."*

- **Best score: 0.** Every guess exactly right.
- **Worst score: 200.** The worst case is guessing ₹0 for something that costs money (section 3 explains why it's 200 and not 100).

> **Tech words:** **A** = actual (the real price), **F** = forecast (your guess). **SMAPE** = Symmetric Mean Absolute Percentage Error. "Mean" = average, "absolute" = ignore the minus sign, "percentage" = divide by the price.

---

## 3. Let's mark some guesses by hand

The real price is always ₹100 here, so you can compare the rows.

| Real | Your guess | How we mark it | Score for this row |
| --- | --- | --- | --- |
| 100 | 100 | off by 0 | **0%** (perfect) |
| 100 | 110 | off by 10, average of 100 and 110 is 105 → 10 ÷ 105 | **9.5%** |
| 100 | 150 | off by 50, average is 125 → 50 ÷ 125 | **40%** |
| 100 | 50 | off by 50, average is 75 → 50 ÷ 75 | **66.7%** |
| 100 | 200 | off by 100, average is 150 → 100 ÷ 150 | **66.7%** |
| 50 | 0 | off by 50, average is 25 → 50 ÷ 25 | **200%** (the worst possible) |
| 0 | 0 | 0 ÷ 0 → we decide this is **perfect** | **0%** |

Look at rows 3 and 4. Both are off by ₹50, but **guessing too low (₹50) is punished more than guessing too high (₹150)**. That's because when you guess low, the "average" you divide by gets smaller, so the percentage gets bigger.

Now look at rows 4 and 5:

- Guessing **half** the price (₹50) → 66.7%
- Guessing **double** the price (₹200) → 66.7%

**Exactly the same score!** Here's the key idea of the whole lesson:

> SMAPE doesn't care about "₹ off". It cares about **"how many times too big or too small"**. Double and half are equally wrong.

---

## 4. Why this means we should think in "times", not "rupees"

Think of prices on a ladder where **each step up doubles the price**:

```
₹25 → ₹50 → ₹100 → ₹200 → ₹400
```

On this ladder, ₹50 and ₹200 are both **one step away** from ₹100. That's exactly how SMAPE sees them: equally wrong.

So instead of teaching our model to guess the price, we teach it to guess **which step of the ladder** the price is on. Then the model's sense of "close" matches the judge's sense of "close".

> **Tech word:** the "step on the ladder" is the **logarithm** (**log**) of the price. Training on `log(price)` instead of `price` is called a **log transform** of the **target** (the target is the thing we're predicting).
>
> We use `log1p(price)` = log(1 + price), so a price of 0 doesn't break the maths (log of 0 is undefined). To turn the model's answer back into rupees we use the reverse, `expm1`.

---

## 5. How the code works (plain version)

```python
denom = (np.abs(y_true) + np.abs(y_pred)) / 2     # the "average" we divide by
diff = np.abs(y_pred - y_true)                    # how far off, ignoring minus
ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
return float(100 * ratio.mean())                  # average, as a percent
```

The only tricky line is the third one. It says: *"divide `diff` by `denom`, **but only where `denom` isn't zero**. Where it is zero, just write 0."*

Why? If the real price is 0 and we guessed 0, we'd compute 0 ÷ 0, which has no answer. In Python that becomes **NaN** ("not a number"). And one NaN makes the whole average NaN, like a single rotten egg spoiling the whole cake. So we decide in advance that 0-for-0 is a perfect guess.

There's also a safety check: if the list of real prices and the list of guesses aren't the same length, the code **stops with an error** instead of quietly giving a wrong score. A misaligned submission should fail loudly.

> **Tech words:** `np` is **NumPy**, the Python library for fast maths on lists of numbers (called **arrays**). `y_true` = real prices, `y_pred` = our predictions. `ValueError` is the kind of error Python raises for "you gave me bad input".

---

## 6. Why we wrote the tests *before* the code

A **test** is a tiny program that checks "when I give the function this, it should give back that". For example: *"SMAPE of real 100, guess 110, should be 9.52."*

We wrote all six tests **first**, ran them, and watched them all **fail**, because the function didn't exist yet. Then we wrote the function until they all **passed**.

Why bother watching them fail? Imagine a smoke alarm that has never beeped. You don't know if it works. Watching a test fail proves it can actually catch a mistake.

**On hackathon day:** the moment the real metric is announced, write its tests straight from the rules page. If you misread the formula (say, forgot the "divide by 2"), a hand-worked example catches it in 2 minutes instead of after a wasted leaderboard submission.

> **Tech words:** this "test first, watch it fail, then make it pass" habit is called **TDD** (Test-Driven Development). We run tests with a tool called **pytest**.

---

## 7. The experiment: what kind of guessing does SMAPE like?

We made up 50,000 fake prices that look like real shop prices: **most things are cheap, a few are very expensive**. The middle price is ₹20, but a few items cost up to ₹2,280.

### Part A — if you could only say ONE price for every product, which should it be?

Think of a classroom where 29 kids have ₹20 pocket money and one kid has ₹2,000.

- The **average** is about ₹86. The one rich kid drags it up, and it's wrong for almost everyone.
- The **middle value** (line everyone up by amount, pick the one in the middle) is ₹20. It's right for almost everyone.

Here's what happened with our fake prices:

| Single price guessed | What it is | SMAPE |
| --- | --- | --- |
| ₹33.2 | the **average** | 76.7 (worse) |
| ₹20.1 | the average of the ladder steps (**geometric mean**) | 70.1 |
| ₹20.0 | the **middle value** (**median**) | 70.1 |
| ₹20.0 | the best single price, found by trying hundreds | 70.1 |

**What we learn:** guessing near the *average* is clearly worse. The judge prefers guesses near the *middle* or on the *ladder*.

> **Tech words:** a model learns *how* to guess from its **loss function**, the rule it uses to punish its own mistakes while learning.
> - **MSE** (Mean Squared Error) punishes big misses a lot (it squares them), and pushes the model toward the **average**. On raw prices that's bad for us.
> - **MSE on log prices** pushes toward the ladder-average (**geometric mean**). Good.
> - **L1 / MAE** (Mean Absolute Error) pushes toward the **median**. Good, and it doesn't panic over a few weird prices.
> - **Huber** is a mix: gentle like L1 for big misses, smooth like MSE for small ones.

### Part B — should we nudge all our guesses up or down?

We made a pretend "decent model" (right on average, with some random error) and multiplied every guess by a number `s`:

| Multiply by `s` | 0.85 | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 |
| --- | --- | --- | --- | --- | --- | --- |
| SMAPE | 40.17 | 39.10 | 38.49 | **38.28** | 38.43 | 38.90 |

The best is **1.00**, meaning don't nudge at all. If the model isn't leaning one way, nudging only hurts.

**So why keep this trick in our plan?** Real models often *do* lean. They usually play it safe and guess too high for cheap things and too low for expensive things. Trying a few `s` values is a 1-minute check that catches that lean. We keep the nudge only if it really improves our score.

> **Tech words:** a model that consistently leans one way has **bias**. The nudge is a **global scale factor**. We test it on **OOF predictions** (out-of-fold: guesses the model made on data it was *not* trained on, so the check is honest; lesson 3 explains folds).

### A mistake this experiment caught

Our plan first said "SMAPE rewards guessing a bit low." **That's wrong.** It's true for a different judge rule called **MAPE**, but not SMAPE. The experiment proved it, and the plan doc is now fixed. This is why we test ideas instead of trusting them.

---

## 8. What we've now decided

| Decision | In plain words |
| --- | --- |
| Train on `log1p(price)`, convert back with `expm1` | Teach the model the "ladder step", not the rupees |
| Use L1 or Huber loss on the log price | Aim for the middle, and don't panic over weird prices |
| Never use MSE on raw prices | That aims at the average, which the judge dislikes |
| Try a scale nudge `s` on OOF guesses; keep it only if it helps | A quick check for a model that leans |
| Never output ₹0 or negative prices | A ₹0 guess is the worst possible row (200%) |

---

## 9. Your turn (exercises)

1. **Guess first, then check.** A model always guesses **double** the real price. What's its SMAPE? (Hint: section 3.) Write your answer, then add a test in `tests/test_metrics.py` to check it.
2. **Think.** What if it always guesses **10×** the price? 100×? What number does the score creep toward, and why can it never go past 200?
3. **Play.** In `experiments/01_smape_demo.py`, make the model lean high by changing the prediction line to `np.log(y) + 0.2 + noise`. Now which `s` wins? Is it close to 0.82? (Hint: 0.82 ≈ 1 ÷ 1.22, which undoes the lean.)
4. **Wonder.** In the real 2025 data, do you think cheap items or expensive items will cause most of our score? Write your guess here; we'll check it in lesson 2.

## Your notes / ideas

_Add anything you noticed or want to try here._

---

### Word list (for quick look-up)

| Word | Simple meaning |
| --- | --- |
| Target | The thing we're trying to predict (here: price) |
| Metric | The judge's scoring rule (here: SMAPE) |
| Loss function | The rule a model uses to punish its own mistakes while learning |
| Log / log transform | The "step on the doubling ladder"; predicting that instead of rupees |
| Mean / average | Add everything up, divide by how many |
| Median | The middle value when everything is lined up in order |
| Geometric mean | The average of the ladder steps, turned back into rupees |
| MSE / L1 / Huber | Different loss functions: aim for the average / the middle / a mix |
| Bias | A model that keeps leaning one way (too high or too low) |
| OOF predictions | Guesses on data the model didn't train on, for an honest check |
| NaN | "Not a number", what you get from 0 ÷ 0 |
| Test / TDD | A small check program; writing it before the code |
