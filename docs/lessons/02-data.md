# Lesson 2 — Getting to know the data (2025 dataset)

**The big idea in one line:** before teaching a model anything, look at the data with your own eyes, like a detective at a crime scene. Every surprise you find now is a mistake you won't make later.

- Script: [`experiments/02_eda.py`](../../experiments/02_eda.py) — run `uv run python experiments/02_eda.py`
- Data: `Prev-year-data-files/.../dataset/` (kept out of git; it's 141 MB)
- Official rules: `Prev-year-data-files/.../student_resource/README.md`

> **Tech word:** this "look around first" step is called **EDA** (**Exploratory Data Analysis**).

---

## 1. What did Amazon give us?

Four files, like a school exam:

| File | What it's like | Rows |
| --- | --- | --- |
| `train.csv` | The **practice questions with answers**: products *and* their prices | 75,000 |
| `test.csv` | The **real exam**: products, but prices hidden | 75,000 |
| `sample_test.csv` / `sample_test_out.csv` | A **sample answer sheet**, showing exactly how to write your answers | 100 |

Every product row has:

| Column | Plain meaning | Example |
| --- | --- | --- |
| `sample_id` | The product's roll number | `0` |
| `catalog_content` | Everything written about the product | "La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)…" |
| `image_link` | A web link to its photo | `https://m.media-amazon.com/images/I/51mo8htwTHL.jpg` |
| `price` | The answer, only in train | `4.89` |

**No missing values anywhere.** That's rare and lucky.

> **Tech words:** the practice set is the **training set**; the hidden-answer set is the **test set**. The price is the **target** or **label**. Each column is a **feature** (a clue the model can use).

### Rules worth remembering (from the official README)

- Answer sheet: exactly two columns, `sample_id,price`, **one row for every test product**. One missing or extra row and you get **no score at all**.
- Prices must be **positive** numbers.
- Any pretrained model you use must be **≤ 8 billion parameters** and have an **MIT or Apache 2.0 licence**.
- **No looking up prices online.** Instant disqualification.
- The live leaderboard only uses **25,000 of the 75,000** test products. The final ranking uses all 75,000 **plus a 1-page write-up** of your method.

---

## 2. What do the prices look like?

Here's how many products fall in each price range (every `#` ≈ 800 products):

```
$0–1        353
$1–2       2435 ###
$2–5      10354 ############
$5–10     15622 ###################
$10–20    19158 #######################
$20–50    18986 #######################
$50–100    6199 #######
$100–200   1595 #
$200–500    281
$500–3000    17
```

- The **middle price (median) is $14**. Half of all products cost less than that.
- The **average (mean) is $23.65**, pulled up by a few very expensive items (the most expensive is $2,796).
- The cheapest item is 13 cents.

It looks like a **slide**: a big pile of cheap products, then a long thin tail of expensive ones. That's exactly the shape we faked in lesson 1, so everything we learned there applies.

Notice the bins get wider as you go right: 1, 2, 5, 10, 20, 50… That's the **"doubling ladder"** from lesson 1. On the ladder, the pile looks like a neat hill instead of a slide.

> **Tech words:** a pile with a long tail on the right is **right-skewed**. On the log ladder it looks roughly like a bell curve, which is called **log-normal**.

### Our first score, with no model at all

What if we guess the **same price for every product**?

| Guess for every product | SMAPE |
| --- | --- |
| $23.65 (the average) | 79.06 |
| $14.00 (the middle) | **72.70** |

Lesson 1 predicted this: the middle beats the average. **72.7 is our floor.** Any real model must beat it, or it has learned nothing.

> **Tech word:** a dumb-but-honest first score like this is a **baseline**.

**Answer to lesson 1, exercise 4** (cheap or expensive items, which dominate?): SMAPE gives **every row the same weight** in percent. About 38% of products cost under $10, so by sheer numbers **cheap products dominate the score**. Being 30% wrong on a $3 shampoo hurts exactly as much as being 30% wrong on a $300 mixer, and there are far more shampoos.

---

## 3. What's hidden inside `catalog_content`?

It looks like one messy blob of text, but it's actually **labelled pieces stuck together**:

```
Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)
Value: 72.0
Unit: Fl Oz
```

| Piece | Found in | What it holds |
| --- | --- | --- |
| `Item Name:` | 100% of rows | Brand + product + size, the most useful line |
| `Bullet Point 1…N:` | about 81% | Selling points ("real butter", "32 cookies total") |
| `Product Description:` | about 43% | A longer paragraph |
| `Value:` | 100% (a number in 98.7%) | **The total amount already worked out** |
| `Unit:` | 100% | What that amount is measured in |

The big gift is **`Value`**. For "12 Ounce (Pack of 6)", `Value` is already **72** (12 × 6). Amazon did the maths for us in about 80% of the rows we checked.

**One trap:** the same unit is spelled many ways, with **64 different spellings** in all: `ounce`, `oz`, `ounces`, `fl oz`, `fl. oz`, `fluid ounce`… To a computer, "oz" and "ounce" are totally different words unless we teach it they're the same.

> **Tech words:** pulling labelled pieces out of text is **parsing** or **feature extraction**. Merging spellings into one standard form is **normalisation**. The tool we'll use to find patterns like "Pack of 6" is a **regex** (regular expression).

---

## 4. Does "more stuff" mean "higher price"?

You'd think a 72-ounce pack costs more than a 12-ounce pack. So does quantity predict price?

We measured how strongly the two move together, on a scale where 0 = no connection and 1 = perfectly in step:

| Unit | Products | Connection between quantity and price |
| --- | --- | --- |
| ounce | 42,939 | 0.24 (weak) |
| count | 18,183 | 0.13 (very weak) |
| fl oz | 11,259 | 0.17 (weak) |

**Surprise: quantity alone barely helps.** Why? Because 72 ounces of *hot sauce* and 72 ounces of *saffron* have wildly different prices. Quantity only matters *once you know what the product is*.

**What this teaches us:**

- The model must understand **what** the product is, and that knowledge lives in the words. So text understanding is the main engine.
- Quantity is still useful, but as a **helper** to text, not on its own.
- This supports the idea from our plan: **predict price per unit, then multiply by quantity**. The "what is it" part sets the price per ounce; the quantity scales it up.

> **Tech word:** that 0-to-1 "move together" number is the **correlation**. Here we computed it on the log ladder for both price and quantity.

---

## 5. Is the exam like the practice? (the most important check)

This decides how we test ourselves honestly. Think of it like this:

- If the exam has **the same kinds of questions** as the practice set, you can test yourself by **hiding random practice questions**.
- If the exam has **completely new topics**, hiding random questions would fool you. You'd need to hide **whole topics** at once.

What we found:

| Check | Result | What it means |
| --- | --- | --- |
| Roll numbers (`sample_id`) | train 0, 2, 3…; test 1, 4…, all mixed together | Products were shuffled and **dealt randomly** into train and test, like cards |
| Test products whose **brand** also appears in train | **73%** | Most exam "topics" were in the practice set |
| Test products with the **exact same photo** as a train product | 6.7% | Some near-twins exist |
| Test products with **identical text** to a train product | 0.3% | Almost no exact copies |

**Verdict: the exam looks like a random slice of the same shop.** So our plan's idea of "hide whole brands when testing ourselves" was **too strict for this data**. It would make us test on harder questions than the real exam has. Hiding random products (while keeping cheap and expensive ones balanced in every group) matches reality better.

The 6.7% photo twins are a hint too: "find the most similar training product and look at its price" should work well. That's the nearest-neighbour idea from our plan.

> **Tech words:** "testing yourself by hiding part of the practice set" is **validation** (or **cross-validation**, when you take turns hiding different parts, called **folds**). Hiding by brand = **group K-fold**; hiding randomly but keeping price ranges balanced = **stratified K-fold**. Checking whether train and test look alike is part of spotting **distribution shift**.

---

## 6. What about the photos?

We downloaded 40 random product photos as a test:

| What we measured | Result |
| --- | --- |
| Downloads that worked | 40 / 40 |
| Speed (16 downloads at once) | about 7 photos per second |
| Average photo size | about 270 KB, up to 2560 × 2560 pixels |

Now the maths for all 150,000 photos (train + test):

- **Time:** 150,000 ÷ 7 per second ≈ **6 hours**.
- **Space:** 150,000 × 270 KB ≈ **40 GB**.

That's a lot of hours and disk space. But image models only need small pictures (about 224–512 pixels), so:

- **Shrink each photo right after downloading.** It drops to roughly 20–30 KB, and the whole set fits in about 4 GB.
- **Start the download in the first minute of the hackathon**, running in the background while we do everything else.
- It must be able to **resume**: if it crashes at photo 90,000, it continues from there instead of starting over.

> **Tech words:** downloading many at once = **parallel** / **multi-threaded** downloading. Shrinking = **resizing**. Picking up where you left off = **resumable**.

---

## 7. What this lesson changed in our plan

| Finding | Decision |
| --- | --- |
| Always guessing the median scores 72.7 | That's the floor. Every model must beat it |
| Train and test were dealt randomly | Test ourselves by **hiding random products, balanced by price range** (stratified K-fold), not by hiding brands |
| `Value` + `Unit` already give the total amount in 98.7% of rows | Parse them first. Only normalise the 64 unit spellings. Reading text off photos (OCR) is even less needed |
| Quantity alone barely predicts price | Text understanding is the main engine; try "price per unit × quantity" |
| 6.7% of test photos also appear in train | Nearest-neighbour "similar product" prices are promising |
| Photos: about 6 hours and 40 GB at full size | Shrink as you download, start at minute 0, make it resumable |
| Rules: models ≤ 8B parameters, MIT/Apache licence; write-up counts in final ranking | Pick models by licence; start the write-up early |

These are now reflected in [`decision-review.md`](../decision-review.md).

---

## 8. Your turn (exercises)

1. **Read 10 real products.** Run this and read the output like a shopper:
   `uv run python -c "import pandas as pd; df=pd.read_csv('Prev-year-data-files/68e8d1d70b66d_student_resource/student_resource/dataset/train.csv'); [print(r.price, '|', r.catalog_content[:200], '\n') for r in df.sample(10).itertuples()]"`
   For each one, guess the price *before* looking. What words in the text helped you guess?
2. **Find the weird ones.** Look at the 10 cheapest and 10 most expensive products (`df.nsmallest(10,'price')`, `df.nlargest(10,'price')`). Are any of them mistakes in the data? Would you trust them?
3. **Unit detective.** List all 64 unit spellings (`value_counts()` on the Unit field). Group them by hand into a few standard units (ounce, fl oz, count, pound, gram…). This becomes our normaliser in a later lesson.
4. **Think.** Photos matched in 6.7% of test products. Why might the same photo have *different* prices? (Hint: within train, same-photo products differ in price by about 1.5× on average. Look at their text.)

## Your notes / ideas

_Add anything you noticed or want to try here._

---

### Word list

| Word | Simple meaning |
| --- | --- |
| EDA | Looking around the data before building anything |
| Training set / test set | Practice questions with answers / the real exam |
| Feature | A clue the model can use (text, photo, quantity) |
| Baseline | A simple first score that every model must beat |
| Right-skewed / log-normal | Big pile of small values plus a long tail of big ones / looks like a neat hill on the log ladder |
| Parsing / feature extraction | Pulling labelled pieces out of messy text |
| Normalisation | Making "oz", "ounce" and "ounces" all mean the same thing |
| Regex | A mini-language for finding text patterns like "Pack of 6" |
| Correlation | How strongly two things move together (0 = not at all, 1 = perfectly) |
| Validation / cross-validation / folds | Testing yourself by hiding parts of the practice set, taking turns |
| Stratified / group K-fold | Hide random rows keeping price ranges balanced / hide whole groups like brands |
| Distribution shift | When the exam looks different from the practice |
| Resumable download | One that continues where it stopped instead of restarting |
