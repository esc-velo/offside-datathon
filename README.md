# Football Goal Prediction | Offside Data-Thon

## Overview

This project was developed for the **Offside Data-Thon**, a machine learning competition focused on predicting whether a football player will score at least one goal in a match.

The task was formulated as a binary classification problem:

**Target Variable:** `scored_flag`

* `0` → Player did not score
* `1` → Player scored at least one goal

Predictions were evaluated using **Average Precision (AP)**, making ranking quality and class imbalance handling critical.

---

## Dataset

The dataset combines:

* Player appearance records
* Market valuation data
* Match context information
* Advanced football analytics (xG, xA, xGChain, xGBuildup, etc.)
* Player demographics and positional data

The training set contained approximately **1.3 million player-match appearances**.

---

## Exploratory Data Analysis

Key findings from the analysis:

### Target Imbalance

Only ~8.5% of appearances resulted in a goal, creating a highly imbalanced classification problem.

### Missing Data Patterns

Several advanced analytics features contained significant missing values.

Investigation showed that:

* `goal_per_cap` was missing whenever a player had no national team experience.
* Advanced metrics such as `avg_xG`, `avg_xA`, and `avg_shots` were strongly linked to `analytics_coverage_flag`.

This suggested that missing values themselves contained useful information rather than representing random data loss.

### Strong Predictive Signals

The following variables showed a clear relationship with scoring probability:

* `avg_xG`
* `avg_npxG`
* `avg_shots`
* `avg_xA`
* `position`
* `market_value_tier`
* `starter_flag`

Players with higher attacking metrics consistently exhibited higher scoring rates.

### Leakage Detection

The provided feature catalog identified several features with high leakage risk.

Removed features included:

* `goals`
* `assists`
* `goal_contributions`
* `goal_difference`
* `minutes_played`
* `yellow_cards`
* `red_cards`
* `result`

Additional investigation revealed that match outcome-related variables could dominate model importance and were treated cautiously during experimentation.

---

## Feature Engineering

Several domain-inspired features were created:

### Attack Strength

```python
attack_strength = avg_xG + avg_npxG + avg_shots / 10
```

### Goal Threat

```python
goal_threat = avg_xG * avg_shots
```

### Market Attack

```python
market_attack = highest_market_value_in_eur * is_attacker
```

### International Experience

```python
international_caps + international_goals
```

### Forward Starter

```python
starter_flag * is_attacker
```

### Goal Creator Ratio

```python
avg_xG / (avg_xA + 0.01)
```

---

## Model

The final solution used:

### CatBoostClassifier

Reasons for selection:

* Native handling of categorical features
* Robust treatment of missing values
* Strong performance on large tabular datasets
* Minimal preprocessing requirements

Configuration:

```python
iterations=500
learning_rate=0.05
depth=6
loss_function="Logloss"
eval_metric="PRAUC"
```

---

## Results

| Metric                   | Score   |
| ------------------------ | ------- |
| Local Validation AP      | ~0.453  |
| Public Leaderboard Score | 0.47570 |

Final leaderboard position at submission time:

**Top 15**

---

## Repository Structure

```text
.
├── v1-offside.ipynb
├── submission.csv
├── README.md
```

---

## Lessons Learned

* Domain knowledge is often as valuable as model complexity.
* Missing values can carry important information.
* Leakage detection is critical in sports analytics problems.
* Strong EDA can reveal more useful insights than extensive hyperparameter tuning.

---

## Author

Rahini Raneru

Engineering Student | Machine Learning & Data Analytics Enthusiast
