# 06_dataops

<!--
Design note for experiments/06_dataops.py (same stem, one to
one with the script). Write and agree on it before creating the script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Does rewriting the 05 model as a skrub DataOp (groups baked into the
computation graph) produce the same grouped-CV scores as experiment 05,
and does the `SkrubLearner` correctly handle the train/test split at
prediction time without leakage?

The secondary goal is correctness: experiment 05 built `cv_splits` as a
separate list object that could silently drift if columns or row order
changed. The DataOp approach makes that impossible — the groups column
lives on the same graph node as X.

## Motivation

Per DAY2.md §6: "GroupKFold only works if `groups` is the `patient_id` of
the *same* rows as `X`. Putting that on the graph is how you stop the
leakage from step 3 from creeping back in through a forgotten argument."

This experiment validates that the DataOp path reproduces 05 and teaches
how `make_learner` handles a test set that has no `target` column.

- **Sourcing strategy:** user (explicit request — DAY2.md §6)
- **Why this matters:** The DataOps pattern is the recommended way to
  productionise experiments in skrub. Confirming parity with 05 and
  verifying the learner works for submission is the gate before §7.

## Why we drop `patient_id` (not just `target`)

The DAY2.md snippet drops only `target`:

```python
X_op = data.drop("target", axis=1).skb.mark_as_X(...)
```

That leaves `patient_id` in the feature table. This is a **leakage risk**:

1. `patient_id` is a high-cardinality string identifier (5,576 unique
   values). `TableVectorizer` classifies it as high-cardinality and feeds
   it to `StringEncoder`, which compresses it into numeric dimensions.
2. At training time the model sees a patient-id embedding that encodes
   the *training* patient. At test time the patients are **completely new**
   (no overlap with train — confirmed by EDA). The embedding for a new
   patient id is meaningless noise, and the model has learned to rely on
   it, inflating training-set CV scores.
3. In experiment 05 we explicitly dropped `patient_id` from X and it was
   not in `cv_splits` either. To reproduce 05 fairly, we must drop it here
   too.

Therefore:

```python
X_op = data.drop("target", axis=1, errors="ignore")  \
           .drop("patient_id", axis=1, errors="ignore") \
           .skb.mark_as_X(cv=GroupKFold(n_splits=5),
                           split_kwargs={"groups": groups})
```

`errors="ignore"` also makes the drop safe at **predict time** (when
`X_test` has no `target` column).

## Method

- **Files touched:**
  - `experiments/06_dataops.py` (new)
  - `journal/06_dataops.md` (this file)
  - `journal/JOURNAL.md` (new row after run)

- **Input table:** `visits = X_train.merge(y_train, on="Index")`
  — all columns including `target`. Same as DAY2.md.

- **DataOp graph:**
  ```python
  data   = skrub.var("visits", visits)
  groups = data["patient_id"]

  X_op = (data
      .drop("target",     axis=1, errors="ignore")
      .drop("patient_id", axis=1, errors="ignore")
      .skb.mark_as_X(
          cv=GroupKFold(n_splits=5),
          split_kwargs={"groups": groups},
      ))

  y_op = data["target"].skb.mark_as_y()

  pred = (X_op
      .skb.apply(skrub.TableVectorizer())
      .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op))
  ```

- **Evaluation:** `skore.evaluate(pred)` — **no `splitter=`**. skore reads
  `cv` and `split_kwargs` from the graph.

- **Learner smoke-test (submission proof):**
  ```python
  learner = pred.skb.make_learner()
  learner.fit({"visits": visits})
  preds = learner.predict({"visits": X_test})   # X_test has no target col
  assert len(preds) == 11013
  ```

- **Hub report:** `project.put("06_dataops", report)`

- **No Kaggle submission** in this experiment — §7 decides the final model
  and writes the CSV.

- **Out of scope:** any hyperparameter tuning, additional features.

## Comparison with experiment 05

| Aspect | 05_tabular_pipeline | 06_dataops |
|---|---|---|
| Feature table | `X_raw.drop(["patient_id"])` | DataOp drops `target` + `patient_id` |
| CV | pre-built `cv_splits` list | baked into graph via `mark_as_X` |
| Encoder | `tabular_pipeline` (TableVectorizer + HGBR) | `TableVectorizer` + bare HGBR |
| `rater_id` | present | present |
| Expected RMSE | 7.521 | ~7.521 (should match if column set is identical) |

The scores **should be identical** if and only if:
1. The column set going into `TableVectorizer` is the same (same drops).
2. The GroupKFold random seed and n_splits match (same fold assignment).
3. The HGBR `random_state` is identical.

Any difference would indicate a column set discrepancy (e.g. `patient_id`
leaking in via the DAY2 snippet) or a different fold assignment.

## Workbook questions (DAY2 §6)
*Filled in after the run.*

**Q1. Are the 06 scores identical to 05? If not, explain precisely why.**
> *To be filled after run.*

**Q2. Which columns does X_op actually contain? Are Index / patient_id absent?**
> *To be filled after run.*
>
> Expected: X_op contains `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`,
> `ledd`, `time_since_intake_on`, `time_since_intake_off`, `rater_id`, `on`,
> `off` — exactly the 11 columns from 05, with `Index` and `patient_id` absent.

## Status

- **State:** planned
- **Approved by user on:** (pending)
- **Headline result:** (pending)
- **Implication for next iteration:** (pending)
