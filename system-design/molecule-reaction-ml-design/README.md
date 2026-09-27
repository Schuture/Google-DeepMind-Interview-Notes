# ML Design: Predicting a Reaction Factor for Pairs of Molecules

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| ML system design | ★★★☆☆ | Medium | MLE · RE · RS | eda, molecular-fingerprints, pairwise-models, symmetry, data-splitting, leakage, graph-neural-networks, active-learning | 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

A chemistry lab has measured pairwise reactions between molecules and recorded, for each pair tested, one
positive number called the *reaction factor*. The raw data is a table of rows
`(molecule1_name, molecule2_name, reaction_factor)`, where `molecule1_name` and `molecule2_name` name two
distinct molecules and `reaction_factor` is the measured outcome of combining them. A separate lookup table
maps every molecule name appearing anywhere in the data to a *SMILES* string (Simplified Molecular Input
Line Entry System) — a compact text notation that encodes a molecule's atoms, bonds and ring structure as a
sequence of characters (for example `CCO` is ethanol). The same molecule can be written as more than one
valid SMILES string; a *canonical SMILES* is the single string a fixed canonicalisation algorithm always
produces for a given structure, regardless of which equivalent string was fed to it. Task: given the
measured rows and the name-to-SMILES lookup, build a model that takes two molecules — by name, if both are
in the lookup, or by SMILES structure directly, for a molecule the lookup has never seen — and predicts
their reaction factor.

*EDA* (exploratory data analysis) is the practice of inspecting a dataset's distributions, counts,
duplicates and anomalies before choosing a model, so that a modelling choice is a response to what the data
actually looks like rather than an assumption a model would otherwise silently absorb. A *molecular
fingerprint* is a fixed-length bit vector summarising a molecule's substructures; the common *ECFP* (Morgan)
fingerprint hashes every local, atom-centred subgraph up to a fixed radius into a bit position, so two
molecules sharing substructures share many set bits, and the fingerprint is computable from a SMILES string
alone, with no training involved. A *scaffold* is a molecule's core ring-and-linker framework obtained by
stripping its side chains (the Bemis–Murcko scaffold is the standard definition), used to group structurally
related molecules together. *Leakage* is evaluation contaminated by information that would not be available
at prediction time — for example, the same physical measurement, or a near-duplicate of it, appearing in
both the training and the test data — which makes a measured metric more optimistic than the model's real
performance. A *molecule-disjoint split* is a train/test split performed by assigning whole molecules, not
individual rows, to train or test, so that no molecule seen during training also appears in a test pair —
the split needed to measure generalisation to a molecule the model has never encountered, rather than to a
merely unseen combination of familiar ones.

Premise, all numbers given:

- $200{,}000$ measured rows, over $20{,}000$ distinct molecules.
- The lookup table gives every molecule name's SMILES structure.
- The reaction factor is strictly positive and spans about four orders of magnitude.
- About $5\%$ of the distinct unordered pairs were measured two or more times; across repeated
  measurements of the same pair, the spread is about $0.1$ in $\log_{10}$ units.
- The order of the two names in a row carries no chemical meaning — the lab confirms the factor measured
  for $(a, b)$ equals the factor for $(b, a)$ — but both orders occur in the raw file: which molecule is
  written as `molecule1_name` is arbitrary per row.
- The model will be used for two purposes: (i) predicting the factor for unmeasured pairs of molecules
  that have each been measured before, in some other pair, and (ii) predicting the factor for pairs where
  one or both molecules have never been measured at all — new candidates a chemist proposes — in order to
  choose which pairs the lab should measure next.

In scope: the modelling work, from the raw table to choosing the next pairs to measure. Out of scope: the
serving infrastructure that would run a trained model in production; the chemical or quantum-mechanical
mechanism that produces the reaction factor; and wet-lab logistics.

Produce:

1. The questions you would ask and the EDA you would run before modelling, and what each check could
   reveal.
2. A representation of a single molecule and of a pair, including how to make the prediction independent
   of the order the two molecules are given in.
3. A data-splitting strategy for each of the two uses above, and what goes wrong with a naive random split
   of rows.
4. A sequence of models from a simple baseline to the strongest reasonable choice, each with its training
   loss and what it can and cannot capture.
5. At least one concrete technique for improving generalisation to molecules with little or no measurement
   history.
6. An evaluation plan: metrics, the noise ceiling, and evaluation sliced by how well-measured a molecule is.
7. A plan for closing the loop: choosing which pairs the lab should measure next.

Questions the interviewer may interleave, once the relevant part of the design comes up:

- Explain the self-attention computation inside a single transformer layer, and state how its compute cost
  scales with sequence length.
- A transformer encoder run directly over a molecule's SMILES string gives different outputs for two valid
  SMILES strings of the same molecule; explain why, and give two fixes.
- A graph neural network's readout step pools its final atom representations — for example by summing or
  averaging over atoms — into one fixed-length vector for the whole molecule; explain why this makes the
  resulting molecule embedding invariant to the order the atoms are listed in.
- State whether an $L_1$ or an $L_2$ loss is the better default for a heavy-tailed regression target such
  as this one, and justify the choice.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Worth confirming before designing: whether downstream consumers only need the reaction factor's relative
ordering and log-scale calibration, or its exact predicted value on the raw scale (assumed here: relative
ordering and log-scale calibration, since that is all the active-learning loop in the deep dives below
actually consumes); and whether the lookup table's SMILES strings are already canonicalised consistently
(assumed here: no — different rows may hold different, but chemically equivalent, SMILES strings for the
same molecule, so canonicalisation is a preprocessing step this design owns rather than a guarantee it is
given).

### Requirements and scale

**Coverage of pair-space.** The number of distinct unordered pairs of $20{,}000$ molecules is
$\binom{20{,}000}{2} = 199{,}990{,}000$, so the $200{,}000$ measured rows cover about

$$\frac{200{,}000}{199{,}990{,}000} \approx 0.1\%$$

of every possible pair — the measured data is an extremely sparse sample of pair-space, and almost
everything the model will ever be asked about (in particular, every candidate the active-learning loop
considers) is unmeasured.

**Average measured partners per molecule.** Each row contributes two endpoints, so
$200{,}000 \times 2 / 20{,}000 = 20$ measured partners per molecule on average. This is an average only:
Data and EDA below looks at the shape of that distribution, which matters because a per-molecule model can
only be fit reliably for a molecule with many partners.

**Fingerprint storage.** A common fixed length for an ECFP/Morgan fingerprint is $2{,}048$ bits. Storing one
per molecule for the whole catalogue costs
$20{,}000 \times 2{,}048 \text{ bits} / 8 = 5{,}120{,}000$ bytes, exactly $5.12$ MB — small enough to hold
the entire fingerprint table in memory for both training and inference, unlike a large-scale retrieval
index.

**The noise floor.** The replicate spread of $0.1$ in $\log_{10}$ units (Data and EDA below derives it, and
Evaluation gives its precise meaning) sets an approximate lower bound on the RMSE any model can be expected
to reach on this target: it is the standard deviation of the noise already present in the label itself, not
a property of any model.

### Data and EDA

**Target distribution.** Plot the distribution of $\log_{10}(\text{reaction factor})$, not the raw factor:
with a four-orders-of-magnitude range, a raw-scale histogram is dominated by its largest values, while the
log-scale distribution shows the natural range to model in and immediately suggests the choice made in
Models below — predict the log, not the raw value.

**Per-molecule measurement counts.** For every molecule, count how many measured rows it appears in. The
mean across the whole set is exactly $20$ (Requirements and scale), but the distribution is heavy-tailed: a
handful of hub molecules — common reagents or solvents measured against many partners — account for a
disproportionate share of rows, while a long tail of molecules has only one or two measurements. This
foreshadows a generalisation gap before any model is fit: a model that relies on per-molecule identity can
only be trusted for the hub end of this distribution.

**Replicate noise.** Group rows by an order-independent pair key (Splits and leakage, below, gives the
construction) and look at the roughly $5\%$ of distinct pairs measured two or more times; the standard
deviation of $\log_{10}(\text{reaction factor})$ within each such group estimates the irreducible
measurement noise. This is the source of the $0.1$ figure in the premise, and it becomes the noise floor
that Evaluation compares every model's RMSE against.

**Outliers and unit errors.** A value many decades away from its neighbours for an otherwise unremarkable
pair, or an exact factor-of-$10$ or factor-of-$1{,}000$ offset shared by a whole subset of rows, is a
classic sign of a units mistake or a transcription error upstream, not a genuinely extreme measurement, and
should be checked before being trusted.

**Order conflicts.** Because order carries no chemical meaning but both orders occur in the file, grouping
by the order-independent pair key surfaces any pair recorded once as $(a, b)$ and again — among the $5\%$
replicated pairs — as $(b, a)$; a large discrepancy between the two recordings of the same physical pair is
a second, independent read on the replicate-noise question above, and also a direct check that the lab's
claim of order-independence actually holds in the data rather than being merely asserted.

### Features and representations

**Representing one molecule,** from weakest to strongest generalisation: (i) an *identity embedding* — a
vector learned per molecule id, exactly like a matrix-factorisation user or item embedding — captures
whatever that molecule's own training rows reveal about it, but is undefined for a molecule absent from
training, so it only ever serves use (i); (ii) a *molecular fingerprint*, most commonly ECFP/Morgan, defined
for any molecule the moment its SMILES is known, with nothing to train to compute it; (iii) hand-built
*descriptors* — scalar physicochemical properties (molecular weight, counts of specific functional groups,
computed $\log P$, …) read directly off the structure; (iv) a *molecular graph* — atoms as nodes, bonds as
edges — consumed by a graph neural network, which learns its own substructure features rather than relying
on a fingerprint's fixed hash. Representations (ii)–(iv) are all computable from a SMILES string alone for
a molecule that has never been measured — exactly the property use (ii) needs, and exactly what (i) lacks.

**Representing a pair, order-independent.** Write $h(\cdot)$ for whichever single-molecule representation
is chosen. A pair function $f(h(a), h(b))$ must satisfy $f(h(a), h(b)) = f(h(b), h(a))$ for every pair,
since the lab confirms the factor itself does not depend on order:

```text
   molecule a --> h(a) --\
                           >-- symmetric combiner --> predicted log10(reaction factor)
   molecule b --> h(b) --/     (sum, product, |diff|,
                                 or a symmetric bilinear form)
```

Four ways to build one:

- Elementwise **sum**, $h(a) + h(b)$ — cheap and order-invariant by construction, but collapses every pair
  with the same sum of representations onto the same input, discarding which specific pair it is.
- Elementwise **product**, $h(a) \odot h(b)$ — order-invariant, and captures a different kind of pair
  information (co-occurrence of specific fingerprint bits or descriptor values in both molecules at once),
  most useful alongside the sum rather than instead of it.
- Elementwise absolute **difference**, $|h(a) - h(b)|$ — order-invariant, useful when the target depends on
  how dissimilar the two molecules are rather than on their joint magnitude.
- A **symmetric bilinear form**, $h(a)^\top S\, h(b)$ for a learned matrix with $S = S^\top$ — order-invariant
  because $h(a)^\top S h(b) = h(b)^\top S h(a)$ exactly when $S$ is symmetric (a general, non-symmetric $S$
  would not have this property), and strictly more expressive than the elementwise product, since it can mix
  different coordinates of $h(a)$ against different coordinates of $h(b)$.

A fifth option symmetrises the *model* instead of the *features*: feed the order-dependent concatenation
$[h(a), h(b)]$ to any model, train on both orders of every pair, and average the two predictions at
inference, $\tfrac12(\hat y(a,b) + \hat y(b,a))$ — exactly order-invariant at inference regardless of what
the model learned, at roughly twice the training and inference cost. In practice the sum and the product (or
a bilinear form) are used together, since each captures pair information the other misses; the checks below
verify with a small fitted model that this combination is exactly order-invariant while a naive ordered
concatenation is not.

### Splits and leakage

**Deduplicate before splitting.** Canonicalise every row to an order-independent pair key — for example the
pair of molecule ids sorted, `(min(id_a, id_b), max(id_a, id_b))` — and collapse rows sharing a key by
averaging their $\log_{10}(\text{reaction factor})$ values into one label; Data and EDA above already
estimated how much within-pair spread this averages away, about $0.1$ in log units, the same figure that
becomes the noise floor. This must happen *before* any split is chosen: splitting duplicate rows naively at
random can put one repeat measurement of a physical pair in training and the other, slightly different,
repeat in test — the model then partly memorises a value it has effectively already seen, an instance of the
*leakage* defined in the problem statement, and the test metric reads better than production performance
will.

**Random row split.** Splitting the deduplicated rows uniformly at random is still the wrong default here:
with an average of $20$ measured partners per molecule, a molecule placed in the test set overwhelmingly
also appears in several training rows too — the split is *row*-disjoint but not *molecule*-disjoint. This
fairly measures use (i), predicting an unmeasured pair between two molecules the model has already seen
elsewhere, but is optimistic for use (ii): a model can partly memorise per-molecule identity and still score
well here, without having learned anything that generalises to a molecule it has never seen at all.

**One-molecule-unseen split.** Assign molecules, not rows, to train and test; a row qualifies for this test
set if exactly one of its two molecules was assigned to test (the other is in train). This measures the more
common case of use (ii): a candidate pair where only one side is genuinely new.

**Both-unseen split.** With the same molecule assignment, a row qualifies for this harder test set only if
*both* of its molecules were assigned to test; a row with one molecule on each side is dropped from both the
training set and this test set entirely, since it would neither test extrapolation to two new molecules nor
be a valid training example for a molecule-disjoint model. This is the most literal test of use (ii)'s worst
case: predicting between two molecules the model has never measured at all.

**Scaffold split.** Assigning molecules to folds uniformly at random still lets a test molecule be a close
structural relative of many training molecules — sharing the same core ring system and differing only in a
side chain — which a fingerprint or graph model can exploit almost as easily as if the molecule itself had
been seen. Assigning molecules to folds by their *scaffold* instead, so every molecule sharing a scaffold
with a training molecule stays in the training fold, is a strictly harder and more realistic test of
extrapolation to a chemically novel series; the gap between a random-molecule split's score and a scaffold
split's score on the same model is itself a useful, checkable measure of how much of the "molecule-disjoint"
result above was really coming from near-duplicates of the training set.

### Models

Every model below predicts $y = \log_{10}(\text{reaction factor})$, never the raw factor: the raw factor's
four-orders-of-magnitude range means its variance is dominated by its largest values, so a loss computed on
the raw scale would spend nearly all its gradient on a few huge measurements; on the log scale the replicate
noise is homoscedastic (a roughly constant standard deviation of $0.1$ regardless of magnitude — additive
noise in log units is multiplicative noise on the raw scale), exactly what a squared-error loss assumes.

1. **Global mean.** Predict $\hat y = \bar y$ for every pair, ignoring molecule identity entirely, fit and
   scored with mean squared error, $\mathrm{MSE} = \frac1n \sum_i (y_i - \bar y)^2$. Cannot distinguish any
   pair from any other; exists to fix $R^2 = 0$, the floor every other model below must clear.
2. **Additive per-molecule model.** $\hat y(a, b) = \mu + b_a + b_b$, one scalar bias per molecule id, fit by
   ordinary least squares. Captures "how reactive is this molecule on average across its partners," nothing
   about which specific *pair*. Defined only for molecules present in training (a molecule absent from every
   training row gets bias $0$ by construction, with no data to fit it from), so the model ignores the new
   side of a pair with one such molecule and collapses to the useless global mean of model 1 on a pair of
   two — the exact failure the checks below measure directly.
3. **Matrix factorisation.** $\hat y(a, b) = \mu + b_a + b_b + u_a^\top u_b$, adding a learned
   low-dimensional latent vector per molecule and its inner product — a symmetric bilinear form with
   $S = I$, in the language of the previous section — to capture pair-specific interaction beyond the two
   molecules' individual averages. Still trained only from identity, so it inherits the additive model's
   blindness to any molecule outside training, but is strictly richer *among* already-measured molecules
   (use (i)), since $u_a^\top u_b$ can express that a specific pair reacts unusually well or badly beyond
   what either molecule's own average predicts alone.
4. **Gradient-boosted trees on symmetric fingerprint features.** Compute a fingerprint $h(\cdot)$ for each
   molecule directly from its SMILES — no training needed to compute it, unlike models 2–3's per-molecule
   parameters — and feed a symmetric pair function of it, for example $[h(a) + h(b),\ h(a) \odot h(b)]$,
   into gradient-boosted trees trained on squared error (or Huber loss, a safer default whenever outliers or
   unit errors flagged by the EDA have not all been cleaned, since past a threshold residual its loss grows
   linearly rather than quadratically, so its gradient stays bounded). Because $h$ comes from structure
   alone, this model is defined for any molecule with a SMILES string, seen in training or not — the
   property use (ii) needs and models 2–3 lack; the checks below confirm, with ridge regression on the same
   symmetric features, that such a model keeps most of its accuracy on a molecule-disjoint split, where
   model 2 falls to the global mean.
5. **GNN or transformer pair encoder.** Replace the fixed fingerprint with a *learned* molecule encoder — a
   graph neural network reading the molecular graph directly, or a transformer reading a SMILES string as a
   token sequence — trained end to end with a symmetric pair head (Features and representations, above) on
   squared or Huber loss. Strictly more capacity than model 4, since the encoder learns which substructures
   matter for this specific target instead of relying on a fixed, hand-designed hash, at the cost of needing
   enough training pairs to fit that encoder, and the extra care worked out in the deep dives below (canonical
   or augmented SMILES for the transformer; nothing extra for the GNN).

### Evaluation

**Metrics.** Report the root-mean-squared error of $y = \log_{10}(\text{reaction factor})$ — directly
comparable across splits and against the noise floor below, unlike a metric computed on the raw factor — and
the Spearman rank correlation between predicted and true $y$ within each split, useful independently of
RMSE since the closing-the-loop use in the deep dives below mainly needs candidate pairs ranked correctly,
not predicted to the exact value.

**The noise floor.** The replicate standard deviation measured in Data and EDA (about $0.1$ in $\log_{10}$
units) is the standard deviation of the noise added to every individual measurement, including every one in
the test set. For a test label generated as (true value) $+$ (independent noise of standard deviation
$\sigma$), the expected mean squared error of even the perfect predictor of the true value is exactly
$\sigma^2$ — RMSE cannot be driven below $\sigma$ by a better model, only by a better measurement. The
checks below fit the best model possible on synthetic data — least squares on the hidden features the
labels are generated from — and its test RMSE lands within $5\%$ of $\sigma$. Every RMSE figure should be
read against this floor, not against zero.

**Slices.** Report every metric broken out at least by split type (random row, one-unseen, both-unseen,
scaffold), so one blended number never hides that a model looks good only because most test rows are the
easy, already-well-measured kind, and by how many training measurements the pair's *less-measured* molecule
had — few versus many partners, in Data and EDA's heavy tail — since that is exactly the axis the
active-learning loop below is trying to move a molecule along.

### Deep dives

**(a) Generalising to molecules with little or no measurement history.** Structure-based features are
already the point of models 4–5 above; beyond that, a molecular encoder *pretrained* on a much larger,
weakly labelled or unlabelled molecule corpus and fine-tuned on this dataset's $200{,}000$ rows gives a
better starting representation for a molecule with few or zero in-domain measurements than an encoder
learned from this dataset alone. Regularisation shrinks the model away from over-fitting the well-measured
hub molecules that Data and EDA's heavy tail identified. Ensembling — training several models on bootstrap
resamples or different random seeds — turns disagreement among them into an uncertainty signal: a novel
molecule the ensemble disagrees sharply about is one the model has not really learned anything reliable
about, feeding directly into the active-learning loop in (b).

**(b) Closing the loop: choosing what to measure next.** Using the model for use (ii) is an
active-learning problem, not a plain prediction problem: the goal is not to predict accurately everywhere,
but to choose measurements that improve the model, or find a high-value pair, as fast as possible — relevant
because Requirements and scale already showed only about $0.1\%$ of pair-space has ever been measured. An
*upper confidence bound* rule ranks candidates by $\hat y + \kappa \hat\sigma$, for a predicted value
$\hat y$, a calibrated uncertainty estimate $\hat\sigma$ (the ensemble spread from (a), calibrated as in the
last follow-up below) and a constant $\kappa$ trading exploiting pairs already predicted high-value against
exploring pairs the model is merely unsure about. *Expected improvement* instead weighs a candidate by how much, in
expectation, measuring it would raise the current best-known value, behaving more conservatively than a
confidence bound once the current best value is already high and few candidates plausibly beat it. Either
rule creates the same downstream problem: the next round of measurements is deliberately biased toward
whatever this round's model considered promising or uncertain, so a model retrained on the combined data
trains on a *shifted* distribution relative to the original rows — the same sampling-bias hazard the splits
above warned about, now self-inflicted; worth monitoring by checking whether newly labelled batches keep
landing in the same region of fingerprint space as earlier rounds (exploiting a possibly narrow, spurious
high-value region) or keep moving to previously poorly covered molecules (the intended effect of the
uncertainty term).

**(c) Self-attention and its cost.** One transformer layer projects a sequence of $L$ token embeddings to
queries, keys and values, $Q = XW_Q$, $K = XW_K$, $V = XW_V$, and computes
$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(QK^\top / \sqrt{d_k}\right) V$: every position attends
to every other position through the $L \times L$ score matrix $QK^\top$. Forming and softmax-normalising
that matrix costs $O(L^2 d_k)$ time and $O(L^2)$ memory, quadratic in sequence length — for a SMILES string,
$L$ is the string's character (or token) length, so a long SMILES string costs disproportionately more than
a short one to encode this way, unlike a graph neural network whose per-layer cost scales with the number of
bonds, not with the square of the atom count.

**(d) Why a SMILES transformer is sensitive to how a molecule is written, and two fixes.** Self-attention has
no built-in notion that a token in one string represents the same atom as a token in a differently-written
string for the same molecule: two syntactically different but chemically identical SMILES strings for one
molecule (starting the traversal from a different atom, or writing branches in a different order) produce
different token sequences, and nothing in the architecture forces the same output embedding for both —
unlike a graph neural network, whose input is the graph itself, not one arbitrary linearisation of it. Two
fixes: always canonicalise a SMILES string before encoding it, so exactly one deterministic string represents
each unique molecule and the ambiguity is removed rather than left for the model to learn away; or
*randomised-SMILES augmentation* — train on many different valid, non-canonical strings per molecule,
resampled at training time — so the model learns approximate invariance to the choice through data rather
than through a preprocessing guarantee. Canonicalisation is cheaper and exact; augmentation costs more
training time but also acts as a general regulariser and remains useful even where canonicalisation has bugs
or the canonicalisation algorithm itself changes between library versions.

**(e) Why a GNN's readout is invariant to atom order.** A graph neural network computes one representation
per atom (node) from the graph's connectivity, then a *readout* pools all of them — summing or averaging —
into one fixed-length vector for the whole molecule. Each message-passing layer updates an atom from its own
vector and a sum, mean or max over its neighbours' vectors, with the same weights for every atom, so
renumbering the atoms only permutes the list of per-atom vectors without changing any of them (the layers
are permutation-equivariant). Sum and mean are symmetric functions of their inputs: addition is commutative
and associative, so reordering the terms being summed or averaged never changes the result. However the
graph's atoms happen to be numbered or listed as input, the readout that pools over all of them produces the
identical embedding — invariance here is architectural, guaranteed by the equivariant layers and the choice
of pooling function, not something training data has to teach the model the way (d) has to teach a
transformer.

**(f) $L_1$ versus $L_2$ loss on a heavy-tailed target.** Squared error's gradient with respect to a
prediction grows linearly with the residual, so one very large error can dominate the total gradient and
pull the fit toward reducing that one point at the expense of the bulk of ordinary ones — a real risk on a
heavy-tailed raw target. Absolute error's gradient has constant magnitude regardless of residual size, so an
outlier pulls the fit no harder than a typical point does, at the cost of a loss that is non-smooth at zero
residual and that targets the conditional median rather than the conditional mean. Modelling
$\log_{10}(\text{reaction factor})$ already pulls in the most extreme raw-scale values (Requirements and
scale turns a four-orders-of-magnitude raw range into a compact log range), so squared error on the log scale
is often an adequate default; genuine unit-error outliers the EDA missed are better handled with Huber loss —
squared error for small residuals, linear past a threshold — than by taking either loss to its extreme.

### Follow-ups

- **Asymmetric reactions.** If a later experiment showed order actually mattered — one molecule is a
  catalyst added to a substrate, and which is added first changes the outcome — every order-invariant
  component above needs replacing or supplementing: the pair function no longer enforces
  $f(h(a), h(b)) = f(h(b), h(a))$, the pair-key canonicalisation in Splits and leakage must canonicalise on
  which molecule is the catalyst rather than collapse both orders together, and the replicate-noise estimate
  must first confirm which measured repeats are true replicates (same order) rather than the other,
  genuinely different, direction.
- **Predicting for three molecules.** A pairwise model does not trivially extend to triples: the natural
  generalisation of a symmetric combiner is a function symmetric over all three inputs — for example summing
  every pairwise bilinear term, $\sum_{i<j} h_i^\top S h_j$, plus a per-participant single-molecule term —
  but the number of possible combinations of a fixed molecule set grows combinatorially with group size, so
  the sparsity already seen at pairs (about $0.1\%$ of pair-space measured) becomes far more severe for
  triples, leaning even more heavily on the structure-based generalisation of deep dive (a) than on anything
  identity-based.
- **Multi-task learning.** If the lab also runs a related assay on the same molecules far more often than
  the pairwise reaction — solubility or stability, say — sharing the molecule encoder between that auxiliary
  task and this one, with separate output heads trained jointly, lets the auxiliary task's larger sample size
  teach the shared encoder a better general-purpose representation, directly benefiting exactly the
  poorly-measured molecules the reaction data alone under-serves.
- **Detecting extrapolation.** Beyond an ensemble's raw disagreement (deep dive (a)), a direct check is
  distance in representation space: the fingerprint or embedding distance from a candidate pair's molecules
  to the nearest molecule actually seen in training. A candidate whose nearest training neighbour is still
  far away is being extrapolated rather than interpolated, and that same distance is a useful additional
  feature for the uncertainty estimate the active-learning loop in (b) consumes.
- **Calibrating uncertainty.** An ensemble's raw spread (deep dive (a)) is not automatically a calibrated
  interval, which the upper-confidence-bound rule in (b) needs it to be: hold out a further slice of
  already-measured data, never used to fit or select among ensemble members, and check that its nominal
  predictive interval contains the true value at roughly its nominal rate, rescaling the spread if it
  systematically does not.

<details>
<summary>Estimate check (runnable)</summary>

```python
import math
from math import comb

import numpy as np
from sklearn.linear_model import Ridge

# ---- requirements-and-scale numbers ----
n_molecules_real = 20_000
n_rows_real = 200_000

n_possible_pairs = comb(n_molecules_real, 2)
assert n_possible_pairs == 199_990_000
frac_measured_pct = 100 * n_rows_real / n_possible_pairs
assert round(frac_measured_pct, 2) == 0.10

avg_partners = 2 * n_rows_real / n_molecules_real   # each row has 2 endpoints
assert avg_partners == 20.0

fp_bits = 2_048
fp_store_bytes = n_molecules_real * fp_bits // 8
assert fp_store_bytes == 5_120_000
fp_store_mb = fp_store_bytes / 1e6
assert fp_store_mb == 5.12

replicate_noise_log10 = 0.1   # given: the spread across repeated measurements of the same pair

print("all requirements-and-scale numbers check out")


# ---- synthetic experiment: hidden molecular features, observed only through noisy binary fingerprints ----
# NOTE: 300 molecules / 4,000 pairs is a small stand-in for the real 20,000 / 200,000 scale -- enough to
# make the qualitative effects below stable, not a re-creation of the real dataset's own numbers.
rng = np.random.default_rng(0)

n_molecules, dim = 300, 6
x = rng.normal(size=(n_molecules, dim))                    # each molecule's hidden, never-observed features

g_coef = rng.uniform(0.5, 1.5, size=dim)                    # the single-molecule term g(x) = x . g_coef
w_diag = rng.uniform(0.3, 0.9, size=dim)
off = rng.normal(scale=0.15, size=(dim, dim))
off = (off + off.T) / 2                                     # NOTE: symmetrise before zeroing the diagonal,
np.fill_diagonal(off, 0.0)                                  # so W stays exactly symmetric (W == W.T) below
W = np.diag(w_diag) + off
assert np.allclose(W, W.T)                                  # the interaction matrix the text calls symmetric W

n_bits_per_dim, fp_noise = 4, 0.5
noise_bits = rng.normal(scale=fp_noise, size=(n_bits_per_dim, n_molecules, dim))
bits = (x[None, :, :] + noise_bits > 0).astype(np.float64)   # a noisy binary reading of each hidden coordinate
fingerprint = bits.transpose(1, 0, 2).reshape(n_molecules, n_bits_per_dim * dim)   # the observed "fingerprint"

all_a, all_b = np.triu_indices(n_molecules, k=1)             # every possible unordered pair, in a fixed,
n_pairs = 4_000                                              # deterministic order -- never a Python set, so
chosen = rng.permutation(len(all_a))[:n_pairs]               # results cannot depend on hash seed or iteration
mol_lo, mol_hi = all_a[chosen], all_b[chosen]                # order

swap = rng.integers(0, 2, size=n_pairs).astype(bool)         # which molecule the raw file happens to write
mol1 = np.where(swap, mol_hi, mol_lo)                        # first is arbitrary per row, exactly as the
mol2 = np.where(swap, mol_lo, mol_hi)                        # premise states


def pairwise_interaction(xa, xb):
    return np.einsum("ij,jk,ik->i", xa, W, xb)


true_log10 = x[mol1] @ g_coef + x[mol2] @ g_coef + pairwise_interaction(x[mol1], x[mol2])
# NOTE: true_log10 depends only on the unordered {mol1, mol2} pair -- swapping which one is "first" above
# leaves it unchanged, matching the premise that the factor itself does not depend on which name is written
# first.

sigma_noise = replicate_noise_log10
y = true_log10 + rng.normal(scale=sigma_noise, size=n_pairs)

noise_floor_rmse = math.sqrt(np.mean((y - true_log10) ** 2))   # the realised noise: the best any model could do
assert abs(noise_floor_rmse - sigma_noise) < 0.02               # matches the sigma used to generate the data

fp1, fp2 = fingerprint[mol1], fingerprint[mol2]
X_sym = np.concatenate([fp1 + fp2, fp1 * fp2], axis=1)        # order-invariant: unchanged by swapping fp1, fp2
X_ord = np.concatenate([fp1, fp2], axis=1)                    # order-dependent: swapping moves values across halves

X_additive = np.zeros((n_pairs, n_molecules))
X_additive[np.arange(n_pairs), mol1] += 1.0
X_additive[np.arange(n_pairs), mol2] += 1.0                    # row @ b = b_mol1 + b_mol2; mu is added separately


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def fit_additive_least_squares(train_mask, test_mask):
    """mu + b_a + b_b: mu is the training mean and the b's are fit to the residuals by ordinary least
    squares. A molecule absent from every training row has an all-zero column there, so lstsq's
    minimum-norm solution leaves its b at 0: a pair of two such molecules is predicted as mu, the global
    mean of model 1."""
    mu = y[train_mask].mean()
    coef, *_ = np.linalg.lstsq(X_additive[train_mask], y[train_mask] - mu, rcond=None)
    pred = mu + X_additive[test_mask] @ coef
    return r2_score(y[test_mask], pred)


def fit_ridge(features, train_mask, test_mask, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(features[train_mask], y[train_mask])
    pred = model.predict(features[test_mask])
    return r2_score(y[test_mask], pred), model


# random row split: molecule-disjointness is not enforced, so most test molecules also appear in training
row_order = rng.permutation(n_pairs)
row_test = np.zeros(n_pairs, dtype=bool)
row_test[row_order[: int(0.2 * n_pairs)]] = True
row_train = ~row_test

# molecule-disjoint split: assign MOLECULES, not rows, to train/test; drop rows straddling both folds
mol_order = rng.permutation(n_molecules)
mol_is_test = np.zeros(n_molecules, dtype=bool)
mol_is_test[mol_order[: int(0.3 * n_molecules)]] = True
row1_test, row2_test = mol_is_test[mol1], mol_is_test[mol2]
disjoint_train = (~row1_test) & (~row2_test)     # both molecules in the training fold
disjoint_test = row1_test & row2_test            # both molecules in the test fold (the "both-unseen" case);
                                                  # a row with one molecule on each side lands in neither mask

add_random_r2 = fit_additive_least_squares(row_train, row_test)
add_disjoint_r2 = fit_additive_least_squares(disjoint_train, disjoint_test)
sym_random_r2, sym_model = fit_ridge(X_sym, row_train, row_test)
sym_disjoint_r2, _ = fit_ridge(X_sym, disjoint_train, disjoint_test)
_, ord_model = fit_ridge(X_ord, row_train, row_test)

print(f"additive model:     random-split R^2={add_random_r2:.2f}, molecule-disjoint R^2={add_disjoint_r2:.2f}")
print(f"symmetric-fp model: random-split R^2={sym_random_r2:.2f}, molecule-disjoint R^2={sym_disjoint_r2:.2f}")

assert add_random_r2 > 0.5                     # (a) the additive model does well on the random row split...
assert add_disjoint_r2 < 0.05                  # ... but no better than predicting the mean once split by molecule
assert sym_disjoint_r2 > 0.4                   # (b) the symmetric-fingerprint model keeps real accuracy there...
assert sym_disjoint_r2 > 0.6 * sym_random_r2   # ... at least 60% of its own random-split R^2...
assert sym_disjoint_r2 - add_disjoint_r2 > 0.3   # ... clearly ahead of the additive model on the hard split

# (c) order sensitivity: predict both orders of every held-out pair and compare
pred_ord_fwd = ord_model.predict(X_ord[row_test])
pred_ord_bwd = ord_model.predict(np.concatenate([fp2[row_test], fp1[row_test]], axis=1))
mean_abs_diff_ord = np.mean(np.abs(pred_ord_fwd - pred_ord_bwd))

pred_sym_fwd = sym_model.predict(X_sym[row_test])
X_sym_swapped = np.concatenate([fp2[row_test] + fp1[row_test], fp2[row_test] * fp1[row_test]], axis=1)
pred_sym_bwd = sym_model.predict(X_sym_swapped)
mean_abs_diff_sym = np.mean(np.abs(pred_sym_fwd - pred_sym_bwd))

print(f"mean |pred(a,b) - pred(b,a)|: ordered features={mean_abs_diff_ord:.3f}, "
      f"symmetric features={mean_abs_diff_sym:.3f}")
assert mean_abs_diff_ord > 0.05    # a real, order-dependent shift, in log10 units
assert mean_abs_diff_sym < 1e-9    # exactly zero (to floating point) once features are order-invariant

# (d) the noise floor: even the best model possible -- least squares on the hidden features, in the exact form
# of the true function (x_a + x_b, plus every symmetric product x_a[j] x_b[k] + x_b[j] x_a[k]) -- only gets
# down to sigma on held-out rows, while the fingerprint model stays far above it
xa, xb = x[mol1], x[mol2]
j_idx, k_idx = np.triu_indices(dim)
sym_products = xa[:, j_idx] * xb[:, k_idx] + xb[:, j_idx] * xa[:, k_idx]
X_oracle = np.column_stack([np.ones(n_pairs), xa + xb, sym_products])
beta, *_ = np.linalg.lstsq(X_oracle[row_train], y[row_train], rcond=None)
oracle_rmse = math.sqrt(np.mean((y[row_test] - X_oracle[row_test] @ beta) ** 2))
sym_random_rmse = math.sqrt(np.mean((y[row_test] - pred_sym_fwd) ** 2))
print(f"noise floor RMSE={noise_floor_rmse:.3f} (sigma used to generate the data={sigma_noise}); "
      f"best possible model's test RMSE={oracle_rmse:.3f}; "
      f"symmetric-fingerprint model's test RMSE={sym_random_rmse:.3f}")
assert 0.95 * sigma_noise < oracle_rmse < 1.05 * sigma_noise   # the best model possible lands on the floor...
assert sym_random_rmse > oracle_rmse                          # ... and the fingerprint model sits above it

print("all checks passed")
```

</details>

</details>
