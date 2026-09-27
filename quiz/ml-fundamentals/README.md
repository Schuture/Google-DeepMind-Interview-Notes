# ML Fundamentals: Metrics, Losses, Optimisers and Regularisation

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with derivations | ★★★★★ | Medium | RS · RE · MLE · Applied AI · Intern | precision-recall, roc-auc, cross-entropy, logistic-regression, adam, weight-decay, bias-variance, normalisation, huber-loss, gan, distribution-shift | 15 questions / 60 min | Skills interview |
<!-- meta:end -->

## Problem

Throughout, $\log$ is the natural logarithm and $\sigma(z) = 1/(1+e^{-z})$ is the logistic sigmoid.

### Evaluation

**Q1.** A binary classifier is evaluated on $1{,}000$ labelled examples and produces the confusion matrix $\mathrm{TP}=40$, $\mathrm{FP}=10$, $\mathrm{FN}=20$, $\mathrm{TN}=930$ (true/false positive/negative counts). Define accuracy, precision, recall and the F1 score in terms of these four counts, compute all four, and say which one is the most misleading summary of this classifier's quality on this data, and why. Give one realistic scenario where you would tune the decision threshold to favour precision over recall, and one where you would do the reverse.

**Q2.** A binary classifier assigns a real-valued score to each example; a prediction is positive exactly when its score exceeds a threshold $t$. Write $\mathrm{TPR}(t)$ and $\mathrm{FPR}(t)$ for the true- and false-positive rates at $t$ — the fraction of actual positives, respectively actual negatives, scored above $t$. The *ROC curve* is $\{(\mathrm{FPR}(t), \mathrm{TPR}(t)) : t \in \mathbb R\}$, and *ROC-AUC* is the area under it. Let $S^+$ and $S^-$ be the scores of an independently drawn positive and negative example. Derive

$$\mathrm{ROC\text{-}AUC} = P(S^+ > S^-) + \tfrac12 P(S^+ = S^-).$$

Then explain why the *precision–recall curve* (precision against recall, swept over the same thresholds) is the more informative summary under heavy class imbalance: state what happens to the ROC curve and to the precision–recall curve when every negative example in the evaluation set is duplicated ten times, and why.

### Losses

**Q3.** A single example has label $y \in \{0,1\}$ and a model produces a logit $z \in \mathbb R$, giving predicted probability $p = \sigma(z)$. Derive $\partial L/\partial z$ for two choices of per-example loss: *binary cross-entropy* $L_{\mathrm{bce}} = -[y \log p + (1-y)\log(1-p)]$, and *squared error* $L_{\mathrm{se}} = (p-y)^2$. Using the two derivatives, explain why training with squared error learns slowly when the model is confidently wrong (say $y=1$ and $z \ll 0$), while cross-entropy does not.

**Q4.** A logistic-regression model has weights $w \in \mathbb R^d$ and, given a design matrix $X \in \mathbb R^{n \times d}$ (row $i$ is $x_i^\top$) and labels $y \in \{0,1\}^n$, predicts $p_i = \sigma(x_i^\top w)$. Write the negative log-likelihood $L(w) = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$, derive its gradient $\nabla L(w) = X^\top(\sigma(Xw) - y)$ (with $\sigma$ applied elementwise) and its Hessian $\nabla^2 L(w) = X^\top S X$ with $S = \mathrm{diag}(p_i(1-p_i))$, and use the Hessian to show $L$ is convex on all of $\mathbb R^d$. Now suppose the data is *linearly separable*: some $w^\star$ satisfies $x_i^\top w^\star > 0$ whenever $y_i = 1$ and $x_i^\top w^\star < 0$ whenever $y_i = 0$, for every $i$. Say what happens to $\inf_w L(w)$ and to $\lVert w \rVert$ under unregularised gradient descent on this data, and how adding an $\ell_2$ penalty $\tfrac\lambda2\lVert w \rVert^2$ to $L$ changes the answer.

**Q5.** A model produces logits $z \in \mathbb R^K$ for a $K$-class problem, with $\mathrm{softmax}(z)_i = e^{z_i}/\sum_k e^{z_k}$, and the loss for an example of true class $y \in \{1,\dots,K\}$ is $L(z) = -\log \mathrm{softmax}(z)_y$. Derive $\nabla_z L = \mathrm{softmax}(z) - e_y$, where $e_y \in \mathbb R^K$ is the standard basis vector for class $y$. Explain, citing the identity that makes it valid, why a correct implementation subtracts $m = \max_k z_k$ from every logit before calling `exp`, both when forming $\mathrm{softmax}(z)$ and when forming $\log \sum_k e^{z_k}$.

### Optimisers

**Q6.** For a scalar parameter $\theta$ with gradient $g_t$ at step $t$ and learning rate $\eta$: plain SGD updates $\theta_{t+1} = \theta_t - \eta g_t$; SGD with momentum $\mu$ keeps a velocity $v_t = \mu v_{t-1} + g_t$ ($v_0 = 0$) and updates $\theta_{t+1} = \theta_t - \eta v_t$; Adam keeps $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ and $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ ($m_0 = v_0 = 0$), bias-corrects them to $\hat m_t = m_t/(1-\beta_1^t)$ and $\hat v_t = v_t/(1-\beta_2^t)$, and updates $\theta_{t+1} = \theta_t - \eta\, \hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Assuming $\mathbb E[g_t] = \bar g$ at every step $t$, derive $\mathbb E[m_t]$ in closed form and use it to explain why $m_t$ is divided by $1-\beta_1^t$. Then show that, for any $g_1$, the first Adam update satisfies $\hat m_1/(\sqrt{\hat v_1}+\epsilon) \approx \mathrm{sign}(g_1)$ — so the first step moves every coordinate by about $\eta$, regardless of how large or small that coordinate's gradient is.

**Q7.** Consider one SGD-family update with gradient $g$, learning rate $\eta$ and coefficient $\lambda > 0$. *$\ell_2$-regularised* SGD minimises $L(w) + \tfrac\lambda2\lVert w\rVert^2$ by gradient descent: $w \leftarrow w - \eta(g + \lambda w)$. *Decoupled weight decay* instead shrinks the weights by a fraction $\gamma$ every step, independently of the loss gradient: $w \leftarrow (1-\gamma) w - \eta g$. Show that these two coincide for every $w$ and $g$ once $\gamma = \eta\lambda$. Now consider Adam. *Adam with $\ell_2$ regularisation* adds $\lambda w$ to $g$ before forming $m_t, v_t$, then takes the ordinary Adam step. *AdamW* forms $m_t, v_t$ from $g$ alone and applies decoupled decay directly: $w \leftarrow (1-\gamma) w - \eta\, \hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Explain, with reference to $\hat v_t$, why these two no longer coincide at $\gamma = \eta\lambda$, or at any other matching of the coefficients.

**Q8.** Training uses the Adam optimiser with target (post-warmup) learning rate $\eta$. *Learning-rate warmup* ramps the learning rate linearly from $0$ to $\eta$ over the first few thousand steps rather than using $\eta$ from step $1$; *learning-rate decay* (cosine or linear) ramps it back down towards $0$ over the last part of training. Explain the mechanism that makes warmup useful with Adam and with large batch sizes, with reference to how reliable an estimate $\hat v_t$ is after only a few steps. Separately, for plain SGD minimising a stochastic objective with a *constant* learning rate $\eta$, state what happens to the iterates instead of converging to the minimum, and how the size of the resulting spread around the minimum depends on $\eta$; use this to explain why decaying $\eta$ later in training helps.

### Generalisation and regularisation

**Q9.** Fix an input $x$ and suppose $y = f(x) + \varepsilon$ for an unknown function $f$ and noise $\varepsilon$ with $\mathbb E[\varepsilon] = 0$, $\mathrm{Var}(\varepsilon) = \sigma^2$, independent of the training data. Given a random training set $D$, a learning algorithm produces a predictor $\hat f_D$; write $\bar f(x) = \mathbb E_D[\hat f_D(x)]$. Derive the bias–variance decomposition

$$\mathbb E_{D,\varepsilon}\bigl[(y - \hat f_D(x))^2\bigr] = \underbrace{(\bar f(x) - f(x))^2}_{\mathrm{bias}^2} + \underbrace{\mathbb E_D\bigl[(\hat f_D(x) - \bar f(x))^2\bigr]}_{\mathrm{variance}} + \sigma^2,$$

stating at which step $\mathbb E[\varepsilon] = 0$ is used and at which step the independence of $\varepsilon$ from $D$ is used.

**Q10.** For a design matrix $X \in \mathbb R^{n \times d}$ with $\mathrm{rank}(X) = d$, targets $y \in \mathbb R^n$ and $\lambda > 0$, *ridge regression* minimises $\lVert y - Xw \rVert^2 + \lambda \lVert w \rVert^2$. Derive the closed form $\hat w_\lambda = (X^\top X + \lambda I)^{-1} X^\top y$. Writing the thin singular value decomposition $X = U \Sigma V^\top$ with singular values $d_1,\dots,d_d$, express $\hat w_\lambda$ in terms of $U, V, \Sigma, y$, and show that, in the coordinates given by $V$, ridge shrinks the ordinary-least-squares coefficient along the $i$-th singular direction by the factor $d_i^2/(d_i^2+\lambda)$. Then take a design with orthonormal columns ($X^\top X = I$): on it, $\arg\min_w \lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2$ (ridge, as above) reduces to $d$ independent one-dimensional problems, and so does $\arg\min_w \tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$ (*lasso*, the $\ell_1$-penalised analogue, written with the customary leading $\tfrac12$ so that its threshold comes out as a clean $\lambda$). Derive both coordinatewise closed forms, and use them to explain why lasso commonly sets coefficients to exactly zero while ridge does not.

**Q11.** *Inverted dropout* with drop probability $p \in (0,1)$ replaces each activation $h$ of a layer, independently, with $\tilde h = \frac{m}{1-p} h$, where $m \sim \mathrm{Bernoulli}(1-p)$ (so the unit is kept, $m=1$, with probability $1-p$). Derive $\mathbb E[\tilde h]$, and say what it would be without the factor $1/(1-p)$. State what a layer using inverted dropout computes at evaluation time, and why no further rescaling is needed there.

**Q12.** For a batch of $N$ examples, each a $C$-dimensional feature vector, written as $h \in \mathbb R^{N \times C}$ with entries $h_{n,c}$: *batch normalisation* computes, per channel $c$, a mean and variance over the $N$ examples and normalises column $c$ with them; *layer normalisation* computes, per example $n$, a mean and variance over its own $C$ features and normalises row $n$ with them. Write both formulas (before any learned affine transform) and state which axis each one reduces over. Describe what batch normalisation does differently at inference time from training time, and why that difference is necessary. Explain why transformer architectures normalise with layer normalisation (or RMSNorm) rather than batch normalisation.

### Regression losses and generative models

**Q13.** A regression model predicts $f(x_i)$ for each of $n$ training examples $(x_i, y_i)$, with residual $r_i = y_i - f(x_i)$; two common training objectives are the *L2 loss* $\sum_i r_i^2$ and the *L1 loss* $\sum_i |r_i|$. For the special case of a constant predictor, $f(x_i) = c$ for every $i$, derive the value of $c$ that minimises $\sum_i (y_i-c)^2$ from its derivative, and the value that minimises $\sum_i |y_i-c|$ from the sign of each term and the subgradient of $|\cdot|$ at $0$ (the interval $[-1,1]$ of slopes consistent with its kink there); name the classical statistic each minimiser is, and explain the consequence for how far a single large outlier in $\{y_i\}$ can move it. Give $\partial r^2/\partial r$ and a subgradient of $|r|$ as functions of the residual $r$, and describe each at $r=0$ and as $|r|\to\infty$. State the noise model on the residuals under which minimising each of the two losses is maximum likelihood estimation, and derive one of the two correspondences from the noise model's log-likelihood. Define the *Huber loss* with threshold $\delta>0$,

$$L_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta\left(|r| - \tfrac12\delta\right) & |r| > \delta \end{cases},$$

give $\partial L_\delta/\partial r$, and say when it is preferred to using the L1 or L2 loss alone. Close with one sentence contrasting the same two losses' *other* common role, as regularisers added to a loss rather than as the loss on the residual itself (see Q10, without repeating its derivation).

**Q14.** A *generative adversarial network* (GAN) trains a generator $G$ and a discriminator $D$ against each other on the minimax objective

$$V(D,G) = \mathbb E_{x\sim p_{\mathrm{data}}}[\log D(x)] + \mathbb E_{z\sim p_z}[\log(1-D(G(z)))],$$

with $D$ maximising $V$ for a fixed $G$ and $G$ minimising $V$ for the resulting $D$; write $p_g$ for the distribution of $G(z)$ when $z\sim p_z$. Fixing $G$ (hence $p_g$), derive $D^\star = \arg\max_D V(D,G)$ pointwise in $x$ — maximise $a\log y + b\log(1-y)$ over $y\in(0,1)$ for constants $a,b>0$ — and show it is $D^\star(x) = p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x)+p_g(x))$. Writing $m = (p_{\mathrm{data}}+p_g)/2$ and $\mathrm{KL}(p\Vert q) = \sum_x p(x)\log(p(x)/q(x))$ for the Kullback–Leibler divergence between discrete distributions $p,q$, the *Jensen–Shannon divergence* is $\mathrm{JSD}(p\Vert q) = \tfrac12\mathrm{KL}(p\Vert m)+\tfrac12\mathrm{KL}(q\Vert m)$; show that $V(D^\star,G) = -\log 4 + 2\,\mathrm{JSD}(p_{\mathrm{data}}\Vert p_g)$, and use $\mathrm{JSD}\ge 0$ to say at which $p_g$ this is globally minimised over $G$. Writing $s$ for the discriminator's pre-sigmoid logit on a generated sample, so $D(G(z)) = \sigma(s)$, compare $\partial/\partial s\,\log(1-\sigma(s))$ — the original generator loss, minimised in the objective above — with $\partial/\partial s\,[-\log\sigma(s)]$ — the *non-saturating* generator loss used in practice instead — at $s \ll 0$, i.e. when $D(G(z))$ is close to $0$, and explain why practice minimises the second rather than the first early in training. Name two distinct failure modes of GAN training other than this vanishing gradient, and give one concrete mitigation for each.

### From offline to online

**Q15.** A new model scores higher than the current production model on the offline held-out test set, but after it is launched, its online metric (for example, click-through rate) is worse than production's. List the distinct causes you would investigate for this discrepancy — five are expected — and define each precisely (for a change in the data distribution, distinguish its standard types by which distribution changes). For each cause, give the concrete check you would run to confirm or rule it out.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two conventions worth pinning down before answering out loud: the momentum and Adam update rules below use an unnormalised velocity/moment term, the common convention in ML courses and frameworks — a textbook that instead weights the momentum term by $(1-\mu)$ describes the same algorithm up to a rescaling of the effective learning rate. And for Q11, "drop probability $p$" is assumed rather than "keep probability"; worth confirming which one an interviewer has in mind before writing $1-p$ or $p$ into a formula.

### Evaluation

**Q1.** Accuracy is $0.97$, precision $0.8$, recall $2/3 \approx 0.667$, F1 $\approx 0.727$, and accuracy is the misleading one. From the counts: $\mathrm{accuracy} = (\mathrm{TP}+\mathrm{TN})/n = 970/1000$; $\mathrm{precision} = \mathrm{TP}/(\mathrm{TP}+\mathrm{FP}) = 40/50$; $\mathrm{recall} = \mathrm{TP}/(\mathrm{TP}+\mathrm{FN}) = 40/60$; F1 is the harmonic mean $2\,\mathrm{precision}\cdot\mathrm{recall}/(\mathrm{precision}+\mathrm{recall})$, not the arithmetic mean — the harmonic mean penalises a low score in either term much more heavily than an average would. The data is imbalanced ($60$ positives out of $1{,}000$), and a classifier that ignores its input and always predicts "negative" already scores $940/1000 = 0.94$ accuracy on it, so this classifier's $0.97$ is barely more informative than that trivial baseline, even though it misses a third of the actual positives (recall $0.667$) — a symptom accuracy hides and recall exposes directly. Precision is the one to favour when a false positive is the expensive mistake, for example flagging a legitimate email as spam; recall is the one to favour when a false negative is the expensive mistake, for example screening for a disease with a cheap test, where a missed case delays treatment and a false alarm only costs a follow-up test.

**Q2.** ROC-AUC is exactly the probability that a random positive outscores a random negative, ties counting for half; sweeping the classification threshold down through the sorted scores turns this into an area computation. Order all $n^++n^-$ scores from high to low and move the threshold down through them one distinct value at a time. A step that newly includes one negative $j$ moves the curve right by $1/n^-$, at whatever height $\mathrm{TPR}$ has already reached — the fraction of positives already included, $\#\{i : s_i^+ > s_j^-\}/n^+$ — while a step that includes a positive moves the curve up without changing $\mathrm{FPR}$. Summing the area contributed by every rightward step gives $\frac{1}{n^+n^-}\sum_{i,j} \mathbb 1(s_i^+ > s_j^-)$, which is exactly $P(S^+ > S^-)$ for a uniformly random pair $(i,j)$. Where several examples share a score they must move together, since no threshold separates them: a group of $a$ tied positives and $b$ tied negatives moves the curve along a straight diagonal, whose trapezoid area is $\mathrm{TPR}\cdot(b/n^-) + \tfrac12(a/n^+)(b/n^-)$ — full credit for the pairs already strictly ahead, and exactly half credit split over the $ab$ newly tied pairs. Summing over the whole sweep gives $P(S^+>S^-) + \tfrac12 P(S^+=S^-)$, matching scikit-learn's `roc_auc_score` to numerical precision, as checked below. $\mathrm{TPR}(t)$ and $\mathrm{FPR}(t)$ are both within-class fractions, so duplicating every negative ten times changes neither of them: the ROC curve and ROC-AUC are exactly unchanged. Precision at $t$ is $\mathrm{TP}(t)/(\mathrm{TP}(t)+\mathrm{FP}(t))$; the duplication multiplies $\mathrm{FP}(t)$ by ten while leaving $\mathrm{TP}(t)$ alone, so precision falls at every threshold with $\mathrm{FP}(t)>0$, and average precision (the area under the precision–recall curve) falls by more than $0.1$ on the data used in the checks below — the practical cost of far more negatives a positive prediction must now be right against, which ROC-AUC cannot see because it never divides by a class count.

### Losses

**Q3.** Both derivatives reduce to a clean closed form; cross-entropy's stays large exactly where squared error's collapses. Writing $\sigma'(z) = \sigma(z)(1-\sigma(z))$ (immediate from $\sigma(z) = 1/(1+e^{-z})$ by the quotient rule) and $\partial p/\partial z = \sigma'(z) = p(1-p)$:

$$\frac{\partial L_{\mathrm{bce}}}{\partial z} = \Bigl(-\frac{y}{p}+\frac{1-y}{1-p}\Bigr)p(1-p) = -y(1-p) + (1-y)p = p - y, \qquad \frac{\partial L_{\mathrm{se}}}{\partial z} = 2(p-y)\cdot p(1-p).$$

At $y=1$, $z \ll 0$ the model is confidently wrong: $p \to 0$. Cross-entropy's gradient $p - y \to -1$, a full-sized push to increase $z$. Squared error's gradient has the extra factor $p(1-p) \to 0$: the sigmoid has saturated, so even though $(p-y)$ is close to its worst possible value, $\partial p/\partial z$ is tiny, and the two nearly cancel — at $z=-20$ the two gradients differ by more than seven orders of magnitude in the checks below. Squared error only starts pushing hard once $p$ has already moved away from the saturated region, so a badly wrong, confident prediction corrects itself very slowly under squared error and quickly under cross-entropy.

**Q4.** The gradient and Hessian follow the same pattern as Q3, summed over examples; convexity follows from the Hessian being a sum of outer products with non-negative weights, and separability breaks the existence of a finite minimiser. With $z_i = x_i^\top w$, $p_i = \sigma(z_i)$, Q3 gives $\partial L/\partial z_i = p_i - y_i$, and by the chain rule $\partial L/\partial w = \sum_i (p_i-y_i)x_i = X^\top(\sigma(Xw)-y)$. Differentiating again, $\partial p_i/\partial w = p_i(1-p_i)x_i$, so $\nabla^2 L(w) = \sum_i p_i(1-p_i)\,x_ix_i^\top = X^\top S X$. For any $v \in \mathbb R^d$, $v^\top X^\top S X v = \sum_i p_i(1-p_i)(x_i^\top v)^2 \ge 0$, since $p_i(1-p_i) \ge 0$ for $p_i \in (0,1)$ and every squared term is non-negative — the Hessian is positive semidefinite everywhere, so $L$ is convex on $\mathbb R^d$. On separable data, scaling $w = t\,w^\star$ and letting $t \to \infty$ drives every signed margin $x_i^\top w$ ($+$ for $y_i=1$, $-$ for $y_i=0$) to $+\infty$, so every $p_i \to y_i$ and $L(tw^\star) \to 0$ monotonically: $\inf_w L(w) = 0$, but it is not attained at any finite $w$, so unregularised gradient descent keeps increasing $\lVert w \rVert$ without bound, for as long as it runs. Adding $\tfrac\lambda2\lVert w\rVert^2$ makes the objective $L(w) + \tfrac\lambda2\lVert w\rVert^2 \to \infty$ as $\lVert w \rVert \to \infty$ (since $L \ge 0$), while its Hessian $X^\top S X + \lambda I$ is now positive *definite* everywhere ($\lambda I$ alone already is), so the regularised objective is strictly convex and coercive: it has a unique, finite minimiser regardless of separability, and gradient descent converges to it instead of diverging.

**Q5.** The gradient is softmax minus the one-hot target; the max-subtraction is an identity that leaves softmax and log-sum-exp exactly unchanged in exact arithmetic while keeping every exponent $\le 0$ in floating point. Write $L(z) = -z_y + \log\sum_k e^{z_k}$. The first term contributes $-\partial z_y/\partial z_i = -\delta_{iy}$; for the second, $\partial/\partial z_i \log\sum_k e^{z_k} = e^{z_i}/\sum_k e^{z_k} = \mathrm{softmax}(z)_i$. So $\partial L/\partial z_i = \mathrm{softmax}(z)_i - \delta_{iy}$, i.e. $\nabla_z L = \mathrm{softmax}(z) - e_y$. For any constant $m$, $e^{z_k} = e^m \cdot e^{z_k - m}$, so $\sum_k e^{z_k} = e^m \sum_k e^{z_k-m}$: in the softmax ratio the $e^m$ factor cancels top and bottom exactly, and $\log \sum_k e^{z_k} = m + \log\sum_k e^{z_k-m}$ exactly, for any $m$ — so subtracting $m = \max_k z_k$ changes neither quantity mathematically. What it changes is the floating-point computation: every shifted exponent $z_k - m$ is $\le 0$, so `exp` can never overflow, and the largest term is exactly $e^0=1$, so the sum can never underflow to $0$ either, however large or spread out the original logits are.

### Optimisers

**Q6.** SGD's step is the gradient times $\eta$; momentum accumulates gradients with decay $\mu$; Adam normalises each coordinate's step by its own recent gradient scale, and its bias correction exists because the moment estimate starts at zero and is dragged toward zero early on. Unrolling $m_t$ from $m_0 = 0$: $m_t = (1-\beta_1)\sum_{k=1}^t \beta_1^{t-k}g_k$, so under $\mathbb E[g_k]=\bar g$ for every $k$,

$$\mathbb E[m_t] = (1-\beta_1)\bar g \sum_{k=1}^t \beta_1^{t-k} = (1-\beta_1)\bar g\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,\bar g,$$

using the finite geometric sum $\sum_{j=0}^{t-1}\beta_1^j = (1-\beta_1^t)/(1-\beta_1)$. So $m_t$ underestimates $\bar g$ by the factor $1-\beta_1^t$, close to $0$ for small $t$ (the average has barely started) and close to $1$ for large $t$; dividing by exactly this factor, $\hat m_t = m_t/(1-\beta_1^t)$, gives $\mathbb E[\hat m_t] = \bar g$ for every $t$, removing the bias completely under this stationarity assumption — confirmed by Monte Carlo below, to within its sampling noise. At $t=1$, $m_1 = (1-\beta_1)g_1$, so $\hat m_1 = g_1$ *exactly*, whatever $\beta_1$ is (the correction factor $1-\beta_1^1=1-\beta_1$ cancels it precisely), and likewise $\hat v_1 = g_1^2$ exactly. So $\hat m_1/(\sqrt{\hat v_1}+\epsilon) = g_1/(|g_1|+\epsilon)$, which equals $\mathrm{sign}(g_1)$ up to an error that vanishes as $|g_1|/\epsilon \to \infty$: the first Adam step has size $\eta\cdot|g_1|/(|g_1|+\epsilon) \approx \eta$ in every coordinate — checked below to within $0.01\%$ of $\eta$ for gradients spanning six orders of magnitude — because step $1$'s bias correction is exact regardless of how large or small that coordinate's gradient happens to be.

**Q7.** For SGD the two update rules are the same formula in different notation; for Adam they are not, because $\ell_2$'s decay term rides through the adaptive denominator and decoupled decay does not. Substituting $\gamma = \eta\lambda$ into decoupled decay, $w \leftarrow (1-\eta\lambda)w - \eta g = w - \eta g - \eta\lambda w$, which is exactly the $\ell_2$-regularised update $w - \eta(g+\lambda w)$ — the two coincide termwise, for every $w$ and $g$, with no further conditions, as checked below. For Adam, $\ell_2$ regularisation folds $\lambda w$ into $g$ *before* $m_t$ and $v_t$ are formed, so the decay contribution is divided by $\sqrt{\hat v_t}$ together with the rest of the gradient: a coordinate with a large recent gradient magnitude (large $\hat v_t$) has its effective decay shrunk relative to a coordinate with a small $\hat v_t$, even though both weights should shrink by the same fraction $\gamma$. AdamW keeps $\lambda$ out of $m_t$ and $v_t$ entirely and applies $(1-\gamma)$ identically to every coordinate, so its decay strength never depends on that coordinate's gradient history. Since $\hat v_t$ genuinely differs across coordinates and across time — that is the entire point of Adam's per-coordinate adaptivity — no single choice of $\gamma$ makes the two update rules coincide in general, as confirmed below on gradients of different per-step scales.

**Q8.** Warmup protects against an unreliable $\hat v_t$ early in training; decay shrinks the noise-driven spread around the minimum that a constant learning rate leaves behind. $\hat v_1 = g_1^2$ (Q6): after a single step, Adam's estimate of a coordinate's gradient scale is built from exactly one noisy sample, so an unlucky large or small $g_1$ is taken entirely at face value, and the resulting step is still $\approx \eta$ regardless — there has been no chance yet for the exponential average to smooth out minibatch noise. With large batches this matters more because the per-step learning rate is itself usually scaled up (to compensate for fewer steps per epoch), so the same relative unreliability in early $\hat v_t$ turns into a larger absolute risk of a bad early step; ramping $\eta$ up from $0$ keeps the *actual* step size small during precisely this unreliable window, giving $\hat v_t$ time to average over enough steps to settle. For the decay half: on the quadratic $f(\theta)=\theta^2/2$ (minimum at $0$) with stochastic gradient $g_t = \theta_t + \xi_t$, $\xi_t \sim N(0,\sigma^2)$ i.i.d., SGD gives $\theta_{t+1} = (1-\eta)\theta_t - \eta\xi_t$, a linear recursion whose stationary variance solves $\mathrm{Var}_\infty = (1-\eta)^2\mathrm{Var}_\infty + \eta^2\sigma^2$, i.e.

$$\mathrm{Var}_\infty = \frac{\eta^2\sigma^2}{1-(1-\eta)^2} = \frac{\eta\sigma^2}{2-\eta}.$$

The iterates do not converge to the minimum at all: they converge in distribution to a spread of variance $\mathrm{Var}_\infty$ around it (confirmed by simulation below, to within the tolerance used there), a "noise ball" whose size grows with $\eta$. Decaying $\eta$ late in training shrinks this ball — continuing the same recursion at a smaller $\eta$ settles to the smaller variance the same formula predicts — trading the faster but noisier early progress of a large step size for a tighter final resting place around the optimum.

### Generalisation and regularisation

**Q9.** Two completions of the square, using $\mathbb E[\varepsilon]=0$ once and the independence of $\varepsilon$ from $D$ once. Write $y - \hat f_D(x) = \bigl(f(x)-\hat f_D(x)\bigr) + \varepsilon$; squaring and taking $\mathbb E_{D,\varepsilon}$, and using the independence of $\varepsilon$ from $D$ to factor the cross term,

$$\mathbb E_{D,\varepsilon}\bigl[(y-\hat f_D(x))^2\bigr] = \mathbb E_D\bigl[(f(x)-\hat f_D(x))^2\bigr] + 2\,\mathbb E_D[f(x)-\hat f_D(x)]\cdot\mathbb E[\varepsilon] + \mathbb E[\varepsilon^2];$$

$\mathbb E[\varepsilon]=0$ kills the middle term outright, and $\mathbb E[\varepsilon^2]=\mathrm{Var}(\varepsilon)=\sigma^2$, leaving $\mathbb E_D[(f(x)-\hat f_D(x))^2] + \sigma^2$. Now split $f(x)-\hat f_D(x) = (f(x)-\bar f(x)) + (\bar f(x)-\hat f_D(x))$ and square again:

$$\mathbb E_D\bigl[(f(x)-\hat f_D(x))^2\bigr] = (f(x)-\bar f(x))^2 + 2(f(x)-\bar f(x))\underbrace{\mathbb E_D[\bar f(x)-\hat f_D(x)]}_{=\,0} + \mathbb E_D\bigl[(\hat f_D(x)-\bar f(x))^2\bigr],$$

where the middle term vanishes because $\mathbb E_D[\hat f_D(x)] = \bar f(x)$ by definition — leaving exactly $\mathrm{bias}^2 + \mathrm{variance}$, and combined with the $\sigma^2$ above, the full decomposition. On a simulation with a badly under-fit model (a straight line fit to a curved true function) the bias² term comes out more than three times the variance term, and the three terms sum to the measured mean squared error to within the Monte Carlo tolerance used below.

**Q10.** Ridge's closed form comes from setting the gradient to zero; its SVD form shows shrinkage acting independently on each principal direction; on an orthonormal design lasso's closed form is soft-thresholding, which has a flat region at exactly zero that ridge's smooth penalty does not. $\nabla_w\bigl[\lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2\bigr] = -2X^\top(y-Xw) + 2\lambda w = 0 \iff (X^\top X+\lambda I)w = X^\top y$, and $X^\top X + \lambda I$ is positive definite (hence invertible) for $\lambda>0$, giving the unique minimiser $\hat w_\lambda = (X^\top X+\lambda I)^{-1}X^\top y$. With $X=U\Sigma V^\top$, $X^\top X = V\Sigma^2 V^\top$, so $X^\top X+\lambda I = V(\Sigma^2+\lambda I)V^\top$ (using $VV^\top=I$) and

$$\hat w_\lambda = V(\Sigma^2+\lambda I)^{-1}\Sigma U^\top y = V\,\mathrm{diag}\Bigl(\frac{d_i}{d_i^2+\lambda}\Bigr)U^\top y.$$

The ordinary-least-squares solution is the $\lambda=0$ case, $\hat w_{\mathrm{ols}} = V\,\mathrm{diag}(1/d_i)U^\top y$; comparing the two term by term in $V$'s coordinates, ridge multiplies the $i$-th OLS coefficient by $d_i^2/(d_i^2+\lambda)$ — close to $1$ when $d_i^2 \gg \lambda$ (well-determined directions barely move) and close to $0$ when $d_i^2 \ll \lambda$ (barely-determined directions are shrunk almost to nothing). When $X^\top X=I$, writing $c = X^\top y$, the same expansion gives $\lVert y-Xw\rVert^2 = \mathrm{const} + \sum_i(w_i-c_i)^2$, so $\lVert y-Xw\rVert^2+\lambda\lVert w\rVert^2$ decouples to $(w_i-c_i)^2+\lambda w_i^2$ per coordinate; setting its derivative to zero, $2(w_i-c_i)+2\lambda w_i=0$, gives $w_i^\star = c_i/(1+\lambda)$ — ridge always returns a non-zero coefficient unless $c_i$ itself is exactly $0$. For $\arg\min_w \tfrac12\lVert y-Xw\rVert^2+\lambda\lVert w\rVert_1$ on the same design, the same completion of the square (now carrying an overall $\tfrac12$) decouples to $\tfrac12(w_i-c_i)^2+\lambda|w_i|$ per coordinate, whose minimiser is the soft threshold $w_i^\star=\mathrm{sign}(c_i)\max(|c_i|-\lambda,0)$: the subgradient of $\lambda|w_i|$ at $0$ spans the whole interval $[-\lambda,\lambda]$, so any $c_i$ with $|c_i|\le\lambda$ already satisfies the zero-subgradient optimality condition at $w_i=0$ exactly, not merely in a limit. Ridge's penalty has no such flat region — its gradient $2\lambda w_i$ is zero only exactly at $w_i=0$ — so it shrinks every coefficient continuously without ever forcing one there. In the checks below, lasso pushes at least two of six coefficients to exactly zero on one design, while every ridge coefficient computed from the closed form above stays larger than $10^{-4}$ in magnitude.

**Q11.** $\mathbb E[\tilde h] = h$ exactly, which is the point of the $1/(1-p)$ factor: $\mathbb E[\tilde h] = \mathbb E[m]\,h/(1-p) = (1-p)h/(1-p) = h$, using $\mathbb E[m]=1-p$ for $m\sim\mathrm{Bernoulli}(1-p)$. Without the factor, masking alone gives $\mathbb E[mh] = (1-p)h$: every activation would be biased down by the factor $1-p$, layer after layer, purely from the mechanics of dropout rather than anything learned — confirmed as the "uncorrected" case in the checks below. Because training's expected forward pass already equals the no-dropout computation, evaluation simply turns dropout off — every unit kept, no masking — and applies no further rescaling: that already matches what training computed in expectation, so nothing is left to correct.

**Q12.** Batch norm reduces over the batch axis, one pair of statistics per channel; layer norm reduces over the feature axis, one pair of statistics per example. Writing $\mu_c = \frac1N\sum_n h_{n,c}$, $\sigma_c^2 = \frac1N\sum_n(h_{n,c}-\mu_c)^2$, batch normalisation computes $\hat h_{n,c} = (h_{n,c}-\mu_c)/\sqrt{\sigma_c^2+\epsilon}$ — statistics shared by every example in the batch, one pair per channel $c$ (axis $0$). Writing $\mu_n = \frac1C\sum_c h_{n,c}$, $\sigma_n^2 = \frac1C\sum_c(h_{n,c}-\mu_n)^2$, layer normalisation computes $\hat h_{n,c} = (h_{n,c}-\mu_n)/\sqrt{\sigma_n^2+\epsilon}$ — statistics private to example $n$, one pair per row (axis $1$), independent of every other row and of the batch size. At inference, batch normalisation stops computing $\mu_c,\sigma_c^2$ from the current batch and uses running averages accumulated during training instead; this is necessary because inference batches can be small or even size $1$, where the variance *within the batch* is degenerate — exactly $0$ for a batch of one, collapsing every normalised value to $0$ regardless of the input, as shown below — a problem layer normalisation never has, since its statistics come from a single example's own features and need no batch context at all, training or not. This is also why transformers favour layer normalisation (or RMSNorm, which drops the mean-centring step and normalises only by $\sqrt{\frac1C\sum_c h_{n,c}^2}$): sequences are processed with variable length and are often decoded one token — one example — at a time, and per-example statistics behave identically whatever the batch size or composition, while batch statistics would couple unrelated sequences together during training and become unreliable or undefined at small batch sizes during inference.

### Regression losses and generative models

**Q13.** The L2 loss is minimised by the mean, the L1 loss by a median: a squared term's derivative is
unbounded and grows with the residual, while $|\cdot|$'s subgradient is bounded in $[-1,1]$ and only encodes
sign — so a single outlier can drag the L2 minimiser arbitrarily far while leaving the L1 minimiser
essentially fixed. Setting $\frac{d}{dc}\sum_i(y_i-c)^2=-2\sum_i(y_i-c)=0$ gives $c^\star=\frac1n\sum_i y_i$,
the mean. For $\sum_i|y_i-c|$, the derivative where it exists is
$\sum_i\mathrm{sign}(c-y_i)=\#\{y_i<c\}-\#\{y_i>c\}$, so a stationary point needs equal counts below and
above $c$; at $c$ equal to some $y_i$, the subgradient is that count difference plus a contribution from
$[-1,1]$, and $0$ lies in it exactly when $c$ is a median — unique when $n$ is odd. Directly,
$\partial r^2/\partial r=2r$ is unbounded as $|r|\to\infty$ and $0$ only at $r=0$; a subgradient of $|r|$ is
$\mathrm{sign}(r)=\pm1$ for $r\ne0$ and any value in $[-1,1]$ at $r=0$: an outlier's enormous $|r_i|$ pulls
$c$ through the L2 derivative in proportion to its size, but only ever contributes $\pm1$ through the L1
derivative — so only which side it is on matters, not its distance, confirmed below by moving one outlier's
value to $10^6$.

Assume residuals $r_i=y_i-f(x_i)$ i.i.d. with density $p$; minimising $-\sum_i\log p(r_i)$ over $f$ is
maximum likelihood. For Gaussian $p(r)\propto e^{-r^2/(2\sigma^2)}$, $-\log p(r)=r^2/(2\sigma^2)+\mathrm{const}$,
so minimising it is exactly minimising $\sum_i r_i^2$: the L2 loss is maximum likelihood under Gaussian
noise. For Laplace $p(r)\propto e^{-|r|/b}$, likewise $-\log p(r)=|r|/b+\mathrm{const}$, making the L1 loss
maximum likelihood under Laplace noise.

The Huber loss's derivative is $\partial L_\delta/\partial r=r$ for $|r|\le\delta$ and $\delta\,\mathrm{sign}(r)$
for $|r|>\delta$ (continuous at $|r|=\delta$): quadratic near a good fit, linear with bounded slope beyond it
— preferred when outlier contamination is present but not overwhelming, keeping the L2 loss's fast
convergence near the optimum while capping any single residual's damage as the L1 loss does. The same
kink-versus-smoothness split reappears when the two norms are regularisers rather than losses (Q10): an
$\ell_1$ penalty's kink at $0$ sets small coefficients to exactly zero, while an $\ell_2$ penalty's smooth
gradient only shrinks them.

**Q14.** The optimal discriminator is $D^\star(x)=p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x)+p_g(x))$; the
resulting value of the game is $-\log4$ plus twice the Jensen–Shannon divergence between the data and
generator distributions, globally minimised over $G$ exactly when $p_g=p_{\mathrm{data}}$; and near the
start of training, when the discriminator confidently rejects generated samples, the original generator
loss's gradient with respect to the discriminator's logit vanishes while the non-saturating loss's does not
— why practice minimises the latter.

For fixed $G$, substituting $x=G(z)$ turns $V(D,G)=\int\bigl[p_{\mathrm{data}}(x)\log D(x)+p_g(x)\log(1-D(x))\bigr]\,\mathrm dx$,
so $D(x)$ is chosen independently at every $x$. Maximising $a\log y+b\log(1-y)$ over $y\in(0,1)$ for
$a,b>0$: the derivative $a/y-b/(1-y)$ is zero at $y^\star=a/(a+b)$, and the second derivative is negative
there, confirming a maximum; with $a=p_{\mathrm{data}}(x)$, $b=p_g(x)$, this gives $D^\star$. Substituting
back with $m=(p_{\mathrm{data}}+p_g)/2$:

$$V(D^\star,G) = \mathrm{KL}(p_{\mathrm{data}}\Vert m) + \mathrm{KL}(p_g\Vert m) - \log4 = 2\,\mathrm{JSD}(p_{\mathrm{data}}\Vert p_g) - \log4,$$

using $\int p_{\mathrm{data}}=\int p_g=1$ to collect the two $-\log2$ terms and the definition of JSD. Since
JSD is an average of two non-negative KL divergences, $\mathrm{JSD}\ge0$ with equality iff
$p_g=p_{\mathrm{data}}$, so $V(D^\star,G)$ is globally minimised over $G$ exactly there, at $-\log4$.

Writing $D(G(z))=\sigma(s)$: the original generator loss $\log(1-\sigma(s))$ has
$\partial/\partial s\,\log(1-\sigma(s))=-\sigma(s)=-D(G(z))$, while the non-saturating loss $-\log\sigma(s)$
has $\partial/\partial s\,[-\log\sigma(s)]=-(1-\sigma(s))=D(G(z))-1$. At $s\ll0$, i.e. $D(G(z))\approx0$ —
the discriminator confidently rejecting the generator's samples, as it typically does early in training —
the first gradient vanishes exactly when $G$ most needs a signal, while the second stays $\approx-1$; so
practice minimises $-\mathbb E[\log D(G(z))]$ instead of $\mathbb E[\log(1-D(G(z)))]$.

Two distinct failure modes beyond this vanishing gradient: *mode collapse*, where $G$ settles on a small set
of outputs that reliably fool $D$ instead of covering $p_{\mathrm{data}}$'s full variety, mitigated by giving
$D$ cross-example batch statistics (*minibatch discrimination*); and unstable, oscillating training, where
alternating gradient steps never settle, mitigated by constraining $D$ to be Lipschitz (a gradient penalty,
as in WGAN-GP).

### From offline to online

**Q15.** Five causes, each with its own check. *Training–serving skew*: a feature computed differently — or
stale, or missing — online than in training and offline evaluation; confirmed by logging what serving
actually computed for live requests, recomputing it offline from the same raw inputs, and diffing the two (a
*shadow deployment* runs this on live traffic before users are affected); a stale or missing feature
typically shows a serving-time distribution collapsed to a default the offline data never takes.
*Distribution shift* between the offline test period and live traffic: *covariate shift* if $p(x)$ changes
but $p(y\mid x)$ does not, *label shift* if $p(y)$ changes but $p(x\mid y)$ does not, *concept shift* if
$p(y\mid x)$ itself changes; confirmed by comparing feature and label distributions between the two periods,
and by a *time-based backtest* — training only up to a cut-off and evaluating strictly after it, rather than
on a random split of all history, which hides drift a forward split catches (checked below). Under covariate
shift specifically, with source and target input densities known or estimable, *importance weighting* the
offline evaluation by $p_{\mathrm{target}}(x)/p_{\mathrm{source}}(x)$ gives a materially better online-accuracy
estimate: since $p(y\mid x)$ is unchanged between domains,
$\mathbb E_{x\sim\mathrm{target}}[\mathrm{correct}(x)] = \mathbb E_{x\sim\mathrm{source}}\bigl[\tfrac{p_{\mathrm{target}}(x)}{p_{\mathrm{source}}(x)}\mathrm{correct}(x)\bigr]$,
turning a source-domain expectation into the target-domain one (confirmed below). *Leakage* inflating the
offline estimate: a random rather than time-respecting split of time-ordered or grouped data, duplicate
examples across train and test, or *target leakage* — a feature encoding information not actually available
at serving time; confirmed the same way as the backtest above (a drifting relationship gives a clearly
higher random-split accuracy than a forward split), and target leakage specifically by checking, feature by
feature, whether its value would actually have existed before the label was known. A *metric–objective
mismatch*: confirmed by checking, across earlier launches and A/B tests, whether the offline metric has
predicted online gains — a weak relationship means it is not a faithful proxy. *Feedback effects*, where the
production model's own past decisions shaped the data: confirmed by evaluating both models on data it did
not select — randomised exploration traffic, or logs reweighted by its propensities — where an advantage
seen only on production-selected data disappears; an A/B test with guardrail metrics (for example, long-term
engagement) then catches loops no data collected under the old model could show.

<details>
<summary>Checks (runnable)</summary>

```python
import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from scipy.special import log_softmax as scipy_log_softmax, softmax as scipy_softmax
from sklearn.linear_model import Lasso, LogisticRegression, QuantileRegressor, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(2026)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---- Q1: confusion-matrix metrics
TP, FP, FN, TN = 40, 10, 20, 930
total = TP + FP + FN + TN
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)  # NOTE: harmonic mean of P, R -- not (P + R) / 2
assert (accuracy, round(precision, 4), round(recall, 4), round(f1, 4)) == (0.97, 0.8, 0.6667, 0.7273)
predict_all_negative_accuracy = (FP + TN) / total          # every actual negative is correct, every positive missed
assert math.isclose(predict_all_negative_accuracy, 0.94)
assert accuracy - predict_all_negative_accuracy < 0.04      # the classifier barely beats the trivial baseline

# ---- Q2: ROC curve and ROC-AUC
def brute_force_auc(pos_scores, neg_scores):
    pos, neg = np.asarray(pos_scores, dtype=float)[:, None], np.asarray(neg_scores, dtype=float)[None, :]
    wins = (pos > neg).sum()
    ties = (pos == neg).sum()                              # NOTE: ties count for one half, not zero or one
    return (wins + 0.5 * ties) / (pos.size * neg.size)


n_pos, n_neg = 60, 200
pos_scores = np.round(rng.normal(1.0, 1.0, n_pos), 1)       # rounded scores so real ties occur
neg_scores = np.round(rng.normal(0.0, 1.0, n_neg), 1)
y_true = np.r_[np.ones(n_pos), np.zeros(n_neg)]
y_score = np.r_[pos_scores, neg_scores]
assert (pos_scores[:, None] == neg_scores[None, :]).sum() > 0          # confirms ties are actually exercised
auc = roc_auc_score(y_true, y_score)
assert math.isclose(auc, brute_force_auc(pos_scores, neg_scores), rel_tol=1e-9)

neg_scores_dup = np.repeat(neg_scores, 10)                  # every negative duplicated ten times
y_true_dup = np.r_[np.ones(n_pos), np.zeros(neg_scores_dup.size)]
y_score_dup = np.r_[pos_scores, neg_scores_dup]
assert math.isclose(roc_auc_score(y_true_dup, y_score_dup), auc, rel_tol=1e-9)     # ROC-AUC: unchanged
ap, ap_dup = average_precision_score(y_true, y_score), average_precision_score(y_true_dup, y_score_dup)
assert ap_dup < ap - 0.1                                     # average precision: markedly lower

# ---- Q3: cross-entropy vs squared-error gradient with respect to the logit
def bce_from_logit(z, y):
    # NOTE: log(sigmoid(z)) and log(1 - sigmoid(z)) lose precision by cancellation once |z| is a few
    # units from 0; log(1 + exp(-z)) and log(1 + exp(z)) (via logaddexp) do not.
    return y * np.logaddexp(0.0, -z) + (1 - y) * np.logaddexp(0.0, z)


def bce_grad(z, y):
    return sigmoid(z) - y


def se_from_logit(z, y):
    return (sigmoid(z) - y) ** 2


def se_grad(z, y):
    p = sigmoid(z)
    return 2 * (p - y) * p * (1 - p)


h_fd = 1e-6
for _ in range(2000):
    z, y = rng.normal(scale=4.0), float(rng.integers(0, 2))
    fd_bce = (bce_from_logit(z + h_fd, y) - bce_from_logit(z - h_fd, y)) / (2 * h_fd)
    fd_se = (se_from_logit(z + h_fd, y) - se_from_logit(z - h_fd, y)) / (2 * h_fd)
    assert math.isclose(fd_bce, bce_grad(z, y), abs_tol=1e-5)
    assert math.isclose(fd_se, se_grad(z, y), abs_tol=1e-5)

z_wrong = -20.0  # confidently wrong: true label is 1, model is sure it is 0
assert abs(bce_grad(z_wrong, 1.0)) > 0.999
assert abs(se_grad(z_wrong, 1.0)) < 1e-7                      # squared error: gradient has collapsed
assert abs(bce_grad(z_wrong, 1.0)) / abs(se_grad(z_wrong, 1.0)) > 1e7    # by more than seven orders of magnitude

# ---- Q4: logistic-regression NLL, gradient, Hessian, convexity, separability
def nll(w, X, y):
    return np.sum(bce_from_logit(X @ w, y))


def nll_grad(w, X, y):
    return X.T @ (sigmoid(X @ w) - y)


def nll_hessian(w, X):
    s = sigmoid(X @ w) * (1 - sigmoid(X @ w))
    return X.T @ (X * s[:, None])


for _ in range(200):
    n, d = rng.integers(5, 15), rng.integers(2, 6)
    X, y, w = rng.normal(size=(n, d)), rng.integers(0, 2, n).astype(float), rng.normal(size=d)
    v = rng.normal(size=d)
    fd_dir = (nll(w + h_fd * v, X, y) - nll(w - h_fd * v, X, y)) / (2 * h_fd)
    assert math.isclose(fd_dir, nll_grad(w, X, y) @ v, abs_tol=1e-4, rel_tol=1e-4)
    fd_hess = (nll_grad(w + h_fd * v, X, y) - nll_grad(w - h_fd * v, X, y)) / (2 * h_fd)
    H = nll_hessian(w, X)
    assert np.allclose(fd_hess, H @ v, atol=1e-4)
    # NOTE: eigvalsh, not eig -- H is symmetric by construction, but eig() can return spurious tiny
    # imaginary parts from rounding, and comparing a complex number with >= raises.
    assert np.linalg.eigvalsh(H).min() >= -1e-10

w_dir = rng.normal(size=5)
w_dir /= np.linalg.norm(w_dir)
half, gap, radius = 100, 3.0, 0.3
pos_half = gap * w_dir + rng.uniform(-1, 1, size=(half, 5)) * radius
neg_half = -gap * w_dir + rng.uniform(-1, 1, size=(half, 5)) * radius
X_sep, y_sep = np.vstack([pos_half, neg_half]), np.r_[np.ones(half), np.zeros(half)]
# every point's margin along w_dir exceeds gap - radius*sqrt(5) > 0 by Cauchy-Schwarz, so this is separable
# for every draw of the noise, not just with high probability
assert np.array_equal((X_sep @ w_dir > 0).astype(float), y_sep)


def gradient_descent(X, y, lam, eta, steps, checkpoint_every=200):
    w = np.zeros(X.shape[1])
    norms = []
    for t in range(steps):
        w = w - eta * (nll_grad(w, X, y) + lam * w)
        if t % checkpoint_every == 0 or t == steps - 1:
            norms.append(np.linalg.norm(w))
    return norms


norms_plain = gradient_descent(X_sep, y_sep, 0.0, 0.01, 4000)
norms_ridge = gradient_descent(X_sep, y_sep, 1.0, 0.01, 4000)
assert norms_plain[-1] > norms_plain[len(norms_plain) // 2] * 1.03     # unregularised: still growing (+3% or more)
tail = norms_ridge[-len(norms_ridge) // 4:]
assert max(tail) - min(tail) < 0.02 * norms_ridge[-1]                  # regularised: has settled (within 2%)

# ---- Q5: softmax cross-entropy gradient with respect to the logits, log-sum-exp
def log_sum_exp(z):
    # NOTE: subtract m before exp, not after -- shifting after exponentiating changes nothing.
    m = np.max(z, axis=-1, keepdims=True)
    return m[..., 0] + np.log(np.sum(np.exp(z - m), axis=-1))


def softmax_stable(z):
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


def softmax_ce_loss(z, y_idx):
    picked = np.take_along_axis(z, y_idx[..., None], axis=-1)[..., 0]
    return log_sum_exp(z) - picked


def softmax_ce_grad(z, y_idx):
    q = softmax_stable(z)
    onehot = np.zeros_like(z)
    np.put_along_axis(onehot, y_idx[..., None], 1.0, axis=-1)
    return q - onehot


for _ in range(500):
    K = int(rng.integers(2, 6))
    z, y_idx = rng.normal(scale=5.0, size=K), np.array(rng.integers(0, K))
    analytic = softmax_ce_grad(z[None, :], y_idx[None])[0]
    fd = np.array([(softmax_ce_loss((z + h_fd * e)[None, :], y_idx[None])[0]
                    - softmax_ce_loss((z - h_fd * e)[None, :], y_idx[None])[0]) / (2 * h_fd)
                   for e in np.eye(K)])
    assert np.allclose(analytic, fd, atol=1e-5)

for _ in range(50):                                      # matches an independent library implementation
    K = int(rng.integers(2, 8))
    z = rng.normal(scale=3.0, size=(4, K))
    assert np.allclose(softmax_stable(z), scipy_softmax(z, axis=-1))
    assert np.allclose(z - log_sum_exp(z)[:, None], scipy_log_softmax(z, axis=-1))

big_logits = np.array([1000.0, 1.0, 0.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(big_logits) / np.exp(big_logits).sum()
assert np.isnan(naive).any()                              # naive softmax: breaks
stable = softmax_stable(big_logits)
assert np.all(np.isfinite(stable)) and math.isclose(stable[0], 1.0, abs_tol=1e-12)     # shifted: exact and finite

# ---- Q6: SGD, momentum, Adam; bias correction; first-step size
def adam_moments(m, v, t, g, beta1, beta2):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    mhat = m / (1 - beta1 ** t)          # NOTE: the exponent is t, the number of updates so far -- using
    vhat = v / (1 - beta2 ** t)          #       t - 1 here would leave the very first step uncorrected
    return m, v, mhat, vhat


BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


def m_t_closed_form(grads, beta1):
    t = len(grads)
    weights = beta1 ** (t - 1 - np.arange(t))
    return (1 - beta1) * np.sum(weights * np.asarray(grads))


for _ in range(20):
    grads = rng.normal(size=int(rng.integers(1, 30)))
    m = 0.0
    for g in grads:
        m, _, _, _ = adam_moments(m, 0.0, 1, g, BETA1, BETA2)   # t is irrelevant to the m update itself
    assert math.isclose(m, m_t_closed_form(grads, BETA1), rel_tol=1e-9)

mu_g, sigma_g, reps = 2.5, 1.0, 20000
for t in (1, 3, 10):
    sample = rng.normal(mu_g, sigma_g, size=(reps, t))
    m = np.zeros(reps)
    for k in range(t):
        m = BETA1 * m + (1 - BETA1) * sample[:, k]
    assert math.isclose(m.mean(), (1 - BETA1 ** t) * mu_g, rel_tol=0.05, abs_tol=0.05)   # m_t is biased low
    assert math.isclose(m.mean() / (1 - BETA1 ** t), mu_g, rel_tol=0.05, abs_tol=0.05)    # correction fixes it

g1 = np.array([1e-3, 1.0, 1e3, -50.0])                    # gradients spanning six orders of magnitude
_, _, mhat1, vhat1 = adam_moments(0.0, 0.0, 1, g1, BETA1, BETA2)
assert np.allclose(mhat1, g1) and np.allclose(vhat1, g1 ** 2)      # exact at t=1, whatever beta1, beta2 are
step1 = 0.01 * mhat1 / (np.sqrt(vhat1) + ADAM_EPS)
assert np.all(np.isclose(np.abs(step1), 0.01, rtol=1e-4))          # every coordinate moves by about lr

# ---- Q7: L2 regularisation versus decoupled weight decay
def sgd_l2_step(w, g, eta, lam):
    return w - eta * g - eta * lam * w


def sgd_decoupled_step(w, g, eta, gamma):
    return (1 - gamma) * w - eta * g


for _ in range(200):
    d = int(rng.integers(1, 6))
    w, g = rng.normal(size=d), rng.normal(size=d)
    eta, lam = rng.uniform(0.001, 0.5), rng.uniform(0.0, 2.0)
    assert np.allclose(sgd_l2_step(w, g, eta, lam), sgd_decoupled_step(w, g, eta, eta * lam), atol=1e-12)


def adam_l2_run(w0, grads, eta, lam):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    for t, g in enumerate(grads, start=1):
        m, v, mhat, vhat = adam_moments(m, v, t, g + lam * w, BETA1, BETA2)      # decay folded into g first
        w = w - eta * mhat / (np.sqrt(vhat) + ADAM_EPS)                          # -- so it is divided by
    return w                                                                     #    sqrt(vhat) too


def adamw_run(w0, grads, eta, gamma):
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    for t, g in enumerate(grads, start=1):
        m, v, mhat, vhat = adam_moments(m, v, t, g, BETA1, BETA2)                # decay never touches g,
        w = (1 - gamma) * w - eta * mhat / (np.sqrt(vhat) + ADAM_EPS)            # m or v -- applied raw
    return w


w0 = rng.normal(size=4)
eta, lam = 0.05, 0.1
grads = [rng.normal(scale=s, size=4) for s in (0.01, 1.0, 5.0)]                  # gradients of very different
assert not np.allclose(adam_l2_run(w0, grads, eta, lam),                          # scale across coordinates
                        adamw_run(w0, grads, eta, eta * lam), atol=1e-3)

# ---- Q8: warmup and later decay, mechanism checked via a noisy quadratic
NOISE_SIGMA = 2.0


def stationary_var(eta):
    return eta * NOISE_SIGMA ** 2 / (2 - eta)                # requires 0 < eta < 2 for the recursion to be stable


def run_noisy_gd(eta, steps, n_chains, theta0):
    theta = np.full(n_chains, theta0)
    for _ in range(steps):
        theta = (1 - eta) * theta - eta * rng.normal(scale=NOISE_SIGMA, size=n_chains)
    return theta


n_chains = 20000
theta_big = run_noisy_gd(0.2, 400, n_chains, 0.0)
assert math.isclose(theta_big.var(), stationary_var(0.2), rel_tol=0.1)
theta_small = run_noisy_gd(0.05, 400, n_chains, 0.0)
assert math.isclose(theta_small.var(), stationary_var(0.05), rel_tol=0.1)
theta_decayed = theta_big.copy()                              # continue the SAME chains at a smaller eta
for _ in range(400):
    theta_decayed = (1 - 0.05) * theta_decayed - 0.05 * rng.normal(scale=NOISE_SIGMA, size=n_chains)
assert math.isclose(theta_decayed.var(), stationary_var(0.05), rel_tol=0.1)
assert theta_decayed.var() < theta_big.var() / 2               # decaying eta shrinks the steady-state spread

# ---- Q9: bias-variance decomposition, by simulation
def true_f(x):
    return np.sin(3.0 * x)


x_train, x0, noise_sigma, degree = np.linspace(-1.0, 1.0, 15), 0.7, 0.3, 1
R = 6000
preds, targets = np.empty(R), np.empty(R)
for r in range(R):
    y_train = true_f(x_train) + rng.normal(scale=noise_sigma, size=x_train.size)
    preds[r] = np.polyval(np.polyfit(x_train, y_train, degree), x0)
    targets[r] = true_f(x0) + rng.normal(scale=noise_sigma)   # NOTE: a *fresh* draw of test-point noise --
                                                                #       reusing training noise would be wrong
bias2, variance = (preds.mean() - true_f(x0)) ** 2, preds.var()
measured_mse = np.mean((targets - preds) ** 2)
assert math.isclose(measured_mse, bias2 + variance + noise_sigma ** 2, rel_tol=0.1)
assert bias2 > 3 * variance                                    # the degree-1 fit is dominated by bias, as expected

# ---- Q10: ridge closed form, SVD shrinkage, lasso vs ridge sparsity
n, d = 60, 8
X = rng.normal(size=(n, d))
y = X @ rng.normal(size=d) + rng.normal(scale=0.5, size=n)
lam_ridge = 2.5
w_closed = np.linalg.solve(X.T @ X + lam_ridge * np.eye(d), X.T @ y)
assert np.allclose(w_closed, Ridge(alpha=lam_ridge, fit_intercept=False).fit(X, y).coef_, atol=1e-8)

U, svals, Vt = np.linalg.svd(X, full_matrices=False)            # NOTE: svd returns V^T, not V
w_svd = Vt.T @ ((svals / (svals ** 2 + lam_ridge)) * (U.T @ y))
assert np.allclose(w_closed, w_svd, atol=1e-8)
ols_in_v_basis = (1.0 / svals) * (U.T @ y)
ridge_in_v_basis = (svals / (svals ** 2 + lam_ridge)) * (U.T @ y)
assert np.allclose(ridge_in_v_basis / ols_in_v_basis, svals ** 2 / (svals ** 2 + lam_ridge), atol=1e-10)

n_orth = 30
Q_orth, _ = np.linalg.qr(rng.normal(size=(n_orth, 6)))           # orthonormal columns: Q^T Q = I
y_orth = Q_orth @ np.array([2.0, -1.5, 0.05, 0.0, 3.0, -0.02]) + rng.normal(scale=0.15, size=n_orth)
c = Q_orth.T @ y_orth                                             # (1/2)||y-Xw||^2 = const + (1/2) sum_i (w_i-c_i)^2

lam_l1 = 0.9                                                      # minimises (1/2)(w_i-c_i)^2 + lam_l1|w_i| per i
soft_threshold = np.sign(c) * np.maximum(np.abs(c) - lam_l1, 0.0)
# NOTE: sklearn's Lasso objective is (1/(2n))||y-Xw||^2 + alpha||w||_1; alpha = lam_l1 / n matches it exactly
# to (1/2)||y-Xw||^2 + lam_l1||w||_1 up to the positive constant factor n, which does not move the minimiser
lasso_coef = Lasso(alpha=lam_l1 / n_orth, fit_intercept=False, tol=1e-12, max_iter=100000).fit(Q_orth, y_orth).coef_
assert np.allclose(soft_threshold, lasso_coef, atol=1e-6)
assert np.sum(soft_threshold == 0.0) >= 2

lam_l2 = 0.9                                                      # ridge: minimises ||y-Xw||^2 + lam_l2||w||^2
ridge_coef = Ridge(alpha=lam_l2, fit_intercept=False).fit(Q_orth, y_orth).coef_
assert np.allclose(ridge_coef, c / (1 + lam_l2), atol=1e-8)       # closed form on this orthonormal design
assert np.all(np.abs(ridge_coef) > 1e-4)                          # never exactly zero

# ---- Q11: inverted dropout preserves the expectation
p_drop, acts, reps = 0.3, np.array([2.5, -4.0, 0.8]), 400000
keep = rng.random((reps, acts.size)) > p_drop
inverted = keep * acts / (1 - p_drop)
assert np.allclose(inverted.mean(axis=0), acts, rtol=0.02, atol=0.02)
assert np.allclose(inverted.var(axis=0), p_drop / (1 - p_drop) * acts ** 2, rtol=0.05)
uncorrected = keep * acts                                          # NOTE: without the 1/(1-p) factor the mean
assert np.allclose(uncorrected.mean(axis=0), (1 - p_drop) * acts, rtol=0.02)     #       is biased down by (1-p)

# ---- Q12: batch norm versus layer norm
def batch_norm(x, eps=1e-5):
    mu, var = x.mean(axis=0, keepdims=True), x.var(axis=0, keepdims=True)   # NOTE: axis=0 -- across the batch,
    return (x - mu) / np.sqrt(var + eps)                                   #       one pair of stats per channel


def layer_norm(x, eps=1e-5):
    mu, var = x.mean(axis=1, keepdims=True), x.var(axis=1, keepdims=True)   # NOTE: axis=1 -- across the features
    return (x - mu) / np.sqrt(var + eps)                                   #       of one example, batch-independent


N, C = 5, 4
acts2 = rng.normal(size=(N, C)) * 3 + 1
bn, ln = batch_norm(acts2), layer_norm(acts2)
assert np.allclose(bn.mean(axis=0), 0.0, atol=1e-8) and np.allclose(bn.std(axis=0), 1.0, atol=1e-3)
assert np.allclose(ln.mean(axis=1), 0.0, atol=1e-8) and np.allclose(ln.std(axis=1), 1.0, atol=1e-3)

perturbed = acts2.copy()
perturbed[0] += 5.0
assert not np.allclose(batch_norm(perturbed)[1:], bn[1:])          # batch norm: every row shares the statistics
assert np.allclose(layer_norm(perturbed)[1:], ln[1:], atol=1e-10)  # layer norm: rows are independent

single = acts2[:1]
assert np.allclose(batch_norm(single), 0.0)                        # a batch of one has zero variance: collapses
running_mean, running_var = acts2.mean(axis=0), acts2.var(axis=0)
inference_bn = (single - running_mean) / np.sqrt(running_var + 1e-5)
assert not np.allclose(inference_bn, 0.0)                          # inference uses the running statistics instead
assert np.allclose(layer_norm(single), ln[:1])                     # layer norm needs no such special-casing

# ---- Q13: L1 vs L2 regression losses, Huber loss, MLE noise models
def huber(r, delta):
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= delta, 0.5 * r ** 2, delta * (np.abs(r) - 0.5 * delta))


def huber_grad(r, delta):
    r = np.asarray(r, dtype=float)
    return np.where(np.abs(r) <= delta, r, delta * np.sign(r))


y_reg = np.concatenate([rng.normal(10.0, 1.0, 30), [80.0]])   # 31 values (odd -- a unique median), one far outlier
bounds_c = (y_reg.min() - 10.0, 2e6)
c_l2 = minimize_scalar(lambda c: np.sum((y_reg - c) ** 2), bounds=bounds_c, method="bounded").x
c_l1 = minimize_scalar(lambda c: np.sum(np.abs(y_reg - c)), bounds=bounds_c, method="bounded").x
assert math.isclose(c_l2, y_reg.mean(), abs_tol=1e-2)
assert math.isclose(c_l1, np.median(y_reg), abs_tol=1e-2)

y_reg_moved = y_reg.copy()
y_reg_moved[-1] = 1e6                                            # push the single outlier much further away
c_l2_moved = minimize_scalar(lambda c: np.sum((y_reg_moved - c) ** 2), bounds=bounds_c, method="bounded").x
c_l1_moved = minimize_scalar(lambda c: np.sum(np.abs(y_reg_moved - c)), bounds=bounds_c, method="bounded").x
assert abs(c_l2_moved - c_l2) > 1000                              # L2 minimiser: dragged far by the outlier's size
assert math.isclose(c_l1_moved, np.median(y_reg_moved), abs_tol=1e-2)
assert abs(c_l1_moved - c_l1) < 1e-1                              # L1 minimiser: essentially unmoved

delta = 1.5
r_grid = np.concatenate([rng.uniform(-4.0, 4.0, 200), [delta, -delta]])   # NOTE: includes both sides of the kink
for r in r_grid:
    fd = (huber(r + h_fd, delta) - huber(r - h_fd, delta)) / (2 * h_fd)
    assert math.isclose(fd, float(huber_grad(r, delta)), abs_tol=1e-4)

n_reg, true_slope = 60, 2.0                                      # L1-loss linear fit versus OLS, contaminated data
x_reg = rng.normal(size=n_reg)
y_reg_lin = true_slope * x_reg + rng.normal(scale=0.2, size=n_reg)
x_outliers = np.array([2.5, -2.5, 2.8, -2.8])
y_outliers = np.array([-20.0, 20.0, -25.0, 25.0])                 # NOTE: sign deliberately opposes the true slope
x_contam = np.concatenate([x_reg, x_outliers])
y_contam = np.concatenate([y_reg_lin, y_outliers])
l1_slope = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(x_contam[:, None], y_contam).coef_[0]
ols_slope = np.polyfit(x_contam, y_contam, 1)[0]
assert abs(l1_slope - true_slope) < 0.3
assert abs(ols_slope - true_slope) > 3 * abs(l1_slope - true_slope)   # L1 fit: far closer to the true slope

# ---- Q14: GAN optimal discriminator, Jensen-Shannon divergence, saturating vs non-saturating generator loss
support_size = 6
p_data = rng.dirichlet(np.ones(support_size))
p_g = rng.dirichlet(np.ones(support_size))


def V_of_D(D_vals, p_data, p_g):
    D_vals = np.clip(D_vals, 1e-12, 1 - 1e-12)                    # NOTE: guard log() for the random D's tested below
    return np.sum(p_data * np.log(D_vals) + p_g * np.log(1 - D_vals))


D_star = p_data / (p_data + p_g)
v_star = V_of_D(D_star, p_data, p_g)
for _ in range(500):
    D_random = rng.uniform(1e-6, 1 - 1e-6, size=support_size)
    assert V_of_D(D_random, p_data, p_g) <= v_star + 1e-9          # D* maximises V pointwise, hence in total

def kl(p, q):
    return np.sum(p * np.log(p / q))


m_mix = (p_data + p_g) / 2
jsd_manual = 0.5 * kl(p_data, m_mix) + 0.5 * kl(p_g, m_mix)
jsd_scipy = jensenshannon(p_data, p_g, base=np.e) ** 2              # NOTE: scipy returns the square root of the JSD
assert math.isclose(jsd_manual, jsd_scipy, rel_tol=1e-9)
assert math.isclose(v_star, -math.log(4) + 2 * jsd_manual, rel_tol=1e-9)

p_same = rng.dirichlet(np.ones(support_size))                       # p_g = p_data: JSD collapses to 0
assert math.isclose(V_of_D(p_same / (2 * p_same), p_same, p_same), -math.log(4), abs_tol=1e-9)

d_val = 1e-3                                                        # D(G(z)) close to 0: early in training
s_val = math.log(d_val / (1 - d_val))                               # the logit with sigmoid(s_val) == d_val
assert math.isclose(sigmoid(s_val), d_val, rel_tol=1e-9)


def saturating_loss(s):
    return math.log(1 - sigmoid(s))


def nonsaturating_loss(s):
    return -math.log(sigmoid(s))


grad_saturating = (saturating_loss(s_val + h_fd) - saturating_loss(s_val - h_fd)) / (2 * h_fd)
grad_nonsaturating = (nonsaturating_loss(s_val + h_fd) - nonsaturating_loss(s_val - h_fd)) / (2 * h_fd)
assert math.isclose(grad_saturating, -d_val, abs_tol=1e-4)          # NOTE: -D(G(z)) -- vanishes as D(G(z)) -> 0
assert math.isclose(grad_nonsaturating, d_val - 1, abs_tol=1e-4)    # NOTE: D(G(z)) - 1 -- stays near -1
assert abs(grad_nonsaturating) > 500 * abs(grad_saturating)         # non-saturating: far larger gradient magnitude

# ---- Q15: offline/online gaps -- concept drift (random vs. forward split), training-serving skew, covariate shift
n_stream, d_stream = 4000, 2
frac = np.arange(n_stream) / (n_stream - 1)
w0_stream = np.array([3.0, 2.0])
theta_max = 3 * math.pi / 4                                        # NOTE: the true decision boundary rotates
theta = theta_max * frac                                          #       smoothly over time -- concept drift
cos_t, sin_t = np.cos(theta), np.sin(theta)
w_t = np.stack([w0_stream[0] * cos_t - w0_stream[1] * sin_t,
                w0_stream[0] * sin_t + w0_stream[1] * cos_t], axis=1)
X_stream = rng.normal(size=(n_stream, d_stream))
y_stream = (rng.random(n_stream) < sigmoid(np.sum(X_stream * w_t, axis=1))).astype(float)

perm = rng.permutation(n_stream)
n_train_s = int(0.7 * n_stream)
train_r, test_r = perm[:n_train_s], perm[n_train_s:]
acc_random = (LogisticRegression(max_iter=1000).fit(X_stream[train_r], y_stream[train_r])
              .score(X_stream[test_r], y_stream[test_r]))

train_f, test_f = np.arange(n_train_s), np.arange(n_train_s, n_stream)
clf_forward = LogisticRegression(max_iter=1000).fit(X_stream[train_f], y_stream[train_f])
acc_forward = clf_forward.score(X_stream[test_f], y_stream[test_f])

t_future = rng.integers(n_train_s, n_stream, size=20000)            # NOTE: fresh draws from the SAME later period
frac_future = t_future / (n_stream - 1)
theta_future = theta_max * frac_future
cos_f, sin_f = np.cos(theta_future), np.sin(theta_future)
w_future = np.stack([w0_stream[0] * cos_f - w0_stream[1] * sin_f,
                      w0_stream[0] * sin_f + w0_stream[1] * cos_f], axis=1)
X_future = rng.normal(size=(t_future.size, d_stream))
y_future = (rng.random(t_future.size) < sigmoid(np.sum(X_future * w_future, axis=1))).astype(float)
acc_future = clf_forward.score(X_future, y_future)

assert acc_random - acc_forward > 0.1         # random split: optimistic, hides the drift
assert abs(acc_forward - acc_future) < 0.03   # forward split: a faithful estimate of true future performance

n2, d2 = 3000, 3                                                     # training-serving skew
X_skew = rng.normal(size=(n2, d2))
w_true_skew = np.array([2.0, -1.5, 1.0])
y_skew = (rng.random(n2) < sigmoid(X_skew @ w_true_skew)).astype(float)
train_sk, test_sk = np.arange(n2 // 2), np.arange(n2 // 2, n2)
clf_skew = LogisticRegression(max_iter=1000).fit(X_skew[train_sk], y_skew[train_sk])
acc_clean = clf_skew.score(X_skew[test_sk], y_skew[test_sk])
X_served = X_skew[test_sk].copy()
X_served[:, 1] *= 5.0                          # NOTE: one feature computed on a different scale online than offline
acc_skewed = clf_skew.score(X_served, y_skew[test_sk])
assert acc_clean - acc_skewed > 0.05           # training-serving skew: a checked drop in accuracy


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


mu_source, mu_target, sigma_cs = 0.0, 2.0, 1.0                       # covariate shift: p(x) differs, p(y|x) shared
n_src_train, n_src_test, n_tgt_eval = 4000, 4000, 200000
x_src_train = rng.normal(mu_source, sigma_cs, n_src_train)
y_src_train = (rng.random(n_src_train) < sigmoid(2.0 * x_src_train)).astype(float)
clf_cs = LogisticRegression(max_iter=1000).fit(x_src_train[:, None], y_src_train)

x_src_test = rng.normal(mu_source, sigma_cs, n_src_test)
y_src_test = (rng.random(n_src_test) < sigmoid(2.0 * x_src_test)).astype(float)
correct_src = (clf_cs.predict(x_src_test[:, None]) == y_src_test).astype(float)
unweighted_estimate = correct_src.mean()
weights = normal_pdf(x_src_test, mu_target, sigma_cs) / normal_pdf(x_src_test, mu_source, sigma_cs)
weighted_estimate = np.sum(weights * correct_src) / np.sum(weights)

x_tgt_eval = rng.normal(mu_target, sigma_cs, n_tgt_eval)
y_tgt_eval = (rng.random(n_tgt_eval) < sigmoid(2.0 * x_tgt_eval)).astype(float)
true_target_accuracy = (clf_cs.predict(x_tgt_eval[:, None]) == y_tgt_eval).mean()

assert abs(weighted_estimate - true_target_accuracy) < abs(unweighted_estimate - true_target_accuracy)
assert abs(weighted_estimate - true_target_accuracy) < 0.02          # importance weighting: close to the truth

print("all checks passed")
```

</details>

</details>
