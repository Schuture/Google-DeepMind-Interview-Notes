# Maths Quiz: Probability, Statistics and Linear Algebra

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with derivations | ★★★☆☆ | Medium | RS · RE · MLE · Intern | bayes-theorem, expectation, markov-chains, maximum-likelihood, map-estimation, kl-divergence, svd, matrix-calculus, conditioning | 12 questions / 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

Throughout, $\log$ is the natural logarithm, $\mathbb{E}[\cdot]$ and $\mathrm{Var}(\cdot)$ are expectation and
variance, and $\mathcal{N}(\mu, \sigma^2)$ denotes the density of a Gaussian with that mean and variance.

### Probability

- **Q1.** Urn A holds 3 red balls and 1 blue ball; urn B holds 1 red ball and 3 blue balls. An urn is chosen
  by a fair coin toss, and one ball is drawn from it: the ball is red. Compute $P(A \mid \text{red})$. A
  second ball is then drawn from the same urn without replacement (the first ball is not returned); compute
  the probability that this second ball is red too.
- **Q2.** A fair coin is flipped repeatedly until the pattern HH first appears (two heads in a row); compute
  the expected number of flips. Do the same for the pattern HT, and explain why the two expectations differ.
- **Q3.** Let $X_1, \dots, X_n$ be independent $\mathrm{Uniform}(0, 1)$ random variables. Compute
  $\mathbb{E}[\max_i X_i]$ and $\mathbb{E}[\min_i X_i]$ as functions of $n$.
- **Q4.** A diagnostic test has sensitivity 99% (the probability of testing positive given the disease) and
  specificity 95% (the probability of testing negative given no disease). Disease prevalence in the tested
  population is 1%. Compute $P(\text{disease} \mid \text{positive})$, and explain in one sentence why it is
  so much lower than the sensitivity.

### Statistics

- **Q5.** Let $X_1, \dots, X_n$ be independent and identically distributed (i.i.d.) samples from
  $\mathcal{N}(\mu, \sigma^2)$ with both parameters unknown. Derive the maximum-likelihood estimates $\hat\mu$
  and $\hat\sigma^2$, and show that $\hat\sigma^2$ is a biased estimator of $\sigma^2$, with
  $\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$.
- **Q6.** Let $y = Xw + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, and put an
  independent $\mathcal{N}(0, \tau^2)$ prior on each weight $w_j$. Show that the maximum a posteriori (MAP)
  estimate of $w$ is the ridge-regression solution $\hat w = \arg\min_w \|Xw - y\|^2 + \lambda \|w\|^2$, for some $\lambda$ that you
  should express in terms of $\sigma^2$ and $\tau^2$. Then show that replacing the Gaussian prior with an
  independent Laplace prior on each $w_j$ makes the MAP estimate the L1-regularised (lasso) solution, and
  express its $\lambda$ the same way.
- **Q7.** Define the Kullback–Leibler divergence $\mathrm{KL}(p\,\|\,q)$ between two densities on the same
  space, and prove that $\mathrm{KL}(p\,\|\,q) \ge 0$, with equality iff $p = q$ almost everywhere. Give a
  two-point (Bernoulli) example with $\mathrm{KL}(p\,\|\,q) \ne \mathrm{KL}(q\,\|\,p)$. Then consider fitting
  a single Gaussian $q$ to a distribution $p$ that is a two-component Gaussian mixture with well-separated
  modes: with a derivation, say what $q$ minimising the forward divergence $\mathrm{KL}(p\,\|\,q)$ looks like
  ("mode-covering"), and, qualitatively, what $q$ minimising the reverse divergence $\mathrm{KL}(q\,\|\,p)$
  looks like ("mode-seeking").
- **Q8.** You want to estimate a classifier's accuracy to within $\pm 1$ percentage point at 95% confidence,
  using the normal approximation to a Bernoulli proportion. How many i.i.d. test examples are needed in the
  worst case over the unknown true accuracy $p$? State the assumption needed for the normal approximation to
  be valid.

### Linear algebra and calculus

- **Q9.** Contrast the eigendecomposition and the singular value decomposition (SVD) of a real matrix
  $X \in \mathbb{R}^{m \times n}$: when does each exist? How do the singular values of $X$ relate to the
  eigenvalues of $X^\top X$? How does the rank of $X$ show up in its SVD? And, given a data matrix with rows
  as observations, how is principal component analysis (PCA) computed from the SVD of the centred data?
- **Q10.** Define positive semidefinite (PSD) for a real symmetric matrix. Prove that every covariance
  matrix is PSD. State and justify the relationship between a PSD Hessian and the convexity of a
  twice-differentiable function.
- **Q11.** Derive $\nabla_x (x^\top A x)$ for $x \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$;
  $\nabla_W \|XW - Y\|_F^2$, the gradient of the squared Frobenius norm (the sum of squared entries) of
  $XW - Y$, for $X \in \mathbb{R}^{m \times p}$, $W \in \mathbb{R}^{p \times k}$, $Y \in \mathbb{R}^{m \times
  k}$; and the Jacobian of the softmax function $s = \operatorname{softmax}(z)$ with respect to
  $z \in \mathbb{R}^K$.
- **Q12.** For $f(x) = \tfrac12 x^\top A x$ with $A \in \mathbb{R}^{n \times n}$ symmetric positive definite,
  with eigenvalues $0 < \lambda_1 \le \dots \le \lambda_n = \lambda_{\max}$, consider gradient descent
  $x_{t+1} = x_t - \eta \nabla f(x_t)$ with step size $\eta = 1/\lambda_{\max}$. Show that the error contracts
  geometrically along each eigenvector of $A$, with the $i$-th eigen-direction shrinking by a factor
  $1 - \lambda_i/\lambda_{\max}$ per step, so that the slowest-converging direction shrinks by $1 - 1/\kappa$,
  where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number. Explain why a large condition number
  slows convergence, and how feature normalisation or preconditioning helps.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Confirm the convention for $\log$ (natural, used throughout) and, where a question asks for a derivation,
whether a standard identity may be quoted or should be shown from first principles; every derivation below is
given in full regardless.

### Probability

**Q1.** $P(A \mid \text{red}) = 3/4$, and the second ball is red with probability $1/2$. By Bayes' theorem,
$P(\text{red} \mid A) = 3/4$ and $P(\text{red} \mid B) = 1/4$, so
$P(\text{red}) = \tfrac12 \cdot \tfrac34 + \tfrac12 \cdot \tfrac14 = \tfrac12$ and

$$P(A \mid \text{red}) = \frac{P(\text{red} \mid A)\,P(A)}{P(\text{red})} = \frac{\tfrac34 \cdot \tfrac12}{\tfrac12} = \frac34.$$

For the second draw, condition on the urn: with probability $3/4$ the urn is A, now holding 2 red and 1 blue
after the first (red) ball is removed, so the second ball is red with probability $2/3$; with probability
$1/4$ the urn is B, which after losing its only red ball holds 0 red and 3 blue, so the second ball is red
with probability $0$. Hence

$$P(\text{2nd red} \mid \text{1st red}) = \frac34 \cdot \frac23 + \frac14 \cdot 0 = \frac12.$$

**Q2.** $\mathbb{E}[\text{flips to HH}] = 6$ and $\mathbb{E}[\text{flips to HT}] = 4$. Track one bit of state:
$0$ if the last flip was not H (or none has been seen yet), $H$ if the last flip was H. For HH, from state
$0$ a flip is H with probability $1/2$ (move to $H$) or T with probability $1/2$ (stay at $0$), so
$E_0 = 1 + \tfrac12 E_H + \tfrac12 E_0$; from state $H$, a flip is H with probability $1/2$ (done) or T with
probability $1/2$ (back to $0$ — the earlier H is wasted), so $E_H = 1 + \tfrac12 E_0$. Solving gives
$E_H = 4$, $E_0 = 6$. For HT, the equation from state $0$ is the same,
$E_0 = 1 + \tfrac12 E_H + \tfrac12 E_0$; but from state $H$, a flip is T with probability $1/2$ (done) or H
with probability $1/2$ (stay at $H$ — this new H can still start a match), so $E_H = 1 + \tfrac12 E_H$, giving
$E_H = 2$, $E_0 = 4$. The two differ because HH overlaps itself (its one-letter prefix H is also its suffix):
failing a near-miss (T after H) discards the H entirely and the search restarts from scratch, whereas failing
a near-miss of HT (another H after H) discards nothing, since that new H is exactly as good a start as the
first one was. Overlapping patterns are, on average, slower to wait for.

**Q3.** $\mathbb{E}[\max] = n/(n+1)$ and $\mathbb{E}[\min] = 1/(n+1)$. The maximum has $P(\max \le x) = x^n$
for $x \in [0, 1]$ (every $X_i \le x$), hence density $n x^{n-1}$ and

$$\mathbb{E}[\max] = \int_0^1 x \cdot n x^{n-1}\,dx = \frac{n}{n+1}.$$

For the minimum, note that if $U \sim \mathrm{Uniform}(0,1)$ then so is $1 - U$, so
$1 - \min_i X_i = \max_i (1 - X_i)$ has the same distribution as $\max_i X_i$; taking expectations,
$1 - \mathbb{E}[\min] = n/(n+1)$, so $\mathbb{E}[\min] = 1/(n+1)$.

**Q4.** $P(\text{disease} \mid \text{positive}) = 1/6 \approx 16.7\%$. By the law of total probability,
$P(\text{positive}) = P(\text{pos} \mid D)P(D) + P(\text{pos} \mid \lnot D)P(\lnot D) = 0.99 \times 0.01 + 0.05
\times 0.99 = 0.0594$, and Bayes' theorem gives

$$P(D \mid \text{positive}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} = \frac16.$$

This is the base-rate effect: because the disease is rare, the healthy population (99% of everyone) is far
larger than the diseased population, so even a small false-positive rate (5%) on that huge group produces
more false positives (4.95% of the population) than there are true positives (0.99% of the population) —
most positives are false alarms.

### Statistics

**Q5.** $\hat\mu = \bar X = \frac1n \sum_i X_i$ and $\hat\sigma^2 = \frac1n \sum_i (X_i - \bar X)^2$, and the
latter is biased low by the factor $(n-1)/n$. The log-likelihood is
$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (X_i - \mu)^2$; setting
$\partial \ell/\partial \mu = 0$ gives $\sum_i (X_i - \mu) = 0$, i.e. $\hat\mu = \bar X$, and setting
$\partial \ell/\partial \sigma^2 = 0$ at $\mu = \hat\mu$ gives $\hat\sigma^2 = \frac1n \sum_i (X_i - \bar X)^2$.
For the bias, expand around the true mean:

$$\sum_i (X_i - \bar X)^2 = \sum_i \bigl[(X_i - \mu) - (\bar X - \mu)\bigr]^2 = \sum_i (X_i - \mu)^2 - n(\bar X - \mu)^2,$$

using $\sum_i (X_i - \mu) = n (\bar X - \mu)$. Taking expectations, $\mathbb{E}\sum_i (X_i - \mu)^2 = n\sigma^2$
and $\mathbb{E}[n(\bar X - \mu)^2] = n\,\mathrm{Var}(\bar X) = \sigma^2$, so
$\mathbb{E}\sum_i (X_i - \bar X)^2 = (n-1)\sigma^2$ and $\mathbb{E}[\hat\sigma^2] = \frac{n-1}{n}\sigma^2$. The
estimator spends one degree of freedom estimating $\mu$ with $\bar X$, which is on average slightly closer to
the sample than the true $\mu$ is, understating the spread; dividing by $n - 1$ instead of $n$ removes the
bias exactly.

**Q6.** $\lambda = \sigma^2/\tau^2$ for ridge; for the Laplace prior with scale $b$, the L1 penalty has
$\lambda = 2\sigma^2/b$. By Bayes' theorem the posterior satisfies $p(w \mid X, y) \propto p(y \mid X, w)
p(w)$, so the MAP estimate maximises $\log p(y \mid X, w) + \log p(w)$, equivalently minimises its negative.
The Gaussian likelihood gives $-\log p(y \mid X, w) = \frac{1}{2\sigma^2}\|Xw - y\|^2 + \text{const}$, and the
Gaussian prior gives $-\log p(w) = \frac{1}{2\tau^2}\|w\|^2 + \text{const}$, so the MAP objective is

$$\frac{1}{2\sigma^2}\|Xw - y\|^2 + \frac{1}{2\tau^2}\|w\|^2,$$

and multiplying by $2\sigma^2$ (which does not change the minimiser) gives $\|Xw - y\|^2 + \lambda \|w\|^2$
with $\lambda = \sigma^2/\tau^2$: exactly ridge regression. Replacing the prior with an independent
Laplace$(0, b)$ on each $w_j$, with density $\frac{1}{2b} e^{-|w_j|/b}$, gives
$-\log p(w) = \frac1b \|w\|_1 + \text{const}$, so the MAP objective becomes
$\frac{1}{2\sigma^2}\|Xw - y\|^2 + \frac1b \|w\|_1$; multiplying by $2\sigma^2$ gives
$\|Xw - y\|^2 + \lambda \|w\|_1$ with $\lambda = 2\sigma^2/b$ — the lasso. A Gaussian prior penalises large
weights quadratically and shrinks every coordinate smoothly toward $0$; a Laplace prior has a sharp peak at
$0$ (its penalty is not differentiable there), and the subgradient at that kink can hold a coordinate exactly
at $0$, which is why the lasso produces sparse solutions and ridge, generically, does not.

**Q7.** $\mathrm{KL}(p\,\|\,q) = \mathbb{E}_p\bigl[\log \frac{p(X)}{q(X)}\bigr] = \int p(x) \log
\frac{p(x)}{q(x)}\,dx \ge 0$. For non-negativity, apply Jensen's inequality to the convex function $-\log$
with $Y = q(X)/p(X)$ under $X \sim p$: $\mathbb{E}_p[Y] = \int p(x) \frac{q(x)}{p(x)}\,dx = \int q(x)\,dx = 1$
(over the support of $p$), so

$$\mathrm{KL}(p\,\|\,q) = \mathbb{E}_p[-\log Y] \ge -\log \mathbb{E}_p[Y] = -\log 1 = 0,$$

with equality iff $Y$ is almost surely constant (since $-\log$ is strictly convex), i.e. $q = p$ almost
everywhere. Asymmetry: for $p = \mathrm{Bernoulli}(0.9)$, $q = \mathrm{Bernoulli}(0.5)$,

$$\mathrm{KL}(p\,\|\,q) = 0.9 \log\frac{0.9}{0.5} + 0.1 \log\frac{0.1}{0.5} \approx 0.368, \qquad \mathrm{KL}(q\,\|\,p) = 0.5 \log\frac{0.5}{0.9} + 0.5 \log\frac{0.5}{0.1} \approx 0.511,$$

which are not equal. For the Gaussian fit, write $q = \mathcal{N}(\mu, \sigma^2)$; minimising
$\mathrm{KL}(p\,\|\,q) = -H(p) + H(p, q)$ over $q$ is the same as minimising the cross-entropy
$H(p, q) = \mathbb{E}_p[-\log q(X)] = \frac12 \log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(X - \mu)^2]}{2\sigma^2}$.
This is minimised over $\mu$ at $\mu = \mathbb{E}_p[X]$ regardless of $\sigma^2$, and then over $\sigma^2$ at
$\sigma^2 = \mathrm{Var}_p(X)$: forward KL is minimised by *moment matching*. Since $p$ is symmetric with
modes at $\pm 3$ and unit variance, $\mathbb{E}_p[X] = 0$ and $\mathrm{Var}_p(X) = \mathbb{E}_p[X^2] = 10$
(each component has $\mathrm{Var} + \text{mean}^2 = 1 + 9 = 10$), so the forward-KL fit is
$\mathcal{N}(0, 10)$: wide and centred between the modes, in order to keep $q$ positive wherever $p$ is (else
$p \log(p/q) \to \infty$). The reverse divergence $\mathrm{KL}(q\,\|\,p) = \mathbb{E}_q[\log q(X) - \log
p(X)]$ instead penalises $q$ for putting mass where $p$ is small, so it has no incentive to cover both modes;
started near one mode, gradient-based fitting finds a $q$ tightly wrapped around that single mode, close to
its own mean and variance, rather than one covering both.

**Q8.** $n \approx 9{,}604$ in the worst case. For $n$ i.i.d. Bernoulli$(p)$ trials, the sample proportion
$\hat p$ has $\mathrm{Var}(\hat p) = p(1-p)/n$ and, by the central limit theorem, is approximately
$\mathcal{N}(p,\, p(1-p)/n)$ once $n$ is large enough. A two-sided 95% interval has half-width
$z_{0.975}\sqrt{p(1-p)/n}$ with $z_{0.975} = 1.96$; setting this equal to the target margin $E = 0.01$ and
solving for $n$ gives

$$n = \frac{z_{0.975}^2\, p(1-p)}{E^2}.$$

Since $p(1-p) \le \frac14$ for every $p \in [0, 1]$, with equality at $p = 1/2$, the worst case is
$n = 1.96^2 \times 0.25 / 0.01^2 = 9{,}604$. This requires $n$ to be large enough that the normal
approximation to the Binomial is accurate — in particular, that the examples are drawn i.i.d. with a fixed
true accuracy $p$ (no distribution shift across the test set) and that $np$ and $n(1-p)$ are both comfortably
above about 5–10, which 9,604 satisfies for any $p$ not extremely close to 0 or 1.

### Linear algebra and calculus

**Q9.** The SVD $X = U\Sigma V^\top$ (with $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$
orthogonal, and $\Sigma$ rectangular-diagonal with entries $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$) exists for
every real matrix of every shape, with no assumptions on $X$. An eigendecomposition $A = Q\Lambda Q^{-1}$
requires a square matrix, is guaranteed to have real $\Lambda$ and orthogonal $Q$ only when $A$ is symmetric
(the spectral theorem), and can fail to exist at all for a general square matrix: the $2 \times 2$ matrix with
a single $1$ in the top-right corner and zeros elsewhere has the repeated eigenvalue $0$ (algebraic
multiplicity 2) but only a one-dimensional space of eigenvectors, so no invertible $Q$ diagonalises it — a
*defective* matrix. Since $X^\top X = V \Sigma^\top U^\top U \Sigma V^\top = V(\Sigma^\top \Sigma) V^\top$ and
$\Sigma^\top \Sigma = \mathrm{diag}(\sigma_i^2)$, this is exactly an eigendecomposition of the symmetric PSD
matrix $X^\top X$: its eigenvalues are $\sigma_i^2$ and its eigenvectors are the columns of $V$, so
$\sigma_i = \sqrt{\lambda_i(X^\top X)}$. Because $U, V$ are invertible, $\mathrm{rank}(X) = \mathrm{rank}
(\Sigma)$, i.e. the number of nonzero singular values. For PCA, centre the data $X_c = X - \bar X$ (the
row-wise mean removed) and take its SVD $X_c = U\Sigma V^\top$; then $X_c^\top X_c = V \Sigma^2 V^\top$ is
$(n-1)$ times the sample covariance matrix, so the principal directions are the columns of $V$, the explained
variance along direction $i$ is $\sigma_i^2/(n-1)$, and the principal-component scores (the data projected
onto those directions) are $X_c V = U\Sigma$.

**Q10.** A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is PSD if $x^\top A x \ge 0$ for every
$x \in \mathbb{R}^n$ (equivalently, every eigenvalue is $\ge 0$). For a random vector $X$ with mean $\mu$ and
covariance $\Sigma = \mathbb{E}[(X - \mu)(X - \mu)^\top]$, and any $a \in \mathbb{R}^n$,

$$a^\top \Sigma a = \mathbb{E}\bigl[a^\top (X-\mu)(X-\mu)^\top a\bigr] = \mathbb{E}\bigl[(a^\top(X-\mu))^2\bigr] \ge 0,$$

since this is the expectation of a squared scalar random variable; the same argument applies verbatim to the
sample covariance, with $\mathbb{E}$ replaced by an average over the sample. So $\Sigma$ is always PSD. A
twice-differentiable $f$ is convex on a convex domain if and only if $\nabla^2 f(x)$ is PSD at every point $x$
in it. One direction: if $f$ is convex, then for any direction $v$ the function $g(t) = f(x + tv)$ is convex
(a convex function composed with an affine map), so $g''(0) = v^\top \nabla^2 f(x) v \ge 0$ for every $v$,
i.e. $\nabla^2 f(x)$ is PSD. Conversely, if $\nabla^2 f$ is PSD everywhere, Taylor's theorem with the Lagrange
remainder gives, for any $x, y$, $f(y) = f(x) + \nabla f(x)^\top (y - x) + \frac12 (y-x)^\top \nabla^2 f(\xi)
(y - x)$ for some $\xi$ on the segment between them; the last term is $\ge 0$ by assumption, so
$f(y) \ge f(x) + \nabla f(x)^\top (y - x)$ for all $x, y$, which is exactly the first-order characterisation
of convexity.

**Q11.** $\nabla_x (x^\top A x) = (A + A^\top) x$; $\nabla_W \|XW - Y\|_F^2 = 2X^\top(XW - Y)$; the softmax
Jacobian is $\operatorname{diag}(s) - s s^\top$. Writing $x^\top A x = \sum_{i,j} A_{ij} x_i x_j$ and
differentiating with respect to $x_k$: $\partial_k (x^\top A x) = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i =
(Ax)_k + (A^\top x)_k$, so $\nabla_x(x^\top A x) = (A + A^\top)x$ (equal to $2Ax$ when $A$ is symmetric). For
the second, write $R = XW - Y$ and use differentials:
$d\|R\|_F^2 = d\,\mathrm{tr}(R^\top R) = 2\,\mathrm{tr}(R^\top dR) = 2\,\mathrm{tr}(R^\top X\,dW) = 2\,
\mathrm{tr}\bigl((X^\top R)^\top dW\bigr)$, and since $d\|R\|_F^2 = \mathrm{tr}(G^\top dW)$ defines the
gradient $G$, reading off gives $G = 2X^\top R = 2X^\top(XW - Y)$. For the softmax, $s_i = e^{z_i}/Z$ with
$Z = \sum_k e^{z_k}$; by the quotient rule, $\partial s_i/\partial z_j = \delta_{ij} e^{z_i}/Z - e^{z_i}
e^{z_j}/Z^2 = \delta_{ij} s_i - s_i s_j = s_i(\delta_{ij} - s_j)$, which in matrix form is
$J = \operatorname{diag}(s) - s s^\top$ — symmetric, and with rows summing to $0$ (perturbing every logit by
the same amount leaves softmax unchanged).

**Q12.** The minimiser is $x^\star = 0$, so the error is $x_t$ itself. $\nabla f(x) = Ax$, so
$x_{t+1} = (I - \eta A) x_t$ with $\eta = 1/\lambda_{\max}$. Diagonalise $A = Q\Lambda Q^\top$ and write
$y_t = Q^\top x_t$ for the coordinates of $x_t$ in the eigenbasis; then $y_{t+1} = (I - \eta\Lambda) y_t$ is
diagonal, so coordinate by coordinate,

$$y_{t+1,i} = \Bigl(1 - \frac{\lambda_i}{\lambda_{\max}}\Bigr) y_{t,i} \quad\Longrightarrow\quad y_{t,i} = \Bigl(1 - \frac{\lambda_i}{\lambda_{\max}}\Bigr)^{t} y_{0,i}.$$

Every coordinate contracts geometrically, at a rate that depends only on its own eigenvalue: the direction of
$\lambda_{\max}$ has $1 - \lambda_{\max}/\lambda_{\max} = 0$ and reaches exactly $0$ after a single step,
while the direction of $\lambda_{\min}$ has the largest (slowest) contraction factor
$1 - \lambda_{\min}/\lambda_{\max} = 1 - 1/\kappa$. Because the error norm $\|x_t\|$ is a sum of these
per-direction terms, once the faster directions have died away it is dominated by the $\lambda_{\min}$ term,
so $\|x_t\| / \|x_{t-1}\| \to 1 - 1/\kappa$: the overall rate is set by the *worst-conditioned* direction,
however fast the others converge. When $\kappa \gg 1$ (ill-conditioning — some directions of $f$ are much
flatter than others), $1 - 1/\kappa$ is close to $1$ and convergence along the flat directions is extremely
slow, even though the step size is tuned optimally for the steepest direction; a smaller step would only slow
the flat directions further, since $\eta = 1/\lambda_{\max}$ is already the largest step that does not
diverge along the steepest direction. Feature normalisation (rescaling inputs to comparable variance) and
preconditioning (replacing $A$ with $P^{-1}A$ for some $P \approx A$, e.g. a diagonal or block approximation
of the Hessian) both act by shrinking the ratio between the largest and smallest curvature, pushing $\kappa$
toward $1$ so that every direction contracts at a similar, fast rate.

<details>
<summary>Checks (runnable)</summary>

```python
import math
from fractions import Fraction

import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize
from sklearn.linear_model import Lasso as SkLasso

rng = np.random.default_rng(2026)

# ---- Q1: urns, Bayes' theorem, exact fractions and Monte Carlo
p_red_A, p_red_B = Fraction(3, 4), Fraction(1, 4)          # urn A: 3 red / 4; urn B: 1 red / 4
p_A_post = p_red_A * Fraction(1, 2) / (p_red_A * Fraction(1, 2) + p_red_B * Fraction(1, 2))
assert p_A_post == Fraction(3, 4)
p_second_red = p_A_post * Fraction(2, 3) + (1 - p_A_post) * Fraction(0, 3)
assert p_second_red == Fraction(1, 2)

N = 2_000_000
urn_is_A = rng.random(N) < 0.5
counts = np.where(urn_is_A[:, None], np.array([3, 1]), np.array([1, 3]))   # [red, blue] remaining
first_red = rng.random(N) < (counts[:, 0] / 4)
remaining = counts.copy()
remaining[first_red, 0] -= 1                                # NOTE: the first ball is not returned
remaining[~first_red, 1] -= 1
assert math.isclose(urn_is_A[first_red].mean(), 0.75, abs_tol=0.01)
second_red = rng.random(N) < (remaining[:, 0] / 3)
assert math.isclose(second_red[first_red].mean(), 0.5, abs_tol=0.01)

# ---- Q2: expected flips to first HH, first HT, by solving the Markov-chain equations and by simulation
# state 0: last flip was not H (or none yet); state H: last flip was H
A_hh = np.array([[0.5, -0.5], [-0.5, 1.0]])                 # E0 - EH = 2 ; -0.5 E0 + EH = 1
E0_hh, EH_hh = np.linalg.solve(A_hh, np.array([1.0, 1.0]))
assert math.isclose(E0_hh, 6.0, rel_tol=1e-9) and math.isclose(EH_hh, 4.0, rel_tol=1e-9)
A_ht = np.array([[0.5, -0.5], [0.0, 0.5]])                  # E0 - EH = 2 ; 0.5 EH = 1
E0_ht, EH_ht = np.linalg.solve(A_ht, np.array([1.0, 1.0]))
assert math.isclose(E0_ht, 4.0, rel_tol=1e-9) and math.isclose(EH_ht, 2.0, rel_tol=1e-9)

L, reps = 60, 300_000
flips = rng.integers(0, 2, size=(reps, L))                  # 1 = H, 0 = T
is_hh = (flips[:, :-1] == 1) & (flips[:, 1:] == 1)
is_ht = (flips[:, :-1] == 1) & (flips[:, 1:] == 0)
assert is_hh.any(axis=1).all() and is_ht.any(axis=1).all()  # NOTE: argmax silently returns 0 with no True value,
wait_hh, wait_ht = is_hh.argmax(axis=1) + 2, is_ht.argmax(axis=1) + 2   #      so a match must be guaranteed first
assert math.isclose(wait_hh.mean(), 6.0, abs_tol=0.05)
assert math.isclose(wait_ht.mean(), 4.0, abs_tol=0.05)

# ---- Q3: E[max] and E[min] of n iid Uniform(0,1), for several n, by quadrature and by simulation
for n3 in (1, 2, 5, 10, 30):
    e_max, e_min = n3 / (n3 + 1), 1 / (n3 + 1)
    assert math.isclose(integrate.quad(lambda x: x * n3 * x ** (n3 - 1), 0, 1)[0], e_max, rel_tol=1e-8)
    assert math.isclose(integrate.quad(lambda x: x * n3 * (1 - x) ** (n3 - 1), 0, 1)[0], e_min, rel_tol=1e-8)
    u3 = rng.random((300_000, n3))
    assert math.isclose(u3.max(axis=1).mean(), e_max, abs_tol=0.01)
    assert math.isclose(u3.min(axis=1).mean(), e_min, abs_tol=0.01)

# ---- Q4: diagnostic test, base rate, exact fractions and Monte Carlo
sens, spec, prev = Fraction(99, 100), Fraction(95, 100), Fraction(1, 100)
true_pos_rate = sens * prev                    # 0.99 x 0.01 = 0.0099, i.e. 0.99% of the whole population
false_pos_rate = (1 - spec) * (1 - prev)        # 0.05 x 0.99 = 0.0495, i.e. 4.95% of the whole population
assert true_pos_rate == Fraction(99, 10000) and false_pos_rate == Fraction(495, 10000)
p_pos = true_pos_rate + false_pos_rate
assert p_pos == Fraction(594, 10000)
p_disease_given_pos = true_pos_rate / p_pos
assert p_disease_given_pos == Fraction(1, 6)

disease = rng.random(3_000_000) < float(prev)
pos = np.where(disease, rng.random(3_000_000) < float(sens), rng.random(3_000_000) < (1 - float(spec)))
assert math.isclose(disease[pos].mean(), 1 / 6, abs_tol=0.01)

# ---- Q5: Gaussian MLE, biased variance, for several n, by simulation
mu_true, sigma_true = 2.0, 3.0
for n5 in (3, 6, 15):
    X5 = rng.normal(mu_true, sigma_true, size=(300_000, n5))
    xbar5 = X5.mean(axis=1)
    sigma2_hat = ((X5 - xbar5[:, None]) ** 2).mean(axis=1)   # NOTE: MLE divides by n, not n - 1
    assert math.isclose(xbar5.mean(), mu_true, abs_tol=0.02)
    assert math.isclose(sigma2_hat.mean(), (n5 - 1) / n5 * sigma_true ** 2, rel_tol=0.02)
    assert math.isclose(sigma2_hat.mean() * n5 / (n5 - 1), sigma_true ** 2, rel_tol=0.02)   # the n/(n-1) correction

# ---- Q6: ridge and lasso as MAP estimates
n6, p6 = 200, 5
X6 = rng.standard_normal((n6, p6))
sigma2, tau2 = 0.5, 2.0
lam_ridge = sigma2 / tau2
y6 = X6 @ np.array([1.0, -2.0, 0.5, 3.0, -1.5]) + rng.normal(scale=math.sqrt(sigma2), size=n6)
w_ridge_closed = np.linalg.solve(X6.T @ X6 + lam_ridge * np.eye(p6), X6.T @ y6)


def neg_log_posterior_gaussian(w):
    return ((X6 @ w - y6) @ (X6 @ w - y6)) / (2 * sigma2) + (w @ w) / (2 * tau2)


def grad_gaussian(w):
    return X6.T @ (X6 @ w - y6) / sigma2 + w / tau2


map_result = minimize(neg_log_posterior_gaussian, np.zeros(p6), jac=grad_gaussian, method="BFGS", tol=1e-12)
assert np.allclose(w_ridge_closed, map_result.x, atol=1e-5)

b_laplace = 0.3
lam_l1 = 2 * sigma2 / b_laplace                              # Laplace(0, b) prior -> L1 penalty lambda = 2 sigma^2 / b


def lasso_ista(X, y, lam, n_iter=4000):
    lipschitz = 2 * np.linalg.eigvalsh(X.T @ X).max()
    step = 1.0 / lipschitz
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        w = w - step * 2 * X.T @ (X @ w - y)
        w = np.sign(w) * np.maximum(np.abs(w) - step * lam, 0.0)   # proximal step: soft-thresholding
    return w


w_ista = lasso_ista(X6, y6, lam_l1)
# NOTE: sklearn's Lasso objective is (1/(2n)) ||y - Xw||^2 + alpha ||w||_1, so alpha = lambda / (2n), not lambda
w_sklearn = SkLasso(alpha=lam_l1 / (2 * n6), fit_intercept=False, tol=1e-14, max_iter=200_000).fit(X6, y6).coef_
assert np.allclose(w_ista, w_sklearn, atol=1e-3)

# ---- Q7: KL divergence: non-negativity, asymmetry, forward vs reverse fit to a bimodal mixture
def kl_discrete(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


for _ in range(2000):                                        # non-negativity on random categorical distributions
    k = rng.integers(2, 8)
    a, b = rng.random(k) + 1e-3, rng.random(k) + 1e-3
    assert kl_discrete(a / a.sum(), b / b.sum()) >= -1e-10

a2, b2 = 0.9, 0.5                                             # asymmetric two-point (Bernoulli) example
p2, q2 = np.array([a2, 1 - a2]), np.array([b2, 1 - b2])
kl_pq, kl_qp = kl_discrete(p2, q2), kl_discrete(q2, p2)
assert math.isclose(kl_pq, a2 * math.log(a2 / b2) + (1 - a2) * math.log((1 - a2) / (1 - b2)), rel_tol=1e-9)
assert math.isclose(kl_pq, 0.368, abs_tol=5e-4) and math.isclose(kl_qp, 0.511, abs_tol=5e-4)
assert kl_pq < kl_qp and not math.isclose(kl_pq, kl_qp, rel_tol=0.05)

M1, M2, SD = -3.0, 3.0, 1.0                                   # two well-separated unit-variance modes


def mixture_pdf(x):
    return 0.5 * stats.norm.pdf(x, M1, SD) + 0.5 * stats.norm.pdf(x, M2, SD)


grid = np.linspace(-20.0, 20.0, 8001)
dx = grid[1] - grid[0]
p_grid = mixture_pdf(grid)


def forward_kl_obj(params):                                  # KL(p || q), q = N(mu, exp(log_sigma)^2)
    mu, log_sigma = params
    q = np.clip(stats.norm.pdf(grid, mu, math.exp(log_sigma)), 1e-300, None)
    return np.sum(p_grid * (np.log(np.clip(p_grid, 1e-300, None)) - np.log(q))) * dx


def reverse_kl_obj(params):                                   # KL(q || p)
    mu, log_sigma = params
    q = stats.norm.pdf(grid, mu, math.exp(log_sigma))
    qc, pc = np.clip(q, 1e-300, None), np.clip(p_grid, 1e-300, None)
    return np.sum(q * (np.log(qc) - np.log(pc))) * dx


fwd = minimize(forward_kl_obj, x0=[1.0, math.log(2.0)], method="BFGS", tol=1e-12)
mu_fwd, sigma_fwd = fwd.x[0], math.exp(fwd.x[1])
assert math.isclose(mu_fwd, 0.0, abs_tol=0.02) and math.isclose(sigma_fwd, math.sqrt(10.0), rel_tol=0.01)

rev = minimize(reverse_kl_obj, x0=[2.5, math.log(1.2)], method="BFGS", tol=1e-12)   # NOTE: started near one mode;
mu_rev, sigma_rev = rev.x[0], math.exp(rev.x[1])                                    #  reverse KL is multimodal
assert math.isclose(mu_rev, M2, abs_tol=0.05) and math.isclose(sigma_rev, SD, abs_tol=0.05)
assert sigma_rev < sigma_fwd / 2

# ---- Q8: sample size for a +-1pp margin at 95% confidence
z95 = stats.norm.ppf(0.975)
assert math.isclose(z95, 1.96, abs_tol=5e-4)
n_needed = math.ceil(z95 ** 2 * 0.25 / 0.01 ** 2)
assert n_needed == 9604
p_grid8 = np.linspace(0.0, 1.0, 100_001)
assert np.all(p_grid8 * (1 - p_grid8) <= 0.25 + 1e-12)         # p(1 - p) <= 1/4, the worst case used above
phat = rng.binomial(n_needed, 0.5, size=20_000) / n_needed
assert math.isclose(np.mean(np.abs(phat - 0.5) <= 0.01), 0.95, abs_tol=0.02)   # the promised coverage, empirically

# ---- Q9: SVD vs eigendecomposition of X^T X, rank, and PCA from the SVD of centred data
X9 = rng.standard_normal((7, 5))
U9, S9, Vt9 = np.linalg.svd(X9, full_matrices=False)           # NOTE: this returns V^T, not V
eigvals9, eigvecs9 = np.linalg.eigh(X9.T @ X9)                 # ascending order
assert np.allclose(S9 ** 2, eigvals9[::-1], atol=1e-8)
signs9 = np.sign(np.sum(Vt9.T * eigvecs9[:, ::-1], axis=0))
assert np.allclose(Vt9.T, eigvecs9[:, ::-1] * signs9, atol=1e-6)

r_true = 3
A_low = rng.standard_normal((8, r_true)) @ rng.standard_normal((r_true, 6))   # an exactly rank-3 matrix
s_low = np.linalg.svd(A_low, compute_uv=False)
assert int((s_low > s_low.max() * 1e-10).sum()) == r_true == np.linalg.matrix_rank(A_low)

n9p, d9p = 300, 4
raw9 = rng.standard_normal((n9p, d9p)) @ rng.standard_normal((d9p, d9p))     # correlated columns
Xc = raw9 - raw9.mean(axis=0, keepdims=True)
Up, Sp, Vtp = np.linalg.svd(Xc, full_matrices=False)
eigvals_c, eigvecs_c = np.linalg.eigh((Xc.T @ Xc) / (n9p - 1))
assert np.allclose(Sp ** 2 / (n9p - 1), eigvals_c[::-1], atol=1e-8)          # explained variance
signs_p = np.sign(np.sum(Vtp.T * eigvecs_c[:, ::-1], axis=0))
assert np.allclose(Xc @ Vtp.T, Up * Sp, atol=1e-8)                           # scores: X_c V = U Sigma
assert np.allclose(Up * Sp, Xc @ (eigvecs_c[:, ::-1] * signs_p), atol=1e-6)  # matches the covariance eigenbasis

# ---- Q10: covariance is PSD; PSD Hessian is equivalent to convexity
data10 = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))
centred10 = data10 - data10.mean(axis=0)
cov10 = (centred10.T @ centred10) / (len(data10) - 1)
assert np.linalg.eigvalsh(cov10).min() >= -1e-8                # NOTE: eigvalsh, not eig -- cov10 is symmetric,
for _ in range(200):                                            #  eig() can return spurious tiny imaginary parts
    a = rng.standard_normal(6)
    assert math.isclose(a @ cov10 @ a, np.mean((centred10 @ a) ** 2) * len(data10) / (len(data10) - 1), rel_tol=1e-9)


def quad_f(A, x):
    return 0.5 * x @ A @ x


Qo, _ = np.linalg.qr(rng.standard_normal((5, 5)))
A_psd = Qo @ np.diag([0.1, 1.0, 2.0, 3.0, 4.0]) @ Qo.T           # PSD: every eigenvalue positive
A_indef = Qo @ np.diag([-0.5, 1.0, 2.0, 3.0, 4.0]) @ Qo.T        # indefinite: one negative eigenvalue
for _ in range(200):                                             # exact identity: the midpoint gap is
    x, y = rng.standard_normal(5), rng.standard_normal(5)        # (1/8)(x - y)^T A (x - y)
    gap = 0.5 * (quad_f(A_psd, x) + quad_f(A_psd, y)) - quad_f(A_psd, (x + y) / 2)
    assert gap >= -1e-9 and math.isclose(gap, 0.125 * (x - y) @ A_psd @ (x - y), rel_tol=1e-9)
neg_dir = Qo[:, 0]                                                # x - y along the negative eigenvector
x0, y0 = rng.standard_normal(5), rng.standard_normal(5)
y0 = x0 - 2.0 * neg_dir
assert 0.5 * (quad_f(A_indef, x0) + quad_f(A_indef, y0)) - quad_f(A_indef, (x0 + y0) / 2) < 0

# ---- Q11: matrix-calculus identities, checked by finite differences
h = 1e-5
A11, x11 = rng.standard_normal((4, 4)), rng.standard_normal(4)
analytic11 = (A11 + A11.T) @ x11
fd11 = np.array([((x11 + h * e) @ A11 @ (x11 + h * e) - (x11 - h * e) @ A11 @ (x11 - h * e)) / (2 * h)
                  for e in np.eye(4)])
assert np.allclose(analytic11, fd11, atol=1e-4)

X11, W11, Y11 = rng.standard_normal((6, 3)), rng.standard_normal((3, 2)), rng.standard_normal((6, 2))
analytic_G = 2 * X11.T @ (X11 @ W11 - Y11)
fd_G = np.zeros_like(W11)
for i in range(3):
    for j in range(2):
        step = np.zeros_like(W11)
        step[i, j] = h
        fd_G[i, j] = (np.linalg.norm(X11 @ (W11 + step) - Y11, "fro") ** 2
                      - np.linalg.norm(X11 @ (W11 - step) - Y11, "fro") ** 2) / (2 * h)
assert np.allclose(analytic_G, fd_G, atol=1e-4)


def softmax11(z):
    e = np.exp(z - z.max())
    return e / e.sum()


z11 = rng.standard_normal(4)
s11 = softmax11(z11)
analytic_J = np.diag(s11) - np.outer(s11, s11)
fd_J = np.column_stack([(softmax11(z11 + h * e) - softmax11(z11 - h * e)) / (2 * h) for e in np.eye(4)])
assert np.allclose(analytic_J, fd_J, atol=1e-5)
assert np.allclose(analytic_J @ np.ones(4), 0.0, atol=1e-10)     # rows sum to 0: shifting every logit changes nothing

# ---- Q12: gradient descent on a quadratic, contraction along each eigendirection of A
Q12o, _ = np.linalg.qr(rng.standard_normal((4, 4)))
lambdas = np.array([1.0, 3.0, 8.0, 25.0])                        # lambda_max = 25 -> kappa = 25
A12 = Q12o @ np.diag(lambdas) @ Q12o.T
eta12 = 1.0 / lambdas.max()
x = rng.standard_normal(4)
coords = [Q12o.T @ x]
for _ in range(200):
    x = x - eta12 * (A12 @ x)
    coords.append(Q12o.T @ x)
Y12 = np.array(coords)                                            # (T + 1, 4): coordinates in the eigenbasis
predicted = 1 - lambdas / lambdas.max()
for t in range(len(Y12) - 1):
    ratio = np.divide(Y12[t + 1], Y12[t], out=np.full(4, np.nan), where=np.abs(Y12[t]) > 1e-12)
    ok = ~np.isnan(ratio)
    assert np.allclose(ratio[ok], predicted[ok], atol=1e-8)
assert math.isclose(Y12[1, np.argmax(lambdas)], 0.0, abs_tol=1e-10)   # top eigen-direction: exactly 0 after 1 step
late_ratio = np.linalg.norm(Y12[-1]) / np.linalg.norm(Y12[-2])
assert math.isclose(late_ratio, 1 - lambdas.min() / lambdas.max(), rel_tol=1e-6)   # dominated by 1 - 1/kappa

print("all checks passed")
```

</details>

</details>
