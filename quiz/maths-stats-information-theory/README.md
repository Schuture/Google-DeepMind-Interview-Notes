# Maths Quiz: Matrix Computation, Statistical Tests and Information Theory

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with derivations | ★★★☆☆ | Medium | RS · RE · MLE · Intern | matrix-multiplication, rank, matrix-inverse, pseudo-inverse, moments, central-limit-theorem, hypothesis-testing, chi-square-test, entropy, mutual-information, integration | 12 questions / 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

Throughout, $\log$ is the natural logarithm, and $\log_2$ is written explicitly wherever a quantity is in
bits. $\|\cdot\|$ is the Euclidean norm of a vector and $\|\cdot\|_F$ the Frobenius norm of a matrix (the
square root of the sum of its squared entries). For $A \in \mathbb{R}^{m \times n}$, its singular value
decomposition is $A = U\Sigma V^\top$ for orthogonal
$U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ and rectangular-diagonal $\Sigma$ with entries
$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(m,n)} \ge 0$, the square roots of the eigenvalues of
$A^\top A$. $I_n$ is the $n \times n$ identity matrix. $\mathrm{KL}(p \,\|\, q)$ is the Kullback–Leibler
divergence between two distributions on the same space; its definition, non-negativity and asymmetry are
given in [Maths Quiz: Probability, Statistics and Linear Algebra](../maths-probability-linear-algebra/README.md)
and are used here without re-derivation.

### Linear algebra

- **Q1.** The schoolbook algorithm multiplies $A \in \mathbb{R}^{m \times n}$ by $B \in \mathbb{R}^{n \times
  p}$ by computing every entry of the $m \times p$ product as a sum of $n$ products. State the number of
  scalar multiply–add operations this takes, and hence the time complexity of multiplying two square
  $n \times n$ matrices, both with this direct algorithm and with Strassen's algorithm. Then, for $A$ of
  shape $1000 \times 10$, $B$ of shape $10 \times 1000$ and $v \in \mathbb{R}^{1000}$, compute the number of
  multiply–adds needed to evaluate $(AB)v$ and $A(Bv)$, each in the order its brackets indicate. Explain what
  this determines about (a) the FLOP cost of the forward pass of a linear layer applied to a batch, and (b) how
  a low-rank update should be applied.
- **Q2.** Define the rank of a matrix and what it means for a square matrix to have full rank. State how
  $\mathrm{rank}(AB)$ relates to $\mathrm{rank}(A)$ and $\mathrm{rank}(B)$, and show that a nonzero outer
  product $uv^\top$ (for nonzero $u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$) has rank 1. List conditions on a
  square matrix $A \in \mathbb{R}^{n \times n}$ that are each equivalent to $A$ being invertible. Explain why
  testing $\det(A) \ne 0$ is a poor numerical test for invertibility — illustrate with $A = 0.1\, I_{100}$,
  whose determinant is $10^{-100}$ although $A$ is perfectly conditioned — and give a test that does not have
  this problem. Say how this bears on (a) when $X^\top X$ is invertible for a data matrix $X$, and (b) the
  rank of a LoRA update $BA$.
- **Q3.** Describe a numerical method for solving a general, square, invertible linear system $Ax = b$, and
  its time complexity. Give the approximate flop count of this method, and compare it with the flop count of
  instead forming $A^{-1}$ explicitly and computing $A^{-1}b$. Explain why the first way is preferred even
  when several right-hand sides $b$ must be solved for the same $A$ — in flop count and in numerical
  accuracy — and describe the further saving available when $A$ is symmetric positive definite. Say what
  this means for (a) solving the normal equations of linear regression, (b) computing a Gaussian-process
  posterior, and (c) taking a Newton step $Hd = -g$.
- **Q4.** Define the Moore–Penrose pseudo-inverse $A^+$ of $A \in \mathbb{R}^{m \times n}$ from its singular
  value decomposition. Show that $x = A^+b$ is the solution of $\min_x \|Ax - b\|$ of smallest Euclidean norm,
  among all $x$ attaining that minimum. Give the closed form of $A^+$ when $A$ has full column rank, and
  express $A^+$ as a limit of ridge-regularised normal equations as the ridge parameter goes to zero. Say how
  this bears on (a) linear regression with more parameters than examples, and (b) what gradient descent
  started from zero converges to on a least-squares problem with more parameters than examples.

### Probability and statistics

- **Q5.** For a random variable $X$ with mean $\mu$, the $k$-th raw moment is $E[X^k]$ and the $k$-th central
  moment is $E[(X - \mu)^k]$; variance is the 2nd central moment, skewness is the 3rd central moment divided
  by $\sigma^3$, and kurtosis is the 4th central moment divided by $\sigma^4$ (excess kurtosis subtracts 3,
  the value for a Gaussian). Define the moment-generating function $M_X(t) = E[e^{tX}]$ and state how the raw
  moments are recovered from its derivatives at $t = 0$. Give an example of a distribution with no mean, and
  one whose moments exist only up to some finite order. For $X \sim \mathrm{Exp}(\lambda)$, with density
  $\lambda e^{-\lambda x}$ for $x \ge 0$, derive $E[X^k]$ for a positive integer $k$, and give its skewness and
  excess kurtosis. Say how this bears on (a) what the moment estimates inside Adam track, and (b) what
  normalisation layers compute.
- **Q6.** State the weak law of large numbers and the central limit theorem for i.i.d. random variables,
  together with the moment assumptions each one needs. Give three ways the central limit theorem can fail to
  give a useful approximation at the sample size available: (a) a distribution whose variance (or mean) is
  infinite; (b) samples that are identically distributed but not independent; (c) a finite-variance but
  strongly skewed distribution at a moderate sample size, where the Berry–Esseen theorem bounds the error of
  the normal approximation by $C\rho/(\sigma^3\sqrt n)$ for an absolute constant $C$, with
  $\rho = E|X - \mu|^3$. For each of the three, give a concrete example and say, qualitatively, what goes
  wrong. Say how this bears on (a) the scaling of minibatch gradient noise with the batch size, and (b) the
  reliability of a normal-approximation confidence interval for a test accuracy near 0% or 100%.
- **Q7.** Define, for a statistical hypothesis test: the null and alternative hypotheses, the test statistic,
  the p-value, the significance level, and the power; state precisely what probability a p-value is (and is
  not) computing. Two models are evaluated on the same $n$-example test set. Compare (i) a paired test that
  uses, for each example, whether the two models agree or disagree (for instance McNemar's test on the
  discordant examples, or a paired bootstrap) against (ii) a test that treats the two accuracy estimates as
  independent samples; say which is more powerful when the two models' correctness is positively correlated
  across examples, and derive the comparison of variances that explains why. Suppose 20 independent
  hypothesis tests are run, each at significance level $\alpha = 0.05$, and every null hypothesis is in fact
  true: compute the probability that at least one test (falsely) rejects its null. Describe the Bonferroni
  correction and the Benjamini–Hochberg procedure, and state what each one controls.
- **Q8.** A fair-die hypothesis is tested by rolling a six-sided die 600 times and recording, for faces
  $1, \dots, 6$ in order, the counts $[90, 110, 95, 105, 120, 80]$. Compute the chi-square goodness-of-fit
  statistic $\chi^2 = \sum_i (O_i - E_i)^2 / E_i$ under the null hypothesis that the die is fair, state the
  number of degrees of freedom (and why it is not simply the number of faces), and compute the critical value
  at the 5% significance level and the p-value. State the conclusion, and say when this test is unreliable.
  Say how this bears on testing whether a sampler's output matches its target distribution.
- **Q9.** You are given two arrays of floats, `a` and `b`, containing i.i.d. samples from two unknown
  one-dimensional distributions, and a further float `x`. Give a rule, with its justification, for deciding
  which array `x` is more likely to have come from, and say what should be reported instead of only a label.
  Then take $a$ to be samples from $\mathcal N(0, 1)$ and $b$ to be samples from $\mathcal N(0, 3^2)$: with
  the true densities and equal array sizes, find the set of $x$ assigned to $a$, and how it changes when $a$
  has three times as many samples as $b$ (with the class prior taken from the array sizes). Say when a rule
  based only on the distance to the nearest sample mean fails, and relate it to the two arrays above.

### Information theory

- **Q10.** For a distribution $p$ over a finite set of size $K$, define the entropy $H(p)$, and, for a second
  distribution $q$ over the same set, the cross-entropy $H(p, q)$. Derive the identity relating $H(p, q)$ to
  $H(p)$ and $\mathrm{KL}(p \,\|\, q)$, and say what it implies about minimising cross-entropy over $q$ for a
  fixed $p$, and about the relationship between minimising cross-entropy and maximum likelihood estimation.
  Prove that $H(p) \le \log K$ for every $p$, with equality iff $p$ is uniform. State the relationship between
  entropy measured with $\log$ and with $\log_2$. Define perplexity as a transform of cross-entropy, and
  compute it (a) for a model that scores 2.0 nats of cross-entropy loss per token, and (b) for a model that
  assigns probability $1/V$ to every one of $V$ possible tokens regardless of the input.
- **Q11.** Define the mutual information $I(X; Y)$ between two random variables as a Kullback–Leibler
  divergence, and derive its two other standard forms: in terms of a marginal and a conditional entropy, and
  in terms of the three entropies $H(X)$, $H(Y)$ and $H(X, Y)$. Show that $I(X; Y) \ge 0$, with equality iff
  $X$ and $Y$ are independent, and that $I(X; Y) = I(Y; X)$. Compute $I(X; Y)$, in bits, for the case where
  $X$ is a uniformly random bit and $Y = X \oplus N$ for an independent noise bit $N$ that equals 1 with
  probability 0.1 (a *binary symmetric channel*). State the relationship between mutual information and (a)
  the information gain used to choose a split in a decision tree, and (b) the bound the InfoNCE contrastive
  loss gives on $I(X; Y)$.

### Calculus

- **Q12.** Compute each of the following by hand, stating the method used: $\int x e^x\,dx$;
  $\int \ln x\,dx$ for $x > 0$; $\int dx / (1 + x^2)$, and hence
  $\int_{-\infty}^{\infty} dx / (\pi(1 + x^2))$; $\int_{-\infty}^{\infty} e^{-x^2/2}\,dx$; and $E[X]$ for
  $X \sim \mathrm{Exp}(\lambda)$, directly from its definition as an integral. Give one use, in machine
  learning, for this kind of integral.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Confirm the convention for $\log$ (natural, used throughout) and whether a named result (Strassen's
algorithm, the Berry–Esseen theorem, McNemar's test) may be quoted or should be derived from first
principles; the answers below quote them.

### Linear algebra

**Q1.** $mnp$ multiply–adds (about $2mnp$ floating-point operations: one multiply and one add per term), and
$O(n^3)$ for square matrices with the schoolbook algorithm. Forming an entry of $C = AB$ sums $n$ products,
and there are $mp$ entries to form, for a total of $mnp$. Strassen's algorithm multiplies two $2 \times 2$
block matrices using 7 sub-multiplications instead of 8, giving the recursion $T(n) = 7T(n/2) + O(n^2)$; by
the master theorem, $T(n) = O(n^{\log_2 7}) \approx O(n^{2.807})$. For $A$ ($1000 \times 10$), $B$
($10 \times 1000$), $v$ ($1000 \times 1$): forming $AB$ first costs $1000 \cdot 10 \cdot 1000 = 10^7$
multiply–adds, and $(AB)v$ a further $1000 \cdot 1000 = 10^6$, for $1.1 \times 10^7$ in total; forming $Bv$
first costs $10 \cdot 1000 = 10^4$, and $A(Bv)$ a further $1000 \cdot 10 = 10^4$, for $2 \times 10^4$ in
total — 550 times cheaper, even though both orders compute the exact same vector, since matrix multiplication
is associative. ML: the forward pass of a linear layer $y = xW$ on a batch of tokens, $x \in \mathbb
R^{B \times d_{in}}$, $W \in \mathbb R^{d_{in} \times d_{out}}$, costs $Bd_{in}d_{out}$ multiply–adds, i.e.
$2Bd_{in}d_{out}$ FLOPs, by the same $mnp$ formula; and a low-rank update $x(AB)$, with $A \in \mathbb
R^{d_{in} \times r}$, $B \in \mathbb R^{r \times d_{out}}$ and $r$ small (as in LoRA), should be computed as
$(xA)B$, for the same reason the second route above is cheaper.

**Q2.** The rank of $A \in \mathbb R^{m \times n}$ is the dimension of its column space (equivalently, of its
row space); a square $A \in \mathbb R^{n \times n}$ has full rank if $\mathrm{rank}(A) = n$. Every column of
$AB$ is a linear combination of the columns of $A$, so the column space of $AB$ is contained in that of $A$,
giving $\mathrm{rank}(AB) \le \mathrm{rank}(A)$; symmetrically, every row of $AB$ is a combination of the
rows of $B$, giving $\mathrm{rank}(AB) \le \mathrm{rank}(B)$, and together
$\mathrm{rank}(AB) \le \min(\mathrm{rank}(A), \mathrm{rank}(B))$. Every column of $uv^\top$ is a scalar
multiple of $u$ (its $j$-th column is $v_j u$), so its column space is the line spanned by $u$: rank 1, for
$u, v$ both nonzero. For square $A$, each of the following is equivalent to $A$ being invertible: $\det(A)
\ne 0$; $A$ has full rank $n$; $Ax = 0$ only for $x = 0$; the columns (or rows) of $A$ are linearly
independent; $0$ is not an eigenvalue of $A$; every singular value of $A$ is nonzero. The determinant is a
poor numerical test because it conflates scale with conditioning: $\det(0.1\, I_{100}) = 0.1^{100} =
10^{-100}$, which any fixed threshold would call zero (it underflows to exactly $0$ in single precision, and
$\det(0.1\, I_{400}) = 10^{-400}$ does so even in double precision), even though $0.1\, I_{100}$ has every
singular value equal to $0.1$ and is as well conditioned as a matrix can be. A test that keeps the two apart
is the smallest singular value $\sigma_{\min}$ (zero, or numerically negligible next to $\sigma_{\max}$,
exactly when $A$ is singular or effectively so), or equivalently the condition number
$\kappa(A) = \sigma_{\max}/\sigma_{\min}$, which is scale-invariant and large exactly when $A$ is
ill-conditioned, regardless of $\det(A)$. ML: for a data matrix $X \in \mathbb R^{n \times d}$, $X^\top X$ is
singular iff there is a nonzero $v$ with $v^\top X^\top X v = \|Xv\|^2 = 0$, i.e. $Xv = 0$, i.e. $X$ does not
have full column rank — which happens with collinear features, or with more features than examples
($d > n$); ridge regression adds $\lambda I$ to guarantee invertibility. A LoRA update $BA$, with
$B \in \mathbb R^{d \times r}$ and $A \in \mathbb R^{r \times d}$, has rank at most $r$ by the bound above,
since $\mathrm{rank}(B) \le r$.

**Q3.** Factorise $A$ once — by Gaussian elimination with partial pivoting, i.e. an $LU$ decomposition
$PA = LU$ for a permutation $P$ and triangular $L, U$ — in $O(n^3)$ time (about $\tfrac23 n^3$ flops), then
solve $Ax = b$ with one forward and one backward triangular solve (each $O(n^2)$). Forming $A^{-1}$
explicitly costs about $2n^3$ flops (the same LU factorisation, followed by $n$ triangular solves, one for
each column of the identity) — roughly three times the factorisation alone — and then $A^{-1}b$ costs a
further $O(n^2)$ per right-hand side, the same order as an extra triangular solve; so there is no saving from
having pre-computed $A^{-1}$, only the much larger upfront cost. Beyond flop count, factor-and-solve is also
more accurate: it is backward stable, so its residual $\|A\hat x - b\|$ stays at the level of machine
precision regardless of how well conditioned $A$ is, whereas the residual after explicitly forming and
applying $A^{-1}$ grows with the condition number of $A$. The forward error $\|\hat x - x\|/\|x\|$ is governed
by the condition number on both routes — below roughly $n\,\kappa(A)$ times machine precision — so there the
inverse route is no better, and which route comes out ahead in a given run depends on rounding details. The
two routes agree closely on a well-conditioned matrix, but on an ill-conditioned one (checked below on Hilbert
matrices) explicit inversion brings the extra cost and a residual at least a thousand times larger for nothing
in return. When $A$ is
symmetric positive definite, Cholesky computes $A = LL^\top$ for a triangular $L$, exploiting symmetry to
finish in about $\tfrac13 n^3$ flops — twice as fast again as a general LU factorisation. ML: the normal
equations $X^\top X w = X^\top y$ have a symmetric positive definite coefficient matrix (when $X$ has full
column rank), so they are solved by Cholesky rather than by inverting $X^\top X$ (and when $X$ is
ill-conditioned, by a QR factorisation of $X$ itself, since forming $X^\top X$ squares the condition number);
a Gaussian-process posterior
is computed from a Cholesky factor of the (symmetric positive definite) kernel matrix; and a Newton step
$Hd = -g$ is taken by factorising $H$ and solving, never by forming $H^{-1}$.

**Q4.** Write $A = U\Sigma V^\top$; the pseudo-inverse is $A^+ = V\Sigma^+U^\top$, where $\Sigma^+$ transposes
the shape of $\Sigma$ and replaces every nonzero singular value $\sigma_i$ by $1/\sigma_i$, leaving zero
entries at zero. For any $x \in \mathbb R^n$, write $x = Vc$ (since $V$ is an orthogonal basis), so that
$Ax = U\Sigma V^\top Vc = U\Sigma c$ and, since $U$ is orthogonal, $\|Ax - b\|^2 = \|\Sigma c - U^\top
b\|^2$ — separable across the coordinates of $c$. For a coordinate $i$ with $\sigma_i > 0$, the term
$(\sigma_i c_i - (U^\top b)_i)^2$ is minimised uniquely at $c_i = (U^\top b)_i / \sigma_i$; for a coordinate
with $\sigma_i = 0$, the term does not depend on $c_i$ at all, so every value of $c_i$ minimises
$\|Ax - b\|$ equally, and since $\|x\|^2 = \|Vc\|^2 = \|c\|^2$ (again because $V$ is orthogonal), the value of
smallest norm is $c_i = 0$. Collecting both cases gives $c = \Sigma^+U^\top b$, i.e.
$x = V\Sigma^+U^\top b = A^+b$: simultaneously a minimiser of $\|Ax - b\|$ and, among all minimisers, the one
of smallest norm. When $A$ has full column rank, every singular value is nonzero and
$A^\top A = V\Sigma^\top\Sigma V^\top$ is invertible with inverse $V(\Sigma^\top\Sigma)^{-1}V^\top$, so
$(A^\top A)^{-1}A^\top = V(\Sigma^\top\Sigma)^{-1}\Sigma^\top U^\top = V\Sigma^+U^\top = A^+$. For the ridge
limit, $(A^\top A + \lambda I)^{-1}A^\top = V(\Sigma^\top\Sigma + \lambda I)^{-1}\Sigma^\top U^\top$, whose
$i$-th diagonal entry is $\sigma_i / (\sigma_i^2 + \lambda)$: as $\lambda \to 0^+$ this tends to $1/\sigma_i$
when $\sigma_i > 0$, and to $0$ when $\sigma_i = 0$ (the numerator already being $0$), matching $\Sigma^+$
exactly. ML: for underdetermined linear regression (more parameters than examples), infinitely many weight
vectors fit the training data exactly, and $A^+b$ picks out the one of smallest norm; gradient descent started
from $w_0 = 0$ on a least-squares objective only ever moves within the row space of $A$ (every gradient
$A^\top(Aw - b)$ lies there), so among the many zero-residual solutions it could reach, it converges to the
one solution that lies in the row space — exactly the minimum-norm solution $A^+b$: an implicit bias toward
small weights from the optimisation itself, with no explicit regulariser.

### Probability and statistics

**Q5.** For $X \sim \mathrm{Exp}(\lambda)$, $E[X^k] = k!/\lambda^k$ for every positive integer $k$; its
skewness is $2$ and its excess kurtosis is $6$ (both independent of $\lambda$). The moment-generating
function is $M_X(t) = E[e^{tX}] = \int_0^\infty e^{tx}\lambda e^{-\lambda x}\,dx = \lambda/(\lambda - t)$ for
$t < \lambda$, and expanding $e^{tX} = \sum_k (tX)^k/k!$ termwise under the expectation gives
$M_X(t) = \sum_k E[X^k]\,t^k/k!$, so $E[X^k] = M_X^{(k)}(0)$, the $k$-th derivative of $M_X$ at $t = 0$
(matching the definition of a Taylor coefficient). Directly, substituting $u = \lambda x$,

$$E[X^k] = \int_0^\infty x^k \lambda e^{-\lambda x}\,dx = \frac{1}{\lambda^k}\int_0^\infty u^k e^{-u}\,du = \frac{k!}{\lambda^k},$$

using $\Gamma(k + 1) = k!$ for integer $k$. From the raw moments, $\mu = 1/\lambda$ and
$\sigma^2 = E[X^2] - \mu^2 = 2/\lambda^2 - 1/\lambda^2 = 1/\lambda^2$; expanding the central moments in raw ones,
$E[(X - \mu)^3] = E[X^3] - 3\mu E[X^2] + 2\mu^3 = (6 - 6 + 2)/\lambda^3 = 2/\lambda^3$ and
$E[(X - \mu)^4] = E[X^4] - 4\mu E[X^3] + 6\mu^2 E[X^2] - 3\mu^4 = (24 - 24 + 12 - 3)/\lambda^4 = 9/\lambda^4$,
so dividing by $\sigma^3$ and $\sigma^4$ gives skewness $2$ and kurtosis $9$, i.e. excess kurtosis $6$. Some
distributions have no moments at all: the standard Cauchy density $1/(\pi(1 + x^2))$ has $E|X| = \infty$ (the
tail integral $\int^M x/(\pi(1+x^2))\,dx$ grows like $\log M$), so it has no mean, and hence no higher moments
either. Others have moments only up to a finite order: the Student-$t$ distribution with $\nu$ degrees of
freedom has tails decaying like $|x|^{-\nu - 1}$, so $E[|X|^k]$ is finite exactly for $k < \nu$ — for
instance, $t_3$ has a finite mean and variance but no finite skewness or kurtosis. ML: Adam keeps $m_t$ and
$v_t$ as exponential moving averages of the gradient and of its elementwise square, i.e. running estimates of
its first and second raw moments; batch and layer normalisation centre and rescale activations with the mean
and variance computed over a batch or over the features, and RMSNorm rescales with the second raw moment alone.

**Q6.** The weak law of large numbers: if $X_1, \dots, X_n$ are i.i.d. with finite mean $\mu$, then
$\bar X_n \to \mu$ in probability as $n \to \infty$. The central limit theorem: if in addition
$\mathrm{Var}(X_i) = \sigma^2 < \infty$, then $\sqrt n(\bar X_n - \mu)/\sigma$ converges in distribution to
$N(0, 1)$. Three ways this can fail or mislead at a finite $n$: (a) *infinite variance or mean* — for i.i.d.
standard Cauchy samples, $\bar X_n$ has exactly the standard Cauchy distribution for every $n$ (checked below
by its interquartile range, which stays at $2$ instead of shrinking), since neither the mean nor the variance
exists for the LLN or the CLT to act on; (b) *dependence* — for equicorrelated $X_i$ with common variance
$\sigma^2$ and pairwise correlation $\rho > 0$,

$$\mathrm{Var}(\bar X_n) = \frac1{n^2}\Bigl(\sum_i \mathrm{Var}(X_i) + \sum_{i \ne j}\mathrm{Cov}(X_i, X_j)\Bigr) = \frac1{n^2}\bigl(n\sigma^2 + n(n-1)\rho\sigma^2\bigr) = \frac{\sigma^2}{n}\bigl(1 + (n-1)\rho\bigr) \xrightarrow{n \to \infty} \rho\sigma^2 \ne 0,$$

so the sample mean never concentrates at $\mu$ — its variance stays above $\rho\sigma^2$ however large $n$ is —
because the shared component of the $X_i$ never averages out; (c) *heavy skew at a moderate $n$* — the Berry–Esseen theorem bounds how far the
standardised sum's distribution function can be from $\Phi$ by $C\rho/(\sigma^3\sqrt n)$, so the bound is
large whenever $\rho/\sigma^3$ (closely related to the skewness) is large, and a strongly skewed distribution
needs a substantially bigger $n$ before its standardised sum looks Gaussian than a symmetric one does;
concretely, the skewness of $\bar X_n$ itself for i.i.d. samples of skewness $\gamma_1$ is $\gamma_1/\sqrt n$
(checked below), which is still far from $0$ at, say, $n = 20$ for an exponential. ML: minibatch gradient
noise is, to leading order, the average of $B$ roughly independent per-example gradient contributions, so its
standard deviation scales as $1/\sqrt B$ (a mean of $B$ independent terms has variance $\sigma^2/B$) and, by the
CLT, its distribution is close to Gaussian; and the normal-approximation confidence interval for a
test accuracy $\hat p$ relies on the same CLT, which is a poor approximation to the true, markedly skewed,
binomial distribution when $\hat p$ is close to $0$ or $1$ and $n$ is not very large.

**Q7.** $H_0$ is the statement under test (e.g. "the die is fair"); $H_1$ is what would be concluded if $H_0$
were rejected. A test statistic $T$ has a distribution under $H_0$ that is known or well approximated; the
p-value is $P(T$ at least as extreme as observed $\mid H_0)$ — the probability, under the null over
hypothetical repeats, of a statistic this extreme, not the probability that $H_0$ is true (which would need a
prior). The significance level $\alpha$ is the type I error rate the test is calibrated to, enforced by
rejecting when $p<\alpha$. The type II error rate $\beta$ is the probability of failing to reject $H_0$ when
a specific $H_1$ holds, and the power $1-\beta$ is the probability of correctly rejecting it then.

Let $C_i^{(1)}, C_i^{(2)}\in\{0,1\}$ record correctness of the two models on example $i$, and
$D_i=C_i^{(1)}-C_i^{(2)}$. The point estimate $\bar D$ is the same either way, but its variance is not:
pairing uses

$$\mathrm{Var}(\bar D) = \tfrac1n\bigl(\mathrm{Var}(C^{(1)}_i)+\mathrm{Var}(C^{(2)}_i)-2\,\mathrm{Cov}(C^{(1)}_i,C^{(2)}_i)\bigr),$$

while treating the two accuracies as independent omits the $-2\,\mathrm{Cov}/n$ term. Two models tend to
agree, beyond chance, on which examples are easy or hard, so $\mathrm{Cov}>0$ in practice; the
independent-sample variance then overstates the true variance, giving less power than the correctly paired
test (McNemar's test on the discordant counts $n_{10},n_{01}$, statistic $(n_{10}-n_{01})^2/(n_{10}+n_{01})$,
approximately $\chi^2_1$ under $H_0$; or an equivalent paired bootstrap). If instead $\mathrm{Cov}<0$, the
independent-sample variance is too small and rejects a true $H_0$ too often; the paired variance is the right
one either way.

With 20 independent tests at $\alpha=0.05$ and every null true, the probability all 20 fail to reject is
$0.95^{20}\approx0.358$, so the probability at least one falsely rejects is $1-0.95^{20}\approx0.642$. The
Bonferroni correction rejects test $i$ only when $p_i<\alpha/m$; by the union bound,
$P(\text{any false rejection})\le\sum_i P(p_i<\alpha/m)=\alpha$, controlling the family-wise error rate at
$\alpha$, at the cost of power when $m$ is large. Benjamini–Hochberg instead sorts
$p_{(1)}\le\dots\le p_{(m)}$, finds the largest $k$ with $p_{(k)}\le(k/m)\alpha$, and rejects
$H_{(1)},\dots,H_{(k)}$; this controls the false discovery rate — the expected proportion of rejections that
are false — at $\alpha$, rather than the probability of any false positive.

**Q8.** With $600$ rolls and a fair die, the expected count per face is $E_i = 100$; the statistic is

$$\chi^2 = \sum_{i=1}^6 \frac{(O_i - E_i)^2}{E_i} = \frac{10^2 + 10^2 + 5^2 + 5^2 + 20^2 + 20^2}{100} = \frac{100 + 100 + 25 + 25 + 400 + 400}{100} = 10.5.$$

The degrees of freedom are $6 - 1 = 5$, not $6$, because the six observed counts are not free to vary
independently: they must sum to the number of rolls, so the six deviations satisfy one linear constraint,
$\sum_i (O_i - E_i) = 600 - 600 = 0$ (and no parameter of the null distribution is estimated from the data,
which would remove one more degree of freedom per parameter). At the $5\%$ level, the critical value is
$\chi^2_{0.95, 5} \approx 11.07$, and the p-value is $P(\chi^2_5 \ge 10.5) \approx 0.062$. Since
$10.5 < 11.07$ (equivalently, $0.062 > 0.05$), the test does not reject the fair-die hypothesis at the $5\%$
level, though the result is close to the boundary and would be judged differently at, say, the $10\%$ level.
The chi-square approximation relies on each expected count being reasonably large (a common rule of thumb: at
least $5$, and violated in at most about a fifth of the cells); with small expected counts, the true sampling
distribution of the statistic departs from the $\chi^2_5$ curve, and an exact multinomial test or a simulated
null distribution should be used instead. ML: the same statistic tests whether samples produced by a
generative or sampling procedure match a known target distribution over a discrete set of outcomes, by
binning the samples and comparing observed to expected counts.

**Q9.** Treat the array of origin as a latent class $C\in\{a,b\}$ with prior $\pi_a,\pi_b$ (the relative
array sizes, unless told otherwise) and class-conditional densities $p_a,p_b$ estimated from each array — by
fitting a parametric family (e.g. Gaussian, from each array's sample mean and variance), by kernel density
estimation, or via a nearest-neighbour rule; the Bayes-optimal rule assigns $x$ to the class of larger
posterior $P(C\mid x)\propto\pi_C p_C(x)$, and the posterior itself,

$$P(a\mid x) = \frac{\pi_a p_a(x)}{\pi_a p_a(x) + \pi_b p_b(x)},$$

should be reported rather than only the arg-max label. With a Gaussian fit and unequal variances, the
boundary $\pi_a p_a(x)=\pi_b p_b(x)$ is quadratic in $x$ after taking logs, so it generally has two roots
rather than one: one array occupies the middle interval and the other the two tails, depending on which has
the larger variance.

For $a\sim N(0,1)$, $b\sim N(0,3^2)$ and equal array sizes ($\pi_a=\pi_b=\tfrac12$), the boundary
$\tfrac1{\sqrt{2\pi}}e^{-x^2/2}=\tfrac1{3\sqrt{2\pi}}e^{-x^2/18}$ simplifies, after multiplying by
$3\sqrt{2\pi}$ and taking logs, to $\ln3=x^2\bigl(\tfrac12-\tfrac1{18}\bigr)=\tfrac{4x^2}9$, i.e.
$x=\pm\tfrac32\sqrt{\ln3}\approx\pm1.572$; since $a$ is the more concentrated density, it wins in the middle,
so $x$ is assigned to $a$ iff $|x|<1.5\sqrt{\ln3}$. With three times as many samples in $a$ as in $b$, the
priors become $\pi_a=\tfrac34,\pi_b=\tfrac14$, and the same steps give
$\tfrac{\pi_b}{\pi_a}=\tfrac13=3e^{-4x^2/9}$, i.e. $x^2=\tfrac94\ln9$, so the threshold widens to
$x=\pm1.5\sqrt{\ln9}\approx\pm2.223$: a prior more favourable to $a$ enlarges the region assigned to it.

A rule based only on the distance to the nearest sample mean fails whenever the two means coincide, exactly
as here (both means are $0$): every $x$ is equidistant from the two means, so the rule cannot discriminate at
all, even though the two distributions are easily told apart by their spread. The same posterior rule
underlies out-of-distribution detection, and is exactly what Gaussian discriminant analysis and naive Bayes
compute from class-conditional densities fitted per class.

### Information theory

**Q10.** For $p$ over a finite set of size $K$, $H(p) = -\sum_x p(x)\log p(x) = E_p[-\log p(X)]$; for a
second distribution $q$ over the same set, $H(p, q) = -\sum_x p(x)\log q(x) = E_p[-\log q(X)]$. Splitting
$\log q(x) = \log p(x) - \log\frac{p(x)}{q(x)}$ inside the sum,

$$H(p, q) = -\sum_x p(x)\log p(x) - \sum_x p(x)\log\frac{q(x)}{p(x)} = H(p) + \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p) + \mathrm{KL}(p \,\|\, q).$$

Since $H(p)$ does not depend on $q$, minimising $H(p, q)$ over $q$ for a fixed $p$ is exactly minimising
$\mathrm{KL}(p \,\|\, q)$; and when $p$ is the true (or empirical) data distribution and $q_\theta$ is a
model, the law of large numbers gives $-\frac1n\sum_i\log q_\theta(x_i) \to H(p, q_\theta)$ over training
examples $x_i \sim p$, so minimising the average negative log-likelihood — empirical cross-entropy — over
$\theta$ is minimising cross-entropy, which is maximum likelihood estimation. For the maximum,
$\mathrm{KL}(p \,\|\, \mathrm{uniform}) = \sum_x p(x)\log\bigl(p(x)K\bigr) = -H(p) + \log K$ (using
$\sum_xp(x)=1$), and this is $\ge 0$ by non-negativity of KL, so $H(p) \le \log K$, with equality iff $p$ is
uniform (the equality case of KL). Entropy measured with $\log_2$ (bits) is entropy measured with $\log$
(nats) divided by $\log 2$, since $\log_2 x = \log x/\log 2$. Perplexity is $\mathrm{PPL} = \exp\bigl(H(p,
q)\bigr)$ when the cross-entropy is in nats: a model scoring $2.0$ nats of cross-entropy loss per token has
perplexity $e^2 \approx 7.39$; and, for $q$ uniform over $V$ tokens, $H(p, q) = -\sum_xp(x)\log(1/V) = \log V$
for *every* $p$ (the $\log(1/V)$ factors out of the sum), so a uniform model has perplexity exactly $V$,
regardless of the true distribution over tokens. ML: cross-entropy is exactly the loss minimised when
training a classifier or a language model by maximum likelihood, and perplexity is the number practitioners
report instead of the raw loss, precisely because of this direct reading as an effective vocabulary size.

**Q11.** $I(X; Y) = \mathrm{KL}\bigl(p(x,y) \,\|\, p(x)p(y)\bigr) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$.
Writing $\frac{p(x,y)}{p(x)p(y)} = \frac{p(x \mid y)}{p(x)}$,

$$I(X; Y) = \sum_{x,y} p(x,y)\log\frac{p(x \mid y)}{p(x)} = -\sum_{x,y} p(x,y)\log p(x) + \sum_{x,y} p(x,y)\log p(x \mid y) = H(X) - H(X \mid Y),$$

where $H(X \mid Y) = -\sum_{x,y}p(x,y)\log p(x\mid y) = E_Y[H(X \mid Y = y)]$ is the conditional entropy. The
chain rule $H(X, Y) = H(Y) + H(X \mid Y)$ follows from $p(x,y) = p(y)p(x\mid y)$ by taking $-\log$ and an
expectation over $p(x,y)$; substituting $H(X\mid Y) = H(X,Y) - H(Y)$ gives the third form,

$$I(X; Y) = H(X) - \bigl(H(X,Y) - H(Y)\bigr) = H(X) + H(Y) - H(X, Y),$$

manifestly symmetric in $X, Y$, so $I(X; Y) = I(Y; X)$. Non-negativity, $I(X; Y) \ge 0$ with equality iff
$p(x,y) = p(x)p(y)$ almost everywhere (i.e. $X \perp Y$), follows directly from the non-negativity of KL
applied to $p(x,y)$ and $p(x)p(y)$. For the binary symmetric channel, $X \sim \mathrm{Bernoulli}(1/2)$ and
$Y = X \oplus N$ with $N \sim \mathrm{Bernoulli}(0.1)$ independent of $X$: since $X$ is uniform and the
channel is symmetric, $Y$ is uniform too, so $H(Y) = 1$ bit; and for either value of $X$, $Y = X \oplus N$
has exactly the entropy of $N$, so $H(Y \mid X) = H_b(0.1)$, the binary entropy of $0.1$ in bits. Hence

$$I(X; Y) = H(Y) - H(Y \mid X) = 1 - H_b(0.1) \approx 1 - 0.469 = 0.531 \text{ bits}.$$

ML: the information gain used to pick a split in a decision tree is exactly the mutual information between
the split variable and the label, $I(\text{split}; Y) = H(Y) - H(Y \mid \text{split})$; and the InfoNCE
contrastive loss $L$, computed from one positive pair and $N - 1$ negatives per anchor, satisfies
$I(X; Y) \ge \log N - L$, so driving $L$ down is driving up a lower bound on the mutual information between
the two views.

### Calculus

**Q12.** $\int xe^x\,dx = (x - 1)e^x + C$: integrate by parts with $u = x$, $dv = e^x\,dx$ (so $du = dx$,
$v = e^x$), giving $\int xe^x\,dx = xe^x - \int e^x\,dx = xe^x - e^x + C = (x-1)e^x + C$. $\int\ln x\,dx =
x\ln x - x + C$ for $x > 0$: by parts with $u = \ln x$, $dv = dx$ ($du = dx/x$, $v = x$),
$\int \ln x\,dx = x\ln x - \int x \cdot \tfrac1x\,dx = x\ln x - x + C$. $\int\frac{dx}{1 + x^2} = \arctan x +
C$, since $\frac{d}{dx}\arctan x = \frac1{1+x^2}$; hence the Cauchy density integrates to $1$,

$$\int_{-\infty}^{\infty} \frac{dx}{\pi(1 + x^2)} = \frac1\pi\Bigl[\arctan x\Bigr]_{-\infty}^{\infty} = \frac1\pi\Bigl(\frac\pi2 - \bigl(-\frac\pi2\bigr)\Bigr) = 1.$$

$\int_{-\infty}^{\infty} e^{-x^2/2}\,dx = \sqrt{2\pi}$: writing $I$ for this integral and squaring it,

$$I^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2 + y^2)/2}\,dx\,dy = \int_0^{2\pi}\int_0^\infty e^{-r^2/2}\,r\,dr\,d\theta = 2\pi\Bigl[-e^{-r^2/2}\Bigr]_0^\infty = 2\pi,$$

switching to polar coordinates ($x = r\cos\theta$, $y = r\sin\theta$, area element $r\,dr\,d\theta$), so
$I = \sqrt{2\pi}$, the positive root, since the integrand is everywhere positive. $E[X] = 1/\lambda$ for
$X \sim \mathrm{Exp}(\lambda)$: by parts with $u = x$, $dv = \lambda e^{-\lambda x}\,dx$ ($v = -e^{-\lambda
x}$),

$$E[X] = \int_0^\infty x\lambda e^{-\lambda x}\,dx = \Bigl[-xe^{-\lambda x}\Bigr]_0^\infty + \int_0^\infty e^{-\lambda x}\,dx = 0 + \frac1\lambda = \frac1\lambda,$$

consistent with the $k = 1$ case of Q5's $E[X^k] = k!/\lambda^k$. ML: integrals of exactly this kind normalise
densities (checking or deriving that a density integrates to $1$, as for the Cauchy and Gaussian above), and
compute the expectations that appear in likelihoods and their gradients.

<details>
<summary>Checks (runnable)</summary>

```python
import math

import numpy as np
from scipy import integrate, optimize, stats
from scipy.linalg import cho_factor, cho_solve, hilbert, lu_factor, lu_solve

rng = np.random.default_rng(2026)


# ---- Q1: multiply-add counts by explicit loops and by the m n p formula, associativity, ML costs
def multiply_adds_and_product(A, B):
    """The schoolbook algorithm computed by explicit loops, counting one multiply-add per inner step."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    count = 0
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
                count += 1
            C[i, j] = s
    return count, C


for (m1, n1, p1) in [(3, 4, 5), (5, 2, 6)]:
    A1, B1 = rng.standard_normal((m1, n1)), rng.standard_normal((n1, p1))
    count1, C1 = multiply_adds_and_product(A1, B1)
    assert count1 == m1 * n1 * p1
    assert np.allclose(C1, A1 @ B1)

assert math.isclose(math.log2(7), 2.807, abs_tol=5e-4)          # Strassen's exponent

m_big, n_big, p_big = 1000, 10, 1000
A_big = rng.standard_normal((m_big, n_big))
B_big, v_big = rng.standard_normal((n_big, p_big)), rng.standard_normal(p_big)
cost_ab_then_v = m_big * n_big * p_big + m_big * p_big          # form AB (m x p), then (AB) v
cost_bv_then_a = n_big * p_big + m_big * n_big                  # form Bv (n,), then A (Bv)
assert cost_ab_then_v == 11_000_000
assert cost_bv_then_a == 20_000
assert np.allclose((A_big @ B_big) @ v_big, A_big @ (B_big @ v_big), atol=1e-6)   # associativity

batch_tok, d_in, d_out = 6, 5, 4                                 # ML: forward pass of a linear layer
X_tok, W_layer = rng.standard_normal((batch_tok, d_in)), rng.standard_normal((d_in, d_out))
count_layer, _ = multiply_adds_and_product(X_tok, W_layer)
assert count_layer == batch_tok * d_in * d_out

batch_lr, d_lr, r_lr, d_out_lr = 50, 100, 4, 100                 # ML: low-rank update x (A B) as (x A) B
X_lr = rng.standard_normal((batch_lr, d_lr))
A_lr, B_lr = rng.standard_normal((d_lr, r_lr)), rng.standard_normal((r_lr, d_out_lr))
cost_full_lr = d_lr * r_lr * d_out_lr + batch_lr * d_lr * d_out_lr
cost_factored_lr = batch_lr * d_lr * r_lr + batch_lr * r_lr * d_out_lr
assert cost_full_lr / cost_factored_lr == 13.5
assert np.allclose(X_lr @ (A_lr @ B_lr), (X_lr @ A_lr) @ B_lr, atol=1e-6)

# ---- Q2: rank, rank(AB) <= min(rank A, rank B), outer products, det vs conditioning, ML consequences
for _ in range(200):
    m2, k2, p2 = (int(rng.integers(2, 6)) for _ in range(3))
    A2, B2 = rng.standard_normal((m2, k2)), rng.standard_normal((k2, p2))
    assert np.linalg.matrix_rank(A2 @ B2) <= min(np.linalg.matrix_rank(A2), np.linalg.matrix_rank(B2))

u_out, v_out = rng.standard_normal(5), rng.standard_normal(4)
assert np.linalg.matrix_rank(np.outer(u_out, v_out)) == 1
assert np.linalg.matrix_rank(np.outer(u_out, v_out) @ rng.standard_normal((4, 6))) == 1   # still <= 1

det_small = np.linalg.det(0.1 * np.eye(100))
assert math.isclose(det_small, 1e-100, rel_tol=1e-6)                     # representable in float64, yet "tiny"
assert np.linalg.det(0.1 * np.eye(100, dtype=np.float32)) == 0.0         # NOTE: 1e-100 underflows to 0 in float32
assert np.linalg.det(0.1 * np.eye(400)) == 0.0                           # and 1e-400 underflows even in float64
assert math.isclose(np.linalg.cond(0.1 * np.eye(100)), 1.0, rel_tol=1e-9)  # perfectly conditioned regardless
assert np.allclose(np.linalg.svd(0.1 * np.eye(100), compute_uv=False), 0.1)

n_feat, d_feat = 30, 5                                            # ML: X^T X invertible iff X full column rank
X_full = rng.standard_normal((n_feat, d_feat))
assert np.linalg.matrix_rank(X_full) == d_feat and np.linalg.cond(X_full.T @ X_full) < 1e10
X_collinear = X_full.copy()
X_collinear[:, -1] = 2.0 * X_collinear[:, 0]                      # collinear feature
assert np.linalg.matrix_rank(X_collinear) == d_feat - 1 and np.linalg.cond(X_collinear.T @ X_collinear) > 1e12
X_wide_feat = rng.standard_normal((5, 20))                         # more features than examples
assert np.linalg.matrix_rank(X_wide_feat) == 5 and np.linalg.cond(X_wide_feat.T @ X_wide_feat) > 1e12

d_lora, r_lora = 64, 4                                             # ML: a LoRA update B A has rank <= r
A_lora, B_lora = rng.standard_normal((r_lora, d_lora)), rng.standard_normal((d_lora, r_lora))
assert np.linalg.matrix_rank(B_lora @ A_lora) <= r_lora

# ---- Q3: solving A x = b, LU/Cholesky factor-and-solve vs the explicit inverse
assert (2 * 200 ** 3) / (2 * 200 ** 3 / 3) == 3.0                  # LU ~ 2n^3/3 flops, inversion ~ 2n^3

n_wc = 200
M_wc = rng.standard_normal((n_wc, n_wc))
A_wc = M_wc @ M_wc.T + n_wc * np.eye(n_wc)                         # a well-conditioned SPD matrix
x_true_wc = rng.standard_normal(n_wc)
b_wc = A_wc @ x_true_wc
lu_wc, piv_wc = lu_factor(A_wc)
x_lu_wc = lu_solve((lu_wc, piv_wc), b_wc)
chol_wc, lower_wc = cho_factor(A_wc)
x_cho_wc = cho_solve((chol_wc, lower_wc), b_wc)
x_inv_wc = np.linalg.inv(A_wc) @ b_wc
assert np.linalg.norm(A_wc @ x_lu_wc - b_wc) < 1e-8               # well conditioned: factor-and-solve...
assert np.linalg.norm(A_wc @ x_inv_wc - b_wc) < 1e-8               # ...and the explicit inverse agree closely
assert np.allclose(x_lu_wc, x_true_wc, atol=1e-8) and np.allclose(x_cho_wc, x_true_wc, atol=1e-8)

# NOTE: on a badly conditioned matrix the inverse route costs more and leaves a far larger residual; average
# over many right-hand sides at each size, since a single one is a noisy comparison
reps_hilbert = 200
err_lu_by_n = {}
for n_h in (6, 8, 10):
    A_h = hilbert(n_h)
    X_true_h = rng.standard_normal((n_h, reps_hilbert))
    B_h = A_h @ X_true_h
    lu_h, piv_h = lu_factor(A_h)
    X_lu_h = lu_solve((lu_h, piv_h), B_h)
    X_inv_h = np.linalg.inv(A_h) @ B_h                             # NOTE: not backward stable -- see below
    res_lu_h = np.linalg.norm(A_h @ X_lu_h - B_h, axis=0).mean()
    res_inv_h = np.linalg.norm(A_h @ X_inv_h - B_h, axis=0).mean()
    err_lu_h = (np.linalg.norm(X_lu_h - X_true_h, axis=0) / np.linalg.norm(X_true_h, axis=0)).mean()
    err_inv_h = (np.linalg.norm(X_inv_h - X_true_h, axis=0) / np.linalg.norm(X_true_h, axis=0)).mean()
    assert res_lu_h < 1e-10                                        # LU stays backward stable regardless of cond
    assert res_inv_h > 1e3 * res_lu_h                               # the inverse route's residual does not
    # NOTE: the two forward errors are not compared with each other: both stay below about n * cond(A) * eps,
    # and which route comes out ahead, and by how much, depends on the BLAS kernels of the machine
    fwd_bound_h = n_h * np.linalg.cond(A_h) * np.finfo(float).eps
    assert err_lu_h < fwd_bound_h and err_inv_h < fwd_bound_h
    err_lu_by_n[n_h] = err_lu_h
assert err_lu_by_n[6] < err_lu_by_n[8] < err_lu_by_n[10]            # LU's forward error grows with cond
assert math.isclose(np.linalg.cond(X_full.T @ X_full), np.linalg.cond(X_full) ** 2, rel_tol=1e-6)  # X^T X squares it

# ---- Q4: Moore-Penrose pseudo-inverse: SVD formula, Penrose conditions, min-norm, ridge limit, GD
def pinv_from_svd(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.where(S > tol * S.max(), 1.0 / S, 0.0)
    return (Vt.T * S_inv) @ U.T


for shape4 in ((8, 5), (5, 8), (6, 6)):                             # tall, wide, square
    A4 = rng.standard_normal(shape4)
    Ap4 = np.linalg.pinv(A4)
    assert np.allclose(pinv_from_svd(A4), Ap4, atol=1e-8)
    assert np.allclose(A4 @ Ap4 @ A4, A4, atol=1e-8)                # the four Penrose conditions
    assert np.allclose(Ap4 @ A4 @ Ap4, Ap4, atol=1e-8)
    assert np.allclose((A4 @ Ap4).T, A4 @ Ap4, atol=1e-8)
    assert np.allclose((Ap4 @ A4).T, Ap4 @ A4, atol=1e-8)

A_tall = rng.standard_normal((9, 4))                                # full column rank: A+ = (A^T A)^-1 A^T
assert np.allclose(np.linalg.pinv(A_tall), np.linalg.inv(A_tall.T @ A_tall) @ A_tall.T, atol=1e-8)

m_wide, n_wide = 4, 9                                               # underdetermined: min-norm property
A_wide = rng.standard_normal((m_wide, n_wide))
b_wide = rng.standard_normal(m_wide)
x_pinv = np.linalg.pinv(A_wide) @ b_wide
res_pinv = np.linalg.norm(A_wide @ x_pinv - b_wide)
null_vec = np.linalg.svd(A_wide)[2][m_wide:][0]     # NOTE: svd returns V^T, so its rows are the directions of V
assert np.allclose(A_wide @ null_vec, 0, atol=1e-8)
for scale in (0.3, 1.0, -2.0):
    x_perturbed = x_pinv + scale * null_vec
    assert math.isclose(np.linalg.norm(A_wide @ x_perturbed - b_wide), res_pinv, abs_tol=1e-8)  # same residual
    assert math.isclose(np.linalg.norm(x_perturbed) ** 2, np.linalg.norm(x_pinv) ** 2 + scale ** 2,
                        rel_tol=1e-9)                                     # larger norm: orthogonal, by Pythagoras

for lam4 in (1e-2, 1e-4, 1e-6, 1e-8):                                # the ridge limit
    ridge_est = np.linalg.inv(A_wide.T @ A_wide + lam4 * np.eye(n_wide)) @ A_wide.T
    assert np.linalg.norm(ridge_est - np.linalg.pinv(A_wide)) < 5 * lam4 ** 0.5

w4 = np.zeros(n_wide)                                                # GD from 0 converges to the min-norm solution
lipschitz4 = np.linalg.eigvalsh(A_wide.T @ A_wide).max()
step4 = 1.0 / lipschitz4
for _ in range(20_000):
    w4 = w4 - step4 * (A_wide.T @ (A_wide @ w4 - b_wide))
assert np.allclose(w4, x_pinv, atol=1e-4)

# ---- Q5: raw and central moments, the MGF, distributions without moments, the exponential's moments
lam5 = 2.0


def mgf_exp(t, lam=lam5):
    return lam / (lam - t)


h5 = 1e-3
f5 = {k: mgf_exp(k * h5) for k in (-2, -1, 0, 1, 2)}
d1 = (f5[1] - f5[-1]) / (2 * h5)
d2 = (f5[1] - 2 * f5[0] + f5[-1]) / h5 ** 2
d3 = (f5[2] - 2 * f5[1] + 2 * f5[-1] - f5[-2]) / (2 * h5 ** 3)
d4 = (f5[2] - 4 * f5[1] + 6 * f5[0] - 4 * f5[-1] + f5[-2]) / h5 ** 4
for k5, dk in zip((1, 2, 3, 4), (d1, d2, d3, d4)):
    assert math.isclose(dk, math.factorial(k5) / lam5 ** k5, rel_tol=2e-2)            # MGF derivatives at 0
    quad_val5, _ = integrate.quad(lambda x, k5=k5: x ** k5 * lam5 * math.exp(-lam5 * x), 0, np.inf)
    assert math.isclose(quad_val5, math.factorial(k5) / lam5 ** k5, rel_tol=1e-8)      # and by direct integration

raw5 = {k5: math.factorial(k5) / lam5 ** k5 for k5 in (1, 2, 3, 4)}                   # central moments from raw ones
mu5 = raw5[1]
assert math.isclose(raw5[3] - 3 * mu5 * raw5[2] + 2 * mu5 ** 3, 2 / lam5 ** 3, rel_tol=1e-12)
assert math.isclose(raw5[4] - 4 * mu5 * raw5[3] + 6 * mu5 ** 2 * raw5[2] - 3 * mu5 ** 4, 9 / lam5 ** 4, rel_tol=1e-12)
_, _, skew5, kurt5 = stats.expon(scale=1 / lam5).stats(moments="mvsk")
assert math.isclose(float(skew5), 2.0, rel_tol=1e-8)
assert math.isclose(float(kurt5), 6.0, rel_tol=1e-8)                                  # excess kurtosis

mvsk_t3 = stats.t(df=3).stats(moments="mvsk")                                         # moments exist only for k < nu
assert math.isclose(float(mvsk_t3[1]), 3.0, rel_tol=1e-8)                             # variance (k=2 < 3): exists
assert not np.isfinite(float(mvsk_t3[3]))               # NOTE: a non-existent moment reports as inf/nan, not an error
cauchy_tail = [integrate.quad(lambda x: abs(x) / (math.pi * (1 + x ** 2)), -c, c)[0] for c in (1e2, 1e4, 1e6)]
assert np.all(np.diff(cauchy_tail) > 0.5)                                             # E|X| unbounded: no mean

# ---- Q6: LLN and CLT, and three ways they fail or mislead
reps6 = 8_000
for n6 in (10, 100, 1_000):
    q1c, q3c = np.percentile(rng.standard_cauchy((reps6, n6)).mean(axis=1), [25, 75])
    assert math.isclose(q3c - q1c, 2.0, abs_tol=0.2)                        # no shrinkage at all with n
    q1n, q3n = np.percentile(rng.standard_normal((reps6, n6)).mean(axis=1), [25, 75])
    assert math.isclose(q3n - q1n, 2 * stats.norm.ppf(0.75) / math.sqrt(n6), rel_tol=0.15)   # shrinks like 1/sqrt(n)

rho6 = 0.3                                                                  # strong dependence: a shared factor
reps6b = 30_000
for n6b in (5, 20, 100, 400):
    common6 = rng.standard_normal(reps6b)
    X6 = math.sqrt(rho6) * common6[:, None] + math.sqrt(1 - rho6) * rng.standard_normal((reps6b, n6b))
    var_mean6 = X6.mean(axis=1).var()
    assert math.isclose(var_mean6, (1 + (n6b - 1) * rho6) / n6b, rel_tol=0.1)
assert var_mean6 > 10 / n6b                                                 # far from the iid rate sigma^2 / n

gamma1_exp = 2.0                                                            # heavy skew at moderate n (Berry-Esseen)
reps6c = 80_000
for n6c in (5, 20, 80):
    means6c = rng.exponential(1.0, (reps6c, n6c)).mean(axis=1)
    assert math.isclose(stats.skew(means6c) * math.sqrt(n6c), gamma1_exp, rel_tol=0.15)
    # the mean of n Exp(1) variables is Gamma(n, scale 1/n): its skewness is exactly 2 / sqrt(n)
    assert math.isclose(float(stats.gamma(a=n6c, scale=1 / n6c).stats(moments="s")), gamma1_exp / math.sqrt(n6c))

# ---- Q7: p-values, familywise error and Bonferroni, and paired versus unpaired power
n7, trials7 = 30, 40_000
z7 = rng.standard_normal((trials7, n7)).mean(axis=1) * math.sqrt(n7)
p7 = 2 * (1 - stats.norm.cdf(np.abs(z7)))
assert stats.kstest(p7, "uniform").pvalue > 0.05 and math.isclose((p7 < 0.05).mean(), 0.05, abs_tol=0.01)

m7, trials7b = 20, 20_000                                                    # 20 independent true nulls
pm7 = 2 * (1 - stats.norm.cdf(np.abs(rng.standard_normal((trials7b, m7, n7)).mean(axis=2) * math.sqrt(n7))))
assert math.isclose(1 - 0.95 ** 20, 0.6415, abs_tol=5e-4)
assert math.isclose((pm7 < 0.05).any(axis=1).mean(), 1 - 0.95 ** 20, abs_tol=0.02)      # naive: much worse than 5%
assert (pm7 < 0.05 / m7).any(axis=1).mean() < 0.07                                      # Bonferroni controls it


def correlated_bernoulli(n, p_1, p_2, corr, size, rng):
    """Two Bernoulli(p_1), Bernoulli(p_2) sequences with correlated correctness, via a Gaussian copula."""
    Z = rng.multivariate_normal([0.0, 0.0], [[1.0, corr], [corr, 1.0]], size=(size, n))
    return (Z[..., 0] < stats.norm.ppf(p_1)).astype(float), (Z[..., 1] < stats.norm.ppf(p_2)).astype(float)


n7c, trials7c = 200, 4_000
c1_7, c2_7 = correlated_bernoulli(n7c, 0.75, 0.70, 0.6, trials7c, rng)
p1_hat, p2_hat = c1_7.mean(axis=1), c2_7.mean(axis=1)
diff7 = c1_7 - c2_7
se_paired = diff7.std(axis=1, ddof=1) / math.sqrt(n7c)
se_unpaired = np.sqrt(p1_hat * (1 - p1_hat) / n7c + p2_hat * (1 - p2_hat) / n7c)   # NOTE: ignores the correlation
power_paired = (np.abs(diff7.mean(axis=1) / se_paired) > stats.norm.ppf(0.975)).mean()
power_unpaired = (np.abs((p1_hat - p2_hat) / se_unpaired) > stats.norm.ppf(0.975)).mean()
assert se_paired.mean() < se_unpaired.mean()               # positive correlation shrinks the paired standard error
assert power_paired > 1.5 * power_unpaired                 # so the paired test is markedly more powerful here

rng7d = np.random.default_rng(7)                           # a separate stream, so later checks keep their draws
c1_neg, c2_neg = correlated_bernoulli(n7c, 0.70, 0.70, -0.6, trials7c, rng7d)   # H0 true, negative covariance
d_neg = c1_neg - c2_neg
se_paired_neg = d_neg.std(axis=1, ddof=1) / math.sqrt(n7c)
p1_neg, p2_neg = c1_neg.mean(axis=1), c2_neg.mean(axis=1)
se_unpaired_neg = np.sqrt(p1_neg * (1 - p1_neg) / n7c + p2_neg * (1 - p2_neg) / n7c)
size_paired = (np.abs(d_neg.mean(axis=1) / se_paired_neg) > stats.norm.ppf(0.975)).mean()
size_unpaired = (np.abs((p1_neg - p2_neg) / se_unpaired_neg) > stats.norm.ppf(0.975)).mean()
assert se_unpaired_neg.mean() < 0.95 * se_paired_neg.mean()                # the independent formula is too small
assert abs(size_paired - 0.05) < 0.015 and size_unpaired > 0.07            # so it rejects a true H0 too often

# ---- Q8: chi-square goodness of fit for the die
counts8 = np.array([90, 110, 95, 105, 120, 80])
expected8 = np.full(6, counts8.sum() / 6)
chi2_stat8 = float(((counts8 - expected8) ** 2 / expected8).sum())
assert math.isclose(chi2_stat8, 10.5, rel_tol=1e-9)
lib8 = stats.chisquare(counts8, expected8)
assert math.isclose(float(lib8.statistic), chi2_stat8, rel_tol=1e-9)
df8 = len(counts8) - 1
assert df8 == 5
crit8, pval8 = stats.chi2.ppf(0.95, df8), stats.chi2.sf(chi2_stat8, df8)
assert math.isclose(crit8, 11.07, abs_tol=5e-3) and math.isclose(pval8, 0.0622, abs_tol=5e-4)
assert math.isclose(pval8, float(lib8.pvalue), rel_tol=1e-9)
assert chi2_stat8 < crit8 and pval8 > 0.05                                   # do not reject fairness at 5%

# ---- Q9: which array does x come from -- Gaussian-discriminant thresholds and a fitted classifier
def log_posterior_gap(x, pi_a, pi_b, mu_a, sig_a, mu_b, sig_b):
    la = math.log(pi_a) - 0.5 * math.log(2 * math.pi * sig_a ** 2) - (x - mu_a) ** 2 / (2 * sig_a ** 2)
    lb = math.log(pi_b) - 0.5 * math.log(2 * math.pi * sig_b ** 2) - (x - mu_b) ** 2 / (2 * sig_b ** 2)
    return la - lb


mu9, sig_a9, sig_b9 = 0.0, 1.0, 3.0
root_equal = optimize.brentq(lambda x: log_posterior_gap(x, 0.5, 0.5, mu9, sig_a9, mu9, sig_b9), 0.1, 5.0)
assert math.isclose(root_equal, 1.5 * math.sqrt(math.log(3)), rel_tol=1e-8)
root_3to1 = optimize.brentq(lambda x: log_posterior_gap(x, 0.75, 0.25, mu9, sig_a9, mu9, sig_b9), 0.1, 5.0)
assert math.isclose(root_3to1, 1.5 * math.sqrt(math.log(9)), rel_tol=1e-8)

n9 = 4_000                                                                    # a fitted classifier on seeded samples
a9, b9 = rng.normal(mu9, sig_a9, n9), rng.normal(mu9, sig_b9, n9)
mu_a9, sig_hat_a9 = a9.mean(), a9.std(ddof=1)
mu_b9, sig_hat_b9 = b9.mean(), b9.std(ddof=1)
for x9, want_a in ((0.5, True), (4.0, False), (-4.0, False)):
    gap9 = log_posterior_gap(x9, 0.5, 0.5, mu_a9, sig_hat_a9, mu_b9, sig_hat_b9)
    assert (gap9 > 0) == want_a
def nearest_mean9(x):
    return "a" if abs(x - mu_a9) < abs(x - mu_b9) else "b"


assert "a" in (nearest_mean9(4.0), nearest_mean9(-4.0))    # both belong to b; nearest-mean sends one of them to a

# ---- Q10: entropy, cross-entropy, the H(p,q) = H(p) + KL(p||q) identity, maximum entropy, perplexity
def entropy(p):
    p = p[p > 0]                                     # NOTE: 0 log 0 is taken to be its limit, 0, not left as nan
    return float(-(p * np.log(p)).sum())


def cross_entropy(p, q):
    mask = p > 0
    return float(-(p[mask] * np.log(q[mask])).sum())


def kl_disc(p, q):
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


for _ in range(500):
    K10 = int(rng.integers(2, 10))
    p10, q10 = rng.random(K10) + 1e-3, rng.random(K10) + 1e-3
    p10, q10 = p10 / p10.sum(), q10 / q10.sum()
    assert math.isclose(cross_entropy(p10, q10), entropy(p10) + kl_disc(p10, q10), rel_tol=1e-8)

for K10b in (2, 5, 10, 50):
    unif10 = np.full(K10b, 1.0 / K10b)
    assert math.isclose(entropy(unif10), math.log(K10b), rel_tol=1e-12)
    for _ in range(50):
        p10b = rng.random(K10b) + 1e-3
        p10b /= p10b.sum()
        assert entropy(p10b) <= math.log(K10b) + 1e-9                                  # maximised by the uniform law
        assert math.isclose(kl_disc(p10b, unif10), math.log(K10b) - entropy(p10b), rel_tol=1e-8)

assert math.isclose(math.exp(2.0), 7.39, abs_tol=5e-3)                                  # perplexity at loss 2.0 nats
K10c = 500
for _ in range(30):
    p10c = rng.random(K10c) + 1e-3
    p10c /= p10c.sum()
    ce10c = cross_entropy(p10c, np.full(K10c, 1.0 / K10c))
    assert math.isclose(ce10c, math.log(K10c), rel_tol=1e-9)                            # true p is irrelevant here
    assert math.isclose(math.exp(ce10c), K10c, rel_tol=1e-8)                            # perplexity = vocabulary size

# ---- Q11: mutual information: the three formulas, independence, the BSC, information gain
def mi_definition(joint):
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    outer, mask = np.outer(px, py), joint > 1e-300
    return float((joint[mask] * np.log(joint[mask] / outer[mask])).sum())


def mi_via_cond_entropy(joint):
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    h_given_y = sum(py[j] * entropy(joint[:, j] / py[j]) for j in range(joint.shape[1]) if py[j] > 1e-300)
    return entropy(px) - h_given_y


def mi_via_joint_entropy(joint):
    return entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0)) - entropy(joint.flatten())


for _ in range(300):
    r11, c11 = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    joint11 = rng.random((r11, c11)) + 1e-3
    joint11 /= joint11.sum()
    mi1, mi2, mi3 = mi_definition(joint11), mi_via_cond_entropy(joint11), mi_via_joint_entropy(joint11)
    assert math.isclose(mi1, mi2, abs_tol=1e-8) and math.isclose(mi1, mi3, abs_tol=1e-8) and mi1 >= -1e-9

px11, py11 = rng.random(4), rng.random(3)
assert abs(mi_definition(np.outer(px11 / px11.sum(), py11 / py11.sum()))) < 1e-10        # zero on a product table


def binary_entropy_bits(p):
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


flip11 = 0.1
joint_bsc = np.array([[0.5 * (1 - flip11), 0.5 * flip11], [0.5 * flip11, 0.5 * (1 - flip11)]])
mi_bsc_bits = mi_definition(joint_bsc) / math.log(2)
assert math.isclose(mi_bsc_bits, 1 - binary_entropy_bits(flip11), rel_tol=1e-9)
assert math.isclose(mi_bsc_bits, 0.531, abs_tol=5e-4)

split11 = rng.integers(0, 2, 400)                                           # information gain = mutual information
label11 = np.where(rng.random(400) < np.where(split11 == 1, 0.8, 0.3), 1, 0)
joint_counts11 = np.zeros((2, 2))
for s11, y11 in zip(split11, label11):
    joint_counts11[s11, y11] += 1
joint_emp11 = joint_counts11 / joint_counts11.sum()
px_split11 = joint_emp11.sum(axis=1)
h_label_given_split = sum(px_split11[s11] * entropy(joint_emp11[s11] / px_split11[s11])
                          for s11 in range(2) if px_split11[s11] > 0)
info_gain11 = entropy(joint_emp11.sum(axis=0)) - h_label_given_split
assert math.isclose(info_gain11, mi_definition(joint_emp11), abs_tol=1e-9)
assert info_gain11 > 0.01

# ---- Q12: five integrals by hand, checked by differentiation or by quadrature
h12 = 1e-6


def check_antiderivative(F, f, xs):
    for x in xs:
        assert math.isclose((F(x + h12) - F(x - h12)) / (2 * h12), f(x), rel_tol=1e-4, abs_tol=1e-4)


check_antiderivative(lambda x: (x - 1) * math.exp(x), lambda x: x * math.exp(x), [-2.0, -0.5, 0.3, 1.0, 2.5])
check_antiderivative(lambda x: x * math.log(x) - x, math.log, [0.1, 0.5, 1.0, 2.0, 5.0])
check_antiderivative(math.atan, lambda x: 1 / (1 + x ** 2), [-3.0, -1.0, 0.0, 1.0, 3.0])

cauchy_integral, _ = integrate.quad(lambda x: 1 / (math.pi * (1 + x ** 2)), -np.inf, np.inf)
assert math.isclose(cauchy_integral, 1.0, rel_tol=1e-10)
gaussian_integral, _ = integrate.quad(lambda x: math.exp(-x ** 2 / 2), -np.inf, np.inf)
assert math.isclose(gaussian_integral, math.sqrt(2 * math.pi), rel_tol=1e-10)
for lam12 in (0.5, 1.0, 3.0):
    mean_exp12, _ = integrate.quad(lambda x: x * lam12 * math.exp(-lam12 * x), 0, np.inf)
    assert math.isclose(mean_exp12, 1 / lam12, rel_tol=1e-8)

print("all checks passed")
```

</details>

</details>
