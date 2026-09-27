# RL for Language Models: Policy Gradients, PPO, GRPO and DPO

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| Type | Priority | Difficulty | Roles | Topics | Format | Round |
| --- | --- | --- | --- | --- | --- | --- |
| Quiz · oral, with derivations | ★★★☆☆ | Hard | RS · RE · MLE | policy-gradient, ppo, grpo, dpo, kl-regularisation, rlhf, reward-hacking, off-policy, importance-sampling | 9 questions / 45–60 min | Skills interview |
<!-- meta:end -->

## Problem

Fix notation shared by every question below: a prompt $x$; a response $y = (y_1,\dots,y_n)$, a sequence of $n$
tokens; a policy $\pi_\theta(y\mid x) = \prod_{t=1}^n \pi_\theta(y_t \mid x, y_{<t})$, the language model being
trained, which generates $y$ one token at a time conditioned on the prompt and the tokens already generated; a
reference policy $\pi_{\text{ref}}$, a frozen copy of the policy fixed before the current training stage; a scalar
reward $r(x,y)$, larger is better; and a KL coefficient $\beta > 0$.

### Policy gradients

**Q1.** For a prompt $x$ fixed throughout, let $J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]$.
Derive $\nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\,\nabla_\theta \log \pi_\theta(y\mid x)\big]$
(the log-derivative trick), starting from $\nabla_\theta \pi_\theta(y\mid x) = \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x)$.
Use the autoregressive factorisation to write $\nabla_\theta \log \pi_\theta(y\mid x)$, and hence $\nabla_\theta J(\theta)$,
as a sum over tokens $t = 1,\dots,n$. Then, for a baseline $b$ that does not depend on $y$ (it may depend on $x$),
show that $\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[(r(x,y) - b)\,\nabla_\theta \log \pi_\theta(y\mid x)\big] = \nabla_\theta J(\theta)$
as well, i.e. subtracting $b$ leaves the estimator unbiased. State what a well-chosen baseline buys.

### PPO

**Q2.** Let $\rho = \pi_\theta(y\mid x) / \pi_{\theta_{\text{old}}}(y\mid x)$, the ratio between the policy being
updated and a fixed snapshot $\theta_{\text{old}}$ taken before the update, and let $A$ be a scalar advantage for
$(x,y)$: positive when $y$ was better than the current policy's average response to $x$, negative when worse. The
PPO clipped surrogate for one sample is

$$L^{\text{clip}}(\rho, A) = \min\big(\rho A,\ \operatorname{clip}(\rho,\, 1-\epsilon,\, 1+\epsilon)\, A\big), \qquad \operatorname{clip}(\rho, l, u) = \min(\max(\rho, l), u),$$

for a clip range $\epsilon \in (0,1)$, maximised in expectation over $\theta$. Splitting $\rho$'s position into
three regions — below $1-\epsilon$, inside $[1-\epsilon, 1+\epsilon]$, above $1+\epsilon$ — state, for each region
crossed with the sign of $A$, the value of $L^{\text{clip}}$ and of $\partial L^{\text{clip}} / \partial \rho$
(five cases in total, since inside the interval the two signs of $A$ behave alike); say which of these cases zero
out the gradient and, for those, explain in terms of a trust region what clipping protects against. Compute $L^{\text{clip}}$
for $\epsilon = 0.2$ at $(\rho, A) \in \{(1.5, 2), (0.5, 2), (1.5, -2), (0.5, -2)\}$.

**Q3.** In PPO applied to language-model post-training, a critic $V_\phi(x, y_{<t})$ is a second, learned network
trained alongside the policy. State what quantity it is an estimate of. The terminal reward $r(x,y)$ from the reward
model or verifier is available only once the full response is complete, and a per-token KL penalty against $\pi_{\text{ref}}$
(scaled by $\beta$) is commonly charged at every step; give the resulting per-token quantity $\tilde r_t$ that
combines the two, and, writing the TD (temporal-difference) error as $\delta_t = \tilde r_t + \gamma V_\phi(x, y_{\le t}) - V_\phi(x, y_{<t})$
for a discount $\gamma \in (0, 1]$ with $V_\phi(x, y_{\le n}) \doteq 0$ at the terminal step, write down generalised
advantage estimation (GAE): the advantage $A_t^{\mathrm{GAE}(\gamma,\lambda)}$, for a trace-decay parameter $\lambda \in [0, 1]$,
as a $(\gamma\lambda)$-geometrically-weighted sum of $\delta_t, \delta_{t+1}, \dots$. Derive the two limiting cases
$\lambda = 0$ and $\lambda = 1$. What does using a critic cost, concretely?

### GRPO

**Q4.** Group relative policy optimisation (GRPO) samples, for one prompt $x$, a group of $G$ responses $y_1,\dots,y_G \sim \pi_{\theta_{\text{old}}}(\cdot\mid x)$
and scores each with a single scalar reward $r_i = r(x,y_i)$ (for instance the output of a verifier: $1$ if $y_i$
passes a test or matches a reference answer, $0$ otherwise). Every token of response $i$ receives the same advantage

$$A_i = \frac{r_i - \bar r}{\operatorname{std}(r) + \delta}, \qquad \bar r = \frac1G\sum_{j=1}^G r_j, \qquad \operatorname{std}(r) = \sqrt{\frac{1}{G-1}\sum_{j=1}^G (r_j-\bar r)^2},$$

for a small constant $\delta > 0$. Explain why this needs no critic, why it particularly suits verifiable rewards
(unit tests, exact-match answers), and what happens to every $A_i$ of a group whose $G$ rewards are all equal.
State two known biases of this formulation: one from normalising each response's contribution to the loss by its
own token length, and one from dividing by the group's reward standard deviation. Compute $A_i$ for a group with
rewards $(1,0,1,0)$ and for a group with rewards $(1,1,1,1)$.

**Q5.** Compare PPO and GRPO in a table: what each uses as a critic, how each computes the advantage, memory and
variance, what each needs from the reward, and when you would choose one over the other.

### KL and preferences

**Q6.** Let $y\sim\pi_\theta$ and $u = \pi_{\text{ref}}(y)/\pi_\theta(y)$. The estimators $k_1 = -\log u$ and $k_3 = (u-1) - \log u$
are both proposed as ways to estimate $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \mathbb E_{y\sim\pi_\theta}[\log(\pi_\theta(y)/\pi_{\text{ref}}(y))]$
from samples. Show that both are unbiased, show that $k_3 \ge 0$ for every sample (not just in expectation), and
explain why $k_3$ is preferred in practice.

**Q7.** Direct Preference Optimisation (DPO). Starting from $\max_{\pi} \mathbb E_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x))$,
maximised over all conditional distributions $\pi(\cdot\mid x)$ for each fixed $x$, derive the closed-form maximiser
$\pi^*(y\mid x) = \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)/Z(x)$ and give $Z(x)$. Solve this relationship for
$r(x,y)$ in terms of $\pi^*$, $\pi_{\text{ref}}$ and $Z(x)$, then substitute into the Bradley–Terry model of a
preference between a winning and a losing response, $P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$
for the logistic function $\sigma(z) = 1/(1+e^{-z})$, to obtain the DPO loss over a preference dataset of triples
$(x,y_w,y_l)$, replacing $\pi^*$ by the trainable $\pi_\theta$. Explain why $Z(x)$ cancels on the way, and state
what data DPO needs to train on.

**Q8.** Reward hacking and over-optimisation. During RLHF training against a learned reward model, describe the
symptoms you would look for to detect that the policy is over-optimising the reward model rather than genuinely
improving, and state concrete mitigations.

### On-policy and off-policy data

**Q9.** A training update is *on-policy* when the samples it learns from come from the same policy it is
updating, and *off-policy* when they come from a different one. Formalise this with a *behaviour policy* $\mu$,
the distribution the training data was actually sampled from, and a *target policy* $\pi$, the distribution
whose expected reward $\mathbb E_{y\sim\pi}[r(y)]$ is being evaluated or improved. Place each of the following on
the spectrum from strictly on-policy to strictly off-policy, and justify the placement: REINFORCE (Q1's
policy-gradient estimator), which samples a fresh $y\sim\pi_\theta$ immediately before every gradient step; PPO
and GRPO, which take several gradient steps against one batch of responses sampled once from
$\pi_{\theta_{\text{old}}}$; Q-learning with a replay buffer of transitions collected under many past policies;
and DPO, trained on a fixed, pre-collected preference dataset.

Write the importance-sampling estimator of $\mathbb E_{y\sim\pi}[r(y)]$ built from an i.i.d. sample $y\sim\mu$,
and show it is unbiased whenever $\mu(y) > 0$ at every $y$ with $\pi(y)\,r(y) \ne 0$. Now model a $T$-token
response as $T$ tokens drawn i.i.d. from a fixed vocabulary — dropping the autoregressive conditioning on
history, to isolate what sequence length alone does to the estimator — and write $w_t = \pi(y_t)/\mu(y_t)$ for
the ratio at token $t$ and $W = \prod_{t=1}^T w_t$ for the resulting sequence-level weight. Using the
chi-squared divergence between the per-token laws, $\chi^2(\pi\,\|\,\mu) = \sum_y (\pi(y)-\mu(y))^2/\mu(y)$,
derive $\operatorname{Var}_\mu(W)$ in closed form as a function of $T$ and $\chi^2(\pi\,\|\,\mu)$, and explain
why it grows exponentially in $T$ even when $\pi$ and $\mu$ are close at every single token. State how applying
PPO's clipped surrogate (Q2) per token, a hard cap directly on the importance weight, and a KL penalty holding
$\pi$ close to $\mu$ each control this variance, and at what cost. Finally, name two ways an LLM post-training
system ends up with $\mu \ne \pi$ without that being a deliberate design choice.

## Reference solution

<details>
<summary>Show the reference solution</summary>

Two things worth settling aloud before answering: every KL term below is the forward $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$
— the direction a per-token penalty added to the reward actually estimates, not the reverse — and GRPO's reward
$r_i$ is one scalar per full response, not per token, since a group-relative advantage only makes sense when there
is exactly one comparable number per sampled response.

### Policy gradients

**Q1.** $\nabla_\theta J(\theta) = \nabla_\theta \sum_y \pi_\theta(y\mid x)\, r(x,y) = \sum_y r(x,y)\, \nabla_\theta \pi_\theta(y\mid x)$,
since $r$ does not depend on $\theta$. Using $\nabla_\theta \log \pi_\theta(y\mid x) = \nabla_\theta \pi_\theta(y\mid x)/\pi_\theta(y\mid x)$
wherever $\pi_\theta(y\mid x) > 0$ (the ordinary chain rule, rearranged — the whole log-derivative trick), $\nabla_\theta \pi_\theta(y\mid x) = \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x)$,
so

$$\nabla_\theta J(\theta) = \sum_y \pi_\theta(y\mid x)\, r(x,y)\, \nabla_\theta \log \pi_\theta(y\mid x) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\,\nabla_\theta \log \pi_\theta(y\mid x)\big].$$

Token by token: $\log \pi_\theta(y\mid x) = \sum_{t=1}^n \log \pi_\theta(y_t\mid x,y_{<t})$ directly from the autoregressive
factorisation, so by linearity of the gradient

$$\nabla_\theta J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\Big[r(x,y)\sum_{t=1}^n \nabla_\theta \log \pi_\theta(y_t\mid x,y_{<t})\Big].$$

For the baseline, take any $b$ with $\nabla_\theta b = 0$ that does not depend on $y$ (it may depend on $x$). Since
every probability distribution sums to $1$ for every $\theta$, $\nabla_\theta \sum_y \pi_\theta(y\mid x) = \nabla_\theta 1 = 0$;
expanding the left side the same way as above,

$$0 = \nabla_\theta \sum_y \pi_\theta(y\mid x) = \sum_y \pi_\theta(y\mid x)\,\nabla_\theta \log \pi_\theta(y\mid x) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[\nabla_\theta \log \pi_\theta(y\mid x)\big],$$

so $b\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[\nabla_\theta \log \pi_\theta(y\mid x)] = 0$ too, and subtracting
it from the reward changes nothing:

$$\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[(r(x,y)-b)\,\nabla_\theta \log \pi_\theta(y\mid x)\big] = \nabla_\theta J(\theta) - 0 = \nabla_\theta J(\theta).$$

A baseline never changes what the estimator converges to; a well-chosen $b(x)$ close to $\mathbb E_{y\sim\pi_\theta}[r(x,y)\mid x]$
changes how noisy a single sample of it is, by removing most of the reward's absolute scale from the term being
multiplied by the score — the sole justification for a learned critic (Q3) or, without training one, a sampled
group mean (Q4).

### PPO

**Q2.** Clipping keeps the gradient of a move that has already gone far enough in its favoured direction at exactly
$0$, while a mistake is always still fully corrected.

| Region of $\rho$ | Sign of $A$ | $L^{\text{clip}}$ | $\partial L^{\text{clip}}/\partial\rho$ |
| --- | --- | --- | --- |
| $[1-\epsilon,\,1+\epsilon]$ | either | $\rho A$ | $A$ |
| $>1+\epsilon$ | $A>0$ | $(1+\epsilon)A$ | $0$ |
| $>1+\epsilon$ | $A<0$ | $\rho A$ | $A$ |
| $<1-\epsilon$ | $A>0$ | $\rho A$ | $A$ |
| $<1-\epsilon$ | $A<0$ | $(1-\epsilon)A$ | $0$ |

(Inside the interval the unclipped and clipped branches of $\min$ coincide for either sign of $A$, so it is one
case rather than two.) The gradient is exactly $0$ only in the two rows where $\rho$ has already moved outside
the interval in the direction the sign of $A$ favours: past $1+\epsilon$ with $A>0$ (the probability of an already-good
response has already been pushed up past the trust region) or past $1-\epsilon$ with $A<0$ (an already-bad response's
probability has already been pushed down past it). In every other row — inside the interval, or outside it in the
direction $A$ disfavours, i.e. an actual mistake — the gradient is the plain $A$, so clipping never blocks a correction,
only the continuation of a move that has already been exploited as far as the trust region allows.

For $\epsilon=0.2$ (interval $[0.8,1.2]$): $(\rho,A)=(1.5,2)$ is above with $A>0$ (clipped): $\min(3.0,\,2.4)=2.4$.
$(\rho,A)=(0.5,2)$ is below with $A>0$ (open): $\min(1.0,\,1.6)=1.0$. $(\rho,A)=(1.5,-2)$ is above with $A<0$ (open):
$\min(-3.0,\,-2.4)=-3.0$. $(\rho,A)=(0.5,-2)$ is below with $A<0$ (clipped): $\min(-1.0,\,-1.6)=-1.6$.

**Q3.** $V_\phi(x,y_{<t})$ estimates the expected future (KL-shaped) return from having generated the prefix $y_{<t}$
— the sum of everything from step $t$ to the end of the response. The terminal reward attaches to the last token,
and a per-token KL penalty is folded directly into a single quantity charged at every step:

$$\tilde r_t = -\beta \log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{ref}}(y_t\mid x,y_{<t})} + \mathbb 1[t=n]\, r(x,y),$$

so every token pays a small KL cost and only the last also receives the terminal reward. Writing $V_k := V_\phi(x, y_{\le k})$
for brevity, GAE is

$$A_t^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{n-t} (\gamma\lambda)^l\, \delta_{t+l}.$$

At $\lambda=0$ only the $l=0$ term survives, $A_t = \delta_t$: the one-step TD residual, low variance but as biased
as $V_\phi$ is wrong. At $\lambda=1$,

$$\sum_{l=0}^{n-t}\gamma^l \delta_{t+l} = \sum_{l=0}^{n-t}\gamma^l\big(\tilde r_{t+l} + \gamma V_{t+l} - V_{t+l-1}\big) = \sum_{l=0}^{n-t}\gamma^l \tilde r_{t+l} + \sum_{l=0}^{n-t}\big(\gamma^{l+1}V_{t+l} - \gamma^l V_{t+l-1}\big),$$

and the second sum telescopes: consecutive terms cancel, leaving only $-V_{t-1}$ from $l=0$ and $\gamma^{\,n-t+1}V_n = 0$
from the last term (the terminal value is pinned to $0$), so $A_t = \sum_{l=0}^{n-t}\gamma^l \tilde r_{t+l} - V_{t-1}$:
the (KL-shaped) Monte Carlo return-to-go minus the baseline, unbiased regardless of $V_\phi$'s quality, but with
the full variance of a sampled return. Intermediate $\lambda$ trades one for the other.

Concretely, the critic costs two things: it is a second network trained alongside the policy — for an LLM this
is usually a copy of the same transformer backbone with a scalar head, so it roughly doubles the parameters and
activation memory kept during training — and it must itself be fit by regression against a target that is still
moving as training proceeds, so a miscalibrated $V_\phi$ biases every advantage built from it, exactly the bias
the $\lambda=1$ limit above removes at the cost of variance.

### GRPO

**Q4.** No critic is needed because $\bar r$, the group's own empirical mean over $G$ fresh samples of this exact
prompt, is already a direct Monte Carlo estimate of $\mathbb E_{y\sim\pi_\theta}[r(x,y)\mid x]$ — precisely the
quantity a learned baseline $V_\phi(x)$ would otherwise be trained to approximate (Q1) — so it removes the reward's
scale from the estimator without a second network; dividing by $\operatorname{std}(r)+\delta$ further rescales
the result so prompts whose rewards happen to be spread out differently contribute advantages of comparable size.
This suits verifiable rewards particularly well: a checker (unit tests, exact-match grading) is cheap and deterministic
to re-run, so sampling $G$ full responses to the same prompt — the one extra cost GRPO pays for dropping the critic
— is affordable in a way it would not be if scoring a response required running an expensive learned reward model
$G$ times as well.

When a group's $G$ rewards are all equal, $\operatorname{std}(r) = 0$ exactly, and $\delta>0$ turns every $A_i$
into $0/(0+\delta)=0$: that prompt contributes no gradient at all on that step, whether every response was right
or every response was wrong.

Two biases of this formulation. First, normalising each response's summed token loss by its own length $|y_i|$
(so the per-token coefficient is $A_i/|y_i|$) gives every token of a long response a smaller pull than every token
of a short one for the same $A_i$: a response of length $10$ with $A_i=-1$ gets a per-token coefficient of $-0.1$,
the same $A_i=-1$ spread over $40$ tokens gets $-0.025$, four times weaker — so a wrong response is punished less
per token simply for being longer, biasing training toward longer wrong answers over time. Second, dividing by
$\operatorname{std}(r)$ reweights prompts by how far their success rate is from $50\%$: for a Bernoulli-like reward
with success probability $p$, $\operatorname{std}(r)\approx \sqrt{p(1-p)}$ is largest at $p=0.5$ and shrinks as
$p\to0$ or $p\to1$, so a group with a $50\%$ success rate (of $100$ responses, $\operatorname{std}(r)\approx0.50$)
turns the same reward gap of $1$ into an advantage-magnitude gap of about $2.0$, while a $10\%$-or-$90\%$-success
group ($\operatorname{std}(r)\approx0.30$) turns it into about $3.3$, and a $2\%$-success group ($\operatorname{std}(r)\approx0.14$)
into about $7.1$ — more than three times the balanced group's step, for prompts whose only distinguishing feature
was being easy or hard.

For rewards $(1,0,1,0)$: $\bar r=0.5$, $\operatorname{std}(r)=\sqrt{1/3}\approx0.5774$, so $A\approx(0.8660,-0.8660,0.8660,-0.8660)$.
For rewards $(1,1,1,1)$: $A=(0,0,0,0)$.

**Q5.** The two land on different sides of a memory, variance and critic-training trade-off:

| | PPO | GRPO |
| --- | --- | --- |
| Critic | Learned $V_\phi(x,y_{<t})$, trained jointly by regression | None |
| Advantage | GAE$(\gamma,\lambda)$ from TD errors against $V_\phi$ (Q3) | Group-normalised $(r_i-\bar r)/(\operatorname{std}(r)+\delta)$ from $G$ rollouts of the same prompt (Q4) |
| Memory / compute | Two networks trained together, often both full size | One network, but $G$ full rollouts per prompt before every update |
| Variance | Lower when $V_\phi$ tracks the true value; biased when it does not | Falls as $1/\sqrt G$; no learned-function bias, but see Q4's two biases |
| What it needs from the reward | Any scalar, at any point in the response — dense per-token shaping is fine | Cheap and repeatable, so sampling $G$ responses per prompt is affordable; ideally verifiable |
| When to choose | Reward is expensive to score many times per prompt, dense shaping is available, or a second network is not the bottleneck | Reward is cheap and verifiable, or critic memory and training instability are the bottleneck |

### KL and preferences

**Q6.** Both estimate the same quantity exactly in expectation; $k_3$ is preferred because it is also non-negative
on every individual sample and has much lower variance where it matters (near $\pi_\theta \approx \pi_{\text{ref}}$).

Unbiasedness of $k_1$: by definition $u=\pi_{\text{ref}}(y)/\pi_\theta(y)$, so $-\log u = \log\big(\pi_\theta(y)/\pi_{\text{ref}}(y)\big)$
exactly, and

$$\mathbb E_{y\sim\pi_\theta}[k_1] = \mathbb E_{y\sim\pi_\theta}\Big[\log\frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}\Big] = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$$

by the definition of the KL divergence — no approximation is involved.

Unbiasedness of $k_3$: first, $\mathbb E_{y\sim\pi_\theta}[u] = \sum_y \pi_\theta(y) \cdot \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} = \sum_y \pi_{\text{ref}}(y) = 1$,
since $\pi_{\text{ref}}$ is itself a probability distribution over the same support. So $\mathbb E[u]-1=0$, and

$$\mathbb E_{y\sim\pi_\theta}[k_3] = \mathbb E[u] - 1 - \mathbb E[\log u] = -\mathbb E[\log u] = \mathbb E[-\log u] = \mathbb E[k_1] = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}).$$

Non-negativity: let $f(u) = u - 1 - \log u$ for $u > 0$, so $k_3 = f(u)$. $f'(u) = 1 - 1/u$ is negative for $u<1$
and positive for $u>1$, so $u=1$ is the unique global minimiser of $f$, with $f(1)=0$; hence $f(u) \ge 0$ for every
$u>0$, i.e. $k_3 \ge 0$ on every sample, with equality only where $\pi_\theta(y)=\pi_{\text{ref}}(y)$.

Why $k_3$ is preferred: write $u = 1+\varepsilon$ for the regime that matters most — a policy kept close to its
reference by the KL penalty itself, so $\varepsilon$ is typically small. Since $\log(1+\varepsilon) = \varepsilon - \varepsilon^2/2 + O(\varepsilon^3)$,

$$k_1 = -\log u = -\varepsilon + O(\varepsilon^2), \qquad k_3 = (u-1)-\log u = \varepsilon - \big(\varepsilon - \tfrac{\varepsilon^2}{2} + O(\varepsilon^3)\big) = \tfrac{\varepsilon^2}{2} + O(\varepsilon^3),$$

so $k_1$'s per-sample fluctuations are first order in $\varepsilon$ and can take either sign, while $k_3$'s are
second order, always non-negative, and a full order smaller — e.g. at $u=1.01$, $k_1\approx-0.00995$ against $k_3\approx4.97\times10^{-5}$,
two orders of magnitude smaller. A running average of $k_1$ can read negative on a finite batch even though the
true KL cannot be; $k_3$ never does, and its lower variance means fewer samples are needed to pin the value down.

**Q7.** For a fixed $x$, treat $\pi(\cdot\mid x)$ as a free probability distribution over responses and maximise

$$J[\pi] = \sum_y \pi(y\mid x)\,r(x,y) - \beta \sum_y \pi(y\mid x) \log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$$

subject to $\sum_y \pi(y\mid x)=1$. Introducing a Lagrange multiplier $\mu$ for the constraint and differentiating
with respect to each $\pi(y\mid x)$ separately,

$$\frac{\partial}{\partial \pi(y\mid x)}\Big[J[\pi] - \mu\Big(\sum_y \pi(y\mid x)-1\Big)\Big] = r(x,y) - \beta\Big(\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+1\Big) - \mu = 0,$$

so $\log\big(\pi(y\mid x)/\pi_{\text{ref}}(y\mid x)\big) = \big(r(x,y)-\mu\big)/\beta - 1$, giving $\pi(y\mid x) = \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)\exp(-\mu/\beta-1)$.
The last factor does not depend on $y$; call it $1/Z(x)$ and fix it by the normalisation constraint:

$$\pi^*(y\mid x) = \frac{\pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big)}{Z(x)}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big).$$

$J$ is strictly concave in $\pi$ (the KL term is a strictly convex function of $\pi$, and the reward term is linear),
so this stationary point is the unique global maximiser.

Solving for the reward: $r(x,y) = \beta \log\dfrac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$.
Substituting into the Bradley–Terry model for a pair $(y_w,y_l)$ (the response judged better and the one judged
worse for the same $x$), $P(y_w \succ y_l\mid x) = \sigma\big(r(x,y_w)-r(x,y_l)\big)$, the reward enters only through
the difference $r(x,y_w)-r(x,y_l)$, and $\beta\log Z(x)$ is exactly the same additive term in $r(x,y_w)$ and in
$r(x,y_l)$ (it depends on $x$ alone, not on $y$), so it cancels without ever being computed:

$$r(x,y_w)-r(x,y_l) = \beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}.$$

Replacing the (unknown) optimum $\pi^*$ by the trainable $\pi_\theta$ turns maximum-likelihood fitting of this
Bradley–Terry model on a preference dataset $\mathcal D = \{(x,y_w,y_l)\}$ into the DPO loss

$$\mathcal L_{\text{DPO}}(\theta) = -\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

DPO needs only a static dataset of preference triples $(x,y_w,y_l)$ — $y_w,y_l$ can come from anywhere, with no
on-policy sampling and no reward model to train — plus a frozen copy of $\pi_{\text{ref}}$ to compute both log-ratios.

**Q8.** The tell-tale symptom is a widening gap between the proxy and the truth: the reward model's own score keeps
climbing while a separate, held-out measurement — human preference win-rate, or accuracy on withheld verifiable
tasks — plateaus and then falls, and responses tend to grow longer, since length is an easy way to look better
to many reward models without being better. Other tells include increasingly repetitive or degenerate phrasing,
sycophancy, and outputs tuned to specific quirks of the reward model rather than to the task.

Mitigations, roughly in order of how directly they attack the cause: raise $\beta$, trading achieved reward for
staying in the region around $\pi_{\text{ref}}$ that the reward model is best calibrated on; use an ensemble of
independently trained reward models (score with their minimum, or penalise their disagreement) so that exploiting
one model's specific blind spot is no longer free, or replace the learned reward with a verifiable one wherever
the task allows it (Q4), removing the attack surface entirely; add an explicit length control or penalty so the
policy cannot win purely by writing more, which directly targets the same length-inflation mechanism as GRPO's
own length-normalisation bias (Q4); and pick the training checkpoint by early stopping against a held-out judge
that is not the training-time reward model, tracked over the whole run, rather than trusting the proxy reward at
the final step.

### On-policy and off-policy data

**Q9.** On-policy means $\mu=\pi$: the policy being updated produced the data; off-policy means $\mu\ne\pi$.
REINFORCE is strictly on-policy: each step samples a fresh $y\sim\pi_\theta$ immediately before the update, so
$\mu=\pi=\pi_\theta$ exactly. PPO and GRPO are on-policy by design but mildly off-policy in execution: the
batch is sampled once from $\mu=\pi_{\theta_{\text{old}}}$, and after the first of several gradient steps
$\pi=\pi_\theta$ has already moved past it — why $\rho\ne1$ in Q2 and why clipping exists. Q-learning with a
replay buffer is fully off-policy by design: it learns a target policy $\pi$ from transitions collected under
whatever exploratory $\mu$ was running when the buffer was filled, possibly many updates earlier. DPO is
off-policy in the strongest sense here: $\mu$ is whatever produced the fixed, pre-collected preference
dataset, never resampled, while $\pi=\pi_\theta$ changes throughout with no importance correction at all.

The importance-sampling estimator $\hat J=(\pi(y)/\mu(y))\,r(y)$ of $\mathbb E_{y\sim\pi}[r(y)]$ from
$y\sim\mu$ is unbiased whenever $\mu(y)>0$ at every $y$ with $\pi(y)r(y)\ne0$:

$$\mathbb E_{y\sim\mu}\Big[\frac{\pi(y)}{\mu(y)}r(y)\Big] = \sum_{y:\mu(y)>0}\pi(y)r(y) = \mathbb E_{y\sim\pi}[r(y)],$$

the terms with $\mu(y)=0$ all being $0$ under that hypothesis.

Modelling a $T$-token response as $T$ i.i.d. tokens, $w_t=\pi(y_t)/\mu(y_t)$, $W=\prod_{t=1}^T w_t$:
$\mathbb E_\mu[w_t]=1$ at every position gives $\mathbb E_\mu[W]=1$ by independence, and likewise
$\mathbb E_\mu[W^2]=\big(\mathbb E_\mu[w_t^2]\big)^T$. Expanding $\chi^2(\pi\,\|\,\mu)=\sum_y\pi(y)^2/\mu(y)-1$
from its definition identifies $\mathbb E_\mu[w_t^2]=1+\chi^2(\pi\,\|\,\mu)$, so

$$\mathbb E_\mu[W^2]=\big(1+\chi^2(\pi\,\|\,\mu)\big)^T,\qquad \operatorname{Var}_\mu(W)=\big(1+\chi^2(\pi\,\|\,\mu)\big)^T-1.$$

With $\kappa=\chi^2(\pi\,\|\,\mu)>0$ possibly tiny per token, this is exponential in $T$ — doubling $T$
squares $(1+\kappa)^T$ — so a mismatch invisible at one token can dominate every other noise source by
$T=100$.

Three controls, each at a cost. PPO's clipped surrogate (Q2), applied per token, pairs each $w_t$ with its
own advantage instead of forming $W$, so nothing is raised to the $T$-th power; the cost is bias, since
per-token ratios ignore how $\pi$ changing also changes the prefix distribution each token conditions on, and
a ratio already past $1\pm\epsilon$ has its gradient discarded outright. Capping the weight,
$\tilde W=\min(W,c)$, bounds $\operatorname{Var}_\mu(\tilde W)\le c^2$ pointwise; the cost is
$\mathbb E_\mu[\tilde W]<1$ whenever $\mu$ puts mass above $c$, growing with that mass. A KL penalty holding
$\pi$ close to $\mu$ shrinks $\chi^2(\pi\,\|\,\mu)$ too, since $\mathrm{KL}\approx\tfrac12\chi^2$ near
$\pi=\mu$; the cost is that $\pi$ is kept from moving as far from $\mu$ as the reward alone would push it.

Two ways $\mu\ne\pi$ arises unintentionally. In asynchronous training, rollout workers generate with weights
lagging the trainer's, so a batch sampled from a snapshot trains against parameters that have already moved
past it, the gap growing with how far generation lags training. And the inference engine generating rollouts
computes slightly different token probabilities than the trainer would for the same weights (different
batching, kernels or precision), so the snapshot used as $\mu$ only approximates what the tokens were
actually drawn from — a gap present even with one gradient step and perfect synchrony.

<details>
<summary>Checks (runnable)</summary>

```python
import math

import numpy as np

rng = np.random.default_rng(0)


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---- Q1: the score-function identity, token by token, and the effect of a baseline
# A 2-token response y = (y1, y2) from a 3-way vocabulary, autoregressive: y2's law depends on y1.
THETA1 = np.array([0.2, -0.5, 0.9])               # logits of pi(y1)
THETA2 = np.array([[0.4, 0.1, -0.3],               # logits of pi(y2 | y1 = 0)
                    [-0.2, 0.6, 0.0],              # logits of pi(y2 | y1 = 1)
                    [0.3, -0.4, 0.5]])             # logits of pi(y2 | y1 = 2)
REWARD = np.array([[1.0, -2.0, 0.5],
                    [0.3, 1.5, -1.0],
                    [-0.5, 0.8, 2.0]])             # r(y1, y2): one fixed number per outcome


def expected_reward(theta1, theta2):
    p1 = softmax(theta1)
    total = 0.0
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        total += p1[y1] * float(p2 @ REWARD[y1])
    return total


def score1(theta1, y1):                            # grad_theta1 log pi(y1) = onehot(y1) - pi(y1)
    p1 = softmax(theta1)
    oh = np.zeros(3); oh[y1] = 1.0
    return oh - p1


def score2_row(theta2_row, y2):                     # grad_(theta2[y1]) log pi(y2|y1) = onehot(y2) - pi(.|y1)
    p2 = softmax(theta2_row)
    oh = np.zeros(3); oh[y2] = 1.0
    return oh - p2


def reinforce_grad_exact(theta1, theta2, reward, baseline=0.0):
    """E_{y~pi}[(r(y) - baseline) * grad log pi(y)], by enumeration; grad log pi(y) is summed token by token."""
    p1 = softmax(theta1)
    g1 = np.zeros(3)
    g2 = np.zeros((3, 3))
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        for y2 in range(3):
            p = p1[y1] * p2[y2]
            adv = reward[y1, y2] - baseline
            g1 += p * adv * score1(theta1, y1)
            g2[y1] += p * adv * score2_row(theta2[y1], y2)
    return g1, g2


def fd_grad(theta1, theta2, h=1e-6):
    """Independent brute force: central finite differences directly on expected_reward, no score-function formula."""
    g1 = np.zeros(3)
    for i in range(3):
        tp, tm = theta1.copy(), theta1.copy()
        tp[i] += h; tm[i] -= h
        g1[i] = (expected_reward(tp, theta2) - expected_reward(tm, theta2)) / (2 * h)
    g2 = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            tp, tm = theta2.copy(), theta2.copy()
            tp[r, c] += h; tm[r, c] -= h
            g2[r, c] = (expected_reward(theta1, tp) - expected_reward(theta1, tm)) / (2 * h)
    return g1, g2


g1_exact, g2_exact = reinforce_grad_exact(THETA1, THETA2, REWARD)
g1_fd, g2_fd = fd_grad(THETA1, THETA2)
assert np.allclose(g1_exact, g1_fd, atol=1e-6)
assert np.allclose(g2_exact, g2_fd, atol=1e-6)

N = 200_000                                         # Monte Carlo agreement (vectorised)
p1 = softmax(THETA1)
y1_samples = rng.choice(3, size=N, p=p1)
y2_samples = np.empty(N, dtype=int)
for y1 in range(3):
    mask = y1_samples == y1
    k = int(mask.sum())
    if k:
        y2_samples[mask] = rng.choice(3, size=k, p=softmax(THETA2[y1]))
r_samples = REWARD[y1_samples, y2_samples]
score1_samples = np.eye(3)[y1_samples] - p1[None, :]
mc_g1 = (r_samples[:, None] * score1_samples).mean(axis=0)
mc_g2 = np.zeros((3, 3))
for y1 in range(3):
    mask = y1_samples == y1
    p2 = softmax(THETA2[y1])
    score2_samples = np.eye(3)[y2_samples[mask]] - p2[None, :]
    mc_g2[y1] = (r_samples[mask][:, None] * score2_samples).sum(axis=0) / N
assert np.allclose(mc_g1, g1_exact, atol=0.02)
assert np.allclose(mc_g2, g2_exact, atol=0.05)

mean_reward = expected_reward(THETA1, THETA2)                       # baseline: mean unaffected, variance changes
BASELINES = [0.0, -5.0, mean_reward, 10.0]
for b in BASELINES:
    g1_b, g2_b = reinforce_grad_exact(THETA1, THETA2, REWARD, baseline=b)
    assert np.allclose(g1_b, g1_exact, atol=1e-10)
    assert np.allclose(g2_b, g2_exact, atol=1e-10)


def estimator_second_moment(theta1, theta2, reward, baseline):
    """E[|| one-sample score-function estimator ||^2], enumerated exactly over all 9 outcomes."""
    p1 = softmax(theta1)
    total = 0.0
    for y1 in range(3):
        p2 = softmax(theta2[y1])
        for y2 in range(3):
            p = p1[y1] * p2[y2]
            adv = reward[y1, y2] - baseline
            s1, s2 = score1(theta1, y1), score2_row(theta2[y1], y2)
            total += p * (adv ** 2) * (float(s1 @ s1) + float(s2 @ s2))
    return total


mean_sq_norm = float(g1_exact @ g1_exact) + float((g2_exact * g2_exact).sum())
variances = {b: estimator_second_moment(THETA1, THETA2, REWARD, b) - mean_sq_norm for b in BASELINES}
assert variances[mean_reward] < variances[0.0] < variances[10.0]

# ---- Q2: PPO's clipped surrogate: five distinguishable cases, and finite differences in rho
EPS = 0.2


def ppo_surrogate(rho, A, eps=EPS):
    clipped = min(max(rho, 1 - eps), 1 + eps)
    return min(rho * A, clipped * A)


cases = [(1.5, 2.0), (0.5, 2.0), (1.5, -2.0), (0.5, -2.0)]
surrogate_values = [ppo_surrogate(rho, A) for rho, A in cases]
assert np.allclose(surrogate_values, [2.4, 1.0, -3.0, -1.6])

h = 1e-6
ppo_grads = [(ppo_surrogate(rho + h, A) - ppo_surrogate(rho - h, A)) / (2 * h) for rho, A in cases]
assert np.allclose(ppo_grads, [0.0, 2.0, -2.0, 0.0], atol=1e-4)

for rho in (0.85, 1.0, 1.15):                        # the fifth case: inside the interval, either sign of A
    for A in (3.0, -3.0):
        g = (ppo_surrogate(rho + h, A) - ppo_surrogate(rho - h, A)) / (2 * h)
        assert math.isclose(g, A, rel_tol=1e-4)
        assert math.isclose(ppo_surrogate(rho, A), rho * A, rel_tol=1e-9)

# ---- Q3: the per-token KL-shaped reward and GAE(gamma, lambda)
GAMMA, LAM = 0.97, 0.9
EPISODE_LEN = 6
V = rng.normal(size=EPISODE_LEN + 1)
V[EPISODE_LEN] = 0.0                       # NOTE: the terminal "value after the last token" is pinned to 0
r_tilde = rng.normal(size=EPISODE_LEN) * 0.1


def gae_recursive(r_tilde, V, gamma, lam):
    delta = r_tilde + gamma * V[1:] - V[:-1]
    steps = len(r_tilde)
    A = np.zeros(steps)
    running = 0.0
    for t in reversed(range(steps)):
        running = delta[t] + gamma * lam * running
        A[t] = running
    return A, delta


def gae_geometric_sum(r_tilde, V, gamma, lam):
    """Same recursion, unrolled explicitly as a (gamma*lam)^l-weighted sum of future deltas: an independent path."""
    delta = r_tilde + gamma * V[1:] - V[:-1]
    steps = len(r_tilde)
    A = np.zeros(steps)
    for t in range(steps):
        A[t] = sum((gamma * lam) ** l * delta[t + l] for l in range(steps - t))
    return A


A_rec, delta = gae_recursive(r_tilde, V, GAMMA, LAM)
A_sum = gae_geometric_sum(r_tilde, V, GAMMA, LAM)
assert np.allclose(A_rec, A_sum, atol=1e-10)

A_lam0, _ = gae_recursive(r_tilde, V, GAMMA, 0.0)               # lambda = 0: one-step TD residual
assert np.allclose(A_lam0, delta, atol=1e-10)

A_lam1, _ = gae_recursive(r_tilde, V, GAMMA, 1.0)                # lambda = 1: telescopes to (return-to-go - V)
return_to_go = np.array([sum(GAMMA ** l * r_tilde[t + l] for l in range(EPISODE_LEN - t))
                         for t in range(EPISODE_LEN)])
assert np.allclose(A_lam1, return_to_go - V[:-1], atol=1e-8)

# the per-token KL-shaped reward itself, worked for a 3-token response
TOK_PI_THETA = np.array([0.5, 0.25, 0.8])                   # pi_theta(y_t | x, y_<t) for t = 1, 2, 3
TOK_PI_REF = np.array([0.4, 0.3, 0.6])                      # pi_ref(y_t | x, y_<t), same tokens
BETA_TOK, TERMINAL_R = 0.1, 2.0
tilde_r_worked = -BETA_TOK * np.log(TOK_PI_THETA / TOK_PI_REF)
tilde_r_worked[-1] += TERMINAL_R                            # NOTE: only the terminal token gets the episode reward
assert math.isclose(tilde_r_worked[0], -BETA_TOK * math.log(0.5 / 0.4), rel_tol=1e-9)
assert math.isclose(tilde_r_worked[1], -BETA_TOK * math.log(0.25 / 0.3), rel_tol=1e-9)
assert math.isclose(tilde_r_worked[2], -BETA_TOK * math.log(0.8 / 0.6) + TERMINAL_R, rel_tol=1e-9)
# summed KL cost is the log of a product of ratios: an independent code path to the elementwise formula above
assert math.isclose(tilde_r_worked.sum() - TERMINAL_R, -BETA_TOK * math.log(float(np.prod(TOK_PI_THETA / TOK_PI_REF))), rel_tol=1e-9)

# ---- Q4: GRPO's group-relative advantage
ADV_DELTA = 1e-6


def grpo_advantages(rewards, delta=ADV_DELTA):
    rewards = np.asarray(rewards, dtype=float)
    mean = rewards.mean()
    std = rewards.std(ddof=1) if len(rewards) > 1 else 0.0  # NOTE: ddof=1 (sample std); numpy's default ddof=0 understates it
    return (rewards - mean) / (std + delta)


adv_mixed = grpo_advantages([1, 0, 1, 0])
expected_mixed = (np.array([1, 0, 1, 0]) - 0.5) / (math.sqrt(1 / 3) + ADV_DELTA)
assert np.allclose(adv_mixed, expected_mixed, atol=1e-6)
assert np.allclose(adv_mixed, [0.8660, -0.8660, 0.8660, -0.8660], atol=1e-3)

adv_constant = grpo_advantages([1, 1, 1, 1])
assert np.allclose(adv_constant, [0.0, 0.0, 0.0, 0.0], atol=1e-9)

for _ in range(200):                                # the group mean of the advantage is 0 whenever there is spread
    r = rng.integers(0, 2, size=8).astype(float)
    if r.std() > 0:
        assert abs(grpo_advantages(r).mean()) < 1e-8

GROUP = 100                                          # a group's success rate, exactly p * GROUP successes out of GROUP
group_stds = {}
for p in (0.5, 0.1, 0.9, 0.02):
    r = np.array([1.0] * round(p * GROUP) + [0.0] * (GROUP - round(p * GROUP)))
    group_stds[p] = r.std(ddof=1)
gaps = {p: 1.0 / (group_stds[p] + ADV_DELTA) for p in group_stds}      # |advantage| between a success and a failure
assert gaps[0.5] < gaps[0.1] < gaps[0.02]
assert gaps[0.5] < gaps[0.9]
for p, std_approx, gap_approx in [(0.5, 0.50, 2.0), (0.1, 0.30, 3.3), (0.02, 0.14, 7.1)]:  # the numbers quoted above
    assert math.isclose(group_stds[p], std_approx, abs_tol=0.01)
    assert math.isclose(gaps[p], gap_approx, rel_tol=0.02)
assert gaps[0.02] > 3 * gaps[0.5]

length_coeff = lambda A_i, length: A_i / length      # the per-token pull after dividing by the response length
assert math.isclose(length_coeff(-1.0, 10), -0.1)
assert math.isclose(length_coeff(-1.0, 40), -0.025)
assert abs(length_coeff(-1.0, 40)) < abs(length_coeff(-1.0, 10)) / 3

# ---- Q6: k1 and k3 estimators of KL(pi_theta || pi_ref)
def exact_kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def k1_k3_expectations(p, q):
    u = q / p  # NOTE: u = pi_ref / pi_theta = q / p; flipping this ratio breaks both estimators silently
    k1, k3 = -np.log(u), (u - 1) - np.log(u)
    return float(np.sum(p * k1)), float(np.sum(p * k3))


for _ in range(50):
    p, q = softmax(rng.normal(size=6)), softmax(rng.normal(size=6))
    kl = exact_kl(p, q)
    e_k1, e_k3 = k1_k3_expectations(p, q)
    assert math.isclose(e_k1, kl, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(e_k3, kl, rel_tol=1e-9, abs_tol=1e-9)
    u_all = q / p
    assert np.all(((u_all - 1) - np.log(u_all)) >= -1e-12)          # k3 >= 0 pointwise, not just in expectation

for eps in (1e-2, 1e-3, 1e-4):                        # near u = 1: k1 is first order, k3 is second order and >= 0
    u = 1 + eps
    assert math.isclose(-math.log(u), -eps, rel_tol=2e-2)
    assert math.isclose((u - 1) - math.log(u), eps ** 2 / 2, rel_tol=2e-2)

assert math.isclose(-math.log(1.01), -0.00995, rel_tol=1e-3)               # the u = 1.01 numbers quoted above
assert math.isclose((1.01 - 1) - math.log(1.01), 4.97e-5, rel_tol=2e-3)

# ---- Q7: the closed-form optimum of the KL-regularised objective, and the DPO loss
def closed_form_pi_star(pi_ref, reward, beta):
    unnorm = pi_ref * np.exp(reward / beta)
    return unnorm / unnorm.sum()


def kl_regularised_objective(pi, pi_ref, reward, beta):
    """J[pi] = E_pi[r] - beta * KL(pi || pi_ref)."""
    mask = pi > 0
    kl = np.sum(pi[mask] * np.log(pi[mask] / pi_ref[mask]))
    return float(np.sum(pi * reward) - beta * kl)


N_RESPONSES, BETA = 6, 0.3
pi_ref = softmax(rng.normal(size=N_RESPONSES))
reward = rng.normal(size=N_RESPONSES) * 2.0
pi_star = closed_form_pi_star(pi_ref, reward, BETA)
assert math.isclose(pi_star.sum(), 1.0, rel_tol=1e-10)
assert np.all(pi_star > 0)

best = kl_regularised_objective(pi_star, pi_ref, reward, BETA)
competitors = rng.dirichlet(np.ones(N_RESPONSES), size=2000)          # many random competing distributions
scores = np.array([kl_regularised_objective(pi, pi_ref, reward, BETA) for pi in competitors])
assert np.all(scores < best - 1e-9)
assert best - scores.max() > 1e-6                     # a real margin, not a numerical coincidence

log_Z = math.log(float((pi_ref * np.exp(reward / BETA)).sum()))
assert math.isclose(best, BETA * log_Z, rel_tol=1e-8)          # the optimal value itself is beta * log Z(x)

implicit_reward = BETA * np.log(pi_star / pi_ref)      # the DPO implicit reward, from pi_star and pi_ref alone
assert np.allclose(implicit_reward, reward - BETA * log_Z, atol=1e-8)
assert abs(BETA * log_Z) > 1e-3                          # NOTE: Z(x) is a real additive shift, not coincidentally 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


for _ in range(30):
    yw, yl = rng.choice(N_RESPONSES, size=2, replace=False)
    assert math.isclose(implicit_reward[yw] - implicit_reward[yl], reward[yw] - reward[yl], rel_tol=1e-8)
    p_true = sigmoid(reward[yw] - reward[yl])
    p_from_policy = sigmoid(implicit_reward[yw] - implicit_reward[yl])
    assert math.isclose(p_true, p_from_policy, rel_tol=1e-8)

# ---- Q9: importance sampling across mu != pi, and the variance of a T-token product of ratios
PI_TOK = softmax(np.array([0.3, -0.1, 0.4, 0.0]))     # target policy's fixed per-token law
MU_TOK = softmax(np.array([0.1, 0.15, 0.05, 0.2]))    # behaviour policy's fixed per-token law
VOCAB = len(PI_TOK)
G_TOK = np.array([1.0, -0.5, 2.0, 0.3])               # a fixed per-token reward g(v); r(y) = sum_t g(y_t)
CHI2 = float(((PI_TOK - MU_TOK) ** 2 / MU_TOK).sum())   # chi^2(pi || mu), straight from its definition
assert math.isclose(float((PI_TOK ** 2 / MU_TOK).sum()) - 1.0, CHI2, rel_tol=1e-12)   # NOTE: E_mu[w^2] - 1 with w = pi/mu, not mu/pi
assert CHI2 > 0.02                                     # a real per-token mismatch, not a coincidental near-match


def token_weight_and_reward(tokens):
    """W = prod_t pi(y_t)/mu(y_t) and r(y) = sum_t g(y_t), vectorised over a batch of sequences (n, t_len)."""
    return (PI_TOK[tokens] / MU_TOK[tokens]).prod(axis=1), G_TOK[tokens].sum(axis=1)


def enumerate_sequences(t_len):
    """Every one of VOCAB**t_len sequences with its exact mu-probability: brute force, only for small t_len."""
    grids = np.meshgrid(*([np.arange(VOCAB)] * t_len), indexing="ij")
    tokens = np.stack([g.ravel() for g in grids], axis=1)
    return tokens, MU_TOK[tokens].prod(axis=1)


T_LIST = (1, 2, 4, 8, 16)
N_IS = 200_000
var_formula = {t: (1 + CHI2) ** t - 1 for t in T_LIST}          # the closed form derived above

for t_len in (1, 2, 4):                                          # brute force: every sequence, no IS formula involved
    tokens, mu_prob = enumerate_sequences(t_len)
    pi_prob = PI_TOK[tokens].prod(axis=1)
    exact_by_enum = float((pi_prob * G_TOK[tokens].sum(axis=1)).sum())
    assert math.isclose(exact_by_enum, t_len * float(G_TOK @ PI_TOK), rel_tol=1e-9)  # matches the i.i.d. linearity argument

var_empirical = {}
for t_len in T_LIST:
    tokens = rng.choice(VOCAB, size=(N_IS, t_len), p=MU_TOK)
    w_seq, r_seq = token_weight_and_reward(tokens)
    exact_reward = t_len * float(G_TOK @ PI_TOK)                          # independence structure: T times one token's expectation
    se_mean = math.sqrt(float((w_seq * r_seq).var(ddof=1)) / N_IS)        # NOTE: ddof=1 (sample std); numpy's default understates it
    assert abs(float((w_seq * r_seq).mean()) - exact_reward) < 6 * se_mean      # the IS estimate of E_pi[r(y)], from mu-samples alone
    se_w = math.sqrt(var_formula[t_len] / N_IS)
    assert abs(float(w_seq.mean()) - 1.0) < 8 * se_w + 1e-9                     # E_mu[W] = 1 at every length
    var_empirical[t_len] = float(w_seq.var(ddof=1))
    assert math.isclose(var_empirical[t_len], var_formula[t_len], rel_tol=0.15, abs_tol=0.03)

assert var_empirical[16] > var_empirical[8] > var_empirical[4] > var_empirical[2] > var_empirical[1]
assert (var_empirical[16] - var_empirical[8]) > 2 * (var_empirical[4] - var_empirical[2])   # accelerating: geometric, not linear
ratio_low = (var_formula[4] + 1) / (var_formula[2] + 1)
ratio_high = (var_formula[16] + 1) / (var_formula[8] + 1)
assert math.isclose(ratio_low, (1 + CHI2) ** 2, rel_tol=1e-9) and math.isclose(ratio_high, (1 + CHI2) ** 8, rel_tol=1e-9)

for eps in (1e-2, 1e-3):                          # near mu, KL and chi^2 agree to second order: KL ~ chi^2 / 2
    pi_near = (1 - eps) * MU_TOK + eps * PI_TOK
    kl_near = float((pi_near * np.log(pi_near / MU_TOK)).sum())
    chi2_near = float(((pi_near - MU_TOK) ** 2 / MU_TOK).sum())
    assert math.isclose(kl_near / chi2_near, 0.5, rel_tol=0.05)

# truncating W at a cap: lower variance, and a bias checked against the exact (enumerated) truncated mean
T_TRUNC = 8
tokens, mu_prob = enumerate_sequences(T_TRUNC)
w_all = PI_TOK[tokens].prod(axis=1) / mu_prob
assert math.isclose(float((mu_prob * w_all).sum()), 1.0, rel_tol=1e-9)    # E_mu[W] = 1 exactly, by enumeration
assert w_all.max() > 6.0                                                   # so every cap below actually binds somewhere

samp_tokens = rng.choice(VOCAB, size=(N_IS, T_TRUNC), p=MU_TOK)
w_samp, _ = token_weight_and_reward(samp_tokens)
for cap in (1.5, 3.0, 6.0):
    exact_truncated_mean = float((mu_prob * np.minimum(w_all, cap)).sum())
    assert exact_truncated_mean < 1.0 - 1e-6              # NOTE: min(W, c) <= W pointwise, so the mean can only fall
    truncated_samp = np.minimum(w_samp, cap)
    se_trunc = float(truncated_samp.std(ddof=1)) / math.sqrt(N_IS)
    assert abs(float(truncated_samp.mean()) - exact_truncated_mean) < 8 * se_trunc + 1e-9
    assert truncated_samp.var(ddof=1) < w_samp.var(ddof=1)   # capping large values can only reduce the variance
    assert truncated_samp.var(ddof=1) <= cap ** 2            # and 0 <= min(W, c) <= c bounds it by c^2 outright

print("all checks passed")
```

</details>

</details>
