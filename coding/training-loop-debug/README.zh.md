# 调试一个学不会的分类器

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 编程 · 机器学习调试（NumPy） | ★★★★☆ | 中等 | RE · RS · MLE · Applied AI | debugging, softmax, data-shuffling, gradient-scaling, momentum, dropout, broadcasting, sanity-checks | 3 个部分 / 45 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

*两层感知机*（two-layer perceptron）把标准化后的输入 $x \in \mathbb{R}^2$，经过一个有 $32$ 个单元的隐藏层，
映射到 $C = 3$ 个类别上的一个分布。隐藏层的前激活是 $z^{(1)} = xW_1 + b_1$（$W_1 \in \mathbb{R}^{2\times32}$，
$b_1 \in \mathbb{R}^{32}$），隐藏层激活是 $h = \mathrm{ReLU}(z^{(1)}) = \max(z^{(1)}, 0)$（逐元素取值），
logits 是 $z^{(2)} = h_{\mathrm{drop}}W_2 + b_2 \in \mathbb{R}^3$（$W_2 \in \mathbb{R}^{32\times3}$，
$b_2\in\mathbb R^3$），其中 $h_{\mathrm{drop}}$ 是 $h$ 经过*反转 dropout*（inverted dropout）之后的结果：
训练时，$32$ 个隐藏单元各自独立地以概率 $p = 0.2$ 被置零，每个存活下来的单元都按 $1/(1-p)$ 放大；评估时不丢弃
任何单元，也不做放大，所以 $h_{\mathrm{drop}} = h$。预测的类别概率是
$\mathrm{probs}_c = e^{z^{(2)}_c} / \sum_{c'=0}^{2} e^{z^{(2)}_{c'}}$，一个批次里 $N$ 个样本（真实标签
$y \in \{0,1,2\}^N$）上的损失，是平均交叉熵 $L = \frac1N\sum_{i=1}^N -\log \mathrm{probs}_{i, y_i}$。

`make_dataset` 在平面上从三个高斯团（Gaussian blob）里抽取 $600$ 个训练点和 $300$ 个验证点（每类 $200$/$100$
个），然后只用训练集自己的均值和标准差对两部分输入特征做*标准化*（standardise），
$x' = (x - \hat\mu) / \hat\sigma$，并把同一组 $\hat\mu, \hat\sigma$ 施加到验证集上。训练用带*动量*
（momentum）的小批量（mini-batch）SGD 跑 $30$ 个 epoch，批大小 $32$，每个 epoch 开始时重新打乱训练数据：
速度 $v$（每个参数一个数组，训练开始前只初始化一次为 $\mathbf 0$）按 $v \leftarrow \mu v - \eta\,
\nabla_\theta L$ 更新，每个参数按 $\theta \leftarrow \theta + v$ 更新，动量 $\mu = 0.9$，学习率为 $\eta$。

### Part 1 —— 找出并修复每一个 bug

下面这段脚本本应在 `make_dataset` 生成的数据上训练这个网络，打印每个 epoch 的平均训练损失和最终的验证准确率。
它恰好含有六个 bug，没有一个会抛出异常。找出并修复每一个。六个都修好之后，`train(seed=0)` 能达到至少 $0.9$
的验证准确率。

```py
import numpy as np

HIDDEN = 32
DROPOUT_P = 0.2
N_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.15
MOMENTUM = 0.9
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 100
CENTERS = np.array([[0.0, 2.0], [-1.7320508, -1.0], [1.7320508, -1.0]])
BLOB_STD = 0.9


def make_dataset(seed: int):
    """Three Gaussian blobs in 2-D, one per class. Returns standardised (X_train, y_train, X_val, y_val)."""
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def forward(params: dict, X: np.ndarray, training: bool, rng: np.random.Generator) -> dict:
    """X: (N, 2). Returns {"probs": (N, 3), "cache": ...}. training selects training- versus eval-time behaviour."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)                                       # ReLU
    keep_prob = 1.0 - DROPOUT_P
    mask = (rng.random(h.shape) < keep_prob) / keep_prob           # inverted dropout
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    """Mean softmax cross-entropy over the batch, and its gradient with respect to every parameter."""
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = probs - onehot
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    """probs: (N, 3) predicted class probabilities. y: (N,) true labels. Returns the fraction correct."""
    pred = np.argmax(probs, axis=1, keepdims=True)
    return float(np.mean(pred == y))


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        X_shuf = X_train[rng.permutation(N)]
        y_shuf = y_train[rng.permutation(N)]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            velocity = {k: np.zeros_like(v) for k, v in params.items()}
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
        print(f"epoch {epoch + 1:2d}  training loss {train_losses[-1]:.4f}")
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    print(f"validation accuracy: {val_accuracy:.3f}")
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}


if __name__ == "__main__":
    train()
```

按原样运行，会打印出类似这样的内容：

```text
epoch  1  training loss 25.5481
epoch  2  training loss 26.7524
epoch  3  training loss 26.8433
...
epoch 16  training loss nan
...
epoch 30  training loss nan
validation accuracy: 0.333
```

### Part 2 —— 预测每个 bug 的症状

对六个 bug 中的每一个，在只保留这一个 bug（其余全部修复）的情况下，说明每个 epoch 平均训练损失的曲线是什么
样子、报告出的验证准确率是多少，与完全正确的运行相比，并解释原因。

### Part 3 —— 能尽早捕捉这些问题的健全性检查

实现 `sanity_checks`：接收上面四个部件和一份数据集，在正式投入一次完整训练之前跑五个快速检查，返回一个
`dict[str, bool]`：

- `initial_loss_near_ln_C`：一个刚初始化的网络在给定数据集上的损失（dropout 关闭）与 $\ln C$ 相差小于
  $0.1$。
- `overfits_tiny_batch`：从一个全新的初始化开始，对数据集的前 $8$ 个样本做 $500$ 步普通梯度下降（学习率
  $0.5$，dropout 关闭），能把损失降到 $0.05$ 以下。
- `gradients_match_finite_differences`：在前 $6$ 个样本上（dropout 关闭，float64），每一个解析梯度都与
  步长为 $10^{-5}$ 的中心有限差分相符，相对误差小于 $10^{-5}$。
- `eval_invariant_to_batching`：对前 $6$ 个样本，一起评估和按打乱后的顺序逐个单独评估（dropout 关闭），
  给出相同的预测概率。
- `accuracy_metric_correct_on_labels`：把真实标签的 one-hot 编码当作预测概率传入时，准确率函数报告恰好
  $1.0$。

```py
def sanity_checks(init_params, forward, loss_and_grads, accuracy_score,
                   X: np.ndarray, y: np.ndarray, n_classes: int, seed: int = 0) -> dict:
    """Runs the five checks above against the given pipeline pieces and dataset."""
```

## 参考解答

<details>
<summary>展开参考解答</summary>

动手写代码之前，有两点值得和面试官确认：评估时 dropout 应当关闭（这是全篇假设的标准反转 dropout 约定），
以及 Part 1 的及格线是验证准确率明显高于 $0.9$，而不是某个具体的精确数字，因为题目本身并没有限定这个合成
数据集的其它性质。

### Part 1

按脚本给出的症状线索依次修复这些 bug，每次都会出现一个新的症状。因为 bug 5 和 bug 6 都可能让*报告*出的
准确率出错，而模型本身并不差，表里同时也跟踪了*真实*准确率：用同一组训练好的参数，不论其它地方还留着哪些
bug，都重新用完全正确的前向传播和准确率函数评估一遍。

| 已修复的 bug | 最初几个 epoch 的训练损失 | 报告的／真实的验证准确率 | 指向下一个 |
| --- | --- | --- | --- |
| 都未修复 | $\approx 26$，一路升到 epoch $16$ 时变成 `nan` | $0.333$ / $0.333$ | softmax 的轴 |
| softmax 的轴 | $\approx 2.0$，仍然远高于健康的运行 | $0.333$ / $0.333$ | 缺失的 $1/N$ |
| + 梯度缩放 | $\approx 1.1$–$1.2$，停在 $\ln 3$ 这个平台 | $0.333$ / $0.243$ | 不匹配的打乱 |
| + 打乱 | 正常下降，epoch $1 \approx 0.39$ | $0.333$ / $0.957$ | 准确率函数 |
| + 准确率函数 | 不变 | $0.963$ / $0.957$ | 接近但不精确——重置的动量 |
| + 动量 | epoch $1$ 改善到 $\approx 0.30$ | $0.953$ / $0.957$ | 仍不精确——评估时的 dropout |
| + 评估时的 dropout（全部六个） | 不变 | $0.957$ / $0.957$ | — |

**Bug 2：softmax 在批次轴上做归约。** `logits.max(axis=0)` 和 `.sum(axis=0)` 是在固定每个类别的情况下，
对批次里的 $N$ 个样本做归约，而不是在固定每个样本的情况下对 $3$ 个类别做归约：结果沿着一行加起来并不是 $1$，
所以它不是任何一个样本的概率分布，还会因批次里恰好有哪些其它样本而变化。类别 $c$ 的总质量
$\sum_i e^{z^{(2)}_{i,c}}$，是按各行自己在这个类别上的 logit 值的比例，分摊到 $N$ 行上的，于是哪一行在类别
$c$ 上的 logit 最大，就几乎独占这份质量，让其余 $N-1$ 行在类别 $c$ 上的概率都接近 $0$——其中对大多数行来说，
这恰好正是它们自己的真实类别——这正是上面看到的 $-\log(\mathrm{probs}_{i,y_i})$ 被推到几十这个量级的原因。

**Bug 3：缺失的 $1/N$。** 前向传播算出的损失是 $N$ 个逐样本项的平均，$L=\frac1N\sum_i \ell_i$，其中
$\ell_i=-\log \mathrm{probs}_{i,y_i} = -z^{(2)}_{i,y_i} + \log\sum_{c'} e^{z^{(2)}_{i,c'}}$，于是
$\partial \ell_i/\partial z^{(2)}_{i,c} = \mathrm{probs}_{i,c}-\mathbb 1[c=y_i]$，
$\partial L/\partial z^{(2)}_{i,c}=\frac1N(\mathrm{probs}_{i,c}-\mathbb 1[c=y_i])$。`dlogits = probs -
onehot` 漏掉了这个 $\frac1N$，使每一个梯度——从而每一次参数更新，因为 $\eta$ 是直接乘在梯度上的——都恰好
大了 $N=32$ 倍：训练的表现就如同学习率是 $32\times 0.15=4.8$ 而不是 $0.15$，损失也就始终没能落回健康运行
的范围。

**Bug 1：不匹配的打乱。** `X_shuf = X_train[rng.permutation(N)]` 和 `y_shuf = y_train[rng.permutation(N)]`
各自抽取一个长度相同、但相互独立的排列，所以一旦 $N>1$，`X_shuf[i]` 和 `y_shuf[i]` 几乎从不对应同一个原始
样本。用这样近乎随机配对的标签训练，网络除了标签的边际分布之外没有别的可拟合的东西：在没有可用信号的情况下
最小化交叉熵，会把 `probs` 的每一行都推向经验类别频率，而三个类别是均衡的，这个频率就是均匀分布
$(1/3,1/3,1/3)$，它对任何一个真实标签的交叉熵都恰好是 $-\log(1/3)=\ln 3$——也就是上面看到的那个平台。

**Bug 6：准确率函数里的 `keepdims=True`。** `np.argmax(probs, axis=1, keepdims=True)` 的形状是 $(N,1)$；
拿它和形状为 $(N,)$ 的 `y` 比较时会右对齐并*广播*（broadcasting）成 $(N,N)$，其 $(i,j)$ 项是
$\mathrm{pred}_i=y_j$——本该落在对角线上的 $N$ 个逐样本正确比较，被 $N(N-1)$ 个无关的非对角项稀释了。记
$q_c$ 为 $N$ 个预测中等于类别 $c$ 的比例，$q'_c$ 为真实比例，那么在全部 $N^2$ 对上取平均就是
$\frac{1}{N^2}\sum_c (Nq_c)(Nq'_c)=\sum_c q_c q'_c$；对一个预测落在 $3$ 个均衡类别上的频率，大致就是各类别
真实出现频率的模型来说，每个 $q_c,q'_c\approx1/3$，报告出的数字就 $\approx 3\cdot(1/3)^2=1/3$，与 $N$ 个
对角项里实际有多少个正确无关。训练本身完全不受影响，所以损失曲线同样看不出任何线索。

**Bug 4：每一步都重置动量。** 在小批量循环的开头把 `velocity` 重新建成全零，意味着每一步算出的都是
$v=\mu\cdot\mathbf 0-\eta\nabla_\theta L=-\eta\nabla_\theta L$：不论 $\mu$ 是多少，都只是一次普通的梯度步。
真正会累积的速度，在梯度 $g$ 大致不变的连续 $k$ 步之后，会达到
$v_k=-\eta g\sum_{j=0}^{k-1}\mu^j=-\eta g\,(1-\mu^k)/(1-\mu)$，随着 $k$ 增大逐渐逼近 $-\eta g/(1-\mu)$，
在 $\mu=0.9$ 时是普通一步的 $10$ 倍；每一步都重置则把每次更新都钉在 $k=1$ 这一项上，所以最初几个 epoch
明显比正确累积的运行要慢，尽管两者在 $30$ 个 epoch 之后会到达同一个地方，因为没有加速的更新有足够的时间
追了上来——这是一个只改变损失曲线*形状*、不改变它最终落点的 bug。

**Bug 5：评估时 dropout 仍然生效。** 随机掩码不受 `training` 控制，是无条件生成的——这意味着
`forward(..., training=False, ...)` 仍然会随机置零 $20\%$ 的隐藏单元，并把其余的按 $1/(1-p)=1.25$ 放大：
对同一个 `X` 调用两次会抽到两个不同的掩码，结果可能不一致，所以评估时 `forward` 不再是其输入的确定性函数。
和 bug 6 一样，训练本身没有任何变化，所以这个 bug 在训练损失曲线上完全看不出来；只有对同样的输入重复评估
并比较，才能发现它。

```python
import numpy as np

HIDDEN = 32
DROPOUT_P = 0.2
N_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.15
MOMENTUM = 0.9
N_PER_CLASS_TRAIN = 200
N_PER_CLASS_VAL = 100
CENTERS = np.array([[0.0, 2.0], [-1.7320508, -1.0], [1.7320508, -1.0]])
BLOB_STD = 0.9


def make_dataset(seed: int):
    rng = np.random.default_rng(seed)

    def blobs(n_per_class, rng):
        X = np.concatenate([CENTERS[c] + BLOB_STD * rng.standard_normal((n_per_class, 2)) for c in range(N_CLASSES)])
        y = np.repeat(np.arange(N_CLASSES), n_per_class)
        return X, y

    X_train, y_train = blobs(N_PER_CLASS_TRAIN, rng)
    X_val, y_val = blobs(N_PER_CLASS_VAL, rng)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / std, y_train, (X_val - mean) / std, y_val


def init_params(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(
        W1=rng.standard_normal((2, HIDDEN)) * np.sqrt(2.0 / 2),
        b1=np.zeros(HIDDEN),
        W2=rng.standard_normal((HIDDEN, N_CLASSES)) * 0.01,
        b2=np.zeros(N_CLASSES),
    )


def forward(params: dict, X: np.ndarray, training: bool, rng: np.random.Generator) -> dict:
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    if training:
        keep_prob = 1.0 - DROPOUT_P
        mask = (rng.random(h.shape) < keep_prob) / keep_prob   # NOTE: inverted dropout, drawn only while training
    else:
        mask = np.ones_like(h)                                  # NOTE: identity at eval -- no masking, no rescaling
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)        # NOTE: max over classes (axis=1), not the batch
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)                 # NOTE: likewise sum over classes -- rows sum to 1
    cache = dict(X=X, z1=z1, mask=mask, h_drop=h_drop)
    return {"probs": probs, "cache": cache}


def loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray, training: bool, rng: np.random.Generator):
    out = forward(params, X, training=training, rng=rng)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = (probs - onehot) / N       # NOTE: /N mirrors the mean in `loss` -- omitting it scales every
                                          #       gradient (and so the effective learning rate) by N
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score(probs: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)      # NOTE: no keepdims -- shape (N,), matching y, so pred == y is elementwise
    return float(np.mean(pred == y))


def sgd_momentum_step(params, velocity, grads, lr, momentum):
    new_v = {k: momentum * velocity[k] - lr * grads[k] for k in params}
    new_p = {k: params[k] + new_v[k] for k in params}
    return new_p, new_v


def train(seed: int = 0) -> dict:
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}   # NOTE: created once, never reset inside the loop
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        perm = rng.permutation(N)
        X_shuf, y_shuf = X_train[perm], y_train[perm]              # NOTE: one shared permutation for X and y
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            loss, grads = loss_and_grads(params, Xb, yb, training=True, rng=rng)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
    val_accuracy = accuracy_score(forward(params, X_val, training=False, rng=rng)["probs"], y_val)
    return {"train_losses": train_losses, "val_accuracy": val_accuracy}
```

### Part 2

| bug（单独出现） | 训练损失曲线 | 报告的验证准确率 |
| --- | --- | --- |
| 1 —— 不匹配的打乱 | 每个 epoch 都停在 $\ln 3\approx1.10$ 附近 | $\approx 1/3$ |
| 2 —— softmax 在批次轴上做归约 | 很大，$17$–$27$，没有下降趋势 | $\approx 1/3$ |
| 3 —— 缺失的 $1/N$ | 偏高且不稳定，$1.1$–$4.7$，始终不收敛 | $\approx 1/3$ |
| 4 —— 每一步都重置动量 | 正常下降，但 epoch $1\approx0.39$，正确版本是 $\approx0.30$ | 与正确运行一致，$\approx 0.96$ |
| 5 —— 评估时的 dropout | 与正确运行一致 | 单次调用接近正确运行，但每次调用之间会变 |
| 6 —— 准确率里的 `keepdims` | 与正确运行一致 | 不论模型多好，都 $\approx 1/3$ |

bug 1、2、3 各自都破坏了梯度本身（或者梯度据以计算的标签），所以每一个都会在损失曲线上留下独特的印记，正如
Part 1 推导的那样：bug 1 停在 $\ln 3$ 的平台，bug 2 因为在错误的轴上归一化而爆炸，bug 3 由于 $4.8$ 的有效
学习率而持续偏高。bug 4 也会影响训练，但只影响它的*速度*：有 $30$ 个 epoch 可以恢复，每一步都是普通梯度步
依然能到达 bug 1–3 永远到不了的地方，所以它表现为最初几个 epoch 变慢，而不是走向错误的终点。bug 5 和 bug 6
都不触及训练能看到的任何东西——评估时的 dropout 和准确率指标都严格发生在最后一次梯度更新之后——所以两者的
损失曲线都和正确版本逐位相同；只有反复评估并比较（bug 5），或者拿一个已知答案的用例去检验这个指标（bug 6），
才能发现它们，这正是 Part 3 的检查要做的事。

### Part 3

```python
def sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X, y, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    results = {}

    # a freshly initialised net should be no more sure of anything than a uniform guess over the classes
    params = init_params(seed)
    loss0, _ = loss_and_grads(params, X, y, training=False, rng=rng)
    results["initial_loss_near_ln_C"] = bool(abs(loss0 - np.log(n_classes)) < 0.1)

    # backprop should be able to memorise a handful of examples almost exactly
    Xb, yb = X[:8], y[:8]
    p = init_params(seed + 1)
    for _ in range(500):
        _, grads = loss_and_grads(p, Xb, yb, training=False, rng=rng)
        p = {k: p[k] - 0.5 * grads[k] for k in p}
    final_loss, _ = loss_and_grads(p, Xb, yb, training=False, rng=rng)
    results["overfits_tiny_batch"] = bool(final_loss < 0.05)

    # the analytic gradient must match a central finite difference, parameter by parameter
    p64 = {k: v.astype(np.float64) for k, v in init_params(seed + 2).items()}
    Xg, yg = X[:6].astype(np.float64), y[:6]
    _, analytic = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
    eps = 1e-5
    worst_rel_err = 0.0
    for name, arr in p64.items():
        grad = analytic[name]
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = arr[idx]
            arr[idx] = orig + eps
            loss_plus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig - eps
            loss_minus, _ = loss_and_grads(p64, Xg, yg, training=False, rng=rng)
            arr[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            denom = max(abs(numeric), abs(grad[idx]), 1e-8)
            worst_rel_err = max(worst_rel_err, abs(numeric - grad[idx]) / denom)
    results["gradients_match_finite_differences"] = bool(worst_rel_err < 1e-5)

    # a prediction must not depend on which other rows happen to share its batch
    params_eval = init_params(seed + 3)
    X_sub = X[:6]
    probs_batch = forward(params_eval, X_sub, training=False, rng=rng)["probs"]
    order = np.random.default_rng(seed + 4).permutation(len(X_sub))
    probs_singleton = np.zeros_like(probs_batch)
    for i in order:
        probs_singleton[i] = forward(params_eval, X_sub[i:i + 1], training=False, rng=rng)["probs"][0]
    results["eval_invariant_to_batching"] = bool(np.allclose(probs_batch, probs_singleton, atol=1e-8))

    # the metric itself must report perfect accuracy when handed perfect predictions
    onehot = np.zeros((len(y), n_classes))
    onehot[np.arange(len(y)), y] = 1.0
    results["accuracy_metric_correct_on_labels"] = bool(accuracy_score(onehot, y) == 1.0)

    return results
```

每个检查针对一种不同的失效模式。`initial_loss_near_ln_C` 能在迈出第一步梯度更新之前，就尽早捕捉到一个缩放
很差的初始化，或者一个被破坏的前向传播——包括 bug 2，它在批次轴上做的 softmax，让一个全新的网络报告出的
损失就已经和 $\ln 3$ 相差很远。因为只需要反向传播，以及在 $8$ 个样本上跑几百步廉价的更新，`overfits_tiny_batch`
通常是在等不到一个完整 epoch 的情况下，捕捉一条断掉的梯度路径（权重转置错了、轴换错了）最快的办法；
`gradients_match_finite_differences` 是同一个想法更彻底的版本，逐个参数去检查，而不是相信损失在下降就意味着
梯度是对的。`eval_invariant_to_batching` 是专门为 Part 2 里那两个无声的 bug 设计的：bug 2 在 axis=0 上做的
softmax，给批大小为 $1$ 的输入和批大小为 $6$ 的输入，算出完全不同（退化）的归一化结果；bug 5 无条件的
dropout 每次调用都会抽出一个独立的随机掩码；两者都没法在同一行被单独评估、而不是跟别的行一起评估时，保持
预测不变。`accuracy_metric_correct_on_labels` 把这个指标从模型里完全隔离出来——喂给它真实标签的 one-hot
编码，就不再需要模型本身有多好，于是 bug 6 的广播错误就没有地方能藏在带噪声的真实预测背后。这样重新插入
bug 2，实际上会让五个检查里的四个失败，而不只是批次一致性这一个，因为一个永远算不出合法逐样本分布的前向
传播，会同时破坏损失、小批次拟合和梯度检查；bug 5 会让两个失败（批次一致性，以及梯度检查，因为有限差分需要
一个确定性的前向传播）；bug 6 完全局限在指标本身，只会让它自己那一项检查失败。

### 追问

- **仅凭损失曲线看不出来的 bug。** bug 5 和 bug 6 让每一个训练损失数字都和完全正确的运行一模一样，因为评估
  时的 dropout 和准确率指标都只在最后一次梯度更新*之后*才运行。再怎么盯着损失曲线看也发现不了它们中的任何
  一个；只有把同样的输入评估两次（bug 5），或者用一个已知答案的用例检验这个指标（bug 6），才能发现，这正是
  `sanity_checks` 存在的意义。
- **学习率范围测试。** 训练几百步，同时把学习率从一个很小的值开始按几何级数增大，把损失画出来，会看到一段
  宽阔、稳定、下降的区域，随后是一次陡峭的爆炸；bug 3 的有效学习率 $32\times\eta$ 远远落在这个点之后，尽管
  单看 $\eta=0.15$ 这个数字，纸面上毫不起眼。
- **训练之前把一个批次连同它的标签画出来。** 标准化之后，取几个小批量，把 `Xb` 按 `yb` 上色画出来，是针对
  bug 1 最省事的检查：如果每种颜色都均匀铺满同一片区域，而不是分成数据实际来自的那三个分开的团，这一眼就能
  看出来，完全不需要损失曲线。
- **两个数据集切分之间的标准化泄漏。** `make_dataset` 只在训练集上拟合 `mean`/`std`，再把它们用到验证集
  上；如果改成在两个切分拼接后的数据上计算，就会把验证集分布的一点信息泄漏进每一个标准化后的输入里，把
  验证准确率抬高一点——这个量小到单看不会显得可疑，又因为不会引发任何报错而很容易被忽略。
- **确定性。** 这里每一个随机性的来源——生成数据、初始化、打乱、dropout——都通过显式的
  `np.random.default_rng` 设定种子，从不裸调用全局的随机函数；正因如此，说 `train(seed=0)` 能达到某个具体
  的准确率才有意义，而不只是说它“通常”能达到。

<details>
<summary>验证代码（可运行）</summary>

```python
result = train(seed=0)
assert len(result["train_losses"]) == EPOCHS
assert result["val_accuracy"] >= 0.9, result["val_accuracy"]


# --- every formula quoted above, checked independently ---

def forward_loop(params, X):                              # forward's formula, written with explicit loops
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N, Din, H, C = X.shape[0], X.shape[1], params["W1"].shape[1], params["W2"].shape[1]
    probs = np.zeros((N, C))
    for i in range(N):
        z1 = [sum(X[i, d] * W1[d, j] for d in range(Din)) + b1[j] for j in range(H)]
        h = [max(v, 0.0) for v in z1]
        logits = [sum(h[j] * W2[j, c] for j in range(H)) + b2[c] for c in range(C)]
        m = max(logits)
        exps = [np.exp(v - m) for v in logits]
        s = sum(exps)
        probs[i] = [e / s for e in exps]
    return probs


rng = np.random.default_rng(3)
params_rand = init_params(3)
X_rand = rng.normal(size=(15, 2))
assert np.allclose(forward(params_rand, X_rand, training=False, rng=rng)["probs"],
                    forward_loop(params_rand, X_rand), atol=1e-8)

# inverted dropout: E[mask] = 1 elementwise, and about DROPOUT_P of units are zeroed
rng = np.random.default_rng(4)
keep_prob = 1.0 - DROPOUT_P
mask_sample = (rng.random((200_000, HIDDEN)) < keep_prob) / keep_prob
assert abs(mask_sample.mean() - 1.0) < 0.01
assert abs(np.mean(mask_sample == 0.0) - DROPOUT_P) < 0.01

# standardisation: the training split has mean 0 and standard deviation 1 per feature
X_train, y_train, X_val, y_val = make_dataset(0)
assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-8)
assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-8)
print("forward pass, dropout statistics, and standardisation match their formulas")


# --- bug machinery: one general, parameterised reimplementation, independent of the fixed code above
#     except for the pieces the corresponding bug leaves untouched -- not string-patching the listing ---

def forward_general(params, X, training, rng, bugs=frozenset()):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    keep_prob = 1.0 - DROPOUT_P
    if training or (5 in bugs):                            # bug 5: dropout stays on when training is False
        mask = (rng.random(h.shape) < keep_prob) / keep_prob
    else:
        mask = np.ones_like(h)
    h_drop = h * mask
    logits = h_drop @ W2 + b2
    axis = 0 if (2 in bugs) else 1                           # bug 2: normalise over the batch, not the classes
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=axis, keepdims=True)
    return {"probs": probs, "cache": dict(X=X, z1=z1, mask=mask, h_drop=h_drop)}


def loss_and_grads_general(params, X, y, training, rng, bugs=frozenset()):
    out = forward_general(params, X, training, rng, bugs)
    probs, cache = out["probs"], out["cache"]
    N = X.shape[0]
    loss = float(np.mean(-np.log(probs[np.arange(N), y] + 1e-12)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dlogits = probs - onehot
    if 3 not in bugs:                                        # bug 3: skip the /N that undoes the batch mean
        dlogits = dlogits / N
    dW2 = cache["h_drop"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = (dlogits @ params["W2"].T) * cache["mask"]
    dz1 = dh * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)


def accuracy_score_general(probs, y, bugs=frozenset()):
    pred = np.argmax(probs, axis=1, keepdims=(6 in bugs))     # bug 6: keepdims broadcasts (N,1) against (N,)
    return float(np.mean(pred == y))


def train_with_bugs(bugs=frozenset(), seed=0):
    """Reruns the training loop with an arbitrary subset of the six bugs reinserted; bugs=frozenset()
    is the fully fixed pipeline. Independent of train() above."""
    X_train, y_train, X_val, y_val = make_dataset(seed)
    params = init_params(seed)
    rng = np.random.default_rng(seed + 1000)
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    N = len(X_train)
    train_losses = []
    for epoch in range(EPOCHS):
        if 1 in bugs:                                          # bug 1: two independent permutations
            X_shuf, y_shuf = X_train[rng.permutation(N)], y_train[rng.permutation(N)]
        else:
            perm = rng.permutation(N)
            X_shuf, y_shuf = X_train[perm], y_train[perm]
        losses = []
        for start in range(0, N, BATCH_SIZE):
            Xb, yb = X_shuf[start:start + BATCH_SIZE], y_shuf[start:start + BATCH_SIZE]
            if 4 in bugs:                                       # bug 4: velocity reset every mini-batch
                velocity = {k: np.zeros_like(v) for k, v in params.items()}
            loss, grads = loss_and_grads_general(params, Xb, yb, True, rng, bugs)
            params, velocity = sgd_momentum_step(params, velocity, grads, LR, MOMENTUM)
            losses.append(loss)
        train_losses.append(float(np.mean(losses)))
    probs = forward_general(params, X_val, False, rng, bugs)["probs"]
    reported_acc = accuracy_score_general(probs, y_val, bugs)
    true_probs = forward_general(params, X_val, False, rng, frozenset())["probs"]
    true_acc = accuracy_score_general(true_probs, y_val, frozenset())
    return {"train_losses": train_losses, "reported_val_accuracy": reported_acc,
            "true_val_accuracy": true_acc, "params": params}


ALL_BUGS = frozenset({1, 2, 3, 4, 5, 6})
import warnings

# --- the buggy script exactly as given (all six bugs): the statement's worked example ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                            # expected overflow once bug 2 and 3 compound
    r_all = train_with_bugs(ALL_BUGS)
assert max(l for l in r_all["train_losses"] if np.isfinite(l)) > 20
assert any(not np.isfinite(l) for l in r_all["train_losses"])
assert abs(r_all["reported_val_accuracy"] - 1 / 3) < 0.05

# --- the fixing order of Part 1's table: 2, 3, 1, 6, 4, 5, one new symptom at a time ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r_fix2 = train_with_bugs(ALL_BUGS - {2})
    r_fix23 = train_with_bugs(ALL_BUGS - {2, 3})
r_fix231 = train_with_bugs(ALL_BUGS - {2, 3, 1})
r_fix2316 = train_with_bugs(ALL_BUGS - {2, 3, 1, 6})
r_fix23164 = train_with_bugs(ALL_BUGS - {2, 3, 1, 6, 4})
r_fixed_all = train_with_bugs(frozenset())

assert r_fixed_all["train_losses"] == result["train_losses"]                # train_with_bugs(frozenset()) agrees
assert r_fixed_all["reported_val_accuracy"] == result["val_accuracy"]       # with train(), bit for bit
assert 1.5 < max(r_fix2["train_losses"]) < 5.0                              # no longer huge, still clearly wrong
assert abs(r_fix23["train_losses"][0] - np.log(3)) < 0.15                    # the ln(3) plateau
assert abs(r_fix23["true_val_accuracy"] - 0.2433) < 0.02                     # shuffle still scrambled -- no better
                                                                              # than chance even off the reported metric
assert (r_fix231["train_losses"][0] < 0.5 and abs(r_fix231["reported_val_accuracy"] - 1 / 3) < 0.05
        and r_fix231["true_val_accuracy"] > 0.9)                             # healthy loss, broken reported accuracy
assert 1e-6 < abs(r_fix2316["reported_val_accuracy"] - r_fix2316["true_val_accuracy"]) < 0.05  # close, not exact
assert r_fix23164["train_losses"][0] < r_fix231["train_losses"][0]           # momentum now speeds up epoch 1
assert r_fixed_all["reported_val_accuracy"] == r_fixed_all["true_val_accuracy"]
print("the fixing order in Part 1's table reproduces its numbers one row at a time")

# --- Part 2: each bug, reinserted in isolation (every other bug fixed) ---
r1 = train_with_bugs(frozenset({1}))
assert abs(r1["reported_val_accuracy"] - 1 / 3) < 0.05 and r1["train_losses"][-1] > 0.9

r2 = train_with_bugs(frozenset({2}))
assert abs(r2["reported_val_accuracy"] - 1 / 3) < 0.05 and max(r2["train_losses"]) > 5.0

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r3 = train_with_bugs(frozenset({3}))
finite3 = [l for l in r3["train_losses"] if np.isfinite(l)]
assert abs(r3["reported_val_accuracy"] - 1 / 3) < 0.05 and min(finite3) > 1.0 and max(finite3) > 3.0

r0 = train_with_bugs(frozenset())
r4 = train_with_bugs(frozenset({4}))
assert r4["train_losses"][0] > 1.1 * r0["train_losses"][0]                   # bug 4: noticeably slower epoch 1
assert r4["reported_val_accuracy"] >= 0.9                                    # same destination by epoch 30

rng_demo = np.random.default_rng(42)
probs_a = forward_general(r0["params"], X_val, False, rng_demo, frozenset({5}))["probs"]
probs_b = forward_general(r0["params"], X_val, False, rng_demo, frozenset({5}))["probs"]
assert not np.allclose(probs_a, probs_b, atol=1e-6)                          # bug 5: two evaluations disagree

r6 = train_with_bugs(frozenset({6}))
assert abs(r6["reported_val_accuracy"] - 1 / 3) < 0.05
assert abs(r6["reported_val_accuracy"] - r6["true_val_accuracy"]) > 0.3      # bug 6: reported far from true
print("each bug reproduces its predicted symptom in isolation")


# --- Part 3: sanity_checks passes on the fixed pipeline, fails the relevant check when a bug is reinserted ---
res_fixed = sanity_checks(init_params, forward, loss_and_grads, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert all(res_fixed.values()), res_fixed


def fwd2(params, X, training, rng):
    return forward_general(params, X, training, rng, frozenset({2}))


def lg2(params, X, y, training, rng):
    return loss_and_grads_general(params, X, y, training, rng, frozenset({2}))


def fwd5(params, X, training, rng):
    return forward_general(params, X, training, rng, frozenset({5}))


def lg5(params, X, y, training, rng):
    return loss_and_grads_general(params, X, y, training, rng, frozenset({5}))


def acc6(probs, y):
    return accuracy_score_general(probs, y, frozenset({6}))


res_bug2 = sanity_checks(init_params, fwd2, lg2, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert res_bug2["eval_invariant_to_batching"] is False, res_bug2
assert sum(res_bug2.values()) == 1                          # only accuracy_metric_correct_on_labels survives

res_bug5 = sanity_checks(init_params, fwd5, lg5, accuracy_score, X_train, y_train, n_classes=3, seed=0)
assert res_bug5["eval_invariant_to_batching"] is False, res_bug5
assert sum(res_bug5.values()) == 3

res_bug6 = sanity_checks(init_params, forward, loss_and_grads, acc6, X_train, y_train, n_classes=3, seed=0)
assert res_bug6["accuracy_metric_correct_on_labels"] is False, res_bug6
assert sum(res_bug6.values()) == 4

print("sanity_checks passes on the fixed pipeline and fails the relevant check for bugs 2, 5 and 6")

print("all checks passed")
```

</details>

</details>
