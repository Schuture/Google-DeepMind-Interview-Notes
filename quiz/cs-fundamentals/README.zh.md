# 计算机基础问答：内存、并发与浮点数

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 知识问答 · 口述，含小演示 | ★★☆☆☆ | 中等 | SWE · Intern · RE | oop, memory-management, garbage-collection, race-conditions, deadlock, gil, cache-locality, floating-point, amortised-analysis | 10 个问题 / 45 分钟 | 技能面（Skills） |
<!-- meta:end -->

## 题目

全文中，复杂度上界均为最坏情形，除非题目明确说“平均”；文中提到的具体运行时行为（GIL、CPython 的回收器、列表
扩容）都是 CPython 3.11 的行为，而非 Python 语言本身的通用性质。

### 编程与内存

**Q1.** 什么是面向对象编程（object-oriented programming，OOP）？分别用一句话定义*封装*（encapsulation）、
*继承*（inheritance）、*多态*（polymorphism）和*抽象*（abstraction）。然后对比下面这两种整数 LIFO（后进先出）栈的设计：

```py
class StackInherit(list):
    """Reuses list by inheritance: is-a list."""
    def push(self, x: int) -> None: ...


class StackCompose:
    """Reuses list by composition: has-a list."""
    def push(self, x: int) -> None: ...
    def pop(self) -> int: ...
    def peek(self) -> int: ...
```

说明哪种设计更好，并结合上面四个概念说明原因。

**Q2.** 将*栈*（stack）和*堆*（heap）定义为两块内存区域，并对比二者的分配和释放方式。在一门带垃圾回收的语言
里，既然回收器从不会让不可达的内存一直占用着，那什么是*内存泄漏*（memory leak）？给出两个具体成因。在 C++
中，`std::unique_ptr`、`std::shared_ptr` 和 `std::weak_ptr` 各自表达了什么所有权关系？`weak_ptr` 解决了
`shared_ptr` 单独无法解决的什么具体问题？

**Q3.** 对比*引用计数*（reference counting）和*追踪式垃圾回收*（tracing garbage collection）这两种回收内存
的策略，并指出：即使一个对象本身已经从程序的其余部分变得不可达，它的引用计数也永远不会降到零的那种情形。解释
CPython 是如何把两者结合起来的——它的分代（generational）环回收器由什么触发，需要考虑哪些对象。演示一个进入
这种环的对象：证明删除所有指向环内的名字并不能把它释放掉，而显式跑一次回收可以。

### 并发

**Q4.** 两个线程并发地操作同一个共享变量 `x`（初始为 `0`），各自执行三步——*load*（把 `x` 读入一个线程私有的
寄存器）、*add*（把寄存器加 1）、*store*（把寄存器写回 `x`）：

```text
线程 1:  L1: r1 = x        A1: r1 = r1 + 1        S1: x = r1
线程 2:  L2: r2 = x        A2: r2 = r2 + 1        S2: x = r2
```

每个线程自己的三步保持顺序（`L` 在 `A` 之前，`A` 在 `S` 之前），但这六步之间可以按任意方式交错。列出这些交错
方式下 `x` 所有可能的最终取值，并准确说出哪些交错方式各自导致哪个取值。然后解释用一把互斥锁保护这三步，或者用
一条原子的*读—加—写*（fetch-and-add）指令代替它们，是如何解决这个问题的。

**Q5.** 定义 Python 的*全局解释器锁*（global interpreter lock，GIL）——它到底串行化了哪一件事？给定一个有多
个线程的程序，解释为什么增加线程数能缩短 I/O 密集型任务（例如若干线程各自在等待网络套接字）的墙钟时间，却不能
缩短用纯 Python 写的 CPU 密集型任务（例如每个线程对一个大范围求和）的墙钟时间。说出两种能让 CPU 密集型工作真
正用上多个核心的办法。

**Q6.** 陈述死锁成立所必须同时满足的四个 Coffman 条件。定义一组线程与资源的*等待图*（wait-for graph）——每个
线程一个节点，只要 $T_i$ 正阻塞地等待着 $T_j$ 当前持有的某个资源，就有一条边 $T_i \to T_j$——并说出这张图上
与死锁等价的那个条件。结合它消除的是哪一条 Coffman 条件，解释为什么让每个线程都按同一个固定的全局顺序去获取
锁可以防止死锁。

### 性能与数值

**Q7.** 一个 `R` 行 `C` 列、元素为 8 字节的二维数组 `A` 按行优先（row-major）顺序存储，因此 `A[i][j]` 相对起
始地址的字节偏移是 `(i * C + j) * 8`。机器的缓存以 64 字节的*缓存行*（cache line）为单位组织：访问任何尚未被
缓存的字节都是一次*缓存行加载*（cache-line load），会取回它所在的整条 64 字节缓存行，如果缓存（这里只有 4
行）已满，就淘汰最近最少使用（least-recently-used）的那一行；访问一个已经在缓存中的字节则不花代价。取
`R = 8`、`C = 16`，分别给出按行遍历 `A`（`for i: for j: visit(i, j)`）和按列遍历 `A`
（`for j: for i: visit(i, j)`）各自产生的缓存行加载次数的准确值，并用*步幅*（stride，相邻两次访问之间的字节
距离）解释两者为何不同。

**Q8.** 在 IEEE 754 双精度下，解释为什么 `0.1 + 0.2 != 0.3`。定义*机器精度*（machine epsilon），并给出双精
度下它的取值。定义*灾难性抵消*（catastrophic cancellation），并解释为什么直接计算 $1 - \cos x$ 在 $x$ 很小
时会损失精度，而代数上等价的 $2\sin^2(x/2)$ 不会。给出应当用来代替 `a == b` 的、基于容差比较两个浮点结果的
判据，并解释为什么它同时需要一个相对项和一个绝对项。

**Q9.** 对 fp32、fp16 和 bf16（每种格式都是一个符号位，加上决定可表示范围的指数（exponent）字段，再加上决定
精度的尾数/小数（mantissa/fraction）字段），分别给出指数位数和尾数位数，并由此求出每种格式能表示的最大有限
值。给出这三种格式各自的*机器精度*——1 与下一个可表示值之间的间隔——将其表示为尾数位数的函数。解释为什么训练
神经网络时通常更偏好 bf16 而不是 fp16，以及在使用 fp16 时，*损失缩放*（loss scaling）具体是为了防止哪种数值
故障。

**Q10.** 定义哈希表的*负载因子*（load factor），并分别用 $n$（已存储的键数）表示出查找的平均情形和最坏情形
时间复杂度。解释最坏情形是由什么造成的，以及为什么扩容——把容量翻倍并重新哈希每个键——能让平均情形随着 $n$
增长始终保持在 $O(1)$。推导：对一张从容量 1 开始、每次满了就翻倍的表做 $n$ 次插入，所有这些扩容加起来搬移元
素的总次数小于 $2n$——这正是让 Python 的 `list.append` 具有摊还（amortised）$O(1)$ 代价的同一个翻倍论证。

## 参考解答

<details>
<summary>展开参考解答</summary>

开口作答前有两件事值得先说清楚：下面的复杂度上界都按最坏情形算，除非题目说的是“平均”；所有关于 GIL、引用计
数或列表扩容的说法针对的都是 CPython 本身——其他实现（有自己的 JIT 和回收器的 PyPy；Jython；GraalPy）完全
可以有不同的行为，而且确实不同。

### 编程与内存

**Q1.** 面向对象编程（OOP）把一个程序组织成一组相互作用的对象，每个对象把状态和操作它的代码捆在一起，并围
绕这样的对象类别来构建代码复用和多态行为。*封装*是把对象的数据和作用在其上的操作捆在一起，并把数据藏在这些
操作背后，使它只能通过受控的接口改变。*继承*是子类获得父类的字段和方法，建模的是一种“is-a”（是一个）关系。
*多态*是对不同类型的对象调用同一个方法名，让每个对象运行自己特定于类型的实现，这个选择在运行时才做出。*抽象*
是通过一个简化的接口暴露对象的核心行为，同时隐藏这个行为具体是怎么实现的。`StackCompose` 是更好的设计：
`StackInherit(list)` 是“is-a”一个 list，因此它继承了 list 的每一个方法，包括 `insert`、`__setitem__` 和
`sort`，这些方法没有一个尊重栈唯一的不变量（元素只能从一端进出）——只要调用者伸手去用一个原本从未打算成为栈
接口一部分的方法，封装就被破坏了，下面的检查会具体演示这一点。`StackCompose` “has-a”（有一个）list，但只对
外暴露 `push`、`pop` 和 `peek`，所以这个不变量在它的接口层面根本无法被破坏。一般的原则是：当一个类需要另一
个类的*实现*、却不想承诺（进而暴露）它完整的*接口*时，优先选组合；只有在确实存在 is-a 关系、并且子类实例本
来就打算在任何用到父类的地方都能替代使用时，才用继承。

**Q2.** *栈*是保存调用帧的内存区域——每次函数调用活跃期间的参数、局部变量和返回地址——调用时压入、返回时弹
出，严格遵循后进先出（last-in-first-out）的顺序；分配和释放都只是移动一下栈指针，因此二者基本上是免费的，
而且一个值的生命期恰好和创建它的作用域绑在一起。*堆*是用来存放那些大小或生命期无法绑定到单个调用帧上的对象
的内存——由程序显式分配（`malloc`/`new`）或由垃圾回收器分配，释放方式则可以是显式释放、由回收器释放，或者
（用 C++ 智能指针时）在持有它的值离开作用域时释放；簿记开销使分配和释放都比在栈上更贵。在一门带垃圾回收的
语言里，回收器只会释放已经变得*不可达*的内存，所以那里的“泄漏”不是不可达的内存没被释放——而是内存沿着程序
其实已经不再需要、却依然存在的某条引用链，保持可达，因而被正确地留在了原地。两个具体成因：一个没有淘汰策
略、把它算过的每个结果都存起来的无界缓存，只要进程还在跑，它的体积就会一直增长；以及注册在一个长期存活的发
布者上、却从未取消订阅的监听器（例如传给 `subscribe` 的一个绑定方法），使得发布者的监听器列表一直让这个订
阅者——以及它的闭包捕获的一切——保持存活，远远超过创建它的代码本该用到它的时间，下面的检查对这两种情形都做
了演示。在 C++ 中，`unique_ptr` 表达的是*独占*所有权：它不能被复制，只能被移动，析构函数在唯一持有它的那个
`unique_ptr` 离开作用域时，把被指向的对象恰好释放一次。`shared_ptr` 通过引用计数（一个带原子强引用计数的控
制块）表达*共享*所有权：复制一次计数加一，销毁一次计数减一，计数降到零时释放被指向的对象。`weak_ptr` 是对
`shared_ptr` 所指对象的*非持有*观察者：持有一个 `weak_ptr` 不会影响强引用计数，安全使用它的方式是调用
`.lock()`，如果对象已经被释放，它会返回一个空的 `shared_ptr`。它存在的原因是，单靠 `shared_ptr` 使用的是没
有环回收器的纯引用计数：两个互相持有对方 `shared_ptr` 的对象（例如一对互相都需要一个指向对方的活指针的父子
对象）会形成一个环，即便所有外部引用都已经消失，这个环的强引用计数也永远不会降到零，于是两者都被永远泄漏
——这正是引用计数在 Q3 里的那个失效模式，只是这里没有追踪式回收器兜底，所以这个环必须手动打破：约定俗成的
做法是让子对象持有一个指回父对象的 `weak_ptr`，而不是 `shared_ptr`。

**Q3.** *引用计数*给每个对象都挂一个计数，记录有多少引用指向它，每多一个引用就加一，每少一个就减一；计数
一降到零，对象立刻被释放。它能立即、确定性地回收内存，但恰好有一个盲区：一组*环*（cycle）——几个对象直接或
者通过一条链互相引用，但环外没有任何东西指向它们——环里的每个对象仍然有一个正的计数，这个计数由环里的另一
个成员贡献，所以没有哪个计数会降到零，即便这个环对程序其余部分而言已经不可达，它还是会整体泄漏。*追踪式垃
圾回收*则是从一组根（root，全局变量、调用栈）出发，遍历可达性图，把能够到达的每个对象都标记出来；没被标记
到的就是垃圾，不管它内部的引用是怎么接起来的，所以环和其他任何东西一样容易被回收，代价是要扫一遍存活的内
存，而不是给每个对象做一次计数递减。CPython 把引用计数当作主要机制——每个对象都带着一个计数，值离开作用域
的普通情形会立刻被释放——并在此之上叠加了一个分代（generational）的追踪式*环回收器*（`gc` 模块），它纯粹是
给引用计数做不到的事情兜底。这个回收器只需要考虑有可能参与成环的“容器”对象（list、dict、类实例、闭包之类
——任何可能持有另一个对象引用的东西）；它按对象扛过了多少轮回收把它们分进三代，只有当某一代自上次扫描以来
积累了足够多的容器分配，才会去扫那一代，所以大多数对象（数字、字符串、不成环的实例）根本不会产生任何开销。
下面的检查构造了两个互相引用的对象，删除所有指向这个环的名字，说明其中一个对象的弱引用（weak reference）依
然存活——引用计数无法释放它——直到显式跑一次 `gc.collect()` 之后它才消失；而第三个不成环的对象，在它最后一
个名字被删除的瞬间就被释放，完全不需要任何回收动作。

### 并发

**Q4.** 这六步按各自线程内部顺序交错，一共有 $\binom{6}{3} = 20$ 种方式（只要选定 6 个位置里哪 3 个属于线
程 1，其余就随之确定）。`x` 最终只可能取两个值：**1** 或 **2**——不可能是 $0$，因为 `x` 只会增加；也不可能
大于 $2$，因为每个线程只执行一次 `add`。当且仅当这个调度是完全串行的——某个线程的 load、add、store 三步全
部完成之后，另一个线程的 load 才开始——最终值才是 **2**，因为这时第二个线程的 load 读到的是第一个线程已经
存好的结果，再在它上面加 1；这样的调度在 20 种里恰好有 2 种（线程 1 整体在前、线程 2 在后，或者反过来）。其
余每一种调度都会让*两个* load 都在 `x` 还是 $0$ 的时候读到它（第二个线程的 load 抢在第一个线程的 store 之
前发生），于是两个线程各自在本地算出 $0 + 1 = 1$，不管哪一个的 store 最后落地，都只是把另一个覆盖成同一个
值——这是典型的*丢失更新*（lost update）：两次自增里有一次完全没起作用。用一把互斥锁保护这三步，强制了两个
线程的临界区互斥，这就把调度空间收缩到只剩那两种完全串行的顺序，二者都给出正确的 $2$。用一条原子的读—加—写
指令则在硬件层面达到同样的效果：load、add、store 作为一步不可分割的操作发生，其他核心不可能看到它执行到一
半的中间状态，所以同样只剩下两种（现在是指令级别的）串行顺序。

**Q5.** GIL 是 CPython 解释器内部的一把互斥锁，执行 Python 字节码时只有一个操作系统线程能持有它；它存在的
原因是，CPython 内部的簿记——最明显的就是 Q3 里的那个引用计数——如果没有它，从多个线程同时更新是不安全的。
它的效果是：即便机器有很多核心，两个 Python 线程也绝不会在同一时刻执行 Python 字节码——任何时候最多只有一
个线程“身处”解释器内部。一个阻塞在 I/O 上的线程（套接字、磁盘读取、`time.sleep`）会在底层这次阻塞调用期间
释放 GIL，直到调用返回才重新获取它，于是它等待的这段时间里，另一个线程可以自由运行——因此若干个 I/O 密集
型线程的*等待*时间能够重叠起来，墙钟吞吐量因此提高，即便执行本身从未真正同时发生过。做纯 Python 计算的线
程不会主动阻塞，所以它会连续持有 GIL，只在周期性的强制切换时松手（CPython 按一个固定的切换间隔——默认每 5
毫秒——打断正在运行的线程，把 GIL 让给另一个正在等待的线程）；这只是把单个核心按时间片分给这些线程，而不是
让它们跑在不同的核心上，所以更多线程并不会提高纯 Python、CPU 密集型工作负载的总吞吐量，算上锁交接的开销
后，通常还会略微变差。真正能从 Python 用上多个核心的两种办法：一是启动独立的操作系统*进程*
（`multiprocessing`）——每个进程都有自己的解释器和自己的 GIL，因此它们能真正并行运行，代价是各自独立的内存
空间和需要显式的进程间通信；二是调用*在其计算核心周围释放 GIL 的原生代码*，例如 NumPy 的 C 循环，它们把真
正做数值运算的部分包在会释放 GIL 的调用里——于是若干个各自在等待某次 NumPy 调用的 Python 线程，其底层的 C
计算可以真正在不同核心上重叠执行。

**Q6.** 死锁要成立必须同时满足的四个 Coffman 条件：（1）*互斥*（mutual exclusion）——至少有一种资源，同一
时刻只能被一个线程持有；（2）*占有并等待*（hold and wait）——一个线程持有至少一个资源的同时，还在等待获取
另一个；（3）*不可剥夺*（no preemption）——资源不能被强行从持有它的线程手里拿走，只能由它自愿释放；（4）
*循环等待*（circular wait）——存在一个线程的环 $T_1, \dots, T_k$，其中每个 $T_i$ 都在等待
$T_{(i \bmod k) + 1}$ 持有的某个资源。在等待图上，系统发生死锁，当且仅当这张图里存在一个环——一组线程彼此
直接或间接地等待着同一组里的另一个，导致没有哪一个能先取得进展、释放下一个所需要的东西。要求每个线程都按
同一个固定的全局顺序（比如给每把锁只分配一次的 ID）去获取锁，消除的正是*循环等待*这一条，而按 Coffman 的
说法，单单消除这一条就足以让死锁不可能发生：如果一个线程持有锁 $a$、正在等待锁 $b$，全局顺序就迫使 $a < b$
（它必须先获取了较小的那把），于是等待图里的每一条边，都是从一个较小的锁 ID 指向一个较大的锁 ID；一个环则
要求沿着它 ID 严格递增，一路绕回到起点，这在一个固定的全序下是不可能的。下面的检查在手工构造的例子（一个
2-环、一个 3-环，以及两个无环图）上，以及在数百个随机图上把一个环检测器和一次独立的可达性计算相互比对，都
确认了这一点，并且另外确认了：边都遵守同一个固定节点顺序的图，永远不会包含环。

### 性能与数值

**Q7.** 按行遍历恰好产生 **16** 次缓存行加载；按列遍历恰好产生 **128** 次——是前者的八倍，基本上每访问一
次就要加载一次。元素 8 字节、缓存行 64 字节，一条缓存行正好装得下一行里连续的 8 个元素，也就是固定 `i` 时连
续 8 个 `j` 的取值。按行顺序（`for i: for j:`）相邻两次访问之间的*步幅*是 8 字节——它沿着一条缓存行的 8 个
元素一路走下去，下一次访问才会落到新的一行，所以每条缓存行恰好只被加载一次，也从不需要重新访问旧的一行；
总数是每访问 8 个元素加载一次，$8 \times 16 / 8 = 16$，而且即便缓存只有一行那么小也是这个结果，因为根本没
有任何重用可言。按列顺序（`for j: for i:`）相邻两次访问之间的步幅是 $C \times 8 = 128$ 字节——每一步都跳到
*下一行*，也就是不同的一条缓存行，具体来说，第 `i` 行落在第 $2i + \lfloor j/8 \rfloor$ 条缓存行上：

```text
按列遍历，line = (i * 16 + j) // 8，前 6 次访问：
  (i=0,j=0) 第 0 行  MISS
  (i=1,j=0) 第 2 行  MISS
  (i=2,j=0) 第 4 行  MISS
  (i=3,j=0) 第 6 行  MISS    <- 缓存（容量 4）现在装着 {0, 2, 4, 6}
  (i=4,j=0) 第 8 行  MISS    <- 第 5 条不同的缓存行：第 0 行（最近最少使用）被淘汰
  (i=5,j=0) 第 10 行 MISS
```

对固定的 `j`，这 8 行本身就已经要碰到 8 条不同的缓存行，而缓存只装得下 4 条，所以等访问到第 4 行时，第 0
行所在的缓存行早已被第 1—3 行挤掉——而且它会一直保持被淘汰的状态，因为移动到 `j + 1`（只要 `j < 8`）会按
完全相同的顺序，重新碰到完全相同的这 8 条缓存行，数量总是超过缓存能装的 4 条。由于从来没有任何重用能撑到被
利用的那一刻，每一次访问都是一次未命中，也就是每个元素都要加载一次，而不是每 8 个元素加载一次：
$8 \times 16 = 128$。

**Q8.** `0.1 + 0.2 != 0.3`，是因为这三个十进制字面量没有一个能在 IEEE 754 双精度下被精确表示——
$0.1 = 1/10$ 没有有限的二进制展开，正如 $1/3$ 没有有限的十进制展开一样——所以每一个都要先被舍入到离它最近
的可表示双精度数，才能参与任何运算，而离 $0.1$ 最近的那个双精度数，加上离 $0.2$ 最近的那个，其结果本身又
被舍入到最近的双精度数，并不会恰好落在离 $0.3$ 最近的那个双精度数上；下面的检查用精确有理数运算确认了这一
点，这种运算本身不带任何舍入，不会模糊这个比较。*机器精度* $\varepsilon$ 是 $1.0$ 与下一个较大的可表示值
之间的间隔——等价地说，是使浮点运算中 $1 + \varepsilon \ne 1$ 成立的最小正 $\varepsilon$；对双精度（52 个
显式尾数位）而言 $\varepsilon = 2^{-52}$，而在任意值 $x$ 附近，可表示的数之间大约相隔 $\varepsilon$ 乘以
$x$ 本身，即*相对*于 $x$ 间隔 $\varepsilon$，$|x|$ 每翻一倍，这些数就稀疏一倍。*灾难性抵消*说的是：当两个
几乎相等的浮点数相减时，它们的有效数字几乎全部抵消，于是每个数原本就带着的舍入误差（相对自身大小最多约
$\varepsilon$）就相对着这个小得多的真实结果显现出来——剩下多少位正确数字，大致上就看抵消掉了多少位。直接
计算 $1 - \cos x$ 在 $x$ 很小时恰好就是这种情形：$\cos x$ 首先被舍入成一个极其接近 $1$ 的双精度数（它的展
开是 $1 - x^2/2 + \dots$，而对很小的 $x$，那一点点 $x^2/2$ 完全可能被舍入吃掉），随后两个几乎相等的双精度
数相减，就把那个固定的绝对舍入误差，摆在了一个量级只有 $x^2$ 的真实答案面前——到 $x = 10^{-8}$ 时，
`cos(x)` 已经被舍入成恰好 `1.0`，所以 `1 - cos(x)` 恰好是 `0.0`：完全抵消，一位正确数字都不剩。恒等式
$1 - \cos x = 2\sin^2(x/2)$ 计算的是同一个量，却从不需要相减两个相近的数——$\sin(x/2)$ 直接算出来，精度就
是普通的 $O(\varepsilon)$ 相对误差，再平方、再乘 2，每一步都没有抵消——所以它在直接公式早已丢光所有数字的
那些 $x$ 上，依然保持准确。比较两个浮点结果，应该用一个组合容差 `abs(a - b) <= atol + rtol * abs(b)`
（`math.isclose`/`numpy.isclose` 就是这么做的），而不是 `a == b` 或者一个纯相对的判据：当数值较大时，相对
项才有意义（浮点数能保证的从来只是固定位数的正确数字，而不是固定的绝对误差），但当 `b` 趋近 $0$ 时它就失
去意义了——一个纯相对的判据，可能仅仅因为两个数相差一个很大的*倍数*，就把两个都已经无限接近零的数判为“不
同”——所以还需要一个绝对项，来规定“接近零”本身指的是什么。

**Q9.** fp32 有 8 个指数位和 23 个尾数位，最大有限值是 $(2 - 2^{-23}) \times 2^{127} \approx
3.4028 \times 10^{38}$。fp16 有 5 个指数位和 10 个尾数位，最大有限值恰好是
$(2 - 2^{-10}) \times 2^{15} = 65504$。bf16（“brain float16”）有 8 个指数位——和 fp32 一样多——却只有 7
个尾数位，最大有限值是 $(2 - 2^{-7}) \times 2^{127} \approx 3.3895 \times 10^{38}$：本质上就是把 fp32 的
尾数截断到 7 位、其余保持 fp32 的范围，而这也正是实践中（以及下面的检查里）产生 bf16 的方式——取一个值
fp32 位模式的高 16 位，丢掉其余部分。机器精度对 fp32 是 $2^{-23}$，对 fp16 是 $2^{-10}$，对 bf16 是
$2^{-7}$——每种情形都是最后一个尾数位上的一个单位。训练时通常更偏好 bf16，是因为它保留了 fp32 完整的指数
范围：任何在 fp32 里放得下的激活值或梯度量级，在 bf16 里同样放得下，单纯做类型转换不会有上溢到无穷或下溢
到零的风险；而它牺牲掉的精度，基于梯度的训练很能容忍，因为小批量采样自带的噪声本来就已经盖过了单个数值的
舍入误差；因此把 fp32 转成 bf16，实际上就是丢掉尾数的低 16 位，指数原封不动。fp16 的指数只有 5 位，范围因
此窄得多——它最小的规格化值大约是 $6.1 \times 10^{-5}$，非规格化数也只能到大约 $6 \times 10^{-8}$——比这
更小的梯度，在网络深处，或者损失本身量级很小之后，是很常见的，这些梯度一旦存成 fp16，就会被直接冲刷成恰好
的零，悄无声息地毁掉那部分学习信号。*损失缩放*防的正是这个：在反向传播之前，把损失、进而根据链式法则把每
一个梯度，都乘上一个很大的 2 的整数次幂常数，把整批梯度量级的分布往上移出下溢区——下面的检查用了一个
$10^{-8}$ 的梯度，fp16 直接存储会得到恰好的零，但在类型转换前先乘以 $2^{16}$ 缩放、之后再除回来，就能正确
地把它恢复出来——而乘以 2 的整数次幂在浮点运算里是精确的（它只移动指数字段），所以这个做法本身不会带来任
何额外的舍入误差。

**Q10.** 负载因子 $\alpha = n/m$ 是已存储的键数 $n$ 除以哈希表的容量 $m$（也就是桶的个数）。平均情形下查
找是 $O(1)$ 的：只要哈希函数能把键大致均匀地打散，并且负载因子保持在某个固定上界以下，平均每个桶就只装着
$O(1)$ 个键，于是查找一个键所需的比较次数期望是常数，与 $n$ 无关。最坏情形下查找是 $O(n)$ 的：如果哈希函
数把 $n$ 个键中的很多个、甚至全部都送进同一个桶——无论是运气不好还是故意用了一个很差的哈希函数——那个桶就
退化成了一个必须线性扫描的普通列表，而且扩容也救不了这一点，因为一个很差的哈希函数不管表变得多大，始终会
把所有东西都送到同一个地方，下面通过强制让每个键都走一个常数哈希函数验证了这一点。扩容——一旦 $\alpha$ 越
过某个固定阈值，就分配一个更大的新桶数组，并把每个已有的键重新哈希进去——正是让*平均*情形能在 $n$ 无限增
长时始终保持 $O(1)$ 的原因：没有它，$\alpha$ 会随 $n$ 线性增长，平均桶长度也会跟着线性增长。至于扩容本身
的代价：从容量为 1 的空表开始，每当一次插入会超出当前容量时，就把容量翻倍——把它当前持有的每个元素都拷贝
进新数组。对 $n$ 次插入而言，设 $m$ 是满足 $m \ge n$ 的最小的 2 的幂；表的大小依次达到 $1, 2, 4, \dots,
m/2$ 时各触发一次扩容（大小为 $m/2$ 时的那次扩容把容量涨到 $m$，足以从容覆盖剩下的插入而不再需要扩容），
每一次都恰好拷贝那么多个元素，所以总拷贝次数是一个等比数列之和

$$1 + 2 + 4 + \cdots + \frac{m}{2} = m - 1.$$

因为 $m$ 是不小于 $n$ 的*最小*的 2 的幂，前一个 2 的幂 $m/2$ 必然小于 $n$——否则 $m$ 就不会是最小的——所以
$m < 2n$，总数 $m - 1 < 2n$：无论 $n$ 有多大，平均每次插入摊到的拷贝都不到两次，下面既用这个精确公式、也
用模拟确认了这一点。这正是 Python 自己的 `list.append` 背后的论证：CPython 每次扩容都会多分配一些，而不是
每次只多分配固定的量，所以同一套等比数列的推理照样适用——它实际的增长因子更小（渐近地大约是 $1.125$ 倍，而
不是 $2$ 倍），这依然给出摊还 $O(1)$，只是常数更大（总拷贝次数是 $9n$ 而不是 $2n$，下面也确认了这一点），
因为更小的增长因子意味着对同样的 $n$ 要更频繁地扩容，总拷贝次数也就更多。

<details>
<summary>验证代码（可运行）</summary>

```python
import gc
import itertools
import math
import sys
import weakref
from collections import Counter, OrderedDict
from fractions import Fraction

import numpy as np

rng = np.random.default_rng(2026)

# ---- Q1: composition vs inheritance -- StackInherit exposes list's whole interface, StackCompose does not
class StackInherit(list):
    def push(self, x: int) -> None:
        self.append(x)


class StackCompose:
    def __init__(self) -> None:
        self._items: list[int] = []

    def push(self, x: int) -> None:
        self._items.append(x)

    def pop(self) -> int:
        return self._items.pop()

    def peek(self) -> int:
        return self._items[-1]


si = StackInherit()
si.push(1)
si.push(2)
assert hasattr(si, "insert")              # inherited from list: breaks the LIFO invariant
si.insert(0, 99)                          # NOTE: nothing in StackInherit's own code stops this
assert list(si) == [99, 1, 2]
sc = StackCompose()
sc.push(1)
sc.push(2)
assert not hasattr(sc, "insert") and not hasattr(sc, "append")   # only push/pop/peek are reachable
assert sc.pop() == 2 and sc.peek() == 1

# ---- Q2: two leak patterns -- an unbounded cache, and a listener that is never unsubscribed
cache: dict[int, int] = {}


def memo(n: int) -> int:
    if n not in cache:
        cache[n] = n * n
    return cache[n]


for i in range(2000):
    memo(i)
assert len(cache) == 2000                 # every distinct call grows it forever: no eviction policy


class Publisher:
    def __init__(self) -> None:
        self._listeners: list = []

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def unsubscribe(self, fn) -> None:
        self._listeners.remove(fn)


class Subscriber:
    def on_event(self) -> None:
        pass


pub = Publisher()
sub = Subscriber()
sub_ref = weakref.ref(sub)
pub.subscribe(sub.on_event)               # a bound method keeps its underlying object alive
del sub                                   # the caller's own name is gone, but the publisher still holds a reference
assert sub_ref() is not None              # NOTE: leaked -- unreachable from the caller, but not from the publisher
pub.unsubscribe(sub_ref().on_event)
assert sub_ref() is None                  # freed the instant the last reference (the publisher's) is dropped

# ---- Q3: a reference cycle needs the tracing collector; a non-cyclic object does not
class Node:
    def __init__(self) -> None:
        self.other: "Node | None" = None


gc.disable()                              # NOTE: otherwise an unrelated threshold-triggered collection could
try:                                      #       run between `del` and the assert below, by coincidence
    a, b = Node(), Node()
    a.other, b.other = b, a               # a -> b -> a: a cycle, unreachable from any name once both are deleted
    ref_a = weakref.ref(a)
    del a, b
    assert ref_a() is not None            # reference counting alone cannot free a cycle
    collected = gc.collect()
    assert ref_a() is None                # the tracing collector finds it and frees it
    assert collected >= 2

    c = Node()                            # no cycle: an ordinary object
    ref_c = weakref.ref(c)
    del c
    assert ref_c() is None                # freed the instant its refcount hits zero -- no gc.collect() needed
finally:
    gc.enable()

# ---- Q4: exhaustive enumeration of the load/add/store interleavings of two threads
def interleavings(len_a: int, len_b: int) -> list[list[tuple[str, int]]]:
    """Every way to merge an A-sequence and a B-sequence of the given lengths, preserving each one's own order."""
    out = []
    for a_positions in itertools.combinations(range(len_a + len_b), len_a):
        a_set = set(a_positions)
        seq, ai, bi = [], 0, 0
        for i in range(len_a + len_b):
            if i in a_set:
                seq.append(("A", ai)); ai += 1
            else:
                seq.append(("B", bi)); bi += 1
        out.append(seq)
    return out


def run_schedule(schedule: list[tuple[str, int]]) -> int:
    """Steps 0/1/2 are load/add/store; a thread's register is private, x is the one shared variable."""
    x = 0
    reg = {"A": None, "B": None}
    for thread, step in schedule:
        if step == 0:
            reg[thread] = x
        elif step == 1:
            reg[thread] = reg[thread] + 1
        else:
            x = reg[thread]
    return x


schedules = interleavings(3, 3)
assert len(schedules) == math.comb(6, 3) == 20
outcomes = {run_schedule(s) for s in schedules}
assert outcomes == {1, 2}
counts = Counter(run_schedule(s) for s in schedules)
serial = [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2)]
serial_rev = [("B", 0), ("B", 1), ("B", 2), ("A", 0), ("A", 1), ("A", 2)]
assert run_schedule(serial) == 2 and run_schedule(serial_rev) == 2
fully_serial = [s for s in schedules if s == serial or s == serial_rev]
assert len(fully_serial) == 2 and counts[2] == 2 and counts[1] == 18   # only the 2 serial schedules avoid the lost update

# a mutex (or one atomic fetch-and-add) restricts execution to exactly the fully-serial schedules
assert {run_schedule(s) for s in fully_serial} == {2}

# ---- Q5: the GIL's default forced switch interval
assert math.isclose(sys.getswitchinterval(), 0.005)      # 5 ms, the value quoted in the answer

# ---- Q6: deadlock as a cycle in the wait-for graph, cross-checked against an independent reachability test
def wait_for_cycle_dfs(graph: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node: str) -> bool:
        color[node] = GRAY
        for nxt in graph.get(node, []):
            if color.get(nxt, WHITE) == GRAY:
                return True
            if color.get(nxt, WHITE) == WHITE and visit(nxt):
                return True
        color[node] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in graph)


def wait_for_cycle_brute(graph: dict[str, list[str]]) -> bool:
    """Independent check: a full reachability closure (Floyd-Warshall); a cycle exists iff some node reaches itself."""
    nodes = sorted(graph)                 # NOTE: sorted -- a stable, order-independent enumeration of the nodes
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    reach = [[False] * n for _ in range(n)]
    for u, targets in graph.items():
        for v in targets:
            reach[idx[u]][idx[v]] = True
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                for j in range(n):
                    reach[i][j] = reach[i][j] or reach[k][j]
    return any(reach[i][i] for i in range(n))


G1 = {"T1": ["T2"], "T2": ["T1"]}                               # 2-cycle: deadlocked
G2 = {"T1": ["T2"], "T2": ["T3"], "T3": ["T1"]}                 # 3-cycle: deadlocked
G3 = {"T1": ["T2"], "T2": ["T3"], "T3": []}                     # a chain: not deadlocked
G4 = {"T1": ["T2", "T3"], "T2": [], "T3": ["T2"]}                # a DAG: not deadlocked
for g, expected in ((G1, True), (G2, True), (G3, False), (G4, False)):
    assert wait_for_cycle_dfs(g) == wait_for_cycle_brute(g) == expected

for _ in range(500):                      # random graphs: the two detectors always agree
    n_nodes = int(rng.integers(2, 8))
    nodes = [f"T{i}" for i in range(n_nodes)]
    graph = {u: [] for u in nodes}
    for u in nodes:
        for v in nodes:
            if u != v and rng.random() < 0.25:
                graph[u].append(v)
    assert wait_for_cycle_dfs(graph) == wait_for_cycle_brute(graph)

for _ in range(500):                      # a global lock order (edges only go to a higher-numbered node) never cycles
    n_nodes = int(rng.integers(2, 8))
    nodes = [f"T{i}" for i in range(n_nodes)]
    ordered_graph = {nodes[i]: [nodes[j] for j in range(n_nodes) if j > i and rng.random() < 0.4]
                     for i in range(n_nodes)}
    assert wait_for_cycle_dfs(ordered_graph) is False

# ---- Q7: cache-line loads for row-major versus column-major traversal, under a tiny LRU model
def line_loads(addresses: list[int], line_size: int, capacity: int) -> int:
    """A trace-driven LRU cache of `capacity` lines; returns the number of misses (cache-line loads)."""
    order: list[int] = []                 # front = most recently used
    misses = 0
    for addr in addresses:
        line = addr // line_size
        if line in order:
            order.remove(line)
            order.insert(0, line)
        else:
            misses += 1
            order.insert(0, line)
            if len(order) > capacity:
                order.pop()
    return misses


def line_loads_ordereddict(addresses: list[int], line_size: int, capacity: int) -> int:
    """Independent re-implementation of the same LRU policy, via OrderedDict instead of a plain list."""
    od: OrderedDict = OrderedDict()
    misses = 0
    for addr in addresses:
        line = addr // line_size
        if line in od:
            od.move_to_end(line)
        else:
            misses += 1
            od[line] = True
            if len(od) > capacity:
                od.popitem(last=False)
    return misses


ROWS, COLS, ELEM_SIZE, LINE_SIZE, CACHE_LINES = 8, 16, 8, 64, 4
row_major_addrs = [(i * COLS + j) * ELEM_SIZE for i in range(ROWS) for j in range(COLS)]
col_major_addrs = [(i * COLS + j) * ELEM_SIZE for j in range(COLS) for i in range(ROWS)]
row_major_misses = line_loads(row_major_addrs, LINE_SIZE, CACHE_LINES)
col_major_misses = line_loads(col_major_addrs, LINE_SIZE, CACHE_LINES)
assert row_major_misses == 16                                # one load per 8 elements: (8*16)/8
assert col_major_misses == 128                                # one load per element: every access misses
assert col_major_misses == 8 * row_major_misses               # exactly the 8 elements-per-line ratio
assert line_loads_ordereddict(row_major_addrs, LINE_SIZE, CACHE_LINES) == row_major_misses
assert line_loads_ordereddict(col_major_addrs, LINE_SIZE, CACHE_LINES) == col_major_misses
# the traced prefix from the statement: the first column exhausts the 4-line cache after 4 distinct rows
first_col_lines = [addr // LINE_SIZE for addr in col_major_addrs[:8]]
assert first_col_lines == [0, 2, 4, 6, 8, 10, 12, 14]

# ---- Q8: the float facts -- 0.1 + 0.2, machine epsilon, catastrophic cancellation, tolerance comparison
assert 0.1 + 0.2 != 0.3
exact_gap = Fraction(0.1) + Fraction(0.2) - Fraction(0.3)     # exact rational arithmetic: no rounding of its own
assert exact_gap != 0                                          # proves the inequality independently of float `!=`
assert 0 < abs(exact_gap) < Fraction(1, 10 ** 16)
assert (0.1 + 0.2) - 0.3 == 2.0 ** -54                          # NOTE: the float *expression* rounds once more than
                                                                 #       exact_gap does, and is not equal to it

eps64 = np.finfo(np.float64).eps
assert eps64 == 2.0 ** -52
assert 1.0 + 2 ** -53 == 1.0                                    # rounds away: below half the spacing at 1.0
assert 1.0 + 2 ** -52 != 1.0                                    # the definition of eps: smallest such that 1+eps != 1


def taylor_one_minus_cos(x: float) -> float:
    """1 - cos(x) via its Taylor series: shrinking, non-cancelling terms, accurate for small x."""
    return x ** 2 / 2 - x ** 4 / 24 + x ** 6 / 720 - x ** 8 / 40320


for x in (1e-4, 1e-6):
    ref = taylor_one_minus_cos(x)
    naive = 1 - math.cos(x)
    stable = 2 * math.sin(x / 2) ** 2
    stable_rel_err = abs(stable - ref) / ref
    naive_rel_err = abs(naive - ref) / ref
    assert stable_rel_err < 1e-9                                # stays at machine-precision accuracy
    assert naive_rel_err > 1000 * stable_rel_err                # cancellation costs several digits, and grows as x shrinks
assert math.cos(1e-8) == 1.0 and 1 - math.cos(1e-8) == 0.0       # total cancellation: not one correct digit left
assert 2 * math.sin(5e-9) ** 2 > 0.0                             # the stable formula has no such floor

assert 0.1 * 3 != 0.3 and math.isclose(0.1 * 3, 0.3)             # a relative tolerance catches ordinary rounding
assert not math.isclose(1e-300, 2e-300, rel_tol=1e-9, abs_tol=0.0)   # purely relative: still "different" near 0
assert math.isclose(1e-300, 2e-300, rel_tol=1e-9, abs_tol=1e-200)    # an absolute term is needed there instead

# ---- Q9: fp32 / fp16 / bf16 -- bit layout, largest value, epsilon; bf16 emulated by truncating fp32's bits
fi32, fi16 = np.finfo(np.float32), np.finfo(np.float16)
assert (fi32.nexp, fi32.nmant) == (8, 23) and (fi16.nexp, fi16.nmant) == (5, 10)
assert math.isclose(float(fi32.max), (2 - 2.0 ** -23) * 2.0 ** 127, rel_tol=1e-6)
assert fi16.max == 65504.0 == (2 - 2.0 ** -10) * 2.0 ** 15
assert fi32.eps == np.float32(2.0 ** -23) and fi16.eps == np.float16(2.0 ** -10)
assert float(fi16.tiny) == 2.0 ** -14                           # fp16's smallest normal value: ~6.1e-5 in the text
assert float(fi16.smallest_subnormal) == 2.0 ** -24              # fp16's smallest subnormal value: ~6e-8 in the text


def to_bf16(x: np.float32) -> np.float32:
    """bf16 has fp32's 8 exponent bits and only 7 mantissa bits: keep the top 16 bits of the fp32 pattern."""
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


bf16_max = to_bf16(np.float32(fi32.max))                        # truncating never overflows the exponent field
assert math.isclose(float(bf16_max), (2 - 2.0 ** -7) * 2.0 ** 127, rel_tol=1e-6)
one_bits = np.float32(1.0).view(np.uint32)
bf16_next = (one_bits + np.uint32(1 << 16)).view(np.float32)    # the smallest step the 7-bit mantissa can take
assert float(bf16_next) == 1.0 + 2.0 ** -7

tiny_grad = 1e-8                                                 # smaller than fp16's smallest subnormal (~6e-8)
assert np.float16(tiny_grad) == 0.0                              # fp16: flushed to exactly zero
scaled = np.float16(tiny_grad * 2.0 ** 16)                       # loss scaling by a power of two: an exact shift
assert scaled != 0.0
recovered = float(scaled) / 2.0 ** 16
assert math.isclose(recovered, tiny_grad, rel_tol=2 ** -9)       # within fp16's own relative precision
bf16_tiny = to_bf16(np.float32(tiny_grad))                       # bf16: fp32's exponent range, so no underflow here
assert bf16_tiny != 0.0
assert math.isclose(float(bf16_tiny), tiny_grad, rel_tol=2 ** -6)

# ---- Q10: doubling gives amortised O(1) insertion (total copies < 2n); average- vs worst-case hash-table lookup
def doubling_copies(n: int, growth: float = 2.0) -> int:
    """Simulate n insertions into an array that starts at capacity 1 and grows by `growth` whenever full.
    Returns the total number of element copies performed across every resize."""
    capacity, size, total_copies = 1, 0, 0
    for _ in range(n):
        if size == capacity:
            total_copies += size          # every existing element is copied into the new array
            capacity = max(capacity + 1, math.ceil(capacity * growth))
        size += 1
    return total_copies


for n in (1, 7, 100, 1_000, 100_000):
    m = 1 << (n - 1).bit_length() if n > 0 else 1     # smallest power of two >= n
    assert doubling_copies(n, growth=2.0) == m - 1     # the exact closed form derived in the text
    assert m - 1 < 2 * n                               # ... and it is always under 2n

# the same geometric argument at Python list's actual (smaller) growth factor: still O(1) amortised, bigger constant
for n in (1_000, 100_000):
    assert doubling_copies(n, growth=1.125) < 9 * n


class SimpleHashTable:
    """Separate chaining; doubles capacity whenever inserting would push the load factor above 0.75."""

    def __init__(self, capacity: int = 8, hash_fn=hash) -> None:
        self._capacity = capacity
        self._hash_fn = hash_fn
        self._buckets: list[list[tuple]] = [[] for _ in range(capacity)]
        self._size = 0
        self.resize_copies = 0

    def _grow(self) -> None:
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        for bucket in old_buckets:
            for key, value in bucket:
                self._buckets[self._hash_fn(key) % self._capacity].append((key, value))
                self.resize_copies += 1

    def insert(self, key, value) -> None:
        if (self._size + 1) / self._capacity > 0.75:
            self._grow()
        bucket = self._buckets[self._hash_fn(key) % self._capacity]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1

    def get(self, key):
        """Returns (value, comparisons) -- comparisons is how many keys were examined: a counting-model stand-in
        for lookup cost, used instead of timing it."""
        bucket = self._buckets[self._hash_fn(key) % self._capacity]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                return v, i + 1
        return None, len(bucket)


for n in (500, 5_000, 50_000):                                   # good hash: average comparisons stay flat in n
    keys = [int(k) for k in np.unique(rng.integers(0, 10 ** 12, size=2 * n))[:n]]
    table = SimpleHashTable()
    for k in keys:
        table.insert(k, k * 2)
    total_comparisons = sum(table.get(k)[1] for k in keys)
    assert total_comparisons / len(keys) < 3.0                   # O(1) average: no growth with n
    assert table.resize_copies < 3 * n                            # its own resizes are linear in n too

bad_n = 2_000                                                     # worst case: every key hashes to the same bucket
bad_keys = [int(k) for k in np.unique(rng.integers(0, 10 ** 12, size=2 * bad_n))[:bad_n]]
bad_table = SimpleHashTable(hash_fn=lambda k: 0)
for k in bad_keys:
    bad_table.insert(k, k)
assert len(bad_table._buckets[0]) == bad_n                        # resizing cannot help: every key still collides
_, comparisons = bad_table.get(bad_keys[-1])
assert comparisons == bad_n                                       # a full linear scan: worst-case O(n)

print("all checks passed")
```

</details>

</details>
