# 设计 Gemini App

[English](README.md) · [中文](README.zh.md)

<!-- meta:begin -->
| 题型 | 优先级 | 难度 | 岗位 | 考点 | 形式 | 轮次 |
| --- | --- | --- | --- | --- | --- | --- |
| 系统设计 | ★★★★☆ | 困难 | SWE · MLE · Applied AI | streaming, conversation-storage, context-management, model-routing, safety-filtering, multimodal-uploads, capacity-planning | 45–60 分钟 | 技能面（Skills） · 终面（Final） |
<!-- meta:end -->

## 题目

设计一个消费级 AI 助手应用的后端，包含网页端和移动端客户端：登录用户可以与一个大语言模型进行多轮对话，可以在消息里附上图片，并且能看到回答逐 token 流式显示出来。

*轮次*（turn）是用户与助手之间的一次往来：用户发出的消息，连同模型对它的回复，合在一起算作一个轮次。*会话*（conversation）是属于同一个登录用户的一串有序轮次，由一个会话 id 标识；一个用户可以拥有多个会话，会话也从不在用户之间共享。*首 token 时间*（time to first token，TTFT）是从客户端发出一个轮次的那一刻，到模型回复的第一个 token 到达该客户端为止所经过的时间。*预填充*（prefill）是模型在一个轮次开始时，把当时已知的一切——系统提示词、组装好的会话历史，以及新消息——处理成模型可以据以生成的形式（一个键/值缓存，key/value cache）；它的开销随处理的 token 数增长。*解码*（decode）是紧随其后的一步：逐个生成回复的输出 token，每一步都要对预填充好的上下文以及此前已经生成的每一个 token 做注意力计算；它的开销随生成的 token 数增长。

为下面的规模设计：

- 60,000,000 名日活跃用户，人均每天 12 个轮次；峰值流量为日均的 3 倍。
- 一个平均轮次向模型发送 2,000 个输入 token（系统提示词 + 会话历史 + 新消息），生成 400 个输出 token。
- 10% 的轮次带有一张约 1 MB 的图片。
- 延迟目标：纯文本轮次的 TTFT 第 95 百分位低于 1.5 秒；一旦开始流式输出，每个回复至少以 30 个输出 token/秒的速度进行。
- 默认模型的一台服务主机能承载 50,000 个预填充 token/秒，或者 5,000 个解码 token/秒——在这次估算里把预填充和解码当作两个独立的主机池，谁也做不了对方的工作。此外还有一个更大的模型可用；不论在哪个池里，它每个 token 的开销都是默认模型的 4 倍。
- 会话历史会一直保留，直到用户删除。可用性目标：99.9%。

范围内：发送一个轮次并流式返回回复；列出用户的会话列表，并翻阅某个会话的历史记录；上传一张图片并把它附加到某个轮次上；从会话历史组装模型的上下文；为一个轮次在默认模型和更大的模型之间做选择；对用户发送的输入和模型流回的输出都做安全过滤；按用户设置限流与配额；记录用户对一条回复的反馈（赞或踩），供后续评估使用。范围外：训练模型；语音输入或输出；公开分享一个会话；计费。

要产出：

1. 需求与规模估算：平均和峰值的轮次/秒；峰值时预填充和解码的 token/秒，以及每个池各需要多少台主机；把一部分轮次路由给更大的模型对这个主机数的影响；会话文本和图片每天的存储增长量（说明你对每 token 字节数和每条消息的假设）。
2. 数据模型与 API：管理会话与附件所用的接口，以及用于流式返回一个轮次的回复所用的协议。
3. 一张架构图，以及对一个带图片的轮次的完整走查，从客户端发起请求，到客户端收到第一个和最后一个流式 token 为止。
4. 深入话题：(a) 流式传输与重新连接——一个在回复过程中掉线的客户端如何在不让模型重新生成任何内容的前提下恢复；(b) 上下文组装——超出 token 预算的历史记录如何被截断或摘要，以及系统提示词和会话历史如何做前缀缓存；(c) 安全——输入和输出分类器分别放在哪里，以及如何对一条已经在向客户端流式输出的回复做内容审核；(d) 系统过载时的模型路由与优雅降级；(e) 配额与防滥用。

## 参考解答

<details>
<summary>展开参考解答</summary>

动手设计之前有两点值得先确认：这个系统服务单一区域，还是从一开始就是全球性的——这会决定会话存储和服务主机池是否从第一天起就要跨区域复制；以及一条仍在生成中的回复是否只属于当前这一个已连接的客户端，还是也必须能被中途打开同一会话的第二台设备看到。这份设计假设先服务单一区域（多区域作为追问之一），并且假设不需要让一条仍在生成中的回复跨设备同步——这一点会在流式传输的深入话题里重新讨论，届时会发现这份设计几乎不用额外代价就能支持它。

### 需求与规模

**轮次速率。** $60{,}000{,}000$ 名日活跃用户，人均每天 12 个轮次：

$$60{,}000{,}000 \times 12 = 720{,}000{,}000 \text{ 个轮次/天，}$$

$$\text{平均轮次/秒} = \frac{720{,}000{,}000}{86{,}400} \approx 8{,}333。$$

按 $3\times$ 的峰均比，峰值吞吐量约为 $25{,}000$ 个轮次/秒。

**峰值时预填充与解码的 token/秒。** 每个轮次发送 2,000 个输入 token、生成 400 个输出 token，所以在 $25{,}000$ 个轮次/秒时：

$$\text{峰值预填充} = 25{,}000 \times 2{,}000 = 50{,}000{,}000 \text{ token/秒} \approx 50\text{M/秒，}$$

$$\text{峰值解码} = 25{,}000 \times 400 = 10{,}000{,}000 \text{ token/秒} \approx 10\text{M/秒。}$$

一台默认模型的主机能承载 50,000 个预填充 token/秒，或者 5,000 个解码 token/秒，而这两个池是分开的，所以裸下限是 $50{,}000{,}000/50{,}000=1{,}000$ 台预填充主机和 $10{,}000{,}000/5{,}000=2{,}000$ 台解码主机。这两个下限都没有给健康检查失败或者滚动发布留出任何余量；加上 20% 的余量，实际配置的机群规模是 $1{,}200$ 台预填充主机和 $2{,}400$ 台解码主机。一个轮次自己的预填充，如果前面没有排队、单独跑，需要 $2{,}000/50{,}000=40$ 毫秒——只占 1.5 秒 TTFT 目标里很小的一部分，把大部分预算留给负载下的排队、安全那部分深入话题里的分类器，以及网络传输。

**路由给更大的模型。** 把“每个 token 开销 4 倍”理解为每个 token 的主机时间是 4 倍，把一部分份额 $s$ 的轮次路由给它，会让处理这些 token 负载所需的主机数乘以 $(1-s) + 4s = 1 + 3s$：这 $s$ 部分的轮次现在每个 token 要花掉 4 倍的主机秒数，其余部分不变。当 $s=10\%$ 时，下限升到 $1{,}000\times1.3=1{,}300$ 台预填充主机和 $2{,}000\times1.3=2{,}600$ 台解码主机；加上同样 20% 的余量，是 $1{,}560$ 和 $3{,}120$。这个关系对 $s$ 是线性的，所以不管路由份额是多少，都按同一套算法定价，不需要为每个份额单独估算一遍。

**存储。** 一条存储的消息需要 `message_id`（16 字节）、`conversation_id`（16 字节）、`turn_seq`（8 字节）、`created_at`（8 字节）和 `role`（2 字节）——固定字段合计 50 字节——外加它的文本，按每 token 平均 4 字节 UTF-8 的假设（这是英文文本常用的经验估算）。一个轮次存两行：用户的新消息，假设平均 40 个 token——远短于 2,000 个 token 的上下文，因为那部分大多是重新组装出来的历史记录和系统提示词，而不是新写下的文字——以及助手的回复，正好是给定的 400 个输出 token。也就是说用户这一行是 $50+40\times4=210$ 字节，助手这一行是 $50+400\times4=1{,}650$ 字节，合计 $1{,}860$ 字节/轮次。按 $720{,}000{,}000$ 个轮次/天算，文本每天增长约 $720{,}000{,}000\times1{,}860\approx1.34$ TB。

10% 的轮次带一张约 1 MB 的图片：$720{,}000{,}000\times0.10=72{,}000{,}000$ 张图片/天，$72{,}000{,}000\times1\text{ MB}=72$ TB/天。图片的字节量比文本高出约 $54$ 倍，这正是为什么真正要按这个规模去构建的是对象存储、而不是上面那个行存储，也是为什么随着保留时间无限累积、需要生命周期或者冷层策略的是它。

### 数据模型与 API

下面每张表都按 `user_id` 分片：一个会话只属于一个用户，这样就能把一个用户的会话、消息、附件和反馈都放在同一个分片上，这也是为什么“直到用户删除为止”可以是一次有界的、单分片的操作，而不是要在整个存储里做分散-聚合（scatter-gather）。

**User（用户）**——`user_id`、`created_at`（鉴权与个人资料字段不在范围内）。

**Conversation（会话）**——`user_id`、`conversation_id`、`title`（由第一个轮次生成）、`last_turn_seq`、`created_at`、`updated_at`。分片键：`user_id`。

**Message（消息）**——`user_id`、`conversation_id`、`message_id`、`turn_seq`（这条消息所属的轮次）、`role`（`user` | `assistant`）、`text`、`attachment_ids`（可能为空）、`token_count`、`model`（生成它的模型；用户自己的消息此字段为空）、`status`（`streaming` | `complete` | `blocked`，只对助手消息有意义）、`created_at`。分片键：`user_id`，按 `(conversation_id, turn_seq)` 聚簇。

**Attachment（附件）**——`user_id`、`attachment_id`、`message_id`、`object_key`（它在对象存储里的位置）、`mime_type`、`byte_size`、`moderation_status`、`created_at`。分片键：`user_id`。字节本身存在对象存储里；这一行只是元数据和一个指针。

**Feedback（反馈）**——`user_id`、`feedback_id`、`message_id`、`conversation_id`、`rating`（`up` | `down`）、`reason`（可选）、`created_at`。分片键：`user_id`。

**TurnBuffer（轮次缓冲区）**（存放在一个快速的外部存储里，不是上面的行存储——生命周期很短，不属于“直到用户删除为止”那部分）——`turn_id` → 这个进行中的轮次目前已经生成的 token，外加一个 `status`（`streaming` | `done` | `blocked`）。这是可续传流的后端存储；流式传输的深入话题会讲到它。键：`turn_id`，TTL 是完成之后的几分钟——足够长，能扛过一次重连，之所以短，是因为一个轮次一旦 `complete`，它的 token 就已经在上面那份持久化的 `Message.text` 里了，一个放弃了这次流、转而重新读取会话的客户端，走那条路径也能拿到同样的答案。

REST，用于管理会话和附件：

1. `GET /v1/conversations?before=&limit=`——调用者的会话，从新到旧，各自带一段预览——供会话列表使用。
2. `POST /v1/conversations`——创建一个空会话，返回 `{conversation_id}`。
3. `GET /v1/conversations/{conversation_id}/messages?before_seq=&limit=`——分页的历史记录，从新到旧。
4. `DELETE /v1/conversations/{conversation_id}`——删除这个会话以及它名下的一切（消息、附件，以及针对这些消息的反馈）。
5. `POST /v1/attachments`——`{conversation_id, mime_type, byte_size}` → `{attachment_id, upload_url}`，一个短期有效的预签名 URL，客户端把图片字节直接上传到这个地址，这样一张 1 MB 的图片就不用经过 API 层中转。
6. `POST /v1/messages/{message_id}/feedback`——`{rating, reason?}` → `200`。

轮次提交与流式传输——选用服务端推送事件（Server-Sent Events，SSE），而不是 WebSocket，因为这个流是单向的（轮次一旦提交，客户端就再也不需要在同一条连接上往回发送任何东西）：

7. `POST /v1/conversations/{conversation_id}/turns`——`{idempotency_key, text, attachment_ids?}` → `202 {turn_id}`。服务器立刻开始组装上下文并生成，不等客户端打开一条流。携带一个在这个会话里已经见过的 `idempotency_key` 的重试调用，会返回已有的 `turn_id`，而不会再开一个新轮次。
8. `GET /v1/turns/{turn_id}/stream?after_seq=`——打开一条 SSE 连接，重放从 `after_seq` 开始的一切（首次连接时为 0），然后继续用实时 token 补上；标准的 SSE `Last-Event-ID` 请求头，配合每个事件把 `id` 设为它的 `seq`，对支持它的客户端来说，不用查询参数也能做到同样的事。事件类型：
   - `token {seq, text}`——一个生成出来的 token（或者为了传输效率打包在一起的一小段）。
   - `done {seq, message_id}`——生成完成并已持久化为 `message_id`；`seq` 是 token 总数。
   - `blocked {seq, reason}`——输出安全分类器提前截断了这条回复。
   - `error {reason}`——生成失败；客户端可以用同一个 `idempotency_key` 重试整个轮次。

### 架构

```text
客户端（网页 / 移动端）
  |
  |  1. POST /v1/attachments -> 预签名上传 URL；客户端直接 PUT 图片字节
  |  2. POST /v1/conversations/{id}/turns  {idempotency_key, text, attachment_ids}
  v
API 网关 -- 对调用者鉴权；按用户限流与配额（深入话题 e）
  |
  v
轮次编排器 -- 端到端负责一个轮次
  |  |
  |  +--> 会话存储（User、Conversation、Message、Attachment、Feedback；按 user_id 分片）
  |  +--> 对象存储（图片字节，由 Attachment.object_key 引用）
  |  +--> 反馈日志（Feedback 的追加式导出，供评估流水线使用）
  v
上下文组装器 -- 系统提示词 + （截断/摘要后的）历史 + 新消息；
  |             为这个会话查找前缀缓存（深入话题 b）
  v
输入安全分类器 -- 一旦发现违规就在这里短路，不会碰到任何模型主机
  v
模型路由器 -- 默认模型还是更大的模型；感知负载，过载时向默认模型降级（深入话题 d）
  v
预填充池（默认 | 更大模型的主机）-- 为上下文计算键/值缓存
  v
解码池（默认 | 更大模型的主机）-- 从交接过来的缓存开始生成输出 token
  v
输出安全分类器 -- 在每一小段新生成的内容放行之前先打分（深入话题 c）
  v
轮次编排器 -- 把每个放行的 token 追加进 TurnBuffer，也发到打开的流里
  v
客户端（网页 / 移动端）-- GET /v1/turns/{turn_id}/stream，可以从任意偏移量续传（深入话题 a）
```

一个用户在新消息里附上一张照片。客户端先调用 `POST /v1/attachments`，带上图片的大小和 MIME 类型，拿到一个预签名的 `upload_url`；它把图片字节直接上传到这个地址上的对象存储，不经过 API 层，并拿到一个 `attachment_id`。接着它调用 `POST /v1/conversations/{id}/turns`，带上消息文本、这个 `attachment_id`，以及一个全新的 `idempotency_key`，同时打开 `GET /v1/turns/{turn_id}/stream` 来接收回复。

API 网关先对请求鉴权、检查用户的限流预算（深入话题 (e)），然后才做别的任何事。轮次编排器为这个轮次里用户的那一半写入一行 `Message`——引用上传完成时已经写好的那行 `Attachment`——并读回足够的会话历史交给上下文组装器。上下文组装器把系统提示词、（可能经过截断或摘要的，深入话题 (b)）历史和新消息组装在一起；因为这个轮次带着一张图片，它还会在这些之上再加上图片的编码表示——超出了 2,000 输入 token、1.5 秒 TTFT 这些数字所描述的纯文本轮次的范围：一个带图片的轮次之所以不在这么紧的界限之内，正是因为这一步额外的编码。组装好的上下文要通过输入安全分类器；在这里没通过的轮次根本不会到达任何模型主机。

模型路由器为这个轮次选定默认模型还是更大的模型（深入话题 (d)），把组装好的上下文交给那个模型的预填充池里的一台主机，由它为整个上下文计算键/值缓存，再把缓存交接给同一个模型的解码池里的一台主机。解码开始生成输出 token；每一个都要经过输出安全分类器检查（深入话题 (c)），一旦放行，既会被追加进这个轮次的 `TurnBuffer`，也会作为一个 `token` 事件推下客户端打开的 SSE 连接——其中第一个事件到达客户端的那一刻，就是 TTFT 目标的衡量点。流式输出会按目标至少以 30 token/秒的速度持续，直到解码结束；编排器随后把生成完的回复写成助手的那行 `Message`，把 `TurnBuffer` 标记为 `done`，并在客户端收到最后一个流式 token 之后紧接着发出 `done` 事件——这次走查到这里结束。如果客户端的连接中途掉线过，重新连接同一个 `GET /v1/turns/{turn_id}/stream`、带上它已经应用过的最后一个 token 的偏移量，就会恰好从掉线的地方续上，包括之后的实时 token，而不需要模型重新生成任何内容（深入话题 (a)）。

### 深入话题

**(a) 流式传输与重新连接。** 解码池放行的每一个 token——在通过输出安全分类器之后——都会先被追加进这个轮次的 `TurnBuffer`，然后才推给当下恰好连着的那个客户端；生成本身会一直跑到完成（或者跑到被 `blocked` 打断）为止，不管有没有客户端连着在看，所以这个缓冲区总是跟得上实际已经生成的一切。客户端的 `GET /v1/turns/{turn_id}/stream?after_seq=`——首次连接时 `after_seq=0`，掉线之后再连时带上它已经应用过的最后一个 token 的偏移量——会重放 `buffer[after_seq:]`，然后继续实时流式输出；这整个过程都不会要求任何模型主机把什么东西再生成第二遍，这正是这道题点名要的那个性质。

*缓冲区放在哪里。* 把一个重新连接的客户端钉死在最初接受这个轮次的那一个编排器实例上——一次连接注册表式的查找，和把一条消息路由到握着某个特定用户连接的那个网关是同一种形状——是可行的，但这样做会额外引入一次查找和一份状态（哪个实例在处理这个 `turn_id`），而一个共享的缓冲区可以让这些都变得不必要：只要这个缓冲区本身存在一个按 `turn_id` 为键的快速外部存储里，而不是某一个实例的内存里，任何一个编排器实例都能处理这次续传请求。代价是每释放一个 token 就要多一次到那个存储的小写入，而不是写本地内存；在峰值 $10{,}000{,}000$ 个解码 token/秒下，这正好就是解码池本来就在生产 token 的速率，所以它只是在已经存在的热路径上多了一次存储写入，而不是一个新的瓶颈，而且正是它让一个水平扩展的编排器机群里的任何实例都能服务任何一次重连——如果开头那条假设以后有变化，同样的机制也能顺带让第二台设备接上同一个 `turn_id`。

*为什么不干脆重新生成。* 在重连时重新提交整个轮次完全不需要缓冲区，但解码是随机采样的，重新生成的回复不一定和客户端已经渲染出来的内容一致——两者可能会在句子中间就明显分叉——而且要为已经付过一次代价的 token 再跑一整遍解码。在每次重连时都从偏移量 0 重发缓冲区的全部内容，而不是从客户端自己的偏移量开始，能避免这种分叉，但避免不了重复发送一条大部分已经看过的回复所浪费的带宽；记录这个偏移量不会比客户端本来就要作为 `Last-Event-ID` 发送的东西多花一分代价。

下面可运行的验证代码模拟了一个轮次逐 token 生成的过程，对手是一个在随机时刻掉线又重连的客户端，其中也包括好几个 token 先在缓冲区里攒起来、再被一次重连一口气排空的情形，并确认客户端重新拼出来的记录和实际生成的记录完全一致——不多不少，顺序不乱——在每一次试验里都成立。

**(b) 上下文组装。** 上下文组装器把系统提示词、会话历史和新消息拼接起来；一旦这会超出上下文组装器愿意发送的 token 预算，最早的那些轮次不会被直接丢弃，而是被折进一份持续更新的、按会话维护的摘要里，这样一个持续很久的会话依然能带着它需要的信息往前走，而不必每个轮次都重新生成一次摘要——只有当真的要淘汰内容、而当前的摘要又已经过期时，才会用一次小模型调用把刚被淘汰的这些轮次折进摘要，这样摘要的开销就完全不会出现在普通轮次的路径上。

*前缀缓存。* 第 $n$ 个轮次的上下文，就是第 $n-1$ 个轮次的上下文加上第 $n-1$ 个轮次的回复、再加上末尾一条新消息——是一个只会往后追加的前缀，而不是每次都是一个全新的、任意的提示词——这让多轮对话很接近复用一份已经算好的键/值缓存的最好情形：如果服务第 $n-1$ 个轮次时建好的缓存，在第 $n$ 个轮次到达时依然常驻在某台解码主机上，就只需要预填充新追加的那些 token，不需要再算一遍整个上下文。用上面的数字来算，命中一次缓存时，只需要预填充假设中 40 个 token 的新消息，而不是完整的 2,000 个 token 的上下文：

$$\text{有效 token 数} = h \times 40 + (1-h) \times 2{,}000，$$

在假设命中率为 80% 时是 $0.8\times40+0.2\times2{,}000=432$ 个 token，相当于把那个轮次原本要付出的预填充负载削减了约 78%。这是叠加在上面那个已配置下限之上的一种优化，不是把配置压到下限以下的理由：一次冷启动、一次机群重新平衡，或者只是新会话里第一个轮次扎堆到来，都会让缓存不命中、回退到完整的上下文——依然正确，只是要付未缓存的那份代价——下限必须按这种情况来定，和上面那些推理服务池的下限是一个道理，因为正确性从不依赖缓存是不是热的。

要真正拿到这样的命中率，需要模型路由器优先把一个会话的下一个轮次送回最近处理过它上一个轮次的那台解码主机上，只要还在一个有界的“热”窗口之内，而不是纯按负载路由；一台有一阵子没见过某个会话流量的主机，会在更近期活跃的会话带来的内存压力下淘汰掉它的缓存——和任何一个共享缓存都需要的最近最少使用（LRU）策略一样——而没命中从来都不是正确性问题，只是让这个轮次慢一点。

**(c) 安全。** 输入分类器对组装好的上下文打分——新消息，连同附带的图片，在上下文组装器把两者都处理成一个文本形式、以及模型消费图片所用的某种表示之后——这发生在模型路由器之前；被它拒绝的轮次根本不会到达任何模型主机，所以输入这一侧不会为一个反正要被拦下的东西花掉任何预填充或解码预算。

输出这一侧的约束更难：回复是一边生成一边流式送出的，一个 token 一旦展示给用户，就没法收回了。只在完整回复生成完之后才做分类，能彻底消除这个风险，但意味着要把每一个 token 都按住，直到生成结束——这正是 $\geq 30$ token/秒的流式目标所不允许的。对生成出来的每一个 token 都单独分类，能避免这一点，但每次调用能给分类器判断的东西几乎没有——单独一个 token 很少能带来足够打分的上下文——同时还要为峰值 $10{,}000{,}000$ 个解码 token/秒里的每一个都加一次分类器调用。这份设计选择的做法是，每次对一小段刚生成出来的、末尾的文本做分类——比如说每 10 个 token 一次，按 30 token/秒的下限算是三分之一秒的生成量——只有这一段通过了才把它放行给客户端；一段没通过的话，就停止生成、不再放行它之后的任何内容，编排器发的是 `blocked` 而不是 `done`。代价是给这条流增加一段小的、恒定的滞留延迟，不会随着回复变长而增长，并且分类器调用是按段而不是按 token 来算的；换来的好处是分类器会拒绝的内容永远不会真正渲染给用户，不像一种只在事后发现违规才去拦住*后续* token 的设计。

**(d) 模型路由与优雅降级。** 路由器最基本的决策——一个轮次该用默认模型还是更大的模型——依据的是对新消息的一次轻量分类，再加上客户端发来的任何明确提示（比如用户自己选了一个“更强”模式），这发生在选定预填充池之前；这个分类本身，完全不考虑负载的话，就是在容量无限的情况下一个轮次会得到的结果，但这份设计实际上并没有照单全收。

*不考虑负载的静态路由*——不管更大那个池的队列深度如何，把分类器判给它的每一个轮次都送过去——被否决了：更大的池是按它预期的份额配置的（上面路由份额那部分的算法），任何超出这个份额的突发，都会直接变成排队延迟，而且恰好压在分类器判定最需要认真作答的那些请求上，没有任何东西能吸收它。*感知负载的路由*（选定）让路由器在参考分类器建议的同时，也读取更大的池当前的队列深度：一旦越过某个阈值，分类器只是弱倾向于更大模型的那些轮次就改送默认池——这种质量上的取舍，用户在一个本就模棱两可的轮次上大概率察觉不到——而分类器有把握确实需要更大模型的轮次，只要那个池还有余量，依然会送过去。这把容量花在最需要它的轮次上，而不是把排队延迟平均摊给所有人，并且只要过载是温和的，这种降级就是不可见的（一个略逊一筹的答案），而不是可见的（一个被打破的延迟目标）。

一旦连默认池都没法把排队延迟压在 TTFT 预算之内，剩下能用的一招，是路由份额那部分算法里已经出现过的那个：准入控制把每个池里在途（排队中或正在执行）的轮次数，限制在这个池稳态吞吐量的一个小倍数以内，超过这个上限到达的轮次会被要求重试，而不是无限期排队——一个 `503`，带一个根据池的当前深度算出来的短暂 `Retry-After`，而不是一个固定延迟。无界队列被否决的理由，和它在一个纯批处理设计里被否决的理由一样：过了这个上限，多等下去换来的是一个已经注定要错过自己延迟目标的轮次，代价却是它最终被服务时仍然要占掉一个主机槽位。

**(e) 配额与防滥用。** 每个用户持有一个持续按补充速率和突发额度补充的令牌桶，在 `POST /v1/conversations/{id}/turns` 时检查；一个会把它耗到负数的请求会收到 `429`，其中的 `Retry-After` 是根据补充速率算出来的。这个桶按实际消耗的 token 计量——输入和输出 token 各算 1 个单位，一个路由给更大模型的轮次两者都乘以 4——而不是按原始的轮次数计量：单看轮次数只能限制一个用户能问多少次，限制不了每一次问的代价有多大，一个总是要求更大的模型给出最长回复的用户，花掉的主机时间最多是同样输入规模的普通默认模型轮次的 $4\times$，正好是上面路由份额算法里已经定过价的那个倍数；按这个倍数实际花掉的东西计量，才能把一个用户在 GPU 时间上的需求真正限住，而不只是限住他的请求次数。

单靠按用户限流，挡不住一个愿意多开几个账号来把额度成倍利用的攻击者：同一条准入路径还会跟踪一个更粗粒度的信号——按设备指纹或者 IP 范围统计的请求量——一旦这个信号明显超出一个人正常使用可能产生的范围，就进一步收紧，或者要求额外验证，而不管背后到底有多少个不同的 `user_id`。耗尽一个用户普通的配额，会让轮次直接失败，但只耗尽了其中更大模型那部分份额，却不会：路由器会把这当成和过载信号（深入话题 (d)）一样的情况处理，默默把这个轮次继续放到默认模型上，而不是让它失败，因为一个能力稍弱一点的答案，对用户和系统来说都比硬性拒绝要好——这正是过载时已经在用的那条兜底路径。

### 追问

- **多区域与数据驻留。** 按用户注册所在的区域给会话存储分片，让这个用户的数据留在那个区域，对外仍然是一个统一的全球 API 域名；预填充池和解码池那时也需要一个能感知区域的路由器，一个正在旅行的用户读取自己在另一个区域的历史记录，只是多了一次跨区域读取的代价，不是一个正确性问题。
- **工具调用与检索增强。** 在一个轮次的生命周期里加一步：模型不再吐出一个 token，而是发出一个结构化的工具调用；由编排器——而不是某个模型主机——去执行这次检索调用，把结果折回上下文，在同一个轮次里再跑一轮预填充加解码——这时 TTFT 目标就要针对第一个真正生成出来的 token 重新表述，而不是针对这一趟工具调用的往返。
- **按用户记忆。** 超出单个会话范围、持续存在的关于某个用户的事实，应该放进一个按 `user_id` 为键的独立小型存储里，用和历史摘要一样的方式整理，再折进上下文组装里系统提示词那部分——恰好是已经被当作最值得缓存的那段前缀，所以只要记忆的变化比一次会话慢，它本身就不会改变上面那套前缀缓存的算法。
- **每轮成本。** 可以直接从上面的数字推出来：一个默认模型轮次的主机时间，正比于它的 2,000 个预填充 token 和 400 个解码 token，按默认模型每台主机的速率；一个更大模型的轮次则是它的 4 倍；乘以机群每主机小时的成本、再除以每台主机每秒的 token 数，就能得到每轮成本，而路由份额那个公式能把它变成任意路由份额下的每日成本。
- **反馈用于训练时的隐私控制。** 一次赞或踩是存在它所指向的那条消息和那个会话名下的，所以要把它用于训练，意味着喂给那条流水线的导出数据必须带着一个指回源消息的引用，而不是一份脱离关系的独立拷贝，并且要遵守同一次删除——那行 `Message` 被删除时，它也要一并失效，而不是作为一条孤儿记录继续留存。

<details>
<summary>估算核对（可运行）</summary>

```python
import math
import random

# ---- turn rate ----
dau = 60_000_000
turns_per_user_per_day = 12
total_turns_per_day = dau * turns_per_user_per_day
assert total_turns_per_day == 720_000_000

avg_turns_per_s = total_turns_per_day / 86_400
assert round(avg_turns_per_s) == 8_333

peak_factor = 3
peak_turns_per_s = avg_turns_per_s * peak_factor
assert round(peak_turns_per_s) == 25_000

# ---- prefill and decode tokens/s at peak, and the hosts each pool needs ----
input_tokens_per_turn = 2_000
output_tokens_per_turn = 400

peak_prefill_tokens_per_s = peak_turns_per_s * input_tokens_per_turn
assert round(peak_prefill_tokens_per_s / 1e6, 1) == 50.0

peak_decode_tokens_per_s = peak_turns_per_s * output_tokens_per_turn
assert round(peak_decode_tokens_per_s / 1e6, 1) == 10.0

prefill_tokens_per_s_per_host = 50_000
decode_tokens_per_s_per_host = 5_000

prefill_hosts_floor = math.ceil(peak_prefill_tokens_per_s / prefill_tokens_per_s_per_host)
assert prefill_hosts_floor == 1_000

decode_hosts_floor = math.ceil(peak_decode_tokens_per_s / decode_tokens_per_s_per_host)
assert decode_hosts_floor == 2_000

headroom = 0.2
prefill_hosts = math.ceil(prefill_hosts_floor * (1 + headroom))
assert prefill_hosts == 1_200

decode_hosts = math.ceil(decode_hosts_floor * (1 + headroom))
assert decode_hosts == 2_400

# a turn's own prefill, run alone with nothing queued ahead of it -- the compute floor under TTFT
prefill_floor_s = input_tokens_per_turn / prefill_tokens_per_s_per_host
assert round(prefill_floor_s * 1000) == 40

# ---- routing a share of turns to the larger (4x host-time per token) model ----
def hosts_with_routing(base_hosts_floor: float, large_share: float, cost_multiplier: float = 4.0) -> float:
    """base_hosts_floor hosts serve every turn at the default model's cost; routing a `large_share`
    fraction of turns to a model that costs `cost_multiplier` times as much host-time per token
    multiplies the floor by (1 - large_share) + large_share * cost_multiplier."""
    return base_hosts_floor * ((1 - large_share) + large_share * cost_multiplier)

assert hosts_with_routing(1_000, 0.0) == 1_000
for share in (0.05, 0.1, 0.2, 0.5, 1.0):
    assert math.isclose(hosts_with_routing(1_000, share), 1_000 * (1 + 3 * share))

large_share_example = 0.10
prefill_hosts_floor_at_share = hosts_with_routing(prefill_hosts_floor, large_share_example)
assert prefill_hosts_floor_at_share == 1_300

decode_hosts_floor_at_share = hosts_with_routing(decode_hosts_floor, large_share_example)
assert decode_hosts_floor_at_share == 2_600

prefill_hosts_at_share = math.ceil(prefill_hosts_floor_at_share * (1 + headroom))
assert prefill_hosts_at_share == 1_560

decode_hosts_at_share = math.ceil(decode_hosts_floor_at_share * (1 + headroom))
assert decode_hosts_at_share == 3_120

# ---- storage per day: conversation text and images ----
bytes_per_token = 4                     # assumption: ~4 bytes of UTF-8 text per token
row_overhead_bytes = 50                 # assumption: message_id16 + conversation_id16 + turn_seq8 + created_at8 + role2
avg_new_message_tokens = 40             # assumption: the user's freshly typed message, not the reconstructed context
avg_reply_tokens = output_tokens_per_turn

user_row_bytes = row_overhead_bytes + avg_new_message_tokens * bytes_per_token
assistant_row_bytes = row_overhead_bytes + avg_reply_tokens * bytes_per_token
assert user_row_bytes == 210
assert assistant_row_bytes == 1_650

text_bytes_per_turn = user_row_bytes + assistant_row_bytes
assert text_bytes_per_turn == 1_860

daily_text_bytes = total_turns_per_day * text_bytes_per_turn
daily_text_tb = daily_text_bytes / 1e12
assert round(daily_text_tb, 2) == 1.34

image_turn_fraction = 0.10
avg_image_bytes = 1_000_000             # 1 MB
daily_image_turns = total_turns_per_day * image_turn_fraction
assert daily_image_turns == 72_000_000

daily_image_bytes = daily_image_turns * avg_image_bytes
daily_image_tb = daily_image_bytes / 1e12
assert daily_image_tb == 72.0

assert round(daily_image_tb / daily_text_tb) == 54

print("all requirements-and-scale numbers check out")


# ---- prefix-cache savings in context assembly ----
def effective_prefill_tokens(hit_rate: float, new_message_tokens: int = avg_new_message_tokens,
                              full_context_tokens: int = input_tokens_per_turn) -> float:
    """On a cache hit only the newly appended tokens need prefilling, since the rest of the context's
    key/value state carries over from the previous turn; on a miss the whole context is prefilled cold."""
    return hit_rate * new_message_tokens + (1 - hit_rate) * full_context_tokens

assert effective_prefill_tokens(0.0) == input_tokens_per_turn
assert effective_prefill_tokens(1.0) == avg_new_message_tokens

example_hit_rate = 0.8
effective = effective_prefill_tokens(example_hit_rate)
assert round(effective) == 432  # NOTE: round(), not == 432 -- (1 - 0.8) is not exact in float64
reduction = 1 - effective / input_tokens_per_turn
assert round(reduction, 2) == 0.78

# ---- output-classifier holdback window, as a fraction of a second at the streaming floor ----
classifier_span_tokens = 10
streaming_floor_tokens_per_s = 30
holdback_s = classifier_span_tokens / streaming_floor_tokens_per_s
assert round(holdback_s, 2) == 0.33

print("context-assembly and safety-holdback numbers check out")


# ---- resume-by-offset streaming: a dropping, reconnecting client sees every token exactly once ----
class TurnBuffer:
    """Server-side per-turn token buffer backing the resumable stream: append-only for the life of the
    turn, readable from any offset by any orchestrator instance."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def append(self, token: str) -> None:
        self.tokens.append(token)

    def read_from(self, offset: int) -> list[str]:
        return self.tokens[offset:]


def simulate_turn(seed: int, n_tokens: int = 50, p_drop: float = 0.15, p_reconnect: float = 0.5):
    """Generates n_tokens one at a time into a TurnBuffer -- generation never pauses for the client's
    connection state -- while a client drops and reconnects at random points, resuming from its own
    offset; returns the client's reassembled transcript plus counts confirming both the live-delivery
    and the catch-up-on-reconnect paths were actually exercised."""
    rng = random.Random(seed)
    buf = TurnBuffer()
    received: list[str] = []
    offset = 0
    connected = True
    live_delivered = 0
    catchup_lengths: list[int] = []

    for i in range(n_tokens):
        token = f"tok{i}"
        buf.append(token)  # NOTE: append happens before delivery, so a mid-turn crash never drops a token
        if connected:
            if rng.random() < p_drop:
                connected = False                 # drops right as this token would have been sent
            else:
                received.append(token)
                offset += 1
                live_delivered += 1
        elif rng.random() < p_reconnect:
            connected = True
            catch_up = buf.read_from(offset)      # never asks for regeneration, only a buffer read
            if catch_up:
                catchup_lengths.append(len(catch_up))
            received.extend(catch_up)
            offset += len(catch_up)

    if offset < len(buf.tokens):                  # the reconnect every real client eventually makes
        catch_up = buf.read_from(offset)
        if catch_up:
            catchup_lengths.append(len(catch_up))
        received.extend(catch_up)
        offset += len(catch_up)

    return received, buf.tokens, live_delivered, catchup_lengths


total_live, total_catchups, max_catchup = 0, 0, 0
for seed in range(40):
    received, generated, live, catchups = simulate_turn(seed)
    assert received == generated, seed            # every token exactly once, in order, despite drops
    total_live += live
    total_catchups += len(catchups)
    max_catchup = max(max_catchup, max(catchups, default=0))

assert total_live > 0                              # some tokens were genuinely delivered live
assert total_catchups > 0                          # some reconnects genuinely had to catch up
assert max_catchup > 1                             # at least one catch-up spanned more than one token

print("resume-by-offset simulation confirms exactly-once, in-order delivery across reconnects")
print("all checks passed")
```

</details>

</details>
