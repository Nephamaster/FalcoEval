## TODO

---

## QA

**一、项目深挖**

1. 你说 FalcoEval 是“大模型评估系统”，它和 lm-evaluation-harness、OpenCompass、HELM 这类已有框架相比，核心差异在哪里？
结论先给：
FalcoEval 的定位不是“通用 benchmark 框架”，而是一个**面向工程闭环的轻量评估系统**。它和 >lm-evaluation-harness、OpenCompass、HELM 的核心差异，本质上在于三点：**设计目标、系统边界、以及工程取舍**。
第一层：设计目标差异（research benchmark vs engineering tool）
这些主流框架本质是“研究导向 benchmark 平台”。
例如：
lm-eval：强调 reproducibility + 标准 benchmark（MMLU、HellaSwag）
OpenCompass：强调大规模模型横向对比 + leaderboard
HELM：强调多维评估（公平性、鲁棒性、风险）
而 FalcoEval 的目标不同，它是：
 **为模型迭代提供快速、可控、可定制的评估闭环**
换句话说，它更像一个**工程内评测系统**，而不是一个“论文 benchmark 平台”。
第二层：系统边界差异（闭环 vs 松散工具）
FalcoEval 是一个**端到端闭环系统**：
配置 → 推理 → 答案抽取 → 指标计算 → 性能分析 → 结果落盘
这一点和 lm-eval 这种“只负责 evaluation，不管生成细节”的框架不同。
具体差异：
FalcoEval：
内置推理（SGLang / Transformers）
内置 prompt 构造 + reference 构造
内置答案抽取 pipeline
内置性能 profiling（p50/p95/tokens/s）
lm-eval / OpenCompass：
更偏“评测调度器”
推理通常依赖外部接口（OpenAI API / vLLM / 自建服务）
很少关心 latency / throughput
所以一句话总结：
 FalcoEval = **评测系统（evaluation system）**
 lm-eval / OpenCompass = **评测框架（evaluation framework）**
第三层：Prompt & Answer 处理机制差异（工程化 vs 静态）
这是一个面试容易加分的点。
FalcoEval 里：
`Promptor`：根据任务动态构造 prompt
`Referencer`：动态生成标准答案
`answer_extraction`：统一抽取模型输出（解决格式不一致问题）
而 lm-eval / OpenCompass：
多数任务是**静态 prompt + 静态 reference**
很少有统一的“答案抽取层”
这带来一个工程上的核心价值：
 FalcoEval 能适配“非标准输出模型”，而 benchmark 框架通常假设模型是规范输出的
第四层：性能评估维度（这是明显差异点）
FalcoEval 明确把**性能指标**纳入一等公民：
TTFT / latency（p50 / p95）
tokens/s
throughput
而：
lm-eval：几乎不关心性能
OpenCompass：性能支持有限
HELM：更偏质量维度
所以它更接近真实业务需求：
 不只是“模型好不好”，而是“能不能上线”
FalcoEval：
优点：
结构简单（predict / eval 解耦）
新任务接入成本低（改 Promptor + Evaluator）
可控性强（本地模型、本地数据）
缺点（可以主动说，面试加分）：
benchmark 覆盖不如 OpenCompass
评测标准不完全统一（需要自己维护一致性）
缺少社区验证
最后给你一版“面试标准总结句”：
FalcoEval 和 lm-eval / OpenCompass / HELM 的核心差异在于：
它不是一个通用 benchmark 平台，而是一个面向模型迭代的工程化评测系统。
它提供从推理、prompt 构造、答案抽取到指标计算和性能分析的完整闭环，更强调可控性、可扩展性和性能评估，而不是大规>模标准 benchmark 对比。


2. 你为什么要自己实现一套评测框架，而不是直接基于 OpenCompass 二次开发？
3. FalcoEval 的整体架构是什么？请你从数据加载、prompt 构造、模型推理、答案解析、指标计算完整讲一遍。
4. 你简历里写“任务路由、数据构造、模型推理、输出解析、指标计算模块解耦”，具体是怎么解耦的？代码里对应哪些类或文件？
5. 新增一个数据集时，需要改哪些地方？如果要做到完全配置化，你会怎么设计？
6. 目前数据集 schema 不统一，你是如何做统一加载和校验的？
7. 为什么选择 jsonl 作为统一数据格式？相比 parquet、json、HF Dataset 有什么优缺点？
8. 多选题、短问答、生成式问答、数学题的 prompt 模板有什么区别？为什么这样设计？
9. 你如何保证不同模型评测时 prompt 是公平的？
我们通过四层机制保证评测公平性：首先用 Promptor 做模板标准化，其次在 prompt 中约束输出格式，然后通过 answer extraction 对不同模型的输出进行统一映射，最后固定推理参数避免采样带来的偏差。同时我们也认为完全公平在不同模型之间是不可达的，因此目标是工程上尽可能减少无关因素的影响。
10. 模型输出经常不是严格格式，比如选择题输出 “I think the answer is B”，你怎么做答案抽取？
11. 当前答案抽取规则有哪些边界问题？比如数学题输出多个数字怎么办？
12. 选择题 reference 是选项字母还是选项文本？为什么？
13. 你实现的 EM/F1 和 SQuAD 官方评测有什么差异？
14. ROUGE、BLEU 适合评估开放式生成任务吗？它们的缺陷是什么？
15. TokenEfficiency 是怎么定义的？这个指标是否合理？
16. 简历里写支持 acc/f1/bleu 等指标，为什么不同任务要选择不同指标？
17. 对长文本 QA、法律、医学任务，简单 EM/F1 是否足够？如果不够你会加什么指标？
18. 如果模型答案语义正确但字面不同，EM/F1 会低估，你怎么解决？
19. 是否考虑过 LLM-as-a-judge？它的可靠性问题怎么控制？
20. 如何防止评测集泄漏？如果模型训练中见过 benchmark，该怎么判断？

**二、性能与推理后端**

21. 你为什么选择 SGLang 作为高吞吐推理后端？
选择 SGLang 的核心原因是：FalcoEval 的瓶颈不是单条样本推理，而是大批量 benchmark 的离线生成吞吐。SGLang 更适合把大量样本组织成高并发、高吞吐的推理任务。
    我的回答可以这样说：
    FalcoEval 的评测流程里，模型生成预测是最耗时的阶段。如果直接用 Transformers 逐条生成，工程上最简单，但 GPU 利用率低，batch 调度能力弱，跑几千到几万条样本时耗时很长。所以我把 SGLang 作为高吞吐后端，把 Transformers 保留为兼容性和 fallback 后端。
    SGLang 的优势主要有三点。
    第一，它适合批量离线推理。评测任务通常是大量 prompt 独立请求，不需要复杂交互状态，天然适合 continuous batching。SGLang 可以把多个请求动态合批，提高 GPU 利用率，降低整体评测时间。
    第二，它对长 prompt、多样本生成更友好。评测集中不同样本的输入长度不一致，如果手写 PyTorch/Transformers batch，padding 浪费会比较明显；SGLang 这类推理后端在调度和 KV cache 管理上更成熟，吞吐更稳定。
    第三，它工程集成成本相对可控。FalcoEval 只需要把 prompt 列表交给推理后端，然后拿回文本输出，不需要深度侵入模型结构。也就是说，SGLang 被设计成一个可替换的 backend，而不是和评测逻辑耦合在一起。
    我没有只保留 SGLang，而是同时保留 Transformers，是因为 SGLang 并不覆盖所有情况。比如有些自定义模型、特殊 tokenizer、特殊 chat template 或本地调试场景，Transformers 更稳。因此系统设计上是：
    SGLang 用于高吞吐正式评测，Transformers 用于兼容性、调试和兜底。
    从软工角度看，这其实是一个 backend abstraction 的设计：评测主流程不关心底层是 SGLang 还是 Transformers，只关心输入 prompt、输出 completion 和性能统计。这样后续如果要接 vLLM、OpenAI API 或内部推理服务，也只需要新增一个 backend adapter，而不需要改评测主流程。
22. SGLang 和 vLLM 的核心区别是什么？
23. SGLang 的批量推理性能提升来自哪里？
24. 简历写“短答案类任务万级样本 5min 推理，单张 A100”，这个结果是怎么测的？
25. 这个 5min 包含模型加载时间吗？包含 warmup 吗？
26. batch size 怎么设置？过大或过小分别有什么问题？
27. 你怎么统计 p50/p95 latency？是单样本 latency 还是 batch 平摊？
    这里要先区分清楚：**p50/p95 latency 统计的是请求级、单样本维度的 latency，不应该用 batch 总耗时简单平摊**。
    在 FalcoEval 里，latency 的统计目标是衡量“一个样本从提交到拿到输出的端到端耗时”，所以更合理的定义是：
    ```text
    latency_i = end_time_i - start_time_i
    ```
    然后对所有样本的 `latency_i` 排序，取中位数作为 p50，取 95 分位作为 p95。
    这和 batch 平摊耗时不同。batch 平摊通常是：
    ```text
    avg_latency = batch_total_time / batch_size
    ```
    它更接近 throughput 指标，而不是用户感知 latency。比如一个 batch 里有 32 个样本，总共 3.2 秒，平摊是 0.1 秒，但实际上每个样本都可能等了接近 3.2 秒才拿到结果。用平摊值算 p50/p95 会严重低估真实延迟。
    所以我会把指标分成两类：
    ```text
    latency：按单样本请求统计，用于 p50 / p95
    throughput：按整体任务统计，例如 samples/s、tokens/s
    ```
    在 SGLang 这种 continuous batching 后端里，这个区分更重要。因为请求会被动态调度，不同样本的排队时间、prefill 时间、decode 时间都可能不同。理想情况下，每条样本要记录自己的 `start_time` 和 `end_time`，得到 request-level latency；而整体吞吐则用：
    ```text
    total_tokens / total_wall_time
    ```
    如果当前实现里是按 batch 记录耗时，那我会在面试里主动说明这是一个工程近似：
 当前版本如果是 batch-level timing，那么 p50/p95 更准确地说是 batch latency 的分位数，不能严格代表单样本 latency。后续应该把 timing 下沉到每条 request，记录 request_id、start_time、first_token_time、end_time，从而区分 queue latency、TTFT 和 total latency。
    这句话很重要，因为它体现你不是只会报指标，而是知道指标的语义边界。
    更完整的答案可以总结成：
 p50/p95 应该统计单样本 request-level latency，而不是 batch 平摊 latency。batch 平摊主要用于估算吞吐，不能代表真实延迟。工程上我会为每个样本记录 start/end time，并进一步记录 first token time，这样可以同时得到 total latency、TTFT、p50/p95 以及 tokens/s。如果现有版本采用 batch-level timing，我会明确标注它是 batch latency，并在后续改造成 request-level profiling。

28. tokens/s 是怎么计算的？输入 token 和输出 token 是否都算？
29. 不同任务 max_new_tokens 不同，如何避免生成长度影响性能比较？
30. SGLang 后端失败时 Transformers 降级方案怎么工作？
31. Transformers 后端为什么吞吐通常低于 SGLang？
32. 如果要支持多卡 tensor parallel，你的代码如何传递 tp_size？
33. tensor parallel 和 data parallel 在评测场景下分别适合什么情况？
34. 多数据集连续评测时，你如何避免重复加载模型？
35. 模型 session 复用可能带来什么状态污染问题？
36. 如果多个用户同时在 Gradio UI 发起评测，会有什么并发问题？
37. 当前系统是否支持断点续评？如果中途失败怎么办？
38. 如果一个数据集有 10 万样本，你会如何做流式处理和结果落盘？
39. 如何设计评测日志，便于复现实验？
40. 如果模型输出为空或推理报错，你的系统如何处理？

**三、结合代码仓库**

41. 我看到你 `Promptor` 和 `Referencer` 各自加载一次数据集，为什么不共享？
42. `build_refernce` 拼写有误，这类问题会影响什么？
43. `WikiEvent` 在配置里存在，但 prompt/reference 没实现，你怎么解释？
44. `2WikiMultihopQA` 数据里有空 answer，当前加载器会全跳过，这个问题怎么修？
45. `LEXam/MedQA/TriviaQA` 配置了但缺少对应 jsonl，如何保证评测入口和数据文件一致？
46. `T-REx` 被标成 Precision，但实际 reference 是 yes/no，这和 span metric 不匹配，你怎么看？
47. 当前 `TokenEfficiency` 用 split 估 token，而推理侧用 tokenizer 统计 token，这两个定义不一致，怎么改？
48. 你的 README 和 UI 中文出现乱码，是什么原因？如何修复？
49. requirements 里 `torch=2.9.1` 写法是否正确？
50. 如果让你把这个项目变成生产可用系统，你优先改哪三点？

**八、常见落地追问**

116. 如果让你为一个业务模型搭建离线评测体系，你会怎么设计？
117. 离线 benchmark 分数和线上用户体验不一致怎么办？
118. 如何设计一套模型上线前的准入标准？
119. 如何评估模型幻觉？
120. 如何评估模型安全性？
121. 如何评估长文本能力？
122. 如何评估 RAG 系统，是评估检索还是生成？
123. 如果线上模型变慢了，你会从哪些角度排查？
124. 如果模型效果下降但代码没变，可能是什么原因？
125. 如何做模型 A/B test？
126. 如何控制评测成本？
127. 如何保证评测结果可复现？
128. 如何判断一个 benchmark 是否已经不可靠？
129. 如果业务数据没有标注，怎么做自动评测？
130. 你觉得大模型算法工程师最核心的能力是什么？

**四、SCS 数据多样性科研**

51. 你的 SCS 指标解决了什么问题？
52. “现有指标会误判离群点”是什么意思？能举例说明吗？
53. 为什么 HDBSCAN 适合做语义聚类？
54. HDBSCAN 相比 KMeans 的优势是什么？
55. 语义凝聚权重是怎么定义的？
56. 为什么要融合空间距离和 LLM 生成概率？
57. LLM 生成概率如何表征样本内部质量？
58. 香农熵在你的方法里具体作用是什么？
59. 如果一个数据集类别很多但样本质量很低，SCS 会怎么反映？
60. 如果一个数据集质量很高但语义很单一，SCS 会怎么反映？
61. 你说 SCS 与性能相关系数 0.985，相关性是 Pearson 还是 Spearman？
62. 这个相关性是在多少组实验上算的？统计显著性如何？
63. 为什么只提升 SOTA 1.3%，这个提升是否显著？
64. 40w 指令数据怎么构建？如何去重、过滤毒性和低质量样本？
65. DataFlow 编排了哪些过滤算子？
66. 复现 4 种数据筛选方法和 12 种评估方法，分别是什么？
67. 如果 embedding 模型换掉，SCS 稳定吗？
68. HDBSCAN 超参如何选择？是否会影响结论？
69. 你的指标是否依赖强 LLM 计算生成概率？成本如何控制？
70. 如果面向中文数据，SCS 是否需要调整？


**七、算法基础和大模型基础**

101. Transformer 的 self-attention 复杂度是多少？
102. Multi-head attention 的作用是什么？
103. RoPE 的原理是什么？
104. KV Cache 的作用是什么？会占用多少显存？
105. LoRA 的核心公式是什么？
106. LoRA rank 怎么选？
107. SFT、DPO、RLHF 的区别是什么？
108. temperature、top-p、top-k 分别影响什么？
109. 为什么评测时通常 temperature=0？
110. beam search 和 greedy decoding 有什么区别？
111. perplexity 是否能直接衡量 instruction following 能力？
112. embedding 相似度常用 cosine，为什么不用欧氏距离？
113. BM25 的核心思想是什么？
114. cross-encoder 和 bi-encoder 的区别是什么？
115. HDBSCAN 的基本思想是什么？

