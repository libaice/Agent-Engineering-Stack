# AutoGen & Microsoft Agent Framework 完整指南

---

## 1. Why AutoGen Matters

AutoGen 诞生于微软研究院，最初是为了解决一个核心问题：**单个 LLM 调用已经不足以处理复杂的现实任务**。它的核心洞察是：让多个 AI Agent 分工协作、相互对话，比让一个超大 prompt 独自完成任务更可靠、更可调试。

AutoGen 的三层架构体现了不同抽象层次的价值：

**Core（核心层）** 是事件驱动的异步运行时，适合有经验的工程师构建可扩展的分布式多智能体系统。它的编程模型更灵活，允许开发者精确控制消息路由和 Agent 行为，支持跨语言（Python、.NET）的分布式 Agent 部署。

**AgentChat（会话层）** 是面向大多数开发者的高层 API，内置了 `AssistantAgent`、`UserProxyAgent` 等预设 Agent 和多种 Team 模式（`RoundRobinGroupChat`、`SelectorGroupChat`、`Swarm` 等），开箱即用。

**Studio（无代码层）** 是基于 Web 的 UI，允许完全不写代码地原型化 Agent 系统，适合产品经理和初次探索者。

AutoGen 之所以重要，在于它将以下概念变成了工程实践：反思（Reflection）、工具调用（Tool Use）、Human-in-the-loop、多 Agent 协作——这些在 2023 年之前还停留在学术论文中的模式，AutoGen 提供了可运行的参考实现，直接影响了整个 AI Agent 领域的工程化路径。

---

## 2. AutoGen Conversation Patterns

AutoGen AgentChat 内置了五种核心对话模式，分别对应不同的协作场景：

**RoundRobinGroupChat（轮询对话）**
所有 Agent 共享同一消息上下文，按照固定顺序轮流发言。每个 Agent 发言后，消息广播给所有其他成员。这是最简单的 Team 配置，经典用例是"生成 + 反思"的双 Agent 循环：一个 primary Agent 生成内容，一个 critic Agent 提供反馈，直到触发终止条件（如 critic 输出 `APPROVE`）。

```python
team = RoundRobinGroupChat([primary_agent, critic_agent],
    termination_condition=TextMentionTermination("APPROVE"))
await team.run(task="Write a short poem about the fall season.")
```

**SelectorGroupChat（选择器对话）**
与轮询不同，每轮对话结束后，由一个 ChatCompletion 模型决定下一个发言的 Agent 是谁。适合任务路由场景，比如将用户请求动态分发给最合适的专家 Agent。

**Swarm（蜂群，局部决策交接）**
每个 Agent 通过发送 `HandoffMessage` 自主决定将任务交给谁，无需中央调度器。所有 Agent 共享消息上下文，Agent 可以将任务交给其他 Agent 或交还给用户（触发暂停等待输入）。适合流程中包含多角色专家协作和人机交互节点的场景，如退款处理、证券研究：

```
用户请求 → travel_agent 路由 → flights_refunder 处理 → 用户补充信息 → flights_refunder 完成 → TERMINATE
```

**MagenticOneGroupChat（通用多 Agent 系统）**
MagenticOne 是 AutoGen 内置的开放式任务求解系统，擅长处理跨域的 Web 和文件任务。它包含一个 Orchestrator（统筹规划）以及 WebSurfer、FileSurfer、Coder、ComputerTerminal 等专用 Agent。

**GraphFlow（图式工作流）**
通过有向图定义 Agent 执行路径，支持顺序（Sequential）、并发（Concurrent）、条件分支等模式。

**终止条件（Termination Conditions）**
AutoGen 提供了丰富的终止条件，并支持通过 `|`（OR）和 `&`（AND）组合：

| 条件类型 | 触发时机 |
|---|---|
| `TextMentionTermination` | 消息中出现指定文本（如 "APPROVE"、"TERMINATE"） |
| `MaxMessageTermination` | 消息总数达到上限 |
| `TimeoutTermination` | 运行时间超出设定值 |
| `HandoffTermination` | 某 Agent 发起交接到指定目标时 |
| `ExternalTermination` | 外部程序调用 `.set()` 主动停止 |
| `FunctionCallTermination` | 某特定工具函数被执行时 |

---

## 3. Why Microsoft Agent Framework

2026 年 4 月 2 日，微软正式发布 **Microsoft Agent Framework (MAF) 1.0 GA**，将 AutoGen 和 Semantic Kernel 合并为一个统一框架。这次整合的动机来自三个现实痛点：

**研究到生产的鸿沟** — AutoGen 起源于微软研究院，设计上偏向灵活性和实验性，缺乏生产级所需的会话管理、中间件、类型安全等企业特性。Semantic Kernel 拥有这些企业能力，但缺少 AutoGen 的多 Agent 编排抽象。两者并行维护造成了资源分散和开发者选择困难。

**统一的工程化路径** — MAF 的设计原则是"从原型到生产"：它保留了 AutoGen 的多 Agent 对话模式，融合了 Semantic Kernel 的会话状态管理（Sessions）、中间件流水线（Middleware）、可观测性（OpenTelemetry）和类型安全，形成一个连贯的开发体验。

**开放标准的拥抱** — MAF 原生支持 MCP（Model Context Protocol）用于工具集成，以及 A2A（Agent-to-Agent）协议用于跨平台跨组织的 Agent 通信（A2A v1 技术指导委员会成员包括 AWS、Google、IBM、Salesforce、SAP、ServiceNow 等），避免了厂商锁定。

**目前的状态**：
- AutoGen 和 Semantic Kernel 进入**维护模式**（仅接受 bug 修复和安全补丁）
- 微软承诺在 MAF GA 后至少维护 SK v1.x 一年
- 新特性投资全部集中在 Agent Framework

---

## 4. Mini AutoGen Pattern Mapping

以下是 AutoGen 核心对话模式到 MAF 工作流模式的映射关系：

| AutoGen 模式 | MAF 对应模式 | 核心机制 | 典型场景 |
|---|---|---|---|
| `RoundRobinGroupChat` | Sequential Workflow | 固定顺序轮流，共享上下文 | 生成→审核→修改循环 |
| `SelectorGroupChat` | Group Chat with Selector | 模型动态选择下一发言者 | 专家路由、意图分发 |
| `Swarm` + `HandoffMessage` | Handoff Orchestration | Agent 自主决策交接，局部控制 | 多角色流程（客服、退款）|
| `MagenticOneGroupChat` | Orchestrator + Specialist Agents | 统筹规划者 + 专用执行者 | 开放式复杂任务（Web/文件） |
| `GraphFlow` | Graph-based Workflows | 有向图，支持顺序/并发/条件 | 确定性业务流程 |
| `AssistantAgent` (single) | Single `Agent` with tools | 工具调用循环，`max_tool_iterations` | 工具驱动的单次任务 |
| `ExternalTermination` | Human-in-the-loop / Approval | 暂停等待外部输入 | 人工审批节点 |
| `BufferedChatCompletionContext` | Context Providers / Memory | 限制上下文窗口大小 | 长对话 token 控制 |

**模式选择决策树：**

```
任务是否需要多个 Agent？
├─ 否 → 单 Agent + Tools (max_tool_iterations)
└─ 是 → 执行路径是否固定？
    ├─ 是 → GraphFlow (Sequential/Concurrent)
    └─ 否 → Agent 是否需要自主决定交接？
         ├─ 是 → Swarm / Handoff Pattern
         └─ 否 → 谁来选择下一个 Agent？
              ├─ LLM 动态决定 → SelectorGroupChat / Group Chat
              └─ 固定轮询 → RoundRobinGroupChat
```

---

## 5. MAF vs LangGraph

根据 LangChain 官方 2026 年框架对比报告及社区反馈，两者的核心差异如下：

**状态管理模型**

LangGraph 使用显式的图节点 + 状态对象，开发者精确控制状态的读取和写入，适合需要细粒度状态追踪的复杂 Agent；MAF 使用 Sessions（会话）管理多轮对话历史，并通过 Checkpointing 支持工作流的持久化和恢复，状态管理更隐式但更轻量。

**语言生态**

LangGraph 是 Python-first，TypeScript 版本为 LangGraph.js；MAF 同时提供 Python 和 C#/.NET 一等公民支持，API 设计在两端保持一致，对于 .NET 企业用户是显著优势。

**云平台集成**

MAF 与 Azure AI Foundry、Azure OpenAI、Microsoft 365 深度集成，支持一键部署为 Foundry Hosted Agents（自动获得 Entra ID 身份、自动扩缩容、会话持久化）；LangGraph 则与 LangSmith（可观测性）紧密捆绑，Cloud 版本可托管在 LangChain 的基础设施上。

| 维度 | MAF | LangGraph |
|---|---|---|
| 语言支持 | Python + C#/.NET（同等一等公民） | Python-first，JS 二等 |
| 状态模型 | Sessions + Checkpointing | 显式 Graph State |
| 工作流范式 | 图式 + 会话式 + Handoff | 图式（DAG + 循环图） |
| 云托管 | Azure Foundry（一键部署） | LangSmith Cloud |
| 可观测性 | 内置 OpenTelemetry → App Insights | LangSmith（更完善的 UI） |
| 责任 AI | Azure AI Foundry 内置内容安全、PII 保护 | 需自行集成 |
| 多 Agent 模式 | Sequential/Concurrent/Handoff/GroupChat 内置 | 通过图节点手动构建 |
| 企业治理 | Agent Governance Toolkit (AGT) 原生集成 | 需第三方工具 |
| 适合团队 | Microsoft 栈企业团队，.NET 团队 | Python 团队，需精细控制状态 |
| GitHub Stars (2026) | ~9.6k | ~29.1k（更成熟社区） |

**社区和成熟度**：LangGraph 目前拥有更大的社区和更多的生产案例积累；MAF 1.0 GA 于 2026 年 4 月发布，生态尚在快速建设中，但背靠微软的企业支持体系。

**一句话总结**：选 LangGraph，如果你需要精细控制 Python 图式状态；选 MAF，如果你在 Microsoft/Azure 栈上，或者需要同时支持 .NET 和 Python，或者需要开箱即用的企业治理和部署能力。

---

## 6. Enterprise Use Cases

MAF 在发布 AGT（Agent Governance Toolkit）集成文档中明确列出了五个企业级行业场景，代表了当前最成熟的落地方向：

**金融 — 贷款审批流程（Loan Processing）**
多 Agent 工作流处理贷款申请：信用评分 Agent 调用外部信评 API，合规 Agent 检查 PII 数据是否泄露，审批 Agent 按金额设置门控（小额自动审批、大额人工确认），Rogue Detection 检测异常转账行为。治理层保证每一步工具调用都有策略审计记录。

**零售 — 客户服务自动化（Customer Service）**
多 Agent 处理退款申请：Triage Agent 分类请求，Refund Agent 处理退款逻辑（防欺诈规则内嵌），支付数据保护中间件阻止 Agent 接触原始卡号。人工升级规则在退款金额超阈值时触发 `HandoffTermination` 交给人工坐席。

**医疗 — 临床助手（Healthcare）**
遵循 HIPAA 要求的 PHI（受保护健康信息）自动过滤，不同部门 Agent 之间的信息隔离（跨科室 Agent 不能访问对方患者数据），处方安全规则防止高风险药物自动下单。

**企业 IT — 智能工单（IT Helpdesk）**
权限提升防护（Agent 不能自行提升自己的权限级别），凭据隔离（密钥、Token 不出现在 Agent 消息中），基础设施保护（破坏性操作需要人工确认）。

**DevOps — 自动化部署（DevOps Deploy）**
生产环境部署门控（必须经过 CI/CD 验证才能触发），破坏性操作拦截（`DROP TABLE`、`rm -rf` 等命令触发策略阻断），部署风暴检测（短时间内异常频繁部署触发 throttle）。

**横向能力支撑**：

上述场景都共同依赖 MAF 的几个横向能力：
- **A2A 协议**：跨平台、跨组织的 Agent 通信，例如与外部合作方的 Agent 对接
- **Foundry Hosted Agents**：每个 Agent Session 拥有独立的 VM 沙箱，自动扩缩容，Session 状态跨 scale-to-zero 持久化
- **FIDES（Flow Integrity Deterministic Enforcement System）**：给所有流经 Agent 的内容打上 trusted/untrusted 标签，防止 Prompt Injection 攻击（OWASP LLM Top 10 #1 风险）
- **集体策略引擎（Collective Policy）**：多 Agent 工作流的全局约束（如全局 API 调用总量限制），而非只约束单个 Agent

---

## 7. Eval Strategy

Agent 系统的评估不同于传统 LLM 评估，需要覆盖**单步行为**和**端到端流程**两个层次。以下是结合 MAF 生态的分层评估策略：

**Layer 1：单 Agent / 单步评估**

评估 Agent 在单次调用中的行为质量，主要关注：
- **工具调用准确性**：给定任务，Agent 是否调用了正确的工具？参数是否正确？（可对比 expected tool call 和 actual tool call）
- **结构化输出质量**：使用 `output_content_type=AgentResponse`（Pydantic 模型）验证输出格式合规性，结合模型评估内容质量
- **反思有效性**：在启用 `reflect_on_tool_use=True` 时，评估反思摘要是否准确

**Layer 2：多轮对话 / Team 评估**

评估多 Agent 协作流程的整体效果：
- **任务完成率（Task Completion Rate）**：在测试集上，Team 是否正确触发了终止条件（而不是无限循环或提前停止）？
- **交接准确性（Handoff Accuracy）**：在 Swarm 模式中，每次 Handoff 是否路由到了正确的 Agent？
- **终止条件触发质量**：`APPROVE`/`TERMINATE` 是在任务真正完成时触发，还是被错误提前触发？

**Layer 3：端到端业务指标评估**

- **LLM-as-a-Judge**：用一个独立的评估 LLM 对 Agent 的最终输出打分（任务遵循度、完整性、安全性），MAF 原生通过 OpenTelemetry + Azure AI Foundry 提供 `task_adherence` 评分
- **Golden Dataset 回归**：维护一个覆盖正常路径 + 边缘案例的标注数据集，每次模型或 prompt 变更后自动运行回归测试
- **代价感知评估（Cost-aware Eval）**：统计每次完成任务的 token 消耗、工具调用次数、耗时，防止优化准确性的同时爆炸式提高成本

**MAF 生态中的评估工具链**：

```
Agent 执行
   ↓ OpenTelemetry Traces
Azure App Insights / LangSmith
   ↓ 采样 + 标注
评估数据集（Golden Dataset）
   ↓ LLM-as-a-Judge / Rule-based Evals
评估报告（task adherence, PII, accuracy）
   ↓ 反馈
Prompt / 工具 / 流程改进
```

MAF 的 AGT 还提供了 **Decision BOM（决策物料清单）**，为每个 Agent 动作生成可追溯的决策依据链（包括信任快照、策略评估结果、执行轨迹），以 Merkle 链方式防篡改，可直接用于合规审计——这在金融和医疗场景中填补了传统 LLM 评估体系的重要空白。

**Eval 的快速落地建议**：先从 `TextMentionTermination` + `MaxMessageTermination` 组合的日志中提取真实运行轨迹，手动标注 100 个样例作为 Golden Dataset，再用 LLM-as-a-Judge 自动扩展到更大规模的回归测试集。

