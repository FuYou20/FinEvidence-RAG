# FinEvidence A2A 服务

FinEvidence 通过 A2A 1.0 JSON-RPC 对外提供证据调查能力。A2A 层只负责
Agent 发现、任务生命周期和协议转换，检索与补证仍由 `EvidenceRuntime`
完成。

## 启动

```bash
python -m src.scripts.a2a_evidence_server
```

默认地址：

- 健康检查：`GET http://127.0.0.1:9999/health`
- Agent Card：`GET http://127.0.0.1:9999/.well-known/agent-card.json`
- A2A JSON-RPC：`POST http://127.0.0.1:9999/a2a`

运行示例客户端：

```bash
python -m src.scripts.a2a_evidence_client
```

## 请求协议

推荐使用 A2A DataPart 传递 `EvidenceRequest`：

```json
{
  "request_id": "req-001",
  "requester": "research-agent",
  "mode": "stance_research",
  "topic": "传音控股在非洲市场具备哪些竞争优势？",
  "direction": "support",
  "research_focus": ["本地化", "渠道", "产品"],
  "max_deepening_steps": 1
}
```

调试时也可以发送内容为上述 JSON 的 TextPart；普通自然语言不会被隐式猜测
为某种调查模式，而会返回 `TASK_STATE_FAILED`。

## 返回协议

一次成功调用按顺序产生：

1. `Task`：服务端接受任务；
2. `TASK_STATE_WORKING`：证据调查正在执行；
3. `evidence_bundle` Artifact：结构化 `EvidenceBundle`；
4. `TASK_STATE_COMPLETED`：调查完成。

证据不足不是协议失败。此时任务仍然完成，但 Artifact 中：

```json
{
  "status": "insufficient",
  "missing_information": ["缺少可验证的最新经营数据"]
}
```

只有参数非法或运行异常才进入 `TASK_STATE_FAILED`。

## 配置

```dotenv
A2A_HOST=127.0.0.1
A2A_PORT=9999
A2A_PUBLIC_URL=http://127.0.0.1:9999/a2a
A2A_ENABLE_WEB=false
A2A_MAX_DEEPENING_STEPS=2
A2A_MAX_CONCURRENCY=1
```

当前 `EvidenceRuntime` 内部包含可变的证据注册表和指标，因此默认将单进程
并发限制为 1。横向扩展时应启动多个独立服务进程，并依靠 Redis 证据缓存及
分布式锁避免重复调查。

当前 A2A TaskStore 使用内存实现，服务重启后任务状态不会保留；证据缓存、
FAISS 索引和项目原有 SQLite 记忆不受影响。需要支持长时间异步任务恢复时，
再将 A2A TaskStore 替换为持久化实现。
