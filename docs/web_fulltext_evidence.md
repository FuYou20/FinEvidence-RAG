# 博查搜索结果的网页正文增强

## 链路

```text
证据 Agent 请求联网补证
  -> 博查返回标题、URL、摘要
  -> 对前 N 个候选 URL 做安全检查
  -> 读取 HTML 并提取正文
  -> 成功：注册 web_fulltext 证据
  -> 失败：保留 search_summary 证据并记录失败类型
```

网页正文是外部不可信证据，正文中的指令不会改变系统 Prompt 或工具权限。

## 数据标记

正文读取成功：

```json
{
  "source_kind": "web",
  "content_origin": "web_fulltext",
  "extraction_status": "success",
  "search_summary": "博查原摘要",
  "text": "抽取后的网页正文"
}
```

正文读取失败：

```json
{
  "source_kind": "web",
  "content_origin": "search_summary",
  "extraction_status": "failed",
  "extraction_error_type": "RequestError",
  "text": "博查原摘要"
}
```

## 网络安全边界

- 只允许 HTTP/HTTPS 和标准端口；
- 禁止 URL 中携带用户名、密码；
- DNS 解析结果必须全部是公网地址；
- 每次重定向都会重新执行 URL 检查；
- 限制超时、响应大小、重定向次数和最终文本长度；
- 只接受 HTML、XHTML 和纯文本；
- 正文过短，或文章 URL 被重定向到站点首页时，判定为提取失败；
- 单页网络失败只降级该页，不中断整次调查；
- 未知编程异常不会被当成普通网络失败吞掉。

这是一层应用级 SSRF 防护。若未来部署到不可信公网环境，还应在网络层设置出口代理、域名/IP 策略和请求审计，以减少 DNS 重绑定等应用层检查难以彻底消除的风险。

## 配置

默认每次联网搜索读取前三个候选网页：

```bash
python -m src.scripts.multi_agent_demo --web-fulltext-limit 3
```

设为 `0` 可以保留博查搜索但不读取网页正文：

```bash
python -m src.scripts.multi_agent_demo --web-fulltext-limit 0
```

网页可能因为登录、付费墙、反爬、JavaScript 动态渲染或内容格式不支持而退回搜索摘要。这是正常的可观察降级，不应伪装成全文读取成功。
