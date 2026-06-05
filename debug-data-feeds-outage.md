# Debug Session: data-feeds-outage

- Status: OPEN
- Started At: 2026-06-01
- Scope:
  - XMAX 新闻无数据
  - 马斯克系新闻无数据
  - 推特监控不新不全

## Hypotheses

1. 前端请求未命中新本地后端路由，导致 404 或命中旧服务。
2. 后端路由可达，但上游 RSS / 订阅源返回异常或被拒绝，解析结果为空。
3. 前后端数据字段不一致，前端渲染层将有效响应误判为空。
4. 推特监控 fallback 未正确触发，或时间字段导致排序异常。
5. 本地运行环境中的 API_BASE / 服务端口与当前页面不一致。

## Evidence Log

- Pre-fix evidence:
  - 当前运行中的本地服务进程 PID 88507 于 2026-05-26 启动，占用 5173，说明页面命中的是旧后端进程。
  - `GET /api/xmax-news` on `127.0.0.1:5173` => 404
  - `GET /api/musk-news` on `127.0.0.1:5173` => 404
  - `GET /api/twitter-monitor` on `127.0.0.1:5173` => 404
  - Debug log lines 1-14: 前端对三个新接口均拿到 404。
- Instrumented server evidence on updated code (`127.0.0.1:5174`):
  - `GET /api/xmax-news` => 200, `count=1`
  - `GET /api/musk-news` => 200, `count=10`
  - `GET /api/twitter-monitor` => RSSHub upstream 404，进入 fallback
  - Debug log lines 15-20: `/api/xmax-news` 与 `/api/musk-news` 正常出数；`/api/twitter-monitor` 上游错误为 `upstream_status=404`。
- Post-fix verification:
  - 前端新增本地 API 自动回退到 `127.0.0.1:5174` 的逻辑，可绕过仍在运行的旧 5173 服务。
  - `/api/twitter-monitor` 新增 Jina Reader fallback。
  - Recheck on `127.0.0.1:5174`:
    - `GET /api/xmax-news` => 200, `count=1`
    - `GET /api/musk-news` => 200, `count=10`
    - `GET /api/twitter-monitor` => 200, `source=JinaReaderFallback`, `count=4`

## Next Step

- 等待用户刷新页面确认三路数据是否恢复；在用户确认前保留调试日志与调试服务。
