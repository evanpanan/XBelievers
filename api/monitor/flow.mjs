// 逐笔成交 → 主动买卖、Delta、价格分布、大单、场所、散户估算
// 免费方案：全市场 SIP 数据延迟15分钟；最近15分钟用 IEX 实时样本补充
import { SYMBOL, send, fetchJSON, alpacaHeaders, etParts, etToUtc, etOffsetMs, isWeekend, marketStatus , checkAuth } from './_lib.mjs';
import { classify, aggregate } from './_classify.mjs';

const BASE = `https://data.alpaca.markets/v2/stocks/${SYMBOL}`;

async function paged(kind, params, headers, maxPages, deadline) {
  const items = []; let token = null, pages = 0, truncated = false;
  do {
    const qs = new URLSearchParams({ ...params, limit: '10000' });
    if (token) qs.set('page_token', token);
    const j = await fetchJSON(`${BASE}/${kind}?${qs}`, { headers }, 9000);
    items.push(...(j[kind] || []));
    token = j.next_page_token; pages++;
    if (token && (pages >= maxPages || Date.now() > deadline)) { truncated = true; break; }
  } while (token);
  return { items, truncated };
}

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  const headers = alpacaHeaders();
  const now = new Date();
  if (!headers) {
    return send(res, {
      available: false,
      reason: '未配置 Alpaca 免费 API Key（ALPACA_KEY_ID / ALPACA_SECRET_KEY），逐笔成交分析暂不可用。',
    }, 's-maxage=60');
  }
  const deadline = Date.now() + 22000;
  const delayedEnd = new Date(now.getTime() - 16 * 60 * 1000);   // 免费方案：SIP 需早于15分钟
  const out = { available: true, generatedAt: now.toISOString(), status: marketStatus(now), errors: [] };

  try {
    // 找到最近一个有成交的交易日
    let session = null;
    for (let back = 0; back < 6 && !session; back++) {
      const day = new Date(now.getTime() - back * 86400000);
      const p = etParts(day);
      if (isWeekend(p.wd)) continue;
      const start = etToUtc(p.y, p.m, p.d, 4, 0);
      const close = etToUtc(p.y, p.m, p.d, 20, 0);
      const end = close < delayedEnd ? close : delayedEnd;
      if (end <= start) continue;
      const tr = await paged('trades', { start: start.toISOString(), end: end.toISOString(), feed: 'sip' }, headers, 6, deadline);
      if (tr.items.length) {
        session = { p, start, end, close, trades: tr.items, truncated: tr.truncated, isToday: back === 0 };
      }
    }
    if (!session) {
      out.errors.push('最近几个交易日没有拿到成交数据');
      return send(res, out, 's-maxage=30');
    }
    const { p } = session;
    const regStart = etToUtc(p.y, p.m, p.d, 9, 30).getTime();
    const regEnd = etToUtc(p.y, p.m, p.d, 16, 0).getTime();
    const offsetSec = etOffsetMs(session.start) / 1000;

    const qt = await paged('quotes', { start: session.start.toISOString(), end: session.end.toISOString(), feed: 'sip' }, headers, 8, deadline);
    const classified = classify(session.trades, qt.items);
    out.delayed = {
      feed: 'SIP 全市场（免费方案延迟约15分钟）',
      sessionDate: `${p.y}-${String(p.m).padStart(2, '0')}-${String(p.d).padStart(2, '0')}`,
      windowStart: session.start.toISOString(), windowEnd: session.end.toISOString(),
      tradesTruncated: session.truncated, quotesTruncated: qt.truncated,
      quoteCount: qt.items.length,
      ...aggregate(classified, { regStart, regEnd, offsetSec }),
    };
    out.offsetSec = offsetSec;

    // 最近15分钟：IEX 实时样本（只占全市场很小一部分）
    if (session.isToday && now < session.close && Date.now() < deadline) {
      try {
        const s = new Date(Math.max(session.start.getTime(), delayedEnd.getTime()));
        const params = { start: s.toISOString(), end: now.toISOString(), feed: 'iex' };
        const [t2, q2] = await Promise.all([
          paged('trades', params, headers, 2, deadline),
          paged('quotes', params, headers, 2, deadline),
        ]);
        const c2 = classify(t2.items, q2.items);
        out.live = { feed: 'IEX 实时（仅 IEX 一家交易所的成交，样本）', windowStart: s.toISOString(), ...aggregate(c2, { regStart, regEnd, offsetSec }) };
      } catch (e) { out.errors.push('IEX 实时样本：' + e.message); }
    }
  } catch (e) {
    out.errors.push(e.message);
    if (/403|subscription/i.test(e.message)) out.errors.push('提示：免费方案只能查询15分钟以前的 SIP 数据，或 Key 填写有误。');
  }
  send(res, out, 's-maxage=25, stale-while-revalidate=60');
}
