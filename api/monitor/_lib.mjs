// 公共工具：时区、请求、缓存。以下划线开头的文件不会被 Vercel 当成接口。
export const SYMBOL = 'XMAX';
export const OLD_SYMBOL = 'XWIN';           // 2026-05-18 前的旧代码
export const SYMBOL_CHANGE_DATE = '2026-05-18';
export const CIK = '0001473334';
export const UA_BROWSER =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';

export function secUA() {
  // SEC 要求 User-Agent 带真实联系邮箱，请在 Vercel 环境变量 SEC_USER_AGENT 里填写
  return process.env.SEC_USER_AGENT || 'XMAX IR Monitor ir-monitor@xmax.com';
}

export function alpacaHeaders() {
  const id = process.env.ALPACA_KEY_ID;
  const secret = process.env.ALPACA_SECRET_KEY;
  if (!id || !secret) return null;
  return { 'APCA-API-KEY-ID': id, 'APCA-API-SECRET-KEY': secret };
}

async function doFetch(url, opts = {}, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { ...opts, signal: ctrl.signal });
    if (!r.ok) {
      const body = await r.text().catch(() => '');
      const err = new Error(`HTTP ${r.status}: ${body.slice(0, 160)}`);
      err.status = r.status;
      throw err;
    }
    return r;
  } catch (e) {
    if (e.name === 'AbortError') throw new Error(`请求超时（${timeoutMs / 1000}秒）`);
    throw e;
  } finally {
    clearTimeout(timer);
  }
}
export async function fetchJSON(url, opts, timeoutMs) {
  const r = await doFetch(url, opts, timeoutMs);
  return r.json();
}
export async function fetchText(url, opts, timeoutMs) {
  const r = await doFetch(url, opts, timeoutMs);
  return r.text();
}

// ---------- 美东时间 ----------
const ET_FMT = new Intl.DateTimeFormat('en-US', {
  timeZone: 'America/New_York', hourCycle: 'h23', weekday: 'short',
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
});
export function etParts(date = new Date()) {
  const o = {};
  for (const p of ET_FMT.formatToParts(date)) o[p.type] = p.value;
  return { y: +o.year, m: +o.month, d: +o.day, h: +o.hour, mi: +o.minute, s: +o.second, wd: o.weekday };
}
// 美东相对 UTC 的偏移（毫秒，冬令时为 -5h）
export function etOffsetMs(date = new Date()) {
  const p = etParts(date);
  return Date.UTC(p.y, p.m - 1, p.d, p.h, p.mi, p.s) - Math.floor(date.getTime() / 1000) * 1000;
}
export function etToUtc(y, m, d, h = 0, mi = 0) {
  const guess = Date.UTC(y, m - 1, d, h, mi);
  let utc = guess - etOffsetMs(new Date(guess));
  const off2 = etOffsetMs(new Date(utc));
  if (guess - off2 !== utc) utc = guess - off2;
  return new Date(utc);
}
export function etDateStr(date = new Date()) {
  const p = etParts(date);
  return `${p.y}-${String(p.m).padStart(2, '0')}-${String(p.d).padStart(2, '0')}`;
}
export function isWeekend(wd) { return wd === 'Sat' || wd === 'Sun'; }

export function marketStatus(date = new Date()) {
  const p = etParts(date);
  const min = p.h * 60 + p.mi;
  if (isWeekend(p.wd)) return { code: 'closed', label: '休市（周末）' };
  if (min >= 240 && min < 570) return { code: 'pre', label: '盘前交易' };
  if (min >= 570 && min < 960) return { code: 'regular', label: '正常交易中' };
  if (min >= 960 && min < 1200) return { code: 'post', label: '盘后交易' };
  return { code: 'closed', label: '休市' };
}

export function send(res, body, cache = 's-maxage=15, stale-while-revalidate=60', status = 200) {
  res.setHeader('Cache-Control', cache);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.status(status).send(JSON.stringify(body));
}

// ---------- Yahoo 图表（非官方公开接口） ----------
export async function yahooChart(symbol, range, interval, prePost = false) {
  const qs = `range=${range}&interval=${interval}&includePrePost=${prePost}&events=div%2Csplit`;
  let lastErr;
  for (const host of ['query1', 'query2']) {
    try {
      const j = await fetchJSON(
        `https://${host}.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?${qs}`,
        { headers: { 'User-Agent': UA_BROWSER, Accept: 'application/json' } }, 8000);
      const r = j?.chart?.result?.[0];
      if (!r) throw new Error(j?.chart?.error?.description || 'Yahoo 无数据');
      const q = r.indicators?.quote?.[0] || {};
      const ts = r.timestamp || [];
      const bars = [];
      for (let i = 0; i < ts.length; i++) {
        if (q.close?.[i] == null || q.open?.[i] == null) continue;
        bars.push({ t: ts[i], o: q.open[i], h: q.high[i], l: q.low[i], c: q.close[i], v: q.volume?.[i] || 0 });
      }
      return { meta: r.meta || {}, bars };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// 访问密码：Vercel 环境变量 MONITOR_PASSWORD；请求需带 ?k=密码
export function checkAuth(req, res) {
  const pw = process.env.MONITOR_PASSWORD;
  if (!pw) return true;
  if (req.query && req.query.k === pw) return true;
  send(res, { error: '需要访问密码', auth: false }, 'no-store', 401);
  return false;
}
