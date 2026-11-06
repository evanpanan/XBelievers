// 实时报价、最近交易日分时、RVOL、大盘对比
// 用最近5天的1分钟数据，按K线自己的美东日期切分交易时段，休市、盘前、盘中、盘后都能正确显示
import { SYMBOL, send, yahooChart, marketStatus, etParts, etToUtc, alpacaHeaders, fetchJSON , checkAuth } from './_lib.mjs';

const INDEXES = [
  { symbol: 'SPY', name: '标普500 ETF' },
  { symbol: 'QQQ', name: '纳指100 ETF' },
  { symbol: 'IWM', name: '罗素2000小盘 ETF' },
];

const dayKey = (sec) => { const p = etParts(new Date(sec * 1000)); return `${p.y}-${String(p.m).padStart(2, '0')}-${String(p.d).padStart(2, '0')}`; };
const minOfDay = (sec) => { const p = etParts(new Date(sec * 1000)); return p.h * 60 + p.mi; };
const isReg = (b) => { const m = minOfDay(b.t); return m >= 570 && m < 960; };

// 把多日K线切成：最近有正常交易的那天（含之后的盘前）、它的前收
function sessionize(bars) {
  const byDay = new Map();
  for (const b of bars) { const k = dayKey(b.t); if (!byDay.has(k)) byDay.set(k, []); byDay.get(k).push(b); }
  const days = [...byDay.keys()].sort();
  const regDays = days.filter((d) => byDay.get(d).some(isReg));
  const lastReg = regDays.at(-1);
  const prevReg = regDays.at(-2);
  if (!lastReg) return null;
  const regBars = byDay.get(lastReg).filter(isReg);
  const prevClose = prevReg ? byDay.get(prevReg).filter(isReg).at(-1).c : null;
  const [y, m, d] = lastReg.split('-').map(Number);
  const regStart = etToUtc(y, m, d, 9, 30).getTime() / 1000;
  const regEnd = etToUtc(y, m, d, 16, 0).getTime() / 1000;
  const chartBars = bars.filter((b) => dayKey(b.t) >= lastReg);       // 最近交易日 + 之后的盘前
  const nextDayBars = chartBars.filter((b) => dayKey(b.t) > lastReg);  // 新一天的盘前
  return { lastReg, regBars, prevClose, regStart, regEnd, chartBars, nextDayBars, dayBars: byDay.get(lastReg) };
}

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  const now = new Date();
  const out = { symbol: SYMBOL, generatedAt: now.toISOString(), status: marketStatus(now), source: null, errors: [] };

  const [intra, m5, ...idx] = await Promise.allSettled([
    yahooChart(SYMBOL, '5d', '1m', true),
    yahooChart(SYMBOL, '1mo', '5m', false),
    ...INDEXES.map((i) => yahooChart(i.symbol, '5d', '5m', false)),
  ]);

  if (intra.status === 'fulfilled') {
    const { meta, bars } = intra.value;
    const s = sessionize(bars);
    if (s) {
      let vol = 0, dollar = 0, hi = -Infinity, lo = Infinity;
      for (const b of s.regBars) { vol += b.v; dollar += b.v * (b.h + b.l + b.c) / 3; hi = Math.max(hi, b.h); lo = Math.min(lo, b.l); }
      // 收盘竞价价格以 Yahoo 官方报价为准（同一交易日时）
      const metaDay = meta.regularMarketTime ? dayKey(meta.regularMarketTime) : null;
      const price = metaDay === s.lastReg && meta.regularMarketPrice ? meta.regularMarketPrice : s.regBars.at(-1).c;
      const pre = s.dayBars.filter((b) => b.t < s.regStart);
      const post = s.dayBars.filter((b) => b.t >= s.regEnd);
      Object.assign(out, {
        source: 'Yahoo Finance（非官方公开接口，通常有数秒至数分钟延迟）',
        sessionDate: s.lastReg,
        price, prevClose: s.prevClose,
        change: s.prevClose ? price - s.prevClose : null,
        changePct: s.prevClose ? (price - s.prevClose) / s.prevClose * 100 : null,
        open: s.regBars[0].o, high: hi, low: lo,
        volume: vol, dollarVolume: dollar, vwapApprox: vol ? dollar / vol : null,
        prePrice: pre.at(-1)?.c ?? null, preVolume: pre.reduce((a, b) => a + b.v, 0),
        postPrice: post.at(-1)?.c ?? null, postVolume: post.reduce((a, b) => a + b.v, 0),
        nextPrePrice: s.nextDayBars.at(-1)?.c ?? null,
        nextPreChangePct: s.nextDayBars.length ? (s.nextDayBars.at(-1).c / price - 1) * 100 : null,
        lastPrice: bars.at(-1)?.c ?? price, lastBarTime: bars.at(-1)?.t ?? null,
        fiftyTwoWeekHigh: meta.fiftyTwoWeekHigh ?? null, fiftyTwoWeekLow: meta.fiftyTwoWeekLow ?? null,
        regularStart: s.regStart, regularEnd: s.regEnd,
        bars: s.chartBars,
      });
    } else out.errors.push('Yahoo 分时：最近5天没有正常时段成交');
  } else out.errors.push('Yahoo 分时：' + intra.reason.message);

  if (out.price == null && alpacaHeaders()) {
    try {
      const s = await fetchJSON(`https://data.alpaca.markets/v2/stocks/${SYMBOL}/snapshot?feed=iex`, { headers: alpacaHeaders() });
      const prevClose = s.prevDailyBar?.c ?? null, price = s.latestTrade?.p ?? s.dailyBar?.c ?? null;
      Object.assign(out, {
        source: 'Alpaca IEX 快照（仅 IEX 一家交易所，成交量远小于全市场）', price, prevClose,
        change: price != null && prevClose ? price - prevClose : null,
        changePct: price != null && prevClose ? (price - prevClose) / prevClose * 100 : null,
        open: s.dailyBar?.o ?? null, high: s.dailyBar?.h ?? null, low: s.dailyBar?.l ?? null, bars: [],
      });
      out.errors.push('当前价格来自 IEX 兜底数据，成交量不代表全市场');
    } catch (e) { out.errors.push('Alpaca 快照：' + e.message); }
  }

  // RVOL：与过去最多20个交易日"同一时刻"的累计成交量比较
  if (m5.status === 'fulfilled') {
    const days = new Map();
    for (const b of m5.value.bars) { const k = dayKey(b.t); if (!days.has(k)) days.set(k, []); days.get(k).push(b); }
    const keys = [...days.keys()].sort();
    const latest = keys.at(-1);
    const nowP = etParts(now);
    let cutoff = nowP.h * 60 + nowP.mi;
    if (latest !== dayKey(now.getTime() / 1000) || out.status.code !== 'regular') cutoff = 960;
    cutoff = Math.max(575, Math.min(960, cutoff));
    const cum = (arr) => arr.filter((b) => minOfDay(b.t) < cutoff).reduce((s, b) => s + b.v, 0);
    const prior = keys.slice(0, -1).slice(-20);
    const priorCum = prior.map((k) => cum(days.get(k)));
    const priorFull = prior.map((k) => days.get(k).reduce((s, b) => s + b.v, 0));
    const avg = priorCum.length ? priorCum.reduce((a, b) => a + b, 0) / priorCum.length : null;
    out.rvol = avg ? cum(days.get(latest)) / avg : null;
    out.rvolDetail = { cutoffMinute: cutoff, sampleDays: prior.length, avgCum: avg,
      avgDailyVolume: priorFull.length ? priorFull.reduce((a, b) => a + b, 0) / priorFull.length : null };
  } else out.errors.push('RVOL 历史：' + m5.reason.message);

  out.indexes = INDEXES.map((ix, i) => {
    const r = idx[i];
    if (r.status !== 'fulfilled') return { ...ix, error: r.reason.message };
    const s = sessionize(r.value.bars);
    if (!s) return { ...ix, error: '无数据' };
    const meta = r.value.meta;
    const p = meta.regularMarketTime && dayKey(meta.regularMarketTime) === s.lastReg && meta.regularMarketPrice ? meta.regularMarketPrice : s.regBars.at(-1).c;
    const pc = s.prevClose;
    return { ...ix, price: p, prevClose: pc, changePct: pc ? (p - pc) / pc * 100 : null,
      sessionDate: s.lastReg, series: s.regBars.map((b) => ({ t: b.t, pct: pc ? (b.c - pc) / pc * 100 : null })) };
  });

  send(res, out, 's-maxage=10, stale-while-revalidate=30');
}
