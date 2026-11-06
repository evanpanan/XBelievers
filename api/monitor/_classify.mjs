// 主动买卖分类与统计（纯计算，便于单独测试）

// Lee-Ready 报价规则 + Tick Rule 兜底；无法判断的标记为 U（未知）
export function classify(trades, quotes) {
  const qs = quotes
    .map((q) => ({ t: Date.parse(q.t), bp: q.bp, ap: q.ap }))
    .sort((a, b) => a.t - b.t);
  const ts = trades
    .map((t) => ({ t: Date.parse(t.t), p: t.p, s: t.s, x: t.x, c: t.c || [] }))
    .sort((a, b) => a.t - b.t);
  let qi = -1, lastPrice = null, lastTickSide = null;
  for (const tr of ts) {
    while (qi + 1 < qs.length && qs[qi + 1].t <= tr.t) qi++;
    const q = qi >= 0 ? qs[qi] : null;
    let side = 'U', method = 'none';
    const usable = q && q.bp > 0 && q.ap > 0 && q.ap >= q.bp && tr.t - q.t < 10 * 60 * 1000;
    if (usable) {
      const mid = (q.ap + q.bp) / 2;
      if (tr.p >= q.ap) side = 'B';
      else if (tr.p <= q.bp) side = 'S';
      else if (tr.p > mid + 1e-9) side = 'B';
      else if (tr.p < mid - 1e-9) side = 'S';
      if (side !== 'U') method = 'quote';
    }
    if (side === 'U' && lastPrice !== null) {
      if (tr.p > lastPrice) side = 'B';
      else if (tr.p < lastPrice) side = 'S';
      else if (lastTickSide) side = lastTickSide;
      if (side !== 'U') method = 'tick';
    }
    if (lastPrice !== null && tr.p !== lastPrice) lastTickSide = tr.p > lastPrice ? 'B' : 'S';
    lastPrice = tr.p;
    tr.side = side; tr.method = method;
    if (q) { tr.bid = q.bp; tr.ask = q.ap; }
  }
  return ts;
}

export const VENUES = {
  A: 'NYSE American', B: 'Nasdaq BX', C: 'NYSE National', D: '场外（FINRA TRF/ADF）',
  H: 'MIAX Pearl', I: 'ISE', J: 'Cboe EDGA', K: 'Cboe EDGX', L: 'LTSE', M: 'NYSE Chicago',
  N: 'NYSE', P: 'NYSE Arca', Q: 'Nasdaq', S: 'Nasdaq', T: 'Nasdaq', U: 'MEMX', V: 'IEX',
  W: 'CBSX', X: 'Nasdaq PSX', Y: 'Cboe BYX', Z: 'Cboe BZX',
};

const SIZE_BUCKETS = [
  { key: 'small', label: '小单（< 2,000美元）', max: 2000 },
  { key: 'mid', label: '中单（2,000–10,000美元）', max: 10000 },
  { key: 'large', label: '大单（10,000–50,000美元）', max: 50000 },
  { key: 'block', label: '特大单（≥ 50,000美元）', max: Infinity },
];

// 场外成交的亚分价格特征估算散户方向（Boehmer-Jones-Zhang-Zhang 方法）
function retailSide(tr) {
  if (tr.x !== 'D') return null;
  const cents = tr.p * 100;
  const frac = Math.round((cents - Math.floor(cents)) * 10000) / 10000;
  if (frac <= 0 || frac >= 1) return null;
  if (frac > 0 && frac < 0.4) return 'S';
  if (frac > 0.6 && frac < 1) return 'B';
  return null;
}

function emptyTotals() {
  return { buyVol: 0, sellVol: 0, unkVol: 0, buyDollar: 0, sellDollar: 0, unkDollar: 0, trades: 0, volume: 0, dollar: 0 };
}
function add(tot, tr) {
  const d = tr.p * tr.s;
  tot.trades++; tot.volume += tr.s; tot.dollar += d;
  if (tr.side === 'B') { tot.buyVol += tr.s; tot.buyDollar += d; }
  else if (tr.side === 'S') { tot.sellVol += tr.s; tot.sellDollar += d; }
  else { tot.unkVol += tr.s; tot.unkDollar += d; }
}
function finish(tot) {
  tot.delta = tot.buyVol - tot.sellVol;
  tot.dollarDelta = tot.buyDollar - tot.sellDollar;
  tot.buyPct = tot.volume ? tot.buyVol / tot.volume * 100 : null;
  tot.sellPct = tot.volume ? tot.sellVol / tot.volume * 100 : null;
  tot.unkPct = tot.volume ? tot.unkVol / tot.volume * 100 : null;
  return tot;
}

// regStart/regEnd: 正常交易时段（毫秒）；offsetSec: 图表用的美东偏移
export function aggregate(ts, { regStart, regEnd, offsetSec }) {
  const all = emptyTotals(), regular = emptyTotals(), ext = emptyTotals();
  const methods = { quote: 0, tick: 0, none: 0 };
  const minutes = new Map();
  const venues = {};
  const sizes = Object.fromEntries(SIZE_BUCKETS.map((b) => [b.key, { label: b.label, ...emptyTotals() }]));
  const retail = { buyVol: 0, sellVol: 0, trades: 0 };
  let oddLotVol = 0, vwapNum = 0, vwapDen = 0;

  for (const tr of ts) {
    add(all, tr);
    const isReg = tr.t >= regStart && tr.t < regEnd;
    add(isReg ? regular : ext, tr);
    if (isReg) { vwapNum += tr.p * tr.s; vwapDen += tr.s; }
    methods[tr.method]++;
    const mk = Math.floor(tr.t / 60000) * 60 + offsetSec;
    if (!minutes.has(mk)) minutes.set(mk, { t: mk, buy: 0, sell: 0, unk: 0, close: tr.p, vol: 0 });
    const m = minutes.get(mk);
    if (tr.side === 'B') m.buy += tr.s; else if (tr.side === 'S') m.sell += tr.s; else m.unk += tr.s;
    m.vol += tr.s; m.close = tr.p;
    const vk = VENUES[tr.x] || tr.x || '未知';
    venues[vk] = venues[vk] || { volume: 0, trades: 0 };
    venues[vk].volume += tr.s; venues[vk].trades++;
    const d = tr.p * tr.s;
    const b = SIZE_BUCKETS.find((x) => d < x.max);
    add(sizes[b.key], tr);
    if (tr.s < 100) oddLotVol += tr.s;
    const rs = retailSide(tr);
    if (rs) { retail.trades++; if (rs === 'B') retail.buyVol += tr.s; else retail.sellVol += tr.s; }
  }
  finish(all); finish(regular); finish(ext);
  for (const k in sizes) finish(sizes[k]);

  let cum = 0;
  const series = [...minutes.values()].sort((a, b) => a.t - b.t).map((m) => {
    cum += m.buy - m.sell;
    return { ...m, delta: m.buy - m.sell, cumDelta: cum };
  });

  // 价格分布（约30档）
  let profile = [];
  if (ts.length) {
    let lo = Infinity, hi = -Infinity;
    for (const t of ts) { lo = Math.min(lo, t.p); hi = Math.max(hi, t.p); }
    const step = Math.max(0.01, Math.ceil(((hi - lo) / 30) * 100) / 100);
    const base = Math.floor(lo * 100) / 100;
    const map = new Map();
    for (const t of ts) {
      const i = Math.floor((t.p - base) / step + 1e-9);
      if (!map.has(i)) map.set(i, { price: +(base + i * step).toFixed(4), buy: 0, sell: 0, unk: 0 });
      const r = map.get(i);
      if (t.side === 'B') r.buy += t.s; else if (t.side === 'S') r.sell += t.s; else r.unk += t.s;
    }
    profile = [...map.values()].sort((a, b) => b.price - a.price);
    profile.step = step;
  }

  // 最长的连续同向主动成交（最近200笔内）
  const recent = ts.slice(-200);
  let run = { side: null, n: 0, vol: 0 }, best = { side: null, n: 0, vol: 0 };
  for (const t of recent) {
    if (t.side !== 'U' && t.side === run.side) { run.n++; run.vol += t.s; }
    else run = { side: t.side === 'U' ? null : t.side, n: t.side === 'U' ? 0 : 1, vol: t.side === 'U' ? 0 : t.s };
    if (run.n > best.n) best = { ...run };
  }

  const large = [...ts].sort((a, b) => b.p * b.s - a.p * a.s).slice(0, 15)
    .sort((a, b) => b.t - a.t).map(slim);
  const tape = ts.slice(-40).reverse().map(slim);

  return {
    all, regular, ext, methods, series, profile, profileStep: profile.step || null,
    venues, sizes, retail, oddLotVol,
    vwap: vwapDen ? vwapNum / vwapDen : null,
    streak: best, large, tape,
    firstTrade: ts[0] ? new Date(ts[0].t).toISOString() : null,
    lastTrade: ts.at(-1) ? new Date(ts.at(-1).t).toISOString() : null,
  };
}

function slim(t) {
  return { t: new Date(t.t).toISOString(), p: t.p, s: t.s, side: t.side, method: t.method, venue: VENUES[t.x] || t.x, bid: t.bid, ask: t.ask };
}
