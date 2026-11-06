// 做空监测：FINRA 每日卖空成交量 + 双周 Short Interest + SEC 总股本
import { SYMBOL, OLD_SYMBOL, SYMBOL_CHANGE_DATE, CIK, send, fetchText, fetchJSON, etParts, secUA, UA_BROWSER , checkAuth } from './_lib.mjs';

function lastWeekdays(n) {
  const out = []; let d = new Date();
  while (out.length < n) {
    const p = etParts(d);
    if (p.wd !== 'Sat' && p.wd !== 'Sun') out.push(`${p.y}${String(p.m).padStart(2, '0')}${String(p.d).padStart(2, '0')}`);
    d = new Date(d.getTime() - 86400000);
  }
  return out;
}

async function dailyShortVolume() {
  const days = lastWeekdays(22);
  const results = await Promise.allSettled(days.map(async (ymd) => {
    const txt = await fetchText(`https://cdn.finra.org/equity/regsho/daily/CNMSshvol${ymd}.txt`, {}, 8000);
    const iso = `${ymd.slice(0, 4)}-${ymd.slice(4, 6)}-${ymd.slice(6)}`;
    const sym = iso < SYMBOL_CHANGE_DATE ? OLD_SYMBOL : SYMBOL;
    const line = txt.split('\n').find((l) => l.split('|')[1] === sym);
    if (!line) return { date: iso, symbol: sym, missing: true };
    const [, , shortV, exemptV, totalV] = line.trim().split('|');
    const s = +shortV, t = +totalV;
    return { date: iso, symbol: sym, short: s, exempt: +exemptV, total: t, ratio: t ? s / t * 100 : null };
  }));
  return results.filter((r) => r.status === 'fulfilled' && !r.value.missing).map((r) => r.value)
    .sort((a, b) => a.date.localeCompare(b.date));
}

const num = (v) => (v == null || v === '' ? null : +String(v).replace(/[,$%]/g, ''));

async function finraShortInterest() {
  const rows = [];
  for (const sym of [SYMBOL, OLD_SYMBOL]) {
    const j = await fetchJSON('https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        limit: 30,
        compareFilters: [{ compareType: 'equal', fieldName: 'symbolCode', fieldValue: sym }],
        sortFields: ['-settlementDate'],
      }),
    }, 8000);
    for (const r of Array.isArray(j) ? j : []) {
      rows.push({
        date: r.settlementDate, symbol: r.symbolCode,
        shortInterest: num(r.currentShortPositionQuantity),
        previous: num(r.previousShortPositionQuantity),
        avgDailyVolume: num(r.averageDailyVolumeQuantity),
        daysToCover: num(r.daysToCoverQuantity),
        changePct: num(r.changePercent),
      });
    }
  }
  if (!rows.length) throw new Error('FINRA 未返回数据');
  return { source: 'FINRA 官方 Short Interest（双周，结算日后约7个工作日发布）', rows: rows.sort((a, b) => b.date.localeCompare(a.date)) };
}

async function nasdaqShortInterest() {
  const j = await fetchJSON(`https://api.nasdaq.com/api/quote/${SYMBOL}/short-interest?assetClass=stocks`, {
    headers: { 'User-Agent': UA_BROWSER, Accept: 'application/json', Origin: 'https://www.nasdaq.com', Referer: 'https://www.nasdaq.com/' },
  }, 8000);
  const rows = (j?.data?.shortInterestTable?.rows || []).map((r) => {
    const [m, d, y] = String(r.settlementDate).split('/');
    return {
      date: y ? `${y}-${m.padStart(2, '0')}-${d.padStart(2, '0')}` : r.settlementDate,
      shortInterest: num(r.interest), avgDailyVolume: num(r.avgDailyShareVolume), daysToCover: num(r.daysToCover),
    };
  });
  if (!rows.length) throw new Error('Nasdaq 未返回数据');
  for (let i = 0; i < rows.length - 1; i++) {
    rows[i].previous = rows[i + 1].shortInterest;
    rows[i].changePct = rows[i + 1].shortInterest ? (rows[i].shortInterest / rows[i + 1].shortInterest - 1) * 100 : null;
  }
  return { source: 'Nasdaq.com 公开页面接口（非官方）', rows };
}

async function sharesOutstanding() {
  const j = await fetchJSON(`https://data.sec.gov/api/xbrl/companyconcept/CIK${CIK}/dei/EntityCommonStockSharesOutstanding.json`,
    { headers: { 'User-Agent': secUA(), Accept: 'application/json' } }, 8000);
  const arr = j?.units?.shares || [];
  const latest = [...arr].sort((a, b) => (a.end + a.filed).localeCompare(b.end + b.filed)).at(-1);
  return latest ? { shares: latest.val, asOf: latest.end, form: latest.form, filed: latest.filed } : null;
}

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  const out = { generatedAt: new Date().toISOString(), errors: [] };
  const [dv, fi, so] = await Promise.allSettled([dailyShortVolume(), finraShortInterest(), sharesOutstanding()]);
  out.dailyShortVolume = dv.status === 'fulfilled' ? dv.value : [];
  if (dv.status === 'rejected') out.errors.push('FINRA 每日卖空量：' + dv.reason.message);
  out.dailyShortVolumeNote = 'FINRA 每日卖空量文件只统计在 FINRA 报告设施（场外/暗池）成交的部分，Total 不是全市场成交量；卖空标记成交包含做市商对冲，不等于新增空头仓位。';

  if (fi.status === 'fulfilled') out.shortInterest = fi.value;
  else {
    out.errors.push('FINRA Short Interest：' + fi.reason.message);
    try { out.shortInterest = await nasdaqShortInterest(); }
    catch (e) { out.errors.push('Nasdaq Short Interest：' + e.message); out.shortInterest = { rows: [] }; }
  }
  out.sharesOutstanding = so.status === 'fulfilled' ? so.value : null;
  if (so.status === 'rejected') out.errors.push('SEC 总股本：' + so.reason.message);
  out.floatNote = '免费数据拿不到统一口径的"自由流通股"，Short % 暂以 SEC 申报的总股本计算，会低估真实的空头占流通股比例。';
  send(res, out, 's-maxage=1800, stale-while-revalidate=3600');
}
