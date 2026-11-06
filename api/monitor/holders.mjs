// 机构持仓（来自 Nasdaq.com 公开页面接口，底层为 13F，季度数据且滞后约45天）
import { SYMBOL, send, fetchJSON, UA_BROWSER , checkAuth } from './_lib.mjs';
const num = (v) => (v == null || v === '' ? null : +String(v).replace(/[,$%]/g, ''));

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  try {
    const j = await fetchJSON(
      `https://api.nasdaq.com/api/company/${SYMBOL}/institutional-holdings?limit=20&type=TOTAL&sortColumn=marketValue&sortOrder=DESC`,
      { headers: { 'User-Agent': UA_BROWSER, Accept: 'application/json', Origin: 'https://www.nasdaq.com', Referer: 'https://www.nasdaq.com/' } }, 9000);
    const d = j?.data || {};
    const rows = (d.holdingsTransactions?.table?.rows || []).map((r) => ({
      name: r.ownerName, date: r.date, shares: num(r.sharesHeld), change: num(r.sharesChange),
      changePct: num(r.sharesChangePCT), value: num(r.marketValue),
    }));
    const summary = d.ownershipSummary || {};
    const active = (d.activePositions?.rows || []).map((r) => ({ type: r.positions, holders: num(r.holders), shares: num(r.shares) }));
    const newSold = (d.newSoldOutPositions?.rows || []).map((r) => ({ type: r.positions, holders: num(r.holders), shares: num(r.shares) }));
    send(res, {
      rows, active, newSold,
      instPct: summary.SharesOutstandingPCT?.value ?? null,
      instShares: summary.ShareoutstandingTotal?.value ?? null,
      instValue: summary.TotalHoldingsValue?.value ?? null,
      source: 'Nasdaq.com 机构持仓（13F 汇总，非官方接口）', generatedAt: new Date().toISOString(),
    }, 's-maxage=21600, stale-while-revalidate=43200');
  } catch (e) {
    send(res, { rows: [], error: e.message }, 's-maxage=300');
  }
}
