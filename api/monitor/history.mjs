// 历史K线（Yahoo 已合并 XWIN 时期的历史行情；如未合并，前端会显示提示）
import { SYMBOL, send, yahooChart , checkAuth } from './_lib.mjs';

const MAP = {
  '5d': ['5d', '15m'], '1mo': ['1mo', '1d'], '6mo': ['6mo', '1d'],
  '1y': ['1y', '1d'], '5y': ['5y', '1wk'], 'max': ['max', '1mo'],
};

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  const key = MAP[req.query.range] ? req.query.range : '6mo';
  const [range, interval] = MAP[key];
  try {
    const { meta, bars } = await yahooChart(SYMBOL, range, interval, false);
    send(res, {
      range: key, interval, bars, firstTradeDate: meta.firstTradeDate ?? null,
      source: 'Yahoo Finance（非官方公开接口）', generatedAt: new Date().toISOString(),
    }, 's-maxage=300, stale-while-revalidate=600');
  } catch (e) {
    send(res, { range: key, bars: [], error: e.message }, 's-maxage=30');
  }
}
