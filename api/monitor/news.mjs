// 新闻：Google News RSS（免费），关键词粗分利好/利空，标题去重
import { send, fetchText, UA_BROWSER , checkAuth } from './_lib.mjs';

const QUERIES = [
  { q: '"XMAX" (Nasdaq OR stock OR shares) OR "XMax Inc"', hl: 'en-US', gl: 'US', ceid: 'US:en' },
  { q: 'XMAX 纳斯达克 OR "XMax Inc"', hl: 'zh-CN', gl: 'CN', ceid: 'CN:zh-Hans' },
];
const NEG = /short seller|short report|lawsuit|class action|investigat|delist|deficien|non-?complian|offering|dilut|fraud|downgrade|subpoena|halt|resign|做空|诉讼|退市|增发|稀释|欺诈|调查|违规/i;
const POS = /buyback|repurchase|regain(ed)? compliance|partnership|agreement|contract|record|growth|upgrade|acquisition|launch|回购|合作|签约|增长|重新符合|收购|发布/i;

function decode(s) {
  return s.replace(/<!\[CDATA\[|\]\]>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/<[^>]+>/g, '').trim();
}
function tag(xml, t) { const m = xml.match(new RegExp(`<${t}[^>]*>([\\s\\S]*?)</${t}>`)); return m ? decode(m[1]) : ''; }

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  const seen = new Set(); const items = []; const errors = [];
  await Promise.all(QUERIES.map(async (qq) => {
    try {
      const url = `https://news.google.com/rss/search?q=${encodeURIComponent(qq.q)}&hl=${qq.hl}&gl=${qq.gl}&ceid=${qq.ceid}`;
      const xml = await fetchText(url, { headers: { 'User-Agent': UA_BROWSER } }, 8000);
      for (const block of xml.split('<item>').slice(1)) {
        const title = tag(block, 'title');
        const source = tag(block, 'source');
        const clean = title.replace(new RegExp(`\\s*-\\s*${source.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`), '');
        const key = clean.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fa5]/g, '').slice(0, 60);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        const tone = NEG.test(clean) ? 'neg' : POS.test(clean) ? 'pos' : 'neu';
        items.push({ title: clean, source, link: tag(block, 'link'), published: new Date(tag(block, 'pubDate')).toISOString(), tone, lang: qq.hl });
      }
    } catch (e) { errors.push(`${qq.hl}：${e.message}`); }
  }));
  items.sort((a, b) => b.published.localeCompare(a.published));
  send(res, {
    items: items.slice(0, 40), errors,
    note: '利好/利空为关键词粗分，仅作提示；"XMAX"一词可能匹配到无关内容，请以原文为准。',
    source: 'Google News RSS', generatedAt: new Date().toISOString(),
  }, 's-maxage=600, stale-while-revalidate=1200');
}
