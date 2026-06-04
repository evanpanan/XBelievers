function stripHtml(s) {
  return String(s || '').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}

function decodeCdata(s) {
  const str = String(s || '');
  const m = str.match(/<!\[CDATA\[([\s\S]*?)\]\]>/i);
  return (m ? m[1] : str).trim();
}

function pickTag(block, tagName) {
  const re = new RegExp(`<${tagName}[^>]*>([\\s\\S]*?)<\\/${tagName}>`, 'i');
  const m = String(block || '').match(re);
  return m ? decodeCdata(m[1]) : '';
}

function parseItems(xml, source) {
  const raw = String(xml || '');
  const blocks = raw.match(/<item\b[\s\S]*?<\/item>/gi) || [];
  const out = [];
  for (const b of blocks) {
    const title = pickTag(b, 'title');
    const link = pickTag(b, 'link');
    const pubDate = pickTag(b, 'pubDate') || pickTag(b, 'dc:date') || pickTag(b, 'isoDate');
    const desc = stripHtml(pickTag(b, 'description') || pickTag(b, 'content:encoded')).slice(0, 320);
    if (!title || !link || !pubDate) continue;
    const ts = Date.parse(pubDate);
    if (!Number.isFinite(ts)) continue;
    out.push({ title, url: link, time: new Date(ts).toISOString(), summary: desc, source, _ts: ts });
  }
  return out;
}

module.exports = async (req, res) => {
  const limitRaw = (req.query && req.query.limit) || '10';
  let limit = 10;
  try {
    limit = Math.min(Math.max(parseInt(limitRaw, 10) || 10, 1), 30);
  } catch (_) {
    limit = 10;
  }

  const googleUrl = 'https://news.google.com/rss/search?q=Elon+Musk+OR+SpaceX+OR+Tesla+OR+xAI&hl=en-US&gl=US&ceid=US:en';
  const teslaratiUrl = 'https://www.teslarati.com/feed/';
  const ua = req.headers['user-agent'] || 'Mozilla/5.0';

  try {
    const rssHeaders = { 'User-Agent': ua, 'Accept': 'application/rss+xml, application/xml;q=0.9, text/xml;q=0.8' };
    const tryFetchXml = async (url, source) => {
      try {
        const r1 = await fetch(url, { headers: rssHeaders });
        if (r1.ok) return { ok: true, xml: await r1.text() };
        const status = r1.status;
        const jina = await fetch(`https://r.jina.ai/${url}`, { headers: { 'Accept': 'text/plain', 'User-Agent': ua } });
        if (jina.ok) {
          const txt = await jina.text();
          const idx = txt.indexOf('<?xml');
          return { ok: true, xml: idx >= 0 ? txt.slice(idx) : txt };
        }
        return { ok: false, status, status2: jina.status, source };
      } catch (e) {
        try {
          const jina = await fetch(`https://r.jina.ai/${url}`, { headers: { 'Accept': 'text/plain', 'User-Agent': ua } });
          if (jina.ok) {
            const txt = await jina.text();
            const idx = txt.indexOf('<?xml');
            return { ok: true, xml: idx >= 0 ? txt.slice(idx) : txt };
          }
          return { ok: false, error: String(e && e.message ? e.message : e), status2: jina.status, source };
        } catch (e2) {
          return { ok: false, error: String(e2 && e2.message ? e2.message : e2), source };
        }
      }
    };

    const [g, t] = await Promise.allSettled([
      tryFetchXml(googleUrl, 'Google News'),
      tryFetchXml(teslaratiUrl, 'Teslarati'),
    ]);

    const errors = [];
    let merged = [];

    if (g.status === 'fulfilled') {
      if (!g.value.ok) errors.push({ source: 'Google News', status: g.value.status, status2: g.value.status2, error: g.value.error });
      else merged = merged.concat(parseItems(g.value.xml, 'Google News'));
    } else {
      errors.push({ source: 'Google News', error: String(g.reason || '') });
    }

    if (t.status === 'fulfilled') {
      if (!t.value.ok) errors.push({ source: 'Teslarati', status: t.value.status, status2: t.value.status2, error: t.value.error });
      else merged = merged.concat(parseItems(t.value.xml, 'Teslarati'));
    } else {
      errors.push({ source: 'Teslarati', error: String(t.reason || '') });
    }

    const now = Date.now();
    const SEVEN_DAYS = 7 * 24 * 60 * 60 * 1000;
    const normUrl = (u) => String(u || '').replace(/[?#].*$/, '').trim();
    const uniq = new Map();
    for (const it of merged) {
      if (!it || !it.url || !it.title || !Number.isFinite(it._ts)) continue;
      if (now - it._ts > SEVEN_DAYS) continue;
      const key = normUrl(it.url) || String(it.title).toLowerCase();
      if (!key || uniq.has(key)) continue;
      uniq.set(key, it);
    }

    const items = Array.from(uniq.values())
      .sort((a, b) => (b._ts || 0) - (a._ts || 0))
      .slice(0, limit)
      .map(({ _ts, ...rest }) => rest);

    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.status(200).send(JSON.stringify({ success: true, items, count: items.length, errors }));
  } catch (e) {
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.status(503).send(JSON.stringify({ success: false, error: e && e.message ? e.message : String(e), items: [] }));
  }
};
