// SEC EDGAR 官方：最新公告
import { CIK, send, fetchJSON, secUA , checkAuth } from './_lib.mjs';

const FORM_CN = {
  '8-K': '重大事项公告', '8-K/A': '重大事项公告（修订）', '10-Q': '季度报告', '10-K': '年度报告',
  '10-K/A': '年报修订', 'DEF 14A': '股东大会委托书', 'PRE 14A': '股东大会委托书（初稿）',
  '4': '内部人交易（Form 4）', '3': '内部人首次持股（Form 3）', '5': '内部人年度持股（Form 5）',
  'SC 13D': '5%以上股东主动持股', 'SC 13G': '5%以上股东被动持股', 'SC 13D/A': '5%股东持股变动', 'SC 13G/A': '5%股东持股变动',
  'SCHEDULE 13D': '5%以上股东主动持股', 'SCHEDULE 13G': '5%以上股东被动持股',
  'SCHEDULE 13D/A': '5%股东持股变动', 'SCHEDULE 13G/A': '5%股东持股变动',
  'S-1': '注册发行', 'S-3': '储架发行注册', 'S-8': '员工股权计划注册', '424B3': '招股说明书补充', '424B5': '招股说明书补充',
  'EFFECT': '注册生效', 'CORRESP': '与SEC往来函件', 'UPLOAD': 'SEC意见函', 'D': '私募豁免发行（Form D）', '6-K': '外国发行人报告',
};
const ITEM_CN = {
  '1.01': '签订重大协议', '1.02': '终止重大协议', '2.01': '完成收购/出售', '2.02': '经营业绩',
  '2.03': '新增重大债务', '3.01': '退市/上市规则合规通知', '3.02': '未注册股份发行', '3.03': '股东权利变更',
  '4.01': '更换审计师', '4.02': '财报不可再依赖', '5.02': '董事/高管变动', '5.03': '章程修改',
  '5.07': '股东投票结果', '7.01': 'Reg FD 信息披露', '8.01': '其他重要事项', '9.01': '财务报表及附件',
};

export default async function handler(req, res) {
  if (!checkAuth(req, res)) return;
  try {
    const j = await fetchJSON(`https://data.sec.gov/submissions/CIK${CIK}.json`,
      { headers: { 'User-Agent': secUA(), Accept: 'application/json' } }, 8000);
    const r = j.filings?.recent || {};
    const cikNum = String(+CIK);
    const filings = (r.form || []).slice(0, 40).map((form, i) => {
      const acc = r.accessionNumber[i];
      const items = (r.items?.[i] || '').split(',').map((s) => s.trim()).filter(Boolean);
      return {
        form, formCN: FORM_CN[form] || form,
        filed: r.filingDate[i], accepted: r.acceptanceDateTime?.[i] || null,
        description: r.primaryDocDescription?.[i] || '',
        items: items.map((it) => ({ code: it, cn: ITEM_CN[it] || '' })),
        url: `https://www.sec.gov/Archives/edgar/data/${cikNum}/${acc.replace(/-/g, '')}/${r.primaryDocument[i]}`,
        indexUrl: `https://www.sec.gov/Archives/edgar/data/${cikNum}/${acc.replace(/-/g, '')}/`,
      };
    });
    send(res, {
      name: j.name, tickers: j.tickers, formerNames: j.formerNames || [],
      filings, source: 'SEC EDGAR 官方 API', generatedAt: new Date().toISOString(),
    }, 's-maxage=120, stale-while-revalidate=300');
  } catch (e) {
    send(res, { filings: [], error: e.message }, 's-maxage=30');
  }
}
