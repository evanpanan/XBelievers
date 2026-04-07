#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深海有象 - 后端代理服务器
负责：
  1. 抓取任意新闻链接的 HTML 内容，绕过 CORS 限制
  2. 通过 LLM API 生成高质量的海报文案、短视频脚本、深度文章
"""

import sys
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import threading
import hashlib
from datetime import datetime
from urllib.parse import urlparse, quote, urlencode
from abc import ABC, abstractmethod
from typing import Optional

app = Flask(__name__)
CORS(app)

# ============================================================
#  LLM 服务抽象层 —— 支持多家 OpenAI 兼容 API
# ============================================================

class LLMProvider(ABC):
    """所有 LLM 提供商的基类"""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """发送对话请求，返回模型文本"""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI / GPT-4o / GPT-4o-mini"""
    NAME = "OpenAI"
    API_BASE = "https://api.openai.com/v1"
    MODELS = [
        {"id": "gpt-4o",      "label": "GPT-4o (最强)"},
        {"id": "gpt-4o-mini", "label": "GPT-4o-mini (经济)"},
        {"id": "gpt-4-turbo", "label": "GPT-4 Turbo"},
        {"id": "gpt-3.5-turbo","label": "GPT-3.5 Turbo"},
    ]
    DEFAULT_MODEL = "gpt-4o-mini"
    KEY_HINT = "sk-..."
    SITE = "https://platform.openai.com/api-keys"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class DeepSeekProvider(LLMProvider):
    """DeepSeek / 深度求索"""
    NAME = "DeepSeek"
    API_BASE = "https://api.deepseek.com/v1"
    MODELS = [
        {"id": "deepseek-chat",      "label": "DeepSeek-V3 (通用)"},
        {"id": "deepseek-reasoner",  "label": "DeepSeek-R1 (推理)"},
    ]
    DEFAULT_MODEL = "deepseek-chat"
    KEY_HINT = "sk-..."
    SITE = "https://platform.deepseek.com/api_keys"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class QwenProvider(LLMProvider):
    """阿里通义千问 (DashScope)"""
    NAME = "通义千问"
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODELS = [
        {"id": "qwen-turbo",         "label": "Qwen Turbo (快)"},
        {"id": "qwen-plus",          "label": "Qwen Plus (均衡)"},
        {"id": "qwen-max",           "label": "Qwen Max (最强)"},
        {"id": "qwen-long",          "label": "Qwen Long (长文)"},
    ]
    DEFAULT_MODEL = "qwen-plus"
    KEY_HINT = "sk-..."
    SITE = "https://dashscope.console.aliyun.com/apiKey"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class MoonshotProvider(LLMProvider):
    """Moonshot / Kimi"""
    NAME = "Moonshot (Kimi)"
    API_BASE = "https://api.moonshot.cn/v1"
    MODELS = [
        {"id": "moonshot-v1-8k",     "label": "Moonshot V1 8K"},
        {"id": "moonshot-v1-32k",    "label": "Moonshot V1 32K"},
        {"id": "moonshot-v1-128k",   "label": "Moonshot V1 128K"},
    ]
    DEFAULT_MODEL = "moonshot-v1-8k"
    KEY_HINT = "sk-..."
    SITE = "https://platform.moonshot.cn/console/api-keys"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class ZhipuProvider(LLMProvider):
    """智谱 AI / GLM"""
    NAME = "智谱AI (GLM)"
    API_BASE = "https://open.bigmodel.cn/api/paas/v4"
    MODELS = [
        {"id": "glm-4-flash",       "label": "GLM-4 Flash (免费)"},
        {"id": "glm-4-air",         "label": "GLM-4 Air"},
        {"id": "glm-4-plus",        "label": "GLM-4 Plus"},
    ]
    DEFAULT_MODEL = "glm-4-flash"
    KEY_HINT = "..."
    SITE = "https://open.bigmodel.cn/usercenter/apikeys"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class DoubaoProvider(LLMProvider):
    """字节跳动豆包 (火山引擎)"""
    NAME = "豆包 (字节跳动)"
    API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
    MODELS = [
        {"id": "doubao-1-5-pro-32k", "label": "Doubao 1.5 Pro"},
        {"id": "doubao-1-5-lite-32k","label": "Doubao 1.5 Lite"},
    ]
    DEFAULT_MODEL = "doubao-1-5-pro-32k"
    KEY_HINT = "..."
    SITE = "https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class YiProvider(LLMProvider):
    """零一万物 (Yi)"""
    NAME = "零一万物 (Yi)"
    API_BASE = "https://api.lingyiwanwu.com/v1"
    MODELS = [
        {"id": "yi-lightning",      "label": "Yi Lightning"},
        {"id": "yi-large",          "label": "Yi Large"},
    ]
    DEFAULT_MODEL = "yi-lightning"
    KEY_HINT = "..."
    SITE = "https://platform.lingyiwanwu.com/apikeys"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


class SparkProvider(LLMProvider):
    """讯飞星火"""
    NAME = "讯飞星火"
    API_BASE = "https://spark-api-open.xf-yun.com/v1"
    MODELS = [
        {"id": "generalv3.5",       "label": "星火 V3.5"},
        {"id": "4.0Ultra",          "label": "星火 V4.0 Ultra"},
    ]
    DEFAULT_MODEL = "generalv3.5"
    KEY_HINT = "..."
    SITE = "https://xinghuo.xfyun.cn/"

    def chat(self, system_prompt, user_prompt, temperature=0.7, max_tokens=4096):
        return _openai_compatible_chat(
            self.API_BASE, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )


# ---- 注册所有提供商 ----
PROVIDERS = {
    cls.NAME: cls for cls in
    [OpenAIProvider, DeepSeekProvider, QwenProvider, MoonshotProvider,
     ZhipuProvider, DoubaoProvider, YiProvider, SparkProvider]
}


def _openai_compatible_chat(api_base, api_key, model,
                            system_prompt, user_prompt,
                            temperature=0.7, max_tokens=4096,
                            max_retries=2):
    """统一的 OpenAI 兼容协议调用，支持 429 重试"""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=60, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                # 频率限制，指数退避重试
                err_body = resp.text[:300]
                last_error = f"LLM API 返回错误 (HTTP 429): {err_body}"
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + 0.5  # 1.5s, 2.5s
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"{last_error}，已重试 {max_retries} 次，请稍后重试")
            else:
                err_body = resp.text[:300]
                raise Exception(f"LLM API 返回错误 (HTTP {resp.status_code}): {err_body}")
        except requests.exceptions.Timeout:
            raise Exception("LLM 请求超时（60秒），请检查网络或稍后重试")
        except requests.exceptions.ConnectionError:
            raise Exception(f"无法连接 LLM 服务 ({api_base})，请检查网络")
    
    raise Exception(last_error or "LLM 请求失败")


def get_provider(provider_name, api_key, model=None):
    """根据名称实例化提供商"""
    cls = PROVIDERS.get(provider_name)
    if not cls:
        raise Exception(f"不支持的 LLM 服务商: {provider_name}，可选: {', '.join(PROVIDERS.keys())}")
    effective_model = model or cls.DEFAULT_MODEL
    return cls(api_key=api_key, model=effective_model)


# ============================================================
#  高质量 System Prompt
# ============================================================

SYSTEM_PROMPTS = {
    "poster": """你是一位顶尖的新媒体文案策划师，擅长将硬核新闻转化为高传播力的社交媒体内容。

你的任务：基于提供的新闻内容，创作一套**海报文案方案**。

要求：
1. **主标题（5-12字）**：必须有冲击力，用最少的字传递最大的信息量。可以用对比、反问、数字等手法。
2. **副标题（15-25字）**：补充关键信息，让读者产生"我必须了解更多"的冲动。
3. **正文文案（80-150字）**：
   - 开头一句抓眼球（场景/冲突/数据）
   - 中段交代核心事实（2-3个关键点）
   - 结尾给出行动号召或引发思考
4. **话题标签（3-5个）**：包含热点话题和领域标签，要有传播潜力
5. **情绪标签**：用一个词概括这条新闻的核心情绪（如：振奋、震惊、期待、警醒）

风格要求：
- 拒绝平庸的标题党，追求"有信息量的吸引力"
- 语言要干脆利落，不啰嗦
- 适合发在朋友圈/微博/小红书等平台

输出格式（严格遵守，用JSON）：
```json
{
  "main_title": "主标题",
  "sub_title": "副标题",
  "body_copy": "正文文案",
  "hashtags": ["标签1", "标签2", "标签3"],
  "emotion": "情绪词",
  "hook": "一句话钩子（用于开头引导）"
}
```""",

    "video": """你是一位资深短视频内容策划师，精通抖音/视频号/B站/小红书等平台的爆款内容创作规律。

你的任务：基于提供的新闻内容，创作一份**60秒短视频口播脚本**。

要求：
1. **节奏感强**：每句话都有存在的意义，删掉任何一句都不完整
2. **开场钩子（0-3秒）**：
   - 绝不能平铺直叙，必须制造悬念/冲突/好奇心
   - 可以用"你绝对想不到"、"就在今天"、"一个数据告诉你"等手法
   - 第一句话就要让观众停下来
3. **事件还原（3-18秒）**：
   - 用通俗语言把复杂信息讲清楚
   - 只讲最关键的事实，删掉一切冗余
   - 用"简单说就是…""换句话说…"来降低理解门槛
4. **深度解读（18-42秒）**：
   - 提出2-3个核心观点，每个观点一句话讲清楚
   - 要有"普通人视角"——这对观众意味着什么？
   - 可以适度加入自己的分析和判断，但标注"个人观点"
5. **价值收尾（42-56秒）**：
   - 用一句话总结核心价值
   - 让观众觉得"这条视频真有用"
6. **互动引导（56-60秒）**：
   - 提一个具体问题引导评论（不要泛泛的"你怎么看"）
   - 或者给出一个行动建议

画面和音效提示：
- 为每个段落标注【画面建议】和【音效/BGM建议】
- 标注【字幕样式】（如：放大关键数据、红色高亮重要词）

输出格式（严格遵守，用JSON）：
```json
{
  "total_duration": "约60秒",
  "segments": [
    {
      "time": "0-3s",
      "phase": "开场钩子",
      "script": "口播文案",
      "visual": "画面建议",
      "sound": "音效/BGM建议",
      "subtitle_note": "字幕样式备注"
    },
    {
      "time": "3-18s",
      "phase": "事件还原",
      "script": "口播文案",
      "visual": "画面建议",
      "sound": "音效/BGM建议",
      "subtitle_note": "字幕样式备注"
    },
    {
      "time": "18-42s",
      "phase": "深度解读",
      "script": "口播文案",
      "visual": "画面建议",
      "sound": "音效/BGM建议",
      "subtitle_note": "字幕样式备注"
    },
    {
      "time": "42-56s",
      "phase": "价值收尾",
      "script": "口播文案",
      "visual": "画面建议",
      "sound": "音效/BGM建议",
      "subtitle_note": "字幕样式备注"
    },
    {
      "time": "56-60s",
      "phase": "互动引导",
      "script": "口播文案",
      "visual": "画面建议",
      "sound": "音效/BGM建议",
      "subtitle_note": "字幕样式备注"
    }
  ],
  "hashtags": ["标签1", "标签2", "标签3"],
  "bgm_style": "BGM风格描述"
}
```""",

    "article": """你是一位深度内容创作者，擅长将新闻事件转化为有洞察力、有深度的分析文章。你的写作风格类似于36氪的深度报道、虎嗅的产业分析、澎湃的评论文章。

你的任务：基于提供的新闻内容，创作一篇**2000-3000字的深度分析文章**。

结构和内容要求：

**引言（200-300字）**：
- 用一个具体的场景/细节/数据开篇，吸引读者
- 快速交代新闻事件的核心信息
- 提出这篇文章要回答的核心问题（1-2个）
- 让读者产生"继续读下去"的欲望

**第一章：事件还原（400-600字）**：
- 像讲故事一样还原事件，不是简单搬运原文
- 交代时间线、关键人物、关键数据
- 突出最令人意外的细节或转折
- 引用原文中的关键信息（标注来源）

**第二章：深度解读（600-800字）**：
- 分析事件背后的深层原因（至少2个维度）
- 联系行业大背景和趋势
- 提供独特的视角和判断（不要泛泛而谈）
- 可以适当做对比分析（历史对比、国际对比等）
- 加入数据支撑你的观点（如果原文中有）

**第三章：影响分析（400-600字）**：
- 对行业的影响
- 对普通人的影响（让读者觉得"这跟我有关"）
- 短期影响 vs 长期影响
- 谁会受益？谁会承压？

**结语（200-300字）**：
- 总结核心观点（不要简单重复，要有升华）
- 给读者一个值得记住的金句
- 引导思考：这件事的下一章会怎样？

写作要求：
- 每个段落要有信息量，拒绝空话套话
- 多用具体数据和事实，少用形容词
- 观点鲜明，但保持理性客观
- 语言流畅但不要刻意文艺，追求"好读的深度"
- 适当使用小标题、加粗、引用等排版方式增强可读性

输出格式（严格遵守，用JSON）：
```json
{
  "title": "文章标题",
  "subtitle": "副标题/一句话摘要",
  "sections": [
    {
      "heading": "章节标题",
      "body": "章节正文（支持用 **加粗** 和换行）"
    }
  ],
  "word_count": "约XXXX字",
  "key_insight": "文章核心观点的一句话总结"
}
```""",
}


_TYPE_DESC = {
    'poster': '一套海报文案',
    'video': '一份60秒短视频口播脚本',
    'article': '一篇深度分析文章',
}

def _type_desc(content_type):
    return _TYPE_DESC.get(content_type, '内容')


def build_user_prompt(content_type, article, extra_requirement=''):
    """根据内容类型和新闻数据构建 user prompt"""
    title = article.get('title', '')
    source = article.get('source', '')
    full_text = article.get('fullText', '')
    summary = article.get('summary', '')
    keywords = article.get('keywords', [])
    url = article.get('url', '')

    prompt = f"""以下是需要处理的真实新闻内容：

【新闻标题】
{title}

【来源】
{source}

【内容摘要】
{summary}

【关键词】
{', '.join(keywords) if keywords else '无'}

【新闻正文】
{full_text}

---

请基于以上新闻内容，创作{_type_desc(content_type)}。

重要提醒：
- 内容必须基于上述真实新闻，不要编造信息
- 可以合理推断和延伸，但必须基于新闻事实
- 追求专业水准，对标头部自媒体的产出质量
- 严格按照要求输出JSON格式，不要输出其他多余内容
- 【关键】所有输出内容必须全部使用中文（包括标题、文案、脚本、文章正文），即使原始新闻是外文的也要翻译为中文创作
{f"- 【用户附加要求】{extra_requirement}" if extra_requirement else ""}"""

    return prompt


# ============================================================
#  模拟真实浏览器的 Headers
# ============================================================

HEADERS_POOL = [
    {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    },
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    },
    {
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
]


# ============================================================
#  新闻抓取模块 — 多层反爬绕过策略
#  Layer 1: curl_cffi  — 模拟真实浏览器 TLS/JA3 指纹
#  Layer 2: cloudscraper — 破解 Cloudflare JS Challenge
#  Layer 3: Jina Reader  — 专业内容提取代理
#  Layer 4: 公共代理 API  — AllOrigins / codetabs 等兜底
# ============================================================

def get_referer(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


# ---------- Layer 1: curl_cffi (TLS 指纹伪装) ----------
def _fetch_via_curl_cffi(url, timeout=15):
    """用 curl_cffi 模拟 Chrome 真实 TLS 指纹，可绕过大多数 TLS 指纹检测"""
    try:
        from curl_cffi import requests as cffi_requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Upgrade-Insecure-Requests': '1',
            'Referer': get_referer(url),
        }
        # impersonate="chrome124" 让 TLS 握手特征与真实 Chrome 完全一致
        resp = cffi_requests.get(url, headers=headers, timeout=timeout,
                                  impersonate="chrome124", allow_redirects=True)
        if resp.status_code == 200:
            text = resp.text
            if len(text) > 300:
                return text, None
            return None, f"curl_cffi: 内容过短({len(text)}字)"
        return None, f"curl_cffi: HTTP {resp.status_code}"
    except ImportError:
        return None, "curl_cffi 未安装"
    except Exception as e:
        return None, f"curl_cffi: {str(e)[:80]}"


# ---------- Layer 2: cloudscraper (Cloudflare JS Challenge) ----------
def _fetch_via_cloudscraper(url, timeout=15):
    """用 cloudscraper 破解 Cloudflare 的 JS 挑战页面"""
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
        )
        scraper.headers.update({
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': get_referer(url),
        })
        resp = scraper.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            text = resp.text
            if len(text) > 300:
                return text, None
            return None, f"cloudscraper: 内容过短({len(text)}字)"
        return None, f"cloudscraper: HTTP {resp.status_code}"
    except ImportError:
        return None, "cloudscraper 未安装"
    except Exception as e:
        return None, f"cloudscraper: {str(e)[:80]}"


# ---------- Layer 3: 标准 requests (多 UA 轮换) ----------
def _fetch_via_requests(url, timeout=15):
    """标准 requests，多 UA 轮换"""
    errors = []
    for i, headers in enumerate(HEADERS_POOL):
        try:
            h = headers.copy()
            h['Referer'] = get_referer(url)
            session = requests.Session()
            session.max_redirects = 5
            resp = session.get(url, headers=h, timeout=timeout,
                               allow_redirects=True, verify=False)
            if resp.status_code == 200:
                resp.encoding = resp.apparent_encoding or 'utf-8'
                content = resp.text
                if len(content) > 300:
                    return content, None
            else:
                errors.append(f"UA{i+1}: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            errors.append(f"UA{i+1}: 超时")
        except Exception as e:
            errors.append(f"UA{i+1}: {str(e)[:50]}")
        time.sleep(0.2)
    return None, '; '.join(errors)


# ---------- Layer 4: Jina Reader ----------
def fetch_url_via_jina(url, timeout=25):
    """通过 r.jina.ai 代理抓取，返回干净 Markdown，能绕过大多数反爬"""
    try:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            'Accept': 'text/plain',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'X-No-Cache': 'true',
            'X-Return-Format': 'markdown',
            'X-Remove-Selector': 'header,footer,nav,.ad,.advertisement,.cookie-notice',
        }
        resp = requests.get(jina_url, headers=headers, timeout=timeout, verify=False)
        if resp.status_code == 200:
            text = resp.text.strip()
            if len(text) > 200:
                return text, None
            return None, f"Jina: 内容过短({len(text)}字)"
        return None, f"Jina: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return None, "Jina: 请求超时"
    except Exception as e:
        return None, f"Jina: {str(e)[:80]}"


# ---------- Layer 5: 公共代理 API 兜底 ----------
def _fetch_via_public_proxy(url, timeout=20):
    """通过多个公共代理 API 抓取原始 HTML"""
    proxies = [
        # AllOrigins — 稳定免费代理
        {
            'name': 'AllOrigins',
            'build': lambda u: f"https://api.allorigins.win/get?url={quote(u)}&disableCache=true",
            'parse': lambda r: r.json().get('contents', ''),
        },
        # codetabs CORS Proxy
        {
            'name': 'codetabs',
            'build': lambda u: f"https://api.codetabs.com/v1/proxy?quest={quote(u)}",
            'parse': lambda r: r.text,
        },
        # htmlpreview / corsproxy.io
        {
            'name': 'corsproxy.io',
            'build': lambda u: f"https://corsproxy.io/?{quote(u)}",
            'parse': lambda r: r.text,
        },
    ]
    errors = []
    for proxy in proxies:
        try:
            proxy_url = proxy['build'](url)
            resp = requests.get(proxy_url, timeout=timeout, verify=False,
                                headers={'Accept': 'application/json,text/html,*/*'})
            if resp.status_code == 200:
                content = proxy['parse'](resp)
                if content and len(content) > 300:
                    print(f"[fetch] 公共代理 {proxy['name']} 成功")
                    return content, None
                errors.append(f"{proxy['name']}: 内容过短")
            else:
                errors.append(f"{proxy['name']}: HTTP {resp.status_code}")
        except Exception as e:
            errors.append(f"{proxy['name']}: {str(e)[:50]}")
    return None, '; '.join(errors)


# ---------- 统一入口：多层回退 ----------
def fetch_url(url, timeout=15):
    """向后兼容的单层请求接口（内部调用 requests）"""
    return _fetch_via_requests(url, timeout)


def fetch_url_fallback(url, timeout=15):
    """
    多层回退抓取策略（按成功率 / 速度排序）：
      Layer 1: curl_cffi  — TLS 指纹伪装（最强，秒级）
      Layer 2: cloudscraper — Cloudflare JS 挑战破解
      Layer 3: 标准 requests 多 UA 轮换
      Layer 4: Jina Reader  — 内容代理（网络稍慢）
      Layer 5: 公共代理 API — 最终兜底
    返回: (content, error, is_markdown)
    """
    all_errors = []

    # Layer 1: curl_cffi
    print(f"[fetch] Layer1 curl_cffi: {url[:80]}")
    html, err = _fetch_via_curl_cffi(url, timeout=timeout)
    if html:
        print(f"[fetch] ✓ curl_cffi 成功，长度={len(html)}")
        return html, None, False
    all_errors.append(f"L1({err})")
    print(f"[fetch] ✗ curl_cffi 失败: {err}")

    # Layer 2: cloudscraper
    print(f"[fetch] Layer2 cloudscraper...")
    html, err = _fetch_via_cloudscraper(url, timeout=timeout)
    if html:
        print(f"[fetch] ✓ cloudscraper 成功，长度={len(html)}")
        return html, None, False
    all_errors.append(f"L2({err})")
    print(f"[fetch] ✗ cloudscraper 失败: {err}")

    # Layer 3: 标准 requests
    print(f"[fetch] Layer3 标准 requests...")
    html, err = _fetch_via_requests(url, timeout=timeout)
    if html:
        print(f"[fetch] ✓ requests 成功，长度={len(html)}")
        return html, None, False
    all_errors.append(f"L3({err})")
    print(f"[fetch] ✗ requests 失败: {err}")

    # Layer 4: Jina Reader（返回 markdown）
    print(f"[fetch] Layer4 Jina Reader...")
    md_text, err = fetch_url_via_jina(url, timeout=25)
    if md_text:
        print(f"[fetch] ✓ Jina 成功，长度={len(md_text)}")
        return md_text, None, True
    all_errors.append(f"L4({err})")
    print(f"[fetch] ✗ Jina 失败: {err}")

    # Layer 5: 公共代理 API
    print(f"[fetch] Layer5 公共代理 API...")
    html, err = _fetch_via_public_proxy(url, timeout=20)
    if html:
        print(f"[fetch] ✓ 公共代理成功，长度={len(html)}")
        return html, None, False
    all_errors.append(f"L5({err})")
    print(f"[fetch] ✗ 公共代理失败: {err}")

    summary = " | ".join(all_errors)
    print(f"[fetch] 全部5层均失败: {summary[:150]}")
    return None, summary, False


def extract_article(html, url):
    soup = BeautifulSoup(html, 'lxml')

    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header',
                               'aside', 'iframe', 'noscript', 'ads', 'advertisement']):
        tag.decompose()
    for tag in soup.find_all(class_=re.compile(r'(nav|menu|sidebar|footer|header|ad|comment|share|relate|recommend|hot|tag)', re.I)):
        tag.decompose()

    # 提取标题
    title = ''
    for sel in ['h1', 'meta[property="og:title"]', 'meta[name="title"]', 'title']:
        el = soup.select_one(sel)
        if el:
            title = el.get('content', '') or el.get_text()
            title = title.strip()
            if title:
                break

    # 提取来源
    source = ''
    for sel in ['meta[property="og:site_name"]', 'meta[name="author"]',
                '.source', '.author', '.media-name', '.from-name']:
        el = soup.select_one(sel)
        if el:
            source = el.get('content', '') or el.get_text()
            source = source.strip()
            if source:
                break
    if not source:
        parsed = urlparse(url)
        host = parsed.netloc.replace('www.', '')
        domain_map = {
            '36kr.com': '36氪', 'thepaper.cn': '澎湃新闻', 'sina.com.cn': '新浪',
            'finance.sina.com.cn': '新浪财经', 'weibo.com': '微博', 'qq.com': '腾讯新闻',
            'news.qq.com': '腾讯新闻', 'caixin.com': '财新网', 'yicai.com': '第一财经',
            'eastmoney.com': '东方财富', 'cls.cn': '财联社', 'jiemian.com': '界面新闻',
            'huxiu.com': '虎嗅', 'ifanr.com': '爱范儿', 'sspai.com': '少数派',
            'cnbeta.com': 'cnBeta', 'sohu.com': '搜狐', 'sina.cn': '新浪',
            'people.com.cn': '人民网', 'xinhuanet.com': '新华网', 'cctv.com': 'CCTV',
            'bloomberg.com': '彭博社', 'reuters.com': '路透社', 'bbc.com': 'BBC',
            'nytimes.com': '纽约时报', 'techcrunch.com': 'TechCrunch', 'wired.com': 'Wired',
        }
        for domain, name in domain_map.items():
            if domain in host:
                source = name
                break
        if not source:
            source = host.split('.')[0].upper()

    # 提取发布时间
    pub_date = ''
    for sel in ['meta[property="article:published_time"]', 'meta[name="pubdate"]',
                'time[datetime]', '.date', '.time', '.pubtime', '.publish-time',
                '.date-info', '.pub-date', '.article-time']:
        el = soup.select_one(sel)
        if el:
            pub_date = el.get('content', '') or el.get('datetime', '') or el.get_text()
            pub_date = pub_date.strip()[:20]
            if pub_date:
                break

    # 提取封面图
    cover = ''
    og_img = soup.select_one('meta[property="og:image"]')
    if og_img:
        cover = og_img.get('content', '')

    # 提取正文
    body_el = None
    body_selectors = [
        'article', '[class*="article-content"]', '[class*="article_content"]',
        '[class*="articleBody"]', '[class*="article-body"]',
        '[class*="news-content"]', '[class*="news_content"]',
        '[class*="content-body"]', '[class*="content_body"]',
        '[class*="post-content"]', '[class*="entry-content"]',
        '[class*="detail-content"]', '[class*="main-text"]',
        '[class*="text-detail"]', '[class*="nr_article"]',
        '[class*="article-wrap"]', '[class*="article_wrap"]',
        '[id*="article-body"]', '[id*="articleBody"]', '[id*="content"]',
        'main .content', 'main', '.body', '#body',
    ]
    for sel in body_selectors:
        try:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 100:
                body_el = el
                break
        except:
            continue
    if not body_el:
        body_el = soup.body

    paragraphs = []
    if body_el:
        # 先提取 <p> 标签段落
        for p in body_el.find_all('p'):
            text = p.get_text(separator=' ', strip=True)
            if len(text) > 15 and not re.search(r'(版权|copyright|©|转载|来源：本站|关注我们|扫码|Subscribe|Sign up|newsletter|cookie|privacy policy)', text, re.I):
                paragraphs.append(text)
        # 如果 <p> 标签太少，尝试提取 <div>/<section> 内的直接文本块
        if len(paragraphs) < 3:
            for block in body_el.find_all(['div', 'section', 'li'], recursive=True):
                # 只取叶子节点或接近叶子的块
                if block.find('div') or block.find('section'):
                    continue
                text = block.get_text(separator=' ', strip=True)
                if len(text) > 30 and not re.search(r'(Subscribe|Sign up|newsletter|cookie|privacy policy|terms of service|click here|read more)', text, re.I):
                    # 去重：避免和已有段落重复
                    if not any(text[:50] in p for p in paragraphs):
                        paragraphs.append(text)

    full_text = '\n\n'.join(paragraphs)
    if len(full_text) < 100 and body_el:
        full_text = body_el.get_text(separator='\n', strip=True)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    summary = generate_summary(paragraphs, title)
    keywords = extract_keywords(title + ' ' + summary)

    return {
        'title': title or '（无法获取标题）',
        'source': source,
        'date': pub_date,
        'cover': cover,
        'summary': summary,
        'fullText': full_text,
        'keywords': keywords,
        'wordCount': len(re.sub(r'\s', '', full_text)),
        'paragraphCount': len(paragraphs),
    }


def extract_article_from_markdown(md_text, url):
    """从 Jina Reader 返回的 Markdown 文本中提取文章内容"""
    lines = md_text.split('\n')
    
    title = ''
    source = ''
    paragraphs = []
    in_content = False

    # Jina 返回格式通常为：
    # Title: xxx 或第一行 # xxx 是标题
    # URL Source: xxx
    # 空行
    # Markdown content...
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 跳过 Jina 元数据行和警告行
        if stripped.startswith('URL Source:') or stripped.startswith('Authors:') or \
           stripped.startswith('Published:') or stripped.startswith('Summary:') or \
           stripped.startswith('Full Article:') or stripped.startswith('Timestamp:') or \
           stripped.startswith('Warning:'):
            if stripped.startswith('URL Source:'):
                source_url = stripped.replace('URL Source:', '').strip()
                parsed = urlparse(source_url)
                host = parsed.netloc.replace('www.', '')
                for domain, name in {
                    'reuters.com': '路透社', 'bbc.com': 'BBC', 'nytimes.com': '纽约时报',
                    'techcrunch.com': 'TechCrunch', 'wired.com': 'Wired', 'bloomberg.com': '彭博社',
                    'cnn.com': 'CNN', 'theguardian.com': '卫报', 'wsj.com': '华尔街日报',
                    'washingtonpost.com': '华盛顿邮报', 'apnews.com': '美联社',
                }.items():
                    if domain in host:
                        source = name
                        break
                if not source:
                    source = host.split('.')[0].upper()
            continue

        # 提取标题（第一行 # 开头或 Title: 开头）
        if not title:
            if stripped.startswith('# '):
                title = stripped[2:].strip()
                continue
            elif stripped.startswith('Title:'):
                title = stripped.replace('Title:', '').strip()
                continue

        # 遇到分隔线后开始正文
        if stripped.startswith('---') or stripped.startswith('***'):
            in_content = True
            continue

        if in_content or (not title and stripped and not stripped.startswith('#')):
            # 提取正文段落
            if stripped and len(stripped) > 20 and not stripped.startswith('![') \
               and not stripped.startswith('[') and not stripped.startswith('http') \
               and not stripped.startswith('> '):
                # 清理 Markdown 标记
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)  # 去掉加粗
                clean = re.sub(r'\*(.+?)\*', r'\1', clean)          # 去掉斜体
                clean = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean)    # 去掉链接
                clean = clean.strip()
                if len(clean) > 20:
                    paragraphs.append(clean)

    # 如果没有通过 --- 分隔，直接取所有长行
    if not paragraphs:
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 30 and not stripped.startswith('#') \
               and not stripped.startswith('![') and not stripped.startswith('URL Source:') \
               and not stripped.startswith('Authors:') and not stripped.startswith('Published:') \
               and not stripped.startswith('Summary:'):
                clean = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
                clean = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', clean)
                if len(clean) > 30:
                    paragraphs.append(clean.strip())

    # 如果还是没标题，用 URL 路径推断
    if not title:
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if path_parts and path_parts[-1]:
            # 把 - 分隔的 slug 转为标题
            slug = path_parts[-1].replace('.html', '').replace('.htm', '')
            title = slug.replace('-', ' ').replace('_', ' ').strip().title()

    full_text = '\n\n'.join(paragraphs)
    summary = generate_summary(paragraphs, title)
    keywords = extract_keywords(title + ' ' + summary)

    return {
        'title': title or '（无法获取标题）',
        'source': source or urlparse(url).netloc.replace('www.', '').split('.')[0].upper(),
        'date': '',
        'cover': '',
        'summary': summary,
        'fullText': full_text,
        'keywords': keywords,
        'wordCount': len(re.sub(r'\s', '', full_text)),
        'paragraphCount': len(paragraphs),
    }


def generate_summary(paragraphs, title):
    if not paragraphs:
        return f'本文报道了关于"{title[:20]}"的最新动态。' if title else ''
    long_paras = [p for p in paragraphs if len(p) > 30][:3]
    if not long_paras:
        return paragraphs[0][:200] if paragraphs else ''
    summary = '。'.join(p[:120] for p in long_paras)
    if len(summary) > 350:
        summary = summary[:350] + '...'
    return summary


def extract_keywords(text):
    stop_words_cn = set('的了是在和与及也都有到不这那一个为以等其中我你他她它上下大小新被对从将之但而于可以已经'.split() +
                     ['报道', '表示', '指出', '认为', '日前', '近日', '据悉', '其中', '目前', '此前'])
    stop_words_en = set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                         'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                         'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                         'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
                         'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
                         'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                         'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                         'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
                         'or', 'if', 'while', 'about', 'that', 'this', 'these', 'those', 'it',
                         'its', 'their', 'they', 'them', 'what', 'which', 'who', 'whom',
                         'said', 'also', 'new', 'one', 'two', 'first', 'last', 'like', 'even',
                         'back', 'still', 'just', 'well', 'way', 'get', 'got', 'make', 'made',
                         'many', 'much', 'now', 'year', 'years', 'time', 'times', 'day', 'days',
                         'month', 'months', 'week', 'weeks', 'will', 'according', 'report'])

    words = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)
    freq = {}
    for w in words:
        if w not in stop_words_cn:
            freq[w] = freq.get(w, 0) + 1

    # 英文关键词提取
    en_words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    for w in en_words:
        if w not in stop_words_en:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:8]]


# ============================================================
#  新闻搜索模块 —— 多源聚合关键词搜索
# ============================================================

# 缓存：避免频繁请求同一个关键词
_news_cache = {}       # key -> {"articles": [...], "timestamp": float}
_news_cache_lock = threading.Lock()
NEWS_CACHE_TTL = 300   # 缓存 5 分钟


def _cache_get(keyword):
    """从缓存获取新闻"""
    key = keyword.strip().lower()
    with _news_cache_lock:
        entry = _news_cache.get(key)
        if entry and (time.time() - entry["timestamp"]) < NEWS_CACHE_TTL:
            return entry["articles"]
    return None


def _cache_set(keyword, articles):
    """写入缓存"""
    key = keyword.strip().lower()
    with _news_cache_lock:
        _news_cache[key] = {"articles": articles, "timestamp": time.time()}


def search_baidu_news(keyword, count=10):
    """从百度新闻搜索结果页面抓取新闻列表"""
    articles = []
    try:
        url = f"https://news.baidu.com/ns?word={quote(keyword)}&tn=news&from=news&cl=2&rn={count}&ct=1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'https://www.baidu.com/',
            'Connection': 'keep-alive',
        }
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=10, verify=False)
        if resp.status_code != 200:
            return articles

        resp.encoding = 'utf-8'
        html_text = resp.text

        # 检测是否被安全验证拦截
        if '百度安全验证' in html_text or '百度安全中心' in html_text:
            print(f"[百度] 触发安全验证，跳过")
            return articles

        soup = BeautifulSoup(html_text, 'lxml')

        # 百度新闻搜索结果：h3 > a 包含标题和链接
        for h3_a in soup.select('h3 a')[:count * 2]:
            try:
                title = h3_a.get_text(strip=True)
                link = h3_a.get('href', '')
                if not link or len(title) < 6:
                    continue
                # 过滤非新闻链接
                if 'top.baidu.com' in link or link == '/' or 'baidu.com/s?' in link:
                    continue

                h3_el = h3_a.parent
                container = h3_el.parent if h3_el else None
                if not container:
                    continue

                source = ''
                pub_time = ''
                summary = ''

                for span in container.select('span.c-color-gray'):
                    t = span.get_text(strip=True)
                    if t and '百度' not in t and '为您找到' not in t and len(t) < 15:
                        source = t
                        break

                for span in container.select('span'):
                    t = span.get_text(strip=True)
                    cls = ' '.join(span.get('class', []))
                    if ('c-color-gray2' in cls or 'c-gap-right' in cls) and re.search(r'\d', t):
                        if re.search(r'(前|小时|分钟|天前|月|日|年|:)', t):
                            pub_time = t
                            break

                for span in container.select('span'):
                    t = span.get_text(strip=True)
                    cls = ' '.join(span.get('class', []))
                    if t and t != title and len(t) > 30 and ('c-color-text' in cls or 'caption' in cls):
                        summary = t[:200]
                        break

                articles.append({
                    'title': title,
                    'url': link,
                    'source': source or '百度新闻',
                    'time': pub_time,
                    'summary': summary,
                    'engine': '百度',
                })

                if len(articles) >= count:
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"[百度新闻搜索失败] {keyword}: {str(e)[:100]}")

    return articles


def search_bing_news(keyword, count=10):
    """从必应新闻搜索结果页面抓取新闻列表"""
    articles = []
    try:
        url = f"https://cn.bing.com/news/search?q={quote(keyword)}&qft=interval%3d\"7\"&form=PTFTNR"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://cn.bing.com/',
        }
        resp = requests.get(url, headers=headers, timeout=10, verify=False)
        if resp.status_code != 200:
            return articles

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        # 必应新闻结果结构
        results = soup.select('.newsitem, .news-card, li.b_algo, .caption, .newsitem_content')
        if not results:
            results = soup.select('article, .t_t, .na_cnt')

        for item in results[:count]:
            try:
                title_el = item.select_one('a.title, a[href], h2 a, a')
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                link = title_el.get('href', '')
                if not link or len(title) < 6:
                    continue

                # 来源和时间
                source = ''
                pub_time = ''
                source_el = item.select_one('.source, .na_cont, .source-fav, span')
                if source_el:
                    st = source_el.get_text(strip=True)
                    # 提取来源和时间
                    time_match = re.search(r'(\d+[分时天前小时分钟]+|\d{1,2}[月日]\s*\d{1,2}[日号]|\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', st)
                    if time_match:
                        pub_time = time_match.group(1)
                        source = st[:time_match.start()].strip().rstrip('· -—')
                    else:
                        source = st.strip().rstrip('· -—')

                # 摘要
                summary = ''
                summary_el = item.select_one('.caption, p, .snippet')
                if summary_el:
                    summary = summary_el.get_text(strip=True)[:150]

                articles.append({
                    'title': title,
                    'url': link,
                    'source': source or '必应新闻',
                    'time': pub_time,
                    'summary': summary,
                    'engine': '必应',
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[必应新闻搜索失败] {keyword}: {str(e)[:100]}")

    return articles


def search_360_news(keyword, count=10):
    """从360新闻搜索抓取"""
    articles = []
    try:
        url = f"https://news.so.com/news?q={quote(keyword)}&src=rel"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.so.com/',
        }
        resp = requests.get(url, headers=headers, timeout=10, verify=False)
        if resp.status_code != 200:
            return articles

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        results = soup.select('.res-list, .news-list, li, .news_item')
        for item in results[:count]:
            try:
                title_el = item.select_one('h3 a, a')
                if not title_el:
                    continue

                title = title_el.get_text(strip=True)
                link = title_el.get('href', '')
                if not link or len(title) < 6:
                    continue

                source = ''
                pub_time = ''
                meta_el = item.select_one('.s-p, .source, .meta, span')
                if meta_el:
                    mt = meta_el.get_text(strip=True)
                    time_match = re.search(r'(\d+[分时天前小时分钟]+|\d{1,2}[月日]\s*\d{1,2}[日号]|\d{4}[-/年])', mt)
                    if time_match:
                        pub_time = time_match.group(1)
                        source = mt[:time_match.start()].strip().rstrip('· -—')
                    else:
                        source = mt.strip().rstrip('· -—')

                summary = ''
                summary_el = item.select_one('.txt-layout, p, .content')
                if summary_el:
                    summary = summary_el.get_text(strip=True)[:150]

                articles.append({
                    'title': title,
                    'url': link,
                    'source': source or '360新闻',
                    'time': pub_time,
                    'summary': summary,
                    'engine': '360',
                })
            except Exception:
                continue
    except Exception as e:
        print(f"[360新闻搜索失败] {keyword}: {str(e)[:100]}")

    return articles


def search_google_news_rss(keyword, count=10):
    """通过 RSS 代理获取 Google News 英文新闻（多镜像方案，确保国内可达）"""
    articles = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=en-US&gl=US&ceid=US:en"

    # 多个镜像方案，按优先级尝试（直连优先，rss2json 作为 fallback）
    mirror_configs = [
        {
            'name': 'direct',
            'url': rss_url,
        },
        {
            'name': 'rss2json',
            'url': f"https://api.rss2json.com/v1/api.json?rss_url={quote(rss_url)}&count={count}",
            'is_json': True,
        },
    ]

    for mirror in mirror_configs:
        try:
            resp = requests.get(mirror['url'], timeout=15, verify=False)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue

            if mirror.get('is_json'):
                # rss2json 格式
                data = resp.json()
                if data.get('status') != 'ok' or not data.get('items'):
                    continue
                for item in data['items'][:count]:
                    try:
                        title = item.get('title', '').strip()
                        link = item.get('link', '')
                        pub_date = item.get('pubDate', '')
                        author = item.get('author', '') or ''

                        if not title or len(title) < 10 or not link:
                            continue

                        description = item.get('description', '') or ''
                        desc_soup = BeautifulSoup(description, 'lxml')
                        summary = desc_soup.get_text(strip=True)[:200] if desc_soup else ''

                        # 从 description 中提取来源
                        source = author
                        if not source:
                            link_el = desc_soup.select_one('a') if desc_soup else None
                            if link_el:
                                source = link_el.get_text(strip=True)
                        if not source:
                            source = 'Google News'

                        articles.append({
                            'title': title,
                            'title_en': title,
                            'title_cn': '',
                            'url': link,
                            'source': source,
                            'time': pub_date,
                            'time_parsed': parse_news_time(pub_date),
                            'summary': summary,
                            'engine': 'Google News',
                            'is_foreign': True,
                        })
                    except:
                        continue
            else:
                # XML RSS 格式（直接解析 XML）
                try:
                    soup = BeautifulSoup(resp.text, 'xml')
                    items = soup.select('item')

                    for item in items[:count]:
                        try:
                            title_el = item.find('title')
                            if not title_el:
                                continue
                            title = title_el.get_text(strip=True)
                            if len(title) < 10:
                                continue

                            link_el = item.find('link')
                            link = link_el.get_text(strip=True) if link_el else ''
                            if not link:
                                continue

                            pub_date_el = item.find('pubDate')
                            pub_date = pub_date_el.get_text(strip=True) if pub_date_el else ''

                            source_el = item.find('source')
                            source = source_el.get_text(strip=True) if source_el else 'Google News'

                            articles.append({
                                'title': title,
                                'title_en': title,
                                'title_cn': '',
                                'url': link,
                                'source': source,
                                'time': pub_date,
                                'time_parsed': parse_news_time(pub_date),
                                'summary': '',
                                'engine': 'Google News',
                                'is_foreign': True,
                            })
                        except:
                            continue
                except:
                    continue

            if articles:
                break
        except Exception as e:
            print(f"[Google News RSS({mirror['name']})失败] {keyword}: {str(e)[:80]}")
            continue

    return articles


def search_google_news_cn(keyword, count=10):
    """通过 RSS 代理获取 Google News 中文新闻"""
    articles = []
    rss_url = f"https://news.google.com/rss/search?q={quote(keyword)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"

    # 直连优先，rss2json 作为 fallback（allorigins 已不可用）
    mirror_configs = [
        {
            'name': 'direct',
            'url': rss_url,
        },
        {
            'name': 'rss2json',
            'url': f"https://api.rss2json.com/v1/api.json?rss_url={quote(rss_url)}&count={count}",
            'is_json': True,
        },
    ]

    for mirror in mirror_configs:
        try:
            resp = requests.get(mirror['url'], timeout=15, verify=False)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue

            if mirror.get('is_json'):
                data = resp.json()
                if data.get('status') != 'ok' or not data.get('items'):
                    continue
                for item in data['items'][:count]:
                    try:
                        title = item.get('title', '').strip()
                        link = item.get('link', '')
                        pub_date = item.get('pubDate', '')
                        if not title or len(title) < 6 or not link:
                            continue
                        articles.append({
                            'title': title,
                            'title_en': '',
                            'title_cn': '',
                            'url': link,
                            'source': item.get('author', '') or 'Google News',
                            'time': pub_date,
                            'time_parsed': parse_news_time(pub_date),
                            'summary': '',
                            'engine': 'Google News 中文',
                            'is_foreign': False,
                        })
                    except:
                        continue
            else:
                try:
                    soup = BeautifulSoup(resp.text, 'xml')
                    for item in soup.select('item')[:count]:
                        try:
                            title_el = item.find('title')
                            if not title_el:
                                continue
                            title = title_el.get_text(strip=True)
                            if len(title) < 6:
                                continue
                            link_el = item.find('link')
                            link = link_el.get_text(strip=True) if link_el else ''
                            if not link:
                                continue
                            pub_date_el = item.find('pubDate')
                            pub_date = pub_date_el.get_text(strip=True) if pub_date_el else ''
                            source_el = item.find('source')
                            source = source_el.get_text(strip=True) if source_el else 'Google News'
                            articles.append({
                                'title': title,
                                'title_en': '',
                                'title_cn': '',
                                'url': link,
                                'source': source,
                                'time': pub_date,
                                'time_parsed': parse_news_time(pub_date),
                                'summary': '',
                                'engine': 'Google News 中文',
                                'is_foreign': False,
                            })
                        except:
                            continue
                except:
                    continue

            if articles:
                break
        except Exception as e:
            print(f"[Google News 中文RSS({mirror['name']})失败] {keyword}: {str(e)[:80]}")
            continue

    return articles


def search_bbc_rss(keyword, count=8):
    """通过 RSS 代理获取 BBC 新闻"""
    articles = []
    rss_url = "https://feeds.bbci.co.uk/news/world/rss.xml"

    # 直连优先，rss2json 作为 fallback（allorigins 已不可用）
    mirror_configs = [
        {
            'name': 'direct',
            'url': rss_url,
        },
        {
            'name': 'rss2json',
            'url': f"https://api.rss2json.com/v1/api.json?rss_url={quote(rss_url)}&count=30",
            'is_json': True,
        },
    ]

    keyword_lower = keyword.lower()
    kw_parts = [p for p in keyword_lower.replace(',', ' ').split() if len(p) > 2]

    for mirror in mirror_configs:
        try:
            resp = requests.get(mirror['url'], timeout=15, verify=False)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue

            items_data = []
            if mirror.get('is_json'):
                data = resp.json()
                if data.get('status') != 'ok' or not data.get('items'):
                    continue
                items_data = data['items']
            else:
                soup = BeautifulSoup(resp.text, 'xml')
                raw_items = soup.select('item')
                for raw in raw_items:
                    items_data.append({
                        'title': raw.find('title').get_text(strip=True) if raw.find('title') else '',
                        'link': raw.find('link').get_text(strip=True) if raw.find('link') else '',
                        'pubDate': raw.find('pubDate').get_text(strip=True) if raw.find('pubDate') else '',
                        'description': raw.find('description').get_text(strip=True) if raw.find('description') else '',
                    })

            for item in items_data:
                if len(articles) >= count:
                    break
                try:
                    title = item.get('title', '').strip()
                    link = item.get('link', '')
                    pub_date = item.get('pubDate', '')
                    description = item.get('description', '') or ''

                    if not title or not link:
                        continue

                    combined = (title + ' ' + description).lower()
                    matched = keyword_lower in combined
                    if not matched:
                        matched = any(p in combined for p in kw_parts)
                    if not matched:
                        continue

                    desc_soup = BeautifulSoup(description, 'lxml')
                    summary = desc_soup.get_text(strip=True)[:200] if desc_soup else ''

                    articles.append({
                        'title': title,
                        'title_en': title,
                        'title_cn': '',
                        'url': link,
                        'source': 'BBC News',
                        'time': pub_date,
                        'time_parsed': parse_news_time(pub_date),
                        'summary': summary,
                        'engine': 'BBC News',
                        'is_foreign': True,
                    })
                except:
                    continue

            if articles:
                break
        except Exception as e:
            print(f"[BBC RSS({mirror['name']})失败] {keyword}: {str(e)[:80]}")
            continue

    return articles


def search_reuters_rss(keyword, count=8):
    """通过 RSS 代理获取海外财经新闻（原 Reuters 域名已失效，改用 Yahoo Finance）"""
    articles = []
    # Yahoo Finance RSS（通过 rss2json 中转，已验证可用）
    rss_url = "https://finance.yahoo.com/news/rssindex"

    # rss2json 是目前唯一可靠的中转（allorigins 已挂，Yahoo RSS 不能直连）
    mirror_configs = [
        {
            'name': 'rss2json',
            'url': f"https://api.rss2json.com/v1/api.json?rss_url={quote(rss_url)}&count=30",
            'is_json': True,
        },
    ]

    keyword_lower = keyword.lower()
    kw_parts = [p for p in keyword_lower.replace(',', ' ').split() if len(p) > 2]

    for mirror in mirror_configs:
        try:
            resp = requests.get(mirror['url'], timeout=15, verify=False)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue

            items_data = []
            if mirror.get('is_json'):
                data = resp.json()
                if data.get('status') != 'ok' or not data.get('items'):
                    continue
                items_data = data['items']
            else:
                soup = BeautifulSoup(resp.text, 'xml')
                for raw in soup.select('item'):
                    items_data.append({
                        'title': raw.find('title').get_text(strip=True) if raw.find('title') else '',
                        'link': raw.find('link').get_text(strip=True) if raw.find('link') else '',
                        'pubDate': raw.find('pubDate').get_text(strip=True) if raw.find('pubDate') else '',
                        'description': raw.find('description').get_text(strip=True) if raw.find('description') else '',
                    })

            for item in items_data:
                if len(articles) >= count:
                    break
                try:
                    title = item.get('title', '').strip()
                    link = item.get('link', '')
                    pub_date = item.get('pubDate', '')
                    description = item.get('description', '') or ''

                    if not title or not link:
                        continue

                    combined = (title + ' ' + description).lower()
                    matched = keyword_lower in combined
                    if not matched:
                        matched = any(p in combined for p in kw_parts)
                    if not matched:
                        continue

                    desc_soup = BeautifulSoup(description, 'lxml')
                    summary = desc_soup.get_text(strip=True)[:200] if desc_soup else ''

                    articles.append({
                        'title': title,
                        'title_en': title,
                        'title_cn': '',
                        'url': link,
                        'source': 'Yahoo Finance',
                        'time': pub_date,
                        'time_parsed': parse_news_time(pub_date),
                        'summary': summary,
                        'engine': 'Yahoo Finance',
                        'is_foreign': True,
                    })
                except:
                    continue

            if articles:
                break
        except Exception as e:
            print(f"[Yahoo Finance RSS({mirror['name']})失败] {keyword}: {str(e)[:80]}")
            continue

    return articles


def search_techcrunch_rss(keyword, count=8):
    """通过 RSS 代理获取 TechCrunch 新闻"""
    articles = []
    rss_url = "https://techcrunch.com/feed/"

    # 直连优先，rss2json 作为 fallback（allorigins 已不可用）
    mirror_configs = [
        {
            'name': 'direct',
            'url': rss_url,
        },
        {
            'name': 'rss2json',
            'url': f"https://api.rss2json.com/v1/api.json?rss_url={quote(rss_url)}&count=30",
            'is_json': True,
        },
    ]

    keyword_lower = keyword.lower()
    kw_parts = [p for p in keyword_lower.replace(',', ' ').split() if len(p) > 2]

    for mirror in mirror_configs:
        try:
            resp = requests.get(mirror['url'], timeout=15, verify=False)
            if resp.status_code != 200 or len(resp.text) < 100:
                continue

            items_data = []
            if mirror.get('is_json'):
                data = resp.json()
                if data.get('status') != 'ok' or not data.get('items'):
                    continue
                items_data = data['items']
            else:
                soup = BeautifulSoup(resp.text, 'xml')
                for raw in soup.select('item'):
                    items_data.append({
                        'title': raw.find('title').get_text(strip=True) if raw.find('title') else '',
                        'link': raw.find('link').get_text(strip=True) if raw.find('link') else '',
                        'pubDate': raw.find('pubDate').get_text(strip=True) if raw.find('pubDate') else '',
                        'description': raw.find('description').get_text(strip=True) if raw.find('description') else '',
                        'categories': [c.get_text(strip=True) for c in raw.select('category')] if raw.select('category') else [],
                    })

            for item in items_data:
                if len(articles) >= count:
                    break
                try:
                    title = item.get('title', '').strip()
                    link = item.get('link', '')
                    pub_date = item.get('pubDate', '')
                    description = item.get('description', '') or ''
                    categories = item.get('categories', []) or []

                    if not title or not link:
                        continue

                    combined = (title + ' ' + description + ' ' + ' '.join(categories)).lower()
                    matched = keyword_lower in combined
                    if not matched:
                        matched = any(p in combined for p in kw_parts)
                    if not matched:
                        continue

                    desc_soup = BeautifulSoup(description, 'lxml')
                    summary = desc_soup.get_text(strip=True)[:200] if desc_soup else ''
                    category = categories[0] if categories else ''

                    articles.append({
                        'title': title,
                        'title_en': title,
                        'title_cn': '',
                        'url': link,
                        'source': f'TechCrunch{(" · " + category) if category else ""}',
                        'time': pub_date,
                        'time_parsed': parse_news_time(pub_date),
                        'summary': summary,
                        'engine': 'TechCrunch',
                        'is_foreign': True,
                    })
                except:
                    continue

            if articles:
                break
        except Exception as e:
            print(f"[TechCrunch RSS({mirror['name']})失败] {keyword}: {str(e)[:80]}")
            continue

    return articles


def search_bing_news_en(keyword, count=10):
    """从 Bing 国际版搜索英文新闻（直连，可能需要代理）"""
    articles = []
    try:
        url = f"https://www.bing.com/news/search?q={quote(keyword)}&qft=interval%3d%227%22&form=PTFTNR&setlang=en-US"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.bing.com/',
        }
        resp = requests.get(url, headers=headers, timeout=12, verify=False,
                             proxies=_get_proxies())
        if resp.status_code != 200:
            return articles

        if '<form id="sb_form"' in resp.text[:2000] and 'news' not in resp.url:
            return articles

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        results = soup.select('.newsitem, .news-card, li.b_algo, .caption, .newsitem_content')
        if not results:
            results = soup.select('article, .t_t, .na_cnt')

        for item in results[:count]:
            try:
                title_el = item.select_one('a.title, a[href], h2 a, a')
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                link = title_el.get('href', '')
                if not link or len(title) < 10:
                    continue

                source = ''
                pub_time = ''
                time_parsed = 0
                source_el = item.select_one('.source, .na_cont, .source-fav, span')
                if source_el:
                    st = source_el.get_text(strip=True)
                    time_match = re.search(r'(\d+[smhd]|minutes?|hours?|days?|ago)', st, re.I)
                    if time_match:
                        pub_time = time_match.group(0)
                        time_parsed = parse_relative_time(pub_time)
                        source = st[:time_match.start()].strip().rstrip('· -—')
                    else:
                        source = st.strip().rstrip('· -—')

                articles.append({
                    'title': title, 'title_en': title, 'title_cn': '',
                    'url': link, 'source': source or 'Bing News',
                    'time': pub_time, 'time_parsed': time_parsed,
                    'summary': '', 'engine': 'Bing International', 'is_foreign': True,
                })
            except:
                continue
    except Exception as e:
        print(f"[Bing国际新闻搜索失败] {keyword}: {str(e)[:100]}")

    return articles


def _get_proxies():
    """获取代理配置（如果设置了 HTTP_PROXY 环境变量则使用）"""
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or os.environ.get('ALL_PROXY') or os.environ.get('all_proxy')
    if http_proxy:
        return {'http': http_proxy, 'https': http_proxy}
    return None


# ============================================================
#  时间解析工具
# ============================================================

def parse_news_time(time_str):
    """将各种时间格式解析为 Unix 时间戳（秒），失败返回 0"""
    if not time_str:
        return 0
    time_str = time_str.strip()

    # RFC 2822 格式（Google News RSS 常用）
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(time_str)
        return int(dt.timestamp())
    except:
        pass

    # ISO 8601 格式
    iso_patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})T(\d{1,2}):(\d{1,2})',
        r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{1,2})',
        r'(\d{4})/(\d{1,2})/(\d{1,2})',
    ]
    for pat in iso_patterns:
        m = re.search(pat, time_str)
        if m:
            try:
                groups = [int(g) for g in m.groups()]
                if len(groups) >= 5:
                    dt = datetime(groups[0], groups[1], groups[2], groups[3], groups[4])
                else:
                    dt = datetime(groups[0], groups[1], groups[2])
                return int(dt.timestamp())
            except:
                continue

    # 相对时间格式
    return parse_relative_time(time_str)


def parse_relative_time(time_str):
    """解析相对时间字符串，返回 Unix 时间戳"""
    if not time_str:
        return 0
    now = int(time.time())

    m = re.search(r'(\d+)\s*(min(?:ute)?s?)\s*ago', time_str, re.I)
    if m:
        return now - int(m.group(1)) * 60

    m = re.search(r'(\d+)\s*(hour|hr)s?\s*ago', time_str, re.I)
    if m:
        return now - int(m.group(1)) * 3600

    m = re.search(r'(\d+)\s*(day)s?\s*ago', time_str, re.I)
    if m:
        return now - int(m.group(1)) * 86400

    m = re.search(r'^(\d+)([smhd])$', time_str, re.I)
    if m:
        val = int(m.group(1))
        unit = m.group(2).lower()
        multiplier = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return now - val * multiplier.get(unit, 60)

    return 0


def is_recent_article(article, max_age_hours=168):
    """检查新闻是否在指定时间范围内（默认7天）"""
    time_parsed = article.get('time_parsed', 0)
    if time_parsed > 0:
        return (time.time() - time_parsed) < max_age_hours * 3600

    # 如果没有解析到时间，检查文本中是否包含旧日期
    time_text = article.get('time', '') or ''
    if not time_text:
        # 没有时间的，放行（宁多勿漏）
        return True

    # 检查文本时间格式中的年份
    year_match = re.search(r'(?:20|19)\d{2}', time_text)
    if year_match:
        year = int(year_match.group())
        current_year = datetime.now().year
        if year < current_year - 1:
            return False
        if year < current_year:
            # 去年的新闻，检查是否超过半年
            return True  # 宽松处理

    return True


def deduplicate_articles(articles):
    """按标题相似度去重"""
    seen = set()
    unique = []
    for a in articles:
        # 同时用中文和英文标题去重
        title_cn = a.get('title', '')
        title_en = a.get('title_en', '')
        norm_cn = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', title_cn).lower()
        norm_en = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', title_en).lower()

        # 也用 URL 去重（不同来源可能相同新闻）
        url_key = ''
        if a.get('url'):
            parsed = urlparse(a['url'])
            url_key = parsed.netloc + parsed.path[:60].rstrip('/')

        if norm_cn and len(norm_cn) > 6 and norm_cn in seen:
            continue
        if norm_en and len(norm_en) > 6 and norm_en in seen:
            continue
        if url_key and url_key in seen:
            continue

        if norm_cn:
            seen.add(norm_cn)
        if norm_en:
            seen.add(norm_en)
        if url_key:
            seen.add(url_key)
        unique.append(a)

    return unique


# ============================================================
#  简易翻译：英文标题 → 中文（使用 free 翻译接口）
# ============================================================

_translate_cache = {}
_translate_cache_lock = threading.Lock()


def translate_title_to_chinese(title_en):
    """将英文标题翻译为中文，使用免费翻译接口"""
    if not title_en or len(title_en) < 5:
        return ''

    # 检查是否主要是英文
    en_chars = len(re.findall(r'[a-zA-Z]', title_en))
    if en_chars < len(title_en) * 0.4:
        return ''

    cache_key = title_en.strip().lower()
    with _translate_cache_lock:
        if cache_key in _translate_cache:
            return _translate_cache[cache_key]

    translated = ''

    # 方案1: 使用 MyMemory 翻译 API（免费，无需 key）
    try:
        url = f"https://api.mymemory.translated.net/get?q={quote(title_en)}&langpair=en|zh-CN"
        resp = requests.get(url, timeout=5, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            translated = data.get('responseData', {}).get('translatedText', '')
            # MyMemory 有时会返回大写文本，做基本清理
            if translated and translated.upper() == translated and len(translated) > 20:
                # 可能翻译失败，尝试下一个方案
                translated = ''
    except:
        pass

    # 方案2: 使用 LibreTranslate（免费开源）
    if not translated:
        try:
            url = f"https://libretranslate.de/translate"
            resp = requests.post(url, json={
                'q': title_en,
                'source': 'en',
                'target': 'zh',
                'format': 'text',
            }, timeout=5, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get('translatedText', '')
        except:
            pass

    with _translate_cache_lock:
        _translate_cache[cache_key] = translated

    return translated


def batch_translate_titles(articles):
    """批量翻译英文标题（带线程并行）"""
    foreign_articles = [a for a in articles if a.get('is_foreign') and a.get('title_en')]
    if not foreign_articles:
        return articles

    def _translate(a):
        cn = translate_title_to_chinese(a['title_en'])
        if cn:
            a['title_cn'] = cn

    threads = []
    for a in foreign_articles:
        t = threading.Thread(target=_translate, args=(a,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=8)

    return articles


# ============================================================
#  英文检测 & LLM 翻译模块
# ============================================================

def is_english_content(text):
    """检测文本是否主要为英文（英文占比超过 60%）"""
    if not text or len(text) < 20:
        return False
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    cn_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    total_alpha = en_chars + cn_chars
    if total_alpha == 0:
        return False
    return en_chars / total_alpha > 0.6


def translate_article_with_llm(article, provider_name, api_key, model=None):
    """使用 LLM 将英文新闻翻译为中文，返回翻译后的 article 副本"""
    title = article.get('title', '')
    full_text = article.get('fullText', '')
    summary = article.get('summary', '')

    # 截取正文（避免超出 token 限制），保留前 6000 字符
    text_to_translate = full_text[:6000] if len(full_text) > 6000 else full_text

    try:
        llm = get_provider(provider_name, api_key, model or None)

        system_prompt = """你是一位专业新闻翻译专家，精通英中双语翻译。
你的任务：将英文新闻内容准确翻译为流畅的中文。

翻译要求：
1. 准确传达原文含义，不要遗漏关键信息
2. 使用符合中文习惯的表达方式，避免"翻译腔"
3. 专有名词（人名、地名、机构名）首次出现时保留英文原文，括号中给出中文
4. 数字、金额保持原文格式
5. 输出为 JSON 格式，包含三个字段：title（标题）、summary（摘要）、fullText（正文）

输出格式（严格遵守JSON）：
```json
{
  "title": "中文标题",
  "summary": "中文摘要",
  "fullText": "中文正文（段落之间用空行分隔）"
}
```"""

        user_prompt = f"""请将以下英文新闻翻译为中文：

【标题】
{title}

【摘要】
{summary}

【正文】
{text_to_translate}

{"（注：正文已截取，仅翻译前半部分内容）" if len(full_text) > 6000 else ""}"""

        raw_response = llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=8192,
        )

        parsed = parse_llm_json(raw_response)

        if parsed.get('_raw'):
            raise Exception(f"翻译结果解析失败: {parsed.get('_parse_error', '')}")

        # 构建翻译后的文章对象
        translated = article.copy()
        translated['title'] = parsed.get('title', title)
        translated['summary'] = parsed.get('summary', summary)
        translated['fullText'] = parsed.get('fullText', full_text)
        translated['keywords'] = extract_keywords(parsed.get('title', '') + ' ' + parsed.get('summary', ''))
        translated['is_translated'] = True
        translated['original_title'] = title

        return translated

    except Exception as e:
        # 翻译失败，返回原文但标记错误
        print(f"[翻译失败] {title[:30]}: {str(e)[:100]}")
        article['translate_error'] = str(e)
        return article


def search_news_aggregated(keyword, count=15, include_foreign=True):
    """聚合多个搜索引擎，返回去重后的新闻列表"""
    keyword = keyword.strip()
    if not keyword:
        return []

    # 先查缓存
    cache_key = f"{keyword}_{include_foreign}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached[:count]

    all_articles = []

    # 并行发起多源搜索
    threads = []
    results_lock = threading.Lock()

    def _search(fn):
        try:
            items = fn(keyword, count)
            with results_lock:
                all_articles.extend(items)
        except Exception:
            pass

    # 国内源
    domestic_sources = [search_baidu_news, search_bing_news, search_360_news]

    # 海外源（直连 + RSS 中转，国内可达）
    foreign_sources = [
        search_google_news_rss,     # Google News 英文（直连优先）
        search_google_news_cn,      # Google News 中文（直连优先）
        search_bbc_rss,             # BBC News（直连优先）
        search_reuters_rss,         # Yahoo Finance（rss2json 中转）
        search_techcrunch_rss,      # TechCrunch（直连优先）
        search_bing_news_en,        # Bing 国际（直连，可能需要代理）
    ]

    all_sources = domestic_sources[:]
    if include_foreign:
        all_sources.extend(foreign_sources)

    for fn in all_sources:
        t = threading.Thread(target=_search, args=(fn,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=15)

    # 如果第一轮没有结果，尝试百度第二轮
    if not all_articles:
        time.sleep(1)
        try:
            items = search_baidu_news(keyword, count)
            all_articles.extend(items)
        except Exception:
            pass

    # 为国内新闻也添加 time_parsed
    for a in all_articles:
        if 'time_parsed' not in a:
            a['time_parsed'] = parse_news_time(a.get('time', ''))
        if 'is_foreign' not in a:
            a['is_foreign'] = False
        if 'title_en' not in a:
            a['title_en'] = ''
        if 'title_cn' not in a:
            a['title_cn'] = ''

    # 过滤旧新闻（超过7天的）
    recent_articles = [a for a in all_articles if is_recent_article(a, max_age_hours=168)]

    # 去重
    unique = deduplicate_articles(recent_articles)

    # 排序：有时间戳的按时间倒序，无时间戳的放后面
    unique.sort(key=lambda x: (
        -x.get('time_parsed', 0),
        x.get('engine', ''),
    ))

    result = unique[:count]

    # 批量翻译海外新闻标题
    if include_foreign:
        batch_translate_titles(result)

    if result:
        _cache_set(cache_key, result)
    else:
        # 不缓存空结果，避免后续请求读到空缓存
        pass

    return result


def search_news_multi_keywords(keywords, count_per_keyword=5, include_foreign=True):
    """多关键词聚合搜索，每个关键词搜索若干条，汇总去重"""
    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        return []

    # 如果只有一个关键词，直接走单关键词逻辑但增加数量
    if len(keywords) == 1:
        return search_news_aggregated(keywords[0], count=count_per_keyword * 3, include_foreign=include_foreign)

    all_articles = []
    threads = []
    results_lock = threading.Lock()

    def _search_kw(kw):
        try:
            items = search_news_aggregated(kw, count=count_per_keyword * 2, include_foreign=include_foreign)
            # 为每条新闻标注匹配的关键词
            for item in items:
                item['matched_keyword'] = kw
            with results_lock:
                all_articles.extend(items)
        except Exception:
            pass

    for kw in keywords:
        t = threading.Thread(target=_search_kw, args=(kw,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=20)

    # 汇总去重
    unique = deduplicate_articles(all_articles)

    # 合并匹配的关键词
    for a in unique:
        matched = set()
        for art in all_articles:
            if art.get('url') == a.get('url') or art.get('title') == a.get('title'):
                if art.get('matched_keyword'):
                    matched.add(art['matched_keyword'])
        a['matched_keywords'] = list(matched)

    # 按时间排序
    unique.sort(key=lambda x: -x.get('time_parsed', 0))

    result = unique[:count_per_keyword * 3]
    return result


# ============================================================
#  API 路由
# ============================================================

@app.route('/api/providers', methods=['GET'])
def list_providers():
    """列出所有可用的 LLM 提供商及模型"""
    result = {}
    for name, cls in PROVIDERS.items():
        result[name] = {
            "models": cls.MODELS,
            "default_model": cls.DEFAULT_MODEL,
            "key_hint": cls.KEY_HINT,
            "site": cls.SITE,
        }
    return jsonify({"success": True, "providers": result})


@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def generate_content():
    """核心接口：调用 LLM 生成内容"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(force=True)
    content_type = data.get('type', '').strip()       # poster / video / article
    article = data.get('article', {})
    provider_name = data.get('provider', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', '').strip()
    extra_requirement = data.get('extra_requirement', '').strip()  # 用户附加要求

    # 参数校验
    if content_type not in SYSTEM_PROMPTS:
        return jsonify({"success": False, "error": f"不支持的内容类型: {content_type}，可选: {', '.join(SYSTEM_PROMPTS.keys())}"}), 400
    if not article or not article.get('title'):
        return jsonify({"success": False, "error": "缺少新闻数据"}), 400
    if not provider_name or provider_name not in PROVIDERS:
        return jsonify({"success": False, "error": f"无效的 LLM 服务商，可选: {', '.join(PROVIDERS.keys())}"}), 400
    if not api_key:
        return jsonify({"success": False, "error": "请提供 API Key"}), 400

    try:
        # 实例化 LLM 提供商
        llm = get_provider(provider_name, api_key, model or None)

        # 构建 prompt
        system_prompt = SYSTEM_PROMPTS[content_type]
        user_prompt = build_user_prompt(content_type, article, extra_requirement)

        # 调用 LLM
        raw_response = llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.75 if content_type == 'video' else 0.7,
            max_tokens=4096 if content_type != 'article' else 8192,
        )

        # 解析 JSON 响应
        parsed = parse_llm_json(raw_response)

        return jsonify({
            "success": True,
            "type": content_type,
            "content": parsed,
            "provider": provider_name,
            "model": llm.model,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def parse_llm_json(raw_text):
    """从 LLM 响应中提取 JSON（兼容 markdown code block 包裹）"""
    text = raw_text.strip()

    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1).strip()

    # 尝试提取 { ... }
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # JSON 解析失败，返回原始文本
        return {
            "_raw": True,
            "_text": raw_text,
            "_parse_error": f"JSON解析失败: {str(e)}",
        }


@app.route('/api/fetch', methods=['POST', 'OPTIONS'])
def fetch_news():
    """抓取并解析新闻链接，如果是英文则自动翻译。
    特殊模式：传入 prefill=true 时，直接使用请求中的数据（跳过抓取），仅做翻译处理。
    """
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(force=True)
    url = data.get('url', '').strip()
    provider_name = data.get('provider', '').strip()
    api_key = data.get('api_key', '').strip()
    model = data.get('model', '').strip()

    # ── 直传模式（雷达新闻直接传入，跳过页面抓取）──
    if data.get('prefill'):
        print(f"[fetch] 直传模式: title='{str(data.get('title',''))[:50]}'")
        article = {
            'title': data.get('title', ''),
            'summary': data.get('summary', ''),
            'fullText': data.get('fullText', '') or data.get('summary', ''),
            'source': data.get('source', '新闻雷达'),
            'date': '',
            'keywords': [],
            'cover': '',
            'wordCount': len(data.get('fullText', '') or data.get('summary', '')),
            'paragraphCount': 1,
            'is_prefill': True,
        }
        article['keywords'] = extract_keywords(
            (article['title'] + ' ' + article['fullText'])[:1000]
        )

        # 检测英文并翻译
        combined_text = article['title'] + ' ' + article['fullText'][:500]
        need_translate = is_english_content(combined_text)
        article['is_english'] = need_translate
        if need_translate and provider_name and api_key:
            print(f"[fetch] 直传模式：开始翻译...")
            article = translate_article_with_llm(article, provider_name, model or None, api_key)
        elif need_translate:
            article['need_translate'] = True
        print(f"[fetch] 直传模式返回: title='{article['title'][:50]}', translated={article.get('is_translated',False)}")
        return jsonify({'success': True, 'url': url or '#', 'article': article})

    if not url:
        return jsonify({'success': False, 'error': '请提供有效的新闻链接'}), 400
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # ── URL 预处理：还原 Google News / 各类加密重定向链接 ──
    original_url = url
    try:
        if 'news.google.com/rss/articles/' in url or 'news.google.com/articles/' in url:
            print(f"[fetch] 检测到 Google News 重定向链接，尝试还原...")
            # Google News 加密重定向链接，通过 HEAD 请求跟随跳转获取真实 URL
            # 先尝试直接 requests 跟随重定向（只取 Location header，不下载内容）
            for headers in HEADERS_POOL[:3]:
                try:
                    h = headers.copy()
                    h['Referer'] = 'https://news.google.com/'
                    resp = requests.head(url, headers=h, timeout=8, allow_redirects=True, verify=False)
                    if resp.status_code == 200 and resp.url and resp.url != url:
                        real_url = resp.url
                        print(f"[fetch] Google News 重定向还原成功: {real_url}")
                        url = real_url
                        break
                except Exception as redirect_err:
                    print(f"[fetch] Google News 重定向还原失败: {redirect_err}")
            # 如果 HEAD 方法不行，尝试 Jina 预解析获取真实 URL
            if url == original_url:
                try:
                    jina_probe_url = f"https://r.jina.ai/headers/{original_url}"
                    probe_resp = requests.get(jina_probe_url, headers={
                        'Accept': 'application/json',
                        'X-No-Cache': 'true',
                    }, timeout=8, verify=False)
                    if probe_resp.status_code == 200:
                        try:
                            probe_data = probe_resp.json()
                            real = probe_data.get('url') or probe_data.get('location', '')
                            if real and real != original_url:
                                url = real
                                print(f"[fetch] Jina 预解析还原 Google News 链接: {url}")
                        except:
                            pass
                except:
                    pass
        # 检测其他常见的重定向/短链接模式
        elif any(pattern in url for pattern in ['t.co/', 'bit.ly/', 'goo.gl/', 'dwz.cn/', 'suo.im/', 't.cn/']):
            print(f"[fetch] 检测到短链接: {url}")
            try:
                resp = requests.head(url, timeout=8, allow_redirects=True, verify=False,
                                    headers={'User-Agent': HEADERS_POOL[0]['User-Agent']})
                if resp.status_code == 200 and resp.url and resp.url != url:
                    print(f"[fetch] 短链接还原: {url} -> {resp.url}")
                    url = resp.url
            except:
                pass
    except Exception as preprocess_err:
        print(f"[fetch] URL 预处理异常: {preprocess_err}")

    print(f"[fetch] 开始解析: {url}")
    print(f"[fetch] LLM配置: provider={provider_name}, model={model}, has_key={'是' if api_key else '否'}")

    content, error, is_markdown = fetch_url_fallback(url)
    if not content:
        print(f"[fetch] 抓取失败: {error}")
        return jsonify({
            'success': False,
            'error': f'已自动尝试 5 种抓取方案（TLS指纹/Cloudflare破解/多UA/Jina Reader/公共代理）均未能访问该链接。\n\n技术详情：{error}\n\n建议：在浏览器中打开该链接，手动复制正文后使用「手动粘贴正文」功能。'
        }), 422

    # 根据内容格式选择不同的提取方式
    if is_markdown:
        print(f"[fetch] 使用 Markdown 提取模式")
        article = extract_article_from_markdown(content, url)
    else:
        article = extract_article(content, url)

    print(f"[fetch] 首次提取结果: title='{article['title'][:50]}', fullText长度={len(article['fullText'])}, paragraphs={article['paragraphCount']}")

    # 内容质量检测：识别反爬/验证码页面
    def _is_low_quality(art):
        text_len = len(art.get('fullText', ''))
        title = art.get('title', '').lower()
        full = art.get('fullText', '').lower()
        # 内容过短
        if text_len < 200:
            return True, f"正文过短({text_len}字)"
        # 检测到反爬/验证码页面特征
        bot_signals = ['are you a robot', 'captcha', 'verify you are human',
                       'access denied', 'please enable javascript', 'enable cookies',
                       '您的访问请求被拒绝', 'cf-browser-verification', 'just a moment']
        for sig in bot_signals:
            if sig in title or sig in full[:500]:
                return True, f"检测到反爬页面特征: {sig}"
        return False, None

    low_quality, reason = _is_low_quality(article)

    # 如果直接请求成功但内容质量差，尝试 Jina 回退
    if not is_markdown and low_quality:
        print(f"[fetch] 内容质量差({reason})，尝试 Jina 精准提取...")
        md_text, jina_error = fetch_url_via_jina(url, timeout=25)
        if md_text:
            article_jina = extract_article_from_markdown(md_text, url)
            low_q_jina, _ = _is_low_quality(article_jina)
            # 只要 Jina 内容更好（更长 或 质量更高）就采用
            if not low_q_jina or len(article_jina['fullText']) > len(article['fullText']):
                article = article_jina
                low_quality, reason = _is_low_quality(article)
                print(f"[fetch] Jina 精准提取成功: title='{article['title'][:50]}', fullText长度={len(article['fullText'])}")
            else:
                print(f"[fetch] Jina 质量也不佳，继续使用原结果")

    # 最终质量检查：如果正文仍然为空或极短，返回有意义的错误
    if len(article['fullText']) < 100:
        # 特殊处理：Google News RSS 链接无法直接抓取原文
        if 'news.google.com' in url:
            return jsonify({
                'success': False,
                'error': 'Google News 的链接是中转链接，无法直接抓取原文内容。\n\n建议：\n1. 在新闻雷达列表中直接点击「使用」按钮（会导入已有的摘要）\n2. 或者在浏览器中打开链接，复制正文后使用「✍️ 手动粘贴正文」功能'
            }), 422
        return jsonify({
            'success': False,
            'error': '已通过多种方案尝试抓取，但无法提取有效正文内容。\n\n可能原因：\n1. 该网站需要登录后才能查看全文\n2. 页面内容由 JavaScript 动态渲染\n3. 网站设置了强力反爬机制\n4. 链接已失效（404/410）\n\n建议：在浏览器中打开该链接，复制正文内容，然后使用「✍️ 手动粘贴正文」功能。'
        }), 422

    # 检测是否为英文内容，自动翻译
    combined_text = article.get('title', '') + ' ' + article.get('fullText', '')[:500]
    need_translate = is_english_content(combined_text)
    article['is_english'] = need_translate
    print(f"[fetch] 英文检测: {need_translate}")

    if need_translate and provider_name and api_key:
        # 有 LLM 配置，使用 LLM 翻译
        print(f"[fetch] 开始翻译...")
        article = translate_article_with_llm(article, provider_name, model or None, api_key)
        print(f"[fetch] 翻译完成: is_translated={article.get('is_translated')}, error={article.get('translate_error', '无')}")
    elif need_translate:
        # 没有 LLM 配置，标记需要翻译
        article['need_translate'] = True
        article['translate_error'] = '未配置 LLM，无法自动翻译英文内容。请在侧边栏配置 LLM 服务后重试。'
        print(f"[fetch] 需要翻译但未配置LLM")

    print(f"[fetch] 返回成功: title='{article['title'][:50]}', keywords={article['keywords'][:3]}")
    return jsonify({'success': True, 'url': url, 'article': article})


def _fetch_fulltext_for_article(article):
    """为单条新闻抓取全文，失败则保留原始 summary 作为 fullText"""
    url = article.get('url', '')
    if not url or not url.startswith('http'):
        article['fullText'] = article.get('summary', '')
        article['fullTextStatus'] = 'no_url'
        return
    try:
        content, error, is_markdown = fetch_url_fallback(url, timeout=12)
        if not content:
            print(f"[fulltext] 抓取失败: {url[:60]}... err={error}")
            article['fullText'] = article.get('summary', '')
            article['fullTextStatus'] = 'failed'
            return
        if is_markdown:
            extracted = extract_article_from_markdown(content, url)
        else:
            extracted = extract_article(content, url)
        full = extracted.get('fullText', '')
        if len(full) > 50:
            article['fullText'] = full
            article['fullTextStatus'] = 'ok'
        else:
            article['fullText'] = article.get('summary', '')
            article['fullTextStatus'] = 'too_short'
    except Exception as e:
        print(f"[fulltext] 异常: {url[:60]}... err={str(e)[:80]}")
        article['fullText'] = article.get('summary', '')
        article['fullTextStatus'] = 'error'


def _batch_fetch_fulltext(articles, max_workers=8):
    """并行批量抓取新闻全文"""
    threads = []
    for art in articles:
        t = threading.Thread(target=_fetch_fulltext_for_article, args=(art,))
        threads.append(t)
        t.start()
        # 控制并发数
        while sum(1 for th in threads if th.is_alive()) >= max_workers:
            time.sleep(0.05)
    for t in threads:
        t.join(timeout=15)
    ok_count = sum(1 for a in articles if a.get('fullTextStatus') == 'ok')
    print(f"[fulltext] 批量抓取完成: {ok_count}/{len(articles)} 成功")


@app.route('/api/news_search', methods=['GET'])
def news_search():
    """搜索关键词相关新闻（聚合多源）"""
    keyword = request.args.get('keyword', '').strip()
    count = request.args.get('count', 15)
    include_foreign = request.args.get('foreign', '1') == '1'
    try:
        count = min(int(count), 30)
    except (ValueError, TypeError):
        count = 15

    if not keyword:
        return jsonify({"success": False, "error": "请提供搜索关键词"}), 400

    try:
        articles = search_news_aggregated(keyword, count, include_foreign=include_foreign)
        print(f"[news_search] keyword={keyword}, results={len(articles)}, foreign={include_foreign}")
        # 批量抓取全文
        _batch_fetch_fulltext(articles)
        return jsonify({
            "success": True,
            "keyword": keyword,
            "count": len(articles),
            "cached": _cache_get(f"{keyword}_{include_foreign}") is not None,
            "foreign_enabled": include_foreign,
            "articles": articles,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/news_radar', methods=['GET'])
def news_radar():
    """多关键词新闻雷达（同时监控多个关键词）"""
    keywords_raw = request.args.get('keywords', '').strip()
    count = request.args.get('count', 20)
    include_foreign = request.args.get('foreign', '1') == '1'
    try:
        count = min(int(count), 50)
    except (ValueError, TypeError):
        count = 20

    if not keywords_raw:
        return jsonify({"success": False, "error": "请提供关键词（多个关键词用逗号分隔）"}), 400

    # 支持多种分隔符：逗号、中文逗号、斜杠、分号
    keywords = re.split(r'[,，/;；|]+', keywords_raw)
    keywords = [k.strip() for k in keywords if k.strip()]

    if not keywords:
        return jsonify({"success": False, "error": "请提供有效的关键词"}), 400

    try:
        per_keyword = max(5, count // len(keywords))
        articles = search_news_multi_keywords(keywords, count_per_keyword=per_keyword, include_foreign=include_foreign)
        # 批量抓取全文
        _batch_fetch_fulltext(articles)
        return jsonify({
            "success": True,
            "keywords": keywords,
            "count": len(articles),
            "articles": articles,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/fetch_single', methods=['POST', 'OPTIONS'])
def fetch_single_article():
    """按需抓取单条新闻全文（用户点击"一键生成"时调用）"""
    if request.method == 'OPTIONS':
        return '', 204

    data = request.get_json(force=True)
    url = data.get('url', '').strip()
    title = data.get('title', '')
    summary = data.get('summary', '')
    source = data.get('source', '')

    if not url or not url.startswith('http'):
        return jsonify({
            "success": True,
            "fullText": summary or '',
            "fullTextStatus": "no_url",
        })

    try:
        # 尝试抓取全文
        content, error, is_markdown = fetch_url_fallback(url, timeout=20)

        if content:
            if is_markdown:
                extracted = extract_article_from_markdown(content, url)
            else:
                extracted = extract_article(content, url)

            full = extracted.get('fullText', '')

            # 质量检测：如果直接抓取结果质量差，尝试 Jina 回退
            if len(full) < 200 and not is_markdown:
                md_text, jina_error = fetch_url_via_jina(url, timeout=20)
                if md_text:
                    article_jina = extract_article_from_markdown(md_text, url)
                    if len(article_jina.get('fullText', '')) > len(full):
                        full = article_jina['fullText']

            if len(full) > 100:
                # 用提取到的标题和摘要（如果原始搜索结果更差的话）
                better_title = extracted.get('title', '') or title
                better_summary = extracted.get('summary', '') or summary
                return jsonify({
                    "success": True,
                    "fullText": full,
                    "fullTextStatus": "ok",
                    "title": better_title if len(better_title) > len(title) else title,
                    "summary": better_summary if len(better_summary) > len(summary) else summary,
                })
            else:
                # 抓到了但内容太短，返回已有摘要
                return jsonify({
                    "success": True,
                    "fullText": summary,
                    "fullTextStatus": "too_short",
                })
        else:
            # 全部抓取方式都失败了
            return jsonify({
                "success": True,
                "fullText": summary,
                "fullTextStatus": "failed",
                "error": error[:200] if error else "所有抓取方式均失败",
            })
    except Exception as e:
        return jsonify({
            "success": True,
            "fullText": summary,
            "fullTextStatus": "error",
            "error": str(e)[:200],
        })


@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    """获取美股实时行情数据（支持 XWIN 等）"""
    ticker = request.args.get('ticker', 'XWIN').upper()

    # 美股数据接口配置
    FINANCE_API_URL = "https://www.codebuddy.cn/v2/tool/financedata"

    try:
        # 调用金融数据 API 获取最新日线数据（含更多字段）
        payload = {
            "api_name": "us_daily",
            "params": {"ts_code": ticker, "limit": 1},
            "fields": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_change,vol,amount,total_mv,pe,pb,turnover_ratio"
        }

        resp = requests.post(FINANCE_API_URL, json=payload, timeout=10)
        result = resp.json()

        if result.get('code') == 0 and result.get('data') and result['data'].get('items'):
            items = result['data']['items']
            if len(items) > 0:
                # 解析返回数据
                item = items[0]
                return jsonify({
                    "success": True,
                    "data": {
                        "symbol": item[0],
                        "trade_date": item[1],
                        "open": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "close": float(item[5]),
                        "pre_close": float(item[6]),
                        "change": float(item[7]) if item[7] else 0,
                        "change_percent": float(item[8]) if item[8] else 0,
                        "volume": int(item[9]) if item[9] else 0,
                        "amount": float(item[10]) if item[10] else 0,
                        "market_cap": float(item[11]) if item[11] else 0,
                        "pe": float(item[12]) if item[12] and item[12] != '-' else None,
                        "pb": float(item[13]) if item[13] and item[13] != '-' else None,
                        "turnover_ratio": float(item[14]) if item[14] and item[14] != '-' else None,
                    }
                })

        # API 额度用完，尝试从雅虎财经获取实时数据
        try:
            yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            yahoo_resp = requests.get(yahoo_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            yahoo_data = yahoo_resp.json()

            if yahoo_data.get('chart') and yahoo_data['chart'].get('result'):
                result_data = yahoo_data['chart']['result'][0]
                meta = result_data['meta']
                quote = result_data.get('indicators', {}).get('quote', [{}])[0]

                # 获取最新价格
                current_price = meta.get('regularMarketPrice', 0)
                previous_close = meta.get('previousClose', 0)
                change = current_price - previous_close
                change_percent = (change / previous_close * 100) if previous_close else 0
                
                # 获取市值、总股数 - 先尝试 Nasdaq API，再尝试 Yahoo quoteSummary
                market_cap = 0
                shares_outstanding = 0  # 总股数
                try:
                    # 方案1：从 Nasdaq 获取总股数，乘以当前价格
                    import subprocess
                    nasdaq_resp = subprocess.run(
                        ['curl', '-s', '-A', 'Mozilla/5.0', '-H', 'Referer: https://www.nasdaq.com/',
                         f'https://api.nasdaq.com/api/company/{ticker}/institutional-holdings?limit=1&type=TOTAL&sortColumn=marketValue',
                         '--max-time', '10'],
                        capture_output=True, text=True, timeout=15
                    )
                    nasdaq_data = json.loads(nasdaq_resp.stdout) if nasdaq_resp.stdout else {}
                    shares_info = nasdaq_data.get('data', {}).get('ownershipSummary', {}).get('ShareoutstandingTotal', {})
                    shares_m = shares_info.get('value')
                    if shares_m:
                        shares_outstanding = float(shares_m) * 1e6
                        market_cap = shares_outstanding * current_price
                except Exception as nasdaq_err:
                    print(f"Nasdaq市值获取失败: {nasdaq_err}")

                pe = None
                if not market_cap:
                    try:
                        # 方案2：Yahoo quoteSummary（同时尝试获取 P/E）
                        stats_url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryDetail,defaultKeyStatistics"
                        stats_resp = requests.get(stats_url, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        stats_data = stats_resp.json()
                        if stats_data.get('quoteSummary') and stats_data['quoteSummary'].get('result'):
                            result0 = stats_data['quoteSummary']['result'][0]
                            summary = result0.get('summaryDetail', {})
                            if summary.get('marketCap'):
                                market_cap = summary['marketCap'].get('raw', 0)
                            if summary.get('trailingPE'):
                                pe = summary['trailingPE'].get('raw', None)
                            # 尝试从 defaultKeyStatistics 获取总股数
                            if not shares_outstanding:
                                ks = result0.get('defaultKeyStatistics', {})
                                if ks.get('sharesOutstanding'):
                                    shares_outstanding = ks['sharesOutstanding'].get('raw', 0)
                    except Exception as stats_err:
                        print(f"Yahoo市值获取失败: {stats_err}")

                vol = meta.get('regularMarketVolume', 0)
                # 换手率 = 当日成交量 / 总股数 × 100%
                turnover_ratio = None
                if vol and shares_outstanding > 0:
                    turnover_ratio = round((vol / shares_outstanding) * 100, 2)

                return jsonify({
                    "success": True,
                    "data": {
                        "symbol": ticker,
                        "trade_date": datetime.now().strftime('%Y%m%d'),
                        "open": meta.get('regularMarketOpen', current_price),
                        "high": meta.get('regularMarketDayHigh', current_price),
                        "low": meta.get('regularMarketDayLow', current_price),
                        "close": current_price,
                        "pre_close": previous_close,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": vol,
                        "amount": round(current_price * vol, 2) if vol else 0,
                        "market_cap": round(market_cap, 2) if market_cap else 0,
                        "pe": pe,
                        "pb": None,
                        "turnover_ratio": turnover_ratio,
                    },
                    "source": "yahoo"
                })
        except Exception as yahoo_err:
            print(f"雅虎财经获取失败: {yahoo_err}")

        # 备用方案3：Google Finance（页面抓取，免费可靠）
        try:
            import re as _re
            gf_url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
            gf_resp = requests.get(gf_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            })
            gf_html = gf_resp.text

            # 提取当前价格
            price_m = _re.search(r'data-last-price="([^"]+)"', gf_html)
            if price_m:
                current_price = float(price_m.group(1))

                # 提取各指标（label 在 <div class="mfs7Fc"> 后紧跟值在 <div class="P6K39c">）
                pairs = _re.findall(r'<div class="mfs7Fc"[^>]*>([^<]+)</div>.*?<div class="P6K39c"[^>]*>([^<]*)', gf_html)
                info = {p[0].strip(): p[1].strip() for p in pairs}

                # 解析 Previous close
                prev_close = 0
                if 'Previous close' in info:
                    prev_str = info['Previous close'].replace('$', '').replace(',', '').strip()
                    prev_close = float(prev_str)

                # 解析 Day range → open/high/low 的近似值
                day_range = info.get('Day range', '').replace('$', '')
                day_low, day_high = 0, 0
                if ' - ' in day_range:
                    parts = day_range.split(' - ')
                    day_low = float(parts[0].strip().replace(',', ''))
                    day_high = float(parts[1].strip().replace(',', ''))

                # Market cap
                market_cap = 0
                if 'Market cap' in info:
                    mcap_str = info['Market cap']
                    mcap_m = _re.search(r'([\d.]+)\s*([BMTK])', mcap_str)
                    if mcap_m:
                        val = float(mcap_m.group(1))
                        unit = mcap_m.group(2)
                        multiplier = {'B': 1e9, 'M': 1e6, 'K': 1e3, 'T': 1e12}
                        market_cap = val * multiplier.get(unit, 1)

                # Volume
                volume = 0
                if 'Avg Volume' in info:
                    vol_str = info['Avg Volume']
                    vol_m = _re.search(r'([\d.]+)\s*([BMTK])', vol_str)
                    if vol_m:
                        val = float(vol_m.group(1))
                        unit = vol_m.group(2)
                        multiplier = {'B': 1e9, 'M': 1e6, 'K': 1e3, 'T': 1e12}
                        volume = int(val * multiplier.get(unit, 1))

                # P/E
                pe = None
                if 'P/E ratio' in info and info['P/E ratio'].strip() not in ('-', 'N/A', ''):
                    try:
                        pe = float(info['P/E ratio'].strip())
                    except:
                        pass

                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0

                # 换手率 = volume / 总股数 × 100%（总股数从市值/价格反推）
                turnover_ratio = None
                if volume > 0 and market_cap > 0 and current_price > 0:
                    shares_outstanding = market_cap / current_price
                    turnover_ratio = round((volume / shares_outstanding) * 100, 2)

                print(f"[Google Finance] {ticker} price=${current_price}, prev=${prev_close}, change={change_percent:+.2f}%")
                return jsonify({
                    "success": True,
                    "data": {
                        "symbol": ticker,
                        "trade_date": datetime.now().strftime('%Y%m%d'),
                        "open": day_low,
                        "high": day_high,
                        "low": day_low,
                        "close": current_price,
                        "pre_close": prev_close,
                        "change": round(change, 4),
                        "change_percent": round(change_percent, 2),
                        "volume": volume,
                        "amount": round(current_price * volume, 2) if volume else 0,
                        "market_cap": market_cap,
                        "pe": pe,
                        "pb": None,
                        "turnover_ratio": turnover_ratio,
                    },
                    "source": "google"
                })
        except Exception as gf_err:
            print(f"Google Finance 获取失败: {gf_err}")

        # 备用方案4：Stock Analysis（简单页面抓取）
        try:
            sa_url = f"https://stockanalysis.com/stocks/{ticker.lower()}/"
            sa_resp = requests.get(sa_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            sa_html = sa_resp.text

            # Stock Analysis 的当前价格通常在 h1 或特定 span 中
            sa_price_m = _re.search(r'<span[^>]*class="[^"]*price[^"]*"[^>]*>(\$?[\d,]+\.\d+)', sa_html, _re.IGNORECASE)
            if not sa_price_m:
                sa_price_m = _re.search(r'data-test="qa-stock-price"[^>]*>(\$?[\d,]+\.\d+)', sa_html)
            if sa_price_m:
                import re as _re
                sa_price = float(sa_price_m.group(1).replace('$', '').replace(',', ''))

                # 尝试找 Previous close
                sa_prev = 0
                prev_m = _re.search(r'Previous Close[^<]*</(?:td|div|span)>\s*<(?:td|div|span)[^>]*>\s*\$?([\d,]+\.\d+)', sa_html, _re.IGNORECASE)
                if prev_m:
                    sa_prev = float(prev_m.group(1).replace(',', ''))

                sa_change = sa_price - sa_prev
                sa_change_pct = (sa_change / sa_prev * 100) if sa_prev else 0

                print(f"[Stock Analysis] {ticker} price=${sa_price}, prev=${sa_prev}")
                return jsonify({
                    "success": True,
                    "data": {
                        "symbol": ticker,
                        "trade_date": datetime.now().strftime('%Y%m%d'),
                        "open": sa_price,
                        "high": sa_price,
                        "low": sa_price,
                        "close": sa_price,
                        "pre_close": sa_prev,
                        "change": round(sa_change, 4),
                        "change_percent": round(sa_change_pct, 2),
                        "volume": 0,
                        "amount": 0,
                        "market_cap": 0,
                        "pe": None,
                        "pb": None,
                        "turnover_ratio": None,
                    },
                    "source": "stockanalysis"
                })
        except Exception as sa_err:
            print(f"Stock Analysis 获取失败: {sa_err}")

        # 所有数据源都失败，返回错误
        return jsonify({
            "success": False,
            "error": "无法获取实时股价数据，请稍后重试"
        }), 503

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _get_kline_4h(ticker, limit):
    """通过 Yahoo Finance 1h 数据聚合为 4h K线"""
    import datetime as dt

    now = dt.datetime.now()
    # 1h数据最多7天（Yahoo限制），需要拉取足够多的1h柱
    # limit根4h蜡烛数，每个蜡烛=4根1h柱，需要 limit*4 + padding
    days_needed = min(30, max(7, (limit * 4 + 24) / 6.5))  # 美股每天约6.5小时
    start = now - dt.timedelta(days=days_needed)

    yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        'period1': int(start.timestamp()),
        'period2': int(now.timestamp()),
        'interval': '1h',
    }

    try:
        yahoo_resp = requests.get(yahoo_url, params=params, timeout=20, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        yahoo_data = yahoo_resp.json()

        if not yahoo_data.get('chart') or not yahoo_data['chart'].get('result'):
            return jsonify({"success": False, "error": "Yahoo 4h数据获取失败"}), 503

        result_data = yahoo_data['chart']['result'][0]
        timestamps = result_data.get('timestamp', [])
        quote = result_data.get('indicators', {}).get('quote', [{}])[0]
        opens = quote.get('open', [])
        highs = quote.get('high', [])
        lows = quote.get('low', [])
        closes = quote.get('close', [])
        volumes = quote.get('volume', [])

        # 先收集1h数据
        hourly = []
        for i in range(len(timestamps)):
            if closes[i] is None:
                continue
            hourly.append({
                'ts': timestamps[i],
                'date': dt.datetime.fromtimestamp(timestamps[i]),
                'open': float(opens[i]) if opens[i] else float(closes[i]),
                'high': float(highs[i]) if highs[i] else float(closes[i]),
                'low': float(lows[i]) if lows[i] else float(closes[i]),
                'close': float(closes[i]),
                'volume': int(volumes[i]) if volumes[i] else 0,
            })

        if not hourly:
            return jsonify({"success": False, "error": "无1h数据"}), 503

        # 按4小时分组聚合
        # 以 0:00, 4:00, 8:00, 12:00, 16:00, 20:00 为边界对齐
        kline_data = []
        group = []
        for h in hourly:
            if not group:
                group.append(h)
                continue
            prev = group[0]
            # 同一个4h窗口：同一小时区间 [0,3], [4,7], [8,11], [12,15], [16,19], [20,23]
            # 且同一天
            same_window = (h['date'].hour // 4 == prev['date'].hour // 4 and h['date'].date() == prev['date'].date())
            if same_window:
                group.append(h)
            else:
                bar = _aggregate_group(group)
                if bar:
                    kline_data.append(bar)
                group = [h]
        # 处理最后一组
        if group:
            bar = _aggregate_group(group)
            if bar:
                kline_data.append(bar)

        # 计算涨跌幅
        for i in range(1, len(kline_data)):
            prev = kline_data[i-1]['close']
            if prev > 0:
                kline_data[i]['pct_change'] = ((kline_data[i]['close'] - prev) / prev) * 100

        kline_data = kline_data[-limit:]

        all_highs = [d['high'] for d in kline_data]
        all_lows = [d['low'] for d in kline_data]
        return jsonify({
            "success": True,
            "data": kline_data,
            "week52_high": max(all_highs) if all_highs else None,
            "week52_low": min(all_lows) if all_lows else None,
            "source": "yahoo_4h",
        })
    except Exception as e:
        print(f"K线4h失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 503


def _aggregate_group(group):
    """将一组1h柱聚合为一根K线"""
    if not group:
        return None
    bar = {
        'date': group[0]['date'].strftime('%m/%d %H:%M'),
        'open': group[0]['open'],
        'high': max(g['high'] for g in group),
        'low': min(g['low'] for g in group),
        'close': group[-1]['close'],
        'volume': sum(g['volume'] for g in group),
        'amount': 0,
        'pct_change': 0,
    }
    return bar


@app.route('/api/stock/kline', methods=['GET'])
def get_stock_kline():
    """获取美股K线历史数据（用于绘制K线图）"""
    ticker = request.args.get('ticker', 'XWIN').upper()
    period = request.args.get('period', 'daily')  # daily / 4h
    limit = min(int(request.args.get('limit', '120')), 500)

    FINANCE_API_URL = "https://www.codebuddy.cn/v2/tool/financedata"

    # 4h 周期仅走 Yahoo Finance（金融API不支持小时级）
    if period == '4h':
        return _get_kline_4h(ticker, limit)

    # 方案1：金融数据 API（仅日线）
    try:
        payload = {
            "api_name": "us_daily",
            "params": {"ts_code": ticker, "limit": limit},
            "fields": "ts_code,trade_date,open,high,low,close,vol,amount,pct_change"
        }
        resp = requests.post(FINANCE_API_URL, json=payload, timeout=15)
        result = resp.json()

        if result.get('code') == 0 and result.get('data') and result['data'].get('items') and len(result['data']['items']) > 1:
            items = result['data']['items']
            items.reverse()
            kline_data = []
            for item in items:
                kline_data.append({
                    "date": str(item[1]),
                    "open": float(item[2]),
                    "high": float(item[3]),
                    "low": float(item[4]),
                    "close": float(item[5]),
                    "volume": int(item[6]) if item[6] else 0,
                    "amount": float(item[7]) if item[7] else 0,
                    "pct_change": float(item[8]) if item[8] else 0,
                })
            all_highs = [d['high'] for d in kline_data]
            all_lows = [d['low'] for d in kline_data]
            return jsonify({
                "success": True,
                "data": kline_data,
                "week52_high": max(all_highs) if all_highs else None,
                "week52_low": min(all_lows) if all_lows else None,
            })
    except Exception as e:
        print(f"K线金融API失败: {e}")

    # 方案2：Yahoo Finance chart API
    try:
        # 计算时间范围
        import datetime as dt
        now = dt.datetime.now()
        ranges = {
            '60': (now - dt.timedelta(days=90), '1d'),
            '120': (now - dt.timedelta(days=180), '1d'),
            '250': (now - dt.timedelta(days=370), '1d'),
        }
        start, interval = ranges.get(str(limit), (now - dt.timedelta(days=90), '1d'))

        yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            'period1': int(start.timestamp()),
            'period2': int(now.timestamp()),
            'interval': interval,
        }
        yahoo_resp = requests.get(yahoo_url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        yahoo_data = yahoo_resp.json()

        if yahoo_data.get('chart') and yahoo_data['chart'].get('result'):
            result_data = yahoo_data['chart']['result'][0]
            timestamps = result_data.get('timestamp', [])
            quote = result_data.get('indicators', {}).get('quote', [{}])[0]
            opens = quote.get('open', [])
            highs = quote.get('high', [])
            lows = quote.get('low', [])
            closes = quote.get('close', [])
            volumes = quote.get('volume', [])

            kline_data = []
            for i in range(len(timestamps)):
                if closes[i] is None: continue
                kline_data.append({
                    "date": dt.datetime.fromtimestamp(timestamps[i]).strftime('%Y%m%d'),
                    "open": float(opens[i]) if opens[i] else float(closes[i]),
                    "high": float(highs[i]) if highs[i] else float(closes[i]),
                    "low": float(lows[i]) if lows[i] else float(closes[i]),
                    "close": float(closes[i]),
                    "volume": int(volumes[i]) if volumes[i] else 0,
                    "amount": 0,
                    "pct_change": 0,
                })
            # 计算涨跌幅
            for i in range(1, len(kline_data)):
                prev = kline_data[i-1]['close']
                if prev > 0:
                    kline_data[i]['pct_change'] = ((kline_data[i]['close'] - prev) / prev) * 100
            # 只保留最后limit条
            kline_data = kline_data[-limit:]

            all_highs = [d['high'] for d in kline_data]
            all_lows = [d['low'] for d in kline_data]
            return jsonify({
                "success": True,
                "data": kline_data,
                "week52_high": max(all_highs) if all_highs else None,
                "week52_low": min(all_lows) if all_lows else None,
                "source": "yahoo",
            })
    except Exception as e:
        print(f"K线Yahoo失败: {e}")

    # 方案3：Stock Analysis（页面表格抓取，免费可靠）
    try:
        import re as _re
        sa_url = f"https://stockanalysis.com/stocks/{ticker.lower()}/history/"
        sa_resp = requests.get(sa_url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        sa_html = sa_resp.text
        tables = _re.findall(r'<table[^>]*>(.*?)</table>', sa_html, _re.DOTALL)
        if tables:
            rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', tables[0], _re.DOTALL)
            kline_data = []
            from datetime import datetime as _dt
            for row in rows[1:]:  # 跳过表头
                cells = _re.findall(r'<td[^>]*>(.*?)</td>', row)
                if len(cells) >= 8:  # Date, Open, High, Low, Close, Adj.Close, Change%, Volume
                    date_str = cells[0].strip()
                    try:
                        date_obj = _dt.strptime(date_str, '%b %d, %Y')
                        date_fmt = date_obj.strftime('%Y%m%d')
                    except:
                        continue
                    open_p = float(cells[1].strip().replace(',', ''))
                    high_p = float(cells[2].strip().replace(',', ''))
                    low_p = float(cells[3].strip().replace(',', ''))
                    close_p = float(cells[4].strip().replace(',', ''))
                    # Volume 在第 8 列 (index 7)
                    vol_str = _re.sub(r'[^0-9]', '', cells[7].strip())
                    volume = int(vol_str) if vol_str else 0
                    kline_data.append({
                        "date": date_fmt,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": volume,
                        "amount": 0,
                        "pct_change": 0,
                    })
            # Stock Analysis 表格从新到旧排列，需反转为旧→新
            kline_data.reverse()
            # 计算涨跌幅
            for i in range(1, len(kline_data)):
                prev = kline_data[i-1]['close']
                if prev > 0:
                    kline_data[i]['pct_change'] = ((kline_data[i]['close'] - prev) / prev) * 100
            kline_data = kline_data[-limit:]
            if len(kline_data) > 1:
                all_highs = [d['high'] for d in kline_data]
                all_lows = [d['low'] for d in kline_data]
                print(f"[Stock Analysis] K线获取成功: {len(kline_data)} 条")
                return jsonify({
                    "success": True,
                    "data": kline_data,
                    "week52_high": max(all_highs) if all_highs else None,
                    "week52_low": min(all_lows) if all_lows else None,
                    "source": "stockanalysis",
                })
    except Exception as e:
        print(f"K线Stock Analysis失败: {e}")

    return jsonify({"success": False, "error": "K线数据获取失败"}), 503


@app.route('/api/stock/financials', methods=['GET'])
def get_stock_financials():
    """获取美股财务指标摘要（PE、PB、换手率等）"""
    ticker = request.args.get('ticker', 'XWIN').upper()

    FINANCE_API_URL = "https://www.codebuddy.cn/v2/tool/financedata"

    try:
        # 获取最近一条日线数据（含财务指标字段）
        payload = {
            "api_name": "us_daily",
            "params": {"ts_code": ticker, "limit": 1},
            "fields": "ts_code,trade_date,close,total_mv,pe,pb,vol,amount,turnover_ratio"
        }
        resp = requests.post(FINANCE_API_URL, json=payload, timeout=10)
        result = resp.json()

        if result.get('code') == 0 and result.get('data') and result['data'].get('items'):
            item = result['data']['items'][0]
            return jsonify({
                "success": True,
                "data": {
                    "trade_date": str(item[1]),
                    "close": float(item[2]),
                    "total_mv": float(item[3]) if item[3] else 0,
                    "pe": float(item[4]) if item[4] and str(item[4]) != '-' else None,
                    "pb": float(item[5]) if item[5] and str(item[5]) != '-' else None,
                    "volume": int(item[6]) if item[6] else 0,
                    "amount": float(item[7]) if item[7] else 0,
                    "turnover_ratio": float(item[8]) if item[8] and str(item[8]) != '-' else None,
                }
            })
        return jsonify({"success": False, "error": "财务指标获取失败"}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/announcements', methods=['GET'])
def get_announcements():
    """获取 XMAX 公司公告（优先东方财富，备用 SEC EDGAR）"""
    ticker = request.args.get('ticker', 'XWIN').upper()

    try:
        announcements = []

        # ============================================================
        # 方案1（优先）: 东方财富美股公告 API
        # API: np-anotice-stock.eastmoney.com/api/security/ann
        # ============================================================
        try:
            eastmoney_url = "https://np-anotice-stock.eastmoney.com/api/security/ann"
            params = {
                'ann_type': 'U,U_Pink,U_ETF',
                'client_source': 'web',
                'stock_list': ticker,
                'page_index': '1',
                'page_size': '10',
                'st': 'notice_date',
                'sr': '-1',
                'cb': '_ntes_quote_callback',
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Referer': f'https://data.eastmoney.com/notices/stock/{ticker}.html',
                'Accept': '*/*',
            }
            resp = requests.get(eastmoney_url, params=params, timeout=15, headers=headers)

            if resp.status_code == 200:
                # 解析 JSONP 响应
                text = resp.text
                # 去掉 JSONP 回调包裹: _ntes_quote_callback({...}); 或 _ntes_quote_callback({...})
                start = text.find('(')
                end = text.rfind(')')
                if start >= 0 and end > start:
                    json_str = text[start+1:end]
                    data = json.loads(json_str)

                    if data.get('success') == 1 and data.get('data', {}).get('list'):
                        items = data['data']['list']
                        for i, item in enumerate(items[:10]):
                            title_ch = item.get('title_ch', '')
                            title_en = item.get('title_en', '')
                            title = item.get('title', '')
                            art_code = item.get('art_code', '')
                            notice_date = item.get('notice_date', '')

                            # 优先使用中文标题
                            display_title = title_ch if title_ch else title_en
                            if not display_title:
                                display_title = title

                            # 提取类别标签
                            columns = item.get('columns', [])
                            col_names = [c.get('column_name', '') for c in columns if c.get('column_name')]
                            col_label = col_names[0] if col_names else ''

                            # 格式化日期
                            time_formatted = ''
                            if notice_date:
                                try:
                                    dt = datetime.strptime(notice_date[:10], '%Y-%m-%d')
                                    time_formatted = dt.strftime('%Y-%m-%d')
                                except:
                                    time_formatted = notice_date[:10]

                            # 东方财富公告详情页 URL
                            filing_url = f"https://data.eastmoney.com/notices/detail/{ticker}/{art_code}.html"
                            if not art_code:
                                filing_url = f"https://data.eastmoney.com/notices/stock/{ticker}.html"

                            if display_title and len(display_title) > 2:
                                announcements.append({
                                    "id": f"EM-{art_code or i+1}",
                                    "title": display_title,
                                    "source": "东方财富",
                                    "time": time_formatted,
                                    "type": col_label,
                                    "url": filing_url
                                })

                        if announcements:
                            print(f"东方财富公告获取成功: {len(announcements)} 条")

        except Exception as e:
            print(f"东方财富公告 API 获取失败: {e}")

        # ============================================================
        # 方案2（备用）: SEC EDGAR Full-Text Search API
        # ============================================================
        if not announcements:
            XMAX_CIK = '0001473334'

            category_map = {
                '8-K': '重大事项报告',
                '8-K/A': '重大事项报告（修正）',
                '10-K': '年度报告',
                '10-Q': '季度报告',
                '424B2': '证券发行说明书',
                '424B3': '证券发行说明书',
                '424B4': '证券发行说明书',
                '424B5': '证券发行说明书',
                'S-1': '招股说明书',
                'SC 14D-9': '重大变更报告',
                'DEF 14A': '委托声明书',
            }

            item_map = {
                '1.01': '签署重大协议',
                '2.01': '资产完成收购',
                '2.02': '经营业绩/财务状况',
                '5.02': '董事/高管离职',
                '5.03': '董事/高管任命',
                '9.01': '财务报表和附件',
            }

            try:
                sec_url = "https://efts.sec.gov/LATEST/search-index?q=%22XMAX%20Inc%22&dateRange=custom&startdt=2024-01-01&enddt=2026-12-31&from=0&num=15"
                sec_resp = requests.get(sec_url, timeout=15, headers={
                    'User-Agent': 'NewsInspiration/1.0 contact@example.com',
                    'Accept': 'application/json',
                })

                if sec_resp.status_code == 200:
                    sec_data = sec_resp.json()
                    hits = sec_data.get('hits', {}).get('hits', []) if isinstance(sec_data.get('hits'), dict) else []

                    for i, hit in enumerate(hits[:15]):
                        source = hit.get('_source', {})
                        ciks = source.get('ciks', [])
                        if isinstance(ciks, list) and XMAX_CIK not in ciks:
                            continue

                        form = source.get('form', '')
                        filed_date = source.get('file_date', '')
                        adsh = source.get('adsh', '')
                        items = source.get('items', [])

                        filing_type_cn = category_map.get(form, form)
                        item_descs = [item_map[x] for x in items if x in item_map]
                        item_text = chr(12290).join(item_descs) if item_descs else ''

                        title = filing_type_cn
                        if item_text:
                            title += f" - {item_text}"

                        time_formatted = ''
                        if filed_date:
                            try:
                                dt = datetime.strptime(filed_date, '%Y-%m-%d')
                                time_formatted = dt.strftime('%Y-%m-%d')
                            except:
                                time_formatted = filed_date

                        filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={XMAX_CIK}"
                        if adsh:
                            adsh_clean = adsh.replace('-', '')
                            filing_url = f"https://www.sec.gov/Archives/edgar/data/1473334/{adsh_clean}/{adsh}-index.html"

                        if title and len(title) > 2:
                            announcements.append({
                                "id": f"SEC-{adsh or i+1}",
                                "title": title,
                                "source": "SEC EDGAR",
                                "time": time_formatted,
                                "type": filing_type_cn,
                                "url": filing_url
                            })

            except Exception as e2:
                print(f"SEC EDGAR 获取失败: {e2}")

        # 按时间倒序排列（最新在前）
        announcements.sort(key=lambda x: x.get('time', ''), reverse=True)

        if not announcements:
            return jsonify({
                "success": False,
                "error": "无法获取公告数据"
            }), 503

        return jsonify({
            "success": True,
            "data": announcements,
            "ticker": ticker,
            "count": len(announcements),
            "source": "东方财富" if announcements[0].get('source') == '东方财富' else "SEC EDGAR"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': '深海有象后端服务运行中'})


# 服务端持久化存储 LLM 配置（刷新不丢失）
_llm_config_store = {}  # 内存存储，服务重启后清除（可选：扩展为文件存储）

@app.route('/api/llm-config', methods=['GET', 'POST'])
def llm_config():
    """持久化 LLM 配置：GET 读取、POST 保存"""
    if request.method == 'GET':
        return jsonify({"success": True, "config": _llm_config_store})
    elif request.method == 'POST':
        data = request.get_json(force=True)
        _llm_config_store.update(data)
        return jsonify({"success": True})


@app.route('/', methods=['GET'])
def index():
    from flask import send_from_directory, make_response
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, 'index.html')
    if os.path.exists(html_path):
        resp = make_response(send_from_directory(base_dir, 'index.html'))
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        # 强制不返回 ETag，防止浏览器用缓存
        resp.headers.pop('ETag', None)
        resp.headers.pop('Last-Modified', None)
        return resp
    return jsonify({'status': 'running'})


# ============================================================
#  XWIN 机构持股数据（仅真实数据，不编造）
#  数据源优先级: Finviz → Nasdaq
# ============================================================

# 机构中文标准译名对照表
INST_CN_NAMES = {
    'VANGUARD GROUP INC': '先锋集团',
    'VANGUARD GROUP, INC.': '先锋集团',
    'BlackRock, Inc.': '贝莱德集团',
    'BLACKROCK INC': '贝莱德集团',
    'GEODE CAPITAL MANAGEMENT, LLC': '极地资本管理',
    'GEODE CAPITAL MANAGEMENT LLC': '极地资本管理',
    'STATE STREET CORP': '道富银行',
    'STATE STREET CORPORATION': '道富银行',
    'MARSHALL WACE, LLP': '马歇尔·韦斯',
    'MARSHALL WACE LLP': '马歇尔·韦斯',
    'GOLDMAN SACHS GROUP INC': '高盛集团',
    'GOLDMAN SACHS & CO. LLC': '高盛集团',
    'OMERS ADMINISTRATION Corp': 'OMERS 管理公司',
    'OMERS ADMINISTRATION CORP': 'OMERS 管理公司',
    'NORTHERN TRUST CORP': '北方信托',
    'NORTHERN TRUST COMPANY': '北方信托',
    'MILLENNIUM MANAGEMENT LLC': '千禧管理',
    'JANE STREET GROUP, LLC': 'Jane Street 集团',
    'JANE STREET CAPITAL LLC': 'Jane Street 资本',
    'JPMORGAN CHASE & CO': '摩根大通',
    'MORGAN STANLEY': '摩根士丹利',
    'CITADEL ADVISORS LLC': '城堡投资',
    'CITADEL LLC': '城堡投资',
    'POINT72 ASSET MANAGEMENT LP': 'Point72 资产管理',
    'TWO SIGMA INVESTMENTS LP': 'Two Sigma 投资',
    'D. E. SHAW & CO. LP': 'D.E. Shaw 对冲基金',
    'BRIDGEWATER ASSOCIATES LP': '桥水基金',
    'RENAISSANCE TECHNOLOGIES LLC': '文艺复兴科技',
    'FIDELITY MANAGEMENT & RESEARCH': '富达管理与研究',
    'DIMENSIONAL FUND ADVISORS LP': '维度基金顾问',
    'TORONTO DOMINION BANK': '多伦多道明银行',
    'BANK OF NEW YORK MELLON CORP': '纽约梅隆银行',
    'BANK OF AMERICA CORP': '美国银行',
    'WELLS FARGO & COMPANY': '富国银行',
    'UBS GROUP AG': '瑞银集团',
    'CREDIT SUISSE AG': '瑞士信贷',
    'BARCLAYS PLC': '巴克莱银行',
    'DEUTSCHE BANK AG': '德意志银行',
    'NOMURA SECURITIES': '野村证券',
    'MITSUBISHI UFJ FINANCIAL GROUP': '三菱日联金融集团',
    'SOCIETE GENERALE': '法国兴业银行',
    'BNP PARIBAS': '法国巴黎银行',
    'INVESCO LTD': '景顺集团',
    'FRANKLIN RESOURCES INC': '富兰克林邓普顿',
    'T. ROWE PRICE GROUP INC': '普信集团',
    'AMVESCAP PLC': '景顺',
    'CAPITAL RESEARCH & MANAGEMENT': '资本集团',
    'CAPITAL GROUP COMPANIES INC': '资本集团',
    'NUVEEN ASSET MANAGEMENT LLC': 'Nuveen 资产管理',
    'LEGAL & GENERAL GROUP PLC': '法通保险',
    'AVANTAX ADVISORY SERVICES INC': 'Avantax 咨询',
    'INVESCO CAPITAL MANAGEMENT LLC': '景顺资本管理',
    'GREAT-WEST LIFECOE INC': '宏利金融',
    'GREAT-WEST LIFE ASSURANCE CO': '宏利保险',
    'WELLS FARGO ADVISORS LLC': '富国银行顾问',
    'DEUTSCHE BANK AG': '德意志银行',
    'VANGUARD EXTENDED MARKET INDEX FUND': '先锋扩展市场指数基金',
    'Fidelity Extended Market Index Fund': '富达扩展市场指数基金',
    'Fidelity Nasdaq Composite Index Fund': '富达纳斯达克综合指数基金',
    'Fidelity Total Market Index Fund': '富达全市场指数基金',
    'Victory Extended Market Index Fund': 'Victory 扩展市场指数基金',
    # SEC 13F 新增机构
    'UBS Group AG': '瑞银集团',
    'CITIGROUP INC': '花旗集团',
    'BNP PARIBAS FINANCIAL MARKETS': '法国巴黎银行金融市场',
    'ROYAL BANK OF CANADA': '加拿大皇家银行',
    'Squarepoint Ops LLC': 'Squarepoint 运营',
    'Qube Research & Technologies Ltd': 'Qube 研究与技术',
    'Hudson Bay Capital Management LP': '哈德逊湾资本管理',
    'Schonfeld Strategic Advisors LLC': 'Schonfeld 战略咨询',
    'Lighthouse Investment Partners, LLC': '灯塔投资伙伴',
    'Scientech Research LLC': 'Scientech 研究',
    'Tower Research Capital LLC': 'Tower Research 资本',
    'MANGROVE PARTNERS IM, LLC': '红树林伙伴投资管理',
    'BALYASNY ASSET MANAGEMENT L.P.': 'Balyasny 资产管理',
    'Global Retirement Partners, LLC': '全球退休伙伴',
    'IFP Advisors, Inc': 'IFP 咨询',
    'State of Wyoming': '怀俄明州政府',
    'JPMORGAN CHASE & CO': '摩根大通',
    'MILLENNIUM MANAGEMENT LLC': '千禧管理',
    'TWO SIGMA INVESTMENTS, LP': 'Two Sigma 投资',
    'RENAISSANCE TECHNOLOGIES LLC': '文艺复兴科技',
}

# 机构所在国家（用于地图定位）
INST_COUNTRIES = {
    'VANGUARD GROUP INC': 'US',
    'VANGUARD GROUP, INC.': 'US',
    'BlackRock, Inc.': 'US',
    'BLACKROCK INC': 'US',
    'GEODE CAPITAL MANAGEMENT, LLC': 'US',
    'GEODE CAPITAL MANAGEMENT LLC': 'US',
    'STATE STREET CORP': 'US',
    'STATE STREET CORPORATION': 'US',
    'MARSHALL WACE, LLP': 'GB',
    'MARSHALL WACE LLP': 'GB',
    'GOLDMAN SACHS GROUP INC': 'US',
    'GOLDMAN SACHS & CO. LLC': 'US',
    'OMERS ADMINISTRATION Corp': 'CA',
    'OMERS ADMINISTRATION CORP': 'CA',
    'NORTHERN TRUST CORP': 'US',
    'NORTHERN TRUST COMPANY': 'US',
    'MILLENNIUM MANAGEMENT LLC': 'US',
    'JANE STREET GROUP, LLC': 'US',
    'JANE STREET CAPITAL LLC': 'US',
    'JPMORGAN CHASE & CO': 'US',
    'MORGAN STANLEY': 'US',
    'CITADEL ADVISORS LLC': 'US',
    'CITADEL LLC': 'US',
    'POINT72 ASSET MANAGEMENT LP': 'US',
    'TWO SIGMA INVESTMENTS LP': 'US',
    'D. E. SHAW & CO. LP': 'US',
    'BRIDGEWATER ASSOCIATES LP': 'US',
    'RENAISSANCE TECHNOLOGIES LLC': 'US',
    'FIDELITY MANAGEMENT & RESEARCH': 'US',
    'DIMENSIONAL FUND ADVISORS LP': 'US',
    'TORONTO DOMINION BANK': 'CA',
    'BANK OF NEW YORK MELLON CORP': 'US',
    'BANK OF AMERICA CORP': 'US',
    'WELLS FARGO & COMPANY': 'US',
    'UBS GROUP AG': 'CH',
    'CREDIT SUISSE AG': 'CH',
    'BARCLAYS PLC': 'GB',
    'DEUTSCHE BANK AG': 'DE',
    'NOMURA SECURITIES': 'JP',
    'MITSUBISHI UFJ FINANCIAL GROUP': 'JP',
    'SOCIETE GENERALE': 'FR',
    'BNP PARIBAS': 'FR',
    'INVESCO LTD': 'US',
    'FRANKLIN RESOURCES INC': 'US',
    'T. ROWE PRICE GROUP INC': 'US',
    'CAPITAL RESEARCH & MANAGEMENT': 'US',
    'CAPITAL GROUP COMPANIES INC': 'US',
    'NUVEEN ASSET MANAGEMENT LLC': 'US',
    'LEGAL & GENERAL GROUP PLC': 'GB',
    'VANGUARD EXTENDED MARKET INDEX FUND': 'US',
    'Fidelity Extended Market Index Fund': 'US',
    'Fidelity Nasdaq Composite Index Fund': 'US',
    'Fidelity Total Market Index Fund': 'US',
    'Victory Extended Market Index Fund': 'US',
    # SEC 13F 新增机构
    'UBS Group AG': 'CH',
    'CITIGROUP INC': 'US',
    'BNP PARIBAS FINANCIAL MARKETS': 'FR',
    'ROYAL BANK OF CANADA': 'CA',
    'Squarepoint Ops LLC': 'US',
    'Qube Research & Technologies Ltd': 'GB',
    'Hudson Bay Capital Management LP': 'US',
    'Schonfeld Strategic Advisors LLC': 'US',
    'Lighthouse Investment Partners, LLC': 'US',
    'Scientech Research LLC': 'US',
    'Tower Research Capital LLC': 'US',
    'MANGROVE PARTNERS IM, LLC': 'US',
    'BALYASNY ASSET MANAGEMENT L.P.': 'US',
    'Global Retirement Partners, LLC': 'US',
    'IFP Advisors, Inc': 'US',
    'State of Wyoming': 'US',
    'JPMORGAN CHASE & CO': 'US',
    'MILLENNIUM MANAGEMENT LLC': 'US',
    'TWO SIGMA INVESTMENTS, LP': 'US',
    'RENAISSANCE TECHNOLOGIES LLC': 'US',
}


def _get_cn_name(name_en):
    """根据英文名查找中文译名"""
    if not name_en:
        return ''
    # 精确匹配
    if name_en in INST_CN_NAMES:
        return INST_CN_NAMES[name_en]
    # 模糊匹配（去掉标点差异）
    clean = name_en.upper().replace('.', '').replace(',', '').replace(';', '').strip()
    for k, v in INST_CN_NAMES.items():
        k_clean = k.upper().replace('.', '').replace(',', '').replace(';', '').strip()
        if k_clean == clean:
            return v
    # 部分匹配
    for k, v in INST_CN_NAMES.items():
        if clean in k.upper() or k.upper() in clean:
            return v
    return ''


def _get_country(name_en):
    """根据英文名推断所在国家"""
    if not name_en:
        return 'US'
    if name_en in INST_COUNTRIES:
        return INST_COUNTRIES[name_en]
    clean = name_en.upper().replace('.', '').replace(',', '').strip()
    for k, v in INST_COUNTRIES.items():
        k_clean = k.upper().replace('.', '').replace(',', '').strip()
        if k_clean == clean:
            return v
    # 关键词推断
    for keyword, country in [('BANK', 'US'), ('TRUST', 'US'), ('CAPITAL', 'US'),
                              ('MANAGEMENT', 'US'), ('INVESTMENT', 'US'),
                              ('SECURITIES', 'US'), ('PARTNERS', 'US'),
                              ('AG', 'CH'), ('PLC', 'GB'), ('LTD', 'GB'),
                              ('GMBH', 'DE'), ('SA', 'FR')]:
        if keyword in clean:
            return country
    return 'US'


def _fetch_finviz_institutional_holdings():
    """从 Finviz 抓取 XWIN 机构持股数据（JSON嵌入在HTML中）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        resp = requests.get('https://finviz.com/quote.ashx?t=XWIN', headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[机构持股] Finviz 返回 {resp.status_code}")
            return None

        # 提取机构持股 JSON 数据
        import json
        match = re.search(
            r'<script id="institutional-ownership-init-data-0"[^>]*>(.*?)</script>',
            resp.text, re.DOTALL
        )
        if not match:
            print("[机构持股] Finviz 页面中未找到机构持股数据")
            return None

        data = json.loads(match.group(1))
        managers = data.get('managersOwnership', [])
        funds = data.get('fundsOwnership', [])

        if not managers:
            print("[机构持股] Finviz managers 数据为空")
            return None

        # 从 Finviz 页面提取总股数和当前价格
        total_shares_str = ''
        price_str = ''
        shares_match = re.search(r'Shs Outstand.*?<b>([\d.]+M)', resp.text)
        if shares_match:
            total_shares_str = shares_match.group(1)
        price_match = re.search(r'quote_price.*?(\d+\.\d+)', resp.text)
        if not price_match:
            price_match = re.search(r'class="[^"]*"[^>]*>(\d+\.\d+)<', resp.text)

        # 计算总股数
        total_shares = 43070000  # 默认值（来自 Finviz 43.07M）
        if total_shares_str:
            try:
                if 'M' in total_shares_str:
                    total_shares = int(float(total_shares_str.replace('M', '')) * 1_000_000)
                elif 'B' in total_shares_str:
                    total_shares = int(float(total_shares_str.replace('B', '')) * 1_000_000_000)
            except:
                pass

        # 获取当前价格（多数据源尝试）
        current_price = 0
        try:
            stock_resp = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/XWIN?range=1d&interval=1d',
                                       headers=headers, timeout=10)
            if stock_resp.status_code == 200:
                price_data = stock_resp.json()
                current_price = price_data['chart']['result'][0]['meta']['regularMarketPrice']
        except:
            pass
        if current_price <= 0:
            try:
                # 备用：从自身API获取
                self_url = os.environ.get('VERCEL_URL', 'http://localhost:5173')
                local_resp = requests.get(f'{self_url}/api/stock?ticker=XWIN', timeout=5)
                if local_resp.status_code == 200:
                    local_data = local_resp.json()
                    # 自身API返回格式: {"data": {"close": 7.24, ...}}
                    if 'data' in local_data and local_data['data'].get('close'):
                        current_price = local_data['data']['close']
                    elif local_data.get('price'):
                        current_price = local_data['price']
            except:
                pass
        if current_price <= 0:
            # 最后备用：从 Finviz 页面提取价格
            try:
                price_match = re.search(r'quote_price_wrapper[^>]*>.*?<strong[^>]*>(\d+\.\d+)</strong>', resp.text)
                if price_match:
                    current_price = float(price_match.group(1))
            except:
                pass

        institutions = []
        for m in managers:
            name = m.get('name', '').strip()
            perc = m.get('percOwnership', 0)
            if not name or perc < 0.01:
                continue
            shares = int(total_shares * perc / 100)
            value = round(shares * current_price, 2) if current_price > 0 else 0

            institutions.append({
                'name': name,
                'name_cn': _get_cn_name(name),
                'shares': shares,
                'value': value,
                'country': _get_country(name),
                'pct_ownership': round(perc, 4),
            })

        # 按持股比例降序排列
        institutions.sort(key=lambda x: x.get('pct_ownership', 0), reverse=True)

        print(f"[机构持股] Finviz 成功获取 {len(institutions)} 家机构, 总股数={total_shares:,}, 价格={current_price}")
        return {
            'source': 'finviz',
            'available': True,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_shares_outstanding': total_shares,
            'current_price': current_price,
            'inst_ownership_pct': round(sum(m.get('percOwnership', 0) for m in managers), 2),
            'institutions': institutions,
        }

    except requests.exceptions.Timeout:
        print("[机构持股] Finviz 请求超时")
        return None
    except json.JSONDecodeError as e:
        print(f"[机构持股] Finviz JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"[机构持股] Finviz 抓取异常: {e}")
        return None


def _fetch_nasdaq_institutional_holdings(current_price=0, total_shares=43070000):
    """从 Nasdaq API 获取 XWIN 机构持股完整数据

    数据源: https://api.nasdaq.com/api/company/XWIN/institutional-holdings
    包含: Owner Name, Date, Shares Held, Change, Change %, Value 等
    注意: 使用 subprocess + curl 代替 requests，避免 Python SSL 超时问题
    """
    import json as _json, subprocess, shlex

    api_url = 'https://api.nasdaq.com/api/company/XWIN/institutional-holdings?limit=50&type=TOTAL&sortColumn=marketValue'

    try:
        # 使用 curl 获取数据（避免 Python requests 的 SSL 握手超时）
        curl_cmd = [
            'curl', '-s', '--max-time', '20',
            '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            '-H', 'Accept: application/json, text/plain, */*',
            '-H', 'Referer: https://www.nasdaq.com/',
            '-H', 'Origin: https://www.nasdaq.com',
            api_url
        ]
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=25)
        if result.returncode != 0:
            print(f"[机构持股] Nasdaq curl 失败: {result.stderr[:200]}")
            return None

        data = _json.loads(result.stdout)
        if data.get('status', {}).get('rCode') != 200:
            print(f"[机构持股] Nasdaq API 业务错误: {data.get('status')}")
            return None

        raw = data.get('data', {})
        ownership = raw.get('ownershipSummary', {})
        transactions = raw.get('holdingsTransactions', {})
        active_pos = raw.get('activePositions', {})
        new_sold = raw.get('newSoldOutPositions', {})

        table_data = transactions.get('table', {})
        rows = table_data.get('rows', [])
        if not rows:
            print("[机构持股] Nasdaq API 返回空表格")
            return None

        # 解析 ownership summary
        inst_ownership_pct = 0
        try:
            pct_str = ownership.get('SharesOutstandingPCT', {}).get('value', '0%')
            inst_ownership_pct = float(pct_str.replace('%', ''))
        except:
            pass

        shares_outstanding_m = 0
        try:
            shares_outstanding_m = float(ownership.get('ShareoutstandingTotal', {}).get('value', '0'))
        except:
            pass
        if shares_outstanding_m > 0:
            total_shares = int(shares_outstanding_m * 1_000_000)

        total_value_m = 0
        try:
            val_str = ownership.get('TotalHoldingsValue', {}).get('value', '$0')
            total_value_m = float(val_str.replace('$', '').replace(',', ''))
        except:
            pass

        institutions = []
        for row in rows:
            name = row.get('ownerName', '').strip()
            if not name:
                continue

            # 解析股数
            shares_str = row.get('sharesHeld', '0').replace(',', '')
            shares = int(shares_str) if shares_str.isdigit() else 0

            # 解析变化
            change_str = row.get('sharesChange', '0').replace(',', '')
            change = int(change_str) if change_str.lstrip('-').isdigit() else 0

            # 解析变化百分比
            change_pct_str = row.get('sharesChangePCT', '')
            change_pct = change_pct_str  # 保持原始格式 ("New", "Sold Out", "448.404%")

            # 解析市值（单位: 千美元）
            mv_str = row.get('marketValue', '').replace('$', '').replace(',', '')
            market_value_k = float(mv_str) if mv_str else 0  # 千美元

            # 日期
            date_str = row.get('date', '')

            # 计算占比
            pct = round(shares / total_shares * 100, 4) if total_shares > 0 and shares > 0 else 0

            # 计算实际市值（美元）
            value_usd = market_value_k * 1000 if market_value_k > 0 else (shares * current_price if shares > 0 else 0)

            institutions.append({
                'name': name,
                'name_cn': _get_cn_name(name),
                'shares': shares,
                'value': round(value_usd, 2),
                'country': _get_country(name),
                'pct_ownership': pct,
                # Nasdaq 专属字段
                'date': date_str,
                'change': change,
                'change_pct': change_pct,
                'market_value_k': round(market_value_k, 1),
                'nasdaq_url': 'https://www.nasdaq.com' + row.get('url', ''),
            })

        # 构建额外统计
        active_summary = {}
        for r in active_pos.get('rows', []):
            active_summary[r['positions']] = {
                'holders': int(r.get('holders', 0)),
                'shares': int(r.get('shares', '0').replace(',', '')),
            }
        new_sold_summary = {}
        for r in new_sold.get('rows', []):
            new_sold_summary[r['positions']] = {
                'holders': int(r.get('holders', 0)),
                'shares': int(r.get('shares', '0').replace(',', '')),
            }

        total_records = int(transactions.get('totalRecords', len(rows)))

        print(f"[机构持股] Nasdaq API 获取 {len(institutions)} 家机构, 机构占比={inst_ownership_pct}%, 价格={current_price}")
        return {
            'source': 'nasdaq',
            'available': True,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_shares_outstanding': total_shares,
            'current_price': current_price,
            'inst_ownership_pct': inst_ownership_pct,
            'total_value_millions': total_value_m,
            'total_records': total_records,
            'period': '2025-Q4 (13F)',
            'active_positions_summary': active_summary,
            'new_sold_summary': new_sold_summary,
            'institutions': institutions,
        }

    except requests.exceptions.Timeout:
        print("[机构持股] Nasdaq API 请求超时")
        return None
    except Exception as e:
        print(f"[机构持股] Nasdaq API 异常: {e}")
        return None


@app.route('/api/institutional_holdings', methods=['GET'])
def get_institutional_holdings():
    """获取 XWIN 全球机构持股数据

    数据源优先级:
    1. Nasdaq API（完整 35 家机构，含 Active Positions / Change / Value 等）
    2. Finviz（备用）
    """
    try:
        # 获取当前价格和总股数（用于计算市值）
        current_price = 0
        total_shares = 43070000  # 默认值

        # 获取价格
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        }
        try:
            stock_resp = requests.get(
                'https://query1.finance.yahoo.com/v8/finance/chart/XWIN?range=1d&interval=1d',
                headers=headers, timeout=10,
            )
            if stock_resp.status_code == 200:
                current_price = stock_resp.json()['chart']['result'][0]['meta']['regularMarketPrice']
        except:
            pass
        if current_price <= 0:
            try:
                self_url = os.environ.get('VERCEL_URL', 'http://localhost:5173')
                local_resp = requests.get(f'{self_url}/api/stock?ticker=XWIN', timeout=5)
                if local_resp.status_code == 200:
                    local_data = local_resp.json()
                    if 'data' in local_data and local_data['data'].get('close'):
                        current_price = local_data['data']['close']
            except:
                pass

        # 优先: Nasdaq API（完整 35 家机构数据）
        nasdaq_data = _fetch_nasdaq_institutional_holdings(current_price, total_shares)
        if nasdaq_data and nasdaq_data.get('institutions') and len(nasdaq_data['institutions']) >= 10:
            return jsonify(nasdaq_data)

        # 备用: Finviz
        finviz_data = _fetch_finviz_institutional_holdings()
        if finviz_data and finviz_data.get('institutions') and len(finviz_data['institutions']) > 0:
            return jsonify(finviz_data)

        # 两个源都不可用
        return jsonify({
            'source': 'nasdaq',
            'available': False,
            'message': 'Institutional Holdings data is currently not available.',
            'institutions': [],
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
    except Exception as e:
        print(f"[机构持股] 错误: {e}")
        return jsonify({
            'source': 'error',
            'available': False,
            'message': f'Failed to fetch: {str(e)}',
            'institutions': [],
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })


if __name__ == '__main__':
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print("🚀 深海有象后端代理服务启动中...")
    print("📡 监听地址: http://localhost:5173")
    print("📋 API端点:")
    print("   GET  /api/health      - 健康检查")
    print("   GET  /api/providers   - 查看可用 LLM 服务商")
    print("   GET  /api/news_search - 关键词新闻搜索")
    print("   GET  /api/news_radar  - 多关键词新闻雷达")
    print("   GET  /api/stock       - 美股实时行情 (XWIN等)")
    print("   GET  /api/announcements - 公司公告监控")
    print("   POST /api/fetch       - 抓取新闻链接")
    print("   POST /api/generate    - LLM 生成内容")
    print("   GET  /api/institutional_holdings - XWIN 机构持股")
    app.run(host='0.0.0.0', port=5173, debug=False)
