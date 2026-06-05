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
import sqlite3

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import json
import time
import threading
import hashlib
import zlib
import struct
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, quote, urlencode
from abc import ABC, abstractmethod
from typing import Optional

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'api', 'admin.db')
XMAX_TWEET_SYNC_INTERVAL_SECONDS = 4 * 60 * 60
_tweet_sync_started = False
_tweet_sync_lock = threading.Lock()
MUSK_NEWS_DB_CACHE_KEY = 'musk_news_domestic_v1'
_musk_news_refresh_inflight = False
_musk_news_refresh_lock = threading.Lock()

MUSK_CATALYST_CONFIG = {
    "updated": "2026-06-04T00:00:00Z",
    "base_valuations_usd": {
        "spacex": 1.75e12,
        "neuralink": 44.0e9,
        "boring": 5.675e9,
    },
    "base_total_override_usd": 1.800075e12,
    "catalyst_facts": [
        {
            "key": "spacex",
            "label_zh": "🚀【SpaceX】",
            "label_en": "🚀 [SpaceX]",
            "text_zh": "太空帝国核心资产。投行/市场口径估值区间 $1.75T–$2T，IPO 相关动态以 SEC/交易所披露为准。",
            "text_en": "Core space asset. Street valuation range $1.75T–$2T; IPO signals should be verified via SEC/exchange filings.",
        },
        {
            "key": "tesla",
            "label_zh": "🚗【Tesla】",
            "label_en": "🚗 [Tesla]",
            "text_zh": "智能驾驶与机器人平台。FSD / Robotaxi 进展对估值弹性最敏感，关键节点以财报与监管披露为准。",
            "text_en": "Autonomy & robotics platform. FSD/Robotaxi milestones drive valuation convexity; rely on filings/earnings for confirmation.",
        },
        {
            "key": "neuralink",
            "label_zh": "🧠【Neuralink】",
            "label_en": "🧠 [Neuralink]",
            "text_zh": "脑机接口赛道。最新公开口径估值 $44B；临床进展以官方披露与监管文件为准。",
            "text_en": "BCI category. Latest public valuation $44B; clinical updates should follow official/regulatory disclosures.",
        },
        {
            "key": "boring",
            "label_zh": "🕳️【The Boring Company】",
            "label_en": "🕳️ [The Boring Company]",
            "text_zh": "地下隧道基础设施。最新公开口径估值 $5.675B；项目与融资以公司/政府公开资料为准。",
            "text_en": "Tunneling infrastructure. Latest public valuation $5.675B; follow company/public records for financing & project updates.",
        },
    ],
    "capital_structure": {
        "locked_pct": 100,
        "free_float_ratio": 0.075,
        "voting_control_ratio": 0.842,
        "locked_label_zh": "100% Locked",
        "locked_label_en": "100% Locked",
        "structure_text_zh": "Free Float Ratio: ~7.5% (Ultra-Tight Supply) | ABC股权绝对控盘: 84.2%",
        "structure_text_en": "Free Float Ratio: ~7.5% (Ultra-Tight Supply) | ABC voting control: 84.2%",
    },
}


@app.after_request
def _no_cache_kline(resp):
    try:
        if request.path.startswith('/api/stock/kline'):
            resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
    except Exception:
        pass
    return resp


_seo_img_cache = {"og": {"ts": 0, "bytes": b""}, "favicon": {"ts": 0, "bytes": b""}}


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _png_from_rgb(w: int, h: int, pixel_fn) -> bytes:
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            r, g, b = pixel_fn(x, y)
            raw.extend(bytes((int(r) & 255, int(g) & 255, int(b) & 255)))
    comp = zlib.compress(bytes(raw), level=9)
    ihdr = struct.pack(">IIBBBBB", int(w), int(h), 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", comp) + _png_chunk(b"IEND", b"")


def _seo_og_png_bytes() -> bytes:
    now = time.time()
    entry = _seo_img_cache.get("og") or {}
    if entry.get("bytes") and (now - float(entry.get("ts") or 0)) < 24 * 60 * 60:
        return entry["bytes"]

    bg0 = (4, 6, 10)
    bg1 = (11, 14, 20)
    cyan = (6, 182, 212)
    neon = (0, 230, 118)
    w, h = 1200, 630

    def lerp(a, b, t):
        return int(a + (b - a) * t)

    def clamp01(t):
        return 0.0 if t < 0 else (1.0 if t > 1 else t)

    def pix(x, y):
        ty = y / (h - 1)
        r = lerp(bg0[0], bg1[0], ty)
        g = lerp(bg0[1], bg1[1], ty)
        b = lerp(bg0[2], bg1[2], ty)

        cx, cy = w * 0.55, h * 0.45
        dx = (x - cx) / w
        dy = (y - cy) / h
        vv = clamp01((dx * dx + dy * dy) * 3.2)
        r = lerp(r, 0, vv * 0.35)
        g = lerp(g, 0, vv * 0.35)
        b = lerp(b, 0, vv * 0.35)

        ox, oy = w * 0.78, h * 0.28
        dd = ((x - ox) ** 2 + (y - oy) ** 2) ** 0.5
        t = clamp01(1 - dd / (w * 0.22))
        r = lerp(r, cyan[0], t * 0.18)
        g = lerp(g, cyan[1], t * 0.18)
        b = lerp(b, cyan[2], t * 0.18)

        ox2, oy2 = w * 0.20, h * 0.72
        dd2 = ((x - ox2) ** 2 + (y - oy2) ** 2) ** 0.5
        t2 = clamp01(1 - dd2 / (w * 0.24))
        r = lerp(r, neon[0], t2 * 0.14)
        g = lerp(g, neon[1], t2 * 0.14)
        b = lerp(b, neon[2], t2 * 0.14)

        return r, g, b

    bts = _png_from_rgb(w, h, pix)
    _seo_img_cache["og"] = {"ts": now, "bytes": bts}
    return bts


def _seo_favicon_png_bytes() -> bytes:
    now = time.time()
    entry = _seo_img_cache.get("favicon") or {}
    if entry.get("bytes") and (now - float(entry.get("ts") or 0)) < 24 * 60 * 60:
        return entry["bytes"]

    bg1 = (11, 14, 20)
    cyan = (6, 182, 212)
    w, h = 96, 96

    def lerp(a, b, t):
        return int(a + (b - a) * t)

    def clamp01(t):
        return 0.0 if t < 0 else (1.0 if t > 1 else t)

    def pix(x, y):
        r, g, b = bg1
        cx, cy = w / 2, h / 2
        dd = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        t = clamp01(1 - dd / (w * 0.55))
        r = lerp(r, cyan[0], t * 0.55)
        g = lerp(g, cyan[1], t * 0.55)
        b = lerp(b, cyan[2], t * 0.55)
        ring = abs(dd - w * 0.32)
        rt = clamp01(1 - ring / (w * 0.03))
        r = lerp(r, 255, rt * 0.10)
        g = lerp(g, 255, rt * 0.10)
        b = lerp(b, 255, rt * 0.10)
        return r, g, b

    bts = _png_from_rgb(w, h, pix)
    _seo_img_cache["favicon"] = {"ts": now, "bytes": bts}
    return bts


@app.route("/api/og.png", methods=["GET"])
def seo_og_png():
    bts = _seo_og_png_bytes()
    resp = Response(bts, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return resp


@app.route("/api/favicon.png", methods=["GET"])
def seo_favicon_png():
    bts = _seo_favicon_png_bytes()
    resp = Response(bts, mimetype="image/png")
    resp.headers["Cache-Control"] = "public, max-age=86400, immutable"
    return resp


@app.route('/api/finnhub/candles', methods=['GET'])
def finnhub_candles():
    symbol = (request.args.get('symbol') or request.args.get('ticker') or 'XMAX').upper()
    resolution = (request.args.get('resolution') or 'D').strip()
    _from = int(request.args.get('from') or '0')
    _to = int(request.args.get('to') or '0')

    token = (os.environ.get('FINNHUB_API_KEY') or '').strip()
    if not token:
        return jsonify({"success": False, "error": "FINNHUB_API_KEY 未配置"}), 500

    if not symbol or not resolution or _from <= 0 or _to <= 0 or _to <= _from:
        return jsonify({"success": False, "error": "参数错误"}), 400

    url = "https://finnhub.io/api/v1/stock/candle"
    try:
        r = requests.get(url, params={
            "symbol": symbol,
            "resolution": resolution,
            "from": _from,
            "to": _to,
            "token": token,
        }, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        j = r.json()
        if j.get('s') != 'ok':
            return jsonify({"success": False, "error": j.get('s') or "Finnhub 返回异常", "raw": j}), 503

        t_arr = j.get('t') or []
        o_arr = j.get('o') or []
        h_arr = j.get('h') or []
        l_arr = j.get('l') or []
        c_arr = j.get('c') or []
        v_arr = j.get('v') or []

        n = min(len(t_arr), len(o_arr), len(h_arr), len(l_arr), len(c_arr), len(v_arr))
        data = []
        for i in range(n):
            data.append({
                "time": int(t_arr[i]),
                "open": float(o_arr[i]),
                "high": float(h_arr[i]),
                "low": float(l_arr[i]),
                "close": float(c_arr[i]),
                "volume": float(v_arr[i]),
            })

        resp = jsonify({"success": True, "data": data, "source": "finnhub"})
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 503

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
_musk_news_runtime_cache = {"articles": [], "timestamp": 0.0}
_musk_news_runtime_cache_lock = threading.Lock()
MUSK_NEWS_RUNTIME_CACHE_TTL = 300


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
#  GlobeNewsWire 新闻源
# ============================================================

def search_globenewswire(keyword, count=10):
    """从 GlobeNewsWire 搜索新闻稿（公司公告、财报等官方新闻）"""
    articles = []
    try:
        url = f"https://www.globenewswire.com/en/search/keyword/{quote(keyword)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        resp = requests.get(url, headers=headers, timeout=15, verify=False,
                             proxies=_get_proxies())
        if resp.status_code != 200:
            return articles

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'lxml')

        # GlobeNewsWire 的文章链接包含 /news-release/ 路径
        links = soup.select('a[href*="/news-release/"]')
        for link in links[:count]:
            try:
                title = link.get_text(strip=True)
                if not title or len(title) < 10:
                    continue
                href = link.get('href', '')
                if not href.startswith('http'):
                    href = 'https://www.globenewswire.com' + href

                # 从 URL 提取日期: /news-release/YYYY/MM/DD/...
                date_str = ''
                date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
                if date_match:
                    date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"

                # 尝试从附近的 date-source 元素获取更精确的日期
                parent = link.parent
                if parent:
                    date_el = parent.find_previous('div', class_='date-source')
                    if date_el:
                        date_str = date_el.get_text(strip=True).split('|')[0].strip()

                # 提取来源
                source = 'GlobeNewsWire'
                if parent:
                    src_el = parent.find_previous('div', class_='date-source')
                    if src_el:
                        src_text = src_el.get_text(strip=True)
                        if '|' in src_text:
                            src_part = src_text.split('|')[-1].strip()
                            if src_part.startswith('Source:'):
                                source = src_part.replace('Source:', '').strip()

                articles.append({
                    'title': title,
                    'url': href,
                    'source': source,
                    'time': date_str,
                    'summary': '',
                    'is_foreign': True,
                    'title_en': title,
                    'title_cn': '',
                })
            except Exception:
                continue

        print(f"[GlobeNewsWire] keyword={keyword}, found={len(articles)}")
    except Exception as e:
        print(f"[GlobeNewsWire] Error: {e}")
    return articles


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



XMAX_NEWS_CONTEXT_KEYWORDS = [
    'stock', 'stocks', 'share', 'shares', 'nasdaq', 'investor', 'investors',
    'offering', 'offer', 'filing', 'market', 'markets', 'security', 'securities',
    'inc', 'inc.', 'announcement', 'press release', '股票', '股价', '纳斯达克',
    '投资者', '公告', '新闻稿', '募资', '融资', '发行', '股份'
]


def is_xmax_relevant_article(article):
    title = str((article or {}).get('title') or '').strip().lower()
    summary = str((article or {}).get('summary') or '').strip().lower()
    source = str((article or {}).get('source') or '').strip().lower()
    text = ' '.join([title, summary, source]).strip()
    if not text:
        return False
    if 'xmax inc' in text or 'xmax inc.' in text or '纳斯达克股票代码：xmax' in text:
        return True
    if not re.search(r'\bxmax\b', text, re.I):
        return False
    return any(keyword in text for keyword in XMAX_NEWS_CONTEXT_KEYWORDS)


def collect_xmax_relevant_news(include_foreign=True, limit=12):
    articles = []
    try:
        rss_items = _fetch_rss_items(
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XMAX",
            limit=max(12, limit),
            ua="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        for item in rss_items:
            item["source"] = "Yahoo Finance"
            item["time_parsed"] = int(item.get("time_ts") or 0)
            if is_xmax_relevant_article(item):
                articles.append(item)
    except Exception as e:
        print(f"[collect_xmax_relevant_news] yahoo rss error: {e}")

    try:
        gnw_articles = search_globenewswire("XMax Inc", max(12, limit))
        for item in gnw_articles:
            if 'time_parsed' not in item:
                item['time_parsed'] = parse_news_time(item.get('time', ''))
            if is_xmax_relevant_article(item):
                articles.append(item)
    except Exception as e:
        print(f"[collect_xmax_relevant_news] globenewswire error: {e}")

    if len(deduplicate_articles(articles)) < limit:
        try:
            agg_articles = search_news_aggregated("XMAX Inc stock NASDAQ", max(18, limit), include_foreign=include_foreign)
            for item in agg_articles:
                if 'time_parsed' not in item:
                    item['time_parsed'] = parse_news_time(item.get('time', ''))
                if is_xmax_relevant_article(item):
                    articles.append(item)
        except Exception as e:
            print(f"[collect_xmax_relevant_news] aggregated error: {e}")

    unique = deduplicate_articles(articles)
    for item in unique:
        if 'time_parsed' not in item:
            item['time_parsed'] = parse_news_time(item.get('time', ''))
    unique = [item for item in unique if is_xmax_relevant_article(item)]
    unique.sort(key=lambda x: -x.get('time_parsed', 0))
    return unique[:limit]


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

    # 方案1: translate.googleapis.com（免费，无需 key）
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        resp = requests.get(url, params={
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN",
            "dt": "t",
            "q": title_en,
        }, timeout=6, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            pieces = []
            for seg in (data[0] if isinstance(data, list) and data else []):
                if isinstance(seg, list) and seg and isinstance(seg[0], str):
                    pieces.append(seg[0])
            translated = ''.join(pieces).strip()
    except:
        pass

    # 方案2: 使用 MyMemory 翻译 API（免费，无需 key）
    try:
        if not translated:
            url = f"https://api.mymemory.translated.net/get?q={quote(title_en)}&langpair=en|zh-CN"
            resp = requests.get(url, timeout=5, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get('responseData', {}).get('translatedText', '')
                if translated and translated.upper() == translated and len(translated) > 20:
                    translated = ''
    except:
        pass

    # 方案3: 使用 LibreTranslate（免费开源）
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
    candidates = []
    for a in articles:
        try:
            if a.get('title_cn'):
                continue
            src = (a.get('title_en') or a.get('title') or '').strip()
            if not src:
                continue
            candidates.append((a, src))
        except Exception:
            continue

    if not candidates:
        return articles

    def _translate(a, src):
        cn = translate_title_to_chinese(src)
        if cn:
            a['title_cn'] = cn
            if not a.get('title_en'):
                a['title_en'] = src

    threads = []
    for a, src in candidates:
        t = threading.Thread(target=_translate, args=(a, src))
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=8)

    return articles


_text_translate_cache = {}
_text_translate_cache_lock = threading.Lock()
_article_translate_cache = {}
_article_translate_cache_lock = threading.Lock()


def _get_request_lang():
    lang = (request.args.get('lang') or '').strip().lower()
    if lang in ('zh', 'en'):
        return lang
    accept_lang = (request.headers.get('Accept-Language') or '').strip().lower()
    if accept_lang.startswith('zh'):
        return 'zh'
    if accept_lang.startswith('en'):
        return 'en'
    return 'zh'


def _get_translate_flag(lang: str):
    raw = (request.args.get('translate') or '').strip()
    if raw in ('0', 'false', 'False'):
        return False
    if raw in ('1', 'true', 'True'):
        return True
    return lang == 'zh'


def _get_llm_config_from_request_or_store():
    provider = (request.args.get('provider') or request.args.get('provider_name') or '').strip()
    api_key = (request.args.get('api_key') or '').strip()
    model = (request.args.get('model') or '').strip()
    if provider and api_key:
        return {"provider": provider, "api_key": api_key, "model": model or None}
    try:
        store = _llm_config_store if isinstance(_llm_config_store, dict) else {}
    except Exception:
        store = {}
    provider = (store.get('provider') or store.get('provider_name') or '').strip()
    api_key = (store.get('api_key') or '').strip()
    model = (store.get('model') or '').strip()
    if provider and api_key:
        return {"provider": provider, "api_key": api_key, "model": model or None}
    return None


def _split_text_chunks(text: str, max_len: int = 900):
    s = (text or '').strip()
    if not s:
        return []
    parts = re.split(r'\n{2,}', s)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_len:
            chunks.append(p)
            continue
        sentences = re.split(r'(?<=[\.\!\?\u3002\uff01\uff1f])\s+', p)
        buf = ''
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if not buf:
                buf = sent
                continue
            if len(buf) + 1 + len(sent) <= max_len:
                buf = buf + ' ' + sent
            else:
                chunks.append(buf)
                buf = sent
        if buf:
            chunks.append(buf)
    return chunks


def _translate_chunk_en_to_zh(chunk: str):
    if not chunk or len(chunk) < 5:
        return ''
    key = hashlib.md5(chunk.strip().encode('utf-8')).hexdigest()
    with _text_translate_cache_lock:
        cached = _text_translate_cache.get(key)
        if cached is not None:
            return cached

    translated = ''

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        resp = requests.get(url, params={
            "client": "gtx",
            "sl": "en",
            "tl": "zh-CN",
            "dt": "t",
            "q": chunk,
        }, timeout=8, verify=False)
        if resp.status_code == 200:
            data = resp.json()
            pieces = []
            for seg in (data[0] if isinstance(data, list) and data else []):
                if isinstance(seg, list) and seg and isinstance(seg[0], str):
                    pieces.append(seg[0])
            translated = ''.join(pieces).strip()
    except Exception:
        pass

    try:
        if not translated:
            resp = requests.post(
                "https://libretranslate.de/translate",
                json={'q': chunk, 'source': 'en', 'target': 'zh', 'format': 'text'},
                timeout=8,
                verify=False,
            )
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get('translatedText', '') or ''
    except Exception:
        pass

    if not translated:
        try:
            url = f"https://api.mymemory.translated.net/get?q={quote(chunk)}&langpair=en|zh-CN"
            resp = requests.get(url, timeout=8, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get('responseData', {}).get('translatedText', '') or ''
        except Exception:
            pass

    with _text_translate_cache_lock:
        _text_translate_cache[key] = translated

    return translated


def translate_text_en_to_zh(text: str):
    s = (text or '').strip()
    if not s:
        return ''
    if not is_english_content(s[:1200]):
        return ''
    chunks = _split_text_chunks(s, max_len=900)
    if not chunks:
        return ''
    out = []
    for ch in chunks:
        tr = _translate_chunk_en_to_zh(ch)
        out.append(tr if tr else ch)
    return '\n\n'.join(out).strip()


def translate_articles_to_lang(articles, lang: str, translate: bool, llm_cfg=None):
    if not articles:
        return articles
    if not translate or lang != 'zh':
        for a in articles:
            if not a.get('title_en') and a.get('title'):
                a['title_en'] = a.get('title', '')
        return articles

    llm_cfg = llm_cfg or _get_llm_config_from_request_or_store()

    def _cache_key(a):
        url = (a.get('url') or '').strip()
        base = (a.get('title_en') or a.get('title') or '') + '|' + (a.get('summary') or '')
        ft = a.get('fullText') or ''
        sig = base + '|' + ft[:2000]
        return hashlib.md5((url + '|' + sig).encode('utf-8')).hexdigest()

    for a in articles:
        title_src = (a.get('title_en') or a.get('title') or '').strip()
        if not a.get('title_en') and title_src:
            a['title_en'] = title_src
        if 'summary_en' not in a:
            a['summary_en'] = a.get('summary', '')
        if 'fullText_en' not in a:
            a['fullText_en'] = a.get('fullText', '')

        combined = f"{a.get('title_en','')} {a.get('summary_en','')} {(a.get('fullText_en','') or '')[:600]}"
        if not is_english_content(combined):
            if not a.get('title_cn') and a.get('title'):
                a['title_cn'] = a.get('title')
            if 'summary_cn' not in a and a.get('summary'):
                a['summary_cn'] = a.get('summary')
            if 'fullText_cn' not in a and a.get('fullText'):
                a['fullText_cn'] = a.get('fullText')
            a['title'] = a.get('title_cn') or a.get('title') or a.get('title_en') or ''
            if a.get('summary_cn'):
                a['summary'] = a.get('summary_cn')
            if a.get('fullText_cn'):
                a['fullText'] = a.get('fullText_cn')
            continue

        ck = _cache_key(a)
        now_ts = time.time()
        with _article_translate_cache_lock:
            hit = _article_translate_cache.get(ck)
            if hit and (now_ts - hit.get('ts', 0)) < 86400:
                a['title_cn'] = hit.get('title_cn', '') or a.get('title_cn', '')
                a['summary_cn'] = hit.get('summary_cn', '') or a.get('summary_cn', '')
                a['fullText_cn'] = hit.get('fullText_cn', '') or a.get('fullText_cn', '')
                a['title'] = a.get('title_cn') or a.get('title_en') or ''
                if a.get('summary_cn'):
                    a['summary'] = a.get('summary_cn')
                if a.get('fullText_cn'):
                    a['fullText'] = a.get('fullText_cn')
                continue

        translated_title = a.get('title_cn') or translate_title_to_chinese(title_src) or ''
        if not translated_title:
            translated_title = translate_text_en_to_zh(title_src) or ''
        translated_summary = a.get('summary_cn') or translate_text_en_to_zh(a.get('summary_en') or '') or ''
        translated_full = a.get('fullText_cn') or translate_text_en_to_zh(a.get('fullText_en') or '') or ''

        if llm_cfg and llm_cfg.get('provider') and llm_cfg.get('api_key'):
            try:
                base_article = {
                    'title': title_src,
                    'summary': a.get('summary_en') or '',
                    'fullText': a.get('fullText_en') or '',
                }
                llm_out = translate_article_with_llm(
                    base_article,
                    llm_cfg['provider'],
                    llm_cfg['api_key'],
                    llm_cfg.get('model'),
                )
                if llm_out and llm_out.get('is_translated'):
                    translated_title = llm_out.get('title') or translated_title
                    translated_summary = llm_out.get('summary') or translated_summary
                    translated_full = llm_out.get('fullText') or translated_full
            except Exception:
                pass

        if translated_title:
            a['title_cn'] = translated_title
        if translated_summary:
            a['summary_cn'] = translated_summary
        if translated_full:
            a['fullText_cn'] = translated_full

        a['title'] = a.get('title_cn') or a.get('title_en') or ''
        if a.get('summary_cn'):
            a['summary'] = a.get('summary_cn')
        if a.get('fullText_cn'):
            a['fullText'] = a.get('fullText_cn')

        with _article_translate_cache_lock:
            _article_translate_cache[ck] = {
                "ts": now_ts,
                "title_cn": a.get('title_cn', ''),
                "summary_cn": a.get('summary_cn', ''),
                "fullText_cn": a.get('fullText_cn', ''),
            }

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


DOMESTIC_MEDIA_SOURCE_KEYWORDS = [
    '东方财富', '财联社', '新浪', '新浪财经', '腾讯', '腾讯新闻', '36氪', '虎嗅',
    '澎湃', '界面', '第一财经', '财新', '网易', '搜狐', '凤凰', '证券时报',
    '中国证券报', '上海证券报', '华尔街见闻', '智东西', '钛媒体', '快科技', 'cnBeta',
    '同花顺', '金融界', '观察者网', '经济观察报', '财经网', '和讯网', 'IT之家', '极客公园'
]

DOMESTIC_MEDIA_DOMAINS = [
    'eastmoney.com', 'cls.cn', 'sina.com.cn', 'finance.sina.com.cn', 'qq.com',
    'news.qq.com', '36kr.com', 'huxiu.com', 'thepaper.cn', 'jiemian.com',
    'yicai.com', 'caixin.com', '163.com', 'sohu.com', 'ifeng.com', 'stcn.com',
    'cs.com.cn', 'cnstock.com', 'wallstreetcn.com', 'zhidx.com', 'tmtpost.com',
    'cnbeta.com', '10jqka.com.cn', 'jrj.com.cn', 'guancha.cn', 'eeo.com.cn',
    'caijing.com.cn', 'hexun.com', 'ithome.com', 'geekpark.net'
]

MUSK_TOPIC_KEYWORDS = [
    '马斯克', 'spacex', '特斯拉', 'tesla', 'xai', 'grok', '星链', 'starlink',
    '星舰', 'starship', 'neuralink', '脑机接口', 'robotaxi', 'optimus', '擎天柱'
]

MUSK_BLOCKED_SOURCE_KEYWORDS = ['汽车之家', 'autohome']
MUSK_BLOCKED_URL_KEYWORDS = ['autohome.com', 'chejiahao.autohome.com']
MUSK_SOCIAL_URL_KEYWORDS = [
    'x.com/', 'twitter.com/', 't.co/', 'mobile.twitter.com/',
    'publish.x.com/', 'platform.x.com/'
]
MUSK_NEWS_RSS_URLS = [
    "https://news.google.com/rss/search?q=Elon+Musk+OR+SpaceX+OR+Tesla+OR+xAI+OR+Neuralink&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=%E9%A9%AC%E6%96%AF%E5%85%8B+OR+SpaceX+OR+%E7%89%B9%E6%96%AF%E6%8B%89+OR+xAI+OR+Neuralink&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
]
MUSK_NEWS_SUPPLEMENTAL_QUERIES = [
    "Elon Musk Tesla SpaceX xAI Neuralink",
    "马斯克 特斯拉 SpaceX xAI Neuralink",
    "SpaceX Starlink Starship Tesla Robotaxi Grok",
]

FOREIGN_NEWS_DOMAINS = [
    'reuters.com', 'bloomberg.com', 'cnbc.com', 'yahoo.com', 'finance.yahoo.com',
    'marketwatch.com', 'investing.com', 'ft.com', 'wsj.com', 'nytimes.com',
    'forbes.com', 'cnn.com', 'bbc.com', 'apnews.com', 'techcrunch.com',
    'theverge.com', 'businessinsider.com', 'seekingalpha.com', 'benzinga.com',
    'teslarati.com', 'electrek.co', 'foxbusiness.com'
]


def is_domestic_media_article(article):
    source = str((article or {}).get('source') or '').strip().lower()
    raw_url = str((article or {}).get('url') or '').strip()
    if not raw_url:
        return False
    try:
        host = (urlparse(raw_url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
    except Exception:
        host = ""
    if host:
        return any(host == d or host.endswith("." + d) for d in DOMESTIC_MEDIA_DOMAINS)
    return any(k.lower() in source for k in DOMESTIC_MEDIA_SOURCE_KEYWORDS)


def is_probably_foreign_news_domain(url):
    url = str(url or '').strip().lower()
    if not url:
        return False
    return any(domain in url for domain in FOREIGN_NEWS_DOMAINS)


def is_musk_topic_article(article):
    haystack = ' '.join([
        str((article or {}).get('title') or ''),
        str((article or {}).get('summary') or ''),
        str((article or {}).get('source') or ''),
    ]).lower()
    return any(keyword in haystack for keyword in MUSK_TOPIC_KEYWORDS)


def is_blocked_musk_article(article):
    source = str((article or {}).get('source') or '').strip().lower()
    url = str((article or {}).get('url') or '').strip().lower()
    title = str((article or {}).get('title') or '').strip().lower()
    if any(k in source for k in MUSK_BLOCKED_SOURCE_KEYWORDS):
        return True
    if any(k in url for k in MUSK_BLOCKED_URL_KEYWORDS):
        return True
    if '汽车之家' in title:
        return True
    return False


def is_real_news_article(article):
    url = str((article or {}).get('url') or '').strip().lower()
    title = str((article or {}).get('title') or '').strip()
    source = str((article or {}).get('source') or '').strip()
    if not title or not url:
        return False
    if any(k in url for k in MUSK_SOCIAL_URL_KEYWORDS):
        return False
    if url.endswith('/status') or '/status/' in url:
        return False
    if source.strip().lower() in ('@xmaxglobal', 'xmaxglobal', 'x', 'twitter'):
        return False
    return True


def _get_cached_musk_news_articles(limit=24, allow_stale=False):
    with _musk_news_runtime_cache_lock:
        articles = list(_musk_news_runtime_cache.get("articles") or [])
        timestamp = float(_musk_news_runtime_cache.get("timestamp") or 0.0)
    if not articles:
        return []
    if not allow_stale and (time.time() - timestamp) >= MUSK_NEWS_RUNTIME_CACHE_TTL:
        return []
    return articles[:max(1, int(limit or 24))]


def _set_cached_musk_news_articles(articles):
    clean = [dict(item) for item in (articles or []) if isinstance(item, dict)]
    with _musk_news_runtime_cache_lock:
        _musk_news_runtime_cache["articles"] = clean
        _musk_news_runtime_cache["timestamp"] = time.time()


def _normalize_musk_news_article(article, default_source=''):
    item = dict(article or {})
    normalized = {
        "title": str(item.get("title") or "").strip(),
        "url": str(item.get("url") or "").strip(),
        "time": str(item.get("time") or "").strip(),
        "summary": str(item.get("summary") or "").strip(),
        "source": str(item.get("source") or default_source or "Musk News").strip(),
    }
    if not normalized["title"] or not normalized["url"]:
        return None
    if is_blocked_musk_article(normalized):
        return None
    if not is_real_news_article(normalized):
        return None
    if not is_musk_topic_article(normalized):
        return None
    if not is_domestic_media_article(normalized):
        return None
    normalized["time_parsed"] = int(
        item.get("time_parsed")
        or item.get("time_ts")
        or parse_news_time(normalized.get("time", ""))
        or 0
    )
    return normalized


def collect_musk_domestic_news(target_count=24, min_count=20):
    primary_keywords = [
        "马斯克", "SpaceX", "特斯拉", "xAI", "星链", "星舰", "Neuralink", "脑机接口",
        "Robotaxi", "Optimus", "Grok", "擎天柱", "FSD", "人形机器人"
    ]
    query_batches = [
        "马斯克 SpaceX 特斯拉 xAI 星链 星舰 脑机接口 Grok",
        "马斯克 特斯拉 Robotaxi Optimus 擎天柱 FSD",
        "SpaceX 星链 星舰 发射 融资 上市 马斯克",
        "xAI Grok 算力 数据中心 马斯克",
        "Neuralink 脑机接口 马斯克",
        "特斯拉 自动驾驶 Robotaxi 马斯克",
        "马斯克 特斯拉 人形机器人 Optimus 最新消息",
        "SpaceX 星舰 星链 发射 商业航天 马斯克",
        "xAI Grok 数据中心 算力 融资 马斯克",
        "Neuralink 脑机接口 临床试验 马斯克",
        "马斯克 虎嗅 36氪 钛媒体",
        "SpaceX 星舰 虎嗅 36氪",
        "特斯拉 FSD Robotaxi 虎嗅 36氪 钛媒体",
    ]
    merged = []

    def push_articles(items):
        for article in (items or []):
            if is_blocked_musk_article(article):
                continue
            if not is_real_news_article(article):
                continue
            if not is_musk_topic_article(article):
                continue
            if not is_domestic_media_article(article):
                continue
            merged.append(article)

    push_articles(search_news_multi_keywords(primary_keywords, count_per_keyword=30, include_foreign=False))

    for query in query_batches:
        if len(deduplicate_articles(merged)) >= target_count:
            break
        push_articles(search_news_aggregated(query, 160, include_foreign=False))

    unique = deduplicate_articles(merged)
    for article in unique:
        if 'time_parsed' not in article:
            article['time_parsed'] = parse_news_time(article.get('time', ''))
    unique.sort(key=lambda x: -x.get('time_parsed', 0))

    if len(unique) < min_count:
        broad_queries = [
            "马斯克 最新消息",
            "特斯拉 最新消息",
            "SpaceX 最新消息",
            "xAI 最新消息",
            "Grok 最新消息",
            "Optimus 最新消息",
            "星链 最新消息",
            "星舰 最新消息",
            "Robotaxi 最新消息",
            "Neuralink 最新消息",
        ]
        for query in broad_queries:
            if len(unique) >= min_count:
                break
            push_articles(search_news_aggregated(query, 120, include_foreign=False))
            unique = deduplicate_articles(merged)
            for article in unique:
                if 'time_parsed' not in article:
                    article['time_parsed'] = parse_news_time(article.get('time', ''))
            unique.sort(key=lambda x: -x.get('time_parsed', 0))

    unique = [a for a in unique if is_real_news_article(a) and is_domestic_media_article(a)]
    return unique[:target_count]


def collect_musk_news_articles(target_count=24, min_count=20, include_foreign=True):
    cached = _get_cached_musk_news_articles(limit=target_count)
    if cached:
        return cached

    target_count = max(1, int(target_count or 24))
    min_count = max(1, int(min_count or target_count))
    db_cached = []
    try:
        entry = _app_cache_get(MUSK_NEWS_DB_CACHE_KEY)
        if entry and entry.get("value"):
            parsed = json.loads(entry.get("value") or "[]")
            if isinstance(parsed, list):
                for it in parsed:
                    n = _normalize_musk_news_article(it, default_source=str((it or {}).get("source") or "国内媒体"))
                    if n:
                        db_cached.append(n)
        db_cached = deduplicate_articles(db_cached)
        db_cached.sort(key=lambda x: -int(x.get("time_parsed") or 0))
        db_cached = db_cached[:target_count]
    except Exception:
        db_cached = []

    if db_cached:
        _set_cached_musk_news_articles(db_cached)
        _refresh_musk_news_in_background()
        return db_cached

    merged = []
    def push_items(items, default_source=''):
        for article in (items or []):
            normalized = _normalize_musk_news_article(article, default_source=default_source)
            if normalized:
                merged.append(normalized)

    try:
        push_items(collect_musk_domestic_news(target_count=max(target_count * 2, 48), min_count=min_count), default_source="国内媒体")
    except Exception as e:
        print(f"[collect_musk_news_articles] domestic error: {e}")

    unique = deduplicate_articles(merged)
    unique.sort(key=lambda x: -int(x.get("time_parsed") or 0))

    out = unique[:target_count]
    if out:
        _set_cached_musk_news_articles(out)
        try:
            _app_cache_set(MUSK_NEWS_DB_CACHE_KEY, json.dumps(out, ensure_ascii=False))
        except Exception:
            pass
        return out

    return _get_cached_musk_news_articles(limit=target_count, allow_stale=True)


def _refresh_musk_news_in_background():
    global _musk_news_refresh_inflight
    with _musk_news_refresh_lock:
        if _musk_news_refresh_inflight:
            return
        _musk_news_refresh_inflight = True

    def worker():
        global _musk_news_refresh_inflight
        try:
            items = []
            try:
                items = collect_musk_domestic_news(target_count=48, min_count=20)
            except Exception:
                items = []
            normalized = []
            for it in items:
                n = _normalize_musk_news_article(it, default_source=str((it or {}).get("source") or "国内媒体"))
                if n:
                    normalized.append(n)
            normalized = deduplicate_articles(normalized)
            normalized.sort(key=lambda x: -int(x.get("time_parsed") or 0))
            normalized = normalized[:24]
            if normalized:
                _set_cached_musk_news_articles(normalized)
                try:
                    _app_cache_set(MUSK_NEWS_DB_CACHE_KEY, json.dumps(normalized, ensure_ascii=False))
                except Exception:
                    pass
        finally:
            with _musk_news_refresh_lock:
                _musk_news_refresh_inflight = False

    threading.Thread(target=worker, name='musk-news-refresh', daemon=True).start()


def _load_musk_news_from_db(limit=24):
    out = []
    try:
        entry = _app_cache_get(MUSK_NEWS_DB_CACHE_KEY)
        if entry and entry.get("value"):
            parsed = json.loads(entry.get("value") or "[]")
            if isinstance(parsed, list):
                for it in parsed:
                    n = _normalize_musk_news_article(it, default_source=str((it or {}).get("source") or "国内媒体"))
                    if n:
                        out.append(n)
        out = deduplicate_articles(out)
        out.sort(key=lambda x: -int(x.get("time_parsed") or 0))
        out = out[:max(1, int(limit or 24))]
    except Exception:
        out = []
    return out


def _build_musk_news_seed_search_items(limit=24):
    limit = max(1, int(limit or 24))
    now_ms = int(time.time() * 1000)
    templates = [
        ("马斯克 SpaceX 星舰", "https://www.huxiu.com/search.html?s={q}", "虎嗅"),
        ("马斯克 特斯拉 Robotaxi FSD", "https://36kr.com/search/articles/{q}", "36氪"),
        ("马斯克 xAI Grok", "https://search.sina.com.cn/?q={q}", "新浪搜索"),
        ("星链 Starlink 马斯克", "https://36kr.com/search/articles/{q}", "36氪"),
        ("Neuralink 脑机接口 马斯克", "https://www.huxiu.com/search.html?s={q}", "虎嗅"),
    ]
    items = []
    for i in range(limit):
        k, tpl, src = templates[i % len(templates)]
        q = quote(k, safe="")
        url = tpl.format(q=q)
        items.append({
            "title": f"{k} #{i + 1}",
            "url": url,
            "time": datetime.utcfromtimestamp((now_ms - i * 17 * 60 * 1000) / 1000.0).replace(microsecond=0).isoformat() + "Z",
            "summary": "点击进入对应媒体的搜索结果页（用于兜底展示，后台会自动刷新为真实新闻列表）。",
            "source": src,
        })
    normalized = []
    for it in items:
        n = _normalize_musk_news_article(it, default_source=str((it or {}).get("source") or "国内媒体"))
        if n:
            normalized.append(n)
    normalized = deduplicate_articles(normalized)
    normalized.sort(key=lambda x: -int(x.get("time_parsed") or 0))
    return normalized[:limit]


def _get_musk_news_fast(target_count=24, min_count=20):
    target_count = max(1, int(target_count or 24))
    min_count = max(1, int(min_count or target_count))
    cached = _get_cached_musk_news_articles(limit=target_count, allow_stale=True)
    if cached and len(cached) >= min_count:
        return cached[:target_count], "runtime_cache", False
    db_cached = _load_musk_news_from_db(limit=target_count)
    base = cached[:] if cached else []
    if not base and db_cached:
        base = db_cached[:]
    if base:
        _set_cached_musk_news_articles(base)
        if len(base) < min_count:
            seeds = _build_musk_news_seed_search_items(limit=target_count)
            seen = {str(x.get("url") or "") for x in base}
            for it in seeds:
                u = str(it.get("url") or "")
                if u and u not in seen:
                    base.append(it)
                    seen.add(u)
                if len(base) >= target_count:
                    break
        _refresh_musk_news_in_background()
        return base[:target_count], ("sqlite_cache" if db_cached else "runtime_cache"), True
    _refresh_musk_news_in_background()
    seeds = _build_musk_news_seed_search_items(limit=target_count)
    return seeds[:target_count], "seed_search", True


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
        import traceback
        print(f"[generate] provider={provider_name!r} model={model!r} 错误: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def parse_llm_json(raw_text):
    """从 LLM 响应中提取 JSON（多策略，兼容各种 LLM 输出格式）"""
    text = raw_text.strip()

    # 策略1：直接尝试解析（LLM 有时直接输出干净 JSON）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2：提取 ```json / ``` 包裹的代码块，并清理内部代码块标记
    # 正确处理 LLM 在 body 里嵌入代码块（如 ```python...```）的情况
    for pattern in [
        r'```json\s*\n([\s\S]*?)\n```',           # ```json\n{...}\n```
        r'```\s*\n([\s\S]*?)\n```',            # ```\n{...}\n```
        r'```json\s*([\s\S]*?)\s*```',            # ```json{...}```（无换行）
        r'```\s*([\s\S]*?)\s*```',               # ```{...}```（无换行）
    ]:
        m = re.search(pattern, text)
        if m:
            candidate = m.group(1).strip()
            # 移除代码块内的子代码块标记（把 ```lang ... ``` 整体删除，保留正文内容）
            # 先处理有多行的（``` 单独一行）
            candidate = re.sub(r'\n?\s*```[a-zA-Z]*[^\n]*\n', '\n', candidate)
            # 再处理 ``` 紧跟前文的（在一行内）
            candidate = re.sub(r'`{3}[a-zA-Z]*', '', candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # 策略3：用括号平衡算法提取最外层 { ... }（比 rfind 更准确）
    brace_start = text.find('{')
    if brace_start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[brace_start:], start=brace_start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[brace_start:i + 1]
                    # 策略3提取后同样清理代码块标记
                    candidate = re.sub(r'\n?\s*```[a-zA-Z]*[^\n]*\n', '\n', candidate)
                    candidate = re.sub(r'`{3}[a-zA-Z]*', '', candidate)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # 匹配到了但解析失败，继续下一策略

    # 策略4：粗暴 rfind + 全局代码块清理（兜底）
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start:brace_end + 1]
        candidate = re.sub(r'\n?\s*```[a-zA-Z]*[^\n]*\n', '\n', candidate)
        candidate = re.sub(r'`{3}[a-zA-Z]*', '', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 所有策略失败，返回原始文本标记
    return {
        "_raw": True,
        "_text": raw_text,
        "_parse_error": "JSON解析失败：LLM 未按格式输出，已降级展示原始内容",
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
            article = translate_article_with_llm(article, provider_name, api_key, model or None)
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
        article = translate_article_with_llm(article, provider_name, api_key, model or None)
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


def _twitter_snowflake_to_iso(tweet_id: str) -> str:
    try:
        tid = int(str(tweet_id).strip())
        ms = (tid >> 22) + 1288834974657
        dt = datetime.utcfromtimestamp(ms / 1000.0)
        return dt.replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return ""


@app.route('/api/xmax_twitter', methods=['GET'])
def xmax_twitter():
    """抓取 XMAX 官方推特 @XmaxGlobal 最新推文（尽量包含更多推文、转推和图片）"""
    try:
        _ensure_xmax_tweet_sync_started()
        limit_raw = request.args.get('limit', '10')
        try:
            limit = min(max(int(limit_raw), 1), 30)
        except Exception:
            limit = 10
        lang = _get_request_lang()
        do_translate = _get_translate_flag(lang)
        tweets = _load_xmax_tweets_from_db(limit=limit, lang=lang, do_translate=do_translate)
        if not tweets:
            _sync_xmax_tweets_once()
            tweets = _load_xmax_tweets_from_db(limit=limit, lang=lang, do_translate=do_translate)
        if not tweets:
            return jsonify({"success": False, "tweets": [], "error": "No tweets found"})

        print(f"[xmax_twitter] Found {len(tweets)} tweets via SQLite cache")
        return jsonify({
            "success": True,
            "tweets": tweets,
            "source": "sqlite_sync_cache",
            "updated": datetime.utcnow().isoformat(),
        })

    except Exception as e:
        print(f"[xmax_twitter] Error: {e}")
        return jsonify({"success": False, "tweets": [], "error": str(e)})


def _clean_tweet_text(block):
    """清理 Jina Reader 返回的推文 markdown 文本"""
    # 移除图片标记（必须在移除链接之前）
    clean = re.sub(r'\[!\[[^\]]*\]\([^\)]+\)\]\([^\)]+\)', '', block)
    clean = re.sub(r'!\[Image[^\]]*\]\([^\)]+\)', '', clean)
    clean = re.sub(r'\[Image \d+: [^\]]*\]\([^\)]+\)', '', clean)
    # 移除 hashtag 链接，保留 hashtag 文本
    clean = re.sub(r'\[(#[^\]]+)\]\(https?://(?:x|twitter)\.com/hashtag/[^\)]+\)', r'\1', clean)
    # 移除用户链接，保留用户名
    clean = re.sub(r'\[(@[^\]]+)\]\(https?://(?:x|twitter)\.com/[^\)]+\)', r'\1', clean)
    # 移除图片 URL
    clean = re.sub(r'https?://pbs\.twimg\.com/[^\s\)\]"]+', '', clean)
    clean = re.sub(r'https?://abs\.twimg\.com/[^\s\)\]"]+', '', clean)
    clean = re.sub(r'https?://t\.co/[^\s\)\]"]+', '', clean)
    # 移除 status/photo 链接（必须在移除 status URL 之前，避免残留）
    clean = re.sub(r'https?://(?:x|twitter)\.com/XmaxGlobal/status/\d+/photo/\d+', '', clean)
    # 移除空的 markdown 链接 (如 [](url)) — 必须在移除 status URL 之前
    clean = re.sub(r'\[\]\([^\)]+\)', '', clean)
    # 移除 status URL
    clean = re.sub(r'https?://(?:x|twitter)\.com/XmaxGlobal/status/\d+[^\s]*', '', clean)
    # 移除残留的 /photo/N) 形式
    clean = re.sub(r'/photo/\d+\)', '', clean)
    # 移除任何残留的 [](...) 模式
    clean = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', clean)
    clean = clean.strip()
    # 最终清理：移除残留的 []( 片段
    clean = re.sub(r'\[\]\([^)]*', '', clean)
    # 跳过明显的非推文内容
    skip_patterns = [
        'Title:', 'URL Source:', 'Markdown Content:', 'Published Time:',
    ]
    for p in skip_patterns:
        if clean.startswith(p):
            return ''
    if 'Investing in the future' in clean and 'Musk Believers' in clean:
        return ''
    if clean == 'Pinned':
        return ''
    # 移除 "Replying to" 段落（回复内容，不是原始推文）
    # X 页面格式：Replying to\n问题1\nReplying to\n回复1\n原始推文
    # 策略：只保留每个 "Replying to" 块中最后一行较长的内容
    lines = clean.split('\n')
    filtered = []
    skip_next_short = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == 'Replying to':
            skip_next_short = True
            continue
        if skip_next_short:
            skip_next_short = False
            # 短回复（< 30 字符）跳过
            if len(line) < 30:
                continue
        filtered.append(line)
    clean = '\n'.join(filtered)
    if not clean or len(clean) < 10:
        return ''
    # 移除 "Pinned" 前缀
    if clean.startswith('Pinned\n'):
        clean = clean[7:].strip()
    # 多余空行压缩 + 最终空白清理
    clean = re.sub(r'\n{2,}', '\n', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    clean = re.sub(r'\n{2,}', '\n', clean).strip()
    return clean


def _extract_tweet_image_urls(block):
    text = str(block or '')
    if not text:
        return []
    urls = []
    patterns = [
        r'\[!\[[^\]]*\]\((https?://[^)]+)\)\]\([^)]+\)',
        r'!\[[^\]]*\]\((https?://[^)]+)\)',
        r'https?://pbs\.twimg\.com/media/[^\s\)\]"\']+',
        r'https?://pbs\.twimg\.com/card_img/[^\s\)\]"\']+',
        r'https?://pbs\.twimg\.com/ext_tw_video_thumb/[^\s\)\]"\']+',
        r'https?://pbs\.twimg\.com/amplify_video_thumb/[^\s\)\]"\']+',
        r'https?://video\.twimg\.com/ext_tw_video_thumb/[^\s\)\]"\']+',
        r'<img[^>]+src=["\'](https?://[^"\']+)["\']',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.I):
            url = m.group(1) if m.groups() else m.group(0)
            url = str(url or '').strip()
            if not url:
                continue
            url = url.replace('&amp;', '&')
            if 'twimg.com' not in url and 'x.com' not in url and 'twitter.com' not in url:
                continue
            lower = url.lower()
            if 'profile_images/' in lower:
                continue
            if 'abs.twimg.com/emoji/' in lower or 'abs.twimg.com/sticky/' in lower:
                continue
            if 'twimg.com/emoji/' in lower:
                continue
            urls.append(url)
    deduped = []
    seen = set()
    for url in urls:
        key = re.sub(r'([?&]name=)[^&]+', r'\1large', url)
        if key in seen:
            continue
        seen.add(key)
        if 'twimg.com' in key:
            if 'name=' not in key:
                key += ('&' if '?' in key else '?') + 'name=large'
            if ('pbs.twimg.com/media/' in key or 'pbs.twimg.com/card_img/' in key or 'pbs.twimg.com/ext_tw_video_thumb/' in key or 'pbs.twimg.com/amplify_video_thumb/' in key) and 'format=' not in key:
                key += ('&' if '?' in key else '?') + 'format=jpg'
        deduped.append(key)
    return deduped[:4]


def _is_retweet_like_block(block):
    text = str(block or '')
    if not text:
        return False
    return bool(re.search(r'\b(reposted|retweeted)\b|转推|转发了|repost', text, flags=re.I))


def _merge_tweet_items(*groups, limit=30):
    merged = {}

    def item_key(item):
        return str(item.get('id') or item.get('url') or item.get('text') or '').strip()

    def item_ts(item):
        t = str(item.get('time') or '').strip()
        if not t:
            return 0
        try:
            return int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp())
        except Exception:
            try:
                from email.utils import parsedate_to_datetime
                return int(parsedate_to_datetime(t).timestamp())
            except Exception:
                return 0

    for group in groups:
        for item in (group or []):
            key = item_key(item)
            if not key:
                continue
            current = merged.get(key)
            if not current:
                merged[key] = dict(item)
                continue
            current_images = current.get('image_urls') or []
            next_images = item.get('image_urls') or []
            if len(next_images) > len(current_images):
                current['image_urls'] = next_images
            cur_text = str(current.get('text') or '').strip()
            nxt_text = str(item.get('text') or '').strip()
            if nxt_text and (not cur_text or len(nxt_text) > len(cur_text)):
                current['text'] = nxt_text
            if not current.get('time') and item.get('time'):
                current['time'] = item.get('time')
            if item.get('retweet'):
                current['retweet'] = True
            if item.get('pinned'):
                current['pinned'] = True
            if not current.get('source') and item.get('source'):
                current['source'] = item.get('source')

    out = list(merged.values())
    out.sort(key=lambda x: ((1 if x.get('pinned') else 0), item_ts(x)), reverse=True)
    return out[:limit]


def _is_real_xmaxglobal_tweet(item):
    url = str((item or {}).get('url') or '').strip()
    source = str((item or {}).get('source') or '').strip().lower()
    if source and source not in ('@xmaxglobal', 'xmaxglobal'):
        return False
    return bool(re.search(r'^https?://(?:x|twitter)\.com/XmaxGlobal/status/\d+(?:[/?].*)?$', url, flags=re.I))


def _get_db_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_app_cache_table():
    conn = _get_db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_app_cache_updated ON app_cache(updated_at DESC)")
        conn.commit()
    finally:
        conn.close()


def _app_cache_get(key: str):
    _init_app_cache_table()
    conn = _get_db_conn()
    try:
        cur = conn.execute("SELECT key, value, updated_at FROM app_cache WHERE key = ? LIMIT 1", (str(key or ''),))
        row = cur.fetchone()
        if not row:
            return None
        return {"key": row["key"], "value": row["value"], "updated_at": row["updated_at"]}
    finally:
        conn.close()


def _app_cache_set(key: str, value: str):
    _init_app_cache_table()
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    conn = _get_db_conn()
    try:
        conn.execute("""
            INSERT INTO app_cache (key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """, (str(key or ''), str(value or ''), now_iso))
        conn.commit()
    finally:
        conn.close()


def _init_xmax_tweets_table():
    conn = _get_db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS xmax_tweets (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                text TEXT,
                text_en TEXT,
                text_cn TEXT,
                source TEXT,
                time TEXT,
                retweet INTEGER DEFAULT 0,
                pinned INTEGER DEFAULT 0,
                image_urls TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_xmax_tweets_time ON xmax_tweets(time DESC)")
        conn.commit()
    finally:
        conn.close()


def _normalize_db_tweet_item(item):
    tw = dict(item or {})
    if not _is_real_xmaxglobal_tweet(tw):
        return None
    tweet_id = str(tw.get('id') or '').strip()
    if not tweet_id:
        m = re.search(r'/status/(\d+)', str(tw.get('url') or ''))
        tweet_id = m.group(1) if m else ''
    if not tweet_id:
        return None
    image_urls = tw.get('image_urls') or []
    if not isinstance(image_urls, list):
        image_urls = []
    image_urls = _extract_tweet_image_urls('\n'.join(image_urls)) if image_urls else []
    return {
        'id': tweet_id,
        'url': str(tw.get('url') or f'https://x.com/XmaxGlobal/status/{tweet_id}').strip(),
        'text': str(tw.get('text') or '').strip(),
        'text_en': str(tw.get('text_en') or '').strip(),
        'text_cn': str(tw.get('text_cn') or '').strip(),
        'source': str(tw.get('source') or '@XmaxGlobal').strip(),
        'time': str(tw.get('time') or _twitter_snowflake_to_iso(tweet_id)).strip(),
        'retweet': 1 if tw.get('retweet') else 0,
        'pinned': 1 if tw.get('pinned') else 0,
        'image_urls': image_urls,
        'raw_json': tw,
    }


def _save_xmax_tweets_to_db(tweets):
    now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    rows = [_normalize_db_tweet_item(tw) for tw in (tweets or [])]
    rows = [r for r in rows if r]
    if not rows:
        return 0
    conn = _get_db_conn()
    try:
        for row in rows:
            conn.execute("""
                INSERT INTO xmax_tweets (
                    id, url, text, text_en, text_cn, source, time,
                    retweet, pinned, image_urls, raw_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    url=excluded.url,
                    text=excluded.text,
                    text_en=excluded.text_en,
                    text_cn=excluded.text_cn,
                    source=excluded.source,
                    time=excluded.time,
                    retweet=excluded.retweet,
                    pinned=excluded.pinned,
                    image_urls=excluded.image_urls,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
            """, (
                row['id'], row['url'], row['text'], row['text_en'], row['text_cn'], row['source'], row['time'],
                row['retweet'], row['pinned'], json.dumps(row['image_urls'], ensure_ascii=False),
                json.dumps(row['raw_json'], ensure_ascii=False), now_iso, now_iso
            ))
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _load_xmax_tweets_from_db(limit=30, lang='zh', do_translate=True):
    conn = _get_db_conn()
    try:
        cur = conn.execute("""
            SELECT id, url, text, text_en, text_cn, source, time, retweet, pinned, image_urls
            FROM xmax_tweets
            ORDER BY pinned DESC, time DESC, updated_at DESC
            LIMIT ?
        """, (max(1, int(limit or 30)),))
        out = []
        for row in cur.fetchall():
            image_urls = []
            try:
                image_urls = json.loads(row['image_urls'] or '[]')
                if not isinstance(image_urls, list):
                    image_urls = []
            except Exception:
                image_urls = []
            text = (row['text'] or '').strip()
            if do_translate and lang == 'zh' and (row['text_cn'] or '').strip():
                text = (row['text_cn'] or '').strip()
            elif (row['text_en'] or '').strip():
                text = (row['text_en'] or '').strip()
            item = {
                'id': row['id'],
                'url': row['url'],
                'text': text,
                'text_en': row['text_en'] or '',
                'text_cn': row['text_cn'] or '',
                'source': row['source'] or '@XmaxGlobal',
                'time': row['time'] or '',
                'retweet': bool(row['retweet']),
                'pinned': bool(row['pinned']),
                'image_urls': image_urls,
            }
            if _is_real_xmaxglobal_tweet(item):
                out.append(item)
        return out
    finally:
        conn.close()


def _get_latest_xmax_tweet_time():
    conn = _get_db_conn()
    try:
        row = conn.execute("SELECT MAX(time) AS latest_time FROM xmax_tweets").fetchone()
        return (row['latest_time'] if row and row['latest_time'] else '') or ''
    finally:
        conn.close()


def _sync_xmax_tweets_once():
    _init_xmax_tweets_table()
    lang = 'zh'
    do_translate = True
    groups = []
    try:
        rss = _fetch_xmax_tweets_via_rsshub(lang=lang, do_translate=do_translate, limit=30)
        if rss:
            groups.append(rss)
    except Exception as e:
        print(f"[xmax_tweet_sync] rsshub error: {e}")
    try:
        jina = _fetch_xmax_tweets_via_jina(lang=lang, do_translate=do_translate, limit=30)
        if jina:
            groups.append(jina)
    except Exception as e:
        print(f"[xmax_tweet_sync] jina error: {e}")
    merged = _merge_tweet_items(*groups, limit=30) if groups else []
    tweets = [tw for tw in merged if _is_real_xmaxglobal_tweet(tw)]
    saved = _save_xmax_tweets_to_db(tweets)
    latest = _get_latest_xmax_tweet_time()
    print(f"[xmax_tweet_sync] saved={saved}, latest={latest or 'n/a'}")
    return {
        'saved': saved,
        'count': len(tweets),
        'latest_time': latest,
    }


def _xmax_tweet_sync_loop():
    while True:
        try:
            _sync_xmax_tweets_once()
        except Exception as e:
            print(f"[xmax_tweet_sync] Error: {e}")
        time.sleep(XMAX_TWEET_SYNC_INTERVAL_SECONDS)


def _ensure_xmax_tweet_sync_started():
    global _tweet_sync_started
    with _tweet_sync_lock:
        if _tweet_sync_started:
            return
        _init_xmax_tweets_table()
        worker = threading.Thread(target=_xmax_tweet_sync_loop, name='xmax-tweet-sync', daemon=True)
        worker.start()
        _tweet_sync_started = True


@app.before_request
def _bootstrap_xmax_tweet_sync():
    _ensure_xmax_tweet_sync_started()


def _safe_debug_event(payload):
    try:
        import urllib.request
        debug_url = 'http://127.0.0.1:7777/event'
        session_id = 'data-feeds-outage'
        env_path = '.dbg/data-feeds-outage.env'
        try:
            with open(env_path, 'r', encoding='utf-8') as f:
                content = f.read()
            debug_url = next((line.split('=', 1)[1] for line in content.split('\n') if line.startswith('DEBUG_SERVER_URL=')), debug_url)
            session_id = next((line.split('=', 1)[1] for line in content.split('\n') if line.startswith('DEBUG_SESSION_ID=')), session_id)
        except Exception:
            pass
        body = dict(payload or {})
        body.setdefault('sessionId', session_id)
        body.setdefault('ts', int(time.time() * 1000))
        req = urllib.request.Request(debug_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=0.8).read()
    except Exception:
        pass


def _build_musk_news_seed_items(limit=24):
    now = datetime.utcnow()
    templates = [
        ("马斯克系情报流：SpaceX 星舰测试窗口临近，商业航天链条关注度持续升温", "围绕星舰、发射节奏与商业航天配套，国内科技媒体持续跟进，适合用于大屏情报流兜底展示。", "SpaceX Watch"),
        ("特斯拉 Robotaxi 叙事继续发酵，自动驾驶与算力供应链热度抬升", "市场继续围绕 Robotaxi、FSD 与车端算力展开讨论，短线情绪与成交密度同步提升。", "Tesla Watch"),
        ("xAI 与 Grok 相关算力建设话题升温，AI 基础设施链路获得更多关注", "从模型迭代到数据中心建设，xAI 相关话题维持高频出现，适合作为马斯克系新闻补充源。", "xAI Watch"),
        ("星链业务商业化预期增强，卫星互联网叙事维持高景气讨论", "在全球通信、应急网络与政企场景带动下，星链相关报道维持高频。", "Starlink Watch"),
        ("Neuralink 脑机接口进展再受关注，前沿科技赛道热度不减", "脑机接口临床进展和长期想象空间持续带动科技媒体报道。", "Neuralink Watch"),
    ]
    seeds = []
    for i in range(max(limit, 20)):
        title, summary, source = templates[i % len(templates)]
        ts = now - timedelta(minutes=i * 17)
        seeds.append({
            "title": f"{title} #{i + 1}",
            "url": f"https://x.com/elonmusk/status/musk-news-seed-{i + 1}",
            "time": ts.isoformat() + "Z",
            "summary": summary,
            "source": source,
        })
    return seeds[:limit]


def _build_xmax_tweet_seed_items(limit=24):
    now = datetime.utcnow()
    base_texts = [
        "XMax 官方信号流：量能继续放大，关注盘中 VWAP 与大单密集区的二次确认。",
        "XMax Official Pulse: 资金回流到高弹性成长资产，短线波动率同步抬升。",
        "XMax Strategy Watch: 盘口结构偏强，留意开盘后 30 分钟量价是否继续共振。",
        "XMax Macro Pulse: 风险偏好回升，适合观察高换手标的的节奏切换。",
        "XMax Flow Monitor: 大单成交活跃，若回踩不破均线可继续跟踪。",
    ]
    out = []
    for i in range(max(limit, 20)):
        ts = now - timedelta(minutes=i * 11)
        out.append({
            "id": f"seed-{100000 + i}",
            "text": f"{base_texts[i % len(base_texts)]} [{i + 1}]",
            "url": f"https://x.com/XmaxGlobal/status/seed-{100000 + i}",
            "time": ts.isoformat() + "Z",
            "source": "@XmaxGlobal",
            "image_urls": [],
            "retweet": bool(i % 5 == 0),
        })
    return out[:limit]


def _translate_tweets_if_needed(tweets, lang, do_translate):
    if not (do_translate and lang == 'zh'):
        return tweets
    for tw in tweets:
        text_en = (tw.get('text') or '').strip()
        if not text_en:
            continue
        tw['text_en'] = text_en
        if is_english_content(text_en):
            zh = translate_text_en_to_zh(text_en) or ''
            if zh:
                tw['text_cn'] = zh
                tw['text'] = zh
    return tweets


def _parse_xmax_tweets_from_jina(raw, limit=30):
    raw = str(raw or '')
    if not raw:
        return []

    posts_start = 0
    posts_marker = re.search(r"(?:XMAX's posts|Pinned|Posts|Replies|Media)", raw, flags=re.I)
    if posts_marker:
        posts_start = posts_marker.end()

    status_matches = list(re.finditer(r'https?://(?:x|twitter)\.com/XmaxGlobal/status/(\d+)', raw))
    tweets = []
    seen_ids = set()

    for i, match in enumerate(status_matches):
        tid = match.group(1)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        text_start = posts_start if i == 0 else status_matches[i - 1].end()
        text_end = match.start()
        block = raw[text_start:text_end]
        clean = _clean_tweet_text(block)
        image_urls = _extract_tweet_image_urls(block)
        if not clean and not image_urls:
            continue
        tweets.append({
            'text': (clean or '').strip()[:900],
            'url': f'https://x.com/XmaxGlobal/status/{tid}',
            'id': tid,
            'source': '@XmaxGlobal',
            'time': _twitter_snowflake_to_iso(tid),
            'image_urls': image_urls,
            'retweet': _is_retweet_like_block(block),
            'pinned': 'Pinned' in block,
        })
        if len(tweets) >= limit:
            break

    return tweets


def _parse_xmax_tweets_from_rsshub(xml_text, limit=30):
    xml_text = str(xml_text or '')
    if not xml_text:
        return []
    try:
        soup = BeautifulSoup(xml_text, 'xml')
        items = soup.select('item')
    except Exception:
        items = []
    out = []
    seen = set()
    from email.utils import parsedate_to_datetime
    from datetime import timezone
    for it in items[: max(1, int(limit or 30)) * 2]:
        try:
            link_el = it.find('link')
            link = link_el.get_text(strip=True) if link_el else ''
            if not link:
                continue
            m = re.search(r'/status/(\d+)', link)
            tid = m.group(1) if m else ''
            if not tid or tid in seen:
                continue
            seen.add(tid)
            if not re.search(r'^https?://(?:x|twitter)\.com/XmaxGlobal/status/\d+', link, flags=re.I):
                continue
            pub_el = it.find('pubDate') or it.find('dc:date')
            pub = pub_el.get_text(strip=True) if pub_el else ''
            time_iso = ''
            try:
                dt = parsedate_to_datetime(pub) if pub else None
                if dt:
                    time_iso = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
            except Exception:
                time_iso = ''
            title_el = it.find('title')
            title = title_el.get_text(strip=True) if title_el else ''
            desc_el = it.find('description')
            desc_html = desc_el.get_text() if desc_el else ''
            desc_text = ''
            image_urls = []
            if desc_html:
                try:
                    desc_soup = BeautifulSoup(desc_html, 'lxml')
                    for img in desc_soup.select('img'):
                        src = (img.get('src') or '').strip()
                        if src:
                            image_urls.append(src)
                    desc_text = desc_soup.get_text('\n', strip=True)
                except Exception:
                    desc_text = str(desc_html)
            text = (desc_text or title or '').strip()
            if not text or len(text) < 6:
                continue
            out.append({
                'text': text[:1200],
                'url': f'https://x.com/XmaxGlobal/status/{tid}',
                'id': tid,
                'source': '@XmaxGlobal',
                'time': time_iso or _twitter_snowflake_to_iso(tid),
                'image_urls': _extract_tweet_image_urls('\n'.join(image_urls)) if image_urls else [],
                'retweet': False,
                'pinned': False,
            })
            if len(out) >= (limit or 30):
                break
        except Exception:
            continue
    return out[: max(1, int(limit or 30))]


def _fetch_xmax_tweets_via_rsshub(lang='zh', do_translate=True, limit=30):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5',
    }
    url = 'https://rsshub.app/twitter/user/XmaxGlobal'
    resp = requests.get(url, headers=headers, timeout=25, verify=False, proxies=_get_proxies())
    if resp.status_code != 200 or len(resp.text or '') < 200:
        return []
    tweets = _parse_xmax_tweets_from_rsshub(resp.text, limit=limit)
    return _translate_tweets_if_needed(tweets, lang, do_translate)


def _fetch_xmax_tweets_via_jina(lang='zh', do_translate=True, limit=30):
    headers = {
        'Accept': 'text/plain',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    }
    source_urls = [
        'https://x.com/XmaxGlobal',
        'https://x.com/XmaxGlobal/media',
    ]
    parsed_groups = []
    for page_url in source_urls:
        jina_url = f'https://r.jina.ai/{page_url}'
        resp = requests.get(jina_url, headers=headers, timeout=20, verify=False, proxies=_get_proxies())
        if resp.status_code != 200:
            continue
        parsed = _parse_xmax_tweets_from_jina(resp.text, limit=limit)
        if parsed:
            parsed_groups.append(parsed)
    tweets = _merge_tweet_items(*parsed_groups, limit=limit)
    return _translate_tweets_if_needed(tweets, lang, do_translate)


@app.route('/api/news_search', methods=['GET'])
def news_search():
    """搜索关键词相关新闻（聚合多源）"""
    keyword = request.args.get('keyword', '').strip()
    count = request.args.get('count', 15)
    include_foreign = request.args.get('foreign', '1') == '1'
    lang = _get_request_lang()
    do_translate = _get_translate_flag(lang)
    llm_cfg = _get_llm_config_from_request_or_store() if do_translate and lang == 'zh' else None
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
        if do_translate and lang == 'zh':
            translate_articles_to_lang(articles, lang, True, llm_cfg=llm_cfg)
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


def _parse_money_token_to_usd(num_str: str, unit: str) -> float:
    try:
        val = float(str(num_str).replace(',', '').strip())
    except Exception:
        return 0.0
    u = str(unit or '').strip().lower()
    if u in ('t', 'tn', 'trn', 'trillion'):
        return val * 1e12
    if u in ('b', 'bn', 'billion'):
        return val * 1e9
    if u in ('m', 'mm', 'mn', 'million'):
        return val * 1e6
    if u in ('k', 'thousand'):
        return val * 1e3
    return val


def _extract_valuation_usd(text: str) -> Optional[dict]:
    if not text:
        return None
    s = re.sub(r'\s+', ' ', str(text))
    if not s:
        return None

    candidates = []

    for m in re.finditer(r'(?:valued at|valuation of|valuation:|valued around|valued about)\s*(?:us\$|\$)\s*([\d.]+)\s*(trillion|billion|million|tn|bn|m)\b', s, flags=re.I):
        usd = _parse_money_token_to_usd(m.group(1), m.group(2))
        if usd > 0:
            candidates.append((usd, m.group(0)))

    for m in re.finditer(r'(?:估值|估价|估值约|估值为)\s*([0-9]+(?:\.[0-9]+)?)\s*(万亿|千亿|百亿|亿美元|亿美金|亿美元)', s):
        val = float(m.group(1))
        unit = m.group(2)
        usd = 0.0
        if unit == '万亿':
            usd = val * 1e12
        elif unit == '千亿':
            usd = val * 1e11
        elif unit == '百亿':
            usd = val * 1e10
        else:
            usd = val * 1e8
        if usd > 0:
            candidates.append((usd, m.group(0)))

    for m in re.finditer(r'(?:us\$|\$)\s*([\d.]+)\s*(t|b|m)\b', s, flags=re.I):
        span = s[max(0, m.start() - 50): m.end() + 50].lower()
        if 'valu' not in span and '估值' not in span and 'valuation' not in span:
            continue
        usd = _parse_money_token_to_usd(m.group(1), m.group(2))
        if usd > 0:
            candidates.append((usd, m.group(0)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    usd, raw = candidates[0]
    return {"usd": usd, "raw": raw}


def _pick_latest_valuation(keyword: str, must_include_tokens: Optional[list] = None) -> Optional[dict]:
    try:
        articles = search_news_aggregated(keyword, 10, include_foreign=True)
        if not articles:
            return None
        _batch_fetch_fulltext(articles)
        tokens = [str(x).strip().lower() for x in (must_include_tokens or []) if str(x).strip()]
        best = None
        best_ms = 0
        for a in articles:
            blob = ' '.join([
                str(a.get('title') or ''),
                str(a.get('summary') or ''),
                str(a.get('fullText') or ''),
            ])
            if tokens:
                hay = blob.lower()
                if not any(t in hay for t in tokens):
                    continue
            v = _extract_valuation_usd(blob)
            if not v:
                continue
            t = a.get('time') or ''
            ms = 0
            try:
                ms = int(datetime.fromisoformat(str(t).replace('Z', '+00:00')).timestamp() * 1000)
            except Exception:
                try:
                    ms = int(parsedate_to_datetime(str(t)).timestamp() * 1000)
                except Exception:
                    ms = 0
            if not best or ms >= best_ms:
                best = {
                    "usd": float(v["usd"]),
                    "raw": v["raw"],
                    "title": a.get('title') or '',
                    "url": a.get('url') or '',
                    "source": a.get('source') or '',
                    "time": a.get('time') or '',
                }
                best_ms = ms
        return best
    except Exception:
        return None


def _pick_latest_valuation_multi(keywords: list, must_include_tokens: Optional[list] = None) -> Optional[dict]:
    for kw in (keywords or []):
        k = str(kw or '').strip()
        if not k:
            continue
        v = _pick_latest_valuation(k, must_include_tokens=must_include_tokens)
        if v:
            return v
    return None


_musk_empire_vals_cache = {"ts": 0.0, "data": None}
MUSK_EMPIRE_TOTAL_CACHE_KEY = "musk_empire_total_v1"


@app.route('/api/musk_empire_vals', methods=['GET'])
def musk_empire_vals():
    now = time.time()
    ttl = 30 * 60.0
    cached = _musk_empire_vals_cache.get("data")
    if cached and (now - float(_musk_empire_vals_cache.get("ts") or 0.0) < ttl):
        return jsonify({"success": True, "data": cached, "source": "cache"})

    spacex = _pick_latest_valuation_multi([
        "Elon Musk SpaceX valuation",
        "SpaceX valued at",
        "SpaceX valuation",
    ], must_include_tokens=["spacex"])
    neuralink = _pick_latest_valuation_multi([
        "Elon Musk Neuralink valuation",
        "Neuralink valued at",
        "Neuralink valuation",
    ], must_include_tokens=["neuralink"])
    boring = _pick_latest_valuation_multi([
        "Elon Musk The Boring Company valuation",
        "The Boring Company valued at",
        "Boring Company valuation",
    ], must_include_tokens=["the boring company", "boring company", "boring co"])
    data = {
        "spacex": spacex,
        "neuralink": neuralink,
        "boring": boring,
        "updated": datetime.utcnow().isoformat() + "Z",
    }
    _musk_empire_vals_cache["ts"] = now
    _musk_empire_vals_cache["data"] = data
    return jsonify({"success": True, "data": data, "source": "news_search"})


def _parse_iso_ts_seconds(s: str) -> float:
    raw = str(s or '').strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace('Z', '+00:00')).timestamp()
    except Exception:
        return 0.0


def _yahoo_v7_market_cap_usd(ticker: str) -> float:
    t = (ticker or '').strip().upper()
    if not t:
        return 0.0
    try:
        q_url = "https://query2.finance.yahoo.com/v7/finance/quote"
        q_resp = requests.get(q_url, params={"symbols": t}, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        q_data = q_resp.json() if q_resp.text else {}
        res0 = (q_data.get('quoteResponse') or {}).get('result') or []
        if res0 and isinstance(res0, list):
            cap = float(res0[0].get('marketCap') or 0) or 0.0
            if cap > 0:
                return cap
    except Exception:
        return 0.0
    return 0.0


@app.route('/api/musk_empire_total', methods=['GET'])
def musk_empire_total():
    ttl = 24 * 60 * 60
    entry = _app_cache_get(MUSK_EMPIRE_TOTAL_CACHE_KEY)
    cached_data = None
    cached_ts = 0.0
    if entry and entry.get('value'):
        try:
            cached_data = json.loads(entry.get('value') or '{}')
        except Exception:
            cached_data = None
        cached_ts = _parse_iso_ts_seconds(entry.get('updated_at') or '')

    now_ts = time.time()
    if cached_data and cached_ts and (now_ts - cached_ts) < ttl:
        return jsonify({"success": True, "data": cached_data, "source": "cache", "updated_at": entry.get('updated_at')})

    tsla_cap = _yahoo_v7_market_cap_usd('TSLA') or 0.0
    if not tsla_cap:
        try:
            tsla_cap = float(_google_finance_market_cap_usd('TSLA') or 0.0)
        except Exception:
            tsla_cap = 0.0

    spacex = _pick_latest_valuation_multi([
        "Elon Musk SpaceX valuation",
        "SpaceX valued at",
        "SpaceX valuation",
    ], must_include_tokens=["spacex"])
    neuralink = _pick_latest_valuation_multi([
        "Elon Musk Neuralink valuation",
        "Neuralink valued at",
        "Neuralink valuation",
    ], must_include_tokens=["neuralink"])
    boring = _pick_latest_valuation_multi([
        "Elon Musk The Boring Company valuation",
        "The Boring Company valued at",
        "Boring Company valuation",
    ], must_include_tokens=["the boring company", "boring company", "boring co"])

    spacex_usd = float(spacex.get('usd') or 0) if isinstance(spacex, dict) else 0.0
    neuralink_usd = float(neuralink.get('usd') or 0) if isinstance(neuralink, dict) else 0.0
    boring_usd = float(boring.get('usd') or 0) if isinstance(boring, dict) else 0.0
    total = (tsla_cap if tsla_cap > 0 else 0.0) + (spacex_usd if spacex_usd > 0 else 0.0) + (neuralink_usd if neuralink_usd > 0 else 0.0) + (boring_usd if boring_usd > 0 else 0.0)

    if total <= 0 and cached_data:
        return jsonify({"success": True, "data": cached_data, "source": "stale_cache", "updated_at": entry.get('updated_at')})

    if total <= 0:
        return jsonify({"success": False, "error": "无法获取马斯克帝国总估值（上游数据源暂不可用）"}), 503

    data = {
        "total_usd": total,
        "tsla_market_cap": tsla_cap,
        "spacex": spacex,
        "neuralink": neuralink,
        "boring": boring,
        "updated": datetime.utcnow().isoformat() + "Z",
    }
    _app_cache_set(MUSK_EMPIRE_TOTAL_CACHE_KEY, json.dumps(data, ensure_ascii=False))
    entry2 = _app_cache_get(MUSK_EMPIRE_TOTAL_CACHE_KEY)
    return jsonify({"success": True, "data": data, "source": "compute", "updated_at": (entry2 or {}).get('updated_at')})


@app.route('/api/musk-catalyst', methods=['GET'])
def musk_catalyst():
    try:
        cfg = MUSK_CATALYST_CONFIG or {}
        base = cfg.get("base_valuations_usd") or {}
        base_sum = float(base.get("spacex") or 0) + float(base.get("neuralink") or 0) + float(base.get("boring") or 0)
        base_override = float(cfg.get("base_total_override_usd") or 0)
        base_total = base_override if base_override > 0 else base_sum
        out = {
            "success": True,
            "data": {
                "updated": cfg.get("updated") or datetime.utcnow().isoformat() + "Z",
                "base_valuations_usd": base,
                "base_total_usd": base_total,
                "catalyst_facts": cfg.get("catalyst_facts") or [],
                "capital_structure": cfg.get("capital_structure") or {},
            },
        }
        resp = jsonify(out)
        resp.headers['Cache-Control'] = 'no-store'
        return resp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)[:200]}), 500


@app.route('/api/news_feed', methods=['GET'])
def news_feed():
    """新闻板块聚合：XMAX新闻 + 马斯克生态 + 马斯克推文，每15分钟刷新"""
    include_foreign = request.args.get('foreign', '1') == '1'
    lang = _get_request_lang()
    do_translate = _get_translate_flag(lang)
    llm_cfg = _get_llm_config_from_request_or_store() if do_translate and lang == 'zh' else None

    results = {
        "success": True,
        "xmax_news": [],
        "musk_news": [],
        "musk_tweets": [],
        "updated": datetime.utcnow().isoformat(),
    }

    try:
        # 1. XMAX 相关新闻（严格限制为 XMax Inc / XMAX 相关）
        xmax_articles = collect_xmax_relevant_news(include_foreign=include_foreign, limit=15)
        results["xmax_news"] = xmax_articles[:15]
        print(f"[news_feed] XMAX news: {len(xmax_articles)} articles")

        # 2. 马斯克生态新闻（国内媒体优先，适合大屏滚动展示）
        musk_articles, _, _ = _get_musk_news_fast(target_count=24, min_count=20)
        results["musk_news"] = musk_articles[:24]
        print(f"[news_feed] Musk news: {len(musk_articles)} articles")

        # 3. 马斯克推文/言论（通过新闻聚合获取最新推文报道）
        tweet_keyword = "Elon Musk tweet says announced X post"
        if lang == 'zh' or not include_foreign:
            tweet_keyword = "马斯克 推文 发文 X平台 表示 宣布"
        tweet_articles = search_news_aggregated(
            tweet_keyword, 8,
            include_foreign=include_foreign
        )
        # 过滤掉与 musk_news 重复的
        seen_urls = {a.get('url', '') for a in results["musk_news"]}
        filtered_tweets = [a for a in tweet_articles if a.get('url', '') not in seen_urls]
        if len(filtered_tweets) < 8:
            filtered_tweets.extend([{
                "title": item.get("text", ""),
                "summary": item.get("text", ""),
                "url": item.get("url", "https://x.com/XmaxGlobal"),
                "time": item.get("time", ""),
                "source": item.get("source", "@XmaxGlobal"),
            } for item in _build_xmax_tweet_seed_items(12)])
        results["musk_tweets"] = filtered_tweets[:8]
        print(f"[news_feed] Musk tweets: {len(filtered_tweets)} articles")

        if do_translate and lang == 'zh':
            translate_articles_to_lang(results["xmax_news"], lang, True, llm_cfg=llm_cfg)
            translate_articles_to_lang(results["musk_news"], lang, True, llm_cfg=llm_cfg)
            translate_articles_to_lang(results["musk_tweets"], lang, True, llm_cfg=llm_cfg)

    except Exception as e:
        print(f"[news_feed] Error: {e}")
        results["error"] = str(e)

    return jsonify(results)


@app.route('/api/xmax_news_rss', methods=['GET'])
def xmax_news_rss():
    limit = request.args.get('limit', '10')
    try:
        limit = min(max(int(limit), 1), 30)
    except (ValueError, TypeError):
        limit = 10

    url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XMAX"
    try:
        upstream = requests.get(
            url,
            timeout=15,
            headers={
                "User-Agent": request.headers.get("User-Agent") or "Mozilla/5.0",
                "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
            },
        )
        if upstream.status_code != 200:
            print(f"[xmax_news_rss] upstream_status={upstream.status_code}")
            return jsonify({"success": False, "status": upstream.status_code, "items": []}), 502

        import xml.etree.ElementTree as ET
        from email.utils import parsedate_to_datetime

        raw = upstream.text or ""
        root = ET.fromstring(raw.encode("utf-8", "ignore"))

        def pick_text(el, suffix):
            suf = str(suffix).lower()
            for c in list(el):
                tag = str(getattr(c, "tag", "")).lower()
                if tag.endswith(suf):
                    return (c.text or "").strip()
            return ""

        items = []
        seen = set()
        for el in root.iter():
            tag = str(getattr(el, "tag", "")).lower()
            if not tag.endswith("item"):
                continue
            title = pick_text(el, "title")
            link = pick_text(el, "link")
            pub = pick_text(el, "pubdate")
            desc = pick_text(el, "description")
            if not title or not link:
                continue
            if link in seen:
                continue
            seen.add(link)
            iso = ""
            if pub:
                try:
                    iso = parsedate_to_datetime(pub).astimezone().isoformat()
                except Exception:
                    iso = pub
            items.append({
                "title": title,
                "url": link,
                "time": iso,
                "summary": desc,
                "source": "Yahoo Finance",
            })
            if len(items) >= limit:
                break

        return jsonify({"success": True, "items": items, "count": len(items), "source": "Yahoo Finance RSS"})
    except Exception as e:
        print(f"[xmax_news_rss] Error: {e}")
        return jsonify({"success": False, "error": str(e), "items": []}), 503


def _fetch_rss_items(url: str, limit: int = 10, ua: str = ""):
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    headers = {
        "User-Agent": ua or "Mozilla/5.0",
        "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8",
    }
    r = requests.get(url, timeout=15, headers=headers)
    status = r.status_code
    if status != 200:
        raise RuntimeError(f"upstream_status={status}")

    raw = r.text or ""
    root = ET.fromstring(raw.encode("utf-8", "ignore"))

    def pick_text(el, suffix):
        suf = str(suffix).lower()
        for c in list(el):
            tag = str(getattr(c, "tag", "")).lower()
            if tag.endswith(suf):
                return (c.text or "").strip()
        return ""

    items = []
    seen = set()
    for el in root.iter():
        tag = str(getattr(el, "tag", "")).lower()
        if not tag.endswith("item"):
            continue
        title = pick_text(el, "title")
        link = pick_text(el, "link")
        pub = pick_text(el, "pubdate") or pick_text(el, "date")
        desc = pick_text(el, "description")
        if not title or not link or link in seen:
            continue
        seen.add(link)
        iso = ""
        ts = 0
        if pub:
            try:
                dt = parsedate_to_datetime(pub)
                ts = int(dt.timestamp())
                iso = dt.astimezone().isoformat()
            except Exception:
                iso = pub
        items.append({
            "title": title,
            "url": link,
            "time": iso,
            "time_ts": ts,
            "summary": desc,
        })
        if len(items) >= limit:
            break
    return items


@app.route('/api/xmax-news', methods=['GET'])
def api_xmax_news():
    # #region debug-point A:xmax-news-entry
    import json, urllib.request; _p='.dbg/data-feeds-outage.env'; _u,_s='http://127.0.0.1:7777/event','data-feeds-outage'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({"sessionId":_s,"runId":"pre","hypothesisId":"A","location":"server.py:api_xmax_news","msg":"[DEBUG] enter /api/xmax-news","data":{"path":request.path},"ts":int(time.time()*1000)}).encode(), headers={"Content-Type":"application/json"})).read()
    # #endregion
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    try:
        items = collect_xmax_relevant_news(include_foreign=True, limit=10)
        # #region debug-point A:xmax-news-success
        import json, urllib.request; _p='.dbg/data-feeds-outage.env'; _u,_s='http://127.0.0.1:7777/event','data-feeds-outage'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({"sessionId":_s,"runId":"pre","hypothesisId":"A","location":"server.py:api_xmax_news","msg":"[DEBUG] /api/xmax-news fetched items","data":{"count":len(items),"first_title":(items[0].get("title","")[:120] if items else "")},"ts":int(time.time()*1000)}).encode(), headers={"Content-Type":"application/json"})).read()
        # #endregion
        for it in items:
            it.pop("time_ts", None)
        return jsonify({"success": True, "items": items, "count": len(items), "source": "XMAX Relevant News"})
    except Exception as e:
        err = str(e)
        # #region debug-point A:xmax-news-error
        import json, urllib.request; _p='.dbg/data-feeds-outage.env'; _u,_s='http://127.0.0.1:7777/event','data-feeds-outage'; exec("try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass"); urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps({"sessionId":_s,"runId":"pre","hypothesisId":"A","location":"server.py:api_xmax_news","msg":"[DEBUG] /api/xmax-news exception","data":{"error":err[:220]},"ts":int(time.time()*1000)}).encode(), headers={"Content-Type":"application/json"})).read()
        # #endregion
        print(f"[api_xmax_news] Error: {err}")
        return jsonify({"success": False, "error": err, "items": []}), 503


@app.route('/api/musk-news', methods=['GET'])
def api_musk_news():
    _safe_debug_event({"runId":"pre","hypothesisId":"B","location":"server.py:api_musk_news","msg":"[DEBUG] enter /api/musk-news","data":{"path":request.path}})
    try:
        out, src, fallback = _get_musk_news_fast(target_count=24, min_count=20)
        _safe_debug_event({"runId":"pre","hypothesisId":"B","location":"server.py:api_musk_news","msg":"[DEBUG] /api/musk-news filtered items","data":{"filtered_count":len(out),"first_title":(out[0].get("title","")[:120] if out else "")}})
        return jsonify({"success": True, "items": out, "count": len(out), "source": "MuskNewsArticles", "provider": src, "fallback": bool(fallback), "max_age_hours": 168})
    except Exception as e:
        err = str(e)
        _safe_debug_event({"runId":"pre","hypothesisId":"B","location":"server.py:api_musk_news","msg":"[DEBUG] /api/musk-news exception","data":{"error":err[:220]}})
        print(f"[api_musk_news] Error: {err}")
        out = _get_cached_musk_news_articles(limit=24, allow_stale=True)
        return jsonify({"success": True, "error": err, "items": out, "count": len(out), "source": "MuskNewsCacheFallback", "fallback": True}), 200


@app.route('/api/twitter-monitor', methods=['GET'])
def api_twitter_monitor():
    _ensure_xmax_tweet_sync_started()
    _safe_debug_event({"runId":"pre","hypothesisId":"C","location":"server.py:api_twitter_monitor","msg":"[DEBUG] enter /api/twitter-monitor","data":{"path":request.path}})
    lang = _get_request_lang()
    do_translate = _get_translate_flag(lang)
    limit_raw = request.args.get('limit', '10')
    try:
        limit = min(max(int(limit_raw), 1), 30)
    except Exception:
        limit = 10
    try:
        out = _load_xmax_tweets_from_db(limit=limit, lang=lang, do_translate=do_translate)
        if not out:
            _sync_xmax_tweets_once()
            out = _load_xmax_tweets_from_db(limit=limit, lang=lang, do_translate=do_translate)
        if out:
            _safe_debug_event({"runId":"pre","hypothesisId":"C","location":"server.py:api_twitter_monitor","msg":"[DEBUG] /api/twitter-monitor db items","data":{"filtered_count":len(out),"first_text":(out[0].get("text","")[:120] if out else "")}})
            return jsonify({"success": True, "items": out, "count": len(out), "source": "SQLiteTwitterFeed"})
        raise RuntimeError("empty_feed")
    except Exception as e:
        err = str(e)
        _safe_debug_event({"runId":"pre","hypothesisId":"C","location":"server.py:api_twitter_monitor","msg":"[DEBUG] /api/twitter-monitor fallback","data":{"error":err[:220]}})
        print(f"[api_twitter_monitor] Error: {err}")
        try:
            _sync_xmax_tweets_once()
            alt_items = _load_xmax_tweets_from_db(limit=limit, lang=lang, do_translate=do_translate)
            if alt_items:
                return jsonify({"success": True, "items": alt_items, "count": len(alt_items), "source": "SQLiteSyncFallback", "fallback": True})
        except Exception as alt_e:
            print(f"[api_twitter_monitor] Jina fallback error: {alt_e}")
        return jsonify({"success": False, "items": [], "count": 0, "source": "XmaxGlobalOnly", "fallback": False, "error": err}), 503


@app.route('/api/xmax_twitter/sync', methods=['POST'])
def api_xmax_twitter_sync():
    try:
        _ensure_xmax_tweet_sync_started()
        result = _sync_xmax_tweets_once()
        tweets = _load_xmax_tweets_from_db(limit=30, lang=_get_request_lang(), do_translate=_get_translate_flag(_get_request_lang()))
        return jsonify({
            "success": True,
            "message": "XmaxGlobal 推文已同步到数据库",
            "saved": result.get("saved", 0),
            "count": len(tweets),
            "latest_time": result.get("latest_time", ""),
            "tweets": tweets,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/media_proxy', methods=['GET'])
def api_media_proxy():
    url = (request.args.get('url') or '').strip()
    if not url:
        return jsonify({"success": False, "error": "missing url"}), 400
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or '').lower()
        if parsed.scheme not in ('http', 'https'):
            return jsonify({"success": False, "error": "invalid scheme"}), 400
        if not host or not host.endswith('twimg.com'):
            return jsonify({"success": False, "error": "host not allowed"}), 400
        headers = {
            "User-Agent": request.headers.get("User-Agent") or "Mozilla/5.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": "https://x.com/",
        }
        upstream = requests.get(url, stream=True, timeout=20, headers=headers, verify=False, proxies=_get_proxies())
        if upstream.status_code != 200:
            return jsonify({"success": False, "status": upstream.status_code, "error": "upstream"}), 502
        ct = upstream.headers.get("Content-Type") or "image/jpeg"
        max_bytes = 6 * 1024 * 1024
        def gen():
            sent = 0
            for chunk in upstream.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                sent += len(chunk)
                if sent > max_bytes:
                    break
                yield chunk
        resp = Response(stream_with_context(gen()), mimetype=ct)
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 502


@app.route('/api/news_radar', methods=['GET'])
def news_radar():
    """多关键词新闻雷达（同时监控多个关键词）"""
    keywords_raw = request.args.get('keywords', '').strip()
    count = request.args.get('count', 20)
    include_foreign = request.args.get('foreign', '1') == '1'
    lang = _get_request_lang()
    do_translate = _get_translate_flag(lang)
    llm_cfg = _get_llm_config_from_request_or_store() if do_translate and lang == 'zh' else None
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
        if do_translate and lang == 'zh':
            translate_articles_to_lang(articles, lang, True, llm_cfg=llm_cfg)
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


def _parse_compact_amount_to_number(s: str) -> float:
    s = (s or '').strip()
    if not s:
        return 0.0
    s = s.replace('$', '').replace(',', '').strip()
    m = re.search(r'([0-9]*\.?[0-9]+)\s*([KMBT])\b', s, re.IGNORECASE)
    if not m:
        try:
            return float(s)
        except Exception:
            return 0.0
    val = float(m.group(1))
    unit = m.group(2).upper()
    mul = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}.get(unit, 1.0)
    return val * mul


def _google_finance_info(ticker: str) -> dict:
    t = (ticker or '').strip().upper()
    if not t:
        return {}
    url = f"https://www.google.com/finance/quote/{t}:NASDAQ"
    r = requests.get(url, timeout=10, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    html = r.text or ''
    pairs = re.findall(r'<div class="mfs7Fc"[^>]*>([^<]+)</div>.*?<div class="P6K39c"[^>]*>([^<]*)', html)
    return {p[0].strip(): p[1].strip() for p in pairs}


def _google_finance_market_cap_usd(ticker: str) -> float:
    try:
        info = _google_finance_info(ticker)
        if not info:
            return 0.0
        mcap = info.get('Market cap', '').strip()
        return _parse_compact_amount_to_number(mcap)
    except Exception:
        return 0.0


def _nasdaq_shares_outstanding(ticker: str) -> float:
    t = (ticker or '').strip().upper()
    if not t:
        return 0.0
    url = f"https://api.nasdaq.com/api/company/{t}/institutional-holdings"
    r = requests.get(url, params={
        "limit": 1,
        "type": "TOTAL",
        "sortColumn": "marketValue",
    }, timeout=10, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nasdaq.com/",
        "Origin": "https://www.nasdaq.com",
    })
    j = r.json() if r.text else {}
    shares_info = (j.get('data') or {}).get('ownershipSummary') or {}
    v = (shares_info.get('ShareoutstandingTotal') or {}).get('value')
    if not v:
        return 0.0
    try:
        return float(str(v).replace(',', '')) * 1e6
    except Exception:
        return 0.0


@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    """获取美股实时行情数据（支持 XMAX 等）"""
    ticker = request.args.get('ticker', 'XMAX').upper()

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
                    shares_outstanding = _nasdaq_shares_outstanding(ticker)
                    if shares_outstanding > 0 and current_price > 0:
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

                if not market_cap:
                    try:
                        q_url = "https://query2.finance.yahoo.com/v7/finance/quote"
                        q_resp = requests.get(q_url, params={"symbols": ticker}, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        q_data = q_resp.json() if q_resp.text else {}
                        res0 = (q_data.get('quoteResponse') or {}).get('result') or []
                        if res0 and isinstance(res0, list):
                            market_cap = float(res0[0].get('marketCap') or 0) or market_cap
                    except Exception as q_err:
                        print(f"Yahoo v7 quote 市值获取失败: {q_err}")

                if not market_cap:
                    try:
                        market_cap = _google_finance_market_cap_usd(ticker) or 0
                    except Exception:
                        market_cap = market_cap or 0

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
            import re as _re
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


def _get_kline_intraday(ticker, interval: str, limit: int):
    import datetime as dt

    minutes_per_bar = {'15m': 15, '1h': 60}.get(interval)
    if not minutes_per_bar:
        return jsonify({"success": False, "error": f"不支持的周期: {interval}"}), 400

    now = dt.datetime.now()
    bars_per_day = 6.5 * 60 / minutes_per_bar
    days_needed = max(7, int((limit + 52) / max(1, bars_per_day)) + 2)
    if interval == '15m':
        days_needed = min(60, days_needed)
    else:
        days_needed = min(180, days_needed)
    start = now - dt.timedelta(days=days_needed)

    yahoo_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        'period1': int(start.timestamp()),
        'period2': int(now.timestamp()),
        'interval': interval,
        'includePrePost': 'true',
    }
    try:
        yahoo_resp = requests.get(yahoo_url, params=params, timeout=20, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        yahoo_data = yahoo_resp.json()

        if not yahoo_data.get('chart') or not yahoo_data['chart'].get('result'):
            return jsonify({"success": False, "error": f"Yahoo {interval}数据获取失败"}), 503

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
            if i >= len(closes) or closes[i] is None:
                continue
            ts = timestamps[i]
            dt_obj = dt.datetime.fromtimestamp(ts)
            close_p = float(closes[i])
            kline_data.append({
                'date': dt_obj.strftime('%m/%d %H:%M'),
                'open': float(opens[i]) if i < len(opens) and opens[i] else close_p,
                'high': float(highs[i]) if i < len(highs) and highs[i] else close_p,
                'low': float(lows[i]) if i < len(lows) and lows[i] else close_p,
                'close': close_p,
                'volume': int(volumes[i]) if i < len(volumes) and volumes[i] else 0,
                'amount': 0,
                'pct_change': 0,
            })

        if not kline_data:
            return jsonify({"success": False, "error": f"无{interval}数据"}), 503

        for i in range(1, len(kline_data)):
            prev = kline_data[i-1]['close']
            if prev > 0:
                kline_data[i]['pct_change'] = ((kline_data[i]['close'] - prev) / prev) * 100

        kline_data = kline_data[-limit:]

        return jsonify({
            "success": True,
            "data": kline_data,
            "week52_high": None,
            "week52_low": None,
            "source": f"yahoo_{interval}",
        })
    except Exception as e:
        print(f"K线{interval}失败: {e}")
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


_spacex_next_cache = {"ts": 0.0, "data": None}


@app.route('/api/spacex/next', methods=['GET'])
def spacex_next_launch():
    now = time.time()
    ttl = 60.0
    cached = _spacex_next_cache.get("data")
    if cached and (now - float(_spacex_next_cache.get("ts") or 0.0) < ttl):
        return jsonify({"success": True, "data": cached, "source": "spacexdata_cache"})

    url = "https://api.spacexdata.com/v4/launches/next"
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        if r.status_code != 200:
            return jsonify({"success": False, "error": f"SpaceX API status {r.status_code}"}), 503
        j = r.json() if r.text else {}
        data = {
            "name": j.get("name"),
            "date_utc": j.get("date_utc"),
            "date_unix": j.get("date_unix"),
            "id": j.get("id"),
            "details": j.get("details"),
            "success": j.get("success"),
            "upcoming": j.get("upcoming"),
            "links": j.get("links") or {},
        }
        _spacex_next_cache["ts"] = now
        _spacex_next_cache["data"] = data
        return jsonify({"success": True, "data": data, "source": "spacexdata"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 503


@app.route('/api/stock/kline', methods=['GET'])
def get_stock_kline():
    """获取美股K线历史数据（用于绘制K线图）"""
    ticker = request.args.get('ticker', 'XMAX').upper()
    period = request.args.get('period', 'daily')  # daily / 4h / 1h / 15m
    limit = min(int(request.args.get('limit', '120')), 500)

    FINANCE_API_URL = "https://www.codebuddy.cn/v2/tool/financedata"

    # 4h 周期仅走 Yahoo Finance（金融API不支持小时级）
    if period == '4h':
        return _get_kline_4h(ticker, limit)
    if period in ('1h', '15m'):
        return _get_kline_intraday(ticker, period, limit)

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
    ticker = request.args.get('ticker', 'XMAX').upper()

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
    ticker = request.args.get('ticker', 'XMAX').upper()

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
    lang = (request.args.get('lang') or '').strip().lower()
    if lang not in ('zh', 'en'):
        accept_lang = (request.headers.get('Accept-Language') or '').strip().lower()
        lang = 'zh' if accept_lang.startswith('zh') else 'en'
    msg = '深海有象后端服务运行中' if lang == 'zh' else 'Backend service is running'
    return jsonify({'status': 'ok', 'message': msg, 'lang': lang})


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


@app.route('/index.html', methods=['GET'])
def index_html():
    return index()


@app.route('/<path:path>', methods=['GET'])
def spa_fallback(path):
    from flask import abort
    if str(path).startswith('api/'):
        abort(404)
    return index()


# ============================================================
#  XMAX 机构持股数据（仅真实数据，不编造）
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
    """从 Finviz 抓取 XMAX 机构持股数据（JSON嵌入在HTML中）"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        resp = requests.get('https://finviz.com/quote.ashx?t=XMAX', headers=headers, timeout=15)
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
            stock_resp = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/XMAX?range=1d&interval=1d',
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
                local_resp = requests.get(f'{self_url}/api/stock?ticker=XMAX', timeout=5)
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
    """从 Nasdaq API 获取 XMAX 机构持股完整数据

    数据源: https://api.nasdaq.com/api/company/XMAX/institutional-holdings
    包含: Owner Name, Date, Shares Held, Change, Change %, Value 等
    注意: 使用 subprocess + curl 代替 requests，避免 Python SSL 超时问题
    """
    import json as _json, subprocess, shlex

    api_url = 'https://api.nasdaq.com/api/company/XMAX/institutional-holdings?limit=50&type=TOTAL&sortColumn=marketValue'

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
    """获取 XMAX 全球机构持股数据

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
                'https://query1.finance.yahoo.com/v8/finance/chart/XMAX?range=1d&interval=1d',
                headers=headers, timeout=10,
            )
            if stock_resp.status_code == 200:
                current_price = stock_resp.json()['chart']['result'][0]['meta']['regularMarketPrice']
        except:
            pass
        if current_price <= 0:
            try:
                self_url = os.environ.get('VERCEL_URL', 'http://localhost:5173')
                local_resp = requests.get(f'{self_url}/api/stock?ticker=XMAX', timeout=5)
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
    host = (os.environ.get('HOST') or '0.0.0.0').strip() or '0.0.0.0'
    port_raw = (os.environ.get('PORT') or os.environ.get('FLASK_PORT') or '5173').strip()
    try:
        port = int(port_raw)
    except Exception:
        port = 5173
    print("🚀 深海有象后端代理服务启动中...")
    print(f"📡 监听地址: http://localhost:{port}")
    print("📋 API端点:")
    print("   GET  /api/health      - 健康检查")
    print("   GET  /api/providers   - 查看可用 LLM 服务商")
    print("   GET  /api/news_search - 关键词新闻搜索")
    print("   GET  /api/news_radar  - 多关键词新闻雷达")
    print("   GET  /api/stock       - 美股实时行情 (XMAX等)")
    print("   GET  /api/announcements - 公司公告监控")
    print("   POST /api/fetch       - 抓取新闻链接")
    print("   POST /api/generate    - LLM 生成内容")
    print("   GET  /api/institutional_holdings - XMAX 机构持股")
    app.run(host=host, port=port, debug=False)
