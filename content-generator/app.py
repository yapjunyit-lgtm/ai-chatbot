"""
✍️ AI 内容生成器 — 一键生成 SEO 文章、社媒帖子、营销文案
支持: OpenAI / DeepSeek / Groq / Claude / 自定义接口
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import os
from datetime import datetime
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from shared.style import inject_css, header, footer

load_dotenv()

# ── 供应商配置 ──────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "prefixes": ["sk-proj-", "sk-admin-"],
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o4-mini"],
    },
    "DeepSeek": {
        "prefixes": ["sk-"],
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Groq": {
        "prefixes": ["gsk_"],
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "qwen-2.5-32b"],
    },
    "Anthropic Claude": {
        "prefixes": ["sk-ant-"],
        "base_url": None,
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8", "claude-fable-5"],
    },
    "Together AI": {
        "prefixes": ["tgp_"],
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1"],
    },
    "OpenRouter": {
        "prefixes": ["sk-or-"],
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro"],
    },
}

# ── 内容类型 + Prompt 模板 ──────────────────────────────────
CONTENT_TYPES = {
    "📝 SEO 博客文章": {
        "prompt": "你是一位资深的 SEO 内容营销专家。请根据以下信息撰写一篇优化的博客文章。",
    },
    "📱 小红书帖子": {
        "prompt": "你是一位小红书爆款文案写手。请撰写一篇小红书帖子，使用 emoji、短句、话题标签，语气亲切自然。",
    },
    "🐦 Twitter / X 帖子": {
        "prompt": "你是一位 Twitter 内容创作者。请撰写一条吸引人的推文（或推文串），言简意赅，适合英文社交平台。",
    },
    "💼 LinkedIn 帖子": {
        "prompt": "你是一位 LinkedIn 个人品牌专家。请撰写一篇专业的 LinkedIn 帖子，适合商业社交平台。",
    },
    "📧 营销邮件": {
        "prompt": "你是一位邮件营销专家。请撰写一封转化率高的营销邮件，包含吸引人的标题、正文和 CTA。",
    },
    "🏷 产品描述": {
        "prompt": "你是一位电商文案专家。请撰写一段吸引人的产品描述，突出卖点和用户痛点的解决方案。",
    },
    "🎬 短视频脚本": {
        "prompt": "你是一位短视频编剧。请撰写一个短视频脚本（60 秒以内），包含画面描述和旁白/台词，注明时间节点。",
    },
    "📢 广告文案": {
        "prompt": "你是一位广告创意总监。请撰写一组多版本的广告文案（标题 + 正文 + CTA），适合投放测试。",
    },
    "🎤 演讲稿": {
        "prompt": "你是一位演讲撰稿人。请撰写一篇演讲稿，结构清晰，有感染力，适合口头表达。",
    },
    "🔑 关键词研究": {
        "prompt": "你是一位 SEO 关键词研究专家。请根据主题生成一组长尾关键词，按搜索意图分类。",
    },
}

TONES = [
    "👔 专业权威", "😊 轻松友好", "😂 幽默风趣",
    "🏛 正式严谨", "💪 激励向上", "🎯 极简直接",
    "🌿 温暖治愈", "⚡ 年轻潮流",
]

PLATFORMS = {
    "小红书": "使用中文 emoji、短段落、话题标签 #、口语化表达、结尾加互动引导",
    "Twitter": "英文为主，简洁有力，可加 hashtag，适合信息密度高的内容",
    "LinkedIn": "专业英文，结构化表达，bullet points 加分，结尾有 CTA",
    "博客/网站": "长文格式，有 H2/H3 标题层级，SEO 友好，可加 meta description",
    "抖音/TikTok": "脚本格式，标注画面和时长，口语化对话风格",
    "Instagram": "视觉描述 + 短文案，emoji 友好，hashtag 策略",
    "微信公众号": "中文长文，讲究排版，有摘要引导，可加分割线",
}


def detect_provider(api_key: str):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


def call_ai(messages: list, api_key: str, model: str, temperature: float, max_tokens: int, is_claude: bool, base_url: str = None):
    """流式调用 AI"""
    if is_claude:
        client = Anthropic(api_key=api_key)
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        claude_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": claude_msgs, "stream": True}
        if system_msg:
            kwargs["system"] = system_msg

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
    else:
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="✍️ AI 内容生成器", page_icon="✍️", layout="wide")
inject_css()
header("✍️", "AI Content Generator", "SEO · Social · Email · Ads · Scripts — 10 types, 8 tones, 7 platforms")

# ── 初始化 session ─────────────────────────────────────────
if "content_history" not in st.session_state:
    st.session_state.content_history = []

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API 设置")

    api_key = os.getenv("API_KEY") or st.text_input("API Key", type="password", placeholder="粘贴任何 API Key...")
    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)

    if api_key and provider_name:
        st.success(f"✅ **{provider_name}**")

    all_providers = list(PROVIDERS.keys()) + ["自定义 (OpenAI 兼容)"]
    manual_provider = st.selectbox("供应商", options=["🔍 自动检测"] + all_providers, index=0)

    if manual_provider == "自定义 (OpenAI 兼容)":
        custom_base_url = st.text_input("Base URL", placeholder="https://api.xxx.com/v1")
    else:
        custom_base_url = None

    if manual_provider.startswith("🔍"):
        active_config = provider_config or {"base_url": None, "models": ["gpt-3.5-turbo"]}
        active_name = provider_name or "未知"
    elif manual_provider == "自定义 (OpenAI 兼容)":
        active_config = {"base_url": custom_base_url, "models": ["default-model"]}
        active_name = "自定义"
    else:
        active_config = PROVIDERS[manual_provider]
        active_name = manual_provider

    is_claude = active_name == "Anthropic Claude"
    model = st.selectbox("模型", options=active_config["models"], index=0)

    temperature = st.slider("创造性", 0.0, 2.0, 0.8, 0.1, help="内容创作建议 0.7-1.0")

    max_tokens = st.slider("最大长度", 512, 8192, 2048, 128)

    st.divider()
    st.caption(f"🟢 **{active_name}** | `{model}`")

    # 历史记录
    st.divider()
    st.subheader("📋 历史记录")
    if st.button("🗑 清空历史", use_container_width=True):
        st.session_state.content_history = []
        st.rerun()
    st.caption(f"已生成 **{len(st.session_state.content_history)}** 篇")

# ── 主区域：表单 ───────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📝 内容设置")

    tab1, tab2 = st.tabs(["🎯 基础设置", "⚡ 高级选项"])

    with tab1:
        content_type = st.selectbox("内容类型", options=list(CONTENT_TYPES.keys()))

        col_a, col_b = st.columns(2)
        with col_a:
            tone = st.selectbox("语气风格", options=TONES)
        with col_b:
            platform = st.selectbox("发布平台", options=list(PLATFORMS.keys()))

        topic = st.text_input("主题 / 产品名", placeholder="例如：如何用 AI 提升工作效率")
        keywords = st.text_input("关键词（逗号分隔）", placeholder="例如：AI工具, 效率提升, 自动化")

    with tab2:
        target_audience = st.text_input("目标受众", placeholder="例如：25-35岁职场白领、中小企业家")
        extra_requirements = st.text_area(
            "额外要求",
            placeholder="例如：开头要吸引眼球 / 包含真实数据 / 结尾有购买引导 / 避免使用'非常''十分'等词...",
            height=100,
        )
        word_count = st.select_slider(
            "字数范围",
            options=["极短 ~100字", "短 ~300字", "中等 ~800字", "长 ~1500字", "超长 ~3000字"],
            value="中等 ~800字",
        )

# ── 右侧：预览 + Prompt ──────────────────────────────────
with col_right:
    st.subheader("🔍 生成预览")

    # 构建 Prompt
    template = CONTENT_TYPES[content_type]
    platform_guide = PLATFORMS[platform]

    system_prompt = f"""{template['prompt']}

语气风格：{tone}
发布平台指导：{platform_guide}

请严格遵循以下规范：
- 输出直接是发布内容，不要加"这是为您生成的"之类的废话
- 字数：{word_count}
- 格式适合直接复制粘贴到 {platform} 发布"""

    with st.expander("📋 完整 Prompt（可修改）", expanded=False):
        final_system = st.text_area("System Prompt", system_prompt, height=250)

    # 生成按钮
    generate_btn = st.button("🚀 生成内容", type="primary", use_container_width=True)

# ── 生成逻辑 ──────────────────────────────────────────────
if generate_btn:
    if not api_key:
        st.error("⚠️ 请先在侧边栏输入 API Key")
    elif not topic.strip():
        st.error("⚠️ 请至少输入主题")
    else:
        # 构建 User Prompt
        user_parts = [f"主题：{topic}"]
        if keywords.strip():
            user_parts.append(f"关键词：{keywords}")
        if target_audience.strip():
            user_parts.append(f"目标受众：{target_audience}")
        if extra_requirements.strip():
            user_parts.append(f"额外要求：{extra_requirements}")
        user_parts.append(f"平台：{platform}")
        user_parts.append(f"内容类型：{content_type}")

        user_prompt = "\n".join(user_parts)

        with st.spinner(f"✍️ 生成中... ({active_name})"):
            output_placeholder = st.empty()
            full_output = ""

            try:
                messages = [
                    {"role": "system", "content": final_system},
                    {"role": "user", "content": user_prompt},
                ]

                for token in call_ai(messages, api_key, model, temperature, max_tokens, is_claude, active_config.get("base_url")):
                    full_output += token
                    output_placeholder.markdown(full_output + "▌")

                output_placeholder.markdown(full_output)

                # 保存历史
                st.session_state.content_history.append({
                    "time": datetime.now().strftime("%H:%M"),
                    "type": content_type,
                    "topic": topic,
                    "content": full_output,
                })

                # 操作按钮
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    st.download_button(
                        "💾 下载 .txt", full_output, f"{topic[:20]}.txt",
                        use_container_width=True,
                    )
                with col_btn2:
                    st.code(full_output, language="markdown")  # 方便复制

            except Exception as e:
                st.error(f"❌ {str(e)[:300]}")

# ── 历史记录 ──────────────────────────────────────────────
if st.session_state.content_history:
    st.divider()
    st.subheader(f"📋 历史记录 ({len(st.session_state.content_history)} 篇)")

    for i, item in enumerate(reversed(st.session_state.content_history[-5:])):
        with st.expander(f"{item['type']} · {item['topic'][:40]} · {item['time']}"):
            st.markdown(item["content"])

footer("💡 Tip: Good topic + keywords + target audience = best AI-generated content.")
