"""
🔍 AI Image Analyzer & Prompt Generator — 上传图片，AI 分析视觉元素并生成图像提示词
支持: OpenAI GPT-4o / DeepSeek / Gemini / 自定义 OpenAI 兼容接口
"""
import streamlit as st
import os
import base64
import json
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 供应商配置 ──────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "prefixes": ["sk-proj-", "sk-admin-"],
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "vision": True,
    },
    "DeepSeek": {
        "prefixes": ["sk-"],
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "vision": False,  # DeepSeek 不支持图片分析！
    },
    "Groq": {
        "prefixes": ["gsk_"],
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "vision": True,
    },
    "Together AI": {
        "prefixes": ["tgp_"],
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"],
        "vision": True,
    },
    "OpenRouter": {
        "prefixes": ["sk-or-"],
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro"],
        "vision": True,
    },
    "Google Gemini": {
        "prefixes": ["AIza"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "vision": True,
        "is_gemini": True,
    },
}


# ── Key 自动检测 ───────────────────────────────────────────
def detect_provider(api_key: str):
    """根据 API Key 前缀自动识别供应商"""
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


# ── AI 系统提示词 ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert visual analyst and AI art director. Analyze the uploaded image and return a single JSON object with exactly this structure:

{
  "elements": {
    "visualElements": ["list of 5-10 key objects, subjects, or visual components visible in the image"],
    "artStyles": ["list of 2-4 art movements, illustration styles, or visual aesthetics this image resembles"],
    "colors": ["list of 4-6 dominant hex color codes found in the image, e.g. #3B82F6"],
    "lighting": "one sentence describing the lighting quality, direction, and mood",
    "fonts": ["list of 2-4 font styles or typeface categories that match the text or aesthetic, e.g. serif, sans-serif, handwritten, display"]
  },
  "aiImagePrompt": "a single detailed paragraph (50-100 words) that could be used as a prompt for Midjourney, DALL-E, or Stable Diffusion to generate a similar image. Include subject, style, composition, lighting, color palette, and mood.",
  "searchTerms": ["5-8 specific keyword phrases for finding images with a similar style on stock photo sites, Google Images, or Pinterest"]
}

Rules:
- Return ONLY the JSON object, no markdown fences, no explanation.
- Every array must have at least 2 items.
- Hex color codes must be valid 6-character hex prefixed with #.
- The aiImagePrompt must be a single string (not an array).
- Be specific and concrete. Avoid vague terms like "nice" or "beautiful."
- Escape ALL double quotes inside string values with backslash."""


# ── 调用 OpenAI 兼容 API ───────────────────────────────────
def call_openai_vision(api_key, base_url, model, image_data_url):
    """发送图片到 OpenAI 兼容 API 进行分析"""
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this image and return the JSON as specified."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    )

    return json.loads(response.choices[0].message.content)


# ── 调用 Gemini API ─────────────────────────────────────────
def call_gemini_vision(api_key, base_url, model, image_data):
    """发送图片到 Gemini API 进行分析（使用 requests 而非 SDK）"""
    import requests

    # 提取 MIME 和 base64 数据
    if image_data.startswith("data:"):
        mime_match = image_data.split(";")[0].replace("data:", "")
        base64_str = image_data.split(",", 1)[1]
    else:
        mime_match = "image/png"
        base64_str = image_data

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    body = {
        "contents": [{
            "parts": [
                {"text": SYSTEM_PROMPT + "\n\nCRITICAL: Return ONLY valid JSON. Escape all double quotes inside strings. Keep aiImagePrompt as one line."},
                {"inlineData": {"mimeType": mime_match, "data": base64_str}}
            ]
        }],
        "generationConfig": {"maxOutputTokens": 1000}
    }

    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    raw = data["candidates"][0]["content"]["parts"][0]["text"]
    # 去除可能的 markdown 代码块
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


# ── 图片转 base64 ──────────────────────────────────────────
def image_to_data_url(image: Image.Image) -> str:
    """将 PIL Image 转为 base64 data URL"""
    # 限制尺寸以节省 token
    max_dim = 2048
    if image.width > max_dim or image.height > max_dim:
        ratio = max_dim / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 Image Analyzer & Prompt Generator",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Image Analyzer & Prompt Generator")
st.caption("上传图片 → AI 分析视觉元素 → 生成 Midjourney/DALL-E 提示词 → 查找相似风格")

# ── 侧边栏：API 设置 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API 设置")

    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or st.text_input(
        "API Key",
        type="password",
        placeholder="粘贴 API Key，自动识别供应商...",
        help="支持 OpenAI、DeepSeek、Groq、Gemini、Together、OpenRouter 等",
    )

    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)

    if api_key and provider_name:
        if provider_config.get("vision", True):
            st.success(f"✅ 识别为 **{provider_name}**（支持图片分析）")
        else:
            st.error(f"⚠️ **{provider_name}** 不支持图片分析！请使用 OpenAI 或 Gemini。")

    all_providers = list(PROVIDERS.keys()) + ["自定义 (OpenAI 兼容)"]
    manual_provider = st.selectbox(
        "供应商",
        options=["🔍 自动检测"] + all_providers,
        index=0,
    )

    if manual_provider == "自定义 (OpenAI 兼容)":
        custom_base_url = st.text_input("Base URL", placeholder="https://api.xxx.com/v1")
    else:
        custom_base_url = None

    # 确定最终配置
    if manual_provider.startswith("🔍"):
        if provider_config:
            active_config = provider_config
            active_name = provider_name
        else:
            active_config = {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o"], "vision": True}
            active_name = "OpenAI（默认）"
    elif manual_provider == "自定义 (OpenAI 兼容)":
        active_config = {"base_url": custom_base_url, "models": ["gpt-4o"], "vision": True}
        active_name = "自定义"
    else:
        active_config = PROVIDERS[manual_provider]
        active_name = manual_provider

    model = st.selectbox(
        "模型",
        options=active_config.get("models", ["gpt-4o"]),
        index=0,
    )

    st.divider()
    st.caption(f"🟢 当前: **{active_name}** | `{model}`")
    st.caption("💡 推荐: **OpenAI GPT-4o** 或 **Google Gemini**（免费额度）")

# ── 主界面：图片上传 ──────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📤 上传图片")
    uploaded_file = st.file_uploader(
        "拖放图片到这里，或点击上传",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        help="支持 JPG、PNG、GIF、WebP，最大 20MB",
    )

    if uploaded_file:
        # 检查文件大小
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("图片太大！请使用小于 20MB 的图片。")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"✅ {uploaded_file.name} ({image.width}×{image.height})", use_container_width=True)

            if st.button("🔍 分析图片", type="primary", use_container_width=True):
                if not api_key:
                    st.error("⚠️ 请先在侧边栏输入 API Key")
                elif active_name == "DeepSeek":
                    st.error("⚠️ DeepSeek 不支持图片分析！请在侧边栏切换到 OpenAI 或 Gemini。")
                else:
                    with st.spinner(f"🔍 正在用 {active_name} ({model}) 分析图片..."):
                        try:
                            data_url = image_to_data_url(image)

                            if active_config.get("is_gemini"):
                                result = call_gemini_vision(api_key, active_config["base_url"], model, data_url)
                            else:
                                result = call_openai_vision(api_key, active_config.get("base_url", "https://api.openai.com/v1"), model, data_url)

                            st.session_state["analysis_result"] = result
                            st.session_state["original_prompt"] = result.get("aiImagePrompt", "")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ 分析失败: {str(e)}")

# ── 结果展示 ──────────────────────────────────────────────
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    elements = result.get("elements", {})

    with col2:
        st.subheader("🎨 分析结果")

        # ── 视觉元素标签 ──
        with st.expander("🔍 视觉元素 & 艺术风格", expanded=True):
            tab1, tab2, tab3 = st.tabs(["🖼 元素", "🎨 风格", "🔤 字体"])

            with tab1:
                visual_els = elements.get("visualElements", [])
                if visual_els:
                    cols = st.columns(min(len(visual_els), 4))
                    for i, item in enumerate(visual_els):
                        cols[i % 4].markdown(
                            f"<span style='background:#3b82f6;color:white;padding:4px 12px;"
                            f"border-radius:20px;font-size:0.8rem;display:inline-block;margin:2px;'>{item}</span>",
                            unsafe_allow_html=True
                        )

            with tab2:
                styles = elements.get("artStyles", [])
                if styles:
                    for s in styles:
                        st.markdown(
                            f"<span style='background:#8b5cf6;color:white;padding:4px 12px;"
                            f"border-radius:20px;font-size:0.8rem;display:inline-block;margin:2px;'>{s}</span>",
                            unsafe_allow_html=True
                        )

            with tab3:
                fonts = elements.get("fonts", [])
                if fonts:
                    for f in fonts:
                        st.markdown(
                            f"<span style='background:#10b981;color:white;padding:4px 12px;"
                            f"border-radius:20px;font-size:0.8rem;display:inline-block;margin:2px;'>{f}</span>",
                            unsafe_allow_html=True
                        )

        # ── 色彩 ──
        colors = elements.get("colors", [])
        if colors:
            with st.expander("🌈 色彩方案", expanded=True):
                cols = st.columns(len(colors))
                for i, hex_color in enumerate(colors):
                    with cols[i]:
                        st.markdown(
                            f"<div style='width:60px;height:60px;background:{hex_color};"
                            f"border-radius:12px;border:2px solid #ddd;margin:0 auto;'></div>"
                            f"<p style='text-align:center;font-family:monospace;font-size:0.8rem;'>{hex_color}</p>",
                            unsafe_allow_html=True
                        )

        # ── 光照 ──
        lighting = elements.get("lighting", "")
        if lighting:
            with st.expander("💡 光照描述"):
                st.info(lighting)

        # ── AI 提示词 ──
        st.subheader("✍️ AI 图像提示词")

        # 可编辑的提示词
        edited_prompt = st.text_area(
            "可直接编辑，然后点击下方按钮复制或重新生成",
            value=st.session_state.get("original_prompt", result.get("aiImagePrompt", "")),
            height=120,
            key="editable_prompt",
            help="你可以直接修改这段提示词，修改后点击「复制」按钮即可复制修改后的版本。",
        )

        col_copy, col_regen = st.columns(2)
        with col_copy:
            st.code(edited_prompt, language=None)  # st.code 自带复制按钮！
        with col_regen:
            st.caption("💡 点击上方 textarea 直接编辑，然后点击代码块的 📋 复制按钮。")

        # ── 自定义 & 重新生成 ──
        st.subheader("🎛️ 自定义 & 重新生成")

        with st.expander("修改元素并重新生成提示词", expanded=False):
            st.caption("修改以下字段，AI 会将你的修改融合进原提示词并生成新的完整版本。")

            c1, c2 = st.columns(2)
            with c1:
                custom_title = st.text_input("📝 主题/内容",
                    value=", ".join(elements.get("visualElements", [])[:3]),
                    help="更改图片的主题或主要内容")
                custom_colors = st.text_input("🌈 色彩",
                    value=", ".join(elements.get("colors", [])),
                    help="修改颜色方案")
                custom_mood = st.text_input("💫 氛围/情绪",
                    placeholder="例如: 宁静祥和、充满活力、神秘莫测...")

            with c2:
                custom_style = st.text_input("🎨 艺术风格",
                    value=", ".join(elements.get("artStyles", [])),
                    help="更改艺术风格")
                custom_lighting = st.text_input("💡 光照",
                    value=elements.get("lighting", ""),
                    help="修改光照描述")
                custom_composition = st.text_input("📐 构图",
                    placeholder="例如: 居中构图、三分法、广角...")

            custom_extra = st.text_area("✨ 额外要求（自由描述）",
                placeholder="例如: 让画面更像电影海报、去除背景文字、改用竖版构图...",
                height=60)

            if st.button("🔄 重新生成提示词", type="primary", use_container_width=True):
                if not api_key:
                    st.error("⚠️ 请先在侧边栏输入 API Key")
                else:
                    with st.spinner("🔄 正在融合你的修改..."):
                        try:
                            # 构建修改请求
                            changes = []
                            if custom_title: changes.append(f"- 主题/内容: {custom_title}")
                            if custom_style: changes.append(f"- 艺术风格: {custom_style}")
                            if custom_colors: changes.append(f"- 色彩: {custom_colors}")
                            if custom_mood: changes.append(f"- 氛围: {custom_mood}")
                            if custom_lighting: changes.append(f"- 光照: {custom_lighting}")
                            if custom_composition: changes.append(f"- 构图: {custom_composition}")
                            if custom_extra: changes.append(f"- 额外: {custom_extra}")

                            regen_prompt = f"""MODIFY THIS IMAGE PROMPT. Apply these changes and output the COMPLETE modified prompt.

=== ORIGINAL PROMPT ===
{edited_prompt}
=== END ORIGINAL ===

=== CHANGES ===
{chr(10).join(changes)}
=== END CHANGES ===

IMPORTANT: Output the FULL modified prompt — every word from start to finish.
No summary. No "Here is". No bullet points. Just the complete final prompt."""

                            if active_config.get("is_gemini"):
                                import requests
                                url = f"{active_config['base_url']}/models/{model}:generateContent?key={api_key}"
                                body = {
                                    "contents": [{"parts": [{"text": "You are a text editor. Take the prompt and directly modify it. Output the ENTIRE modified prompt — every word. No summary, no listing changes.\n\n" + regen_prompt}]}],
                                    "generationConfig": {"maxOutputTokens": 1000}
                                }
                                resp = requests.post(url, json=body, timeout=60)
                                resp.raise_for_status()
                                new_prompt = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                            else:
                                client = OpenAI(api_key=api_key, base_url=active_config.get("base_url", "https://api.openai.com/v1"))
                                regen_resp = client.chat.completions.create(
                                    model=model,
                                    max_tokens=1000,
                                    messages=[
                                        {"role": "system", "content": "You are a text editor. Output the COMPLETE modified prompt — every word. Never summarize or list changes. Just the full final prompt."},
                                        {"role": "user", "content": regen_prompt}
                                    ]
                                )
                                new_prompt = regen_resp.choices[0].message.content.strip()

                            st.session_state["original_prompt"] = new_prompt
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ 重新生成失败: {str(e)}")

        # ── 搜索链接 ──
        search_terms = result.get("searchTerms", [])
        if search_terms:
            with st.expander("🔗 查找相似风格", expanded=True):
                st.caption("点击链接在新标签页中打开搜索结果。")
                for term in search_terms:
                    google_url = f"https://www.google.com/search?tbm=isch&q={term}"
                    pinterest_url = f"https://www.pinterest.com/search/pins/?q={term}"
                    st.markdown(
                        f"**{term}** &nbsp; "
                        f"[🔍 Google Images]({google_url}) &nbsp;|&nbsp; "
                        f"[📌 Pinterest]({pinterest_url})",
                        unsafe_allow_html=True
                    )

# ── 初始状态 ──
else:
    with col2:
        st.info("👈 上传一张图片，然后点击「分析图片」开始分析。")
        st.markdown("""
        ### 功能说明
        1. **上传图片** — 拖放或点击上传
        2. **AI 分析** — 识别视觉元素、风格、色彩、字体、光照
        3. **生成提示词** — 可直接用于 Midjourney / DALL-E / Stable Diffusion
        4. **自定义修改** — 修改元素后 AI 重新融合生成
        5. **查找相似** — 一键在 Google Images 和 Pinterest 搜索相似风格
        """)

# ── 页脚 ──
st.divider()
st.caption("🔍 Image Analyzer | 支持 OpenAI GPT-4o · Google Gemini · 多供应商 | 你的 API Key 只保存在本地")
