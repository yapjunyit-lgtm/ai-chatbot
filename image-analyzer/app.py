"""
🔍 AI Image Analyzer & Prompt Generator — 上传图片，AI 分析视觉元素并生成图像提示词
支持: OpenAI GPT-4o / Google Gemini / DeepSeek / 自定义 OpenAI 兼容接口
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
        "vision": False,
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


def detect_provider(api_key: str):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


# ── CSS 注入 ────────────────────────────────────────────────
def inject_css():
    """注入自定义 CSS，匹配 HTML 版 UI 设计"""
    st.markdown("""
    <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── 基础重置 ── */
        .stApp {
            background: #f8fafc;
        }

        /* ── 隐藏 Streamlit 默认元素 ── */
        #MainMenu, footer, .stDeployButton { visibility: hidden; }

        /* ── 全局字体 ── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #1e293b;
        }

        /* ── 标题区 ── */
        .app-header {
            text-align: center;
            padding: 2.5rem 0 2rem;
        }
        .app-header .icon {
            font-size: 3rem;
            display: block;
            margin-bottom: 0.5rem;
        }
        .app-header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: #1e293b;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .app-header p {
            color: #64748b;
            font-size: 1.05rem;
            margin-top: 0.5rem;
        }

        /* ── 卡片 ── */
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* ── 上传区 ── */
        .upload-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748b;
            margin-bottom: 0.25rem;
        }
        .stFileUploader section {
            border: 2px dashed #e2e8f0 !important;
            border-radius: 16px !important;
            padding: 2rem !important;
            background: #ffffff !important;
            transition: all 250ms ease;
        }
        .stFileUploader section:hover {
            border-color: #6366f1 !important;
            background: #e0e7ff !important;
        }
        .stFileUploader section p {
            font-family: 'Inter', sans-serif !important;
            color: #64748b !important;
        }

        /* ── 按钮 ── */
        .stButton > button {
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            transition: all 150ms ease !important;
        }
        .stButton > button[kind="primary"] {
            background: #6366f1 !important;
            border: none !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #4f46e5 !important;
        }
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid #e2e8f0 !important;
            color: #1e293b !important;
        }
        .stButton > button[kind="secondary"]:hover {
            background: #f8fafc !important;
        }

        /* ── 标签胶囊 ── */
        .tag {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            color: white;
            margin: 2px;
        }
        .tag-visual { background: #3b82f6; }
        .tag-style  { background: #8b5cf6; }
        .tag-font   { background: #10b981; }

        /* ── 色彩色块 ── */
        .swatch {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin: 4px;
        }
        .swatch-circle {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 2px solid rgba(0,0,0,0.1);
            flex-shrink: 0;
        }
        .swatch-hex {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            font-weight: 500;
        }

        /* ── 光照卡片 ── */
        .lighting-card {
            padding: 1rem;
            background: #f8fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            font-size: 0.95rem;
            color: #1e293b;
            line-height: 1.6;
        }

        /* ── 提示词框 (深色) ── */
        .prompt-box {
            background: #1e293b;
            color: #e2e8f0;
            border-radius: 10px;
            padding: 1.25rem;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.9rem;
            line-height: 1.7;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .stTextArea textarea {
            background: #1e293b !important;
            color: #e2e8f0 !important;
            border: none !important;
            border-radius: 10px !important;
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            font-size: 0.9rem !important;
            line-height: 1.7 !important;
            padding: 1.25rem !important;
        }

        /* ── 搜索链接卡片 ── */
        .search-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            margin-bottom: 8px;
        }
        .search-term {
            font-size: 0.9rem;
            font-weight: 500;
            color: #1e293b;
        }
        .search-links a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border-radius: 6px;
            background: white;
            border: 1px solid #e2e8f0;
            color: #1e293b;
            text-decoration: none;
            font-size: 0.75rem;
            font-weight: 600;
            transition: all 150ms ease;
        }
        .search-links a:hover {
            background: #e0e7ff;
            border-color: #6366f1;
            color: #6366f1;
        }

        /* ── 安全提示 ── */
        .security-note {
            font-size: 0.8rem;
            color: #64748b;
            padding: 0.75rem;
            background: #f8fafc;
            border-radius: 6px;
            border-left: 3px solid #f59e0b;
            line-height: 1.5;
        }
        .security-note strong { color: #f59e0b; }

        /* ── 自定义字段 ── */
        .stTextInput input, .stTextArea textarea {
            border-radius: 10px !important;
            border-color: #e2e8f0 !important;
            font-family: 'Inter', sans-serif !important;
            transition: border-color 150ms ease !important;
        }
        .stTextInput input:focus, .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px #e0e7ff !important;
        }

        /* ── 侧边栏 ── */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] .stTextInput input {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
        }

        /* ── Expander ── */
        .stExpander details {
            border: none !important;
        }

        /* ── 页脚 ── */
        .app-footer {
            text-align: center;
            padding: 2rem 0;
            color: #64748b;
            font-size: 0.85rem;
        }

        /* ── spinner ── */
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ── AI 提示词 ──────────────────────────────────────────────
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


# ── API 调用 ────────────────────────────────────────────────
def call_openai_vision(api_key, base_url, model, image_data_url):
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze this image and return the JSON as specified."},
                {"type": "image_url", "image_url": {"url": image_data_url, "detail": "high"}}
            ]}
        ]
    )
    return json.loads(response.choices[0].message.content)


def call_gemini_vision(api_key, base_url, model, image_data):
    import requests
    if image_data.startswith("data:"):
        mime_match = image_data.split(";")[0].replace("data:", "")
        base64_str = image_data.split(",", 1)[1]
    else:
        mime_match = "image/png"
        base64_str = image_data

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [
            {"text": SYSTEM_PROMPT + "\n\nCRITICAL: Return ONLY valid JSON. Escape all double quotes inside strings. Keep aiImagePrompt as one line."},
            {"inlineData": {"mimeType": mime_match, "data": base64_str}}
        ]}],
        "generationConfig": {"maxOutputTokens": 1000}
    }
    resp = requests.post(url, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    return json.loads(raw)


def image_to_data_url(image: Image.Image) -> str:
    max_dim = 2048
    if image.width > max_dim or image.height > max_dim:
        ratio = max_dim / max(image.width, image.height)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── 页面初始化 ──────────────────────────────────────────────
st.set_page_config(page_title="🔍 Image Analyzer & Prompt Generator", page_icon="🔍", layout="wide")
inject_css()

# ── 页面标题 ──────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="icon">🔍</span>
    <h1>Image Analyzer &amp; Prompt Generator</h1>
    <p>Upload an image, and AI will break down its visual elements,
       generate a Midjourney/DALL-E prompt, and find similar styles.</p>
</div>
""", unsafe_allow_html=True)

# ── 侧边栏 ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Key & Provider")

    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key here...",
    )

    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)

    if api_key and provider_name:
        if provider_config.get("vision", True):
            st.success(f"✅ **{provider_name}**")
        else:
            st.error(f"⚠️ **{provider_name}** does NOT support vision!")

    all_providers = list(PROVIDERS.keys()) + ["Custom (OpenAI-compatible)"]
    manual_provider = st.selectbox("AI Provider", options=["🔍 Auto-Detect"] + all_providers, index=0)

    if manual_provider == "Custom (OpenAI-compatible)":
        custom_base_url = st.text_input("Base URL", placeholder="https://api.openai.com/v1")
    else:
        custom_base_url = None

    if manual_provider.startswith("🔍"):
        if provider_config:
            active_config = provider_config
            active_name = provider_name
        else:
            active_config = {"base_url": "https://api.openai.com/v1", "models": ["gpt-4o"], "vision": True}
            active_name = "OpenAI (default)"
    elif manual_provider == "Custom (OpenAI-compatible)":
        active_config = {"base_url": custom_base_url, "models": ["gpt-4o"], "vision": True}
        active_name = "Custom"
    else:
        active_config = PROVIDERS[manual_provider]
        active_name = manual_provider

    model = st.selectbox("Model", options=active_config.get("models", ["gpt-4o"]), index=0)

    st.divider()
    st.caption(f"🟢 Current: **{active_name}** · `{model}`")

    st.markdown("""
    <div class="security-note">
        <strong>🔒 Learning note:</strong> Your API key stays in your
        browser and is sent directly to your chosen AI provider. For a
        production app, use a backend to protect the key.
    </div>
    """, unsafe_allow_html=True)

# ── 主布局：左(上传) + 右(结果) ──────────────────────────
left, right = st.columns([1, 1.5], gap="large")

with left:
    # ── 上传卡片 ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop an image here, or click to browse — JPEG, PNG, GIF, WebP (max 20MB)",
        type=["jpg", "jpeg", "png", "gif", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("Image too large. Please use an image under 20MB.")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"✅ {uploaded_file.name} ({image.width}×{image.height})", use_container_width=True)

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please enter your API key in the sidebar first.")
                elif active_name == "DeepSeek":
                    st.error("DeepSeek does NOT support image/vision analysis. Switch to OpenAI or Gemini.")
                else:
                    with st.spinner(f"🔍 Analyzing with {active_name} ({model})... This may take 10-20 seconds."):
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
                            st.error(f"Analysis failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# ── 结果展示 ──────────────────────────────────────────────
if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]
    elements = result.get("elements", {})

    with right:
        # ── 🎨 视觉元素卡片 ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎨 Visual Elements</div>', unsafe_allow_html=True)

        # 元素标签
        visual_els = elements.get("visualElements", [])
        art_styles = elements.get("artStyles", [])
        fonts = elements.get("fonts", [])

        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin-bottom:4px;">Objects & Subjects</p>', unsafe_allow_html=True)
        if visual_els:
            tags_html = " ".join(
                f'<span class="tag tag-visual">{item}</span>' for item in visual_els
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:1rem 0 4px;">Art Styles</p>', unsafe_allow_html=True)
        if art_styles:
            tags_html = " ".join(
                f'<span class="tag tag-style">{item}</span>' for item in art_styles
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        # 色彩色块
        colors = elements.get("colors", [])
        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:1rem 0 4px;">Colors</p>', unsafe_allow_html=True)
        if colors:
            swatches_html = " ".join(
                f'<span class="swatch"><span class="swatch-circle" style="background:{hex_color};"></span>'
                f'<span class="swatch-hex">{hex_color}</span></span>'
                for hex_color in colors
            )
            st.markdown(swatches_html, unsafe_allow_html=True)

        # 光照
        lighting = elements.get("lighting", "")
        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:1rem 0 4px;">Lighting</p>', unsafe_allow_html=True)
        if lighting:
            st.markdown(f'<div class="lighting-card">{lighting}</div>', unsafe_allow_html=True)

        # 字体
        st.markdown('<p style="font-size:0.85rem;color:#64748b;margin:1rem 0 4px;">Fonts / Typography</p>', unsafe_allow_html=True)
        if fonts:
            tags_html = " ".join(
                f'<span class="tag tag-font">{item}</span>' for item in fonts
            )
            st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── ✍️ 提示词卡片 ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ AI Image Prompt</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#64748b;margin-bottom:12px;">'
            'Ready-to-use prompt for Midjourney, DALL-E 3, or Stable Diffusion. '
            '<strong>Edit the text below</strong> to customize it before copying.'
            '</p>',
            unsafe_allow_html=True
        )

        edited = st.text_area(
            "Prompt",
            value=st.session_state.get("original_prompt", result.get("aiImagePrompt", "")),
            height=140,
            key="editable_prompt",
            label_visibility="collapsed",
        )

        col_copy, _ = st.columns([1, 3])
        with col_copy:
            st.code(edited, language=None)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── 🎛️ 自定义卡片 ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎛️ Customize & Regenerate</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">'
            'Change any element below, then click <strong>Regenerate Prompt</strong> '
            'to create a new AI prompt with your changes applied.'
            '</p>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            custom_title = st.text_input("📝 Title / Subject",
                value=", ".join(elements.get("visualElements", [])[:3]))
            custom_colors = st.text_input("🌈 Colors",
                value=", ".join(elements.get("colors", [])))
            custom_mood = st.text_input("💫 Mood / Vibe",
                placeholder="e.g., peaceful and contemplative, energetic and bold")

        with c2:
            custom_style = st.text_input("🎨 Art Style",
                value=", ".join(elements.get("artStyles", [])))
            custom_lighting = st.text_input("💡 Lighting",
                value=elements.get("lighting", ""))
            custom_composition = st.text_input("📐 Composition",
                placeholder="e.g., centered subject, rule of thirds, wide angle")

        custom_extra = st.text_area("✨ Extra Instructions (freeform)",
            placeholder="Anything else you want to change... e.g., make it more cinematic, add fog, remove background text",
            height=68)

        regen_col, reset_col = st.columns([2, 1])
        with regen_col:
            if st.button("🔄 Regenerate Prompt with Changes", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please enter your API key first.")
                else:
                    with st.spinner("🔄 Regenerating prompt with your changes..."):
                        try:
                            changes = []
                            if custom_title: changes.append(f"- Subject/Title: {custom_title}")
                            if custom_style: changes.append(f"- Art Style: {custom_style}")
                            if custom_colors: changes.append(f"- Colors: {custom_colors}")
                            if custom_mood: changes.append(f"- Mood/Vibe: {custom_mood}")
                            if custom_lighting: changes.append(f"- Lighting: {custom_lighting}")
                            if custom_composition: changes.append(f"- Composition: {custom_composition}")
                            if custom_extra: changes.append(f"- Extra: {custom_extra}")

                            regen_prompt = f"""MODIFY THIS IMAGE PROMPT. Apply these changes and output the COMPLETE modified prompt.

=== ORIGINAL PROMPT ===
{edited}
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
                                    "contents": [{"parts": [{"text": "You are a text editor. Output the ENTIRE modified prompt — every word. No summary, no listing changes.\n\n" + regen_prompt}]}],
                                    "generationConfig": {"maxOutputTokens": 1000}
                                }
                                resp = requests.post(url, json=body, timeout=60)
                                resp.raise_for_status()
                                new_prompt = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                            else:
                                client = OpenAI(api_key=api_key, base_url=active_config.get("base_url", "https://api.openai.com/v1"))
                                regen_resp = client.chat.completions.create(
                                    model=model, max_tokens=1000,
                                    messages=[
                                        {"role": "system", "content": "You are a text editor. Output the COMPLETE modified prompt — every word. Never summarize or list changes. Just the full final prompt."},
                                        {"role": "user", "content": regen_prompt}
                                    ]
                                )
                                new_prompt = regen_resp.choices[0].message.content.strip()

                            st.session_state["original_prompt"] = new_prompt
                            st.rerun()

                        except Exception as e:
                            st.error(f"Regeneration failed: {str(e)}")

        with reset_col:
            if st.button("↻ Reset Fields", use_container_width=True):
                st.session_state["original_prompt"] = result.get("aiImagePrompt", "")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ── 🔗 搜索链接卡片 ──
        search_terms = result.get("searchTerms", [])
        if search_terms:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔗 Find Similar Styles</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.85rem;color:#64748b;margin-bottom:16px;">'
                'Click any link to search for visually similar images.</p>',
                unsafe_allow_html=True
            )

            for term in search_terms:
                google_url = f"https://www.google.com/search?tbm=isch&q={term}"
                pinterest_url = f"https://www.pinterest.com/search/pins/?q={term}"
                st.markdown(
                    f'<div class="search-item">'
                    f'  <span class="search-term">{term}</span>'
                    f'  <span class="search-links">'
                    f'    <a href="{google_url}" target="_blank" title="Google Images">G</a>'
                    f'    <a href="{pinterest_url}" target="_blank" title="Pinterest">P</a>'
                    f'  </span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

else:
    with right:
        st.info("👈 Upload an image and click **Analyze Image** to get started.")

# ── 页脚 ──
st.markdown("""
<div class="app-footer">
    Powered by AI Vision &nbsp;|&nbsp;
    Supports OpenAI GPT-4o &amp; Google Gemini &nbsp;|&nbsp;
    Your API key never leaves your browser
</div>
""", unsafe_allow_html=True)
