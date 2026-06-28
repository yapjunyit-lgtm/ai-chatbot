"""
📄 AI PDF 分析器 — 上传 PDF，一键摘要，智能问答
支持: OpenAI / DeepSeek / Groq / Claude / 自定义接口
"""
import streamlit as st
import os
from io import BytesIO
from pypdf import PdfReader
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

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


def detect_provider(api_key: str):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


def call_ai(messages: list, api_key: str, model: str, temperature: float, max_tokens: int, is_claude: bool, base_url: str = None):
    """统一的 AI 调用接口，支持 Claude 和 OpenAI 兼容"""
    full_response = ""

    if is_claude:
        client = Anthropic(api_key=api_key)
        system_msg = None
        claude_msgs = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            elif m["role"] in ("user", "assistant"):
                claude_msgs.append({"role": m["role"], "content": m["content"]})

        kwargs = {"model": model, "max_tokens": max_tokens, "messages": claude_msgs, "stream": True}
        if system_msg:
            kwargs["system"] = system_msg

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                full_response += text
                yield text
    else:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="📄 AI PDF 分析器",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI PDF 分析器")
st.caption("上传 PDF → 一键摘要 → 智能问答。律师、HR、研究员必备。")

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API 设置")

    api_key = os.getenv("API_KEY") or st.text_input(
        "API Key",
        type="password",
        placeholder="粘贴任何 API Key...",
    )

    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)

    if api_key and provider_name:
        st.success(f"✅ **{provider_name}**")
    elif api_key:
        st.warning("⚠️ 未识别，作为 OpenAI 兼容处理")

    all_providers = list(PROVIDERS.keys()) + ["自定义 (OpenAI 兼容)"]
    manual_provider = st.selectbox("供应商", options=["🔍 自动检测"] + all_providers, index=0)

    if manual_provider == "自定义 (OpenAI 兼容)":
        custom_base_url = st.text_input("Base URL", placeholder="https://api.xxx.com/v1")
    else:
        custom_base_url = None

    # 确定供应商
    if manual_provider.startswith("🔍"):
        if provider_config:
            active_config = provider_config
            active_name = provider_name
        else:
            active_config = {"base_url": None, "models": ["gpt-3.5-turbo"]}
            active_name = "未知"
    elif manual_provider == "自定义 (OpenAI 兼容)":
        active_config = {"base_url": custom_base_url, "models": ["default-model"]}
        active_name = "自定义"
    else:
        active_config = PROVIDERS[manual_provider]
        active_name = manual_provider

    is_claude = active_name == "Anthropic Claude"

    model = st.selectbox("模型", options=active_config["models"], index=0)

    st.divider()
    st.caption(f"🟢 当前: **{active_name}** | `{model}`")

    st.divider()
    st.header("📄 文档操作")

    summarize_btn = st.button("📝 一键摘要", use_container_width=True, type="primary")
    generate_questions_btn = st.button("❓ 生成关键问题", use_container_width=True)

# ── 主区域：上传 & 预览 ───────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 上传 PDF")
    uploaded_files = st.file_uploader(
        "选择一个或多个 PDF 文件",
        type="pdf",
        accept_multiple_files=True,
        help="支持中文、英文 PDF，最大 200MB",
    )

    if uploaded_files:
        all_texts = {}
        for f in uploaded_files:
            with st.spinner(f"📖 读取 {f.name}..."):
                try:
                    reader = PdfReader(BytesIO(f.getvalue()))
                    text_parts = []
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            # 按段落分割，标注页码
                            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                            tagged_paragraphs = []
                            for j, para in enumerate(paragraphs):
                                tagged_paragraphs.append(f"[第{i+1}页·第{j+1}段] {para}")
                            text_parts.append("\n".join(tagged_paragraphs))

                    full_text = "\n\n".join(text_parts)
                    all_texts[f.name] = {
                        "text": full_text,
                        "pages": len(reader.pages),
                        "words": len(full_text.split()),
                        "chars": len(full_text),
                    }
                except Exception as e:
                    st.error(f"❌ 无法读取 {f.name}: {e}")

        # 存入 session
        st.session_state.pdf_data = all_texts

        # 显示文档信息卡片
        st.subheader("📊 文档概览")
        total_words = sum(d["words"] for d in all_texts.values())
        total_pages = sum(d["pages"] for d in all_texts.values())

        metric_cols = st.columns(3)
        metric_cols[0].metric("文件数", len(all_texts))
        metric_cols[1].metric("总页数", total_pages)
        metric_cols[2].metric("总字数", f"{total_words:,}")

        for name, data in all_texts.items():
            with st.expander(f"📄 {name} — {data['pages']} 页 · {data['words']:,} 词"):
                preview = data["text"][:2000]
                st.text_area("文本预览", preview + ("..." if len(data["text"]) > 2000 else ""), height=200, key=f"preview_{name}")
    else:
        st.info("👆 请上传 PDF 文件开始分析")
        st.session_state.pdf_data = {}

# ── 右侧：AI 分析区 ────────────────────────────────────────
with col2:
    st.subheader("🤖 AI 分析")

    if not st.session_state.get("pdf_data"):
        st.info("👈 先上传 PDF 文件")
    else:
        pdf_texts = st.session_state.pdf_data

        # ── Tab 1: AI 摘要 ──
        tab1, tab2 = st.tabs(["📝 摘要", "💬 问答"])

        with tab1:
            if summarize_btn:
                if not api_key:
                    st.error("⚠️ 请先在侧边栏输入 API Key")
                else:
                    for name, data in pdf_texts.items():
                        with st.expander(f"📄 {name} 摘要", expanded=True):
                            placeholder = st.empty()
                            full = ""
                            try:
                                system_prompt = """你是一个专业的文档分析师。请用中文总结以下文档的核心内容。

文本中的引用标记格式：[第X页·第Y段] 表示第X页第Y段。

请按以下结构输出：

### 📄 文档类型和主题
（一句话概括）

### 🔑 关键要点
1. 要点一 （第X页，第Y段）
2. 要点二 （第X页，第Y段）
3. 要点三 （第X页，第Y段）
...（3-5 个）

### 📌 主要结论
（1-2 句话总结，附带页面引用）

**每个要点必须附带具体的页码和段落引用。**"""
                                text_to_summarize = data["text"][:10000]  # 增加长度容纳标记

                                messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"请总结以下文档内容：\n\n{text_to_summarize}"},
                                ]

                                for token in call_ai(messages, api_key, model, 0.3, 2048, is_claude, active_config.get("base_url")):
                                    full += token
                                    placeholder.markdown(full + "▌")

                                placeholder.markdown(full)
                                # 保存摘要
                                pdf_texts[name]["summary"] = full

                            except Exception as e:
                                st.error(f"❌ {str(e)[:200]}")

        # ── Tab 2: 智能问答 ──
        with tab2:
            st.caption("对文档内容提问，AI 会基于文档回答")

            # 初始化问答历史
            if "qa_history" not in st.session_state:
                st.session_state.qa_history = []

            # 显示历史
            for qa in st.session_state.qa_history:
                with st.chat_message("user"):
                    st.markdown(f"**{qa['q']}**")
                with st.chat_message("assistant"):
                    st.markdown(qa["a"])

            # 提问输入
            question = st.chat_input("对文档提问...")

            if question:
                if not api_key:
                    st.error("⚠️ 请先在侧边栏输入 API Key")
                else:
                    with st.chat_message("user"):
                        st.markdown(f"**{question}**")

                    with st.chat_message("assistant"):
                        placeholder = st.empty()
                        full = ""

                        try:
                            # 构建上下文（取所有文档文本）
                            context_parts = []
                            for name, data in pdf_texts.items():
                                context_parts.append(f"【文档：{name}】\n{data['text'][:8000]}")

                            context = "\n\n---\n\n".join(context_parts)

                            system_prompt = """你是一个精确的文档分析助手。请严格基于提供的文档内容回答问题。

文本中的引用标记格式：[第X页·第Y段] 表示第X页第Y段。

回答规则：
1. 每个观点/事实必须附带具体的引用来源，格式：（第X页，第Y段）
2. 如果文档中没有相关信息，请明确说明'文档中未提及此内容'
3. 如有多个出处，列出所有相关引用
4. 用中文回答，清晰分段"""
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"参考文档内容：\n\n{context}\n\n---\n\n用户问题：{question}\n\n请基于以上文档回答。"},
                            ]

                            for token in call_ai(messages, api_key, model, 0.5, 3072, is_claude, active_config.get("base_url")):
                                full += token
                                placeholder.markdown(full + "▌")

                            placeholder.markdown(full)

                        except Exception as e:
                            full = f"❌ 出错了: {str(e)[:200]}"
                            st.error(full)

                    st.session_state.qa_history.append({"q": question, "a": full})

        # ── 生成关键问题 ──
        if generate_questions_btn:
            if not api_key:
                st.error("⚠️ 请先在侧边栏输入 API Key")
            else:
                with st.spinner("🤔 生成关键问题..."):
                    try:
                        first_text = list(pdf_texts.values())[0]["text"][:6000]
                        system_prompt = "你是一个专业的文档分析师。基于文档内容，生成 5-8 个读者可能会问的关键问题。用中文，每个问题一行，以编号开头。"
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"基于以下文档生成关键问题：\n\n{first_text}"},
                        ]
                        full = "".join(list(call_ai(messages, api_key, model, 0.5, 1024, is_claude, active_config.get("base_url"))))
                        st.success("### 💡 建议问题")
                        st.markdown(full)
                    except Exception as e:
                        st.error(f"❌ {str(e)[:200]}")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("💡 提示：上传 PDF 后，先点「一键摘要」快速了解内容，再用「问答」深入提问。")
