"""
🤖 Universal AI Chatbot — 自动识别 API Key，支持多家 AI 供应商
支持: OpenAI / DeepSeek / Groq / Anthropic Claude / 自定义 OpenAI 兼容接口
"""
import streamlit as st
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from shared.style import inject_css, header, footer

load_dotenv()

# ── 供应商配置 ──────────────────────────────────────────────
# 格式: { 供应商名: { "prefix": 前缀列表, "base_url": API地址, "models": 可用模型 } }
PROVIDERS = {
    "OpenAI": {
        "prefixes": ["sk-proj-", "sk-admin-"],
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o4-mini", "o3-mini"],
    },
    "DeepSeek": {
        "prefixes": ["sk-"],
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Groq": {
        "prefixes": ["gsk_"],
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it", "qwen-2.5-32b"],
    },
    "Anthropic Claude": {
        "prefixes": ["sk-ant-"],
        "base_url": None,  # 用 Anthropic SDK，不用 OpenAI base_url
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8", "claude-fable-5"],
    },
    "Together AI": {
        "prefixes": ["tgp_"],
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
    },
    "OpenRouter": {
        "prefixes": ["sk-or-"],
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-6", "google/gemini-2.5-pro", "meta-llama/llama-4-maverick"],
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


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="🤖 Universal AI Chatbot",
    page_icon="🤖",
    layout="centered",
)
inject_css()
header("🤖", "Universal AI Chatbot", "Auto-detects API keys — OpenAI · DeepSeek · Groq · Claude · Custom")

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")

    # API Key
    api_key = os.getenv("API_KEY") or st.text_input(
        "API Key",
        type="password",
        placeholder="粘贴任何 API Key，自动识别供应商...",
        help="支持 OpenAI、DeepSeek、Groq、Claude、Together、OpenRouter 等",
    )

    # 自动检测
    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)

    if api_key and provider_name:
        st.success(f"✅ 识别为 **{provider_name}**")
    elif api_key and not provider_name:
        st.warning("⚠️ 未识别的 Key，将作为 OpenAI 兼容接口处理")

    # 手动选择供应商（覆盖自动检测）
    all_providers = list(PROVIDERS.keys()) + ["自定义 (OpenAI 兼容)"]
    manual_provider = st.selectbox(
        "供应商（自动检测 / 手动选择）",
        options=["🔍 自动检测"] + all_providers,
        index=0,
    )

    # 自定义 Base URL
    if manual_provider == "自定义 (OpenAI 兼容)":
        custom_base_url = st.text_input(
            "Base URL",
            placeholder="https://api.xxx.com/v1",
            help="OpenAI 兼容接口地址",
        )
    else:
        custom_base_url = None

    # 确定最终用的供应商配置
    if manual_provider.startswith("🔍"):
        # 自动模式
        if provider_config:
            active_config = provider_config
            active_name = provider_name
        else:
            active_config = {"base_url": None, "models": ["gpt-3.5-turbo"]}
            active_name = "未知供应商"
    elif manual_provider == "自定义 (OpenAI 兼容)":
        active_config = {"base_url": custom_base_url, "models": ["default-model"]}
        active_name = "自定义"
    else:
        active_config = PROVIDERS[manual_provider]
        active_name = manual_provider

    # 模型选择
    is_claude = active_name == "Anthropic Claude"
    default_idx = 0
    model = st.selectbox(
        "模型",
        options=active_config["models"],
        index=default_idx,
        help=f"当前供应商: {active_name}",
    )

    # Temperature
    temperature = st.slider(
        "创造性 (Temperature)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
    )

    # Max Tokens（Claude 需要）
    max_tokens = st.slider(
        "最大输出长度",
        min_value=256,
        max_value=8192,
        value=4096,
        step=256,
    )

    st.divider()
    st.caption(f"🟢 当前: **{active_name}** | 模型: `{model}`")

    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 对话历史 ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的 AI 助手。有什么我可以帮你的吗？"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 用户输入 ──────────────────────────────────────────────
if prompt := st.chat_input("在这里输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not api_key:
        st.error("⚠️ 请先在侧边栏输入 API Key")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner(f"思考中... ({active_name})"):
            try:
                if is_claude:
                    # ── Anthropic Claude（原生 SDK）─────
                    client = Anthropic(api_key=api_key)

                    # Claude 需要 system prompt 单独传递
                    system_msg = None
                    messages_for_claude = []
                    for m in st.session_state.messages:
                        if m["role"] == "system":
                            system_msg = m["content"]
                        elif m["role"] in ("user", "assistant"):
                            messages_for_claude.append({"role": m["role"], "content": m["content"]})

                    kwargs = {
                        "model": model,
                        "max_tokens": max_tokens,
                        "messages": messages_for_claude,
                        "stream": True,
                    }
                    if system_msg:
                        kwargs["system"] = system_msg

                    full_response = ""
                    placeholder = st.empty()
                    with client.messages.stream(**kwargs) as stream:
                        for text in stream.text_stream:
                            full_response += text
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                else:
                    # ── OpenAI 兼容（OpenAI / DeepSeek / Groq / 自定义）──
                    base_url = active_config.get("base_url")
                    if base_url:
                        client = OpenAI(api_key=api_key, base_url=base_url)
                    else:
                        client = OpenAI(api_key=api_key)

                    messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]

                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

                    full_response = ""
                    placeholder = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                # 保存回复
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ 出错了: {error_msg[:300]}")

                # 给出友好提示
                if "401" in error_msg or "Incorrect API key" in error_msg or "invalid" in error_msg.lower():
                    st.info("💡 API Key 无效。请检查 Key 是否正确，以及供应商是否选对。")
                elif "429" in error_msg or "quota" in error_msg.lower() or "insufficient" in error_msg.lower():
                    st.info("💡 余额不足或请求频率超限，请检查账户余额。")
                elif "Connection" in error_msg or "timed out" in error_msg.lower():
                    st.info("💡 网络连接失败，请检查网络或 Base URL。")
                elif "model" in error_msg.lower() and ("not found" in error_msg.lower() or "does not exist" in error_msg.lower()):
                    st.info(f"💡 模型 `{model}` 不存在或没有权限，请换一个模型试试。")
                else:
                    st.info("💡 常见原因：Key 无效、余额不足、模型不存在、或网络问题。")

                st.session_state.messages.pop()
                st.stop()
