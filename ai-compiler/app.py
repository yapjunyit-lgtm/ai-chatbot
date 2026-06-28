"""
⚡ AI Compiler v3 — 一个 Prompt 同时发给多个 AI，并排对比，一键编译
💰 极致省钱：每家默认最便宜模型，旁边标价格，🆓 免费模型优先
"""
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── 供应商配置 ──────────────────────────────────────────────
# price_tag: 💰 = 付费但便宜, 🆓 = 免费层可用
PROVIDERS = {
    "DeepSeek": {
        "prefixes": ["sk-"],
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "price": "💰 ¥0.14/百万token",
        "free_tier": "注册送额度",
        "get_key_url": "https://platform.deepseek.com/api_keys",
    },
    "Google Gemini": {
        "prefixes": ["AIza"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        "price": "🆓 免费层",
        "free_tier": "每分钟60次免费",
        "get_key_url": "https://aistudio.google.com/apikey",
    },
    "Groq": {
        "prefixes": ["gsk_"],
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "qwen-2.5-32b"],
        "price": "🆓 免费层",
        "free_tier": "速率限制",
        "get_key_url": "https://console.groq.com/keys",
    },
    "OpenAI": {
        "prefixes": ["sk-proj-", "sk-admin-"],
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"],
        "price": "💰 $0.15/百万token",
        "free_tier": "新用户送$5",
        "get_key_url": "https://platform.openai.com/api-keys",
    },
    "Mistral": {
        "prefixes": ["ms-"],
        "base_url": "https://api.mistral.ai/v1",
        "models": ["mistral-small-latest", "mistral-large-latest", "pixtral-large-latest"],
        "price": "🆓 免费层",
        "free_tier": "每秒1次免费",
        "get_key_url": "https://console.mistral.ai/api-keys/",
    },
    "Together AI": {
        "prefixes": ["tgp_"],
        "base_url": "https://api.together.xyz/v1",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1", "Qwen/Qwen2.5-72B-Instruct-Turbo"],
        "price": "💰 $0.10/百万token",
        "free_tier": "注册送$5",
        "get_key_url": "https://api.together.ai/settings/api-keys",
    },
    "OpenRouter": {
        "prefixes": ["sk-or-"],
        "base_url": "https://openrouter.ai/api/v1",
        "models": ["openai/gpt-4o-mini", "google/gemini-2.0-flash", "deepseek/deepseek-chat"],
        "price": "💰 聚合加价",
        "free_tier": "无免费层",
        "get_key_url": "https://openrouter.ai/keys",
    },
}

# ── 默认槽位（最便宜方案）───────────────────────────────────
DEFAULT_SLOTS = [
    {"provider": "DeepSeek", "model": "deepseek-chat"},
    {"provider": "Google Gemini", "model": "gemini-2.5-flash-lite"},
    {"provider": "Groq", "model": "llama-3.1-8b-instant"},
]


def detect_provider(api_key: str):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


def call_ai(messages: list, api_key: str, model: str, temperature: float, max_tokens: int, base_url: str):
    """流式调用 AI — 全部走 OpenAI 兼容接口"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="⚡ AI Compiler", page_icon="⚡", layout="wide")

st.title("⚡ AI Compiler")
st.caption("一个 Prompt → 多个 AI 同时回答 → 并排对比 → 一键编译最优答案")

# ── 初始化 session ─────────────────────────────────────────
if "slots" not in st.session_state:
    st.session_state.slots = [dict(s) for s in DEFAULT_SLOTS]

if "slot_keys" not in st.session_state:
    st.session_state.slot_keys = {}

if "slot_responses" not in st.session_state:
    st.session_state.slot_responses = {}

if "compile_response" not in st.session_state:
    st.session_state.compile_response = ""

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 全局设置")

    temperature = st.slider("创造性 (Temperature)", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("最大输出长度", 256, 8192, 2048, 128)

    st.divider()

    # ── 槽位配置 ──
    st.header("📊 槽位配置")
    slots_to_remove = []

    for idx, slot in enumerate(st.session_state.slots):
        provider = slot["provider"]
        config = PROVIDERS.get(provider, {})
        price = config.get("price", "")
        get_key = config.get("get_key_url", "#")

        with st.expander(
            f"🔹 Slot {idx + 1}: {provider} — `{slot['model']}`  {price}",
            expanded=len(st.session_state.slots) <= 3,
        ):
            # ── 供应商选择 ──
            provider_names = list(PROVIDERS.keys())
            current_pidx = provider_names.index(provider) if provider in provider_names else 0
            new_provider = st.selectbox(
                "供应商",
                options=provider_names,
                index=current_pidx,
                key=f"slot_provider_{idx}",
            )
            slot["provider"] = new_provider
            new_config = PROVIDERS[new_provider]

            # ── API Key 输入 ──
            key_col1, key_col2 = st.columns([3, 1])
            with key_col1:
                default_key = st.session_state.slot_keys.get(idx, "")
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=default_key,
                    placeholder=f"粘贴 {new_provider} Key...",
                    key=f"slot_key_{idx}",
                )
                if api_key:
                    st.session_state.slot_keys[idx] = api_key

            with key_col2:
                st.markdown(f"""<a href="{new_config.get('get_key_url', '#')}" target="_blank">
                    <div style="font-size:12px; color:#667eea; margin-top:28px;">🔗 获取 Key</div>
                </a>""", unsafe_allow_html=True)

            # ── 价格信息 ──
            st.caption(f"{new_config.get('price', '')}  |  {new_config.get('free_tier', '')}")

            # ── 模型选择 ──
            models = new_config["models"]
            try:
                current_midx = models.index(slot["model"]) if slot["model"] in models else 0
            except ValueError:
                current_midx = 0
            slot["model"] = st.selectbox(
                "模型",
                options=models,
                index=current_midx,
                key=f"slot_model_{idx}",
            )

            # ── 检测 Key ──
            if api_key:
                detected, _ = detect_provider(api_key)
                if detected and detected != new_provider:
                    st.warning(f"⚠️ Key 看起来是 {detected} 的，但你选了 {new_provider}")

    # 处理移除
    for idx in reversed(slots_to_remove):
        if len(st.session_state.slots) > 1:
            st.session_state.slots.pop(idx)
            st.rerun()

    # 添加槽位
    if len(st.session_state.slots) < 8:
        if st.button("➕ 添加槽位", use_container_width=True):
            st.session_state.slots.append({"provider": "OpenAI", "model": "gpt-4o-mini"})
            st.rerun()

    st.caption(f"共 **{len(st.session_state.slots)}** 个槽位（最多 8 个）")

    st.divider()

    # ── 编译设置 ──
    st.header("🎯 编译合成")
    compile_slot_idx = st.selectbox(
        "用哪个槽位做编译？",
        options=list(range(len(st.session_state.slots))),
        format_func=lambda i: f"Slot {i+1}: {st.session_state.slots[i]['provider']}",
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 清空回复", use_container_width=True):
            st.session_state.slot_responses = {}
            st.session_state.compile_response = ""
            st.rerun()
    with col2:
        if st.button("🗑 清空 Key", use_container_width=True):
            st.session_state.slot_keys = {}
            st.rerun()

    # ── 免费 Key 获取指引 ──
    st.divider()
    st.subheader("🆓 免费 Key 获取")
    for name, config in PROVIDERS.items():
        if "🆓" in config.get("price", ""):
            st.markdown(f"🔗 [{name}]({config.get('get_key_url', '#')}) — {config.get('free_tier', '')}")

# ── 主区域 ────────────────────────────────────────────────
st.subheader("💬 输入 Prompt")
prompt = st.chat_input("输入你的问题，所有 AI 会同时回答...")

# ── 结果区域：N 列并排 ──
num_slots = len(st.session_state.slots)
if num_slots > 0:
    st.subheader(f"📊 {num_slots} 个 AI 对比")
    result_cols = st.columns(num_slots)

    for idx, (col, slot) in enumerate(zip(result_cols, st.session_state.slots)):
        with col:
            provider = slot["provider"]
            model = slot["model"]
            config = PROVIDERS.get(provider, {})
            api_key = st.session_state.slot_keys.get(idx, "")
            price = config.get("price", "")

            st.markdown(f"**Slot {idx + 1}**")
            st.caption(f"{provider} · `{model}`  {price}")

            placeholder = st.empty()

            if not api_key:
                placeholder.warning("🔑 需要 API Key")
            elif prompt:
                slot_key = f"{idx}_{prompt}_{model}"
                if slot_key not in st.session_state.slot_responses:
                    with st.spinner("⏳ 思考中..."):
                        try:
                            messages = [{"role": "user", "content": prompt}]
                            full = ""
                            for token in call_ai(
                                messages, api_key, model, temperature, max_tokens,
                                config.get("base_url", ""),
                            ):
                                full += token
                                placeholder.markdown(full + "▌")
                            placeholder.markdown(full)
                            st.session_state.slot_responses[slot_key] = full
                        except Exception as e:
                            error_msg = str(e)[:200]
                            placeholder.error(f"❌ {error_msg}")
                            st.session_state.slot_responses[slot_key] = f"**错误**: {error_msg}"
                else:
                    placeholder.markdown(st.session_state.slot_responses[slot_key])
            else:
                placeholder.info("等待输入...")

# ── 编译合成 ──
if prompt and num_slots > 1:
    # 检查至少 2 个槽位有回复
    ready_slots = []
    for idx in range(num_slots):
        slot = st.session_state.slots[idx]
        slot_key = f"{idx}_{prompt}_{slot['model']}"
        if slot_key in st.session_state.slot_responses:
            ready_slots.append((idx, slot, slot_key))

    if len(ready_slots) >= 2:
        st.divider()
        st.subheader("🎯 编译合成")

        compile_model = st.session_state.slots[compile_slot_idx]["model"]
        compile_provider = st.session_state.slots[compile_slot_idx]["provider"]
        compile_config = PROVIDERS.get(compile_provider, {})
        compile_key = st.session_state.slot_keys.get(compile_slot_idx, "")

        compile_btn = st.button(
            f"🚀 用 Slot {compile_slot_idx + 1} (`{compile_model}`) 编译合成",
            type="primary", use_container_width=True,
            disabled=not compile_key,
        )

        if compile_btn and compile_key:
            # 构建编译 prompt
            responses_text = ""
            for idx, slot, slot_key in ready_slots:
                response = st.session_state.slot_responses.get(slot_key, "")
                responses_text += f"\n\n### {slot['provider']} ({slot['model']}):\n{response}"

            system_prompt = f"""你是高级综合编辑。以下是 {len(ready_slots)} 个不同 AI 对同一问题的回答。

请综合所有观点，生成完整准确的最终答案：
1. 保留各回答的独特见解
2. 如有矛盾，选逻辑最严密的
3. 去掉重复信息
4. 忽略明显错误的回答
5. 用原问题的语言回答"""

            compile_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"原始问题：{prompt}\n\n各 AI 回答：{responses_text}\n\n请综合编译。"},
            ]

            st.markdown("---")
            st.markdown(f"**📝 编译结果** (由 `{compile_model}` 合成)")

            compile_placeholder = st.empty()
            full_compile = ""

            try:
                for token in call_ai(
                    compile_messages, compile_key, compile_model, 0.5, 3072,
                    compile_config.get("base_url", ""),
                ):
                    full_compile += token
                    compile_placeholder.markdown(full_compile + "▌")
                compile_placeholder.markdown(full_compile)
                st.session_state.compile_response = full_compile
            except Exception as e:
                compile_placeholder.error(f"❌ 编译失败: {str(e)[:200]}")
    else:
        st.caption("⏳ 等待至少 2 个 AI 完成回复...")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("💰 省钱技巧：DeepSeek ¥0.14 + Gemini 🆓 + Groq 🆓 = 一次对比几乎免费。每家注册拿免费 Key，不花冤枉钱。")
