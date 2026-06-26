"""
⚡ AI Compiler — 一个 Prompt 同时发给多个 AI，并排对比，一键编译合成最终答案
支持: OpenAI / DeepSeek / Groq / Claude / Together / OpenRouter
"""
import streamlit as st
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ── 供应商配置 ──────────────────────────────────────────────
PROVIDERS = {
    "OpenAI": {
        "prefixes": ["sk-proj-", "sk-admin-"],
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "o4-mini", "o3-mini"],
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
        "base_url": None,
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

# ── 默认槽位 ────────────────────────────────────────────────
DEFAULT_SLOTS = [
    {"provider": "DeepSeek", "model": "deepseek-chat"},
    {"provider": "Groq", "model": "llama-3.3-70b-versatile"},
    {"provider": "OpenAI", "model": "gpt-4o-mini"},
]


def detect_provider(api_key: str):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


def call_ai(messages: list, api_key: str, model: str, temperature: float, max_tokens: int, is_claude: bool, base_url: str = None):
    """流式调用 AI — 通用生成器"""
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
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="⚡ AI Compiler", page_icon="⚡", layout="wide")

st.title("⚡ AI Compiler")
st.caption("一个 Prompt → 多个 AI 同时回答 → 并排对比 → 一键编译合成最优答案")

# ── 初始化 session ─────────────────────────────────────────
if "slots" not in st.session_state:
    st.session_state.slots = [dict(s) for s in DEFAULT_SLOTS]  # deep copy

if "slot_responses" not in st.session_state:
    st.session_state.slot_responses = {}

if "compile_response" not in st.session_state:
    st.session_state.compile_response = ""

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 全局设置")

    api_key = os.getenv("API_KEY") or st.text_input(
        "API Key（所有槽位共用）",
        type="password",
        placeholder="粘贴 API Key...",
    )

    provider_name, _ = detect_provider(api_key) if api_key else (None, None)
    if api_key and provider_name:
        st.success(f"✅ 识别为 **{provider_name}**")
    elif api_key:
        st.warning("⚠️ 未识别供应商")

    temperature = st.slider("创造性 (Temperature)", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("最大输出长度", 256, 8192, 2048, 128)

    st.divider()

    # ── 槽位配置 ──
    st.header("📊 槽位配置")

    slots_to_remove = []

    for idx, slot in enumerate(st.session_state.slots):
        with st.expander(f"🔹 Slot {idx + 1}: {slot['provider']} — {slot['model']}", expanded=len(st.session_state.slots) <= 3):
            col_a, col_b = st.columns([3, 1])

            with col_a:
                # 供应商选择
                provider_names = list(PROVIDERS.keys())
                current_provider_idx = provider_names.index(slot["provider"]) if slot["provider"] in provider_names else 0
                new_provider = st.selectbox(
                    "供应商",
                    options=provider_names,
                    index=current_provider_idx,
                    key=f"slot_provider_{idx}",
                )

                # 模型选择
                models = PROVIDERS[new_provider]["models"]
                current_model_idx = models.index(slot["model"]) if slot["model"] in models else 0
                new_model = st.selectbox(
                    "模型",
                    options=models,
                    index=current_model_idx,
                    key=f"slot_model_{idx}",
                )

                slot["provider"] = new_provider
                slot["model"] = new_model

            with col_b:
                st.caption("操作")
                if st.button("🗑", key=f"remove_{idx}", help="移除此槽位"):
                    slots_to_remove.append(idx)

    # 处理移除
    for idx in reversed(slots_to_remove):
        if len(st.session_state.slots) > 1:
            st.session_state.slots.pop(idx)
            st.rerun()

    # 添加槽位按钮
    if len(st.session_state.slots) < 8:
        if st.button("➕ 添加槽位", use_container_width=True):
            st.session_state.slots.append({"provider": "OpenAI", "model": "gpt-4o-mini"})
            st.rerun()

    st.caption(f"共 **{len(st.session_state.slots)}** 个槽位（最多 8 个）")

    st.divider()

    # ── 编译设置 ──
    st.header("🎯 编译合成")
    compile_slot_idx = st.selectbox(
        "用哪个槽位的 AI 做编译？",
        options=list(range(len(st.session_state.slots))),
        format_func=lambda i: f"Slot {i+1}: {st.session_state.slots[i]['provider']} — {st.session_state.slots[i]['model']}",
    )

    st.divider()

    if st.button("🗑 清空全部回复", use_container_width=True):
        st.session_state.slot_responses = {}
        st.session_state.compile_response = ""
        st.rerun()

# ── 主区域：Prompt ─────────────────────────────────────────
st.subheader("💬 输入 Prompt")
prompt = st.chat_input("输入你的问题，所有 AI 会同时回答...")

# ── 结果区域：N 列并排 ─────────────────────────────────────
num_slots = len(st.session_state.slots)
if num_slots > 0:
    st.subheader("📊 AI 对比结果")
    result_cols = st.columns(num_slots)

    # 渲染每列
    for idx, (col, slot) in enumerate(zip(result_cols, st.session_state.slots)):
        with col:
            provider = slot["provider"]
            model = slot["model"]
            config = PROVIDERS.get(provider, {})
            is_claude = provider == "Anthropic Claude"

            st.markdown(f"**Slot {idx + 1}**")
            st.caption(f"{provider} · `{model}`")

            placeholder = st.empty()

            # 初始化或显示已有回复
            if prompt:
                slot_key = f"{idx}_{prompt}"
                if slot_key not in st.session_state.slot_responses:
                    # 发送给 AI
                    with st.spinner(f"⏳ {provider} 思考中..."):
                        try:
                            messages = [{"role": "user", "content": prompt}]
                            full = ""
                            for token in call_ai(
                                messages, api_key, model, temperature, max_tokens,
                                is_claude, config.get("base_url"),
                            ):
                                full += token
                                placeholder.markdown(full + "▌")
                            placeholder.markdown(full)
                            st.session_state.slot_responses[slot_key] = full
                        except Exception as e:
                            placeholder.error(f"❌ {str(e)[:150]}")
                            st.session_state.slot_responses[slot_key] = f"**错误**: {str(e)[:300]}"
                else:
                    # 已有回复，直接显示
                    placeholder.markdown(st.session_state.slot_responses[slot_key])
            else:
                # 还没输入 prompt
                placeholder.info(f"等待输入...")

# ── 编译合成 ──────────────────────────────────────────────
if prompt and num_slots > 1:
    st.divider()
    st.subheader("🎯 编译合成")

    # 检查是否所有槽位都有回复
    slot_keys_for_prompt = [f"{idx}_{prompt}" for idx in range(num_slots)]
    all_ready = all(k in st.session_state.slot_responses for k in slot_keys_for_prompt)

    if all_ready:
        compile_btn = st.button(
            f"🚀 用 Slot {compile_slot_idx + 1} 编译合成所有答案",
            type="primary",
            use_container_width=True,
        )

        if compile_btn:
            compile_config = PROVIDERS.get(st.session_state.slots[compile_slot_idx]["provider"], {})
            compile_model = st.session_state.slots[compile_slot_idx]["model"]
            compile_is_claude = st.session_state.slots[compile_slot_idx]["provider"] == "Anthropic Claude"

            # 构建编译 prompt
            responses_text = ""
            for idx in range(num_slots):
                slot = st.session_state.slots[idx]
                response = st.session_state.slot_responses.get(f"{idx}_{prompt}", "")
                responses_text += f"\n\n### Slot {idx + 1} ({slot['provider']} — {slot['model']}):\n{response}"

            system_prompt = f"""你是一位高级综合编辑。以下是 {num_slots} 个不同的 AI 对同一个问题的回答。

请综合所有观点，生成一份完整、准确、无冗余的最终答案：

规则：
1. 取各家最精华的部分，保留独特见解
2. 如果多个回答有矛盾，选择逻辑最严密的那个
3. 避免信息重复
4. 如果某个回答有明显错误，忽略它
5. 最终答案结构清晰，适合直接使用"""

            compile_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"原始问题：{prompt}\n\n以下是各 AI 的回答：{responses_text}\n\n请综合编译为最终答案。"},
            ]

            st.markdown("---")
            st.markdown(f"**📝 编译结果** (由 {st.session_state.slots[compile_slot_idx]['provider']} — `{compile_model}` 合成)")

            compile_placeholder = st.empty()
            full_compile = ""

            try:
                for token in call_ai(
                    compile_messages, api_key, compile_model, 0.5, 3072,
                    compile_is_claude, compile_config.get("base_url"),
                ):
                    full_compile += token
                    compile_placeholder.markdown(full_compile + "▌")
                compile_placeholder.markdown(full_compile)
                st.session_state.compile_response = full_compile
            except Exception as e:
                compile_placeholder.error(f"❌ 编译失败: {str(e)[:200]}")
    else:
        st.caption("⏳ 等待所有 AI 完成回复...")

    # 显示缓存的上次编译结果
    if st.session_state.compile_response and not prompt:
        st.markdown("**📝 上次编译结果**")
        st.markdown(st.session_state.compile_response)

elif prompt and num_slots <= 1:
    st.info("👆 需要至少 2 个槽位才能编译合成。请在侧边栏添加槽位。")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("💡 技巧：用不同供应商的模型对比同一个问题，DeepSeek 擅长推理、Claude 擅长写作、GPT-4o 擅长全面分析。编译合成取各家长处！")
