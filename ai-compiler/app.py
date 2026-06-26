"""
⚡ AI Compiler — 一个 Prompt 同时发给多个 AI，并排对比，一键编译合成
🔑 一键登录 OpenRouter，无需手动输入 API Key
"""
import streamlit as st
import requests
import urllib.parse
import os
from openai import OpenAI

# ── 常量 ────────────────────────────────────────────────────
OPENROUTER_AUTH_URL = "https://openrouter.ai/auth"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_EXCHANGE = f"{OPENROUTER_API_URL}/auth/keys"
OPENROUTER_MODELS_URL = f"{OPENROUTER_API_URL}/models"

# 默认槽位（OpenRouter 格式模型名）
DEFAULT_SLOTS = [
    {"model": "openai/gpt-4o"},
    {"model": "anthropic/claude-sonnet-4-6"},
    {"model": "google/gemini-2.5-pro"},
]

# 热门模型快捷列表（首次加载/网络失败时的后备）
POPULAR_MODELS = [
    # OpenAI
    "openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o", "openai/gpt-4o-mini",
    "openai/o4-mini", "openai/o3-mini",
    # Anthropic
    "anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-8",
    "anthropic/claude-haiku-4-5", "anthropic/claude-fable-5",
    # Google
    "google/gemini-2.5-pro", "google/gemini-2.5-flash", "google/gemini-2.0-flash",
    # Meta
    "meta-llama/llama-4-maverick", "meta-llama/llama-3.3-70b-instruct",
    # DeepSeek
    "deepseek/deepseek-chat", "deepseek/deepseek-r1",
    # Mistral
    "mistralai/mistral-large", "mistralai/mixtral-8x22b",
    # Qwen
    "qwen/qwen-2.5-72b-instruct", "qwen/qwq-32b",
    # xAI
    "x-ai/grok-3-beta",
]


# ── OAuth 逻辑 ──────────────────────────────────────────────
def get_app_url():
    """获取当前 App 的完整 URL（用作 OAuth callback）"""
    try:
        # Streamlit Cloud 环境
        if "STREAMLIT_SERVER_PORT" in os.environ:
            port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
            return f"http://localhost:{port}"
    except Exception:
        pass
    return "http://localhost:8505"


def exchange_code_for_key(code: str) -> str | None:
    """用 OAuth code 换取 OpenRouter API Key"""
    try:
        resp = requests.post(
            OPENROUTER_KEY_EXCHANGE,
            json={"code": code},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("key")
        else:
            st.error(f"❌ 换取 Key 失败 ({resp.status_code}): {resp.text[:300]}")
            return None
    except Exception as e:
        st.error(f"❌ 网络错误: {e}")
        return None


def get_openrouter_models(api_key: str) -> list[str]:
    """从 OpenRouter 拉取可用模型列表"""
    try:
        resp = requests.get(
            OPENROUTER_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                # 过滤掉不合适的模型
                if model_id and not any(skip in model_id.lower() for skip in [
                    "evl-", "liquid-", "nvidia/", ":free", ":extended",
                ]):
                    models.append(model_id)
            return sorted(models)
        else:
            return POPULAR_MODELS
    except Exception:
        return POPULAR_MODELS


def call_ai(messages: list, api_key: str, model: str, temperature: float, max_tokens: int):
    """流式调用 AI — 全部通过 OpenRouter API"""
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_API_URL)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        extra_headers={
            "HTTP-Referer": get_app_url(),
            "X-Title": "AI Compiler",
        },
    )
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="⚡ AI Compiler", page_icon="⚡", layout="wide")

st.title("⚡ AI Compiler")
st.caption("一个 Prompt → 多个 AI 同时回答 → 并排对比 → 一键编译合成最优答案")

# ── 初始化 session ─────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "slots" not in st.session_state:
    st.session_state.slots = [dict(s) for s in DEFAULT_SLOTS]

if "slot_responses" not in st.session_state:
    st.session_state.slot_responses = {}

if "compile_response" not in st.session_state:
    st.session_state.compile_response = ""

if "models_cache" not in st.session_state:
    st.session_state.models_cache = POPULAR_MODELS

# ── OAuth 回调处理 ─────────────────────────────────────────
query_params = st.query_params
if "code" in query_params and not st.session_state.api_key:
    code = query_params["code"]
    with st.spinner("🔑 正在登录 OpenRouter..."):
        key = exchange_code_for_key(code)
        if key:
            st.session_state.api_key = key
            # 拉取模型列表
            st.session_state.models_cache = get_openrouter_models(key)
            st.query_params.clear()
            st.rerun()
        else:
            st.error("登录失败，请重试。")
            st.query_params.clear()
            st.rerun()

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 账户")

    if st.session_state.api_key:
        st.success(f"✅ 已登录 OpenRouter")

        # 拉取模型列表（如果还没拉）
        if st.button("🔄 刷新模型列表"):
            st.session_state.models_cache = get_openrouter_models(st.session_state.api_key)
            st.rerun()

        st.metric("可用模型", len(st.session_state.models_cache))

        if st.button("🚪 登出", use_container_width=True):
            st.session_state.api_key = None
            st.session_state.slot_responses = {}
            st.session_state.compile_response = ""
            st.rerun()
    else:
        st.warning("⚠️ 未登录")

        # 登录按钮
        app_url = get_app_url()
        callback_url = urllib.parse.quote(app_url, safe="")
        login_url = f"{OPENROUTER_AUTH_URL}?callback_url={callback_url}"

        st.markdown(f"""
        <a href="{login_url}" target="_self">
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 14px 20px; border-radius: 10px;
                text-align: center; font-size: 16px; font-weight: bold;
                cursor: pointer; margin: 10px 0;
            ">
                🔑 登录 OpenRouter
            </div>
        </a>
        """, unsafe_allow_html=True)

        st.caption("""
        **不需要 API Key！**
        点击登录 → 跳转 OpenRouter 授权 → 自动回来。

        一个账号访问 200+ 模型：
        GPT-4o · Claude · Gemini · DeepSeek · Llama · Grok · Qwen
        """)

        st.info("💡 首次使用需要注册 [OpenRouter](https://openrouter.ai) 账号（免费）")

    st.divider()

    # ── 全局设置 ──
    st.header("⚙️ 参数设置")

    temperature = st.slider("创造性 (Temperature)", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("最大输出长度", 256, 8192, 2048, 128)

    st.divider()

    # ── 槽位配置 ──
    st.header("📊 槽位配置")
    models = st.session_state.models_cache
    slots_to_remove = []

    for idx, slot in enumerate(st.session_state.slots):
        model = slot["model"]
        with st.expander(f"🔹 Slot {idx + 1}: `{model}`", expanded=len(st.session_state.slots) <= 3):
            col_a, col_b = st.columns([4, 1])

            with col_a:
                # 找当前模型在列表中的位置
                try:
                    current_idx = models.index(model)
                except ValueError:
                    current_idx = 0

                new_model = st.selectbox(
                    "模型",
                    options=models,
                    index=current_idx,
                    key=f"slot_model_{idx}",
                    help=f"搜索模型名...",
                )
                slot["model"] = new_model

            with col_b:
                st.caption(" ")
                if st.button("🗑", key=f"remove_{idx}", help="移除此槽位"):
                    slots_to_remove.append(idx)

    # 处理移除
    for idx in reversed(slots_to_remove):
        if len(st.session_state.slots) > 1:
            st.session_state.slots.pop(idx)
            st.rerun()

    # 添加槽位
    if len(st.session_state.slots) < 8:
        if st.button("➕ 添加槽位", use_container_width=True):
            st.session_state.slots.append({"model": models[0] if models else "openai/gpt-4o-mini"})
            st.rerun()

    st.caption(f"共 **{len(st.session_state.slots)}** 个槽位（最多 8 个）")

    st.divider()

    # ── 编译设置 ──
    st.header("🎯 编译合成")
    compile_slot_idx = st.selectbox(
        "用哪个槽位做编译？",
        options=list(range(len(st.session_state.slots))),
        format_func=lambda i: f"Slot {i+1}: {st.session_state.slots[i]['model']}",
    )

    st.divider()

    if st.button("🗑 清空全部回复", use_container_width=True):
        st.session_state.slot_responses = {}
        st.session_state.compile_response = ""
        st.rerun()

# ── 主区域 ────────────────────────────────────────────────
if not st.session_state.api_key:
    st.info("👈 请先在侧边栏登录 OpenRouter。一个账号，访问所有 AI 模型。")
else:
    st.subheader("💬 输入 Prompt")
    prompt = st.chat_input("输入你的问题，所有 AI 会同时回答...")

    # ── 结果区域：N 列并排 ──
    num_slots = len(st.session_state.slots)
    if num_slots > 0:
        st.subheader(f"📊 {num_slots} 个 AI 对比")
        result_cols = st.columns(num_slots)

        for idx, (col, slot) in enumerate(zip(result_cols, st.session_state.slots)):
            with col:
                model = slot["model"]
                st.markdown(f"**Slot {idx + 1}**")
                st.caption(f"`{model}`")

                placeholder = st.empty()

                if prompt:
                    slot_key = f"{idx}_{prompt}"
                    if slot_key not in st.session_state.slot_responses:
                        with st.spinner("⏳ 思考中..."):
                            try:
                                messages = [{"role": "user", "content": prompt}]
                                full = ""
                                for token in call_ai(messages, st.session_state.api_key, model, temperature, max_tokens):
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
        st.divider()
        st.subheader("🎯 编译合成")

        slot_keys_for_prompt = [f"{idx}_{prompt}" for idx in range(num_slots)]
        all_ready = all(k in st.session_state.slot_responses for k in slot_keys_for_prompt)

        if all_ready:
            compile_model = st.session_state.slots[compile_slot_idx]["model"]

            compile_btn = st.button(
                f"🚀 用 Slot {compile_slot_idx + 1} (`{compile_model}`) 编译合成",
                type="primary", use_container_width=True,
            )

            if compile_btn:
                # 构建编译 prompt
                responses_text = ""
                for idx in range(num_slots):
                    response = st.session_state.slot_responses.get(f"{idx}_{prompt}", "")
                    model = st.session_state.slots[idx]["model"]
                    responses_text += f"\n\n### AI {idx + 1} ({model}):\n{response}"

                system_prompt = f"""You are a senior synthesis editor. Below are {num_slots} different AI responses to the same question.

Synthesize them into ONE definitive answer. Rules:
1. Keep the best unique insights from each response
2. If responses contradict, prefer the most logically sound one
3. Remove redundant information
4. If a response is clearly wrong, ignore it
5. Structure the final answer clearly and comprehensively

Respond in the same language as the original question."""

                compile_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original question: {prompt}\n\nResponses:{responses_text}\n\nSynthesize into final answer:"},
                ]

                st.markdown("---")
                st.markdown(f"**📝 编译结果** (由 `{compile_model}` 合成)")

                compile_placeholder = st.empty()
                full_compile = ""

                try:
                    for token in call_ai(compile_messages, st.session_state.api_key, compile_model, 0.5, 3072):
                        full_compile += token
                        compile_placeholder.markdown(full_compile + "▌")
                    compile_placeholder.markdown(full_compile)
                    st.session_state.compile_response = full_compile
                except Exception as e:
                    compile_placeholder.error(f"❌ 编译失败: {str(e)[:200]}")
        else:
            st.caption("⏳ 等待所有 AI 完成回复...")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("💡 通过 OpenRouter 一个账号访问 200+ AI 模型 — GPT-4o · Claude · Gemini · DeepSeek · Llama · Grok · Qwen ...")
