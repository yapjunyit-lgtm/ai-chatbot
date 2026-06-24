"""
AI Chatbot — 你的第一个 AI 应用
技术栈: Streamlit + OpenAI API
"""
import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的 API Key
load_dotenv()

# ── 页面设置 ──────────────────────────────────
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 AI Chatbot")
st.caption("你的第一个 AI 应用 — 可以回答问题、写代码、翻译、聊天")

# ── 侧边栏：API 配置 ──────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")

    # API Key 输入（优先用 .env，其次手动输入）
    api_key = os.getenv("OPENAI_API_KEY") or st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="在 https://platform.openai.com/api-keys 获取",
    )

    # 模型选择
    model = st.selectbox(
        "选择模型",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4o-mini 最便宜够用，gpt-4o 最强",
    )

    # Temperature（创造性）
    temperature = st.slider(
        "创造性 (Temperature)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="越高越有创意，越低越严谨",
    )

    st.divider()

    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── 对话历史 ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的 AI 助手。有什么我可以帮你的吗？"}
    ]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 用户输入 ──────────────────────────────────
if prompt := st.chat_input("在这里输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 AI
    if not api_key:
        st.error("⚠️ 请先在侧边栏输入 OpenAI API Key")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                client = OpenAI(api_key=api_key)

                # 构建消息列表
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                # 调用 OpenAI API
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,  # 流式输出，打字机效果
                )

                # 逐字输出
                full_response = ""
                placeholder = st.empty()
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                # 保存 AI 回复（只在成功时）
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"❌ 出错了: {str(e)}")
                st.info("💡 常见原因：API Key 无效、余额不足、或网络问题")

                # 移除用户消息（避免对话历史出问题）
                st.session_state.messages.pop()
                st.stop()
