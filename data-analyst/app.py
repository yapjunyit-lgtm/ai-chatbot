"""
📊 AI 数据分析师 — 上传表格，自动洞察 + 图表 + 自然语言问答
支持: OpenAI / DeepSeek / Groq / Claude / 自定义接口
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import os

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
        "models": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "qwen-2.5-32b"],
    },
    "Anthropic Claude": {
        "prefixes": ["sk-ant-"],
        "base_url": None,
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-8"],
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


def detect_provider(api_key):
    for name, config in PROVIDERS.items():
        for prefix in config["prefixes"]:
            if api_key.startswith(prefix):
                return name, config
    return None, None


def call_ai(messages, api_key, model, temperature, max_tokens, is_claude, base_url=None):
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


def summarize_dataframe(df):
    """生成 DataFrame 的文字摘要，发给 AI"""
    cols = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    shape = df.shape
    missing = df.isnull().sum().to_dict()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # 数值列统计
    stats = {}
    for col in numeric_cols[:10]:
        stats[col] = {
            "mean": round(df[col].mean(), 2) if pd.notna(df[col].mean()) else "N/A",
            "median": round(df[col].median(), 2) if pd.notna(df[col].median()) else "N/A",
            "min": round(df[col].min(), 2) if pd.notna(df[col].min()) else "N/A",
            "max": round(df[col].max(), 2) if pd.notna(df[col].max()) else "N/A",
        }

    sample = df.head(5).to_string()

    return f"""
数据集概览：
- 行数: {shape[0]}，列数: {shape[1]}
- 列名: {cols}
- 数据类型: {dtypes}
- 缺失值: {missing}
- 数值列统计: {stats}

前5行数据:
{sample}
"""


# ── 页面设置 ──────────────────────────────────────────────
st.set_page_config(page_title="📊 AI 数据分析师", page_icon="📊", layout="wide")
st.title("📊 AI 数据分析师")
st.caption("上传 CSV / Excel → 自动图表 + AI 洞察 + 自然语言提问")

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API 设置")
    api_key = os.getenv("API_KEY") or st.text_input("API Key", type="password", placeholder="粘贴任何 API Key...")
    provider_name, provider_config = detect_provider(api_key) if api_key else (None, None)
    if api_key and provider_name:
        st.success(f"✅ **{provider_name}**")

    all_providers = list(PROVIDERS.keys()) + ["自定义 (OpenAI 兼容)"]
    manual_provider = st.selectbox("供应商", options=["🔍 自动检测"] + all_providers, index=0)
    custom_base_url = None
    if manual_provider == "自定义 (OpenAI 兼容)":
        custom_base_url = st.text_input("Base URL", placeholder="https://api.xxx.com/v1")

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
    temperature = st.slider("创造性", 0.0, 2.0, 0.3, 0.1)
    st.divider()
    st.caption(f"🟢 **{active_name}** | `{model}`")

    st.divider()
    st.subheader("📤 上传数据")
    uploaded_file = st.file_uploader("CSV 或 Excel 文件", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        st.success(f"✅ 已加载: {uploaded_file.name}")

# ── 主区域 ────────────────────────────────────────────────
if uploaded_file:
    # 读取数据
    @st.cache_data
    def load_data(file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)

    df = load_data(uploaded_file)

    # ── Tab 结构 ──
    tab_overview, tab_charts, tab_insights, tab_ask = st.tabs([
        "📋 数据概览", "📈 智能图表", "🤖 AI 洞察", "💬 自然语言提问",
    ])

    # ── Tab 1: 数据概览 ──
    with tab_overview:
        metric_cols = st.columns(4)
        metric_cols[0].metric("行数", f"{len(df):,}")
        metric_cols[1].metric("列数", len(df.columns))
        numeric_count = len(df.select_dtypes(include="number").columns)
        metric_cols[2].metric("数值列", numeric_count)
        missing_pct = round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
        metric_cols[3].metric("缺失值 %", f"{missing_pct}%")

        st.subheader("📋 数据预览")
        st.dataframe(df.head(100), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("📊 数值统计")
            st.dataframe(df.describe(), use_container_width=True)
        with col_b:
            st.subheader("🔍 数据类型 & 缺失值")
            info_df = pd.DataFrame({
                "类型": df.dtypes.astype(str),
                "缺失值": df.isnull().sum(),
                "缺失%": (df.isnull().sum() / len(df) * 100).round(2),
                "唯一值": df.nunique(),
            })
            st.dataframe(info_df, use_container_width=True)

    # ── Tab 2: 智能图表 ──
    with tab_charts:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            st.subheader("📈 自动图表生成")

            col_left, col_right = st.columns([1, 3])

            with col_left:
                chart_type = st.selectbox(
                    "图表类型",
                    ["📊 柱状图 (Bar)", "📉 折线图 (Line)", "🥧 饼图 (Pie)",
                     "📦 箱线图 (Box)", "🎻 小提琴图 (Violin)", "🔵 散点图 (Scatter)",
                     "🔥 热力图 (Correlation)"],
                )

                x_col = st.selectbox("X 轴", options=df.columns.tolist())
                y_col = st.selectbox("Y 轴", options=numeric_cols, index=0 if numeric_cols else None) if chart_type != "🔥 热力图 (Correlation)" else None
                color_col = st.selectbox("分组颜色 (可选)", options=["无"] + df.columns.tolist())

            with col_right:
                try:
                    color_param = None if color_col == "无" else color_col

                    if chart_type == "📊 柱状图 (Bar)":
                        fig = px.bar(df.head(200), x=x_col, y=y_col, color=color_param, title=f"{y_col} by {x_col}")
                    elif chart_type == "📉 折线图 (Line)":
                        fig = px.line(df.head(200), x=x_col, y=y_col, color=color_param, title=f"{y_col} over {x_col}")
                    elif chart_type == "🥧 饼图 (Pie)":
                        pie_data = df[x_col].value_counts().head(10)
                        fig = px.pie(values=pie_data.values, names=pie_data.index, title=f"{x_col} 分布")
                    elif chart_type == "📦 箱线图 (Box)":
                        fig = px.box(df, x=x_col, y=y_col, color=color_param)
                    elif chart_type == "🎻 小提琴图 (Violin)":
                        fig = px.violin(df.head(500), x=x_col, y=y_col, color=color_param, box=True)
                    elif chart_type == "🔵 散点图 (Scatter)":
                        fig = px.scatter(df.head(500), x=x_col, y=y_col, color=color_param,
                                         size=y_col if y_col in numeric_cols else None,
                                         hover_data=df.columns.tolist())
                    elif chart_type == "🔥 热力图 (Correlation)":
                        corr = df[numeric_cols].corr()
                        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                        title="数值列相关性热力图")

                    fig.update_layout(height=500, template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ 无法生成图表: {e}")
        else:
            st.info("👆 数据中没有数值列，无法自动生成图表。")

    # ── Tab 3: AI 洞察 ──
    with tab_insights:
        st.subheader("🤖 AI 自动分析")

        if st.button("🚀 生成洞察", type="primary"):
            if not api_key:
                st.error("⚠️ 请先输入 API Key")
            else:
                with st.spinner("🤖 AI 正在分析你的数据..."):
                    summary = summarize_dataframe(df)
                    system_prompt = """你是一位资深数据分析师。请基于数据概览，用中文给出有价值的洞察。

请按以下结构输出：

### 🔍 关键发现
列出 3-5 个最重要的发现，每个用一句话。

### 📈 趋势与模式
数据中隐藏的趋势或模式。

### ⚠️ 需要注意的问题
数据质量问题、异常值、缺失值影响等。

### 💡 行动建议
针对数据洞察，给出 3-5 条可执行的业务建议。

用要点形式，简洁有力。"""

                    try:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": summary},
                        ]
                        placeholder = st.empty()
                        full = ""
                        for token in call_ai(messages, api_key, model, temperature, 2048, is_claude, active_config.get("base_url")):
                            full += token
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)
                    except Exception as e:
                        st.error(f"❌ {str(e)[:300]}")

    # ── Tab 4: 自然语言提问 ──
    with tab_ask:
        st.subheader("💬 用自然语言提问你的数据")

        if "data_qa_history" not in st.session_state:
            st.session_state.data_qa_history = []

        for qa in st.session_state.data_qa_history:
            with st.chat_message("user"):
                st.markdown(qa["q"])
            with st.chat_message("assistant"):
                st.markdown(qa["a"])

        question = st.chat_input("例如：哪个产品的销量最高？年龄和收入有什么关系？")

        if question:
            if not api_key:
                st.error("⚠️ 请先输入 API Key")
            else:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    data_summary = summarize_dataframe(df)
                    system_prompt = "你是数据分析专家。基于提供的数据回答用户问题。如果数据不足以回答，请诚实说明。用中文回答，简洁准确。如果涉及数值，给出具体数字。"

                    try:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"数据：\n{data_summary}\n\n问题：{question}"},
                        ]
                        placeholder = st.empty()
                        full = ""
                        for token in call_ai(messages, api_key, model, 0.3, 1024, is_claude, active_config.get("base_url")):
                            full += token
                            placeholder.markdown(full + "▌")
                        placeholder.markdown(full)

                        st.session_state.data_qa_history.append({"q": question, "a": full})
                    except Exception as e:
                        st.error(f"❌ {str(e)[:300]}")

else:
    # 没有数据时显示示例
    st.info("👈 请先在侧边栏上传 CSV 或 Excel 文件")
    st.markdown("""
    ### 📊 功能预览

    | 功能 | 说明 |
    |------|------|
    | 📋 **数据概览** | 自动统计、缺失值检测、数据类型识别 |
    | 📈 **智能图表** | 柱状图 / 折线图 / 饼图 / 箱线图 / 散点图 / 热力图 |
    | 🤖 **AI 洞察** | 一键生成关键发现 + 趋势分析 + 行动建议 |
    | 💬 **自然语言提问** | 用中文直接问数据问题 |

    ---
    💡 **没有数据？** 用这个命令生成测试数据：
    ```python
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=100),
        "销售额": np.random.randint(1000, 5000, 100),
        "利润": np.random.randint(100, 800, 100),
        "客户数": np.random.randint(10, 100, 100),
        "渠道": np.random.choice(["线上", "线下", "分销"], 100),
    })
    df.to_csv("sample.csv", index=False)
    ```
    """)

st.divider()
st.caption("💡 提示：数据包含中文列名时，图表可能显示为方块 — 这是 Plotly 字体问题，不影响分析结果。")
