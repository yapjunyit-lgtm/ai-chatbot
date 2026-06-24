# 🤖 AI Chatbot

我的第一个 AI 应用 — 基于 Streamlit + OpenAI API 的智能聊天机器人。

## 功能

- 💬 实时对话（流式输出，打字机效果）
- 🎛 可调节创造性 (Temperature)
- 🔄 多模型切换 (GPT-4o / GPT-4o-mini)
- 📝 对话历史保持

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 OpenAI API Key

# 4. 运行
streamlit run app.py
```

## 技术栈

- **Python** 3.13
- **Streamlit** — Web UI
- **OpenAI API** — AI 对话能力

## 截图

> 运行后在这里贴截图
