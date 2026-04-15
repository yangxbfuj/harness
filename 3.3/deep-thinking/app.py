import streamlit as st
import asyncio
import nest_asyncio
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from agent import DeepResearchAgent

# 允许在 Streamlit 中嵌套使用 asyncio 事件循环
nest_asyncio.apply()

# ── 页面基础配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deep Thinking AI 助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* 聊天区域最大宽度 */
.block-container { max-width: 900px; margin: auto; }

/* 工具步骤卡片 */
.step-card {
    background: #f8f9fa;
    border-left: 4px solid #4c8bf5;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
}
.step-result {
    background: #f0faf0;
    border-left: 4px solid #34a853;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session State 初始化 ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [{question, steps, answer}]

if "agent" not in st.session_state:
    st.session_state.agent = DeepResearchAgent()

# ── 工具函数 ──────────────────────────────────────────────────────────────────

def parse_messages(all_messages: list) -> tuple[list, str]:
    """
    将 LangGraph 返回的完整消息列表解析为：
    - steps: [{"tool": str, "args": dict, "result": str}]
    - final_answer: str
    """
    steps = []
    final_answer = ""
    pending_calls: dict[str, dict] = {}   # call_id -> step dict

    for msg in all_messages:
        if isinstance(msg, HumanMessage):
            continue
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    step = {
                        "tool": tc["name"],
                        "args": tc["args"],
                        "result": "",
                        "call_id": tc["id"],
                    }
                    steps.append(step)
                    pending_calls[tc["id"]] = step
            else:
                # 没有 tool_calls 的 AIMessage 就是最终回答
                final_answer = msg.content
        elif isinstance(msg, ToolMessage):
            matched = pending_calls.get(msg.tool_call_id)
            if matched is not None:
                matched["result"] = str(msg.content)

    return steps, final_answer


def render_thinking_expander(steps: list, key_prefix: str):
    """在可折叠面板中渲染思考过程（工具调用 + 结果）"""
    if not steps:
        return
    label = f"🔍 思考过程  ·  共调用 {len(steps)} 次工具"
    with st.expander(label, expanded=False):
        for idx, step in enumerate(steps):
            st.markdown(
                f'<div class="step-card">🔧 <b>第 {idx + 1} 步 &nbsp;·&nbsp; 工具：<code>{step["tool"]}</code></b></div>',
                unsafe_allow_html=True,
            )
            col_in, col_out = st.columns(2, gap="medium")
            with col_in:
                st.markdown("📥 **输入参数**")
                st.json(step.get("args", {}), expanded=False)
            with col_out:
                st.markdown("📤 **执行结果**")
                result_text = step.get("result", "（无结果）")
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "\n\n…（内容已截断，仅展示前 1000 字符）"
                st.text_area(
                    label="执行结果",
                    value=result_text,
                    height=160,
                    key=f"{key_prefix}_result_{idx}",
                    disabled=True,
                    label_visibility="collapsed",
                )
            if idx < len(steps) - 1:
                st.divider()


# ── 渲染历史对话 ──────────────────────────────────────────────────────────────
st.title("🧠 DeepResearchAgent AI 助手")
st.caption("基于 LangGraph + Qwen 的深度思考对话助手，支持网络搜索与深度分析，结果以 Markdown 格式呈现")
st.divider()

for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        render_thinking_expander(chat.get("steps", []), key_prefix=f"hist_{i}")
        st.markdown(chat["answer"])

# ── 对话输入 ──────────────────────────────────────────────────────────────────
question = st.chat_input("💬 请输入您的问题，例如：帮我分析某某股票的最新动态…")

if question:
    # 立即渲染用户消息
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        steps: list = []
        final_answer: str = ""

        # ── 进度状态面板（流式实时更新） ────────────────────────────────────
        with st.status("🤔 深度思考中，请稍候…", expanded=True) as status_box:
            steps_placeholder = st.empty()    # 专用于渲染步骤列表
            status_msg_placeholder = st.empty()  # 专用于底部状态提示行

            # 用可变容器在闭包内共享状态（避免 nonlocal）
            live_steps: list = []       # 实时积累的步骤信息
            pending_calls: dict = {}    # call_id -> step dict
            result_box: list = []       # 最终消息列表

            def _refresh_live_display():
                """将当前 live_steps 渲染到占位符"""
                with steps_placeholder.container():
                    for s in live_steps:
                        icon = "✅" if s["done"] else "⏳"
                        # 截短参数，只展示前 120 字符，避免过长
                        args_str = str(s["args"])
                        if len(args_str) > 120:
                            args_str = args_str[:120] + "…"
                        st.markdown(
                            f'{icon} **第 {s["index"]} 步** &nbsp;·&nbsp; '
                            f'工具：`{s["tool"]}` &nbsp;·&nbsp; `{args_str}`'
                        )

            async def _run_agent_streaming():
                call_count = [0]  # 用列表包装以便在内层修改

                async for node_name, new_msgs, all_msgs in \
                        st.session_state.agent.stream_run_with_history(question):

                    result_box.clear()
                    result_box.extend(all_msgs)

                    if node_name == "llm_call":
                        for msg in new_msgs:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    call_count[0] += 1
                                    step = {
                                        "index": call_count[0],
                                        "tool": tc["name"],
                                        "args": tc["args"],
                                        "done": False,
                                        "call_id": tc["id"],
                                    }
                                    live_steps.append(step)
                                    pending_calls[tc["id"]] = step
                                _refresh_live_display()
                            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                                # LLM 无工具调用 → 正在生成最终答案（追加在步骤列表下方）
                                status_msg_placeholder.markdown("✍️ **正在生成最终分析报告…**")

                    elif node_name == "environment":
                        for msg in new_msgs:
                            if isinstance(msg, ToolMessage):
                                matched = pending_calls.get(msg.tool_call_id)
                                if matched:
                                    matched["done"] = True
                        _refresh_live_display()

            try:
                asyncio.run(_run_agent_streaming())
                all_messages = result_box
                steps, final_answer = parse_messages(all_messages)
                status_box.update(
                    label=f"✅ 思考完成！共调用 {len(steps)} 次工具",
                    state="complete",
                    expanded=False,
                )
            except Exception as exc:
                import traceback
                err_detail = traceback.format_exc()
                steps_placeholder.markdown(f"❌ 执行出错：{exc}")
                status_box.update(label="❌ 执行失败", state="error", expanded=True)
                st.error(f"错误详情：\n```\n{err_detail}\n```")
                final_answer = f"> ⚠️ Agent 执行失败：{exc}"

        # ── 最终报告 ────────────────────────────────────────────────────────
        st.markdown("### 📋 分析报告")
        st.markdown(final_answer)

        # ── 保存到历史 ──────────────────────────────────────────────────────
        st.session_state.chat_history.append({
            "question": question,
            "steps": steps,
            "answer": final_answer,
        })
