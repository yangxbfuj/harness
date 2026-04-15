from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing_extensions import Literal
from webtools import widesearch_for_toolstr
from llm import Tongyi
from prompts import system_prompt, get_current_date
import asyncio

mcp_tools = {}
mcp_tools["common"] = {
    "url": "http://localhost:38001/mcp",
    "transport": "streamable_http",
}

llm = Tongyi()

class DeepResearchAgent:

    async def _build_agent(self):
        """构建并返回编译好的 LangGraph Agent 及工具列表"""
        mcp_client = MultiServerMCPClient(mcp_tools)
        tools = await mcp_client.get_tools()
        tools = tools + [widesearch_for_toolstr]
        tools_by_name = {tool.name: tool for tool in tools}
        llm_with_tools = llm.bind_tools(tools)

        async def llm_call(state: MessagesState):
            print("llm_call")
            messages = [
                SystemMessage(content=system_prompt.format(current_date=get_current_date())),
            ] + state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            state["messages"].append(response)
            return state

        async def tool_node(state):
            print("tool_node")
            for tool_call in state["messages"][-1].tool_calls:
                tool = tools_by_name[tool_call["name"]]
                print("工具名称:", tool_call["name"])
                print("工具参数:", tool_call["args"])
                try:
                    observation = await tool.ainvoke(tool_call["args"])
                    state["messages"].append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
                except Exception as e:
                    error_msg = f"工具执行失败: {str(e)}\n错误类型: {type(e).__name__}"
                    print(f"工具执行错误: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    state["messages"].append(ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call["id"]
                    ))
            return state

        def should_continue(state) -> Literal["environment", "END"]:
            if state["messages"][-1].tool_calls:
                return "environment"
            return "END"

        agent_builder = StateGraph(MessagesState)
        agent_builder.add_node("llm_call", llm_call)
        agent_builder.add_node("environment", tool_node)
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_conditional_edges(
            "llm_call",
            should_continue,
            {
                "environment": "environment",
                "END": END,
            },
        )
        agent_builder.add_edge("environment", "llm_call")
        return agent_builder.compile()

    _RUN_CONFIG = {"recursion_limit": 100}

    async def run(self, question):
        """运行 Agent，返回最终答案字符串"""
        agent = await self._build_agent()
        messages = [HumanMessage(content=question)]
        ret = await agent.ainvoke({"messages": messages}, config=self._RUN_CONFIG)
        return ret["messages"][-1].content

    async def run_with_history(self, question):
        """运行 Agent，返回完整的消息历史（包含工具调用和结果等中间步骤）"""
        agent = await self._build_agent()
        messages = [HumanMessage(content=question)]
        ret = await agent.ainvoke({"messages": messages}, config=self._RUN_CONFIG)
        return ret["messages"]

    async def stream_run_with_history(self, question):
        """
        流式运行 Agent，每个节点执行完毕后 yield (node_name, new_msgs, all_msgs)。
        - node_name : 'llm_call' 或 'environment'
        - new_msgs  : 本次节点【真正新增】的消息列表（去除历史重复）
        - all_msgs  : 截止目前累计的全部消息列表

        注意：llm_call / tool_node 均返回完整 state，astream chunk 里的
        messages 是全量列表而非增量，因此用游标 prev_count 切出真正新增部分。
        """
        agent = await self._build_agent()
        init_messages = [HumanMessage(content=question)]
        all_messages = list(init_messages)

        async for chunk in agent.astream({"messages": init_messages}, config=self._RUN_CONFIG):
            for node_name, output in chunk.items():
                chunk_msgs = output.get("messages", [])
                # chunk_msgs 是全量消息，prev_count 之后的才是本节点新增的
                prev_count = len(all_messages)
                new_msgs = chunk_msgs[prev_count:]
                if new_msgs:
                    all_messages.extend(new_msgs)
                    yield node_name, new_msgs, list(all_messages)

if __name__ == "__main__":
    agent = DeepResearchAgent()
    result = asyncio.run(agent.run("请帮我总结一下OpenClaw相关的股市热点新闻以及相关概念股，最后生成一份完整的MarkDown格式的分析报告"))
    print(result)