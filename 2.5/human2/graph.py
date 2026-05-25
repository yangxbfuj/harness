from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START, END
from llm import Tongyi, myDeepSeek
from prompts import PLAN_PROMPT, PLAN_EXECUTE_PROMPT
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import Literal
from tools import tools, tools_by_name
import asyncio
import sys
import io
import threading

llm_client = myDeepSeek()
llm_with_tools = llm_client.bind_tools(tools)


class PlanState(MessagesState):
    plan: str


async def run_graph():
    async def plan_node(state: PlanState):
        # 调用 LLM
        response = await llm_client.ainvoke(
            [SystemMessage(content=PLAN_PROMPT), state["messages"][0]]
        )

        state["plan"] = response.content
        print(f"[{threading.current_thread().name}] 计划：\n{state["plan"]}")
        return state

    async def human_node(state: PlanState):
        """打断, 等待用户输入, 这里的打断, 本质上并不一定是要等待human, 也可能等其他事件
        """
        # 从 Command resume 中获得信息
        content = interrupt(state["plan"])
        print(f"[{threading.current_thread().name}] 用户输入是:", content)

        if content == "确认":
            return state
        else:
            state["plan"] = content
            return state

    async def execute_node(state: PlanState):
        """这个节点会生成 tool_calls
        """
        messages = [
            SystemMessage(content=PLAN_EXECUTE_PROMPT.format(plan=state["plan"]))
        ] + state["messages"]

        response = await llm_with_tools.ainvoke(messages)
        state["messages"].append(response)

        return state

    async def tool_node(state: PlanState):
        """调用工具, 这里是为上下文生成了 list[ToolMessage]
        """
        for tool_call in state["messages"][-1].tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            state["messages"].append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
        return state

    async def should_continue(state: PlanState) -> Literal["tool_node", "END"]:
        messages = state["messages"]
        last_message = messages[-1]
        if "Final Answer" in last_message.content:
            return "END"
        return "tool_node"

    agent_builder = StateGraph(PlanState)

    # Add nodes
    agent_builder.add_node("plan_node", plan_node)
    agent_builder.add_node("execute_node", execute_node)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("human_node", human_node)

    # Add edges
    agent_builder.add_edge(START, "plan_node")
    agent_builder.add_edge("plan_node", "human_node")
    agent_builder.add_edge("human_node", "execute_node")
    agent_builder.add_conditional_edges(
        "execute_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "END": END,
        },
    )
    agent_builder.add_edge("tool_node", "execute_node")

    # Checkpoint保存在保留图状态的线程中，并且可以在图执行完成后访问。 这允许图执行在特定点暂停，等待人类批准，然后从最后一个checkpoint恢复执行
    memory = MemorySaver()
    agent = agent_builder.compile(checkpointer=memory)

    thread_config = {"configurable": {"thread_id": "123"}}

    ret = await agent.ainvoke(
        {"plan": "", "messages": [HumanMessage(content="帮我写一份潮汕旅游攻略")]},
        config=thread_config,
    )
    ret = await agent.ainvoke(
        {"plan": "", "messages": [HumanMessage(content="帮我写一份潮汕旅游攻略")]},
        config={"configurable": {"thread_id": "a_123"}},
    )
    # 创建了 Plan 之后
    if ret["plan"] != "":
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
        get_user_input = input("请输入用户输入：")
        # 再次激活agent工作流
        ret = await agent.ainvoke(Command(resume=get_user_input), config={"configurable": {"thread_id": "a_123"}})
        # 创建了 Plan 之后
    if ret["plan"] != "":
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
        get_user_input = input("请输入用户输入：")
        # 再次激活agent工作流
        ret = await agent.ainvoke(Command(resume=get_user_input), config=thread_config)
    return ret["messages"][-1].content


if __name__ == "__main__":
    ret = asyncio.run(run_graph())
    print(ret)
