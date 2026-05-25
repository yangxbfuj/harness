from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph import START, END
from regex import P
from llm import Tongyi, DeepSeek
from prompts import PLAN_PROMPT, PLAN_EXECUTE_PROMPT
from typing_extensions import Literal
from tools import tools, tools_by_name

my_llm = DeepSeek()
llm_with_tools = my_llm.bind_tools(tools)


class PlanState(MessagesState):
    """规划状态，包含以下字段, 在某个角度上看，可以认为是上下文对象：
    - messages: 消息列表，包含用户输入、工具调用和工具返回的结果
    - plan: 规划的旅游攻略大纲
    """

    plan: str


def plan_node(state: PlanState):
    """规划节点
    1. 根据用户撰写大纲
    """

    print("-" * 50 + "plan" + "-" * 50)
    print("用户输入：\n" + state["messages"][0].content)  # type: ignore
    # 调用 LLM
    response = my_llm.invoke([SystemMessage(content=PLAN_PROMPT), state["messages"][0]])

    state["plan"] = str(response.content)
    print("计划：\n" + state["plan"])
    return state


def execute_node(state: PlanState):
    """执行节点
    1. 查阅和收集数据
    2. 根据用户数据和手机到的数据，撰写旅游攻略

    """
    print("-" * 50 + "execute" + "-" * 50)
    messages = [
        SystemMessage(content=PLAN_EXECUTE_PROMPT.format(plan=state["plan"]))
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)
    state["messages"].append(response)
    print("执行结果：\n" + str(state["messages"]))  # type: ignore
    return state


def tool_node(state: PlanState):
    """工具节点
    1. 执行工具, 将数据添加到上下文中
    """
    print("-" * 50 + "tool" + "-" * 50)
    for tool_call in state["messages"][-1].tool_calls:  # type: ignore
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])  # type: ignore
        state["messages"].append(
            ToolMessage(content=observation, tool_call_id=tool_call["id"])
        )
        print("执行结果：\n" + str(state["messages"]))
    return state


def should_continue(state: PlanState) -> Literal["tool_node", "END"]:
    """判断是否需要继续执行工具节点"""

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

# Add edges
agent_builder.add_edge(START, "plan_node")
agent_builder.add_edge("plan_node", "execute_node")
agent_builder.add_conditional_edges(
    "execute_node",
    should_continue,
    {
        "tool_node": "tool_node",
        "END": END,
    },
)
agent_builder.add_edge("tool_node", "execute_node")

agent = agent_builder.compile()

ret = agent.invoke(
    {"plan": "", "messages": [HumanMessage(content="帮我写一份潮汕旅游攻略")]}
)
print(ret["messages"][-1].content)
