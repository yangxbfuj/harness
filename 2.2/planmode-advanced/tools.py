from langchain.agents import tool

@tool
def add(a: int, b:int) -> str:
    """执行加法计算"""
    return a+b

@tool
def multiply(a: int, b:int) -> str:
    """执行乘法计算"""
    return a*b

tools = [add, multiply]
tools_by_name = {tool.name: tool for tool in tools}