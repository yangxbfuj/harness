"""
Agent 主程序
实现了一个支持工具调用的对话 Agent，可以自动调用网络搜索工具来回答问题
"""

import json
import os
from llm import send_messages
from tools import searxngsearch
import asyncio

async def main(query: str) -> str:
    """
    主函数：处理用户查询，支持多轮对话和工具调用
    
    Args:
        query: 用户输入的查询字符串
    """
    # 初始化消息列表，第一条为用户消息
    messages = [{"role": "user", "content": query}]
    
    # 循环处理对话，直到 LLM 不再调用工具
    while True:
        # 发送消息到 LLM 并获取响应
        response = await send_messages(messages)

        # 如果 LLM 决定调用工具
        if response.choices[0].message.tool_calls != None:
            # 将 assistant 的消息添加到对话历史中
            messages.append(response.choices[0].message)
            
            # 遍历所有工具调用
            for tool_call in response.choices[0].message.tool_calls:
                # 处理网络搜索工具调用
                if tool_call.function.name == "searxngsearch":
                    print("工具名称：", tool_call.function.name)
                    # 解析工具调用的参数（JSON 格式）
                    arguments_dict = json.loads(tool_call.function.arguments)
                    print("工具参数：", arguments_dict['query'])
                    # 调用网络搜索工具
                    search_result = await searxngsearch(arguments_dict['query'])
                    print(search_result)
                    
                    # 确保 content 是字符串格式（API 要求）
                    if isinstance(search_result, dict):
                        # 如果是字典，转换为 JSON 字符串
                        content = json.dumps(search_result, ensure_ascii=False)
                    else:
                        # 其他类型直接转换为字符串
                        content = str(search_result)
                    
                    # 将工具执行结果添加到对话历史中
                    messages.append({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call.id
                    })
        else:
            # 如果没有工具调用，说明对话结束，退出循环
            break
    return response.choices[0].message.content

if __name__ == "__main__":
    # 运行主函数，处理用户查询
    ret = asyncio.run(main("中国在2026年米兰冬奥会获得了几枚金牌？"))
    print("最终结果：")
    print(ret)
            