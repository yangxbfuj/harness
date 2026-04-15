"""
工具函数模块
提供 Agent 可以调用的各种工具函数，以及工具的定义配置
"""

import json
import os

import requests

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

async def bochasearch(query: str):
    """
    使用 Bocha AI 进行网络搜索
    
    Args:
        query: 搜索关键词字符串
        
    Returns:
        dict: 搜索结果（JSON 格式），如果出错则返回包含错误信息的字典
    """
    # 从环境变量中获取 API 密钥
    bochakey = os.getenv("BOCHA_API_KEY")
    # Bocha AI 网络搜索 API 端点
    ep = "https://api.bochaai.com/v1/web-search"
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {bochakey}",
        "Content-Type": "application/json"
    }
    
    # 构建请求数据
    data = {
        "query": query,      # 搜索关键词
        "summary": True,      # 返回摘要
        "count": 10,         # 返回结果数量
    }
    
    # 发送 POST 请求到 API
    response = requests.post(ep,
                             data=json.dumps(data),
                             headers=headers)
    
    # 尝试解析 JSON 响应
    try:
        return response.json()
    except Exception as e:
        # 如果解析失败，返回错误信息
        return {"error": str(e)}

# 工具定义列表，用于 LLM 的工具调用功能
# 这个列表定义了 Agent 可以使用的工具及其参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "bochasearch",                    # 工具函数名称
            "description": "使用该工具进行网络搜索",  # 工具描述，LLM 会根据这个决定是否调用
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",  # 参数描述
                    }
                },
                "required": ["query"]                 # 必需参数列表
            },
        }
    },
]