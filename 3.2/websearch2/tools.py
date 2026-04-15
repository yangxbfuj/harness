"""
工具函数模块
提供 Agent 可以调用的各种工具函数，以及工具的定义配置
"""

import json

import requests


async def searxngsearch(query: str):
    """使用 SearXNG 进行网络搜索"""
    ep = "http://121.41.120.130:18080/search"
    
    params = {
        'q': query,                                    # 搜索查询关键词
        'format': 'json',                              # 返回格式：JSON
        'language': 'en',                               # 搜索语言：英语
        'pageno': 1,                                   # 页码：第1页
        'safesearch': 0,                               # 安全搜索级别：0=关闭，1=中等，2=严格
        'engines': '360search,baidu,quark,sogou',      # 使用的搜索引擎列表
    }
    
    try:
        response = requests.get(ep, params=params)
        
        return response.json()
            
    except requests.exceptions.RequestException as e:
        return {"error": f"网络请求失败: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON 解析失败: {str(e)}"}
    except Exception as e:
        return {"error": f"未知错误: {str(e)}"}

tools = [
    {
        "type": "function",
        "function": {
            "name": "searxngsearch",
            "description": "使用该工具进行网络搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"]
            },
        }
    },
]
