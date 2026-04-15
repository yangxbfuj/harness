import json
import os

import requests
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import tool
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

@tool
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

@tool
async def widesearch_for_toolstr(query:str):
    """
    使用searx搜索工具进行网络搜索。
    参数:
        query (str): 搜索查询字符串。
    返回:
        str: 搜索结果的markdown格式化字符串，每个结果包含标题、简介和链接。
    """

    print(f"搜索查询字符串: {query}")
    
    engines = ["baidu", "sogou", "quark"]
    search = SearxSearchWrapper(
        searx_host="http://localhost:18080/",
    )  # k用于最大项目数
    search_ret = search.results(query, num_results=10,
                         time_range="year",
                         engines=engines)
    strtemplate = """
    标题:{}
    简介:{}
    链接:{}

    """

    ret = ""
    for data in search_ret:
        # 安全地获取字段值，如果不存在则使用默认值
        title = data.get("title", "无标题")
        snippet = data.get("snippet", "无简介")
        link = data.get("link", "无链接")

        ret += strtemplate.format(title, snippet, link)
    return ret

