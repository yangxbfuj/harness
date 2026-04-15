import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tools import tools

# 加载 .env 文件
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def send_messages(messages):
    response = await client.chat.completions.create(
        model="qwen-max",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    return response