import os
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "user", "content": "大明的儿子叫小明，你了解了就可以，无需做具体回答"}
    ],
)

print("##############第一次回答################")
print(response.choices[0].message.content)
print("#######################################")

response2 = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "user", "content": "大明的儿子叫小明，你了解了就可以，无需做具体回答"},
        {"role": "assistant", "content": "我了解了"},
        {"role": "user", "content": "小明的爸爸是谁？如果不清楚，就回复我不知道"}
    ],
)

print("##############第二次回答################")
print(response2.choices[0].message.content)
print("#######################################")