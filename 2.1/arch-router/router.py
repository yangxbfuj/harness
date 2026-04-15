from langchain_ollama import OllamaLLM
from typing import Dict, Any
import json

TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<conversation>

{conversation}

</conversation>
"""

FORMAT_PROMPT = """
Your task is to decide which route is best suit with user intent on the conversation in <conversation></conversation> XML tags.  Follow the instruction:
1. If the latest intent from user is irrelevant or user intent is full filled, response with other route {"route": "other"}.
2. You must analyze the route descriptions and find the best match route for user latest intent. 
3. You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats if you decide to match any route:
{"route": "route_name"} 
"""

route_config = [
    {
        "name": "sports",
        "description": "运动，体育相关主题",
    },
    {
        "name": "weather",
        "description": "天气相关主题",
    },
    {
        "name": "news",
        "description": "新闻相关主题",
    },
]

def format_prompt(route_config: list[Dict[str, Any]], conversation: list[Dict[str, Any]]):
    return (
        TASK_INSTRUCTION.format(
            routes=json.dumps(route_config), 
            conversation=json.dumps(conversation)
        ) +
        FORMAT_PROMPT
    )

def main():
    llm = OllamaLLM(base_url="http://localhost:8889", 
                    model="archrouter:latest")
    
    conversation = [
        {
            "role": "user",
            "content": "我想了解今天的天气情况",
        }
    ]

    messages = [
        {
            "role": "user",
            "content": format_prompt(route_config, conversation),
        }
    ]

    print(llm.invoke(messages))

if __name__ == "__main__":
    main()