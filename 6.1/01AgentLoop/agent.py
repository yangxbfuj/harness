import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
from utils import lined_print, framed_print
import sys

# 加载 .env 文件
load_dotenv()

client = OpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),  
    base_url="https://api.minimaxi.com/v1"
)

def send_messages(messages):
    response = client.chat.completions.create(
        model="MiniMax-M2.7",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        extra_body={"reasoning_split": True},
    )
    return response

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "使用该工具执行bash命令",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "bash命令",
                    }
                },
                "required": ["command"]
            },
        }
    },
]

def run_bash(command: str) -> str:
    framed_print(f"Tool (RUN_BASH)", command, "success")
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def agent_loop(messages):
    max_rounds = 10
    current_round = 0

    while True:
        current_round += 1
        lined_print(f"Calling LLM (round {current_round})")

        if current_round > max_rounds:
            print(f"Maximum rounds {max_rounds} reached, exiting")
            sys.exit(0)

        response = send_messages(messages)

        if response.choices[0].message.reasoning_details[0]['text'] != "":
            framed_print(f"Thinking", response.choices[0].message.reasoning_details[0]['text'], "info")

        if response.choices[0].message.content != "":
            framed_print(f"Answer", response.choices[0].message.content, "info")

        if response.choices[0].message.tool_calls != None:
            messages.append(response.choices[0].message)
            
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "run_bash":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = run_bash(arguments_dict['command'])
                    
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
        else:
            break

if __name__ == "__main__":
    #messages = [{"role": "user", "content": "/root/test下有哪些文件？"}]
    history = [{"role": "system", "content": "你是在 {os.getcwd()} 目录下的Code Agent。使用 bash 来完成任务。直接行动，不要解释。"}]
    while True:
        try:
            query = input("\033[36muser >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        print(response_content)
        print()
    
            