import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
from utils import lined_print, framed_print
import sys
from pathlib import Path

# 加载 .env 文件
load_dotenv()

WORKDIR = Path.cwd()

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
    {"type": "function", "function": {"name": "run_bash", "description": "使用该工具执行bash命令",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "bash命令"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "run_read", "description": "使用该工具读取文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "limit": {"type": "integer", "description": "限制读取的行数"}}, "required": ["path", "limit"]}}},
    {"type": "function", "function": {"name": "run_write", "description": "使用该工具写入文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "文件内容"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_edit", "description": "使用该工具编辑文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "old_text": {"type": "string", "description": "旧文本"}, "new_text": {"type": "string", "description": "新文本"}}, "required": ["path", "old_text", "new_text"]}}}
]

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        framed_print(f"Tool (RUN_READ):{path}", f"", "success")
        return "\n".join(lines)[:50000]
    except Exception as e:
        framed_print("Readfile error", f'{e}\nRetrying...', "warning")
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        framed_print(f"Tool (RUN_WRITE):{path}", content, "success")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        framed_print("Writefile error", f'{e}\nRetrying...', "warning")
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        framed_print(f"Tool (RUN_EDIT):{path}", f"", "success")
        return f"Edited {path}"
    except Exception as e:
        framed_print("Editfile error", f'{e}\nRetrying...', "warning")
        return f"Error: {e}"

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
                elif tool_call.function.name == "run_read":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = run_read(arguments_dict['path'], arguments_dict['limit'])
                elif tool_call.function.name == "run_write":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = run_write(arguments_dict['path'], arguments_dict['content'])
                elif tool_call.function.name == "run_edit":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = run_edit(arguments_dict['path'], arguments_dict['old_text'], arguments_dict['new_text'])
                else:
                    result = "Error: Unknown tool"
                    
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })
        else:
            break

if __name__ == "__main__":
    #messages = [{"role": "user", "content": "/root/test下有哪些文件？"}]
    history = [{"role": "system", "content": "你是在 {WORKDIR} 目录下的Code Agent。使用工具进行任务的处理。直接行动，不要解释。"}]
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
    
            