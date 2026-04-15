import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
from utils import lined_print, framed_print
import sys
import time
from pathlib import Path

# 加载 .env 文件
load_dotenv()

WORKDIR = Path.cwd()
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
THRESHOLD = 1000

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
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "old_text": {"type": "string", "description": "旧文本"}, "new_text": {"type": "string", "description": "新文本"}}, "required": ["path", "old_text", "new_text"]}}},
     {"type": "function", "function": {"name": "compact", "description": "触发手工的对话压缩",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "摘要中需要保留的内容"}}}}},
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

def micro_compact(messages: list) -> list:
    tool_results: list[tuple[int, dict]] = []
    # Collect tool result messages (these are appended as dicts in agent_loop).
    for msg_idx, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            tool_results.append((msg_idx, msg))

    print(f"[micro_compact] Found {len(tool_results)} tool result messages")
    
    if len(tool_results) <= KEEP_RECENT:
        print(f"[micro_compact] No compression needed (<= {KEEP_RECENT} tool results)")
        return messages

    print(f"[micro_compact] Compressing: keeping last {KEEP_RECENT}, clearing {len(tool_results) - KEEP_RECENT} old tool results")

    # Map tool_call_id -> tool function name from prior assistant messages.
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "assistant" or not hasattr(msg, "tool_calls") or not getattr(msg, "tool_calls", None):
            continue
        for tool_call in msg.tool_calls:
            tool_call_id = getattr(tool_call, "id", None)
            tool_fn_name = getattr(getattr(tool_call, "function", None), "name", None)
            if tool_call_id and tool_fn_name:
                tool_name_map[str(tool_call_id)] = str(tool_fn_name)

    # Clear old results (keep last KEEP_RECENT), replacing content with a placeholder.
    to_clear = tool_results[:-KEEP_RECENT]
    cleared_count = 0
    for idx, result in to_clear:
        content = result.get("content")
        if isinstance(content, str) and len(content) > 100:
            tool_id = str(result.get("tool_call_id", ""))
            tool_name = tool_name_map.get(tool_id, "unknown")
            old_content_preview = content[:50].replace("\n", " ") + "..."
            print(f"[micro_compact] Replacing {tool_name} result (was {len(content)} chars): {old_content_preview}")
            result["content"] = f"[Previous: used {tool_name}]"
            cleared_count += 1
    
    print(f"[micro_compact] Compression complete: {cleared_count} tool results replaced with placeholders")

    return messages

# -- Layer 2: auto_compact - save transcript, summarize, replace messages --
def auto_compact(messages: list) -> list:
    # Save full transcript to disk
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")
    # Ask LLM to summarize
    conversation_text = json.dumps(messages, default=str)[:80000]
    message = """
    Summarize this conversation for continuity. Include: 
    1) What was accomplished, 2) Current state, 3) Key decisions made. 
    Be concise but preserve critical details.\n\n
""" + conversation_text
    response = send_messages(messages)
    summary = response.choices[0].message.content 
    # Replace all messages with compressed summary
    return [
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
        {"role": "assistant", "content": "Understood. I have the context from the summary. Continuing."},
    ]

def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 chars per token."""
    return len(str(messages)) // 4

def agent_loop(messages):
    current_round = 0

    while True:
        # Layer 1: micro_compact before each LLM call
        micro_compact(messages)
        # Layer 2: auto_compact if token estimate exceeds threshold
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)
        current_round += 1
        lined_print(f"Calling LLM (round {current_round})")

        response = send_messages(messages)

        if response.choices[0].message.reasoning_details[0]['text'] != "":
            framed_print(f"Thinking", response.choices[0].message.reasoning_details[0]['text'], "info")

        if response.choices[0].message.content != "":
            framed_print(f"Answer", response.choices[0].message.content, "info")

        if response.choices[0].message.tool_calls != None:
            messages.append(response.choices[0].message)
            manual_compact = False
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
                elif tool_call.function.name == "compact":
                    manual_compact = True
                    result = "Compressing..."
                else:
                    result = "Error: Unknown tool"
                    
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id
                })

            if manual_compact:
                print("[manual compact]")
                messages[:] = auto_compact(messages)
        else:
            break

if __name__ == "__main__":
    #messages = [{"role": "user", "content": "/root/test下有哪些文件？"}]
    history = [{"role": "system", "content": "你是在 {WORKDIR} 目录下的Code Agent。使用工具来完成任务。直接行动，不要解释。"}]
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
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
    
            