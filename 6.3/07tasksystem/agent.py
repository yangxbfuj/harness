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
TASKS_DIR = WORKDIR / ".tasks"

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def send_messages(messages):
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        #extra_body={"reasoning_split": True},
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
    {"type": "function", "function": {"name": "task_create", "description": "创建一个新的任务",
     "parameters": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}}},
    {"type": "function", "function": {"name": "task_update", "description": "更新任务的状态或依赖",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "addBlocks": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}}},
    {"type": "function", "function": {"name": "task_list", "description": "列出所有的任务状态摘要",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "task_get", "description": "通过ID获取任务的全部细节",
     "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}},
]

class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2))

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id, "subject": subject, "description": description,
            "status": "pending", "blockedBy": [], "blocks": [], "owner": "",
        }
        framed_print(f"Tool (TASK_CREATE)", subject, "success")
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

    def get(self, task_id: int) -> str:
        ret = json.dumps(self._load(task_id), indent=2)
        framed_print(f"Tool (TASK_GET)", ret, "success")
        return ret

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, add_blocks: list = None) -> str:
        task = self._load(task_id)
        
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # When a task is completed, remove it from all other tasks' blockedBy
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))
            # Bidirectional: also update the blocked tasks' blockedBy lists
            for blocked_id in add_blocks:
                try:
                    blocked = self._load(blocked_id)
                    if task_id not in blocked["blockedBy"]:
                        blocked["blockedBy"].append(task_id)
                        self._save(blocked)
                except ValueError:
                    pass
        self._save(task)

        ret = json.dumps(task, indent=2)
        framed_print(f"Tool (TASK_UPDATE)", ret, "success")

        return ret

    def _clear_dependency(self, completed_id: int):
        """Remove completed_id from all other tasks' blockedBy lists."""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        framed_print(f"Tool (TASK_LIST)", "", "success")
        tasks = []
        for f in sorted(self.dir.glob("task_*.json")):
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)

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

        #if response.choices[0].message.reasoning_details[0]['text'] != "":
        #    framed_print(f"Thinking", response.choices[0].message.reasoning_details[0]['text'], "info")

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
                elif tool_call.function.name == "task_create":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = TASKS.create(arguments_dict['subject'], arguments_dict.get('description', ''))
                elif tool_call.function.name == "task_update":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = TASKS.update(arguments_dict['task_id'], arguments_dict.get('status'), arguments_dict.get('addBlockedBy'), arguments_dict.get('addBlocks'))
                elif tool_call.function.name == "task_list":
                    result = TASKS.list_all()
                elif tool_call.function.name == "task_get":
                    arguments_dict = json.loads(tool_call.function.arguments)
                    result = TASKS.get(arguments_dict['task_id'])
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
    history = [{"role": "system", "content": "你是在 {WORKDIR} 目录下的Code Agent。在执行任务的过程中，必须使用task_create, task_update, task_list, task_get工具来规划和跟踪工作。首次执行任务必须使用task_create进行任务的创建。"}]
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
    
            