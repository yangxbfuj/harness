import json
import os
import subprocess
import threading
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from utils import lined_print, framed_print

load_dotenv()

WORKDIR = Path.cwd()

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

client = OpenAI(
    api_key=os.getenv("TONGYI_API_KEY"),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def send_messages(messages,tools):
    response = client.chat.completions.create(
        model="qwen3-max",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        #extra_body={"reasoning_split": True},
    )
    return response

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        framed_print(f"Tool (SEND)", f"Sent {msg_type} from {sender} to {to}\nContent: {content[:100]}..." if len(content) > 100 else f"Sent {msg_type} from {sender} to {to}\nContent: {content}", "success")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        if messages:
            framed_print(f"Tool (READ_INBOX)", f"{name} received {len(messages)} message(s)", "info")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        framed_print(f"Tool (BROADCAST)", f"Broadcast from {sender} to {count} teammates", "success")
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: persistent named agents with config.json --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
            framed_print(f"Leader (SPAWN)", f"Reactivating '{name}' (role: {role})", "info")
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
            framed_print(f"Leader (SPAWN)", f"Creating new teammate '{name}' (role: {role})", "info")
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        if prompt:
            BUS.send("lead", name, f"Your initial task: {prompt}", "message")
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        lined_print(f"[{name}] Thread Started")
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        tools = self._teammate_tools()
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            try:
                response = send_messages(messages,tools)
            except Exception:
                break
           
            if response.choices[0].message.content != "":
                framed_print(f"[{name}] Answer", response.choices[0].message.content, "info")

            if response.choices[0].message.tool_calls is None:
                break

            messages.append(response.choices[0].message)

            for tool_call in response.choices[0].message.tool_calls:
                results = self._exec(name, tool_call.function.name, json.loads(tool_call.function.arguments))

                messages.append({
                    "role": "tool",
                    "content": str(results),
                    "tool_call_id": tool_call.id
                })

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()
            lined_print(f"[{name}] Thread Finished (status: idle)")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        # these base tools are unchanged from s02
        return [
            {"type": "function", "function": {"name": "bash", "description": "使用该工具执行bash命令",
             "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "bash命令"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "使用该工具读取文件",
             "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "limit": {"type": "integer", "description": "限制读取的行数"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "使用该工具写入文件",
             "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "文件内容"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "使用该工具编辑文件",
             "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "old_text": {"type": "string", "description": "旧文本"}, "new_text": {"type": "string", "description": "新文本"}}, "required": ["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "send_message", "description": "使用该工具发送消息给队友",
             "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "收件人"}, "content": {"type": "string", "description": "消息内容"}, "msg_type": {"type": "string", "description": "消息类型", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
            {"type": "function", "function": {"name": "read_inbox", "description": "使用该工具读取并清空你的收件箱",
             "parameters": {"type": "object", "properties": {}}}},
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead tool dispatch (9 tools) --
TOOL_HANDLERS = {
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

# these base tools are unchanged from s02
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "使用该工具执行bash命令",
     "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "bash命令"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "使用该工具读取文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "limit": {"type": "integer", "description": "限制读取的行数"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "使用该工具写入文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "content": {"type": "string", "description": "文件内容"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "使用该工具编辑文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "文件路径"}, "old_text": {"type": "string", "description": "旧文本"}, "new_text": {"type": "string", "description": "新文本"}}, "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "spawn_teammate", "description": "使用该工具创建运行在线程中的持久化队友",
     "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "队友名称"}, "role": {"type": "string", "description": "角色"}, "prompt": {"type": "string", "description": "初始提示词"}}, "required": ["name", "role", "prompt"]}}},
    {"type": "function", "function": {"name": "list_teammates", "description": "使用该工具列出所有队友的名称、角色和状态",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "send_message", "description": "使用该工具发送消息到队友的收件箱",
     "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "收件人"}, "content": {"type": "string", "description": "消息内容"}, "msg_type": {"type": "string", "description": "消息类型", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}}},
    {"type": "function", "function": {"name": "read_inbox", "description": "使用该工具读取并清空你的收件箱",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "broadcast", "description": "使用该工具向所有队友发送消息",
     "parameters": {"type": "object", "properties": {"content": {"type": "string", "description": "消息内容"}}, "required": ["content"]}}},
]


def agent_loop(messages: list):
    lined_print("Calling LLM (Lead)")
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            framed_print(f"Lead Inbox", f"Received {len(inbox)} message(s)", "info")
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
            messages.append({
                "role": "assistant",
                "content": "Noted inbox messages.",
            })
        response = send_messages(messages, TOOLS)

        if response.choices[0].message.content != "":
            framed_print(f"Lead Answer", response.choices[0].message.content, "info")

        if response.choices[0].message.tool_calls is not None:
            messages.append(response.choices[0].message)

            for tool_call in response.choices[0].message.tool_calls:
                arguments_dict = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "bash":
                    result = _run_bash(arguments_dict['command'])
                elif tool_call.function.name == "read_file":
                    result = _run_read(arguments_dict['path'], arguments_dict.get('limit'))
                elif tool_call.function.name == "write_file":
                    result = _run_write(arguments_dict['path'], arguments_dict['content'])
                elif tool_call.function.name == "edit_file":
                    result = _run_edit(arguments_dict['path'], arguments_dict['old_text'], arguments_dict['new_text'])
                elif tool_call.function.name == "spawn_teammate":
                    result = TEAM.spawn(arguments_dict['name'], arguments_dict['role'], arguments_dict['prompt'])
                elif tool_call.function.name == "list_teammates":
                    result = TEAM.list_all()
                elif tool_call.function.name == "send_message":
                    result = BUS.send("lead", arguments_dict['to'], arguments_dict['content'], arguments_dict.get('msg_type', 'message'))
                elif tool_call.function.name == "read_inbox":
                    result = json.dumps(BUS.read_inbox("lead"), indent=2)
                elif tool_call.function.name == "broadcast":
                    result = BUS.broadcast("lead", arguments_dict['content'], TEAM.member_names())
                else:
                    result = "Error: Unknown tool"
                messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": tool_call.id
                })
        else:
            break


if __name__ == "__main__":
    history = [
        {"role": "system", "content": SYSTEM}
    ]
    while True:
        try:
            query = input("\033[36muser >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            framed_print(f"Team Status", TEAM.list_all(), "info")
            continue
        if query.strip() == "/inbox":
            inbox = BUS.read_inbox("lead")
            if inbox:
                framed_print(f"Lead Inbox", json.dumps(inbox, indent=2), "info")
            else:
                print("No messages in inbox")
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
