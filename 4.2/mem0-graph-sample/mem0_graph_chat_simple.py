"""
纯图谱记忆对话系统（使用 mem0 库）- 简化版
==========================================
基于 mem0 Graph Memory 文档：https://docs.mem0.ai/open-source/features/graph-memory

架构：
  - 图谱存储  : Neo4j 单节点（通过 mem0 管理）
  - 向量存储  : mem0 默认向量存储（用于记忆检索）
  - LLM       : 通义千问 DashScope（通过 mem0 管理）

mem0 Graph Memory 工作原理（根据文档）：
  1. memory.add() 会自动提取实体和关系，存储到向量数据库和图数据库
  2. memory.search() 执行向量搜索，Graph Memory 并行运行并在 relations 数组中添加相关实体
  3. relations 不会重新排序向量搜索结果，只是提供额外的上下文

依赖安装：
  pip install "mem0ai[graph]" openai python-dotenv
  或
  pip install mem0ai langchain-neo4j openai python-dotenv
"""

import os
from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

# ── 配置 ──────────────────────────────────────────────────────────────────────
TONGYI_API_KEY  = os.getenv("TONGYI_API_KEY", "")
TONGYI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

NEO4J_URL      = "bolt://121.41.120.130:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "mem0neo4j2024")

# 对话模型
CHAT_MODEL = "qwen3-max"

# ── 初始化 mem0 Memory ──────────────────────────────────────────────────────

def init_memory():
    """初始化 mem0 Memory 实例，配置 Neo4j 图存储和通义千问 LLM"""
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": "qwen3-max",
                "api_key": TONGYI_API_KEY,
                "openai_base_url": TONGYI_BASE_URL,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v3",
                "api_key": TONGYI_API_KEY,
                "openai_base_url": TONGYI_BASE_URL,
                "embedding_dims": 1024,
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test",
                "host": "121.41.120.130",
                "port": 6333,
                "embedding_model_dims": 1024
            }
        },
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": NEO4J_URL,
                "username": NEO4J_USER,
                "password": NEO4J_PASSWORD,
            }
        },
    }

    return Memory.from_config(config)


# ── 上下文格式化 ──────────────────────────────────────────────────────────────

def format_graph_context(search_results: dict) -> str:
    """将 mem0 搜索结果格式化为 System Prompt 注入文本"""
    lines = []
    
    # 提取记忆内容
    memories = search_results.get("results", [])
    if memories:
        lines.append("[历史记忆]")
        for mem in memories[:6]:  # 限制最多 6 条
            content = mem.get("memory", "") or mem.get("content", "")
            if content:
                lines.append(f"  · {content}")
    
    # 提取关系信息（根据文档，relations 格式为 [{"source": ..., "relationship": ..., "target": ...}, ...]）
    relations = search_results.get("relations", [])
    if relations:
        lines.append("\n[实体关系图谱]")
        seen_relations = set()  # 用于去重
        for rel in relations[:10]:  # 限制最多 10 条关系
            # 根据文档，关系格式为 {"source": ..., "relationship": ..., "target": ...}
            # 但实际可能使用 "destination" 而不是 "target"
            source = rel.get("source", "")
            relationship = rel.get("relationship", "") or rel.get("type", "")
            target = rel.get("target", "") or rel.get("destination", "")
            
            # 创建唯一标识符用于去重
            rel_key = (source, relationship, target)
            if rel_key in seen_relations:
                continue
            seen_relations.add(rel_key)
            
            # 过滤掉无效的关系
            if source and relationship and target and target != "unknown":
                # 过滤掉源和目标相同的关系（除非是自环关系）
                if source == target and relationship not in ["self", "same", "equals"]:
                    continue
                lines.append(f"  · {source} —[{relationship}]→ {target}")
    
    return "\n".join(lines) if lines else "（暂无相关历史记忆）"


# ── 对话核心 ──────────────────────────────────────────────────────────────────

def chat_with_graph_memory(
    llm: OpenAI,
    memory: Memory,
    user_message: str,
    user_id: str,
    conversation_history: list,
) -> str:
    """
    单轮带图谱记忆的对话

    流程：
      1. 使用 mem0 搜索关联记忆（自动返回向量结果和图谱关系）
      2. 构建含图谱上下文的 System Prompt
      3. 调用 LLM 生成回答
    """

    # ── Step 1: 使用 mem0 搜索 ────────────────────────────────────────────────
    print("\n📚 检索图谱记忆...")
    search_results = memory.search(user_message, user_id=user_id, limit=6)
    context = format_graph_context(search_results)
    print(f"   图谱上下文：\n{context}\n")

    # ── Step 2: 构建 System Prompt ───────────────────────────────
    system_prompt = f"""你是一个具有长期记忆能力的 AI 助手。
你能记住用户在历次对话中分享的信息，并在回答时自然地融入这些记忆。

以下是从记忆中检索到的、与当前问题相关的历史信息：
{context}

请根据以上背景知识，结合用户的当前问题给出准确、自然的回答。
如果记忆中没有相关信息，诚实地告知用户你不知道，而不是凭空捏造。"""

    # ── Step 3: 调用 LLM 生成回答 ───────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
    )
    assistant_reply = response.choices[0].message.content

    # ── Step 4: 更新短期上下文 ──────────────────────────────────
    conversation_history.append({"role": "user",      "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    print()
    return assistant_reply


# ── 退出时批量保存记忆 ────────────────────────────────────────────────────────

def save_memory(memory: Memory, conversation_history: list, user_id: str) -> None:
    """退出时将完整多轮对话一次性保存到 mem0（提取实体关系并写入向量库和图数据库）"""
    if not conversation_history:
        print("   （本次无对话记录，跳过保存）")
        return

    print("💾 正在保存本次会话记忆...")
    try:
        result = memory.add(conversation_history, user_id=user_id)
        memories_count = len(result.get("results", []))
        relations = result.get("relations", [])
        relations_count = len(relations) if isinstance(relations, list) else 0
        print(f"   ✅ 已保存 {memories_count} 条记忆，{relations_count} 条关系")
    except Exception as e:
        import traceback
        print(f"   ⚠️  保存失败: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            print(f"   详细错误: {traceback.format_exc()}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print("⚙️  正在初始化 mem0 Memory...")
    memory = init_memory()
    print(f"✅ mem0 已初始化\n")

    llm = OpenAI(api_key=TONGYI_API_KEY, base_url=TONGYI_BASE_URL)

    user_id = input("请输入用户名（默认: user_001）: ").strip() or "user_001"
    print(f"👤 当前用户: {user_id}\n")

    MAX_SHORT_TERM = 10
    conversation_history: list = []

    print("💡 输入 quit 退出并保存记忆\n")

    try:
        while True:
            try:
                user_input = input(f"[{user_id}] 你: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            # 限制短期上下文长度
            if len(conversation_history) > MAX_SHORT_TERM * 2:
                conversation_history = conversation_history[-(MAX_SHORT_TERM * 2):]

            reply = chat_with_graph_memory(
                llm=llm,
                memory=memory,
                user_message=user_input,
                user_id=user_id,
                conversation_history=conversation_history,
            )
            print(f"🤖 助手: {reply}\n")
    finally:
        save_memory(memory, conversation_history, user_id)
        print("👋 再见！")


if __name__ == "__main__":
    main()
