"""
带Redis缓存的记忆对话系统
=========================
基于 mem0 的长期记忆对话系统，集成 Redis 缓存层以提升检索性能

架构：
  - 记忆存储  : Qdrant 向量数据库（通过 mem0 管理）
  - 缓存层    : Redis（用于加速记忆检索）
  - LLM       : 通义千问 DashScope（通过 mem0 管理）

缓存策略：
  - Cache-Aside 读取：先查 Redis 缓存，未命中则查询 Qdrant 并回写缓存
  - Write-Through 写入：添加新记忆时自动失效相关缓存

依赖安装：
  pip install mem0ai redis openai python-dotenv
"""

import os
import json
import hashlib
from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv
import redis

load_dotenv()

# ── 配置 ──────────────────────────────────────────────────────────────────────
TONGYI_API_KEY  = os.getenv("TONGYI_API_KEY", "")
TONGYI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Redis 连接配置
REDIS_HOST = os.environ.get("REDIS_HOST", "121.41.120.130")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))  # 默认缓存 1 小时

# 对话模型
CHAT_MODEL = "qwen3-max"

# ── Redis 客户端 ──────────────────────────────────────────────────────────────

def get_redis_client():
    """获取 Redis 客户端实例"""
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True
    )

# ── 初始化 mem0 Memory ──────────────────────────────────────────────────────

def init_memory():
    """初始化 mem0 Memory 实例，配置 Qdrant 向量存储和通义千问 LLM"""
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
        }
    }
    return Memory.from_config(config)


# ── 带缓存的记忆检索 ──────────────────────────────────────────────────────────

def search_with_cache(
    memory: Memory,
    redis_client: redis.Redis,
    query: str,
    user_id: str,
    limit: int = 6
) -> list:
    """
    带缓存的记忆搜索（Cache-Aside 模式）

    流程：
      1. 生成缓存 key
      2. 先查 Redis 缓存
      3. 缓存命中直接返回，未命中则查询 Qdrant 并回写缓存
    """
    # 生成缓存 key
    cache_key = f"mem0:search:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"

    # 尝试从缓存读取
    cached = redis_client.get(cache_key)
    if cached:
        print("[Cache HIT] 直接从 Redis 返回结果")
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            # 缓存数据损坏，重新查询
            print("[Cache ERROR] 缓存数据损坏，重新查询...")
            redis_client.delete(cache_key)

    # 缓存未命中，查询 Qdrant
    print(f"[Cache MISS] 查询 Qdrant (query: {query})...")
    search_response = memory.search(query=query, user_id=user_id, limit=limit)

    # mem0.search() 返回字典格式 {'results': [...], ...}
    results = search_response.get('results', []) if isinstance(search_response, dict) else search_response

    # 回写缓存，设置 TTL
    redis_client.setex(cache_key, CACHE_TTL, json.dumps(results, ensure_ascii=False))
    print(f"[Cache SET] 结果已缓存，TTL={CACHE_TTL}s")

    return results


# ── 缓存失效 ────────────────────────────────────────────────────────────────

def invalidate_user_cache(redis_client: redis.Redis, user_id: str) -> int:
    """
    失效指定用户的所有查询缓存

    在添加新记忆后调用，确保后续查询能获取最新数据
    """
    pattern = f"mem0:search:{user_id}:*"
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)
        print(f"[Cache INVALIDATE] 已清除 {len(keys)} 条缓存 key")
        return len(keys)
    else:
        print("[Cache INVALIDATE] 该用户暂无缓存，无需清除")
        return 0


# ── 上下文格式化 ──────────────────────────────────────────────────────────────

def format_memory_context(search_results: list) -> str:
    """将 mem0 搜索结果格式化为 System Prompt 注入文本"""
    if not search_results:
        return "（暂无相关历史记忆）"

    lines = ["[历史记忆]"]
    for mem in search_results:
        # 处理两种情况：mem 是字典或字符串
        if isinstance(mem, dict):
            content = mem.get("memory", "") or mem.get("content", "")
        elif isinstance(mem, str):
            content = mem
        else:
            content = str(mem)

        if content:
            lines.append(f"  · {content}")

    return "\n".join(lines) if len(lines) > 1 else "（暂无相关历史记忆）"


# ── 对话核心 ──────────────────────────────────────────────────────────────────

def chat_with_cached_memory(
    llm: OpenAI,
    memory: Memory,
    redis_client: redis.Redis,
    user_message: str,
    user_id: str,
    conversation_history: list,
) -> str:
    """
    单轮带缓存的记忆对话

    流程：
      1. 使用带缓存的 mem0 搜索关联记忆
      2. 构建含记忆上下文的 System Prompt
      3. 调用 LLM 生成回答
    """
    # ── Step 1: 使用带缓存的记忆搜索 ────────────────────────────────────────
    print("\n📚 检索记忆...")
    search_results = search_with_cache(
        memory=memory,
        redis_client=redis_client,
        query=user_message,
        user_id=user_id,
        limit=6
    )
    context = format_memory_context(search_results)
    print(f"   记忆上下文：\n{context}\n")

    # ── Step 2: 构建 System Prompt ─────────────────────────────────────────────
    system_prompt = f"""你是一个具有长期记忆能力的 AI 助手。
你能记住用户在历次对话中分享的信息，并在回答时自然地融入这些记忆。

以下是从记忆中检索到的、与当前问题相关的历史信息：
{context}

请根据以上背景知识，结合用户的当前问题给出准确、自然的回答。
如果记忆中没有相关信息，诚实地告知用户你不知道，而不是凭空捏造。"""

    # ── Step 3: 调用 LLM 生成回答 ─────────────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    response = llm.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
    )
    assistant_reply = response.choices[0].message.content

    # ── Step 4: 更新短期上下文 ───────────────────────────────────────────────
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": assistant_reply})

    print()
    return assistant_reply


# ── 退出时批量保存记忆 ────────────────────────────────────────────────────────

def save_memory(
    memory: Memory,
    redis_client: redis.Redis,
    conversation_history: list,
    user_id: str
) -> None:
    """
    退出时将完整多轮对话一次性保存到 mem0，并失效相关缓存

    流程：
      1. 将对话历史保存到 Qdrant
      2. 失效该用户的所有查询缓存
    """
    if not conversation_history:
        print("   （本次无对话记录，跳过保存）")
        return

    print("💾 正在保存本次会话记忆...")
    try:
        result = memory.add(conversation_history, user_id=user_id)
        memories_count = len(result.get("results", []))
        print(f"   ✅ 已保存 {memories_count} 条记忆")

        # 失效相关缓存
        invalidated_count = invalidate_user_cache(redis_client, user_id)
        print(f"   🔄 已失效 {invalidated_count} 条缓存记录")

    except Exception as e:
        import traceback
        print(f"   ⚠️  保存失败: {e}")
        if os.getenv("DEBUG", "").lower() == "true":
            print(f"   详细错误: {traceback.format_exc()}")


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    print("⚙️  正在初始化 mem0 Memory 和 Redis 连接...")
    memory = init_memory()
    redis_client = get_redis_client()
    print(f"✅ mem0 已初始化")
    print(f"✅ Redis 已连接 (host={REDIS_HOST}, port={REDIS_PORT})\n")

    llm = OpenAI(api_key=TONGYI_API_KEY, base_url=TONGYI_BASE_URL)

    user_id = input("请输入用户名（默认: user_001）: ").strip() or "user_001"
    print(f"👤 当前用户: {user_id}\n")

    MAX_SHORT_TERM = 10
    conversation_history: list = []

    print("💡 输入 quit 退出并保存记忆")
    print("💡 输入 cache <query> 测试缓存（例如: cache 我喜欢吃什么）")
    print()

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

            # 测试缓存命令
            if user_input.startswith("cache "):
                test_query = user_input[6:].strip()
                if test_query:
                    print("\n🧪 测试缓存检索...")
                    results = search_with_cache(
                        memory=memory,
                        redis_client=redis_client,
                        query=test_query,
                        user_id=user_id,
                        limit=6
                    )
                    print("📋 检索结果:")
                    for i, mem in enumerate(results, 1):
                        # 处理两种情况：mem 是字典或字符串
                        if isinstance(mem, dict):
                            content = mem.get("memory", "") or mem.get("content", "")
                        elif isinstance(mem, str):
                            content = mem
                        else:
                            content = str(mem)
                        print(f"   {i}. {content}")
                    print()
                continue

            # 限制短期上下文长度
            if len(conversation_history) > MAX_SHORT_TERM * 2:
                conversation_history = conversation_history[-(MAX_SHORT_TERM * 2):]

            reply = chat_with_cached_memory(
                llm=llm,
                memory=memory,
                redis_client=redis_client,
                user_message=user_input,
                user_id=user_id,
                conversation_history=conversation_history,
            )
            print(f"🤖 助手: {reply}\n")

    finally:
        save_memory(memory, redis_client, conversation_history, user_id)
        print("👋 再见！")


if __name__ == "__main__":
    main()
