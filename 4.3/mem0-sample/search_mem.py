import os
import json
import hashlib
import dotenv
import redis
from mem0 import Memory

dotenv.load_dotenv()

# ── Redis 连接 ──────────────────────────────────────────────
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))  # 默认缓存 1 小时

# ── mem0 配置 ───────────────────────────────────────────────
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3-max",
            "api_key": os.environ.get("TONGYI_API_KEY"),
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",
            "api_key": os.environ.get("TONGYI_API_KEY"),
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "embedding_dims": 1024,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1024,
        },
    },
}

m = Memory.from_config(config)

# ── 查询参数 ────────────────────────────────────────────────
query = "小明的爸爸喜欢喝什么饮料？"
user_id = "xyy"

# ── Cache-Aside 读流程 ──────────────────────────────────────
cache_key = f"mem0:search:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"

cached = redis_client.get(cache_key)
if cached:
    print("[Cache HIT] 直接从 Redis 返回结果")
    related_memories = json.loads(str(cached))
else:
    print("[Cache MISS] 查询 Qdrant ...")
    related_memories = m.search(query=query, filters={"user_id": user_id}, top_k=5)
    # 回写 Redis，设置 TTL
    redis_client.setex(
        cache_key, CACHE_TTL, json.dumps(related_memories, ensure_ascii=False)
    )
    print(f"[Cache SET] 结果已缓存，TTL={CACHE_TTL}s")

print(related_memories)
