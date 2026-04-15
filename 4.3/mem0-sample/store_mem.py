import os
import dotenv
import redis
from mem0 import Memory

dotenv.load_dotenv()

# ── Redis 连接 ──────────────────────────────────────────────
redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "121.41.120.130"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

# ── mem0 配置 ───────────────────────────────────────────────
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "qwen3-max",
            "api_key": os.environ.get("TONGYI_API_KEY"),
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-v3",
            "api_key": os.environ.get("TONGYI_API_KEY"),
            "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "embedding_dims": 1024
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

m = Memory.from_config(config)

user_id = "xyy"
messages = [
    {"role": "user", "content": "小张和小明是什么关系？"},
    {"role": "assistant", "content": "小张是小明的爸爸"},
    {"role": "user", "content": "小张喜欢喝什么饮料？"},
    {"role": "assistant", "content": "小张喜欢喝大窑。"}
]

# ── 写入 Qdrant ─────────────────────────────────────────────
ret = m.add(messages, user_id=user_id, metadata={"category": "drink"}, infer=False)
print(ret)

# ── Cache Invalidation：清除该用户的所有查询缓存 ────────────
pattern = f"mem0:search:{user_id}:*"
keys = redis_client.keys(pattern)
if keys:
    redis_client.delete(*keys)
    print(f"[Cache INVALIDATE] 已清除 {len(keys)} 条缓存 key")
else:
    print("[Cache INVALIDATE] 该用户暂无缓存，无需清除")
