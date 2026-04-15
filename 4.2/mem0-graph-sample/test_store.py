import os
from mem0 import Memory
import dotenv
dotenv.load_dotenv()

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
            "url": "bolt://121.41.120.130:7687",
            "username": "neo4j",
            "password": "mem0neo4j2024",
        },
    },
}

memory = Memory.from_config(config)

conversation = [
    {"role": "user", "content": "大明的儿子叫小明"},
    {"role": "assistant", "content": "好的，我会记住的"},
    {"role": "user", "content": "大明在阿里云做产品经理工作"},
    {"role": "assistant", "content": "好的，我会记住的"},
    {"role": "user", "content": "大明的儿子喜欢和可口可乐"},
    {"role": "assistant", "content": "好的，我会记住的"},
]

memory.add(conversation, user_id="demo-user")