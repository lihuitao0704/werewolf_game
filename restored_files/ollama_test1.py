"""
完整示例：使用 ollama Python 包调用本地模型
安装依赖：pip install ollama
运行前确保：
  1. Ollama 已安装并启动
  2. 已下载模型：ollama pull qwen2.5:0.5b
运行方式：python 02_ollama_chat.py
"""
import ollama

MODEL = "qwen2.5:0.5b"

# ========== 方式一：简单问答（单次调用） ==========
print("=== 简单问答 ===")
response = ollama.generate(
    model=MODEL,
    prompt="用一句话解释什么是 Token"
)
print(response["response"])

# ========== 方式二：多轮对话（带角色设定） ==========
print("\n=== 多轮对话 ===")

# 第一轮
print("=== 第一轮对话 ===")
messages = [
    {"role": "system", "content": "你是一个耐心的编程老师，用简单的话解释概念，每个回答不超过 3 句话"},
    {"role": "user", "content": "什么是 Token？"},
]

response = ollama.chat(model=MODEL, messages=messages)
answer = response["message"]["content"]
print("AI:", answer)

# 第二轮（带上之前的对话历史，AI 才知道"上下文"）
print("=== 第二轮对话 ===")
messages.append({"role": "assistant", "content": answer})  # 把 AI 的回答加入历史
messages.append({"role": "user", "content": "那上下文窗口又是什么？"})

response = ollama.chat(model=MODEL, messages=messages)
print("AI:", response["message"]["content"])

# 💡 注意：每次调用都是"无状态"的——对话历史需要你自己维护
# 这就是术语手册中 Context Engineering（上下文工程）要解决的问题