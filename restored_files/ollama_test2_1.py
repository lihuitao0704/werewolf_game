"""
完整⽰例：⽤ FastAPI 把本地模型包装成 API 服务
安装依赖：pip install fastapi uvicorn ollama
运⾏⽅式：python 04_fastapi_server.py
启动后访问：
 - GET http://localhost:8000 查看状态
 - POST http://localhost:8000/chat 发送消息
运⾏前确保：
 1. Ollama 已安装并启动
 2. 已下载模型：ollama pull qwen2.5:0.5b
"""

from fastapi import FastAPI
from pydantic import BaseModel
import ollama
# ========== 配置 ==========
MODEL = "qwen2.5:0.5b"
app = FastAPI(title="本地 AI 助⼿")
# ========== 定义请求格式 ==========
class ChatRequest(BaseModel):
 message: str # ⽤⼾消息（必填）
 system_prompt: str = "你是⼀个有⽤的助⼿" # 系统提⽰词（可选，有默认值）
class ChatResponse(BaseModel):
 answer: str # AI 回答
# ========== API 接⼝ ==========
@app.get("/")
def health():
    """健康检查——确认服务正常运⾏"""
    return {"status": "ok", "model": MODEL}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    接收⽤⼾消息，调⽤本地模型，返回回答
    使⽤⽰例（另⼀个终端中执⾏）：
    curl -X POST http://localhost:8000/chat \\
    -H "Content-Type: application/json" \\
    -d '{"message": "你好，介绍⼀下⾃⼰"}'
    """
    response = ollama.chat(
       model=MODEL,messages=[
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.message},
        ]
    )
    return ChatResponse(answer=response["message"]["content"])
# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 服务启动中... 模型: {MODEL}")
    print(f"📖 健康检查: http://localhost:8000")
    print(f"💬 发送消息: POST http://localhost:8000/chat")
    print(f"📝 测试命令: curl -X POST http://localhost:8000/chat -H 'Content-Type:application/json' -d '{{\"message\": \"你好\"}}'")
    uvicorn.run(app, host="0.0.0.0", port=8000)