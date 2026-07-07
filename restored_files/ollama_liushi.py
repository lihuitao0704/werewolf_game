"""
完整示例：FastAPI + Ollama 流式输出API
安装依赖：pip install fastapi uvicorn ollama
运行方式：python 04_fastapi_server.py
启动后访问：
 - GET http://localhost:8000 查看状态
 - POST http://localhost:8000/chat 普通一次性返回
 - POST http://localhost:8000/stream_chat 流式逐字输出
文档测试地址：http://localhost:8000/docs
运行前确保：
 1. Ollama 已安装并启动
 2. 已下载模型：ollama pull qwen2.5:0.5b
"""
import json
import time
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from fastapi.responses import StreamingResponse

# ========== 配置 ==========
MODEL = "qwen2.5:0.5b"
app = FastAPI(title="本地AI助手-流式输出版")

# ========== 定义请求格式 ==========
class ChatRequest(BaseModel):
    message: str  # 用户消息（必填）
    system_prompt: str = "你是一个有用的助手"  # 系统提示词（可选，有默认值）
    stream :bool = False

class ChatResponse(BaseModel):      
    answer: Optional[str] = None  # AI完整回答

# ========== API 接口 ==========
@app.get("/")
def health():
    """健康检查——确认服务正常运行"""
    return {"status": "ok", "model": MODEL}

# 原有一次性返回接口（保留不变）
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """一次性返回完整回答"""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": req.system_prompt},
            {"role": "user", "content": req.message},
        ]

    )
    return ChatResponse(answer=response["message"]["content"])

## 生成流式迭代器（SSE数据生成）
def stream_chat_generator(model: str, messages: list):
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        # 每次返回一段文本，SSE标准格式
        content = chunk["message"]["content"]
        yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
    # 流结束标记
    yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

#Ollama 流式 / 非流式自动切换
@app.post("/chat/stream")
def chat(req: ChatRequest):
    """
    统一对话接口
    stream=True: SSE流式分段输出
    stream=False: 一次性返回完整字符串JSON
    """ 
    messages = [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.message},
    ]

    # 流式分支
    if req.stream:
        return StreamingResponse(
            stream_chat_generator(MODEL, messages),
            media_type="text/event-stream"
        )
    # 非流式原有逻辑
    else:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=False
        )
        return ChatResponse(answer=response["message"]["content"])


# # 新增流式对话接口
# @app.post("/stream_chat")
# def stream_chat(req: ChatRequest):
#     """流式逐字输出接口，文档/curl均可测试"""
#     def generate_stream():
#         # 开启ollama流式迭代
#         stream = ollama.chat(
#             model=MODEL,
#             messages=[
#                 {"role": "system", "content": req.system_prompt},
#                 {"role": "user", "content": req.message},
#             ],
#             stream=True
#         )
#         # 逐块返回文本
#         for chunk in stream:
#             # 每次返回分片内容，加换行方便前端分割
#             content = chunk["message"]["content"]
#             yield content

#     # 返回文本流式响应
#     return StreamingResponse(generate_stream(), media_type="text/plain")


# SSE标准流式接口 + 延时，肉眼逐字展示
@app.post("/stream_chat")
def stream_chat(req: ChatRequest):
    def sse_generator():
        stream = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.message},
            ],
            stream=True
        )
        for chunk in stream:
            text = chunk["message"]["content"]
            # 拆分单个字符逐个输出，增加延时，流式感极强
            for char in text:
                yield f"data: {char}\n\n"
                time.sleep(0.03)  # 控制打字速度，数值越大越慢
        # 流式结束标记
        yield "data: [DONE]\n\n"

    # SSE专用媒体类型
    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 服务启动中... 模型: {MODEL}")
    print(f"📖 健康检查: http://localhost:8000")
    print(f"💬 一次性接口 POST /chat")
    print(f"⚡ 流式输出接口 POST /stream_chat")
    print(f"📝 流式curl测试命令：curl -X POST http://localhost:8000/stream_chat -H 'Content-Type:application/json' -d '{{\"message\": \"40+2等于多少\"}}'")
    uvicorn.run(app, host="0.0.0.0", port=8000)