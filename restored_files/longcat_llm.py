from openai import OpenAI
from dotenv import load_dotenv
import os
# 加载.env文件内的环境变量
load_dotenv()


client = OpenAI(
    api_key=os.environ.get('LONGCAT_API_KEY'),
    base_url="https://api.longcat.chat/openai"
    )


stream = client.chat.completions.create(
    model="LongCat-2.0",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and len(chunk.choices) > 0:
        content = chunk.choices[0].delta.content or ""
        print(content, end="")
print("\n\n生成完成")