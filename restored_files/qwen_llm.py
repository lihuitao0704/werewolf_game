import os
from openai import OpenAI
from dotenv import load_dotenv
# 加载.env文件内的环境变量
load_dotenv()


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://ws-sz2lpqpiqu2qcjv5.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen3.7-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ]
    
)
print(completion.model_dump_json())
print('------------')
answer = completion.choices[0].message.content
print(answer)