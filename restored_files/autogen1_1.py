import asyncio
import io
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ── 配置 LLM（用通义千问） ─────────────────────────────────
model_client = OpenAIChatCompletionClient(
    model="qwen3.7-plus",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://ws-sz2lpqpiqu2qcjv5.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    },
)

# ── 创建逗哏 Agent ─────────────────────────────────────────
dougen = AssistantAgent(
    name="HAOTIAN",
    model_client=model_client,
    system_message="""你是一个暴躁的贴吧老哥，你叫浩天。
你的任务：
- 疯狂的贬低对方
- 每次发言不超过 50 字
- 喜欢抬杠，喜欢嘲讽

""",
)

# ── 创建捧哏 Agent ─────────────────────────────────────────
penggen = AssistantAgent(
    name="SIJIN",
    model_client=model_client,
    system_message="""你是一个知乎的海龟精英， 你叫思锦。
你的任务：
- 喜欢阴阳
- 每次发言不超过 50 字
- 喜欢装逼
""",
)


async def main():
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination

    # 创建轮转群聊（逗哏和捧哏交替发言）
    team = RoundRobinGroupChat(
        [dougen, penggen],
        termination_condition=MaxMessageTermination(max_messages=20),
    )

    try:
        print("🎭 相声开始！\n")
        await Console(team.run_stream(task="成都只有一个名额，为了争抢这个名额吵架"))
    finally:
        await model_client.close() 


asyncio.run(main())
