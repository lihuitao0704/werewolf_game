import os
import pprint

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# 定义工具 (使用 @tool 装饰器)
@tool
def search_weather(city: str) -> str:
    """查询天气"""
    return f"{city}明天晴，15-25°C"


@tool
def book_flight(origin: str, dest: str, date: str) -> str:
    """订机票"""
    return f"已订票：{origin}→{dest}，{date}"


@tool
def search_grade(name: str) -> str:
    """查询某个学生的成绩"""
    mock = {
        '思锦': 0,
        '浩天': 99,
        '李磊': 89
    }
    return f"{name}的成绩是{mock.get(name, 60)} 分"


tools = [search_weather, book_flight, search_grade]

# 初始化模型
# 将环境变量作为参数传入 (如果模型提供商兼容 OpenAI API)
model = ChatOpenAI(
    model="LongCat-2.0",  # 根据您的服务端修改模型名称
    base_url="https://api.longcat.chat/openai",
    api_key=os.getenv("LONGCAT_API_KEY")
)

# 初始化 Agent
agent = create_agent(model=model, tools=tools)
agent2 = create_agent(model=model, tools=[book_flight, search_grade])

# 运行 Agent
# response = agent.invoke({"messages":
#                              [{"role": "user",
#                                "content": "辉哥是沃林的学习成绩提升导师，思锦和 浩天谁需要辅导 ，"
#                                           "辉哥在深圳阴天时候更喜欢辅导思锦，"
#                                           "辉哥在深圳晴天时候更喜欢辅导浩天，"
#                                           "辉哥只辅导成绩低的，不辅导成绩高的，"
#                                           "谁会被辅导"}]})

response = agent.invoke({"messages":
                             [{"role": "user",
                               "content": "北京天气如何？" }]})



# 将 response 转换为字典后，格式化打印
pprint.pprint(response, width=120, sort_dicts=False)