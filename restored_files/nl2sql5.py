"""
NL2SQL V5: FastAPI 服务 + 表结构自动加载
功能：自然语言 → SQL → 执行 → 润色回答
表结构来源：用户手动填写 TABLE_INFO + 自动从数据库读取补充

启动：uvicorn nl2sql5:app --reload --host 0.0.0.0 --port 8000
测试：curl -X POST http://localhost:8000/nl2sql -H "Content-Type: application/json" -d '{"question":"..."}'
"""
import os
import json
import datetime
import pymysql
from decimal import Decimal
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ========== LLM 配置 ==========
client = OpenAI(
    api_key=os.getenv("LONGCAT_API_KEY"),
    base_url="https://api.longcat.chat/openai",
)
MODEL = "LongCat-2.0"


def call_llm(prompt: str) -> str:
    """调用 LLM 并返回纯文本"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


# ========== 数据库配置 ==========
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'aistudy',
    'charset': 'utf8mb4'
}

# ========== 用户手动填写的表结构信息 ==========
# 👇 请在此处填写你的数据库表结构，格式参考下方示例：
TABLE_INFO = """
## 请替换为你的实际表结构描述

示例格式：
数据库中有两张表：

1. students 表（学生信息）：
   - id: INT, 主键, 学生ID
   - name: VARCHAR(50), 学生姓名
   - gender: ENUM('男','女'), 性别
   - age: INT, 年龄

2. scores 表（学生成绩）：
   - id: INT, 主键, 成绩ID
   - student_id: INT, 外键, 关联 students.id
   - subject: VARCHAR(50), 科目
   - score: DECIMAL(5,2), 分数

关系：一个学生有多条成绩记录（一对多）
查询时需要用 student_id 关联两张表。

业务规则：
- 分数范围 0-100
- 学科包括：数学、语文、英语、物理、化学
"""


# ========== 表结构自动加载 ==========
def load_schema() -> str:
    """
    合并用户手动填写的 TABLE_INFO 和从数据库自动读取的表结构。
    若数据库连接失败，仅返回 TABLE_INFO。
    """
    schema_parts = [TABLE_INFO.strip()]

    try:
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]

                schema_parts.append("\n## 自动从数据库读取的表结构：\n")
                for table in tables:
                    cursor.execute(f"DESC `{table}`")
                    columns = cursor.fetchall()
                    schema_parts.append(f"\n表 `{table}`：")
                    for col in columns:
                        # col: (Field, Type, Null, Key, Default, Extra)
                        flags = []
                        if col[3] == 'PRI':
                            flags.append('主键')
                        if col[5] == 'auto_increment':
                            flags.append('自增')
                        if col[2] == 'YES':
                            flags.append('允许NULL')
                        else:
                            flags.append('NOT NULL')
                        schema_parts.append(
                            f"  - {col[0]}: {col[1]}, {', '.join(flags)}"
                        )
        finally:
            conn.close()
    except Exception as e:
        schema_parts.append(f"\n(⚠️ 自动读取数据库失败: {e}，仅使用手动填写的 TABLE_INFO)")

    return "\n".join(schema_parts)


# ========== SQL 生成 ==========
def generate_sql(question: str) -> list[str]:
    schema = load_schema()

    prompt = f"""你是一个 MySQL 专家。根据用户的问题、数据库表结构和业务规则，生成对应的 SQL 语句。

数据库表结构：
{schema}

用户问题：{question}

要求：
1. 只返回 JSON 格式的 SQL 字符串数组，不要其他解释
2. 根据意图生成合适的 SQL（SELECT 查询、INSERT 插入、UPDATE 更新、DELETE 删除）
3. 如果涉及多步操作（如先查ID再插入），在数组中按顺序放多条 SQL
4. 使用正确的字段名和表名
5. 如果需要关联表，使用 JOIN
6. INSERT 时忽略自增主键字段
7. UPDATE/DELETE 必须带 WHERE 条件，禁止全表操作

直接返回 JSON 数组，例如：
["SELECT * FROM users WHERE name = '张三'"]
"""

    raw = call_llm(prompt).strip()
    # 清理可能的 markdown 代码块
    raw = raw.replace('```json', '').replace('```', '').strip()

    sql_list = json.loads(raw)
    if isinstance(sql_list, str):
        return [sql_list]
    return sql_list


# ========== SQL 执行 ==========
def execute_sql(sql_list: list[str]):
    """
    遍历执行 SQL 数组。
    - SELECT: 返回最后一条查询的结果(rows)和列名(columns)
    - INSERT/UPDATE/DELETE: 自动提交，返回受影响的行数
    - 任意语句失败时回滚并抛出异常
    """
    conn = pymysql.connect(**DB_CONFIG)
    try:
        last_results, last_columns = None, []

        for sql in sql_list:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                sql_type = sql.strip().upper().split()[0]

                if sql_type == 'SELECT':
                    last_results = cursor.fetchall()
                    last_columns = (
                        [desc[0] for desc in cursor.description]
                        if cursor.description else []
                    )
                else:
                    conn.commit()
                    last_results = cursor.rowcount
                    last_columns = []

        return last_results, last_columns
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ========== 结果润色 ==========
def polish_result(question: str, sql: str, results: any, columns: list[str]) -> str:
    """用 LLM 将查询数据润色为自然语言回答"""
    prompt = f"""你是一个友好的数据助手。根据用户的问题和查询结果，生成简洁自然的中文回答。

用户问题：{question}
执行的 SQL：{sql}
列名：{columns}
查询结果：{results}

要求：
1. 500字以内，简洁自然的中文
2. 直接给出答案，不要解释 SQL 或技术细节
3. 如果结果为空，礼貌地告知用户未找到匹配数据
4. 数字结果要带单位（如：分、人、元等，根据上下文推断）
"""

    return call_llm(prompt).strip()


# ========== 核心编排 ==========
def nl2sql(question: str) -> dict:
    """完整 NL2SQL 流程：自然语言 → SQL → 执行 → 润色"""
    sql_list = generate_sql(question)

    results, columns = execute_sql(sql_list)

    last_sql = sql_list[-1]
    sql_type = last_sql.strip().upper().split()[0]

    if sql_type == 'SELECT':
        answer = polish_result(question, last_sql, results, columns)
        return {
            "success": True,
            "question": question,
            "sql": sql_list,
            "type": sql_type,
            "columns": columns,
            "data": [list(row) for row in results] if results else [],
            "answer": answer,
        }
    else:
        op_names = {'INSERT': '插入', 'UPDATE': '更新', 'DELETE': '删除'}
        op_name = op_names.get(sql_type, '执行')
        return {
            "success": True,
            "question": question,
            "sql": sql_list,
            "type": sql_type,
            "affected_rows": results,
            "answer": f"数据已{op_name}完成。",
        }


# ========== FastAPI 接口 ==========
app = FastAPI(title="NL2SQL V5 API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    success: bool
    question: str
    sql: list[str] = []
    type: str = ""
    columns: Optional[list[str]] = None
    data: Optional[list] = None
    affected_rows: Optional[int] = None
    answer: Optional[str] = None
    error: Optional[str] = None


@app.get("/")
def root():
    return {"message": "NL2SQL V5 服务运行中", "docs": "/docs", "endpoint": "/nl2sql"}



@app.post("/nl2sql", response_model=QueryResponse)
def query(request: QueryRequest):
    """NL2SQL 接口：传入自然语言问题，返回 SQL 执行结果"""
    try:
        result = nl2sql(request.question)
        return QueryResponse(**result)
    except json.JSONDecodeError as e:
        return QueryResponse(
            success=False,
            question=request.question,
            error=f"SQL 生成解析失败：{e}",
        )
    except pymysql.err.OperationalError as e:
        return QueryResponse(
            success=False,
            question=request.question,
            error=f"数据库连接失败：{e}",
        )
    except Exception as e:
        return QueryResponse(
            success=False,
            question=request.question,
            error=f"执行失败：{e}",
        )


@app.get("/frontend")
def serve_frontend():
    return FileResponse(r"C:\Users\Windows\Desktop\nl2sql5.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
