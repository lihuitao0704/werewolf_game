# NL2SQL5 FastAPI Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-ready NL2SQL FastAPI service that converts natural language to SQL, executes it against MySQL, and returns polished natural language answers.

**Architecture:** Single-file FastAPI app with three core functions (generate_sql, execute_sql, polish_result) plus a schema auto-loader. The user provides table structure via a `TABLE_INFO` variable, which is enriched by auto-fetching `SHOW TABLES` + `DESC <table>` from the live database. LLM calls use LongCat-2.0 via OpenAI-compatible API.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pymysql, openai (LongCat-2.0), python-dotenv

## Global Constraints

- File location: `restored_files/nl2sql5.py`
- LLM: LongCat-2.0 via `https://api.longcat.chat/openai`
- API_KEY from `.env` variable `LONGCAT_API_KEY`
- Database driver: `pymysql`
- Must support SELECT / INSERT / UPDATE / DELETE
- Must return JSON with: success, question, sql, type, columns/data/affected_rows, answer, error
- UseTABLE_INFO as the primary schema source; auto-fetch from DB as fallback/supplement

---

## Task 1: Project scaffolding and configuration

**Files:**
- Create: `restored_files/nl2sql5.py`

**Interfaces:**
- Produces: `client` (OpenAI client), `MODEL` (str), `DB_CONFIG` (dict), `TABLE_INFO` (str)
- Produces: `call_llm(prompt: str) -> str`

- [ ] **Step 1: Create the file with imports, env loading, LLM config, DB config, and empty TABLE_INFO**

```python
"""
NL2SQL V5: FastAPI service with auto schema loading
功能：自然语言 → SQL → 执行 → 润色回答
表结构来源：用户手动填写 TABLE_INFO + 自动从数据库读取补充
启动：uvicorn nl2sql5:app --reload --host 0.0.0.0 --port 8000
测试：curl -X POST http://localhost:8000/nl2sql -H "Content-Type: application/json" -d '{"question":"..."}'
"""
import os
import json
import pymysql
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
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
```

- [ ] **Step 2: Verify the file imports cleanly**

Run: `cd restored_files && python -c "import nl2sql5; print('OK')"`
Expected: `OK` (may show pydantic warning which is fine)

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: scaffold nl2sql5 FastAPI service with config and TABLE_INFO"
```

---

## Task 2: Schema auto-loader (TABLE_INFO + live DB)

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `DB_CONFIG` (dict), `TABLE_INFO` (str)
- Produces: `load_schema() -> str` — merges user TABLE_INFO with auto-fetched DB schema

- [ ] **Step 1: Add `load_schema()` function that auto-fetches tables and columns from the live database**

```python
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
                        schema_parts.append(
                            f"  - {col[0]}: {col[1]}, "
                            f"{'主键' if col[3] == 'PRI' else ''}"
                            f"{'自增' if col[5] == 'auto_increment' else ''}"
                            f"{'允许NULL' if col[2] == 'YES' else 'NOT NULL'}"
                        )
        finally:
            conn.close()
    except Exception as e:
        schema_parts.append(f"\n(⚠️ 自动读取数据库失败: {e}，仅使用手动填写的 TABLE_INFO)")

    return "\n".join(schema_parts)
```

- [ ] **Step 2: Verify schema loading works (or gracefully degrades)**

Run: `cd restored_files && python -c "from nl2sql5 import load_schema; print(load_schema()[:500])"`
Expected: Printed schema text (either merged or TABLE_INFO only if DB unreachable)

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add load_schema to auto-fetch DB schema and merge with TABLE_INFO"
```

---

## Task 3: SQL generation function

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `call_llm(prompt: str) -> str`, `load_schema() -> str`
- Produces: `generate_sql(question: str) -> list[str]` — returns list of SQL statements

- [ ] **Step 1: Add `generate_sql()` function**

```python
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
```

- [ ] **Step 2: Test SQL generation**

Run: `cd restored_files && python -c "from nl2sql5 import generate_sql; print(generate_sql('查询所有学生的姓名'))"`
Expected: A list of SQL strings like `["SELECT name FROM students"]`

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add generate_sql function with merged schema injection"
```

---

## Task 4: SQL execution function

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `DB_CONFIG` (dict)
- Produces: `execute_sql(sql_list: list[str]) -> tuple[any, list[str]]` — returns (results, columns)

- [ ] **Step 1: Add `execute_sql()` function**

```python
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
```

- [ ] **Step 2: Test execution with a simple SELECT**

Run: `cd restored_files && python -c "from nl2sql5 import execute_sql; print(execute_sql(['SELECT 1 AS test']))"`
Expected: `((1,), ['test'])`

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add execute_sql function supporting SELECT/INSERT/UPDATE/DELETE"
```

---

## Task 5: Result polishing function

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `call_llm(prompt: str) -> str`
- Produces: `polish_result(question: str, sql: str, results: any, columns: list[str]) -> str`

- [ ] **Step 1: Add `polish_result()` function**

```python
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
```

- [ ] **Step 2: Test polishing with mock data**

Run: `cd restored_files && python -c "from nl2sql5 import polish_result; print(polish_result('张三数学多少分', 'SELECT score FROM ... WHERE name=\"张三\"', [(95,)], ['score']))"`
Expected: A natural language string about the score

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add polish_result function for natural language answers"
```

---

## Task 6: Core orchestration function

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `generate_sql`, `execute_sql`, `polish_result`
- Produces: `nl2sql(question: str) -> dict` — structured result with success/sql/type/columns/data/affected_rows/answer/error

- [ ] **Step 1: Add the `nl2sql()` orchestration function**

```python
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
```

- [ ] **Step 2: Verify the orchestration chain works end-to-end (may fail without DB, that's OK)**

Run: `cd restored_files && python -c "from nl2sql5 import nl2sql; print(nl2sql('查询所有学生'))"`
Expected: Either a correct result dict, or a dict with `success: false` and an error message

- [ ] **Step 3: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add nl2sql orchestration function"
```

---

## Task 7: FastAPI endpoints

**Files:**
- Modify: `restored_files/nl2sql5.py`

**Interfaces:**
- Consumes: `nl2sql(question: str) -> dict`
- Produces: FastAPI `app` with `GET /` and `POST /nl2sql` endpoints

- [ ] **Step 1: Add FastAPI app, request/response models, and endpoints at the bottom of the file**

```python
# ========== FastAPI 接口 ==========
app = FastAPI(title="NL2SQL V5 API", version="5.0")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Start the server and test with curl**

Terminal 1:
```bash
cd restored_files && uvicorn nl2sql5:app --reload --host 0.0.0.0 --port 8000
```
Expected: Server starts, shows "Application startup complete"

Terminal 2:
```bash
curl -X POST http://localhost:8000/nl2sql -H "Content-Type: application/json" -d '{"question":"查询所有学生的姓名"}'
```
Expected: JSON response with success/sql/answer fields

- [ ] **Step 3: Verify /docs is accessible**

Run: `curl -s http://localhost:8000/docs | head -5`
Expected: HTML containing "NL2SQL V5 API"

- [ ] **Step 4: Commit**

```bash
git add restored_files/nl2sql5.py
git commit -m "feat: add FastAPI endpoints GET / and POST /nl2sql"
```

---

## Task 8: Final verification

- [ ] **Step 1: Confirm the final file structure**

Run: `wc -l restored_files/nl2sql5.py`
Expected: Approximately 150-200 lines

- [ ] **Step 2: Run smoke test against live service with 3 question types**

```bash
# SELECT
curl -s -X POST http://localhost:8000/nl2sql -H "Content-Type: application/json" -d '{"question":"查询所有学生姓名"}'
```

Expected: 200 OK, valid JSON, `success: true`

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add restored_files/nl2sql5.py
git commit -m "fix: address smoke test findings" --allow-empty
```
