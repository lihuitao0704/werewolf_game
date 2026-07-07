"""
NL2SQL Pro: 面向大规模表场景的 FastAPI 服务
==============================================
核心改进：两阶段检索架构（500+ 表适用）

架构：
  ① 离线：启动时一次性扫描所有表，构建 TABLE_INDEX 元数据缓存
  ② 在线第一阶段：关键词 + Embedding 检索，500 张表 → Top 5 相关表
  ③ 在线第二阶段：只把相关表的精简 schema 注入 LLM，生成 SQL

额外能力：
  - 缓存检索结果（同问题不重复检索）
  - 表间关系自动发现（外键 / 同名推断）
  - SQL 安全校验（禁止 DROP/TRUNCATE，UPDATE/DELETE 必须带 WHERE）
  - 结果行数限制，防止 SELECT * 打爆内存

启动：uvicorn nl2sql_pro:app --reload --host 0.0.0.0 --port 8000
"""
import os
import re
import json
import time
import hashlib
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


# ============================================================
#  TABLE INDEX — 离线构建的元数据缓存
# ============================================================

# 这是核心数据结构：表名 → 可检索文本 + 列信息 + 关系
TABLE_INDEX: dict[str, dict] = {}

# 可选：用户手动补充的表注释（用于纠正英文名称表难以理解的情况）
# 例如：{"t_usr": "用户表，存储注册用户的基础信息"}
MANUAL_TABLE_HINTS: dict[str, str] = {
    # 👇 在这里补充中文释义，提升关键词命中率
    # "ods_log_api": "API请求日志，记录所有接口的调用情况",
}


def build_table_index() -> dict[str, dict]:
    """
    一次性扫描数据库所有表，构建元数据索引。
    返回格式:
    {
        "table_name": {
            "searchable_text": "表名 注释 字段1 字段2 ...",  # 用于检索
            "comment": "表注释",
            "columns": [(field, type, null, key, default, extra), ...],
            "related_tables": ["other_table", ...],  # 推断的关联表
        }
    }
    """
    index: dict[str, dict] = {}
    conn = pymysql.connect(**DB_CONFIG)
    # 修复 MariaDB/MySQL 表注释编码问题
    conn.query("SET NAMES utf8mb4")
    conn.query("SET character_set_connection=utf8mb4, character_set_results=utf8mb4, character_set_client=utf8mb4")
    try:
        with conn.cursor() as cursor:
            # 获取所有表名 + 表注释
            cursor.execute(f"""
                SELECT TABLE_NAME, TABLE_COMMENT
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = '{DB_CONFIG['database']}'
            """)
            tables = cursor.fetchall()

            for table_name, table_comment in tables:
                # 获取字段信息
                cursor.execute(f"DESC `{table_name}`")
                columns = cursor.fetchall()

                # 构建可检索文本：表名 + 注释 + 所有字段名
                col_names = [col[0] for col in columns]
                searchable_parts = [table_name]
                if table_comment:
                    searchable_parts.append(table_comment)
                if table_name in MANUAL_TABLE_HINTS:
                    searchable_parts.append(MANUAL_TABLE_HINTS[table_name])
                searchable_parts.extend(col_names)

                index[table_name] = {
                    "searchable_text": " ".join(searchable_parts),
                    "comment": table_comment,
                    "columns": columns,
                    "col_names": col_names,
                    "related_tables": [],
                }

            # 自动发现表间关系（同名推断法）
            _infer_relationships(index)
    finally:
        conn.close()

    print(f"[NL2SQL Pro] 索引构建完成，共 {len(index)} 张表")
    return index


def _infer_relationships(index: dict[str, dict]) -> None:
    """
    自动推断表间关联关系：
    规则1: 字段名 = 其他表名 + _id → 外键关联（如 student_id → students.id）
    规则2: 字段名直接匹配其他表的主键
    """
    for table_name, meta in index.items():
        for col_name in meta["col_names"]:
            # 规则：xxx_id 格式，检查是否有对应表
            if col_name.endswith("_id") and col_name not in ("id",):
                ref_table = col_name[:-3]  # "student_id" → "student"
                ref_table_plural = ref_table + "s"  # 尝试复数形式
                if ref_table in index:
                    meta["related_tables"].append(ref_table)
                elif ref_table_plural in index:
                    meta["related_tables"].append(ref_table_plural)

            # 规则：字段名直接等于某表名
            if col_name in index and col_name != table_name:
                meta["related_tables"].append(col_name)

        # 去重
        meta["related_tables"] = list(set(meta["related_tables"]))


# ============================================================
#  两阶段检索
# ============================================================

# 👇 中文业务关键词 → 英文表名/字段名的映射
# 当用户说中文时，自动翻译为可用于匹配表名的关键词
KEYWORD_SYNONYMS: dict[str, list[str]] = {
    # 例子：如果你的表用英文字段名，用户用中文问
    "学生": ["students", "student"],
    "老师": ["teachers", "teacher", "teacher_id"],
    "教师": ["teachers", "teacher", "teacher_id"],
    "班级": ["classes", "class", "class_id"],
    "成绩": ["scores", "score"],
    "分数": ["scores", "score"],
    "就业": ["employments", "employment", "job", "employed"],
    "工作": ["employments", "employment", "job"],
    "公司": ["employments", "company"],
    "教室": ["room"],
    "年级": ["grade"],
    "学科": ["subject"],
    "性别": ["gender"],
    "年龄": ["age"],
    "电话": ["phone"],
    "薪水": ["salary", "工资", "薪资", "月薪", "年薪"],
    "薪资": ["salary"],
    "地址": ["address"],
}

def retrieve_relevant_tables(question: str, top_k: int = 5) -> list[str]:
    """
    第一阶段：从 TABLE_INDEX 中检索与问题最相关的 N 张表。

    策略（混合检索）：
    1. 字段名/表名直接匹配（英文，高精度低召回）
    2. 关键词同义词扩展后再匹配（中文 → 英文）
    3. 字符级交集（兜底）
    4. 命中表的关联表也纳入（传播）
    """
    q_lower = question.lower()
    q_chars = set(q_lower)

    # 构建扩展查询：原始问题 + 同义词翻译出的英文关键词
    expanded_terms = set()
    for cn_keyword, en_terms in KEYWORD_SYNONYMS.items():
        if cn_keyword in question:
            expanded_terms.update(en_terms)

    # 合并：原始问题 + 扩展英文关键词
    expanded_question = q_lower + " " + " ".join(expanded_terms)

    scores: dict[str, float] = {}

    for table_name, meta in TABLE_INDEX.items():
        text_lower = meta["searchable_text"].lower()

        score = 0.0

        # 规则 1: 表名直接出现在原始问题中或同义词中（最高精度）
        if table_name.lower() in q_lower:
            score += 20
        for term in expanded_terms:
            if table_name.lower() == term.lower():
                score += 15  # 同义词精确命中表名

        # 规则 2: 字段名匹配（精确命中加分，子串命中减分）
        for col_name in meta["col_names"]:
            col_lower = col_name.lower()
            if col_lower in q_lower:
                score += 8
            else:
                # 同义词扩展命中字段名
                for term in expanded_terms:
                    if term == col_lower:
                        score += 6  # 精确字段匹配
                    elif len(term) > 2 and term in col_lower:
                        score += 2  # 子串匹配，权重较低

        # 规则 3: 字符级交集（仅兜底，权重极低）
        char_overlap = len(q_chars & set(c for c in text_lower if c.isascii()))
        score += char_overlap * 0.05

        if score > 0:
            scores[table_name] = score

    # --- 关联表传播：高分的表，其关联表也得基础分 ---
    top_direct = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    expanded_scores = dict(scores)

    for t_name, t_score in top_direct:
        if t_name in TABLE_INDEX:
            for related in TABLE_INDEX[t_name]["related_tables"]:
                if related not in expanded_scores or expanded_scores[related] < t_score * 0.3:
                    expanded_scores[related] = max(expanded_scores.get(related, 0), t_score * 0.3)

    # --- 取 Top-K，至少返回 1 张表 ---
    ranked = sorted(expanded_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in ranked[:top_k]]

    # 兜底：如果什么都没命中，返回所有表（小数据库场景）
    if not selected and TABLE_INDEX:
        selected = list(TABLE_INDEX.keys())[:top_k]

    print(f"[NL2SQL Pro] 检索相关表: {selected}, scores: {dict(sorted(expanded_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])}")
    return selected


# ============================================================
#  SQL 生成（基于检索后的精简 schema）
# ============================================================

def build_focused_schema(table_names: list[str]) -> str:
    """根据给定的表名列表，拼装精简 schema 文本"""
    parts = []
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            for table_name in table_names:
                if table_name not in TABLE_INDEX:
                    continue
                meta = TABLE_INDEX[table_name]
                parts.append(f"\n表 `{table_name}`（{meta['comment'] or '无注释'}）：")

                for col in meta["columns"]:
                    flags = []
                    if col[3] == 'PRI':
                        flags.append('主键')
                    if col[5] == 'auto_increment':
                        flags.append('自增')
                    related = ""
                    if col[0].endswith("_id") and col[0][:-3] + "s" in TABLE_INDEX:
                        related = f" ← 关联 {col[0][:-3] + 's'}"
                    elif col[0].endswith("_id") and col[0][:-3] in TABLE_INDEX:
                        related = f" ← 关联 {col[0][:-3]}"
                    parts.append(
                        f"  - {col[0]}: {col[1]}"
                        f"{' [' + ', '.join(flags) + ']' if flags else ''}"
                        f"{related}"
                    )

                # 关联表提示
                if meta["related_tables"]:
                    parts.append(f"  关联表: {', '.join(meta['related_tables'])}")
    finally:
        conn.close()

    return "\n".join(parts)


def generate_sql(question: str) -> tuple[list[str], list[str]]:
    """
    两阶段 SQL 生成。
    返回 (sql_list, selected_tables)
    """
    # 阶段 1：检索相关表
    selected_tables = retrieve_relevant_tables(question, top_k=5)

    # 阶段 2：只取相关表的 schema
    focused_schema = build_focused_schema(selected_tables)

    print(f"[NL2SQL Pro] Schema 大小: {len(focused_schema)} chars")

    prompt = f"""你是 MySQL 专家。以下是与用户问题相关的表结构（已从 {len(TABLE_INDEX)} 张表中智能筛选）：

{focused_schema}

用户问题：{question}

要求：
1. 只返回 JSON 格式的 SQL 字符串数组，不要其他解释
2. 如果涉及多步操作，在数组中按顺序放多条 SQL
3. 使用 JOIN 关联表时注意字段名
4. INSERT 时忽略自增主键字段
5. UPDATE/DELETE 必须带 WHERE 条件，禁止全表操作
6. SELECT 查询加 LIMIT 100 防止返回过多数据
7. 仅使用上面列出的表和字段，不要猜测不存在的字段
8. enum 类型的值必须从 schema 中列出的选项选择

直接返回 JSON 数组，例如：
["SELECT name, age FROM students WHERE class_id = 1 LIMIT 100"]
"""

    raw = call_llm(prompt).strip()
    raw = raw.replace('```json', '').replace('```', '').strip()

    sql_list = json.loads(raw)
    if isinstance(sql_list, str):
        sql_list = [sql_list]

    return sql_list, selected_tables


# ============================================================
#  SQL 安全校验
# ============================================================

FORBIDDEN_KEYWORDS = ('drop', 'truncate', 'alter', 'create', 'rename', 'grant', 'revoke')


def validate_sql(sql_list: list[str]) -> Optional[str]:
    """
    安全校验，返回错误信息或 None（通过）。
    """
    for sql in sql_list:
        upper = sql.strip().upper()

        # 1. 禁止危险操作（startswith 和 in 都要用大写比较）
        for kw in FORBIDDEN_KEYWORDS:
            kw_upper = kw.upper()
            if upper.startswith(kw_upper) or f" {kw_upper} " in upper:
                return f"安全拦截：禁止执行 {kw_upper} 操作"

        # 2. UPDATE / DELETE 必须带 WHERE
        if upper.startswith(('UPDATE', 'DELETE')):
            if 'WHERE' not in upper:
                return "安全拦截：UPDATE/DELETE 必须指定 WHERE 条件"

        # 3. 禁止注释注入
        if '/*' in sql or '--' in sql or ';' in sql[:-1]:
            return "安全拦截：SQL 中包含非法字符"

    return None


# ============================================================
#  SQL 执行
# ============================================================

def execute_sql(sql_list: list[str]):
    """
    遍历执行 SQL 数组。
    - SELECT → 返回 (rows, columns)
    - INSERT/UPDATE/DELETE → 返回 (affected_rows, [])
    - 任意失败 → 回滚 + 抛异常
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


# ============================================================
#  结果润色
# ============================================================

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


# ============================================================
#  核心编排
# ============================================================

# 简单缓存：同样问题 60 秒内不重复检索
_CACHE: dict[tuple[str, float], dict] = {}


def nl2sql(question: str) -> dict:
    """完整 NL2SQL 流程"""
    # 缓存检查
    now = time.time()
    cache_key = question.strip().lower()
    if cache_key in _CACHE:
        cached_result, cached_time = _CACHE[cache_key]
        if now - cached_time < 60:
            print(f"[NL2SQL Pro] 命中缓存")
            return cached_result

    # 生成 SQL（两阶段检索 + 精简 schema）
    sql_list, selected_tables = generate_sql(question)

    # 安全校验
    error = validate_sql(sql_list)
    if error:
        return {
            "success": False,
            "question": question,
            "sql": sql_list,
            "type": "BLOCKED",
            "error": error,
        }

    # 执行
    results, columns = execute_sql(sql_list)

    last_sql = sql_list[-1]
    sql_type = last_sql.strip().upper().split()[0]

    if sql_type == 'SELECT':
        answer = polish_result(question, last_sql, results, columns)
        result = {
            "success": True,
            "question": question,
            "sql": sql_list,
            "type": sql_type,
            "columns": columns,
            "data": [list(row) for row in results] if results else [],
            "selected_tables": selected_tables,
            "answer": answer,
        }
    else:
        op_names = {'INSERT': '插入', 'UPDATE': '更新', 'DELETE': '删除'}
        op_name = op_names.get(sql_type, '执行')
        result = {
            "success": True,
            "question": question,
            "sql": sql_list,
            "type": sql_type,
            "affected_rows": results,
            "selected_tables": selected_tables,
            "answer": f"数据已{op_name}完成。",
        }

    # 写入缓存
    _CACHE[cache_key] = (result, now)
    return result


# ============================================================
#  FastAPI 接口
# ============================================================

app = FastAPI(title="NL2SQL Pro API", version="1.0")


class QueryRequest(BaseModel):
    question: str


class TableInfo(BaseModel):
    name: str
    comment: str
    columns: int
    related: list[str]


class QueryResponse(BaseModel):
    success: bool
    question: str
    sql: list[str] = []
    type: str = ""
    columns: Optional[list[str]] = None
    data: Optional[list] = None
    affected_rows: Optional[int] = None
    selected_tables: Optional[list[str]] = None
    answer: Optional[str] = None
    error: Optional[str] = None


@app.on_event("startup")
def startup():
    """启动时构建表索引"""
    global TABLE_INDEX
    TABLE_INDEX = build_table_index()


@app.get("/")
def root():
    return {
        "message": "NL2SQL Pro 服务运行中",
        "total_tables": len(TABLE_INDEX),
        "docs": "/docs",
        "endpoints": ["/nl2sql", "/tables"],
    }


@app.post("/nl2sql", response_model=QueryResponse)
def query(request: QueryRequest):
    """NL2SQL 接口"""
    try:
        result = nl2sql(request.question)
        return QueryResponse(**result)
    except json.JSONDecodeError as e:
        return QueryResponse(
            success=False, question=request.question, error=f"SQL 解析失败：{e}"
        )
    except pymysql.err.OperationalError as e:
        return QueryResponse(
            success=False, question=request.question, error=f"数据库错误：{e}"
        )
    except Exception as e:
        return QueryResponse(
            success=False, question=request.question, error=f"执行失败：{e}"
        )


@app.get("/tables")
def list_tables():
    """查看所有已索引的表及其关系"""
    return {
        "total": len(TABLE_INDEX),
        "tables": [
            TableInfo(
                name=name,
                comment=meta["comment"],
                columns=len(meta["columns"]),
                related=meta["related_tables"],
            )
            for name, meta in TABLE_INDEX.items()
        ],
    }


@app.get("/tables/{table_name}")
def table_detail(table_name: str):
    """查看某张表的完整 schema"""
    if table_name not in TABLE_INDEX:
        return {"error": f"表 {table_name} 不存在"}

    meta = TABLE_INDEX[table_name]
    return {
        "name": table_name,
        "comment": meta["comment"],
        "columns": [
            {
                "field": col[0],
                "type": col[1],
                "null": "YES" if col[2] == "YES" else "NO",
                "key": col[3],
                "default": col[4],
                "extra": col[5],
            }
            for col in meta["columns"]
        ],
        "related_tables": meta["related_tables"],
    }


@app.post("/reindex")
def reindex():
    """手动触发重新构建索引（新增表后调用）"""
    global TABLE_INDEX
    TABLE_INDEX = build_table_index()
    return {"message": "索引重建完成", "total_tables": len(TABLE_INDEX)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
