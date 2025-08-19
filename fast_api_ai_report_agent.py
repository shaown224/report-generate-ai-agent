"""
FastAPI AI Report Agent with Grok AI

Files included: This single Python file contains a ready-to-run FastAPI application that:
- Accepts a natural-language prompt asking for a report
- Uses API to convert prompt -> SQL (guided by your DB schema)
- Safely validates SQL (read-only checks)
- Executes SQL against your database (SQLAlchemy)
- Returns results as JSON + Markdown table and provides an Excel file download

Setup (quick):
1. Create a virtualenv and install requirements:
   pip install -r requirements.txt

2. Create a .env file next to this script with:
   DATABASE_URL=postgresql://readonly_user:password@dbhost:5432/dbname
   OP_API_KEY=xai-your-api-key-here
   ALLOWED_TABLES=public.orders,public.customers  # optional, comma-separated

3. Run the app:
   uvicorn fast_api_ai_report_agent:app --reload --port 8000

Endpoints:
- POST /report  { "prompt": "Show total sales by region for July" }
  returns: { markdown: ..., rows: ..., excel_url: ... }

Security notes:
- Use a read-only DB user.
- Review generated SQL before allowing production writes.
- The simple sanitizer below blocks destructive keywords but is not foolproof.

Limitations & next steps:
- For large schemas, feed a focused subset of tables to the LLM.
- Add function-calling / LangChain toolkit for better SQL generation.
- Add caching, authentication, rate-limiting, and query cost controls.

"""

import os
import re
import tempfile
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Optional, Any

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OP_API_KEY = os.getenv("OP_API_KEY")
ALLOWED_TABLES = os.getenv("ALLOWED_TABLES")  # optional comma-separated list

if not DATABASE_URL:
    raise RuntimeError("Please set DATABASE_URL in your environment (.env)")
if not OP_API_KEY:
    raise RuntimeError("Please set OP_API_KEY in your environment (.env)")

app = FastAPI(title="AI Report Agent with")

engine = create_engine(DATABASE_URL, future=True)

# --- Models ---
class ReportRequest(BaseModel):
    prompt: str
    max_rows: int = 1000

class ReportResponse(BaseModel):
    data: List[Dict[str, Any]]  # JSON data from the DataFrame
    rows: int                   # Number of rows returned
    sql: Optional[str] = None   # Optional: the generated SQL query

# --- Utility functions ---


def get_schema_overview(limit_tables: int = 10) -> str:
    """
    Return a small textual schema overview (tables and columns).
    Works with both MySQL and PostgreSQL.
    """
    query = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys', 'pg_catalog')
        ORDER BY table_schema, table_name;
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)
    print(df, '####################')
    # Normalize column names to lowercase (avoids MySQL uppercase issue)
    df.columns = [col.lower() for col in df.columns]

    # Optionally filter allowed tables
    if ALLOWED_TABLES:
        allowed = {t.strip() for t in ALLOWED_TABLES.split(",") if t.strip()}
        # Fix: Filter by table_name instead of full_name (schema.table)
        df = df[df["table_name"].isin(allowed)]

    # Group and format
    out_lines = []
    grouped = df.groupby(["table_schema", "table_name"], sort=False)
    for count, ((schema, table), g) in enumerate(grouped, start=1):
        cols = ", ".join(g["column_name"].tolist())
        out_lines.append(f"{schema}.{table}({cols})")
        if count >= limit_tables:
            break

    return "\n".join(out_lines)


def generate_sql_from_prompt(prompt: str, schema_overview: str) -> str:
    """Ask AI to generate a safe SELECT SQL statement only.
    Uses the xAI API to communicate with models.
    """
    system = (
        "You are a SQL generator. Given a user's request and a database schema, "
        "produce a single READ-ONLY SQL SELECT statement that answers the request. "
        "Do not include any explanation or text, output only the SQL. "
        "If the request cannot be answered with SQL, output the single word: CANNOT_ANSWER."
    )

    user_msg = (
        "Schema (tables and columns):\n"
        f"{schema_overview}\n\n"
        "User request:\n"
        f"{prompt}\n\n"
        "Important constraints:\n"
        "- Only provide a SELECT statement (no semicolons at the end).\n"
        "- Limit results using ORDER BY / LIMIT if appropriate.\n"
        "- Use table/column names exactly as in the schema overview.\n"
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OP_API_KEY,
    )
    print("----------------------")
    print(user_msg)
    print("----------------------")

    completion = client.chat.completions.create(
        # extra_headers={
        #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        # },
        extra_body={},
        model="deepseek/deepseek-r1-0528:free",
        messages=[
            {
            "role": "user",
            "content": user_msg
            }
        ]
    )
    print("Generated SQL:", completion.choices[0].message.content)
    return completion.choices[0].message.content


DANGEROUS_PATTERNS = re.compile(
    r"\b(drop|delete|update|insert|truncate|alter|create|replace|merge)\b",
    flags=re.IGNORECASE,
)


def sanitize_sql(sql: str) -> str:
    """Basic sanitizer: ensure the SQL is a SELECT and doesn't contain dangerous keywords."""
    if sql.strip().upper() == "CANNOT_ANSWER":
        raise ValueError("AI could not produce a SQL statement for this request.")

    if DANGEROUS_PATTERNS.search(sql):
        raise ValueError("SQL contains potentially dangerous keywords.")

    # Enforce it starts with SELECT
    # if not sql.lstrip().lower().startswith("select") or not sql.lstrip().lower().startswith("elect"):
    #     raise ValueError("Only SELECT statements are allowed.")

    # Remove trailing semicolon if present
    sql = sql.strip().rstrip(";")
    return sql


def execute_sql(sql: str, max_rows: int = 1000) -> pd.DataFrame:
    """Execute the SQL and return a DataFrame. Impose a MAX ROWS cap."""
    # Wrap query to enforce limit if user didn't
    limit_clause = f" LIMIT {max_rows}"
    modified_sql = sql
    # crude: append LIMIT if not present
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE) is None:
        modified_sql = f"{sql} {limit_clause}"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(modified_sql), conn)

    print(df, '####################')
    return df


def df_to_excel_file(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    df.to_excel(path, index=False)
    return path


# --- FastAPI endpoints ---
@app.post("/report", response_model=ReportResponse)
def create_report(req: ReportRequest):
    """Main entry: accept a natural language prompt and return a report."""
    prompt = req.prompt
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # 1) Get schema overview
    schema_overview = get_schema_overview(limit_tables=15)
    print("Schema overview:", schema_overview)

    # 2) Generate SQL via AI
    try:
        sql = generate_sql_from_prompt(prompt, schema_overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI error: {e}")

    # 3) Sanitize
    try:
        sql = sanitize_sql(sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4) Execute
    try:
        df = execute_sql(sql, max_rows=req.max_rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # 5) Convert DataFrame to JSON data
    data = df.to_dict('records')  # Convert to list of dictionaries
    
    return ReportResponse(
        data=data,
        rows=len(df),
        sql=sql  # Optional: include the generated SQL for debugging
    )


@app.get("/download")
def download_file(path: str):
    # Security: ensure path is inside tmp dir
    tmpdir = tempfile.gettempdir()
    if not os.path.commonpath([os.path.abspath(path), tmpdir]) == os.path.abspath(tmpdir):
        raise HTTPException(status_code=403, detail="Forbidden file path")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, filename=os.path.basename(path), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# --- Requirements (put these in requirements.txt) ---
# fastapi
# uvicorn[standard]
# sqlalchemy
# psycopg2-binary  # or mysqlclient depending on your DB
# python-dotenv
# pandas
# requests
# openpyxl

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("FastAPI_AI_Report_Agent:app", host="0.0.0.0", port=8000, reload=True)