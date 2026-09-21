"""
setup_datasources.py

Downloads and organizes the three data sources for the agentic RAG project:

1.Documents - docs/ (FastAPI markdown docs, unstructured text)
2.Code - code/ (a sample open-source python repo)
3.Structured - sql/ (Chinook sample SQLite database)

"""
import os
import subprocess
import urllib.request
import sqlite3


BASE_DIR = os.path.join(os.getcwd(),"rag_data")
DOCS_DIR = os.path.join(BASE_DIR,"docs")
CODE_DIR = os.path.join(BASE_DIR,"code")
SQL_DIR = os.path.join(BASE_DIR,"sql")


# 1. Create Folder Structure

def part1_create_folder():
    print("Part 1 Creating folder Structure....")
    for d in [BASE_DIR,DOCS_DIR,CODE_DIR,SQL_DIR]:
        os.makedirs(d,exist_ok=True)
        print(f"Created {d}")


# 2. Documents source: clone FastAPI's markdown docs

def part2_fetch_docs():
    print("Fetching documents source (FastAPI docs)....")
    repo_url="https://github.com/tiangolo/fastapi.git"
    clone_path=os.path.join(DOCS_DIR,"fastapi_docs_repo")

    if os.path.exists(clone_path):
        print("Already present, skipping clone.")
    else:
        subprocess.run(
            ["git","clone","--depth","1",repo_url,clone_path],
            check=True,
        )

    # Count how many markdown files we actually have to index

    md_count=0

    for root, _, files in os.walk(os.path.join(clone_path,"docs")):
        md_count+=sum(1 for f in files if f.endswith(".md"))
    print(f"found {md_count} markdown files under docs/docs")


# 3. Code source: clone a small,well-known open-source repo

def part3_fetch_code():
    print("Fetching source code (request library)...")
    repo_url="https://github.com/psf/requests.git"
    clone_path=os.path.join(CODE_DIR,"requests_repo")

    if os.path.exists(clone_path):
        print("Already present,skipping clone")
    else:
        subprocess.run(
            ["git","clone","--depth","1",repo_url,clone_path],
            check=True,)

    py_count=0
    for root, _, files in os.walk(clone_path):
        py_count +=sum(1 for f in files if f.endswith(".py"))
    print(f"Found {py_count} python files to chunk at function/class level")

# 4.Structured source: download the chinook sample SQLite database (this gives us a realistic multi-table schema (artists,albums,tracks,customers,invoices)) to exercise the text-to-sql branch.

def part4_fetch_sql():
    print("Fetching structure data source (Chinook SQLite db).....")

    url=("https://raw.githubusercontent.com/lerocha/chinook-database/"
         "master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite")
    db_path=os.path.join(SQL_DIR,"chinook.db")

    if os.path.exists(db_path):
        print("Already present, skipping download")
    else:
        urllib.request.urlretrieve(url,db_path)


    con=sqlite3.connect(db_path)
    cur=con.cursor()
    cur.execute("Select name FROM sqlite_master WHERE type='table'")
    tables=[row[0] for row in cur.fetchall()]
    cur.close()
    print(f"Tables Available: {tables}")

# 5. print a schema summary 

def part5_print_schema_for_prompt():
    print("Schema summary for your text-to-sql prompt:")
    db_path= os.path.join(SQL_DIR,"chinook.db")
    con=sqlite3.connect(db_path)
    cur=con.cursor()
    cur.execute("Select name FROM sqlite_master WHERE type='table'")
    tables=[row[0] for row in cur.fetchall()]

    for table in tables:
        cur.execute(f"PRAGMA table_info ({table})")
        cols=[f"{row[1]} ({row[2]})" for row in cur.fetchall()]
        print(f"{table}: {', '.join(cols)}")
    con.close()

if __name__=="__main__":
    part1_create_folder()
    part2_fetch_docs()
    part3_fetch_code()
    part4_fetch_sql()
    part5_print_schema_for_prompt()

    print(f"\n All data sources ready under : {BASE_DIR}")
    