"""This file Consumes rag_data/ (built by setup_datasources.py) and produces a local, persistent ChromaDB vector store with two collections:
    -"docs_and_code" - markdown docs + code chunks (used for hybrid retrieval)
    -the SQL side has no vector index; its schema is read directly at query
    time by the text-to-sql branch.

Everything here runs 100% locally and free:
    -embeddings come from Ollama's bge-m3 model (http://localhost:11434)
    -storage is ChromaDB in local persistent mode(no cloud account needed)

Prerequisites:
    1.Ollama installed and running: http://ollama.com
    2.Pull the embedding model once: ollama pull bge-m3
    3.pip install chromadb requests--break-system-packages

Run: python3 ingest_and_embed.py
"""

import os
import ast
import json
import sqlite3
import requests
import chromadb


BASE_DIR= os.path.join(os.getcwd(),"rag_data")
DOCS_DIR= os.path.join(BASE_DIR,"docs")
CODE_DIR= os.path.join(BASE_DIR,"code")
SQL_DB_PATH= os.path.join(BASE_DIR,"sql","chinook.db")

CHROMA_DIR = os.path.join(os.getcwd(),"chroma_store")
COLLECTION_NAME= "docs_and_code"

OLLAMA_URL ="http://localhost:11434/api/embed"
EMBED_MODEL="bge-m3"

CHUNK_SIZE=700 # characters, (roughly 150-180 tokesn)
CHUNK_OVERLAP=100
MAX_FILES_PER_SOURCE=15 #Covers the full FASTAPI docs(-155 files) and requests code(-20 files)

# 1. Markdown Chunking (fixed-size window with overlap)

def chunk_text(text,chunk_size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    text=" ".join(text.split()) #collapse whitespace

    if not text:
        return []
    chunks=[]

    start=0
    while start<len(text):
        end=start+chunk_size
        chunks.append(text[start:end])
        start+=chunk_size-overlap
    return chunks


def collect_doc_chunks():
    print("\n Chunking Markdown docs....")

    records=[]
    md_files=[]

    for root,_, files in os.walk(DOCS_DIR):
        for f in files:
            if f.endswith(".md"):
                md_files.append(os.path.join(root,f))

    md_files= md_files[:MAX_FILES_PER_SOURCE]

    for path in md_files:
        with open(path,"r",encoding="utf-8",errors="ignore") as fh:
            content=fh.read()
            rel_path = os.path.relpath(path, DOCS_DIR).replace(os.sep, "__")

        for i,chunk in enumerate(chunk_text(content)):
            records.append({
                "id": f"doc::{rel_path}::{i}",
                "text":chunk,
                "metadata":{
                    "source_type":"doc",
                    "file_name":os.path.basename(path),
                    "chunk_index":i,
                },
            })

    print(f"{len(md_files)} files - {len(records)}doc chunks")
    return records


#2. Code chunking at function/class level using Python's ast module (This is the key upgrade over fixed-size chunking: each chunk is a whole function or class, so retrieval returns complete,meaningfule code units.)

def chunk_python_file(path):
    with open(path,"r",encoding="utf-8",errors="ignore") as fh:
        source=fh.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines=source.splitlines()

    chunks=[]

    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            start=node.lineno-1
            end=getattr(node,"end_lineno",start+1)
            snippet="\n".join(lines[start:end])
            docstring=ast.get_docstring(node) or ""
            chunks.append({
                "name":node.name,
                "kind":type(node).__name__,
                "code":snippet,
                "docstring":docstring,
            })

    return chunks

def collect_code_chunks():
    print("\nChunking code at function/class levels...")
    records=[]
    py_files=[]

    for root,_, files in os.walk(CODE_DIR):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root,f))


    py_files=py_files[:MAX_FILES_PER_SOURCE]

    for path in py_files:
        for i, unit in enumerate(chunk_python_file(path)):
            # Embed docstring + code together so semantic search matches
            # both natural-language intent and literal code terms.
            rel_path = os.path.relpath(path, CODE_DIR).replace(os.sep, "__")

            embed_text =f"{unit['kind']} {unit['name']}\n{unit['docstring']}\n{unit['code']}"
            records.append({
                "id": f"code::{rel_path}::{unit['name']}::{i}",
                "text":embed_text[:2000], #guard against very long functions
                "metadata":{
                    "source_type":"code",
                    "file_name":os.path.basename(path),
                    "function_name":unit["name"],
                    "kind":unit["kind"],
                },
            })
    print(f"{len(py_files)} files- {len(records)} code chunks")

    return records
# 3. Embedding via local Ollama (bge-m3),batched

# def embed_batch(texts,batch_size=16):
#     all_embeddings=[]
#     for i in range(0,len(texts),batch_size):
#         batch=texts[i:i+batch_size]
#         resp=requests.post(OLLAMA_URL,json={"model":EMBED_MODEL,"input":batch})
#         resp.raise_for_status()
#         all_embeddings.extend(resp.json()["embeddings"])
#         print(f"embedded {min(i+batch_size,len(texts))}/{len(texts)}")
#     return all_embeddings

import time

def embed_batch(texts,batch_size=16,max_retries=3):
    all_embeddings=[]
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]

        for attempt in range(max_retries):
            resp=requests.post(OLLAMA_URL,json={"model":EMBED_MODEL,"input":batch})
            if resp.status_code == 200:
                break
            print(f"  [retry {attempt+1}/{max_retries}] batch {i} failed: {resp.text[:150]}")
            time.sleep(3)
        else:
            print(f"  [SKIPPED] batch {i} failed after {max_retries} retries, skipping these chunks")
            continue

        all_embeddings.extend(resp.json()["embeddings"])
        print(f"embedded {min(i+batch_size,len(texts))}/{len(texts)}")
    return all_embeddings
# 4. Build the ChromaDB collection (vector index doubles as the hybrid search base; Chroma's default search is vector similarity, add a BM25 pass yourself later for true hybrid-see note at bottom)

def build_vector_store(records):
    print(f"Building ChromaDB collection at {CHROMA_DIR}...")
    client=chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection=client.create_collection(COLLECTION_NAME)
    texts=[r["text"] for r in records]
    ids =[r["id"] for r in records]
    metadatas=[r["metadata"] for r in records]

    print(f" requesting embeddings for {len(texts)} chunks from Ollama ({EMBED_MODEL})....")
    embeddings=embed_batch(texts)

    collection.add(ids=ids,embeddings=embeddings,documents=texts,metadatas=metadatas)
    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}'")
    return collection

# 5. Load the SQL schema as plain text context (no embedding needed)
# This is what you feed the LLM for the text-to-sql branch for the router.

def load_sql_schema_text():
    print("\n Loading SQL schema for the text-to-sql branch...")
    con=sqlite3.connect(SQL_DB_PATH)
    cur=con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables=[row[0] for row in cur.fetchall()]

    lines=[]

    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols=[f"{row[1]} {row[2]}" for row in cur.fetchall()]
        lines.append(f"Table {table}({','.join(cols)})")
    con.close()

    schema_text= "\n".join(lines)
    schema_path=os.path.join(BASE_DIR,"sql","schema.txt")
    with open(schema_path,"w") as fh:
        fh.write(schema_text)

    print(f"wrote schema description to {schema_path}")
    return schema_text

# 6. Quick sanity-check query against the vector store

def test_query(collection,query_text,n_results=3):
    print(f"\nTest query: {query_text}\n")
    query_embedding =embed_batch([query_text])[0]
    results=collection.query(query_embedding=[query_embedding],n_results=n_results)

    for i, doc_id in enumerate(results["ids"][0]):
        meta=results["metadatas"][0][i]
        print(f"{i+1}.{doc_id} [{meta.get('source_type')}]")

if __name__=="__main__":
    doc_records= collect_doc_chunks()
    code_records= collect_code_chunks()
    all_records= doc_records + code_records

    with open("all_chunks.json","w",encoding="utf-8") as f:
        json.dump(all_records,f,ensure_ascii=False,indent=2)
    print(f"Saved {len(all_records)} chunks to all_chunks.json")

    collection=build_vector_store(all_records)
    load_sql_schema_text()

    print("\n Done. Vector store ready at:",CHROMA_DIR)
    print("SQL schema ready at:",os.path.join(BASE_DIR,"sql","schema.txt"))



