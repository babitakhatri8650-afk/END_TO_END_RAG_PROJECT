"""
agent_pipeline.py

The online query pipeline: takes a user question and produces a grounded, cited answer by combining hybrid retrieval (vector + BM25 + rerank) with a text-to-SQL branch, routed by a small LLM decision step.

Depends on:
    -rag_data/ + chroma_store/ (built by setup_datasources.py + ingest_and_embed.py)
    - Ollama running locally with:
        ollama pull bge-m3 (embeddings)
        ollama pull llama3.2 (router+synthesis+text-to-SQL)
    -pip install rank_bm25 sentence-transformers chromadb requests --break-system-packages

    Run: python3 agent_pipeline.py "your question is here"
"""
import os
import re
import json
import sqlite3
import requests
import chromadb
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.join(os.getcwd(),"rag_data")
SQL_DB_PATH = os.path.join(BASE_DIR,"sql","chinook.db")
SCHEMA_PATH = os.path.join(BASE_DIR,"sql","schema.txt")
CHROMA_DIR = os.path.join(os.getcwd(),"chroma_store")
COLLECTION_NAME="docs_and_code"

OLLAMA_GENERATE_URL= "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL="http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"
# LLM_MODEL= "llama3.2:1b"
LLM_MODEL = "llama3.2"


TOP_K_RETRIEVE =15 # Candidates pulled from each of vector + BM25
TOP_K_FINAL =5 #chunks kept after fusion + rerank

# 1. Agent router: one LLM call ,forced into JSON , decides which branch(es) this question needs.

ROUTER_PROMPT="""You are a routing agent for a technical assistant that can search documentation or code (retrieval and query a SQL database (sql).

IMPORTANT: If the user asks "how many","count","total","list" or ask about specific numeric/data values, sets needs_sql to true.
Decide which sources are needed to answer the user's question.
Respond with ONLY valid JSON, no other text, in this exact shape: 
{{"needs_retrieval":true or false, "needs_sql":true or false}}
Question:{question}"""


def call_ollama_generate(prompt,model=LLM_MODEL):
    resp = requests.post(OLLAMA_GENERATE_URL,json={
        "model":model,"prompt":prompt,"stream":False,
    })
    resp.raise_for_status()
    return resp.json()["response"]

def route_query(question):
    raw = call_ollama_generate(ROUTER_PROMPT.format(question=question))
    match=re.search(r"\{.*\}",raw,re.DOTALL)

    if not match:
        # fail safe: if the router output is unparseable,try both branches
        return {"needs_retrieval":True,"needs_sql":True}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"needs_retrieval":True,"needs_sql":True}

# 2. Hybrid retrieval: vector search (ChromaDB)+BM25 keyword search, merged with Reciprocal Rank Fusion (RRF). RRF is simple,has no score normalization issues, and is the standard way to combine two rankers.

def load_collection_for_bm25():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection=client.get_collection(COLLECTION_NAME)
    everything = collection.get(include=["documents","metadatas"])
    return collection, everything

def build_bm25_index(everything):
    tokenized=[doc.lower().split() for doc in everything["documents"]]
    bm25 = BM25Okapi(tokenized)
    return bm25

def embed_query(text):
    resp = requests.post(OLLAMA_EMBED_URL,json={"model":EMBED_MODEL,"input":[text]})
    resp.raise_for_status()
    return resp.json()["embeddings"][0]

def reciprocal_rank_fusion(rankings,k=60):
    """rankings: list of ranked-id-lists (best first). Returns fused ranking."""
    scores={}
    for ranking in rankings:
        for rank,doc_id in enumerate(ranking):
            scores[doc_id]=scores.get(doc_id,0)+1.0/(k+rank+1)

    return sorted(scores.keys(),key=lambda d:scores[d],reverse=True)

def hybrid_retrieve(question,collection,everything,bm25):

    #vector ranking
    q_embedding = embed_query(question)
    vec_results = collection.query(query_embeddings=[q_embedding],n_results = TOP_K_RETRIEVE)
    vec_ranking = vec_results["ids"][0]

    #BM25 ranking
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(bm25_scores)),key=lambda i: bm25_scores[i],reverse=True)
    bm25_ranking = [everything["ids"][i] for i in ranked_indices[:TOP_K_RETRIEVE]]

    #Fuse
    fused_ids = reciprocal_rank_fusion([vec_ranking,bm25_ranking])[:TOP_K_RETRIEVE]

    id_to_index = {doc_id: i for i, doc_id in enumerate(everything["ids"])}
    fused_chunks =[]
    for doc_id in fused_ids:
        idx = id_to_index[doc_id]
        fused_chunks.append({
            "id":doc_id,
            "text":everything["documents"][idx],
            "metadata":everything["metadatas"][idx],
        })

    return fused_chunks

# 3. Reranking with a cross-encoder. Cross-encoders score the (query,chunk) pair jointly, which is far more accurate than cosine similarity alone, at the cost of being slower - hence only run it on the -15 fused candidates, not the whole corpus.

def rerank(question,chunks,top_n=TOP_K_FINAL):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print(" [warn] sentence-transformers not installed,skipping rerank")
        return chunks[:top_n]
    

    try:
        model=CrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception as e:
        print(f" [warn] could not load reranker model ({e}), skipping rerank")
        return chunks[:top_n]


    pairs = [[question,c["text"]] for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks,scores),key=lambda x: x[1],reverse=True)
    return [c for c, _ in ranked[:top_n]]

# 4. Text-to-SQL branch: LLM writes a query against the known schema,the query is validated before execution to block destructive statements.

SQL_PROMPT = """Given this SQLite schema:

{schema}
Write a single SELECT query (no other statement type) that answers this quetion.
Respond with ONLY the SQL query, no explanation, no markdown fences.
Question: {question}"""

FORBIDDEN_SQL_KEYWORDS = ["DROP","DELETE","INSERT","UPDATE","ALTER","ATTACH","PRAGMA",";--"]

def is_sql_safe(query):
    upper = query.upper()
    if not upper.strip().startswith("SELECT"):
        return False
    return not any(kw in upper for kw in FORBIDDEN_SQL_KEYWORDS)

def run_text_to_sql(question):
    with open(SCHEMA_PATH) as f:
        schema = f.read()
    raw_sql = call_ollama_generate(SQL_PROMPT.format(schema=schema,question=question)).strip()
    raw_sql=raw_sql.strip("`").replace("sql\n","",1) if raw_sql.startswith("`")else raw_sql

    if not is_sql_safe(raw_sql):
        return {"query": raw_sql,"error":"Blocked: query failed the safety check","rows":[]}

    try:
        con=sqlite3.connect(SQL_DB_PATH)
        cur = con.cursor()
        cur.execute(raw_sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        con.close()
        return {"query": raw_sql,"columns":columns,"rows":rows[:20],"error":None}
    except sqlite3.Error as e:
        return {"query":raw_sql,"error":str(e),"rows":[]}

# 5. Answer synthesis: Combine retrieved chunks +SQL results into one grounded answer, with citations back to source file/table.

SYNTHESIS_PROMPT= """Answer the user's question using only the context below. Cite the specific file/function or table for each fact you use. If the context doesn't contain the answer, say so plainly.

RETRIEVED CONTEXT:
{context}

SQL_RESULT:
{sql_result}

Question: {question}

ANSWER:

"""

def synthesize_answer(question,chunks,sql_result):
    context_lines  =[]
    for c in chunks:
        meta = c["metadata"]
        label=meta.get("file_name","unknown")
        if meta.get("source_type")=="code":
            label+=f":: {meta.get('function_name','')}"
        context_lines.append(f"[{label}] {c['text'][:400]}")
    context_text ="\n\n".join(context_lines) if context_lines else "(no retrieval performed)"

    sql_text = "(not SQL query run)"
    if sql_result:
        if sql_result.get("error"):
            sql_text=f"SQL error: {sql_result['error']}"

        else:
            sql_text=f"The database was queried and returned this actual result - use these exact numbers directly in your answer:\nQuery: {sql_result['query']}\nColumns: {sql_result['columns']}\nRows:{sql_result['rows']}"
            

    prompt=SYNTHESIS_PROMPT.format(context=context_text,sql_result=sql_text,question=question)
    return call_ollama_generate(prompt)

# 6. Orchestration: write the router, both branches,and synthesis

def answer_question(question):
    print(f"\n Question: {question}")
    print("[router] deciding retrieval path...")
    decision=route_query(question)
    print(f"[router] decision:{decision}")

    chunks=[]
    sql_result =None

    if decision.get("needs_retrieval"):
        print("[retrieval] running hybrid search...")
        collection,everything = load_collection_for_bm25()
        bm25 = build_bm25_index(everything)
        fused = hybrid_retrieve(question,collection,everything,bm25)
        print(f"[retrieval] fused {len(fused)} candidates,reranking...")
        chunks=rerank(question,fused)
        for c in chunks:
            print(f" -{c['id']}")

    if decision.get("needs_sql"):
        print("[sql] generating and running query...")
        sql_result=run_text_to_sql(question)
        print(f" query: {sql_result.get('query')}")

        if sql_result.get("error"):
            print(f" error: {sql_result['error']}")

    print("[synthesis] generating final answer...")
    answer = synthesize_answer(question,chunks,sql_result)
    print("\n --ANSWER-- ")
    print(answer)
    return answer

if __name__=="__main__":
    import sys
    q=" ".join(sys.argv[1:]) or "What tables are available in the database?"
    answer_question(q)
