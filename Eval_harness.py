"""
Eval_harness.py

A lightweight, RAGAs-style evaluation harness that runs entirely on you local Ollama model (no paid API needed). It measures:

    -faithfulness   :is the answer actually grounded in retrieved context, or did the model make things up?
    -answer_relevancy: does the answer actually address the question asked?
    -router_accuracy: did the agent router pick the source(s) you expected?


Why  not the 'rags' pip package directly: at the time this was written, raga's dafault import chain pulls in an unrelated Vertex AI integration that fails to import in a clean environment. This harness measures the same core ideas (LLM-as-judge scoring against retrieved context) without that dependency, and runs for free against a local model.Swap in the real 'ragas' library later if you want - the test set and comparison table format below still applies.
 
Depends on: agent_pipeline.py (for answer_question / route_query /hybrid_retrieve)

Run: python3 eval_harness.py
"""
import json
import re
import statistics
import requests

OLLAMA_GENERATE_URL="http://localhost:11434/api/generate"
# JUDGE_MODEL="llama3.2:1b"
JUDGE_MODEL="llama3.2"

# 1. Test question set. Each question has an expected router decision (for router_accuracy) so you catch misrouting,which is the most common real failure mode in an agentic RAG system.

TEST_SET=[
    {
        "question":"How does the requests library handle connection timeouts?",
        "expected_retrieval":True,
        "expected_sql":False,
    },
    {
        "question":"How many customers are in the database?",
        "expected_retrieval":False,
        "expected_sql":True,

    },
    {
        "question":"What does the merge_setting function do, and how many albums does artist AC/DC have?",
        "expected_retrieval":True,
        "expected_sql":True,
    },
    {
        "question":"What is FastAPI's approach to dependency injection?",
        "expected_retrieval":True,
        "expected_sql":False,
    },
    {
        "question":"List the 5 most expensive tracks in the store.",
        "expected_retrieval":False,
        "expected_sql":True,
    },
    {
        "question":"What's the weather like today?",
        "expected_retrieval":False,
        "expected_sql":False,
    },
]

# 2. LLM-as-judge scoring prompts. Each returns a 1-5 integer score, parsed strictly so a malformed judge response doesn't silently corrupt the average.

FAILTHFULNESS_PROMPT = """You are grading whether an AI answer is faithful to its provided context,i.e every claim in the answer is actually supported by the context (not invented).

CONTEXT:
{context}

ANSWER:
{answer}

Score faithfulness from 1 (answer contains claims not in context) to 5 (fully grounded).
Respond with ONLY a single integer 1-5."""

RELEVANCY_PROMPT="""You are grading whether an AI answer actually addresses the question asked.

QUESTION:{question}

ANSWER:{answer}

Score relevancy from 1 (off-topic doesn't answer the question) to 5 (directly and fully answers it).
Respond with ONLY a single integer 1-5.
"""

def call_judge(prompt):
    resp = requests.post(OLLAMA_GENERATE_URL,json={
        "model":JUDGE_MODEL,"prompt":prompt,"stream":False,
        "options":{"temperature":0}
    })
    resp.raise_for_status()
    raw = resp.json()["response"]
    match=re.search(r"[1-5]",raw)
    return int(match.group(0)) if match else None

def score_faithfulness(context,answer):
    return call_judge(FAILTHFULNESS_PROMPT.format(context=context,answer=answer))

def score_relevancy(question,answer):
    return call_judge(RELEVANCY_PROMPT.format(question=question,answer=answer))

# 3. Router accuracy: compare the pipeline's actual routing decision against the expected one for each test question.

def score_router(actual_decision,expected):
    correct = (
        actual_decision.get("needs_retrieval",False)==expected["expected_retrieval"]
        and actual_decision.get("needs_sql",False)==expected["expected_sql"]
    )
    return 1 if correct else 0

# 4. Run the full harness against the live pipeline (agent_pipeline.py)

def run_harness():
    import agent_pipeline as pipeline # local import so this file works standalone for parts 1-3

    results=[]
    for item in TEST_SET:
        question=item["question"]
        print(f"\nEvaluation: {question}")

        decision = pipeline.route_query(question)
        router_score = score_router(decision,item)

        chunks=[]
        sql_result =None

        if decision.get("needs_retrieval"):
            collection,everything = pipeline.load_collection_for_bm25()
            bm25 = pipeline.build_bm25_index(everything)
            fused = pipeline.hybrid_retrieve(question,collection,everything,bm25)
            chunks = pipeline.rerank(question,fused)

        if decision.get("needs_sql"):
            sql_result=pipeline.run_text_to_sql(question)

        answer = pipeline.synthesize_answer(question,chunks,sql_result)

        context_text = "\n".join(c["text"][:400] for c in chunks)

        if sql_result:
            context_text+=f"\nSQL:{sql_result.get('query')}-> {sql_result.get('rows')}"
        faithfulness = score_faithfulness(context_text,answer) if context_text.strip() else None
        relevancy = score_relevancy(question,answer)

        results.append({
            "question":question,
            "router_correct":router_score,
            "faithfulness":faithfulness,
            "relevancy":relevancy,
        })
        print(f"router_correct ={router_score} faithfulness ={faithfulness} relevancy={relevancy}")
    return results

# 5. Aggregate into a summary table.

def summarize(results,config_name="default"):
    router_acc = statistics.mean(r["router_correct"] for r in results)
    faith_scores = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    rel_scores = [r["relevancy"] for r in results if r["relevancy"] is not None]

    summary = {
        "config":config_name,
        "router_accuracy":round(router_acc,2),
        "avg_faithfulness":round(statistics.mean(faith_scores),2) if faith_scores else None,
        "avg_relevancy":round(statistics.mean(rel_scores),2) if rel_scores else None,
        "n_questons":len(results),
    }
    return summary

def save_results(results,summary,path="Eval_results.json"):
    with open(path,"w") as f:
        json.dump({"summary":summary,"per_question":results},f,indent=2)
    print(f"\nSaved detailed results to {path}")

if __name__=="__main__":
    results = run_harness()
    summary = summarize(results,config_name="hybrid_plus_rerank")
    # summary = summarize(results,config_name="Vectory_only_no_rerank")

    save_results(results,summary,path="Eval_results_no_rerank")

    print("\n--- SUMMARY ---")

    for k,v in summary.items():
        print(f"{k}:{v}")

    print("\nTrip: comment out the reranker call in agent_pipeline.py's rerank(),")
    print("\nrerun this harness, and compare the two summaries - that before/after")
