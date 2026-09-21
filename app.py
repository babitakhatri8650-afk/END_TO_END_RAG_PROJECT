"""Streamlit chat UI for the agentic RAG assistant. Shows the router's decision as a badge, the synthesized answer, and a source panel with citations back to the exact file/function/table used - matching the citation-first design that actually proves the system isn't hallucinating.

Depends on: agent_pipeline.py , and a populated rag_data/+chroma_store/
Run: streamlit run app.py
"""

import  streamlit as st
import agent_pipeline as pipeline

st.set_page_config(page_title="Course + Repo Assistant",page_icon="💬",layout="centered")

# 1. page header + session state for chat history

st.title("Course + repo assistant")
st.caption("Ask about the docs, the codebase, or the database - one router decides where to look.")

if "history" not in st.session_state:
    st.session_state.history =[] # list of dicts: question,answer,decision,chunks,sql_result

# 2. Render past turns: user bubble,router badge, answer,sources

def render_router_badge(decision):
    labels=[]
    if decision.get("needs_retrieval"):
        labels.append("docs/code")
    if decision.get("needs_sql"):
        labels.append("sql")

    if not labels:
        labels.append("out of scope")

    st.markdown(
        f"<span style='background:#f3f4f6;border-radius:6px;padding:2px 10px;"
        f"font-size:12px;color:#374151;'>router:{'+'.join(labels)}</span>",
        unsafe_allow_html=True,
    )


def render_sources(chunks,sql_result):
    if not chunks and not (sql_result and not sql_result.get("error")):
        return
    with st.expander(f"Sources ({len(chunks)} retrieved" + (" +sql)" if sql_result else ")")):
        for c in chunks:
            meta=c["metadata"]

            if meta.get("source_type")=="code":
                st.markdown(f" **code** - `{meta.get('file_name')}`,function `{meta.get('function_name')}`")
            else:
                st.markdown(f" **doc** -`{meta.get('file_name')}`")
            st.caption(c["text"][:200]+"...")
        if sql_result and not sql_result.get("error"):
            st.markdown(f" **SQL** -`{sql_result['query']}`")
            st.caption(f"columns: {sql_result['columns']} | {len(sql_result['rows'])} rows returned")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        render_router_badge(turn["decision"])
        st.write(turn["answer"])
        render_sources(turn["chunks"],turn["sql_result"])

# 3. Handle new input: run the full pipeline, stream progress,store the turn in session history so it persists in the chat view.

question = st.chat_input("Ask a question about the docs, code or database...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        status=st.status("Routing question...",expanded=False)

        decision= pipeline.route_query(question)
        status.update(label=f"Routed to: {decision}")

        chunks=[]
        sql_result=None

        if decision.get("needs_retrieval"):
            status.update(label="Running hybrid search...")
            collection,everything = pipeline.load_collection_for_bm25()
            bm25 = pipeline.build_bm25_index(everything)
            fused = pipeline.hybrid_retrieve(question,collection,everything,bm25)
            status.update(label="Reranking candidates...")
            chunks=pipeline.rerank(question,fused)

        if decision.get("needs_sql"):
            status.update(label="Generating SQL query...")
            sql_result=pipeline.run_text_to_sql(question)

        status.update(label="Synthesizing answer...")
        answer = pipeline.synthesize_answer(question,chunks,sql_result)
        status.update(label="Done",state="complete")

        render_router_badge(decision)
        st.write(answer)
        render_sources(chunks,sql_result)

    st.session_state.history.append({
        "question":question,
        "answer":answer,
        "decision":decision,
        "chunks":chunks,
        "sql_result":sql_result,
    })

# 4. Sidebar: quick stats + reset

with st.sidebar:
    st.header("Session")
    st.write(f"Question asked: {len(st.session_state.history)}")
    if st.button("Clear conversation"):
        st.session_state.history=[]
        st.rerun()

    st.divider()
    st.caption(
        "Architecture:  agent router - hybrid retrieval"
        "(vector + BM25 +rerank) and/or text-to-sql - answer synthesis"
        "with citations. All models run locally via Ollama - no API cost."
    )
    