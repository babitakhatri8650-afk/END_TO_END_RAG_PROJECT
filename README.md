# Agentic RAG Assistant — Docs + Code + SQL

A smart Q&A assistant that can answer questions using three different kinds of information:
written documentation, source code, and a database. Instead of guessing which one to check,
it has a small "router" that decides where to look before answering — and it always shows
where the answer came from.

## What problem does this solve?

Most simple chatbots only search one type of data — usually just documents. But real
questions often need more than one source. For example: *"What does this function do, and
how many customers are in our database?"* needs both the code and the database. This
project handles that by picking the right source (or sources) automatically for each
question.

## How it works (in plain terms)

1. **You ask a question.**
2. **A small AI model reads your question and decides:** does this need me to search
   documents/code, look something up in the database, or both?
3. **If it needs to search documents/code:** it looks for matching text two different ways
   (by meaning, and by exact keywords), combines both results, then double-checks the best
   matches to make sure they're actually relevant.
4. **If it needs the database:** it writes a database query on its own, checks that the
   query is safe (won't delete or change anything), and runs it.
5. **Finally, it combines whatever it found and writes one clear answer** — always pointing
   back to exactly which file, function, or database table the information came from.

## Why this is more than a basic chatbot

- It doesn't just search one place — it picks the right source(s) for each question.
- It shows its sources, so you can check the answer isn't made up.
- It was tested with real questions to measure how accurate it actually is (not just "it
  seems to work").
- Everything runs on your own computer for free — no paid subscription or API needed.

## What it's built from

| Source | What it is |
|---|---|
| Documentation | FastAPI's official docs (you can swap this for your own documents) |
| Code | The `requests` Python library's source code (swap for your own project) |
| Database | A sample music-store database (artists, albums, customers, etc.) |

## Tools used, and why

- **Ollama** — runs the AI models on your own computer, for free, with no internet needed
  after setup.
- **bge-m3** — turns text into a format the computer can compare for meaning (this is what
  "search by meaning" uses).
- **llama3.2** — the AI model that decides which source to search, writes database queries,
  and writes the final answer.
- **ChromaDB** — stores all the document/code text in a searchable form.
- **BM25** — classic keyword search, good for finding exact terms like function names or
  error codes.
- **A reranker model** — double-checks the top search results to make sure the best ones
  come first.
- **Streamlit** — builds the simple chat website you interact with.

## How to run it yourself

```bash
# 1. Install Ollama (the free local AI runner) from https://ollama.com
# 2. Download the two AI models it needs:
ollama pull bge-m3
ollama pull llama3.2

# 3. Install the Python tools this project needs:
pip install chromadb rank_bm25 sentence-transformers streamlit requests --break-system-packages

# 4. Put your documents, code, and database into the rag_data folder
#    (or run this script to download example ones automatically):
python3 setup_datasources.py

# 5. Build the searchable index from that data:
python3 ingest_and_embed.py

# 6. Start the chat assistant:
streamlit run app.py

# 7. (Optional) Check how accurate it is:
python3 eval_harness.py
```

## See it in action

The included demo video [Demo](demo.mp4) shows the assistant handling four different kinds of questions in
one session: a documentation question, a question needing both code and the database, a
database-only question, and a question it correctly refuses to answer because it's outside
what it knows (like asking about the weather).

## How well does it actually work?

I tested it with 6 sample questions and had a second AI model grade the answers — checking
whether each answer was actually backed up by real information (not made up), and whether
it actually answered the question asked.

| Setup | Picked the right source? | Answer backed by real info? | Actually answered the question? |
|---|---|---|---|
| Basic search only | 83% | 3.75 / 5 | 3.33 / 5 |
| Basic search + double-checking step | 83% | 3.80 / 5 | 3.50 / 5 |

The double-checking step (reranking) gave a small improvement. The improvement is small
partly because I only tested 6 questions — testing with 50-100 questions would give a much
clearer picture, and that's the natural next step to make this more rigorous.

One thing I found: the question "List the 5 most expensive tracks" was consistently
misrouted — the assistant sometimes searched documents instead of the database for this
specific phrasing. This is a good, honest example of the kind of mistake small AI models
make, and it's exactly the kind of issue this testing process is designed to catch.

## What's inside each file

```
setup_datasources.py   -> downloads/prepares the documents, code, and database
ingest_and_embed.py    -> breaks everything into searchable pieces and indexes them
agent_pipeline.py      -> the core logic: deciding where to search, searching, answering
eval_harness.py        -> the testing script that measures accuracy
app.py                 -> the chat website you actually talk to
```

## A few design choices explained

- **Code is split by function, not by fixed chunks of text** — so when the assistant shows
  you a piece of code, it's always a complete, working function, not a random cut-off
  fragment.
- **Two search methods are combined** (search by meaning + exact keyword search) because
  each one is bad at what the other is good at — meaning-based search misses exact terms
  like function names, and keyword search misses paraphrased questions.
- **Every database query is checked before running** — only simple "look something up"
  queries are allowed; anything that could delete or change data is blocked automatically.
- **If the decision-making step gives a confusing answer**, the system plays it safe and
  searches everywhere rather than searching nowhere.
- **The double-checking model is only loaded once**, not every single time you ask a
  question — this makes the second and later questions much faster.

## What this project doesn't do (and that's okay)

This is a learning/portfolio project, not something ready for real companies to use as-is.
It doesn't handle multiple users with different permissions, doesn't have monitoring for
when something goes wrong in production, and hasn't been tested at large scale. Those are
the natural next steps if this were turned into a real product.
