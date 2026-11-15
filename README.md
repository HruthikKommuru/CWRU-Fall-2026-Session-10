# Session 10 · RAG and a Document Chatbot

Turn Session 9's semantic search into a chatbot that answers from the TrailMix support handbook and cites its evidence. **Load → Chunk → Store → Retrieve → Augment → Generate → Chat → Evaluate.** For example, “Can I return a tent I used once?” should explain the used-item policy with `[Policies > Returns]`; the follow-up “What about boots?” should retrieve the footwear exception and explain the one-time size/width exchange within 30 days, with no refund. TrailMix Outdoors is fictional, and its handbook is teaching material.

| Job | Service / model |
| --- | --- |
| Embeddings | Gemini `gemini-embedding-001` |
| Chat, question rewriting, and answer judge | Gemini `gemini-3.7-flash` |
| Vector database | Pinecone, index `trailmix-handbook`, as in Session 9 |
| Session 10 namespaces | `heading-800` for the chatbot; `fixed-350` for the notebook comparison |

## Setup

Use **Python 3.13**, the version used to verify the pinned dependencies together. The Pinecone integration pin requires Python below 3.14. One Google key powers all Gemini operations; Pinecone needs a separate key. Running service cells uses API quota and may incur charges on your accounts.

```bash
git clone https://github.com/HruthikKommuru/CWRU-Fall-2026-Session-10.git
cd CWRU-Fall-2026-Session-10
python3.13 -m venv .venv          # Create the environment once.
source .venv/bin/activate        # Activate it in this terminal.
pip install -r requirements.txt
cp .env.example .env             # Do this once; then fill in your keys.
```

On **Windows PowerShell**, create with `py -3.13 -m venv .venv`, activate with `.\.venv\Scripts\Activate.ps1`, install with `python -m pip install -r requirements.txt`, and copy the template with `Copy-Item .env.example .env`. In Command Prompt, use `.venv\Scripts\activate.bat` to activate and `copy .env.example .env` to copy the template.

In each new terminal, return to the repository folder and **activate the existing environment again**; you do not need to recreate it or overwrite `.env`. Check `python --version` reports 3.13 and `python -m pip --version` shows a path inside `.venv`. Run `deactivate` when finished.

Edit `.env` with `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey) and `PINECONE_API_KEY` from the [Pinecone console](https://app.pinecone.io/). `.env` stores configuration; `.venv/` stores Python packages. Both are ignored by Git. The two model strings are constants in both implementations; there is no automatic model/provider fallback.

## Two ways to run

**Learn each step in the notebook.** Launch Jupyter from the repository folder, select the environment's kernel, and run top to bottom. The notebook is saved with no outputs and has not been executed; live cells need your keys. Its final cell is an interactive chat loop; type `quit` to stop.

```bash
jupyter notebook session10_lab.ipynb
```

**Run the complete chatbot.** Ingest once, then ask or chat; flags can be combined.

```bash
python chatbot.py --ingest
python chatbot.py --ask "Can I return a tent I used once?"
python chatbot.py --chat
python chatbot.py --eval
python chatbot.py --ingest --eval
python chatbot.py --chat --verbose
```

`--ingest` loads and stores the main chunks only. `--ask` answers one question without persistent history; `--chat` rewrites follow-ups using the last six messages (three user/assistant turns) and keeps running after a failed request. `--eval` reports retrieval hit rate and judged answer correctness for all ten questions, returning a non-zero exit code for fewer than eight retrieval hits or a service failure. `--verbose` adds retrieved labels and rewritten questions to the time-stamped logs; normal progress goes to logs and bot answers go to standard output.

## What changes from Session 9

- **Two-stage chunking:** split at `#` section and `##` topic headings, then recursively split long topics with `chunk_size=800` and `chunk_overlap=100` characters. Heading metadata is retained; overlap is a target and natural paragraph/word boundaries can reduce it.
- **Failure demo:** hard-cut each labeled heading piece into 350-character chunks with `separator=""` and no overlap. Heading pieces supply citation metadata only; this fixed splitter does not protect sentence or word boundaries. The used-item rule is visibly split inside “Warranty”.
- **Grounding and citations:** top four main-namespace chunks become labeled excerpts in the system prompt. The model is instructed to cite every claim or say “The handbook doesn't cover this — route to General Support.” Inspect real outputs because a prompt does not guarantee compliance.
- **Memory:** a separate rewrite chain restores the meaning of a follow-up before retrieval. The answer still comes from the handbook, and unsuccessful turns do not enter history.
- **Evaluation:** retrieval checks for one chunk matching both the expected section and topic; a Gemini judge compares each answer with the CSV's required fact using a plain JSON schema. Judge grades need human spot-checking and are separate from citation verification.

## Keep the namespaces separate

Session 9's records remain in the **default namespace**. Session 10 writes only to `heading-800` and, in the notebook, `fixed-350`, using stable IDs such as `heading-800-0`. Retrieval passes both `k` and the namespace explicitly, so the chunking experiments do not mix.

Repeated ingestion of the unchanged handbook overwrites the same IDs. If you change chunking or source content, use a new experiment namespace or clear only the affected Session 10 namespace before ingesting: a smaller new chunk set can otherwise leave old IDs behind. Never clear the default namespace for this lab.

## Repository structure

```text
CWRU-Fall-2026-Session-10/
├── README.md                    # Setup, usage, evaluation, and troubleshooting
├── requirements.txt             # Verified direct dependency pins
├── .env.example                 # Empty Google and Pinecone key fields
├── .gitignore                   # Keys, environments, caches, Word lock files
├── data/
│   ├── support_handbook.docx     # Original handbook, unchanged
│   └── eval_questions.csv        # Ten questions, expected headings, required facts
├── session10_lab.ipynb           # Each RAG step, chunk comparison, memory, evaluation
└── chatbot.py                   # Runnable ingestion, ask, chat, and evaluation CLI
```

## Done means

- Hit rate @4 is at least **8/10** in `heading-800`.
- Every factual claim cites a real handbook section and is supported by its passage.
- The CEO hiking-trail question gets “doesn't cover this”.
- “What about boots?” works as a follow-up to the used-item return question.
- No API keys are committed.

Local verification finds **7 sections, 58 topics, 59 heading pieces, 67 main chunks, and 108 fixed-size chunks**. Returns produces **2 main chunks** (788 and 230 characters). The original Word file is unchanged, and both chunk sets can be built without keys; the notebook is unexecuted. Live model availability, retrieval hit rate, and answer/judge quality require your keys and have not been measured here.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Missing key | Copy `.env.example` to `.env` once and fill both values; never commit them. An authentication error means check your key and account access. |
| Handbook missing | Put the unchanged file at `data/support_handbook.docx`. |
| Dimension or metric mismatch | The lab refuses to write. An index cannot change dimension in place: deletion and recreation require re-ingestion, and **deleting this shared index removes Session 9 data too**. Preserve the original data/settings and arrange to restore Session 9 before recreating `trailmix-handbook` with the measured dimension and cosine metric; the code never deletes it automatically. |
| 429 / rate limit | Wait and retry; check quota and billing. The chat, rewrite, and judge requests use up to three wrapper attempts before reporting failure. |
| Empty retrieval | Did you run `--ingest`? Wait a few more seconds after upsert, then retry; Pinecone search visibility is eventual. |
| `iter_inner_content` missing | Upgrade with `pip install --upgrade 'python-docx>=1.1'`; this lab pins 1.2.0. |
| Session 9 data looks different | Session 9 uses the default namespace; Session 10 uses its own namespaces. Confirm which namespace you are viewing in Pinecone. |
| Model unavailable | Check access in AI Studio and discuss the model constants with your instructor; no model is silently replaced. |
| Fixed-size answer looks correct | Top-four retrieval may recover both sides of a cut. Inspect the printed boundary and evidence; the demo exposes a failure mode rather than promising every generated answer will fail. |
| Notebook kernel restarted | Rerun setup and rebuild the Python variables/chains; stored Pinecone records persist separately. |
