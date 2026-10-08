# AGENTS.md

Notes for contributors and coding agents working on this repo.

## Install

```sh
python -m pip install -r requirements-dev.txt   # app requirements + pytest + playwright
python -m playwright install chromium           # add --with-deps on a fresh Linux host
```

## Test tiers

| Tier | What it covers | Secrets | When it runs |
| --- | --- | --- | --- |
| 1. Streamlit smoke | Boots `chat_with_pdf.py` headlessly, checks the UI renders, uploads a small PDF, sends a chat message, and fails on any traceback in the page or server log. Couchbase and Gemini are replaced by in-memory fakes (`tests/smoke_app.py`). | None | Every PR and push to `master` (`Streamlit smoke (no secrets)` job). This is the required check. |
| 2. Gemini provider smoke | Calls the live Gemini API with the exact model names in `chat_with_pdf.py`: `embed_query` must return a non-empty numeric vector and each LLM must return a non-empty response. Couchbase is not used. | `GOOGLE_API_KEY` | `Gemini provider smoke (optional, GOOGLE_API_KEY)` job, on PRs from this repo, pushes to `master`, and manual `workflow_dispatch`. If the secret is not available (fork and Dependabot PRs), the job passes with a "skipped" notice. |
| 3. Live Couchbase RAG validation | Full app against a real cluster and Search vector index. | All app secrets | Manual only (see below). |

Commands:

```sh
# Tier 1: no secrets needed
python -m pytest tests/test_streamlit_smoke.py -v

# Tier 2: skipped unless GOOGLE_API_KEY is set in the environment
python -m pytest tests/test_gemini_provider.py -v -rs
```

To run tier 2 on a Dependabot branch, start the `Smoke tests` workflow with
`workflow_dispatch` on that branch, or add `GOOGLE_API_KEY` as a Dependabot
secret as well.

## Secrets

Names only. Never commit values. `.streamlit/secrets.toml` must keep empty values.

- `GOOGLE_API_KEY`: GitHub Actions secret, used by tier 2. Also needed for running the app.
- `DB_CONN_STR`, `DB_USERNAME`, `DB_PASSWORD`, `DB_BUCKET`, `DB_SCOPE`,
  `DB_COLLECTION`, `INDEX_NAME`, `LOGIN_PASSWORD`: app settings for tier 3
  only. They are not in CI.

## Manual live validation for dependency-update PRs

CI does not run the app against Couchbase. Before merging a dependency update
(especially `langchain-*`, `couchbase`, `streamlit`, `pypdf`), someone should
run the app against a real cluster and attach evidence to the PR:

1. **App boot**: `streamlit run chat_with_pdf.py` starts with no errors (paste the console output).
2. **Couchbase / vector index state**: the cluster is reachable, and the Search
   index (`demoSearchIndex.json`, 768-dim `embedding` field) exists on the
   configured bucket/scope/collection and is ready.
3. **Provider credentials**: `GOOGLE_API_KEY` is set and tier 2 passes (or
   explain why it was skipped).
4. **Sample input**: upload a small PDF and note its name/size.
5. **Query and observed response**: ask a question answered by the PDF. Record
   the question, the RAG answer, and the pure-LLM answer.
6. **Screenshots or traces** of the chat when practical.
