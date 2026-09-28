# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Project overview

AnimeLens is a small full-stack FastAPI application that identifies anime, Chinese animation, comic, illustration, and poster images. The backend accepts a JPG/PNG upload at `/api/search` and uses a three-level cascade: Trace.moe first, a configured OpenAI-compatible Vision LLM when Trace.moe has no reliable match, and SauceNAO as the final fallback. The browser UI is a single page in `static/index.html`; there is no frontend build pipeline, and Tailwind CSS is loaded from the CDN at runtime.

## Development commands

Create and activate a virtual environment, then install the dependency ranges:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Run the development server from the repository root because the application uses relative paths for `static/`:

```bash
uvicorn main:app --reload
```

The app is served at `http://127.0.0.1:8000`. Use `/docs` for Swagger UI, `/openapi.json` for the generated schema, and `/api/health` for a basic health check.

The service reads configuration from environment variables and a local `.env` file via `python-dotenv`. The key variables are `SAUCENAO_API_KEY` for the third-level fallback, plus `VISION_API_KEY`, `VISION_BASE_URL`, and optional `VISION_MODEL` for the Vision LLM. `VISION_BASE_URL` should be the provider root; the service appends `/v1/chat/completions`. For example, in PowerShell:

```powershell
$env:SAUCENAO_API_KEY = "your-key"
$env:VISION_API_KEY = "your-key"
$env:VISION_BASE_URL = "https://api.openai.com"
$env:VISION_MODEL = "gpt-4o-mini"
```

There is no configured build command, linter, formatter, frontend package manager, or test suite at present. Run the available static checks with:

```bash
python -m compileall main.py config.py schemas.py services.py
python -m pip check
```

If tests are added, run one test with `pytest path/to/test_file.py::test_name -q` or the full suite with `pytest -q`.

## Architecture and request flow

`main.py` owns the HTTP application and orchestration. The FastAPI lifespan creates one shared `httpx.AsyncClient` with connect/read/write/pool timeouts and closes it during shutdown. The `/api/search` handler validates both the filename extension and MIME type, reads at most 10 MB plus one byte to detect oversized uploads, then executes the cascade below. Preserve the existing status-code behavior and let one upstream failure move the request to the next level rather than aborting the whole search.

The first level is `_request_tracemoe` in `services.py`, which uploads the image to Trace.moe, normalizes result titles and similarity values, and uses AniList GraphQL lookups when Trace.moe does not include a usable title. A Trace.moe result set is returned immediately only when its best normalized similarity is at least `TRACE_CONFIDENCE_THRESHOLD` (currently `0.85`).

The second level is `_request_llm_vision`. It base64-encodes the image and sends an OpenAI-compatible `POST` request to `{VISION_BASE_URL}/v1/chat/completions`, using the configured model and a JSON-only anime-identification prompt. The function converts a valid non-`UNKNOWN` response into a `SearchResult` with `engine="ai_vision"`, the returned title, optional `character`, and optional `description`. Missing configuration, timeouts, HTTP errors, malformed JSON, and `UNKNOWN` responses are represented as `UpstreamSearchError` so `main.py` can continue to SauceNAO.

The third level is `_request_saucenao`. It requires `SAUCENAO_API_KEY`, sends the original upload to SauceNAO, and normalizes source/title, similarity, image, and thumbnail fields into `SearchResult` objects. If SauceNAO returns usable results, they are returned. If it fails and Trace.moe produced low-confidence results, the handler may return those original Trace.moe results as the best available response; if no usable engine result exists, it returns a friendly HTTP error.

`config.py` centralizes upstream URLs, credentials, the Vision model, thresholds, file limits, allowed MIME types/extensions, logging, and the in-memory AniList title cache. `schemas.py` defines the stable API contract: `SearchResult.engine` is one of `tracemoe`, `ai_vision`, or `saucenao`; similarity is normalized to `0..1`; `video_url`, image fields, episode/time fields, `character`, and `description` are optional. `SearchResponse` also exposes the selected engine and Trace.moe `frameCount` using Pydantic aliases.

The frontend is embedded in `static/index.html` and handles drag-and-drop, file selection, clipboard image paste, client-side type/size checks, loading and error states, and result-card rendering. It displays engine badges and supports results without a video URL, as expected for Vision LLM and SauceNAO. API failures should continue to use FastAPI's `detail` field because the page renders that message directly. Keep user-controlled result values escaped through the existing `escapeHtml` boundary before inserting them into HTML. The page is mounted at `/static` and served at `/`.

CORS is intentionally open to all origins because the API may be developed separately from the static page; change `allow_origins` in `main.py` if deployment needs a restricted list. Trace.moe, AniList, the Vision provider, and SauceNAO are external services with independent rate limits and availability, so network changes must preserve timeout handling, response validation, and graceful cascade behavior.
