from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    logger,
)
from schemas import SearchResponse
from services import (
    BaseSearchEngine,
    SauceNaoEngine,
    TraceMoeEngine,
    VisionLLMEngine,
    get_chain_terminal_error,
    get_trace_frame_count,
    reset_search_context,
)


trace_moe_engine: BaseSearchEngine = TraceMoeEngine()
vision_llm_engine: BaseSearchEngine = VisionLLMEngine()
saucenao_engine: BaseSearchEngine = SauceNaoEngine()
trace_moe_engine.set_next(vision_llm_engine).set_next(saucenao_engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(connect=10.0, read=45.0, write=45.0, pool=10.0)
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="AnimeLens API",
    description=(
        "动漫截图三级级联识图服务：优先查询 Trace.moe，低置信度时调用 Vision LLM，最后使用 SauceNAO。"
        "可在 /docs 中使用 Swagger UI 上传图片并测试接口。"
    ),
    version="3.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "system", "description": "服务状态检查"},
        {"name": "search", "description": "动漫截图识别"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


#@app.get("/", include_in_schema=False)
#async def index() -> FileResponse:
#   return FileResponse("static/index.html")
# 修改 main.py 中的路由部分

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    # 根目录默认展示新版 V2
    return FileResponse("static/persona5.html")

@app.get("/v1", include_in_schema=False)
async def index_old() -> FileResponse:
    # 访问 http://127.0.0.1:8000/v1 可以随时看回旧版
    return FileResponse("static/index.html")

@app.get(
    "/api/health",
    tags=["system"],
    summary="检查服务健康状态",
)
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/search",
    response_model=SearchResponse,
    response_model_by_alias=True,
    tags=["search"],
    summary="使用三级级联引擎识别动漫截图",
    description=(
        "上传 JPG/JPEG 或 PNG 图片（最大 10MB）。按 Trace.moe、Vision LLM、SauceNAO 的顺序级联识别。"
        "返回的 results 使用统一结构，similarity 为 0 到 1 的小数。"
    ),
    responses={
        400: {"description": "图片为空"},
        413: {"description": "图片超过 10MB"},
        415: {"description": "图片格式不支持"},
        502: {"description": "三级识图服务均不可用或未配置"},
        504: {"description": "上游请求超时"},
    },
)
async def search_image(file: UploadFile = File(..., description="待识别的 JPG/PNG 图片")) -> SearchResponse:
    filename = file.filename or "uploaded-image"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = (file.content_type or "").lower()

    if content_type not in ALLOWED_CONTENT_TYPES or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="仅支持 JPG/JPEG 和 PNG 图片。",
        )

    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="图片大小不能超过 10MB。",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="上传的图片为空。",
        )

    client: httpx.AsyncClient = app.state.http_client
    reset_search_context()
    results = await trace_moe_engine.search(
        image_bytes=contents,
        filename=filename,
        content_type=content_type,
        client=client,
    )

    if results:
        return SearchResponse(
            results=results,
            frameCount=get_trace_frame_count(),
            engine=results[0].engine,
        )

    terminal_error = get_chain_terminal_error()
    if terminal_error is not None:
        logger.error("All three search engines failed: %s", terminal_error)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Trace.moe、Vision LLM 和 SauceNAO 均暂时不可用，请稍后重试或检查 API 配置。",
        ) from terminal_error

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="三个识图引擎都没有找到可靠的匹配结果，请换一张更清晰的截图重试。",
    )
