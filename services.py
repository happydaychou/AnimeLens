from __future__ import annotations

import base64
import json
import math
import re
from typing import Any

import httpx

from config import (
    ANILIST_GRAPHQL_URL,
    ANILIST_TITLE_CACHE,
    SAUCENAO_API_KEY,
    SAUCENAO_URL,
    TRACE_MOE_URL,
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_MODEL,
    logger,
)
from schemas import SearchResult, UpstreamSearchError


async def _fetch_anilist_cover(
    client: httpx.AsyncClient,
    anime_title: str,
) -> str | None:
    """Look up an official AniList cover for a title identified by Vision LLM."""
    query = """
    query ($search: String) {
      Media(search: $search, type: ANIME) {
        coverImage { large }
      }
    }
    """
    try:
        response = await client.post(
            ANILIST_GRAPHQL_URL,
            json={"query": query, "variables": {"search": anime_title}},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            logger.warning("AniList cover lookup returned GraphQL errors: %s", payload["errors"])
            return None
        cover_url = payload.get("data", {}).get("Media", {}).get("coverImage", {}).get("large")
        return _normalise_name(cover_url)
    except (httpx.TimeoutException, httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
        logger.warning("AniList cover lookup failed for %r: %s", anime_title, exc)
        return None


async def _request_llm_vision(
    client: httpx.AsyncClient,
    contents: bytes,
    content_type: str,
) -> SearchResult:
    """Identify non-Trace.moe images through an OpenAI-compatible vision API."""
    if not VISION_API_KEY:
        raise UpstreamSearchError(
            "Vision LLM is not configured; set the VISION_API_KEY environment variable"
        )

    image_data = base64.b64encode(contents).decode("ascii")
    system_prompt = (
        "你是一个资深动漫百科全书。请识别用户上传图片的动漫、国漫或漫画出处。"
        "请在返回的 JSON 中包含字段："
        "title（作品名称；不知道请填 UNKNOWN）；"
        "character（角色名字；未知可填 null）；"
        "description（一句简短的背景或特征说明）；"
        "confidence（你的判断置信度，必须是 0.60 到 0.98 之间的浮点数，不要总是给 1.0 或 0.99）。"
        "打分参考：0.92-0.98 表示角色标志性特征极强，或截图中含有明确台词、水印等直接证据；"
        "0.80-0.91 表示画风或角色非常熟悉，大概率是该作品但缺乏决定性特征；"
        "0.65-0.79 表示大众画风或常见人设，属于合理推测；"
        "低于 0.60 时请将 title 填为 UNKNOWN。"
        "如果图片中有文字，请结合文字推理。必须严格返回 JSON，不要添加额外解释。"
    )
    user_prompt = (
        '请只返回 JSON，例如：{"title":"作品名称","character":"角色",'
        '"description":"简短描述","confidence":0.86}。'
        '如果完全不知道出处或这不是动漫，请返回 {"title":"UNKNOWN"}。'
    )
    payload = {
        "model": VISION_MODEL,
        "temperature": 0,
        #"response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{image_data}"
                        },
                    },
                ],
            },
        ],
    }
    
    # 智能处理 URL，防止出现 /v1/v1/ 的情况
    base_url = VISION_BASE_URL.rstrip("/")
    endpoint = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
    try:
        response = await client.post(
            endpoint, # 使用处理后的 endpoint
            headers={
                "Authorization": f"Bearer {VISION_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=payload,
            timeout=30.0  # 建议给大模型加上较长的超时时间
        )
        response.raise_for_status()
        response_payload = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.warning("Vision LLM request failed: %s", exc)
        raise UpstreamSearchError("Vision LLM request failed") from exc

    try:
        content = response_payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise UpstreamSearchError("Vision LLM returned an invalid response") from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise UpstreamSearchError("Vision LLM returned an invalid JSON message")

    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    try:
        result_data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise UpstreamSearchError("Vision LLM returned malformed JSON") from exc

    if not isinstance(result_data, dict):
        raise UpstreamSearchError("Vision LLM returned an invalid JSON object")
    title = _normalise_name(result_data.get("title"))
    if not title or title.upper() == "UNKNOWN":
        raise UpstreamSearchError("Vision LLM could not identify the anime")

    raw_confidence = result_data.get("confidence", 0.82)
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.82
    if not math.isfinite(confidence):
        confidence = 0.82
    confidence = max(0.0, min(confidence, 1.0))
    if confidence < 0.60 or confidence > 0.98:
        confidence = 0.82

    cover_url = await _fetch_anilist_cover(client, title)
    return SearchResult(
        engine="ai_vision",
        title=title,
        similarity=confidence,
        video_url=None,
        image_url=cover_url,
        thumbnail_url=cover_url,
        character=_normalise_name(result_data.get("character")),
        description=_normalise_name(result_data.get("description")),
    )


def _normalise_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_similarity(value: Any) -> float | None:
    """Convert Trace.moe decimals and SauceNAO percentage strings to 0..1."""
    if value is None:
        return None
    try:
        score = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if score > 1:
        score /= 100
    return max(0.0, min(score, 1.0))


def _trace_title(item: dict[str, Any]) -> str:
    """Extract a title from the variants used by Trace.moe and AniList."""
    candidates: list[Any] = []
    anime = item.get("anime")
    if isinstance(anime, dict):
        candidates.extend([anime.get("title"), anime])
    candidates.extend([item.get("title"), item.get("source")])

    for candidate in candidates:
        if isinstance(candidate, dict):
            nested_title = candidate.get("title")
            if isinstance(nested_title, dict):
                for key in ("chinese", "userPreferred", "romaji", "native", "english"):
                    value = _normalise_name(nested_title.get(key))
                    if value:
                        return value
            for key in ("chinese", "userPreferred", "romaji", "native", "english"):
                value = _normalise_name(candidate.get(key))
                if value:
                    return value
        else:
            value = _normalise_name(candidate)
            if value:
                return value
    return "未知动漫"


async def _request_anilist_titles(
    client: httpx.AsyncClient,
    anilist_ids: set[int],
) -> dict[int, str]:
    """Resolve missing Trace.moe titles in one AniList GraphQL request."""
    missing_ids = [anime_id for anime_id in anilist_ids if anime_id not in ANILIST_TITLE_CACHE]
    if not missing_ids:
        return {
            anime_id: ANILIST_TITLE_CACHE[anime_id]
            for anime_id in anilist_ids
            if anime_id in ANILIST_TITLE_CACHE
        }

    query = """
    query ($ids: [Int]) {
      Page(perPage: 50) {
        media(id_in: $ids, type: ANIME) {
          id
          title { romaji native english }
        }
      }
    }
    """
    try:
        response = await client.post(
            ANILIST_GRAPHQL_URL,
            json={"query": query, "variables": {"ids": missing_ids}},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.warning("AniList title lookup failed: %s", exc)
        return {
            anime_id: ANILIST_TITLE_CACHE[anime_id]
            for anime_id in anilist_ids
            if anime_id in ANILIST_TITLE_CACHE
        }

    media_items = payload.get("data", {}).get("Page", {}).get("media", [])
    for media in media_items:
        if not isinstance(media, dict):
            continue
        try:
            anime_id = int(media.get("id"))
        except (TypeError, ValueError):
            continue
        title = _trace_title({"title": media.get("title")})
        if title != "未知动漫":
            ANILIST_TITLE_CACHE[anime_id] = title

    return {
        anime_id: ANILIST_TITLE_CACHE[anime_id]
        for anime_id in anilist_ids
        if anime_id in ANILIST_TITLE_CACHE
    }


def _clean_trace_result(item: dict[str, Any], title: str | None = None) -> SearchResult:
    return SearchResult(
        engine="tracemoe",
        title=title or _trace_title(item),
        similarity=_normalise_similarity(item.get("similarity")) or 0.0,
        video_url=_normalise_name(item.get("video")),
        image_url=_normalise_name(item.get("image")),
        thumbnail_url=_normalise_name(item.get("thumbnail")),
        episode=item.get("episode"),
        **{"from": item.get("from"), "to": item.get("to")},
    )


async def _clean_trace_results(
    client: httpx.AsyncClient,
    raw_results: list[Any],
) -> list[SearchResult]:
    valid_items = [item for item in raw_results if isinstance(item, dict)]
    missing_ids: set[int] = set()
    for item in valid_items:
        if _trace_title(item) != "未知动漫":
            continue
        try:
            missing_ids.add(int(item["anilist"]))
        except (KeyError, TypeError, ValueError):
            continue

    title_map = await _request_anilist_titles(client, missing_ids)
    results: list[SearchResult] = []
    for item in valid_items:
        title = _trace_title(item)
        if title == "未知动漫":
            try:
                title = title_map.get(int(item.get("anilist")), title)
            except (TypeError, ValueError):
                pass
        results.append(_clean_trace_result(item, title=title))
    return results


def _clean_saucenao_result(item: dict[str, Any]) -> SearchResult | None:
    header = item.get("header")
    data = item.get("data")
    if not isinstance(header, dict) or not isinstance(data, dict):
        return None

    similarity = _normalise_similarity(header.get("similarity"))
    if similarity is None:
        return None
    title = (
        _normalise_name(data.get("source"))
        or _normalise_name(data.get("title"))
        or _normalise_name(data.get("eng_name"))
        or "未知动漫"
    )
    image_url = next(
        (
            _normalise_name(data.get(key))
            for key in ("source_url", "url", "image", "img_url")
            if _normalise_name(data.get(key))
        ),
        None,
    )
    ext_urls = data.get("ext_urls")
    if image_url is None and isinstance(ext_urls, list):
        image_url = next((_normalise_name(url) for url in ext_urls if _normalise_name(url)), None)
    thumbnail_url = next(
        (
            _normalise_name(header.get(key)) or _normalise_name(data.get(key))
            for key in ("thumbnail", "thumbnail_url", "sample", "sample_url")
            if _normalise_name(header.get(key)) or _normalise_name(data.get(key))
        ),
        None,
    )
    return SearchResult(
        engine="saucenao",
        title=title,
        similarity=similarity,
        video_url=None,
        image_url=image_url,
        thumbnail_url=thumbnail_url,
    )


async def _request_tracemoe(
    client: httpx.AsyncClient,
    filename: str,
    contents: bytes,
    content_type: str,
) -> tuple[list[SearchResult], int | None]:
    try:
        response = await client.post(
            TRACE_MOE_URL,
            files={"file": (filename, contents, content_type)},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.warning("Trace.moe request failed: %s", exc)
        raise UpstreamSearchError("Trace.moe request failed") from exc

    results = await _clean_trace_results(client, payload.get("result", []))
    return results, payload.get("frameCount")


async def _request_saucenao(
    client: httpx.AsyncClient,
    filename: str,
    contents: bytes,
    content_type: str,
) -> list[SearchResult]:
    if not SAUCENAO_API_KEY:
        raise UpstreamSearchError(
            "SauceNAO is not configured; set the SAUCENAO_API_KEY environment variable"
        )

    try:
        response = await client.post(
            SAUCENAO_URL,
            params={"db": 999, "output_type": 2, "api_key": SAUCENAO_API_KEY},
            files={"file": (filename, contents, content_type)},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.TimeoutException, httpx.HTTPError, ValueError) as exc:
        logger.warning("SauceNAO request failed: %s", exc)
        raise UpstreamSearchError("SauceNAO request failed") from exc

    return [
        result
        for item in payload.get("results", [])
        if isinstance(item, dict)
        for result in [_clean_saucenao_result(item)]
        if result is not None
    ]
