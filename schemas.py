from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """The stable result shape shared by Trace.moe and SauceNAO."""

    model_config = ConfigDict(populate_by_name=True)

    engine: Literal["tracemoe", "ai_vision", "saucenao"] = Field(
        description="The engine that produced this result."
    )
    title: str = Field(description="The best available anime or source title.")
    similarity: float = Field(
        ge=0,
        le=1,
        description="Normalized similarity score from 0.0 to 1.0.",
    )
    video_url: str | None = Field(
        default=None,
        description="Trace.moe preview video URL, or null when unavailable.",
    )
    image_url: str | None = Field(
        default=None,
        description="Direct result image URL when provided by the recognition engine.",
    )
    thumbnail_url: str | None = Field(
        default=None,
        description="Thumbnail URL used as a fallback when the direct image is unavailable.",
    )
    character: str | None = Field(
        default=None, description="Character name identified by the vision model."
    )
    description: str | None = Field(
        default=None, description="Short AI-generated description of the style or background."
    )
    episode: int | float | str | None = Field(
        default=None, description="Matched episode when the engine provides one."
    )
    from_: float | None = Field(
        default=None, alias="from", description="Matched scene start time in seconds."
    )
    to: float | None = Field(
        default=None, description="Matched scene end time in seconds."
    )


class SearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    results: list[SearchResult]
    frame_count: int | None = Field(default=None, alias="frameCount")
    engine: Literal["tracemoe", "ai_vision", "saucenao"] | None = Field(
        default=None,
        description="Engine used for the returned result set, if any.",
    )


class UpstreamSearchError(Exception):
    """Raised when an upstream engine cannot return a usable response."""
