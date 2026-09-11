"""Mood check-in and private image route definitions."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse

from api_contracts import MoodCheckinListResponse, MoodCheckinRequest, MoodCheckinResponse, MoodImage, WeeklyMoodResponse
from mood_image_store import MAX_IMAGE_BYTES, delete_mood_checkin_images, delete_mood_image, get_mood_image_path, list_mood_images, save_mood_image
from mood_store import add_mood_checkin, delete_mood_checkin, format_mood_fluctuation_analysis, format_weekly_mood_summary, get_weekly_mood_points, load_mood_checkins


def _record_with_images(user_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "images": list_mood_images(user_id, str(record["date"]))}


def create_mood_router(
    *,
    current_user: Callable[..., str],
    csrf_protected_user: Callable[..., str],
    read_upload_limited: Callable[[UploadFile, int], Awaitable[bytes]],
) -> APIRouter:
    """Create mood routes while sharing the application's bounded upload reader."""
    router = APIRouter()

    @router.post("/api/v1/mood/checkins", response_model=MoodCheckinResponse)
    def create_mood_checkin(
        request: MoodCheckinRequest, user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        try:
            record = add_mood_checkin(
                user_id=user_id,
                mood=request.mood,
                intensity=request.intensity,
                note=request.note,
                checkin_date=request.checkin_date,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        return {"record": _record_with_images(user_id, record)}

    @router.get("/api/v1/mood/checkins", response_model=MoodCheckinListResponse)
    def list_mood_records(user_id: str = Depends(current_user)) -> dict[str, Any]:
        return {"records": [_record_with_images(user_id, record) for record in load_mood_checkins(user_id)]}

    @router.post(
        "/api/v1/mood/checkins/{checkin_date}/images",
        response_model=MoodImage,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_mood_image(
        checkin_date: str,
        file: Annotated[UploadFile, File(...)],
        user_id: str = Depends(csrf_protected_user),
    ) -> dict[str, Any]:
        try:
            if not any(record["date"] == checkin_date for record in load_mood_checkins(user_id)):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood Check-in was not found.")
            return save_mood_image(user_id, checkin_date, await read_upload_limited(file, MAX_IMAGE_BYTES))
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        finally:
            await file.close()

    @router.get("/api/v1/mood/checkins/{checkin_date}/images/{image_id}")
    def get_mood_image(checkin_date: str, image_id: str, user_id: str = Depends(current_user)) -> FileResponse:
        path = get_mood_image_path(user_id, checkin_date, image_id)
        if path is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood Check-in image was not found.")
        image = next((item for item in list_mood_images(user_id, checkin_date) if item["id"] == image_id), None)
        if image is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood Check-in image was not found.")
        return FileResponse(path, media_type=image["content_type"], headers={"Cache-Control": "private, no-store"})

    @router.delete("/api/v1/mood/checkins/{checkin_date}/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_mood_image(
        checkin_date: str,
        image_id: str,
        user_id: str = Depends(csrf_protected_user),
    ) -> None:
        if not delete_mood_image(user_id, checkin_date, image_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood Check-in image was not found.")

    @router.delete("/api/v1/mood/checkins/{checkin_date}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_mood_checkin(checkin_date: str, user_id: str = Depends(csrf_protected_user)) -> None:
        try:
            deleted = delete_mood_checkin(user_id, checkin_date)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood check-in was not found.")
        delete_mood_checkin_images(user_id, checkin_date)

    @router.get("/api/v1/mood/weekly", response_model=WeeklyMoodResponse)
    def weekly_mood(
        user_id: str = Depends(current_user),
        end_date: str | None = None,
        days: int = Query(default=7, ge=1, le=31),
        locale: str = Query(default="zh"),
    ) -> dict[str, Any]:
        try:
            points = get_weekly_mood_points(user_id, end_date=end_date, days=days)
            return {
                "points": points,
                "summary": format_weekly_mood_summary(user_id, end_date=end_date, days=days, locale=locale),
                "analysis": format_mood_fluctuation_analysis(points, locale=locale),
            }
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    return router
