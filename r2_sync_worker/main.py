from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .sync import SyncError, dataset_status, sync_dataset


class SyncDatasetBody(BaseModel):
    dataset_id: str = Field(..., alias="datasetId")
    bucket: str
    prefix: str
    overwrite: bool = False

    class Config:
        populate_by_name = True


app = FastAPI(title="R2 Sync Worker", version="0.1.0")


@app.get("/healthz")
async def healthcheck():
    settings = get_settings()
    return {
        "status": "ok",
        "datasetRoot": settings.dataset_root,
    }


@app.get("/dataset-status/{dataset_id}")
async def get_dataset_status(dataset_id: str):
    local_path = dataset_status(dataset_id)
    if not local_path:
        raise HTTPException(status_code=404, detail="Dataset not synced yet")
    return {"ok": True, "datasetId": dataset_id, "localPath": local_path}


@app.post("/sync-dataset")
async def sync_dataset_endpoint(body: SyncDatasetBody):
    try:
        stats = sync_dataset(
            dataset_id=body.dataset_id,
            bucket=body.bucket,
            prefix=body.prefix,
            overwrite=body.overwrite,
        )
    except SyncError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"ok": True, **stats}


def run():
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "r2_sync_worker.main:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
    )


if __name__ == "__main__":
    run()

