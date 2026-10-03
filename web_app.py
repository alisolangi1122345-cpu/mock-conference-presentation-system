import os
import sys
import threading
import uuid
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # the analysis modules use relative paths like "output"
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

import pipeline  # noqa: E402
from languages import available_languages, current_language_code  # noqa: E402

# ==========================================
# AI-Based Mock Conference Presentation System
# Web app. Start with:  python web_app.py
# ==========================================

HOST = "127.0.0.1"
PORT = 8000
MAX_UPLOAD_MB = 200
INDEX_FILE = os.path.join(BASE_DIR, "static", "index.html")

app = FastAPI(title="Mock Conference Presentation System")


@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.get("/api/config")
def config():
    return {
        "languages": [{"code": c, "name": n} for c, n in available_languages()],
        "default_language": current_language_code(),
        "default_target_minutes": 3,
    }


@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    target_minutes: float = Form(3.0),
):
    if language not in {c for c, _ in available_languages()}:
        raise HTTPException(400, f"Language '{language}' is not enabled.")
    if not 0.5 <= target_minutes <= 60:
        raise HTTPException(400, "Target length must be between 0.5 and 60 minutes.")

    ext = os.path.splitext(audio.filename or "")[1].lower()
    if not ext or len(ext) > 6:
        ext = ".webm"

    folder = os.path.join(pipeline.DATA_DIR, uuid.uuid4().hex[:10])
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "audio" + ext)

    size = 0
    too_big = False
    with open(path, "wb") as out:
        while chunk := await audio.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                too_big = True
                break
            out.write(chunk)

    if too_big:
        os.remove(path)
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    if size < 1000:
        raise HTTPException(400, "The recording is empty.")

    job_id = pipeline.start_job(path, language, target_minutes)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = pipeline.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return {
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "session_id": job["session_id"],
        "error": job["error"],
    }


@app.get("/api/sessions")
def sessions():
    return pipeline.list_sessions()


@app.get("/api/sessions/{sid}")
def session(sid: str):
    result = pipeline.get_session(sid)
    if result is None:
        raise HTTPException(404, "Session not found.")
    return result


@app.delete("/api/sessions/{sid}")
def remove_session(sid: str):
    if not pipeline.delete_session(sid):
        raise HTTPException(404, "Session not found.")
    return {"deleted": sid}


if __name__ == "__main__":
    import uvicorn

    url = f"http://{HOST}:{PORT}"
    print(f"\nMock conference app running at {url}")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
