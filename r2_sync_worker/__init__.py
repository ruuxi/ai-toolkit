"""
R2 sync worker package.

The worker exposes a small FastAPI service that mirrors datasets from
Cloudflare R2 (S3-compatible) into the ai-toolkit datasets folder before
we submit jobs via /api/jobs.
"""

