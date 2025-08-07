#!/usr/bin/env python3
"""
Ollama Mirror Server

A Python-based mirror server for Ollama model pulling that provides:
- Local caching of models and blobs
- Proxy functionality to upstream Ollama registry
- Compatible API endpoints for seamless integration
- Bandwidth optimization for multiple users
- Offline capability after initial caching

Usage:
    python ollama_mirror.py

Environment Variables:
    OLLAMA_MIRROR_HOST: Host to bind the server (default: 0.0.0.0)
    OLLAMA_MIRROR_PORT: Port to bind the server (default: 11434)
    OLLAMA_CACHE_DIR: Directory to store cached models (default: ./cache)
    OLLAMA_UPSTREAM_URL: Upstream Ollama registry URL (default: https://registry.ollama.ai)
    OLLAMA_LOG_LEVEL: Logging level (default: INFO)
"""

import os
import json
import hashlib
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException, Response, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
from config import load_config, setup_logging, get_config, OllamaMirrorConfig
from monitoring import get_metrics_collector, get_health_checker, periodic_metrics_collection

logger = logging.getLogger(__name__)

# Pydantic models for API


class PullRequest(BaseModel):
    model: str = Field(..., description="Model name to pull")
    insecure: Optional[bool] = Field(
        False, description="Allow insecure connections")
    stream: Optional[bool] = Field(True, description="Stream response")


class PullResponse(BaseModel):
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = True


class BlobInfo(BaseModel):
    digest: str
    size: int
    path: Path
    cached_at: datetime


class ModelCache:
    """Manages local model caching and storage"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.manifests_dir = cache_dir / "manifests" / "registry.ollama.ai" / "library"
        self.blobs_dir = cache_dir / "blobs"
        self.metadata_file = cache_dir / "metadata.json"

        # Create directories
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.blobs_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        self.metadata = self._load_metadata()

        logger.info(f"Cache initialized at {cache_dir}")

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {"models": {}, "blobs": {}}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_manifest_path(self, model_name: str, tag: str = "latest") -> Path:
        """Get path to model manifest"""
        return self.manifests_dir / model_name / tag

    def get_blob_path(self, digest: str) -> Path:
        """Get path to blob file"""
        # Convert sha256:abc123... to sha256-abc123...
        blob_name = digest.replace(":", "-")
        return self.blobs_dir / blob_name

    def has_model(self, model_name: str, tag: str = "latest") -> bool:
        """Check if model is cached"""
        manifest_path = self.get_manifest_path(model_name, tag)
        return manifest_path.exists()

    def has_blob(self, digest: str) -> bool:
        """Check if blob is cached"""
        blob_path = self.get_blob_path(digest)
        return blob_path.exists()

    def cache_manifest(self, model_name: str, tag: str, manifest_data: Dict):
        """Cache model manifest"""
        manifest_path = self.get_manifest_path(model_name, tag)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)

        # Update metadata
        model_key = f"{model_name}:{tag}"
        self.metadata["models"][model_key] = {
            "cached_at": datetime.now().isoformat(),
            "size": len(json.dumps(manifest_data)),
            "path": str(manifest_path)
        }
        self._save_metadata()

        logger.info(f"Cached manifest for {model_key}")

    def cache_blob(self, digest: str, data: bytes):
        """Cache blob data"""
        blob_path = self.get_blob_path(digest)

        with open(blob_path, 'wb') as f:
            f.write(data)

        # Update metadata
        self.metadata["blobs"][digest] = {
            "cached_at": datetime.now().isoformat(),
            "size": len(data),
            "path": str(blob_path)
        }
        self._save_metadata()

        logger.info(f"Cached blob {digest} ({len(data)} bytes)")

    def get_cached_models(self) -> List[Dict]:
        """Get list of cached models"""
        models = []
        for model_key, info in self.metadata["models"].items():
            models.append({
                "name": model_key,
                "cached_at": info["cached_at"],
                "size": info["size"]
            })
        return models

    def cleanup_old_cache(self, retention_days: int = 30):
        """Remove old cached files"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0

        # Clean up old models
        for model_key, info in list(self.metadata["models"].items()):
            cached_at = datetime.fromisoformat(info["cached_at"])
            if cached_at < cutoff_date:
                try:
                    Path(info["path"]).unlink(missing_ok=True)
                    del self.metadata["models"][model_key]
                    removed_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to remove old model {model_key}: {e}")

        # Clean up old blobs
        for digest, info in list(self.metadata["blobs"].items()):
            cached_at = datetime.fromisoformat(info["cached_at"])
            if cached_at < cutoff_date:
                try:
                    Path(info["path"]).unlink(missing_ok=True)
                    del self.metadata["blobs"][digest]
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove old blob {digest}: {e}")

        if removed_count > 0:
            self._save_metadata()
            logger.info(f"Cleaned up {removed_count} old cache entries")


class UpstreamClient:
    """Handles communication with upstream Ollama registry"""

    def __init__(self, config: OllamaMirrorConfig):
        self.upstream_url = config.upstream_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=config.upstream_timeout)
        self.download_semaphore = asyncio.Semaphore(
            config.max_concurrent_downloads)
        self.config = config

    async def get_manifest(self, model_name: str, tag: str = "latest") -> Dict:
        """Fetch model manifest from upstream"""
        url = f"{self.upstream_url}/v2/library/{model_name}/manifests/{tag}"

        async with self.download_semaphore:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.json()

    async def get_blob(self, model_name: str, digest: str) -> bytes:
        """Fetch blob data from upstream"""
        url = f"{self.upstream_url}/v2/library/{model_name}/blobs/{digest}"

        async with self.download_semaphore:
            response = await self.client.get(url)
            response.raise_for_status()
            return response.content

    async def stream_blob(self, model_name: str, digest: str):
        """Stream blob data from upstream"""
        url = f"{self.upstream_url}/v2/library/{model_name}/blobs/{digest}"

        async with self.download_semaphore:
            async with self.client.stream('GET', url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

    async def close(self):
        """Close the client"""
        await self.client.aclose()


# Global instances (will be initialized after config is loaded)
cache: Optional[ModelCache] = None
upstream: Optional[UpstreamClient] = None
app_config: Optional[OllamaMirrorConfig] = None

# FastAPI app
app = FastAPI(
    title="Ollama Mirror Server",
    description="A mirror server for Ollama model pulling with local caching",
    version="1.0.0"
)


def get_cache() -> ModelCache:
    """Get the global cache instance"""
    global cache
    if cache is None:
        config = get_config()
        cache = ModelCache(config.cache_dir)
    return cache


def get_upstream() -> UpstreamClient:
    """Get the global upstream client instance"""
    global upstream
    if upstream is None:
        config = get_config()
        upstream = UpstreamClient(config)
    return upstream


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the server"""
    config = get_config()

    logger.info("Starting Ollama Mirror Server")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Upstream URL: {config.upstream_url}")

    # Initialize global instances
    get_cache()
    get_upstream()

    # Start background tasks
    asyncio.create_task(periodic_cache_cleanup())
    if config.enable_metrics:
        asyncio.create_task(periodic_metrics_collection())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Ollama Mirror Server")
    upstream_client = get_upstream()
    await upstream_client.close()


async def periodic_cache_cleanup():
    """Periodic cache cleanup task"""
    config = get_config()
    cache_instance = get_cache()

    while True:
        try:
            await asyncio.sleep(config.cache_cleanup_interval)
            cache_instance.cleanup_old_cache(config.cache_retention_days)
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

# API Endpoints


@app.get("/")
async def root():
    """Health check endpoint"""
    config = get_config()

    return {
        "status": "healthy",
        "service": "ollama-mirror",
        "version": "1.0.0",
        "cache_dir": str(config.cache_dir),
        "upstream": config.upstream_url
    }


@app.get("/api/tags")
async def list_local_models():
    """List locally cached models (compatible with Ollama API)"""
    cache_instance = get_cache()
    models = cache_instance.get_cached_models()

    return {
        "models": [
            {
                "name": model["name"],
                "size": model["size"],
                "digest": "",  # Would need to calculate from manifest
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "unknown",
                    "quantization_level": "unknown"
                },
                "modified_at": model["cached_at"]
            }
            for model in models
        ]
    }


@app.post("/api/pull")
async def pull_model(request: PullRequest, background_tasks: BackgroundTasks):
    """Pull a model (main mirror functionality)"""
    cache_instance = get_cache()
    metrics = get_metrics_collector()
    config = get_config()

    model_name, tag = (request.model.split(':', 1) + ["latest"])[:2]

    logger.info(f"Pull request for {model_name}:{tag}")

    # Check security restrictions
    if config.get_allowed_models() and request.model not in config.get_allowed_models():
        raise HTTPException(
            status_code=403, detail=f"Model {request.model} not allowed")

    if config.get_blocked_models() and request.model in config.get_blocked_models():
        raise HTTPException(
            status_code=403, detail=f"Model {request.model} is blocked")

    # Check if model is already cached
    if cache_instance.has_model(model_name, tag):
        logger.info(f"Model {model_name}:{tag} found in cache")
        metrics.record_cache_hit(request.model)

        if not request.stream:
            return {"status": "success"}

        async def cached_response():
            yield json.dumps({"status": "already cached"}) + "\n"
            yield json.dumps({"status": "success"}) + "\n"

        return StreamingResponse(cached_response(), media_type="application/x-ndjson")

    # Record cache miss
    metrics.record_cache_miss(request.model)

    # Stream the pull process
    if request.stream:
        return StreamingResponse(
            stream_pull_model(model_name, tag),
            media_type="application/x-ndjson"
        )
    else:
        # Non-streaming pull
        await download_model(model_name, tag)
        return {"status": "success"}


async def stream_pull_model(model_name: str, tag: str):
    """Stream model pulling progress"""
    cache_instance = get_cache()
    upstream_client = get_upstream()
    metrics = get_metrics_collector()

    try:
        yield json.dumps({"status": "resolving manifest"}) + "\n"

        # Get manifest from upstream
        manifest = await upstream_client.get_manifest(model_name, tag)

        yield json.dumps({"status": "downloading manifest"}) + "\n"

        # Cache manifest
        cache_instance.cache_manifest(model_name, tag, manifest)

        # Extract blob digests
        blobs_to_download = []

        # Config blob
        if "config" in manifest and "digest" in manifest["config"]:
            blobs_to_download.append(manifest["config"]["digest"])

        # Layer blobs
        if "layers" in manifest:
            for layer in manifest["layers"]:
                if "digest" in layer:
                    blobs_to_download.append(layer["digest"])

        total_blobs = len(blobs_to_download)
        completed_blobs = 0

        # Download missing blobs
        for digest in blobs_to_download:
            if cache_instance.has_blob(digest):
                completed_blobs += 1
                yield json.dumps({
                    "status": "pulling blob",
                    "digest": digest,
                    "total": total_blobs,
                    "completed": completed_blobs
                }) + "\n"
                continue

            yield json.dumps({
                "status": "downloading blob",
                "digest": digest,
                "total": total_blobs,
                "completed": completed_blobs
            }) + "\n"

            # Download blob
            blob_data = await upstream_client.get_blob(model_name, digest)
            cache_instance.cache_blob(digest, blob_data)

            completed_blobs += 1
            yield json.dumps({
                "status": "pulled blob",
                "digest": digest,
                "total": total_blobs,
                "completed": completed_blobs
            }) + "\n"

        yield json.dumps({"status": "verifying sha256 digest"}) + "\n"
        yield json.dumps({"status": "writing manifest"}) + "\n"
        yield json.dumps({"status": "removing any unused layers"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"

        # Record successful caching
        metrics.record_model_cached(f"{model_name}:{tag}")

        logger.info(f"Successfully pulled and cached {model_name}:{tag}")

    except Exception as e:
        logger.error(f"Failed to pull {model_name}:{tag}: {e}")
        yield json.dumps({"status": "error", "error": str(e)}) + "\n"


async def download_model(model_name: str, tag: str):
    """Download model without streaming"""
    cache_instance = get_cache()
    upstream_client = get_upstream()

    manifest = await upstream_client.get_manifest(model_name, tag)
    cache_instance.cache_manifest(model_name, tag, manifest)

    # Download blobs
    blobs_to_download = []

    if "config" in manifest and "digest" in manifest["config"]:
        blobs_to_download.append(manifest["config"]["digest"])

    if "layers" in manifest:
        for layer in manifest["layers"]:
            if "digest" in layer:
                blobs_to_download.append(layer["digest"])

    for digest in blobs_to_download:
        if not cache_instance.has_blob(digest):
            blob_data = await upstream_client.get_blob(model_name, digest)
            cache_instance.cache_blob(digest, blob_data)


@app.get("/v2/library/{model_name}/manifests/{tag}")
async def get_manifest(model_name: str, tag: str):
    """Get model manifest (registry API compatibility)"""
    cache_instance = get_cache()
    upstream_client = get_upstream()

    if cache_instance.has_model(model_name, tag):
        manifest_path = cache_instance.get_manifest_path(model_name, tag)
        with open(manifest_path, 'r') as f:
            return json.load(f)

    # Fetch from upstream and cache
    try:
        manifest = await upstream_client.get_manifest(model_name, tag)
        cache_instance.cache_manifest(model_name, tag, manifest)
        return manifest
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))


@app.get("/v2/library/{model_name}/blobs/{digest}")
async def get_blob(model_name: str, digest: str):
    """Get blob data (registry API compatibility)"""
    cache_instance = get_cache()
    upstream_client = get_upstream()

    if cache_instance.has_blob(digest):
        blob_path = cache_instance.get_blob_path(digest)
        return FileResponse(blob_path)

    # Stream from upstream and cache
    try:
        blob_data = await upstream_client.get_blob(model_name, digest)
        cache_instance.cache_blob(digest, blob_data)

        # Return the cached file
        blob_path = cache_instance.get_blob_path(digest)
        return FileResponse(blob_path)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """Proxy generate requests to local Ollama instance"""
    config = get_config()

    if not config.enable_api_proxy:
        raise HTTPException(status_code=404, detail="API proxy is disabled")

    # This endpoint proxies to a local Ollama instance if available
    # For a pure mirror, you might want to remove this or make it optional
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=request.dict(),
                timeout=config.upstream_timeout
            )
            return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Local Ollama instance not available: {e}")


@app.get("/api/version")
async def version():
    """Version endpoint (Ollama compatibility)"""
    return {"version": "0.1.0-mirror"}


@app.get("/mirror/stats")
async def mirror_stats():
    """Mirror-specific statistics endpoint"""
    cache_instance = get_cache()
    config = get_config()
    metrics = get_metrics_collector()

    models = cache_instance.get_cached_models()
    total_size = sum(model["size"] for model in models)

    return {
        "cached_models": len(models),
        "total_cache_size": total_size,
        "cache_directory": str(config.cache_dir),
        "upstream_url": config.upstream_url,
        "models": models,
        "metrics": metrics.export_metrics() if config.enable_metrics else None
    }


@app.get("/mirror/health")
async def health_check():
    """Detailed health check endpoint"""
    config = get_config()
    health_checker = get_health_checker(config.cache_dir, config.upstream_url)

    return await health_checker.check_health()


@app.get("/mirror/metrics")
async def metrics_endpoint():
    """Metrics endpoint for monitoring systems"""
    config = get_config()

    if not config.enable_metrics:
        raise HTTPException(
            status_code=404, detail="Metrics collection is disabled")

    metrics = get_metrics_collector()
    return metrics.export_metrics()


@app.delete("/mirror/cache")
async def clear_cache():
    """Clear all cached data"""
    config = get_config()

    try:
        import shutil
        shutil.rmtree(config.cache_dir)

        # Recreate cache
        global cache
        cache = ModelCache(config.cache_dir)

        return {"status": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear cache: {e}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description="Ollama Mirror Server")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--host", help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to bind to")

    args = parser.parse_args()

    # Load configuration
    global config
    from config import config as global_config
    global_config = load_config(args.config)

    # Override with command line arguments
    if args.host:
        global_config.host = args.host
    if args.port:
        global_config.port = args.port

    # Setup logging
    setup_logging(global_config)

    logger.info("Starting Ollama Mirror Server")
    logger.info(
        f"Configuration loaded from: {args.config or 'environment variables'}")

    uvicorn.run(
        "ollama_mirror:app",
        host=global_config.host,
        port=global_config.port,
        log_level=global_config.log_level.lower(),
        reload=global_config.reload
    )


if __name__ == "__main__":
    main()
