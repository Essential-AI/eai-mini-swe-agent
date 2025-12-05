import json
from pathlib import Path

from google.cloud import storage

from minisweagent import global_config_dir
from minisweagent.utils.log import logger

def _parse_gcs_url(url: str) -> tuple[str, str]:
    if not url.startswith("gs://"):
        raise ValueError(f"Unsupported URL (expected gs://): {url}")
    rest = url[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid GCS URL (missing object path): {url}")
    bucket, name = rest.split("/", 1)
    return bucket, name

def _cache_root(custom: Path | None) -> Path:
    return (custom or (global_config_dir / "gcs_cache")).resolve()

def _meta_path(p: Path) -> Path:
    return p.with_name(p.name + ".meta.json")

def ensure_local_gcs_file(url: str, *, cache_root: Path | None = None, force: bool = False) -> Path:
    """
    Ensure a GCS file (gs://bucket/object) exists locally in a cache and return its local Path.
    If the file is already cached and 'force' is False, it is returned as-is.
    """
    bucket_name, object_name = _parse_gcs_url(url)
    root = _cache_root(cache_root)
    local_path = (root / bucket_name / object_name).resolve()
    if local_path.exists() and not force:
        return local_path

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")

    logger.info(f"Downloading '{url}' to cache: '{local_path}'")
    blob.download_to_filename(tmp_path.as_posix())
    tmp_path.replace(local_path)

    blob.reload()  # populate metadata if available
    meta = {
        "url": url,
        "bucket": bucket_name,
        "name": object_name,
        "generation": str(blob.generation) if blob.generation is not None else None,
        "etag": blob.etag,
        "size": blob.size,
    }
    _meta_path(local_path).write_text(json.dumps(meta, indent=2))

    return local_path
