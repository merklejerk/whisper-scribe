import aiohttp
import os
from typing import Optional, cast

async def create_gist(
    files: dict[str, str],
    description: str = "",
    public: bool = True,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Create a GitHub Gist from a dictionary of filenames and contents.

    Args:
        files: dict mapping filename to file content (as string)
        description: description for the gist
        public: whether the gist should be public (default False for privacy)
        token: GitHub token (if None, uses GITHUB_TOKEN env var)
    Returns:
        The URL of the created gist, or None if not available (should not happen if successful)
    Raises:
        RuntimeError: if the gist creation fails (non-201 response)
    """
    api_url = "https://api.github.com/gists"
    headers = {"Accept": "application/vnd.github+json"}
    if token is None:
        token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    data = {
        "description": description,
        "public": public,
        "files": {fn: {"content": content} for fn, content in files.items()}
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=data, headers=headers) as resp:
            if resp.status == 201:
                resp_data = await resp.json()
                return cast(Optional[str], resp_data.get("html_url"))
            else:
                raise RuntimeError(f"Failed to create gist: HTTP {resp.status} {await resp.text()}")

async def create_gist_from_paths(
    paths: list[str],
    description: str = "",
    public: bool = True,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Create a GitHub Gist from a list of file paths.

    Args:
        paths: list of file paths to upload
        description: description for the gist
        public: whether the gist should be public (default False for privacy)
        token: GitHub token (if None, uses GITHUB_TOKEN env var)
    Returns:
        The URL of the created gist, or None if not available (should not happen if successful)
    Raises:
        FileNotFoundError, OSError: if any file cannot be read
        ValueError: if no files are provided
        RuntimeError: if the gist creation fails
    """
    files: dict[str, str] = {}
    for path in paths:
        with open(path, "r") as f:
            files[os.path.basename(path)] = f.read()
    if not files:
        raise ValueError("No files to upload to gist.")
    return await create_gist(files, description=description, public=public, token=token)
