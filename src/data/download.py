"""Dataset download utilities."""
import argparse
import re
import socket
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from tqdm import tqdm


RAW_DIR = Path("data/raw")
NLM_CXR_BASE_URL = (
    "https://data.lhncbc.nlm.nih.gov/public/"
    "Tuberculosis-Chest-X-ray-Datasets/"
)


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def _is_valid_zip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def _remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except (OSError, urllib.error.URLError):
        return None


def _download_file(
    url: str,
    dest: Path,
    retries: int = 10,
    validate_zip: bool = False,
    show_progress: bool = True,
):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if validate_zip and _is_valid_zip(dest):
        print(f"[skip] {dest} is already downloaded")
        return
    if not validate_zip and dest.exists():
        expected_size = _remote_size(url)
        if expected_size is not None and dest.stat().st_size == expected_size:
            return

    downloaded = dest.stat().st_size if dest.exists() else 0
    if show_progress:
        print(f"Downloading {url} -> {dest}")
    if downloaded and show_progress:
        print(f"Resuming from {downloaded / (1024 ** 2):.1f} MiB")

    attempt = 0
    progress = None
    while True:
        request = urllib.request.Request(url)
        if downloaded:
            request.add_header("Range", f"bytes={downloaded}-")

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status = getattr(response, "status", response.getcode())
                if downloaded and status != 206:
                    print("Server did not accept resume; restarting the file.")
                    downloaded = 0
                    dest.unlink(missing_ok=True)

                content_range = response.headers.get("Content-Range", "")
                match = re.search(r"/(\d+)$", content_range)
                total = int(match.group(1)) if match else downloaded + int(
                    response.headers.get("Content-Length", 0)
                )

                if progress is None and show_progress:
                    progress = tqdm(
                        total=total or None,
                        initial=downloaded,
                        unit="B",
                        unit_scale=True,
                    )

                mode = "ab" if downloaded else "wb"
                with open(dest, mode) as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress.update(len(chunk))

            if progress:
                progress.close()
            if validate_zip and not _is_valid_zip(dest):
                raise zipfile.BadZipFile("downloaded file is not a valid ZIP")
            return
        except (
            ConnectionError,
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
            zipfile.BadZipFile,
        ) as exc:
            attempt += 1
            if attempt > retries:
                if progress:
                    progress.close()
                raise RuntimeError(
                    f"Download failed after {retries} retries. "
                    f"Run the command again to resume from {downloaded} bytes."
                ) from exc
            wait_seconds = min(30, 2 ** min(attempt, 5))
            print(
                f"\nConnection interrupted ({exc}). "
                f"Retrying in {wait_seconds}s [{attempt}/{retries}]..."
            )
            time.sleep(wait_seconds)


def _download(url: str, dest: Path, retries: int = 10):
    _download_file(url, dest, retries=retries, validate_zip=True)


def _extract_zip(zip_path: Path, extract_to: Path):
    print(f"Extracting {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def download_isic2018(base_dir: Path = RAW_DIR / "isic2018"):
    """Download ISIC 2018 Task 1 dataset (images + masks)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    urls = {
        "images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    }
    for name, url in urls.items():
        zip_path = base_dir / f"{name}.zip"
        _download(url, zip_path)
        _extract_zip(zip_path, base_dir)
    print("ISIC 2018 download complete!")


def download_chest_xray(
    base_dir: Path = RAW_DIR / "chest_xray",
    datasets=("montgomery", "shenzhen"),
):
    """Download the official Montgomery and Shenzhen CXR datasets from NLM."""
    base_dir.mkdir(parents=True, exist_ok=True)
    roots = {
        "montgomery": urljoin(
            NLM_CXR_BASE_URL,
            "Montgomery-County-CXR-Set/MontgomerySet/",
        ),
        "shenzhen": urljoin(
            NLM_CXR_BASE_URL,
            "Shenzhen-Hospital-CXR-Set/",
        ),
    }

    jobs = []
    for dataset_name in datasets:
        if dataset_name not in roots:
            raise ValueError(f"Unknown Chest X-ray dataset: {dataset_name}")
        root_url = roots[dataset_name]
        jobs.extend(_list_remote_files(root_url, base_dir / dataset_name))

    print(f"Found {len(jobs)} Chest X-ray files.")
    failures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _download_file,
                url,
                destination,
                10,
                False,
                False,
            ): (url, destination)
            for url, destination in jobs
        }
        with tqdm(total=len(futures), unit="file", desc="Chest X-ray") as progress:
            for future in as_completed(futures):
                url, destination = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append((url, destination, exc))
                progress.update(1)

    if failures:
        details = "\n".join(
            f"- {url} -> {destination}: {exc}"
            for url, destination, exc in failures[:10]
        )
        raise RuntimeError(
            f"Failed to download {len(failures)} files. "
            f"Run the command again to resume.\n{details}"
        )
    print(f"Chest X-ray download complete: {base_dir.resolve()}")


def _list_remote_files(root_url: str, local_root: Path):
    jobs = []
    pending = [(root_url, local_root)]
    visited = set()

    while pending:
        directory_url, directory_path = pending.pop()
        if directory_url in visited:
            continue
        visited.add(directory_url)

        index_url = urljoin(directory_url, "index.html")
        print(f"Reading {index_url}")
        with urllib.request.urlopen(index_url, timeout=60) as response:
            html = response.read().decode("utf-8", errors="replace")

        parser = _LinkParser()
        parser.feed(html)
        for href in parser.links:
            if href.startswith(("?", "#")) or href in ("../", "/"):
                continue

            item_url = urljoin(directory_url, href)
            if not item_url.startswith(root_url):
                continue

            if href.endswith("/index.html"):
                child_url = item_url[: -len("index.html")]
                child_name = unquote(
                    Path(urlparse(child_url.rstrip("/")).path).name
                )
                pending.append((child_url, directory_path / child_name))
                continue

            item_name = unquote(Path(urlparse(item_url).path).name)
            if not item_name or item_name == "index.html":
                continue

            if href.endswith("/"):
                pending.append((item_url, directory_path / item_name))
            elif not item_name.endswith(".json"):
                jobs.append((item_url, directory_path / item_name))

    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MedSeg datasets")
    parser.add_argument(
        "--dataset",
        choices=("all", "isic2018", "chest_xray", "montgomery", "shenzhen"),
        default="all",
    )
    args = parser.parse_args()

    if args.dataset in ("all", "isic2018"):
        download_isic2018()
    if args.dataset in ("all", "chest_xray"):
        download_chest_xray()
    elif args.dataset in ("montgomery", "shenzhen"):
        download_chest_xray(datasets=(args.dataset,))
