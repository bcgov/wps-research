'''download data from data mart
'''
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote

BASE_URL = "https://www.for.gov.bc.ca/ftp/HPR/external/!publish/BCWS_DATA_MART/"
LOCAL_ROOT = "BCWS_DATA_MART"

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

visited = set()

def local_path_from_url(url):
    base_path = urlparse(BASE_URL).path
    rel_path = urlparse(url).path[len(base_path):]
    rel_path = unquote(rel_path).lstrip("/")
    return os.path.join(LOCAL_ROOT, rel_path)

def crawl(url):
    if url in visited:
        return
    visited.add(url)

    print(f"Entering: {url}")

    r = session.get(url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Skip navigation links
        if href in ("../", "./"):
            continue

        full_url = urljoin(url, href)

        # Stay inside the dataset root
        if not full_url.startswith(BASE_URL):
            continue

        if full_url.endswith("/"):
            crawl(full_url)
        else:
            local_path = local_path_from_url(full_url)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            if os.path.exists(local_path):
                continue

            print(f"Downloading: {full_url}")
            with session.get(full_url, stream=True) as resp:
                resp.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

def main():
    os.makedirs(LOCAL_ROOT, exist_ok=True)
    crawl(BASE_URL)
    print("Done.")

if __name__ == "__main__":
    main()



