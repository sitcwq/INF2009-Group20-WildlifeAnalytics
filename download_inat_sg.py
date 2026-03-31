import argparse
import csv
import re
import time
from pathlib import Path
from typing import List, Tuple

import requests

INAT_API = "https://api.inaturalist.org/v1/observations"
SG_PLACE_ID = 6734  # Singapore
MY_SG_PLACE_ID = 90097
MY_PLACE_ID = 7155

PLACE_PREFERENCE = [SG_PLACE_ID, MY_SG_PLACE_ID, MY_PLACE_ID]

# Your class list (taxon_id from iNaturalist taxon pages)
TAXA: List[Tuple[str, int]] = [
    ("macaque", 43459),        # Macaca fascicularis
    ("otter", 526556),         # Lutrinae
    ("wild_boar", 42134),      # Sus scrofa
    ("snake", 85553),          # Serpentes
    ("bat", 40268),            # Chiroptera
    ("palm_civet", 854000),    # Paradoxurus musanga (iNat)
    ("monitor_lizard", 39393), # Varanus
    ("squirrel", 45933),       # Sciuridae
    ("human", 43584),          # Homo sapiens
    ("common_flameback", 204504)  # Dinopium javanense (iNat)
]

def upgrade_photo_url(url: str, size: str) -> str:
    """
    iNat photo URLs often contain a size token like 'square', 'small', 'medium', 'large', 'original'.
    Convert to a chosen size for training.
    """
    # Common patterns: .../photos/<id>/square.jpg or .../square.jpeg
    return re.sub(r"/(square|small|medium|large|original)\.(jpg|jpeg|png)$",
                  rf"/{size}.\2", url)

def safe_get(session: requests.Session, url: str, params: dict, timeout: int, max_retries: int, sleep_s: float):
    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            # rate limited; back off
            time.sleep(max(2.0, sleep_s) * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"Failed after retries (status {r.status_code}): {r.text[:200]}")

def download_file(session: requests.Session, url: str, out_path: Path, timeout: int, max_retries: int, sleep_s: float) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True  # already downloaded

    for attempt in range(max_retries):
        r = session.get(url, stream=True, timeout=timeout)
        if r.status_code == 429:
            time.sleep(max(2.0, sleep_s) * (attempt + 1))
            continue
        if r.status_code >= 400:
            time.sleep(sleep_s)
            continue

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 0

    return False

def fetch_observations(session: requests.Session, taxon_id: int, place_id: int, per_page: int, page: int, quality_grade: str, timeout: int, max_retries: int, sleep_s: float):
    params = {
        "place_id": place_id,
        "taxon_id": taxon_id,
        "photos": "true",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
        # optional filters:
        #"quality_grade": quality_grade,  # "research" or "needs_id" or "casual"
        #"verifiable": "true",
    }
    r = safe_get(session, INAT_API, params, timeout, max_retries, sleep_s)
    return r.json()

def main():
    ap = argparse.ArgumentParser(description="Download Singapore wildlife images from iNaturalist into class folders.")
    ap.add_argument("--out", default="dataset", help="Output dataset folder")
    ap.add_argument("--per-class", type=int, default=300, help="Max images to download per class (label)")
    ap.add_argument("--per-page", type=int, default=200, help="API per_page (max 200 recommended)")
    ap.add_argument("--quality", default="research", choices=["research", "needs_id", "casual"], help="iNat quality_grade filter")
    ap.add_argument("--photo-size", default="large", choices=["square","small","medium","large","original"], help="Which photo size to download")
    ap.add_argument("--sleep", type=float, default=0.35, help="Sleep between API calls (seconds)")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=5, help="Retries for API/download")
    ap.add_argument(
    "--only",
    nargs="*",
    default=None,
    help="Optional list of labels to download (e.g., --only palm_civet otter). If omitted, download all."
)
    args = ap.parse_args()

    selected_taxa = TAXA
    if args.only:
        only_set = set(args.only)
        selected_taxa = [(lbl, tid) for (lbl, tid) in TAXA if lbl in only_set]
        if not selected_taxa:
            raise ValueError(f"--only labels not found in TAXA: {args.only}. Available: {[l for l,_ in TAXA]}")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    # Load existing manifest to avoid redownloading (track by obs_id + photo_id)
    seen_keys = set()  # (obs_id, photo_id) as strings
    seen_count_by_label = {}  # label -> count of unique keys already downloaded

    if manifest_path.exists():
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (str(row["obs_id"]), str(row["photo_id"]))
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                lbl = row.get("label")
                if lbl:
                    seen_count_by_label[lbl] = seen_count_by_label.get(lbl, 0) + 1

    session = requests.Session()
    session.headers.update({"User-Agent": "INF2009-edge-wildlife-downloader/1.0"})

    # Prepare manifest writer (append mode)
    new_file = not manifest_path.exists()
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(mf, fieldnames=["label","taxon_id","obs_id","photo_id","local_path","url"])
    if new_file:
        writer.writeheader()

    for label, taxon_id in selected_taxa:
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        already_have = seen_count_by_label.get(label, 0)
        downloaded_new = 0
        target_total = args.per_class
        page = 1

    print(f"\n=== {label} (taxon_id={taxon_id}) already_have={already_have} target={target_total} ===")

    for place_id in PLACE_PREFERENCE:
        page = 1
        print(f"  [place] trying place_id={place_id}")

        while (already_have + downloaded_new) < target_total:
            data = fetch_observations(session, taxon_id, place_id, args.per_page, page, args.quality,
                                    args.timeout, args.retries, args.sleep)

            results = data.get("results", [])
            if not results:
                print(f"  [place] exhausted place_id={place_id}")
                break

            for obs in results:
                if (already_have + downloaded_new) >= target_total:
                    break

                obs_id = obs.get("id")
                photos = obs.get("photos") or []
                if not photos:
                    continue

                p0 = photos[0]
                photo_id = p0.get("id")
                url = p0.get("url")
                if not (obs_id and photo_id and url):
                    continue

                key = (str(obs_id), str(photo_id))
                if key in seen_keys:
                    continue

                url2 = upgrade_photo_url(url, args.photo_size)

                ext = "jpg"
                m = re.search(r"\.(jpg|jpeg|png)$", url2, re.IGNORECASE)
                if m:
                    ext = m.group(1).lower().replace("jpeg", "jpg")

                out_path = label_dir / f"obs_{obs_id}_photo_{photo_id}.{ext}"
                local_path_str = str(out_path)

                ok = download_file(session, url2, out_path, args.timeout, args.retries, args.sleep)
                time.sleep(args.sleep)

                if ok:
                    writer.writerow({
                        "label": label,
                        "taxon_id": taxon_id,
                        "obs_id": obs_id,
                        "photo_id": photo_id,
                        "local_path": local_path_str,
                        "url": url2
                    })
                    mf.flush()
                    seen_keys.add(key)
                    downloaded_new += 1

                    total_now = already_have + downloaded_new
                    if downloaded_new % 25 == 0:
                        print(f"Downloaded new {downloaded_new} (total now {total_now}/{target_total})...")

            page += 1
            time.sleep(args.sleep)

        if (already_have + downloaded_new) >= target_total:
            break  # reached target, stop trying fallback places

    print(f"Done: {label} total={already_have + downloaded_new} (new this run={downloaded_new}).")

    mf.close()
    print(f"\nManifest written to: {manifest_path}")

if __name__ == "__main__":
    main()