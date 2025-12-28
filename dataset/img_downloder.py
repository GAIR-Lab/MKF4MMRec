import json
import os
import requests
import ast
from concurrent.futures import ThreadPoolExecutor

DATASET_NAMES = ["Baby", "Sports", "Clothing"]  # Dataset names
NUM_THREADS = 200  # Number of threads, can be adjusted according to network conditions

def download_image(save_dir, failed_log, asin, url):
    """Download a single image, skip if it already exists"""
    if not url:
        return

    ext = os.path.splitext(url)[-1] or ".jpg"  # Get extension
    filename = f"{asin}{ext}"
    filepath = os.path.join(save_dir, filename)

    # Check if already downloaded
    if os.path.exists(filepath):
        print(f"Exists, skipping: {filepath}")
        return

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filepath, "wb") as img_file:
                for chunk in response.iter_content(1024):
                    img_file.write(chunk)
            print(f"Successfully downloaded: {filepath}")
        else:
            print(f"Failed to download [{response.status_code}]: {url}")
            with open(failed_log, "a", encoding="utf-8") as log:
                log.write(f"{asin}\t{url}\n")

    except requests.RequestException as e:
        print(f"Request error: {url} - {e}")
        with open(failed_log, "a", encoding="utf-8") as log:
            log.write(f"{asin}\t{url}\n")

def process_dataset(dataset_name):
    """Process a single dataset, download all its images"""
    json_file = f"dataset/meta_{dataset_name}.json"
    save_dir = f"dataset/{dataset_name}/image/"
    failed_log = f"dataset/{dataset_name}/failed_downloads_{dataset_name}.txt"

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read JSON file and extract data
    download_tasks = []
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = ast.literal_eval(line)  # Forgiving parser
                asin = data.get("asin", "unknown")
                image_url = data.get("imUrl")

                if image_url:
                    download_tasks.append((save_dir, failed_log, asin, image_url))

            except Exception as e:
                print(f"[Line {idx} parsing failed] {e}")

    # Use multi-threading to download images
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda args: download_image(*args), download_tasks)

    print(f"{dataset_name} dataset download complete!")

# Process all datasets
for dataset in DATASET_NAMES:
    process_dataset(dataset)

print("All dataset image download tasks are complete!")