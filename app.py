import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from io import BytesIO
import os
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import time
import shutil
import uuid
import re

# === Global Configuration (Will be filled dynamically) ===
CONFIG = {}
MIN_SCORE_THRESHOLD = 0.4
STANDARD_DIMENSIONS = {
    "chairs": {"width": 45, "depth": 50, "height": 90, "weight": 4.5},
    "stool": {"width": 35, "depth": 35, "height": 45, "weight": 2.5},
    "table": {"width": 120, "depth": 60, "height": 75, "weight": 15},
    "cabinet": {"width": 80, "depth": 40, "height": 180, "weight": 35},
    "bed": {"width": 160, "depth": 200, "height": 45, "weight": 40}
}

# === Logging ===
def setup_logging():
    logger = logging.getLogger("ikea_matcher")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for h in logger.handlers:
            logger.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# === Helper Functions ===
def get_file_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

def get_file_hash_from_path(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Error calculating hash for {path}: {str(e)}")
        return None

def ensure_dirs():
    for path in [CONFIG["cache_dir"], CONFIG["report_path"]]:
        os.makedirs(path, exist_ok=True)

def preprocess_image(image):
    if image is None:
        return None
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, CONFIG["image_size"])

@lru_cache(maxsize=128)
def get_image_url(product_url):
    try:
        if product_url.startswith("file://"):
            return product_url
        response = requests.get(
            product_url,
            headers=CONFIG["request_headers"],
            timeout=CONFIG["request_timeout"]
        )
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            meta = soup.find("meta", {"property": "og:image"})
            if meta and meta.get("content"):
                return meta["content"]
            img_tags = soup.find_all("img", class_=["product-image", "main-image"])
            for img in img_tags:
                if img.get("src") and "thumbnail" not in img["src"]:
                    return img["src"]
    except Exception as e:
        logger.warning(f"Image URL fetch error for {product_url}: {str(e)}")
    return None

def fetch_and_cache_image(image_url, cache_dir):
    cache_file = os.path.join(cache_dir, f"{get_file_hash(image_url)}.jpg")
    try:
        if image_url.startswith("file://"):
            local_path = image_url[7:]
            if os.path.exists(local_path):
                return cv2.imread(local_path)
            else:
                logger.warning(f"Local file not found: {local_path}")
                return None
        if os.path.exists(cache_file):
            return cv2.imread(cache_file)
        res = requests.get(image_url, timeout=CONFIG["request_timeout"])
        if res.status_code != 200:
            logger.warning(f"Failed to fetch image: {image_url}, status: {res.status_code}")
            return None
        image = Image.open(BytesIO(res.content)).convert("RGB")
        image.save(cache_file)
        return cv2.imread(cache_file)
    except Exception as e:
        logger.warning(f"Error processing image {image_url}: {str(e)}")
        return None

def compare_images(input_img, compare_url, cache_dir):
    try:
        image = fetch_and_cache_image(compare_url, cache_dir)
        if image is None:
            return None
        image_processed = preprocess_image(image)
        if image_processed is None or input_img is None:
            return None
        score = ssim(input_img, image_processed)
        return score
    except Exception as e:
        logger.warning(f"Error comparing images: {str(e)}")
        return None

def check_for_exact_match(input_image_path, df):
    input_hash = get_file_hash_from_path(input_image_path)
    if not input_hash:
        return None
    if 'item_id' not in df.columns:
        logger.warning("Column 'item_id' not found in dataset")
        return None
    custom_items = df[pd.notna(df['item_id'])]
    custom_items = custom_items[custom_items['item_id'].apply(
        lambda x: isinstance(x, str) and x.startswith('CUSTOM_')
    )]
    if custom_items.empty:
        return None
    for _, row in custom_items.iterrows():
        if isinstance(row['link'], str) and row['link'].startswith('file://'):
            file_path = row['link'][7:]
            if os.path.exists(file_path):
                custom_hash = get_file_hash_from_path(file_path)
                if custom_hash and custom_hash == input_hash:
                    logger.info(f"✅ Exact match found! Item already in dataset with ID: {row['item_id']}")
                    return row.to_dict()
    return None

def process_product(row, input_img, cache_dir):
    if not isinstance(row["link"], str):
        return None
    if row["link"].startswith("file://"):
        image_url = row["link"]
    elif row["link"].startswith("http"):
        image_url = get_image_url(row["link"])
        if not image_url:
            return None
    else:
        return None
    score = compare_images(input_img, image_url, cache_dir)
    if score is None:
        return None
    return {
        "score": score,
        "link": row["link"],
        "image_url": image_url,
        "details": row.to_dict()
    }

def generate_html_report(input_image_path, matched_data=None, fallback=None):
    output_html_file = os.path.join(CONFIG["report_path"], "report.html")
    if matched_data:
        display_image_url = matched_data['image_url']
        if display_image_url.startswith("file://"):
            display_image_url = display_image_url[7:]
        html_content = f"""
        <html>
        <head><title>Match Report</title></head>
        <body>
            <h1>Match Score: {matched_data.get('score', 'Exact Match')}</h1>
            <p>Product ID: {matched_data['details'].get('item_id', 'N/A')}</p>
            <p><a href="{matched_data['link']}" target="_blank">Product Link</a></p>
            <img src="{input_image_path}" width="300">
            <img src="{display_image_url}" width="300">
        </body>
        </html>
        """
    else:
        html_content = f"""
        <html>
        <head><title>No Match Found</title></head>
        <body>
            <h1>No Match Found</h1>
            <p>Using standard dimensions for: {CONFIG['target_category']}</p>
            <pre>{fallback}</pre>
        </body>
        </html>
        """
    with open(output_html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_file

def add_to_dataset(input_image_path, standard_dimensions):
    try:
        item_id = f"CUSTOM_{uuid.uuid4().hex[:8]}"
        df = pd.read_csv(CONFIG["csv_path"])
        sno = df['Sno'].max() + 1 if 'Sno' in df.columns else 1
        image_filename = f"{item_id}.jpg"
        cached_image_path = os.path.join(CONFIG["cache_dir"], image_filename)
        img = Image.open(input_image_path).convert("RGB")
        img.save(cached_image_path)
        safe_path = cached_image_path.replace("\\", "/")
        dummy_link = f"file://{safe_path}"
        new_row = {
            'Sno': sno,
            'item_id': item_id,
            'category': CONFIG["target_category"],
            'link': dummy_link,
            'width': standard_dimensions.get('width', 0),
            'depth': standard_dimensions.get('depth', 0),
            'height': standard_dimensions.get('height', 0),
            'weight': standard_dimensions.get('weight', 0)
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CONFIG["csv_path"], index=False)
        logger.info(f"✅ Added new item to dataset: {item_id}")
        return True, item_id
    except Exception as e:
        logger.error(f"❌ Error adding item to dataset: {str(e)}")
        return False, None

# === Main Execution Function ===
def main(config):
    global CONFIG
    CONFIG = config
    start_time = time.time()
    logger.info("🔍 IKEA product matcher started")
    ensure_dirs()

    try:
        df = pd.read_csv(CONFIG["csv_path"])
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return

    input_img = cv2.imread(CONFIG["input_path"])
    if input_img is None:
        logger.error(f"❌ Image not found at {CONFIG['input_path']}")
        return
    input_img_processed = preprocess_image(input_img)

    exact_match = check_for_exact_match(CONFIG["input_path"], df)
    if exact_match:
        generate_html_report(CONFIG["input_path"], {"details": exact_match, "link": exact_match["link"], "image_url": exact_match["link"]})
        return

    filtered_df = df
    if CONFIG["target_category"]:
        filtered_df = df[df['category'].str.lower() == CONFIG["target_category"].lower()]
        if filtered_df.empty:
            logger.warning(f"No items found for category: {CONFIG['target_category']}")
            filtered_df = df

    best_match = None
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = [executor.submit(process_product, row, input_img_processed, CONFIG["cache_dir"]) for _, row in filtered_df.iterrows()]
        for future in as_completed(futures):
            result = future.result()
            if not result:
                continue
            if best_match is None or result["score"] > best_match["score"]:
                best_match = result

    if best_match and best_match["score"] >= MIN_SCORE_THRESHOLD:
        generate_html_report(CONFIG["input_path"], best_match)
    else:
        category_key = CONFIG["target_category"].lower()
        standard = STANDARD_DIMENSIONS.get(category_key, {})
        if not standard:
            for key in STANDARD_DIMENSIONS:
                if key in category_key or category_key in key:
                    standard = STANDARD_DIMENSIONS[key]
                    break
            if not standard and STANDARD_DIMENSIONS:
                standard = next(iter(STANDARD_DIMENSIONS.values()))
        add_to_dataset(CONFIG["input_path"], standard)
        generate_html_report(CONFIG["input_path"], None, standard)

    logger.info(f"✅ Done in {time.time() - start_time:.2f}s")

# === CLI Entry Point ===
if __name__ == "__main__" and os.environ.get("FLASK_RUN_FROM_CLI") != "true":
    import argparse
    parser = argparse.ArgumentParser(description="Furniture Matcher")
    parser.add_argument("--input_path", required=True, help="Path to input image")
    parser.add_argument("--category", required=False, help="Target category (e.g., Chairs)")
    args = parser.parse_args()

    cli_config = {
        "csv_path": "./ikea.csv",  # Use relative paths for server compatibility
        "input_path": args.input_path,
        "cache_dir": "./image_cache_ssim",
        "report_path": "./report",
        "target_category": args.category if args.category else "",
        "max_workers": 16,
        "image_size": (300, 300),
        "request_timeout": 10,
        "request_headers": {'User-Agent': 'Mozilla/5.0'}
    }

    main(cli_config)

