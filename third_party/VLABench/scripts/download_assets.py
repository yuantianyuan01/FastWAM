import gdown
import os
import zipfile
import argparse
import VLABench

asset_url = "https://drive.google.com/file/d/1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-/view?usp=sharing"
scene_url = "https://drive.google.com/file/d/1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw/view?usp=drive_link"

asset_id = asset_url.split("/d/")[1].split("/")[0]
scene_id = scene_url.split("/d/")[1].split("/")[0]

def download_assets():
    target_path = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
    zip_path = os.path.join(target_path, "obj.zip")
    gdown.download(f"https://drive.google.com/uc?id={asset_id}", zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    os.remove(zip_path)
    print(f"Asset data has been downloaded, extracted to {target_path}, and the zip file has been deleted.")
    
def download_scene():
    target_path = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
    zip_path = os.path.join(target_path, "scene.zip")
    gdown.download(f"https://drive.google.com/uc?id={scene_id}", zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    os.remove(zip_path)
    print(f"Scene data has been downloaded, extracted to {target_path}, and the zip file has been deleted.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download assets and scene data.')
    parser.add_argument('--choice', default="all", choices=['all', 'asset', 'scene'], help='Download assets')
    args = parser.parse_args()
    
    if args.choice == "asset":
        download_assets()
    elif args.choice == "scene":
        download_scene()
    elif args.choice == "all":
        download_assets()
        download_scene()
    else:
        raise ValueError("Invalid choice, must be one of 'all', 'asset', 'scene'")