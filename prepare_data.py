import json
import urllib.request
import os

CATEGORIES = ["apple", "cat", "clock"]
SAMPLES_PER_CATEGORY = 500
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"

def download_and_parse(category):
    url = BASE_URL.format(category.replace(" ", "%20"))
    print(f"Downloading {category}...")
    
    samples = []
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            for i, line in enumerate(response):
                if i >= SAMPLES_PER_CATEGORY:
                    break
                data = json.loads(line)
                
                # data["drawing"] is a list of strokes
                # each stroke is [x_array, y_array]
                
                samples.append({
                    "c": category, # class
                    "d": data["drawing"] # strokes
                })
    except Exception as e:
        print(f"Error downloading {category}: {e}")
        
    return samples

def main():
    dataset = []
    for category in CATEGORIES:
        samples = download_and_parse(category)
        dataset.extend(samples)
        
    import random
    random.seed(42)
    random.shuffle(dataset)
    
    split_idx = int(len(dataset) * 0.8)
    out_data = {
        "train": dataset[:split_idx],
        "test": dataset[split_idx:]
    }
    
    out_file = "dataset.json"
    with open(out_file, "w") as f:
        # compress format slightly
        json.dump(out_data, f, separators=(',', ':'))
        
    print(f"Exported {len(dataset)} samples (Train: {len(out_data['train'])}, Test: {len(out_data['test'])}) to {out_file} ({os.path.getsize(out_file)/1024:.2f} KB)")

if __name__ == "__main__":
    main()
