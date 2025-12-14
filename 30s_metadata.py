# Cell: Generate 30s Metadata (Short Version)
import os
import pandas as pd
import glob
import random

# --- CONFIG ---
AUDIO_DIR = "./data/gtzan"
OUTPUT_DIR = "metadata_30s"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Collect all 30s files
data = []
print("Scanning files...")

# Loop through genres (folders)
for genre in os.listdir(AUDIO_DIR):
    genre_path = os.path.join(AUDIO_DIR, genre)
    if not os.path.isdir(genre_path): continue
        
    # Find all .wav files
    for filepath in glob.glob(os.path.join(genre_path, "*.wav")):
        # Force Linux slashes
        filepath = filepath.replace(os.sep, '/')
        filename = os.path.basename(filepath)
        
        # Create Song ID (e.g. blues.00001)
        parts = filename.split('.')
        song_id = f"{parts[0]}.{parts[1]}"
        
        data.append({
            "filepath": filepath,
            "label": genre,
            "song_id": song_id
        })

# 2. Shuffle & Split
random.seed(42)
random.shuffle(data)

df = pd.DataFrame(data)

# 80% Train, 10% Val, 10% Test
n = len(df)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

# 3. Save
train_df.to_csv(f"{OUTPUT_DIR}/train_30s.csv", index=False)
val_df.to_csv(f"{OUTPUT_DIR}/val_30s.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test_30s.csv", index=False)

print(f"✅ Done! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")