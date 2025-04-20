#!/usr/bin/env python3
"""
A complete pipeline for:
1) Sampling key frames from a video.
2) Embedding frames with Moondream vision model.
3) Building a FAISS index + SQLite metadata DB.
4) Querying the index with text questions via sentence-transformers.
"""

import os
import cv2
import sqlite3
import faiss
import numpy as np
import torch
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import moondream as md
from sentence_transformers import SentenceTransformer

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === CONFIGURATION ===
VIDEO_PATH   = "/home/alpha/Downloads/alpha.mp4"          # Input video file
FRAME_DIR    = "frames"                     # Where to store extracted frames
INTERVAL_SEC = 2                            # Seconds between sampled frames
MOONDREAM_PATH = "/home/alpha/Downloads/moondream-2b-int8.mf"  # Path to Moondream model
DB_PATH      = "frames.db"                  # SQLite metadata
INDEX_PATH   = "index.faiss"                # FAISS index file
TOP_K        = 5                            # Number of neighbors to retrieve
EMBEDDING_DIM = 384                         # SentenceTransformer embedding dimension

# Ensure output dirs exist
os.makedirs(FRAME_DIR, exist_ok=True)

# === FRAME EXTRACTION ===
def extract_and_save_frames(video_path, frame_dir, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * interval_sec)
    saved = []
    idx = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = os.path.join(frame_dir, f"frame_{frame_id:05d}.jpg")
            Image.fromarray(rgb).save(path, quality=85)
            saved.append((frame_id, ts, path))
            frame_id += 1
        idx += 1
    cap.release()
    print(f"Extracted {len(saved)} frames from video")
    return saved

# === MODEL INITIALIZATION ===
def load_models():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Initialize Moondream model
    moondream_model = md.vl(model=MOONDREAM_PATH)

    # Initialize SentenceTransformer with CUDA if available
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    return moondream_model, sentence_model

# === IMAGE EMBEDDING ===
def embed_image(moondream_model, img_path):
    """Extract features from an image using Moondream's encode_image function."""
    image = Image.open(img_path)
    encoded_image = moondream_model.encode_image(image)

    # Extract the actual embedding from the Moondream encoded image
    # Since we don't know the exact structure, we'll use the caption
    # functionality to create a description embedding
    caption_result = moondream_model.caption(encoded_image)
    caption = caption_result["caption"]

    # Use SentenceTransformer to create an embedding from this caption
    # This works as a workaround if we can't directly access the image embedding
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    embedding = sentence_model.encode(caption, convert_to_numpy=True)

    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

# === TEXT EMBEDDING ===
def embed_text(sentence_model, text):
    """Embed text using SentenceTransformer."""
    embedding = sentence_model.encode(text, convert_to_numpy=True)
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.reshape(1, -1)  # Reshape for FAISS compatibility

# === DATABASE & INDEX SETUP ===
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY,
            timestamp REAL,
            path TEXT,
            caption TEXT
        )
    """)
    conn.commit()
    return conn

def add_frame_metadata(conn, frame_id, timestamp, path, caption):
    conn.execute(
        "INSERT INTO frames (id, timestamp, path, caption) VALUES (?,?,?,?)",
        (int(frame_id), float(timestamp), path, caption)
    )
    conn.commit()

def build_index(dim):
    return faiss.IndexFlatIP(dim)  # Inner product for cosine similarity with normalized vectors

# === PROCESSING WITH PARALLEL EXECUTION ===
def process_frames(frames, moondream_model, conn, index):
    """Process frames sequentially to avoid any memory issues."""
    processed = []

    # Use SentenceTransformer model for generating embeddings from captions
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    print(f"Processing {len(frames)} frames...")

    for i, (fid, ts, path) in enumerate(frames):
        try:
            # Load image
            image = Image.open(path)

            # Get encoded image from Moondream
            encoded_image = moondream_model.encode_image(image)

            # Generate caption
            caption_result = moondream_model.caption(encoded_image)
            caption = caption_result["caption"]
            print(f"Frame {i+1}/{len(frames)}: {caption}")

            # Create embedding from caption
            embedding = sentence_model.encode(caption, convert_to_numpy=True)
            embedding = embedding / np.linalg.norm(embedding)

            # Add to index
            index.add(embedding.reshape(1, -1))

            # Store metadata
            add_frame_metadata(conn, fid, ts, path, caption)
            processed.append((fid, ts, path))

        except Exception as e:
            print(f"Error processing frame {path}: {e}")

    print(f"Successfully processed {len(processed)} frames")
    return index

# === QUERY FUNCTION ===
def query_scene(index, conn, sentence_model, question, top_k=TOP_K):
    txt_emb = embed_text(sentence_model, question)
    D, I = index.search(txt_emb, top_k)

    # If no results found
    if len(I[0]) == 0:
        return []

    placeholders = ",".join("?" * len(I[0]))
    rows = conn.execute(
        f"SELECT timestamp, path, caption FROM frames WHERE id IN ({placeholders})",
        tuple(map(int, I[0]))
    ).fetchall()

    # Sort results by similarity score
    results = [(D[0][i], ts, path, caption) for i, (ts, path, caption) in enumerate(rows)]
    results.sort(reverse=True)  # Higher score is better

    return [(ts, path, caption, score) for score, ts, path, caption in results]

# === MAIN PIPELINE ===
if __name__ == "__main__":
    # 1) Extract & save frames
    frames = extract_and_save_frames(VIDEO_PATH, FRAME_DIR, INTERVAL_SEC)

    # 2) Load models
    moondream_model, sentence_model = load_models()

    # Get embedding dimension from sentence transformer model
    emb_dim = sentence_model.get_sentence_embedding_dimension()
    print(f"Using embedding dimension: {emb_dim}")

    # 3) Init DB & FAISS
    conn = init_db(DB_PATH)
    index = build_index(dim=emb_dim)

    # 4) Process frames and build index
    index = process_frames(frames, moondream_model, conn, index)

    # 5) Save FAISS index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"Built and saved index to '{INDEX_PATH}' and metadata to '{DB_PATH}'.")

    # 6) Simple interactive query loop
    print("Ready for queries! Type your question or 'exit' to quit.")
    while True:
        q = input(">> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        results = query_scene(index, conn, sentence_model, q)
        if not results:
            print("No relevant frames found.")
        else:
            print(f"Found {len(results)} matches:")
            for ts, path, caption, score in results:
                print(f"Score: {score:.4f} | Time: {ts:.2f}s | Caption: {caption}")
                print(f"File: {path}")
                print("---")

                # Optionally ask Moondream about the frame
                """
                image = Image.open(path)
                encoded_image = moondream_model.encode_image(image)
                answer = moondream_model.query(encoded_image, q)["answer"]
                print(f"Moondream says: {answer}")
                """
