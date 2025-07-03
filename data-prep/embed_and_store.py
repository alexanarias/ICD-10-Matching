import os
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Dimensions for text-embedding-3-small

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Local state tracking
PROCESSED_IDS_LOG_FILE = 'data-prep/processed_ids.log'

# Batching settings
EMBEDDING_BATCH_SIZE = 500  # Number of texts to embed in one OpenAI call
UPSERT_BATCH_SIZE = 200     # Number of vectors to upsert in one Pinecone call

def load_and_prepare_data() -> pd.DataFrame:
    """
    Loads CM parsed data and prepares it for embedding.
    PCS data loading is temporarily disabled.
    """
    print("Loading and preparing ICD-10-CM data...")
    
    # Load diagnosis codes (CM)
    with open('data-prep/json/icd10cm_parsed.json', 'r') as f:
        cm_data = json.load(f)
    cm_df = pd.DataFrame(cm_data)
    cm_df['type'] = 'DIAGNOSIS'  # This is the critical metadata field for future use

    # --- Temporarily disabling PCS data loading ---
    # print("Loading and preparing data...")
    # with open('data-prep/icd10pcs_parsed.json', 'r') as f:
    #     pcs_data = json.load(f)
    # pcs_df = pd.DataFrame(pcs_data)
    # pcs_df['type'] = 'PROCEDURE'
    # combined_df = pd.concat([cm_df, pcs_df], ignore_index=True)
    
    # For now, we only use the CM dataframe
    combined_df = cm_df

    # Ensure 'code' is the unique ID and 'rich_text' is not empty
    combined_df = combined_df.dropna(subset=['code', 'rich_text'])
    combined_df = combined_df[combined_df['rich_text'].str.strip() != '']
    combined_df['id'] = combined_df['code']
    
    print(f"Loaded a total of {len(combined_df)} codes to process.")
    return combined_df

def initialize_pinecone():
    """Initializes and returns a Pinecone index object."""
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",  # Cosine similarity is standard for text embeddings
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index created. Waiting for it to initialize...")
        # Wait a moment for the index to be ready
        time.sleep(10)
    else:
        print(f"Found existing index: '{PINECONE_INDEX_NAME}'.")
        
    return pc.Index(PINECONE_INDEX_NAME)

def get_processed_ids_from_log() -> set:
    """
    Reads the log file of successfully processed IDs to prevent re-processing.
    This is the source of truth for resumability.
    """
    if not os.path.exists(PROCESSED_IDS_LOG_FILE):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(PROCESSED_IDS_LOG_FILE), exist_ok=True)
        return set()
    
    print(f"Reading already processed IDs from {PROCESSED_IDS_LOG_FILE}...")
    with open(PROCESSED_IDS_LOG_FILE, 'r') as f:
        processed_ids = {line.strip() for line in f}
    print(f"Found {len(processed_ids)} processed IDs in local log.")
    return processed_ids

def log_processed_ids(ids_to_log: list):
    """Appends a list of successfully processed IDs to the log file."""
    with open(PROCESSED_IDS_LOG_FILE, 'a') as f:
        for item_id in ids_to_log:
            f.write(f"{item_id}\n")

def create_embeddings(texts: list, client: OpenAI) -> list:
    """Creates embeddings for a batch of texts using OpenAI."""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [record.embedding for record in response.data]

def main():
    """Main function to run the embedding and storage process."""
    
    # --- 1. Initialization ---
    all_data_df = load_and_prepare_data()
    index = initialize_pinecone()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # --- 2. Implement Resumability using a Local Log File ---
    processed_ids = get_processed_ids_from_log()
    
    if processed_ids:
        # Filter out data that has already been processed based on the local log
        unprocessed_df = all_data_df[~all_data_df['id'].isin(processed_ids)]
        print(f"Resuming process. Found {len(unprocessed_df)} new codes to embed.")
    else:
        unprocessed_df = all_data_df
    
    if unprocessed_df.empty:
        print("All codes are already embedded and stored. Exiting.")
        return

    # --- 3. Sequential Batch Processing Loop ---
    total_batches = (len(unprocessed_df) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    print(f"Starting embedding process in {total_batches} batches...")

    for i in tqdm(range(0, len(unprocessed_df), EMBEDDING_BATCH_SIZE), desc="Embedding Batches"):
        batch_df = unprocessed_df.iloc[i:i + EMBEDDING_BATCH_SIZE]
        
        # A. Generate Embeddings (Costly Step)
        texts_to_embed = batch_df['rich_text'].tolist()
        try:
            embeddings = create_embeddings(texts_to_embed, openai_client)
        except Exception as e:
            print(f"An error occurred during OpenAI embedding generation: {e}")
            print("Skipping this batch and continuing. It will be retried on the next run.")
            continue
        
        # B. Prepare vectors and metadata for Pinecone
        vectors_to_upsert = []
        for idx, row in enumerate(batch_df.itertuples()):
            # Create a comprehensive metadata payload.
            # Pinecone metadata values must be strings, numbers, booleans, or lists of strings.
            metadata = {
                "text": str(row.rich_text or ''),
                "type": str(row.type or ''),
                "description": str(row.description or ''),
            }

            if row.type == 'DIAGNOSIS':
                metadata['chapter'] = str(row.chapter_name or '')
                metadata['section'] = str(row.section_name or '')
                # Safely add notes and keywords, ensuring they are lists of strings
                metadata['includes_notes'] = [str(n) for n in row.includes_notes] if hasattr(row, 'includes_notes') and row.includes_notes else []
                metadata['excludes1_notes'] = [str(n) for n in row.excludes1_notes] if hasattr(row, 'excludes1_notes') and row.excludes1_notes else []
                metadata['excludes2_notes'] = [str(n) for n in row.excludes2_notes] if hasattr(row, 'excludes2_notes') and row.excludes2_notes else []
                metadata['code_first'] = [str(n) for n in row.code_first] if hasattr(row, 'code_first') and row.code_first else []
                metadata['code_also'] = [str(n) for n in row.code_also] if hasattr(row, 'code_also') and row.code_also else []
                metadata['use_additional_code'] = [str(n) for n in row.use_additional_code] if hasattr(row, 'use_additional_code') and row.use_additional_code else []
                metadata['keywords'] = [str(k) for k in row.all_keywords] if hasattr(row, 'all_keywords') and row.all_keywords else []

            vectors_to_upsert.append({
                "id": row.id,
                "values": embeddings[idx],
                "metadata": metadata
            })
        
        # C. Upsert to Pinecone and Log Success
        for j in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
            chunk = vectors_to_upsert[j:j + UPSERT_BATCH_SIZE]
            try:
                index.upsert(vectors=chunk)
                # If upsert is successful, log the IDs immediately
                successful_ids = [v['id'] for v in chunk]
                log_processed_ids(successful_ids)
            except Exception as e:
                print(f"An error occurred during Pinecone upsert: {e}")
                print(f"Failed to upsert IDs: {[v['id'] for v in chunk]}.")
                print("These will be retried on the next run. Continuing...")

    print("Embedding and storage process completed successfully.")

if __name__ == "__main__":
    main() 