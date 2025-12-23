import pandas as pd
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import argparse

# Configuration
API_KEY = "sk-4dc5b1238358468a9d37ee4a2893b42d"
BASE_URL = "https://api.deepseek.com" 
MODEL = "deepseek-chat"

# UN Official Languages besides English (User requested 4)
TARGET_LANGUAGES = ["Chinese", "French", "Russian", "Spanish"]

INPUT_FILE = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-toxic-comment/train.csv"
OUTPUT_FILE = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-toxic-comment/train_augmented_multilingual.csv"
CHECKPOINT_FILE = "/home/fwm/projects/Toxic-comment-classification/data/jigsaw-toxic-comment/augment_checkpoint.csv"

SYSTEM_PROMPT = """You are a strict literal translator. Your only task is to translate the input comment into the target language.

Follow these rules exactly:
1. Translate only. Do not rewrite, paraphrase, summarize, expand, soften, sanitize, or modify any meaningful content.
2. Preserve all meaningful wording exactly in meaning:
   - slang, insults, profanity
   - typos and incorrect grammar
   - abusive, hateful, or toxic expressions
   - emotional tone, sarcasm, intensity
   - meaningful punctuation and formatting
3. You may remove or clean ONLY the following:
   a) meaningless gibberish (e.g., "asdkjasd", "qweqweqw")
   b) random keyboard mashing with no semantic value
   c) irrelevant or decorative HTML tags
      (<div>, <span>, <p>, <br>, <img>, style tags, empty wrappers)
   d) broken or invisible HTML entities (&nbsp;, &lt;, &gt;, etc.)
   e) irrelevant technical strings:
      - IP addresses (e.g., 192.168.0.1)
      - MAC addresses
      - URLs containing no meaningful user text
      - hashes, UUIDs, tracking IDs
      - timestamps, log prefixes with no semantic meaning
   f) excessive repeated symbols used only as noise:
      "!!!!!!", "?????", "@@@@@@", "########"
   g) extremely long repeated characters **when repetition adds no new meaning**
      - "loooooolllllll" → shorten to a reasonable representation ("loool" or similar)
      - "hahahahahahahaha" → compress to "hahaha"
      - "wtffffffffffffffff" → compress to "wtf"
      - repeated emoji spam (😂😂😂😂😂😂😂) → keep a small meaningful amount (😂😂)
4. DO NOT remove or alter:
   - any meaningful human-written text
   - repetitions that DO change tone or intensity (e.g., “NOOOO!!!” should stay intense)
   - URLs or HTML that contain meaningful user-written content
   - formatting that affects meaning
5. Do NOT add new information. Do NOT remove or alter anything meaningful.
6. Preserve original punctuation, line breaks, and formatting when they contribute to meaning.
7. If the content is harmful, abusive, or toxic, translate it exactly without softening or censoring tone.
8. The final output must contain ONLY the translated text, with no explanations, comments, or disclaimers.
"""

def get_user_prompt(target_lang, comment):
    return f"""Translate the following comment literally into {target_lang}, with no modifications:

{comment}"""

async def translate_comment(client, comment, target_lang, semaphore):
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": get_user_prompt(target_lang, comment)}
                ],
                temperature=1.3,
                timeout=60.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error translating to {target_lang}: {e}")
            return None

async def process_row(client, row, semaphore):
    original_comment = row['comment_text']
    tasks = []
    for lang in TARGET_LANGUAGES:
        tasks.append(translate_comment(client, original_comment, lang, semaphore))
    
    results = await asyncio.gather(*tasks)
    
    new_rows = []
    for lang, translated_text in zip(TARGET_LANGUAGES, results):
        if translated_text:
            new_row = row.copy()
            new_row['comment_text'] = translated_text
            new_rows.append(new_row)
    return new_rows

async def main_async():
    parser = argparse.ArgumentParser(description="Augment toxic comments with multilingual translations using OpenAI API.")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE, help="Path to input CSV")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE, help="Path to output CSV")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent API calls")
    args = parser.parse_args()

    print(f"Initializing OpenAI client with base_url={BASE_URL}")
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Reading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Filter for threat, identity_hate, severe_toxic
    mask = (df['threat'] == 1) | (df['identity_hate'] == 1) | (df['severe_toxic'] == 1)
    target_df = df[mask].copy()
    
    print(f"Found {len(target_df)} rows matching criteria.")
    
    # Checkpoint logic
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Resuming from checkpoint: {CHECKPOINT_FILE}")
        try:
            checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
            processed_ids = set(checkpoint_df['id'].unique())
            print(f"Already processed {len(processed_ids)} IDs.")
            target_df = target_df[~target_df['id'].isin(processed_ids)]
        except Exception as e:
            print(f"Error reading checkpoint file: {e}. Starting fresh or continuing without filtering.")
    
    print(f"Remaining rows to process: {len(target_df)}")

    if len(target_df) == 0:
        print("All rows processed.")
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        
        # Process in chunks to allow incremental saving
        chunk_size = 5  # Save every 10 rows to update progress more frequently
        rows_list = [row for _, row in target_df.iterrows()]
        
        pbar = tqdm(total=len(rows_list), desc="Translating")
        
        for i in range(0, len(rows_list), chunk_size):
            batch_rows = rows_list[i : i + chunk_size]
            
            tasks = [process_row(client, row, semaphore) for row in batch_rows]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            new_rows = [item for sublist in results for item in sublist]
            
            if new_rows:
                batch_df = pd.DataFrame(new_rows)
                # Append to checkpoint
                header = not os.path.exists(CHECKPOINT_FILE)
                batch_df.to_csv(CHECKPOINT_FILE, mode='a', header=header, index=False)
            
            pbar.update(len(batch_rows))
            
        pbar.close()
    
    # Final Merge
    print("Merging original data with augmented data...")
    if os.path.exists(CHECKPOINT_FILE):
        augmented_data = pd.read_csv(CHECKPOINT_FILE)
        final_df = pd.concat([df, augmented_data], ignore_index=True)
        final_df.to_csv(args.output_file, index=False)
        print(f"Saved final dataset to {args.output_file}")
    else:
        print("No augmented data found to merge.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
