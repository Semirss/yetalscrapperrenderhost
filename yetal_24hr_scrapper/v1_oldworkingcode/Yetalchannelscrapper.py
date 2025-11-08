import os
import re
import json
import asyncio
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from telethon import TelegramClient
from dotenv import load_dotenv
import pandas as pd
from telethon.errors import ChatForwardsRestrictedError, FloodWaitError, RPCError
from transformers import pipeline, AutoTokenizer, AutoModel
import spacy
from uuid import uuid4
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === 🔐 Load environment ===
load_dotenv()
API_ID = int(os.getenv("API_ID", "24916488"))
API_HASH = os.getenv("API_HASH", "3b7788498c56da1a02e904ff8e92d494")
BOT_TOKEN = os.getenv("BOT_TOKEN")  # your bot token
MONGO_URI = os.getenv("MONGO_URI")

USER_SESSION = "user_session"
BOT_SESSION = "bot_session"
DOWNLOAD_DIR = "downloaded_images"
TARGET_CHANNEL = "@Outis_ss1643"
FORWARDED_FILE = "forwarded_messages.json"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# === ⚡ MongoDB Setup ===
mongo_client = MongoClient(
    MONGO_URI, 
    serverSelectionTimeoutMS=30000, 
    connectTimeoutMS=50000,
    socketTimeoutMS=50000
)
db = mongo_client["yetal"]
collection = db["yetalcollection"]
categories_collection = db["categories"]

channels = [ch["username"] for ch in collection.find({})]
if not channels:
    print("⚠️ No channels found in DB. Add some with your bot first!")
    exit()

# === 🤖 AI Models Setup ===
print("Loading AI models...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")  # For keyword extraction and NER
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# === 📚 Dynamic Categories ===
def load_categories():
    """Load categories from MongoDB with their representative descriptions."""
    default_categories = [
        {"name": "Electronics", "description": "Devices like phones, laptops, and gadgets"},
        {"name": "Fashion", "description": "Clothing, shoes, and accessories"},
        {"name": "Home Goods", "description": "Furniture, decor, and household items"},
        {"name": "Beauty", "description": "Cosmetics, skincare, and makeup products"},
        {"name": "Sports", "description": "Sporting equipment and gear"},
        {"name": "Books", "description": "Books, novels, and literature"},
        {"name": "Groceries", "description": "Food and grocery items"},
        {"name": "Vehicles", "description": "Cars, bikes, and vehicles"},
        {"name": "Medicine", "description": "Medicines and health remedies"},
        {"name": "Perfume", "description": "Fragrances, colognes, and perfumes"}
    ]
    stored_categories = categories_collection.find_one({"_id": "categories"})
    if stored_categories and "categories" in stored_categories:
        return stored_categories["categories"]
    else:
        categories_collection.insert_one({"_id": "categories", "categories": default_categories})
        return default_categories

def save_categories(categories):
    """Save updated categories to MongoDB."""
    categories_collection.update_one(
        {"_id": "categories"},
        {"$set": {"categories": categories}},
        upsert=True
    )

def get_text_embedding(text):
    """Generate embedding for a text using sentence-transformers."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = embedder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def propose_new_category(text, classification_results, existing_categories):
    """Propose a general category using semantic similarity and keyword extraction."""
    text = clean_text(text).lower()
    doc = nlp(text)
    
    # Use NER to exclude brand names and products
    entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]}
    
    # Extract single-word nouns, excluding noise and entities
    keywords = [
        token.text for token in doc 
        if token.pos_ in ["NOUN"] and  # Only nouns, not PROPN
        len(token.text) > 3 and 
        token.text.lower() not in entities and 
        token.text.lower() not in ['item', 'product', 'thing', 'restocked', 'detail', 'catch', 'new', 'sale']
    ]
    
    # Generate embedding for the input text
    text_embedding = get_text_embedding(text)
    
    # Compare with existing category descriptions
    category_names = [cat["name"] for cat in existing_categories]
    category_descriptions = [cat["description"] for cat in existing_categories]
    category_embeddings = [get_text_embedding(desc) for desc in category_descriptions]
    
    similarities = cosine_similarity(text_embedding, category_embeddings)[0]
    max_similarity = max(similarities) if similarities.size > 0 else 0
    best_category_idx = np.argmax(similarities) if similarities.size > 0 else -1
    
    # If similarity is high enough, use the closest existing category
    if max_similarity > 0.7:  # Adjustable threshold
        return category_names[best_category_idx]
    
    # If keywords exist, use the first one as a new category
    if keywords:
        new_category = keywords[0].capitalize()
        # Avoid overly specific categories by checking against existing ones
        for cat in existing_categories:
            if new_category.lower() in cat["description"].lower():
                return cat["name"]
        return new_category
    
    # Fallback to classifier's top category
    if classification_results and classification_results["labels"]:
        return classification_results["labels"][0]
    
    return "Miscellaneous"

# === 🧹 Text Helpers ===
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s,.]', '', text)
    # Remove promotional and noise terms
    text = re.sub(r'\b(restocked|detail|catch|new|sale|nishane|montale|phoera|libre|vanille)\b', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_info(text, message_id):
    text = clean_text(text)
    
    title_match = re.split(r'\n|💸|☘️☘️PRICE|Price\s*:|💵', text)[0].strip()
    title = title_match[:100] if title_match else "No Title"
    
    phone_matches = re.findall(r'(\+251\d{8,9}|09\d{8})', text)
    phone = phone_matches[0] if phone_matches else ""
    
    price_match = re.search(
        r'(Price|💸|☘️☘️PRICE)[:\s]*([\d,]+)|([\d,]+)\s*(ETB|Birr|birr|💵)', 
        text, 
        re.IGNORECASE
    )
    price = ""
    if price_match:
        price = price_match.group(2) or price_match.group(3) or ""
        price = price.replace(',', '').strip()
    
    location_match = re.search(
        r'(📍|Address|Location|🌺🌺)[:\s]*(.+?)(?=\n|☘️|📞|@|$)', 
        text, 
        re.IGNORECASE
    )
    location = location_match.group(2).strip() if location_match else ""
    
    channel_mention = re.search(r'(@\w+)', text)
    channel_mention = channel_mention.group(1) if channel_mention else ""
    
    return {
        "title": title,
        "description": text,
        "price": price,
        "phone": phone,
        "location": location,
        "channel_mention": channel_mention,
        "product_ref": str(message_id) 
    }

# === 🤖 AI Enrichment Function with Dynamic Categories ===
def enrich_product(title, desc):
    text = desc if isinstance(desc, str) and len(desc) > 10 else title
    text = clean_text(text)

    # Load current categories
    categories = load_categories()

    # Category classification
    try:
        classification = classifier(text, candidate_labels=[cat["name"] for cat in categories])
        top_category = classification["labels"][0]
        top_score = classification["scores"][0]

        # Use semantic similarity for unseen data
        new_category = propose_new_category(text, classification, categories)
        if new_category not in [cat["name"] for cat in categories]:
            # Add new category with a description
            new_cat_description = text[:100]  # Use first 100 chars as description
            categories.append({"name": new_category, "description": new_cat_description})
            save_categories(categories)
            print(f"📚 Added new category: {new_category}")
        category = new_category
    except Exception as e:
        print(f"⚠️ Classification error: {e}")
        category = "Unknown"

    # Summarized description
    try:
        if len(text) > 50:
            summary = summarizer(text, max_length=40, min_length=10, do_sample=False)[0]["summary_text"]
        else:
            summary = text
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        summary = text[:80]

    return category, summary

# === 📦 Scraper Function with AI Enrichment ===
async def scrape_and_enrich(client, timeframe="24h"):
    results = []  
    seen_posts = set()  

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24) if timeframe == "24h" else now - timedelta(days=7)
    
    try:
        target_entity = await client.get_entity(TARGET_CHANNEL)
        print(f"✅ Target channel resolved: {target_entity.title}")
    except Exception as e:
        print(f"❌ Could not resolve target channel {TARGET_CHANNEL}: {e}")
        return
    
    source_messages = []
    for channel in channels:
        print(f"📡 Scanning channel: {channel}")
        try:
            channel_entity = await client.get_entity(channel)
            print(f"✅ Channel resolved: {channel_entity.title}")
            
            async for message in client.iter_messages(channel_entity, limit=None):
                if not message.text:
                    continue
                if message.date < cutoff:
                    break
                source_messages.append({
                    'text': message.text,
                    'date': message.date,
                    'source_channel': channel,
                    'source_message_id': message.id
                })
        except Exception as e:
            print(f"❌ Error processing channel {channel}: {e}")
            continue

    print(f"🔍 Searching for matching messages in target channel...")
    async for message in client.iter_messages(target_entity, limit=None):
        if not message.text:
            continue
        if message.date < cutoff:
            break
        if message.id in seen_posts:
            continue
        seen_posts.add(message.id)

        matching_source = None
        for source_msg in source_messages:
            if (source_msg['text'] in message.text or 
                message.text in source_msg['text'] or
                source_msg['text'][:100] in message.text):
                matching_source = source_msg
                break

        if not matching_source:
            continue

        info = extract_info(message.text, message.id)
        print(f"🤖 Enriching product: {info['title'][:50]}...")
        predicted_category, generated_description = enrich_product(info["title"], info["description"])
        
        if getattr(target_entity, "username", None):
            post_link = f"https://t.me/{target_entity.username}/{message.id}"
        else:
            internal_id = str(target_entity.id)
            if internal_id.startswith("-100"):
                internal_id = internal_id[4:]
            post_link = f"https://t.me/c/{internal_id}/{message.id}"

        post_images = []
        post_data = {
            "title": info["title"],
            "description": info["description"],
            "price": info["price"],
            "phone": info["phone"],
            "images": post_images,
            "location": info["location"],
            "date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": info["channel_mention"] if info["channel_mention"] else matching_source['source_channel'],
            "post_link": post_link,   
            "product_ref": str(message.id),
            "predicted_category": predicted_category,
            "generated_description": generated_description
        }
        results.append(post_data)

    results = [
        post for post in results
        if datetime.strptime(post["date"], "%Y-%m-%d %H:%M:%S") >= cutoff.replace(tzinfo=None)
    ]

    df = pd.DataFrame(results)
    filename_parquet = f"scraped_{timeframe}_enriched.parquet"
    df.to_parquet(filename_parquet, engine="pyarrow", index=False)

    print(f"\n✅ Done. Scraped and enriched {len(results)} posts ({timeframe}).")
    print(f"📁 Enriched data saved to {filename_parquet}")
    
    return df

# === 📤 Forwarding Function ===
async def forward_messages(user, bot, days: int):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    if os.path.exists(FORWARDED_FILE):
        with open(FORWARDED_FILE, "r") as f:
            forwarded_data = json.load(f)
            forwarded_ids = {
                int(msg_id): datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") 
                for msg_id, ts in forwarded_data.items()
            }
    else:
        forwarded_ids = {}

    forwarded_ids = {msg_id: ts for msg_id, ts in forwarded_ids.items() if ts >= cutoff.replace(tzinfo=None)}

    messages_to_forward_by_channel = {channel: [] for channel in channels}
    for channel in channels:
        async for message in user.iter_messages(channel, limit=None):
            if message.date < cutoff:
                break
            if message.id not in forwarded_ids and (message.text or message.media):
                messages_to_forward_by_channel[channel].append(message)

    total_forwarded = 0
    for channel, messages_list in messages_to_forward_by_channel.items():
        if not messages_list:
            continue
        messages_list.reverse()
        for i in range(0, len(messages_list), 100):
            batch = messages_list[i:i+100]
            try:
                await asyncio.wait_for(
                    bot.forward_messages(
                        entity=TARGET_CHANNEL,
                        messages=[msg.id for msg in batch],
                        from_peer=channel
                    ),
                    timeout=20
                )
                await asyncio.sleep(1)
                for msg in batch:
                    forwarded_ids[msg.id] = msg.date.replace(tzinfo=None)
                    total_forwarded += 1
            except ChatForwardsRestrictedError:
                print(f"🚫 Forwarding restricted for channel {channel}, skipping...")
                break
            except FloodWaitError as e:
                print(f"⏳ Flood wait error ({e.seconds}s). Waiting...")
                await asyncio.sleep(e.seconds)
                continue
            except asyncio.TimeoutError:
                print(f"⚠️ Forwarding timed out for {channel}, skipping batch...")
                continue
            except RPCError as e:
                print(f"⚠️ RPC Error for {channel}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Unexpected error forwarding from {channel}: {e}")
                continue
           
    with open(FORWARDED_FILE, "w") as f:
        json.dump({str(k): v.strftime("%Y-%m-%d %H:%M:%S") for k, v in forwarded_ids.items()}, f)

    if total_forwarded > 0:
        print(f"\n✅ Done. Forwarded {total_forwarded} new posts ({days}d) to {TARGET_CHANNEL}.")
    else:
        print("\nℹ️ No new posts to forward. All messages already exist in the target channel.")

# === ⚡ Main execution block ===
async def main():
    user = TelegramClient(USER_SESSION, API_ID, API_HASH)
    await user.start()

    bot = TelegramClient(BOT_SESSION, API_ID, API_HASH)
    await bot.start(bot_token=BOT_TOKEN)

    print("\nStarting 24-hour scrape with AI enrichment...")
    df_24h = await scrape_and_enrich(user, timeframe="24h")

    print("\nStarting 7-day scrape with AI enrichment...")
    df_7d = await scrape_and_enrich(user, timeframe="7d")

    print("\nStarting 7-day forwarding to channel...")
    await forward_messages(user, bot, days=7)

    print("\n" + "="*50)
    print("📊 SCRAPING & ENRICHMENT SUMMARY")
    print("="*50)
    print(f"24-hour posts: {len(df_24h)} products")
    print(f"7-day posts: {len(df_7d)} products")
    
    if len(df_7d) > 0:
        category_counts = df_7d['predicted_category'].value_counts()
        print("\n📈 Category distribution (7-day):")
        for category, count in category_counts.items():
            print(f"  {category}: {count} products")
    
    print("="*50)

    await user.disconnect()
    await bot.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
