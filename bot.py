import os
import asyncio
import json
import re
import pandas as pd
import requests
import time
import boto3
import io
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from telethon import TelegramClient
from telethon.errors import ChatForwardsRestrictedError, FloodWaitError, RPCError, ChannelInvalidError, UsernameInvalidError, UsernameNotOccupiedError
from telethon.sessions import SQLiteSession
from flask import Flask
import threading
from uuid import uuid4

# === 🔐 Load environment variables ===
load_dotenv()

# Telegram API
API_ID = int(os.getenv("API_ID", "24916488"))
API_HASH = os.getenv("API_HASH", "3b7788498c56da1a02e904ff8e92d494")
BOT_TOKEN = os.getenv("BOT_TOKEN")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")

# Target channel
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "@Outis_ss1643")

# AWS S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

# Hugging Face API
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# File names for S3
USER_SESSION_FILE = "telegram_session.session"
FORWARDED_FILE = "forwarded_messages.json"
SCRAPED_24H_FILE = "scraped_24h_enriched.parquet"

# Initialize services
app = Flask(__name__)
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# === ⚡ MongoDB Setup ===
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["yetal"]
collection = db["yetalcollection"]
categories_collection = db["categories"]

# === 🤖 Hugging Face API Functions ===
def query_huggingface_api(payload, model_name, max_retries=3):
    """Generic function to query Hugging Face API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                wait_time = 10 * (attempt + 1)
                print(f"⏳ Model loading, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timeout (attempt {attempt + 1})")
            continue
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    return None

def ai_classify_text(text, categories):
    """Classify text using Hugging Face zero-shot classification"""
    candidate_labels = [cat["name"] for cat in categories]
    
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": candidate_labels}
    }
    
    result = query_huggingface_api(payload, "facebook/bart-large-mnli")
    
    if result and "labels" in result and "scores" in result:
        return {
            "labels": result["labels"],
            "scores": result["scores"]
        }
    return None

def ai_summarize_text(text):
    """Summarize text using Hugging Face summarization"""
    if len(text) < 50:
        return text
    
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 60,
            "min_length": 20,
            "do_sample": False
        }
    }
    
    result = query_huggingface_api(payload, "facebook/bart-large-cnn")
    
    if result and isinstance(result, list) and len(result) > 0:
        summary = result[0].get("summary_text", text[:80])
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary
    return text[:80]

# === 📚 Dynamic Categories ===
def load_categories():
    """Load categories from MongoDB"""
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
        {"name": "Perfume", "description": "Fragrances, colognes, and perfumes"},
        {"name": "Real Estate", "description": "Properties, houses, and apartments"},
        {"name": "Jobs", "description": "Employment opportunities"},
        {"name": "Services", "description": "Various services"},
        {"name": "Urgent", "description": "Urgent sales and offers"}
    ]
    
    stored_categories = categories_collection.find_one({"_id": "categories"})
    if stored_categories and "categories" in stored_categories:
        return stored_categories["categories"]
    else:
        categories_collection.insert_one({"_id": "categories", "categories": default_categories})
        return default_categories

def save_categories(categories):
    """Save updated categories to MongoDB"""
    categories_collection.update_one(
        {"_id": "categories"},
        {"$set": {"categories": categories}},
        upsert=True
    )

def extract_keywords_with_regex(text):
    """Enhanced keyword extraction using regex patterns"""
    text = clean_text(text).lower()
    
    patterns = {
        'Electronics': r'\b(phone|smartphone|laptop|tablet|computer|tv|television|headphone|earphone|camera|watch|macbook|iphone|samsung)\b',
        'Fashion': r'\b(shirt|dress|jeans|shoe|sneaker|bag|jacket|coat|accessory|jewelry|clothing|wear)\b',
        'Home Goods': r'\b(furniture|sofa|chair|table|bed|mattress|decor|kitchen|appliance|house|apartment)\b',
        'Beauty': r'\b(cosmetic|makeup|skincare|perfume|fragrance|cream|lotion|shampoo|beauty|glam)\b',
        'Sports': r'\b(sport|football|basketball|tennis|gym|fitness|equipment|gear|exercise|training)\b',
        'Vehicles': r'\b(car|bike|motorcycle|vehicle|auto|automobile|toyota|honda|bmw|mercedes|ford|chevrolet)\b',
        'Books': r'\b(book|novel|literature|magazine|textbook|reading|author|story)\b',
        'Groceries': r'\b(food|grocery|fruit|vegetable|meat|drink|beverage|rice|pasta|bread)\b',
        'Real Estate': r'\b(house|apartment|property|land|rent|sale|villa|condo|building|realestate)\b',
        'Jobs': r'\b(job|employment|work|career|vacancy|position|hire|recruitment|opportunity)\b',
        'Services': r'\b(service|repair|maintenance|cleaning|delivery|transport|consultation)\b'
    }
    
    for category, pattern in patterns.items():
        if re.search(pattern, text):
            return category
    
    # Check for urgency
    if re.search(r'\burgent\b|\brush\b|\bemergency\b|\bquick\b|\bfast\b', text, re.IGNORECASE):
        return "Urgent"
    
    # Extract nouns
    words = re.findall(r'\b[a-z]{4,}\b', text)
    common_nouns = [word for word in words if word not in [
        'item', 'product', 'thing', 'restocked', 'detail', 'catch', 'new', 'sale', 
        'price', 'contact', 'sell', 'buy', 'market', 'telegram', 'channel'
    ]]
    
    return common_nouns[0].capitalize() if common_nouns else "Other"

def propose_new_category(text, classification_results, existing_categories):
    """Enhanced category proposal using API classification"""
    text_lower = text.lower()
    
    # Check for urgency first
    if any(word in text_lower for word in ['urgent', 'rush', 'emergency', 'quick sale']):
        return "Urgent"
    
    if classification_results and classification_results["labels"]:
        top_category = classification_results["labels"][0]
        top_score = classification_results["scores"][0]
        
        if top_score > 0.6:
            return top_category
    
    # Enhanced keyword extraction
    keyword_category = extract_keywords_with_regex(text)
    
    # Check if keyword category matches any existing category
    for category in existing_categories:
        if keyword_category.lower() in category["name"].lower():
            return category["name"]
    
    # Special handling for vehicle-related content
    if any(word in text_lower for word in ['toyota', 'honda', 'bmw', 'mercedes', 'car', 'vehicle', 'auto']):
        return "Vehicles"
    
    return keyword_category if keyword_category != "Other" else "Miscellaneous"

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

def enrich_product_with_ai(title, desc):
    """Enhanced product enrichment using Hugging Face API"""
    text = desc if isinstance(desc, str) and len(desc) > 10 else title
    text = clean_text(text)
    
    if not text or len(text) < 10:
        return "Unknown", "No description available"

    # Load current categories
    categories = load_categories()

    # Category classification using API
    try:
        classification_results = ai_classify_text(text, categories)
        
        if classification_results:
            category = propose_new_category(text, classification_results, categories)
        else:
            # Fallback if API fails
            category = extract_keywords_with_regex(text)
            
        # Add new category if it doesn't exist and seems valid
        if (category not in [cat["name"] for cat in categories] and 
            category not in ["Unknown", "Other", "Miscellaneous"] and
            len(category) > 2):
            new_cat_description = f"Products related to {category.lower()}"
            categories.append({"name": category, "description": new_cat_description})
            save_categories(categories)
            print(f"📚 Added new category: {category}")
            
    except Exception as e:
        print(f"⚠️ Classification error: {e}")
        category = extract_keywords_with_regex(text) or "Unknown"

    # Enhanced summarized description using API
    try:
        if len(text) > 50:
            summary = ai_summarize_text(text)
            summary = re.sub(r'(URGENT SELL|URGENT|SELL|BUY)\s+', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'\s+', ' ', summary).strip()
        else:
            summary = text
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        summary = text[:80] + "..." if len(text) > 80 else text

    return category, summary

# === ☁️ AWS S3 File Management ===
def file_exists_in_s3(s3_key):
    """Check if file exists in S3"""
    try:
        s3.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        return True
    except s3.exceptions.NoSuchKey:
        return False
    except Exception as e:
        print(f"❌ Error checking S3 file {s3_key}: {e}")
        return False

def load_json_from_s3(s3_key):
    """Load JSON data directly from S3"""
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        print(f"✅ Loaded JSON from S3: {s3_key}")
        return data
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ JSON file {s3_key} not found in S3, returning empty dict")
        return {}
    except Exception as e:
        print(f"❌ Error loading JSON from S3: {e}")
        return {}

def save_json_to_s3(data, s3_key):
    """Save JSON data directly to S3"""
    try:
        s3.put_object(
            Bucket=AWS_BUCKET_NAME,
            Key=s3_key,
            Body=json.dumps(data).encode('utf-8')
        )
        print(f"✅ Saved JSON to S3: {s3_key}")
        return True
    except Exception as e:
        print(f"❌ Error saving JSON to S3: {e}")
        return False

def load_parquet_from_s3():
    """Load parquet data directly from S3"""
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=f"data/{SCRAPED_24H_FILE}")
        df = pd.read_parquet(io.BytesIO(response['Body'].read()))
        print(f"✅ Loaded parquet from S3: {SCRAPED_24H_FILE}")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ Parquet file {SCRAPED_24H_FILE} not found in S3, returning empty DataFrame")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error loading parquet from S3: {e}")
        return pd.DataFrame()

def save_parquet_to_s3(df):
    """Save parquet data to S3"""
    try:
        if df.empty:
            print("⚠️ DataFrame is empty, nothing to save")
            return False
            
        print(f"💾 Attempting to save {len(df)} records to S3...")
        
        # Use in-memory buffer
        buffer = io.BytesIO()
        
        # Try different parquet engines
        engines = ['pyarrow', 'fastparquet', 'auto']
        success = False
        
        for engine in engines:
            try:
                print(f"🔄 Trying parquet engine: {engine}")
                buffer.seek(0)
                df.to_parquet(buffer, engine=engine, index=False)
                success = True
                print(f"✅ Success with engine: {engine}")
                break
            except Exception as e:
                print(f"❌ Engine {engine} failed: {e}")
                continue
        
        if not success:
            print("❌ All parquet engines failed")
            return False
        
        buffer.seek(0)
        
        # S3 key with proper path
        s3_key = f"data/{SCRAPED_24H_FILE}"
        print(f"📤 Uploading to S3 bucket: {AWS_BUCKET_NAME}")
        
        # Upload to S3
        s3.upload_fileobj(
            buffer, 
            AWS_BUCKET_NAME, 
            s3_key,
            ExtraArgs={'ContentType': 'application/octet-stream'}
        )
        
        print(f"✅ Successfully uploaded {len(df)} records to S3")
        
        # Verify upload
        try:
            response = s3.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
            file_size = response['ContentLength']
            last_modified = response['LastModified']
            print(f"✅ Upload verification successful!")
            print(f"📏 File size: {file_size} bytes")
            print(f"🕒 Last modified: {last_modified}")
            return True
        except Exception as e:
            print(f"⚠️ Upload verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error saving to S3: {e}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

def ensure_s3_structure():
    """Ensure the required S3 folder structure exists"""
    try:
        # Create sessions folder
        s3.put_object(Bucket=AWS_BUCKET_NAME, Key="sessions/")
        print("✅ Created sessions/ folder in S3")
    except Exception:
        print("✅ sessions/ folder already exists in S3")
    
    try:
        # Create data folder
        s3.put_object(Bucket=AWS_BUCKET_NAME, Key="data/")
        print("✅ Created data/ folder in S3")
    except Exception:
        print("✅ data/ folder already exists in S3")

# === 🔄 Session Management with S3 ===
async def get_telethon_client():
    """Get the main Telethon client with S3 session handling"""
    client = None
    max_retries = 3
    retry_delay = 2
    
    # Download session file from S3 to memory
    session_data = None
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=f"sessions/{USER_SESSION_FILE}")
        session_data = response['Body'].read()
        print(f"✅ Downloaded session file from S3: {USER_SESSION_FILE}")
    except s3.exceptions.NoSuchKey:
        print(f"❌ Session file not found in S3: {USER_SESSION_FILE}")
        return None
    except Exception as e:
        print(f"❌ Error downloading session from S3: {e}")
        return None
    
    for attempt in range(max_retries):
        try:
            print(f"🔧 Attempt {attempt + 1}/{max_retries} to connect Telethon client...")
            
            # Create a temporary writable session file
            temp_session_file = f"temp_{USER_SESSION_FILE}"
            with open(temp_session_file, 'wb') as f:
                f.write(session_data)
            
            # Set proper permissions for the temp file
            try:
                os.chmod(temp_session_file, 0o644)
            except:
                pass
            
            # Use the temporary session file
            session = SQLiteSession(temp_session_file)
            client = TelegramClient(session, API_ID, API_HASH)
            
            await asyncio.wait_for(client.connect(), timeout=15)
            
            if not await client.is_user_authorized():
                error_msg = "Session not authorized"
                print(f"❌ {error_msg}")
                await client.disconnect()
                # Clean up temporary file
                if os.path.exists(temp_session_file):
                    os.remove(temp_session_file)
                return None
            
            me = await asyncio.wait_for(client.get_me(), timeout=10)
            print(f"✅ Telethon connected successfully as: {me.first_name} (@{me.username})")
            
            # Store the temp file name for cleanup
            client.temp_session_file = temp_session_file
            return client
            
        except Exception as e:
            error_msg = f"Connection error (attempt {attempt + 1}): {str(e)}"
            print(f"❌ {error_msg}")
            
            if client:
                try:
                    await client.disconnect()
                except:
                    pass
            
            # Clean up temporary file on error
            if 'temp_session_file' in locals() and os.path.exists(temp_session_file):
                try:
                    os.remove(temp_session_file)
                except:
                    pass
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                return None
    
    return None

# === 📤 Forwarding Function ===
async def forward_messages(days: int = 1):
    """Forward messages from source channels to target channel"""
    user = None
    try:
        user = await get_telethon_client()
        if not user:
            print("❌ Failed to initialize Telethon client")
            return False, "Could not establish connection"

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=days)

        # Load forwarded messages from S3
        forwarded_data = load_json_from_s3(f"data/{FORWARDED_FILE}")
        forwarded_ids = {
            int(msg_id): datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") 
            for msg_id, ts in forwarded_data.items()
        } if forwarded_data else {}

        # Remove old forwarded IDs
        forwarded_ids = {msg_id: ts for msg_id, ts in forwarded_ids.items() 
                        if ts >= cutoff.replace(tzinfo=None)}

        # Get channels from MongoDB
        channels = [ch["username"] for ch in collection.find({})]
        if not channels:
            return False, "No channels found in database"

        messages_to_forward_by_channel = {channel: [] for channel in channels}
        for channel in channels:
            try:
                async for message in user.iter_messages(channel, limit=None):
                    if message.date < cutoff:
                        break
                    if message.id not in forwarded_ids and (message.text or message.media):
                        messages_to_forward_by_channel[channel].append(message)
            except Exception as e:
                print(f"❌ Error fetching messages from {channel}: {e}")
                continue

        total_forwarded = 0
        for channel, messages_list in messages_to_forward_by_channel.items():
            if not messages_list:
                continue
            messages_list.reverse()
            
            for i in range(0, len(messages_list), 10):
                batch = messages_list[i:i+10]
                try:
                    await asyncio.wait_for(
                        user.forward_messages(
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

        # Save updated forwarded IDs to S3
        save_json_to_s3(
            {str(k): v.strftime("%Y-%m-%d %H:%M:%S") for k, v in forwarded_ids.items()},
            f"data/{FORWARDED_FILE}"
        )

        if total_forwarded > 0:
            return True, f"✅ Successfully forwarded {total_forwarded} new posts to {TARGET_CHANNEL}."
        else:
            return False, "📭 No new posts to forward."

    except Exception as e:
        return False, f"❌ Critical error: {str(e)}"
    finally:
        if user:
            try:
                await user.disconnect()
                # Upload updated session file to S3
                print("📤 Uploading updated session file to S3...")
                if os.path.exists(USER_SESSION_FILE):
                    with open(USER_SESSION_FILE, 'rb') as f:
                        s3.put_object(
                            Bucket=AWS_BUCKET_NAME,
                            Key=f"sessions/{USER_SESSION_FILE}",
                            Body=f.read()
                        )
                    print(f"✅ Session file uploaded to S3: {USER_SESSION_FILE}")
                    # Clean up temporary file
                    os.remove(USER_SESSION_FILE)
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

# === 📦 Scraping Function with AI Enrichment ===
async def scrape_and_enrich(timeframe="24h"):
    """Scrape and enrich posts with AI from target channel"""
    user = None
    try:
        user = await get_telethon_client()
        if not user:
            return False, "Could not establish connection for scraping"

        # Get channels from MongoDB
        channels = [ch["username"] for ch in collection.find({})]
        if not channels:
            return False, "No channels found in database"

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24) if timeframe == "24h" else now - timedelta(days=7)
        
        try:
            target_entity = await user.get_entity(TARGET_CHANNEL)
            print(f"✅ Target channel resolved: {target_entity.title}")
        except Exception as e:
            return False, f"Could not resolve target channel {TARGET_CHANNEL}: {e}"

        source_messages = []
        for channel in channels:
            print(f"📡 Scanning channel: {channel}")
            try:
                channel_entity = await user.get_entity(channel)
                print(f"✅ Channel resolved: {channel_entity.title}")
                
                async for message in user.iter_messages(channel_entity, limit=None):
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
        results = []
        seen_posts = set()
        
        async for message in user.iter_messages(target_entity, limit=None):
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
            print(f"🤖 AI enriching product: {info['title'][:50]}...")
            predicted_category, generated_description = enrich_product_with_ai(info["title"], info["description"])
            
            if getattr(target_entity, "username", None):
                post_link = f"https://t.me/{target_entity.username}/{message.id}"
            else:
                internal_id = str(target_entity.id)
                if internal_id.startswith("-100"):
                    internal_id = internal_id[4:]
                post_link = f"https://t.me/c/{internal_id}/{message.id}"

            post_data = {
                "title": info["title"],
                "description": info["description"],
                "price": info["price"],
                "phone": info["phone"],
                "location": info["location"],
                "date": message.date.strftime("%Y-%m-%d %H:%M:%S"),
                "channel": info["channel_mention"] if info["channel_mention"] else matching_source['source_channel'],
                "post_link": post_link,
                "product_ref": str(message.id),
                "predicted_category": predicted_category,
                "generated_description": generated_description,
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(post_data)

        # Filter by cutoff date
        results = [
            post for post in results
            if datetime.strptime(post["date"], "%Y-%m-%d %H:%M:%S") >= cutoff.replace(tzinfo=None)
        ]

        # Save to S3
        df = pd.DataFrame(results)
        success = save_parquet_to_s3(df)
        
        if success:
            # Print AI enhancement summary
            if not df.empty and 'predicted_category' in df.columns:
                category_counts = df['predicted_category'].value_counts()
                print("🤖 AI Enhancement Summary:")
                for category, count in category_counts.items():
                    print(f"  • {category}: {count} products")
            
            return True, f"✅ Scraped and enriched {len(results)} posts. Data saved to S3."
        else:
            return False, "❌ Failed to save data to S3."

    except Exception as e:
        return False, f"❌ Scraping error: {str(e)}"
    finally:
        if user:
            try:
                await user.disconnect()
                # Upload session file to S3
                print("📤 Uploading updated session file to S3...")
                if os.path.exists(USER_SESSION_FILE):
                    with open(USER_SESSION_FILE, 'rb') as f:
                        s3.put_object(
                            Bucket=AWS_BUCKET_NAME,
                            Key=f"sessions/{USER_SESSION_FILE}",
                            Body=f.read()
                        )
                    print(f"✅ Session file uploaded to S3: {USER_SESSION_FILE}")
                    # Clean up temporary file
                    os.remove(USER_SESSION_FILE)
            except Exception as e:
                print(f"⚠️ Error during cleanup: {e}")

# === 🕒 Scheduled Task ===
async def scheduled_task():
    """Main scheduled task that runs every 24 hours"""
    print("🕒 Starting scheduled 24-hour task...")
    
    # Step 1: Forward new messages
    print("📤 Step 1: Forwarding new messages...")
    success, message = await forward_messages(days=1)
    print(message)
    
    # Step 2: Scrape and enrich with AI
    print("🤖 Step 2: Scraping and AI enrichment...")
    success, message = await scrape_and_enrich(timeframe="24h")
    print(message)
    
    # Step 3: Print summary
    print("📊 Step 3: Generating summary...")
    df = load_parquet_from_s3()
    if not df.empty:
        print(f"📈 Total AI-enhanced products: {len(df)}")
        if 'predicted_category' in df.columns:
            category_counts = df['predicted_category'].value_counts()
            print("📊 Category distribution:")
            for category, count in category_counts.items():
                print(f"  • {category}: {count} products")
    
    print("✅ Scheduled task completed!")

def run_scheduled_task():
    """Run the scheduled task in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scheduled_task())
    loop.close()

# === 🌐 Flask App for Render ===
@app.route("/")
def home():
    return "🤖 Telegram Scraper Bot is running!"

@app.route("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.route("/run-now")
def run_now():
    """Manual trigger for the scheduled task"""
    threading.Thread(target=run_scheduled_task, daemon=True).start()
    return {"message": "Scheduled task started manually"}

@app.route("/status")
def status():
    """Check system status"""
    try:
        # Check S3 files
        s3_status = {
            "session_file": file_exists_in_s3(f"sessions/{USER_SESSION_FILE}"),
            "forwarded_file": file_exists_in_s3(f"data/{FORWARDED_FILE}"),
            "data_file": file_exists_in_s3(f"data/{SCRAPED_24H_FILE}")
        }
        
        # Load data stats
        df = load_parquet_from_s3()
        data_stats = {
            "total_records": len(df),
            "categories": df['predicted_category'].value_counts().to_dict() if not df.empty and 'predicted_category' in df.columns else {}
        }
        
        return {
            "status": "running",
            "s3_files": s3_status,
            "data_stats": data_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# === 🚀 Main Execution ===
def main():
    """Main function to start the application"""
    print("🚀 Starting Telegram Scraper Bot...")
    
    # Ensure S3 structure exists
    ensure_s3_structure()
    
    # Check S3 files status
    print("🔍 Checking S3 files...")
    files_to_check = {
        "Session File": f"sessions/{USER_SESSION_FILE}",
        "Forwarded Messages": f"data/{FORWARDED_FILE}",
        "Scraped Data": f"data/{SCRAPED_24H_FILE}"
    }
    
    for file_type, s3_key in files_to_check.items():
        exists = file_exists_in_s3(s3_key)
        status = "✅" if exists else "❌"
        print(f"{status} {file_type}: {s3_key}")
    
    # Start Flask app
    port = int(os.environ.get("PORT", 5000))
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=port, debug=False), daemon=True).start()
    print(f"🌐 Flask app started on port {port}")
    
    # Schedule the 24-hour task
    def schedule_daily_task():
        import schedule
        import time
        
        # Schedule task to run every 24 hours
        schedule.every(24).hours.do(run_scheduled_task)
        
        # Also run immediately on startup
        print("⏰ Running initial task...")
        run_scheduled_task()
        
        print("⏰ Scheduled task set to run every 24 hours")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Start scheduler in separate thread
    threading.Thread(target=schedule_daily_task, daemon=True).start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("👋 Shutting down...")

if __name__ == "__main__":
    main()