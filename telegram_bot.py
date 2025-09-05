import asyncio
import logging
import json
import sqlite3
import pickle
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import threading
import queue
import numpy as np
from collections import deque

# Konfigurasi logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Konfigurasi
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MEMORY_DB = "character_memory.db"
MAX_CONTEXT_LENGTH = 8192
LEARNING_RATE = 0.0001

class CharacterMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database untuk menyimpan memori karakter"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabel untuk menyimpan percakapan
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sentiment REAL,
                language TEXT,
                context_embedding BLOB
            )
        ''')
        
        # Tabel untuk personality traits yang dipelajari
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                trait_type TEXT,
                trait_value TEXT,
                confidence REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabel untuk preferensi pengguna
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_language TEXT,
                conversation_style TEXT,
                topics_of_interest TEXT,
                relationship_stage TEXT,
                last_interaction DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, user_id, user_message, bot_response, sentiment=0.0, language="id"):
        """Simpan percakapan ke database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_id, user_message, bot_response, sentiment, language)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, user_message, bot_response, sentiment, language))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, user_id, limit=10):
        """Ambil riwayat percakapan untuk konteks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, bot_response, timestamp 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        history = cursor.fetchall()
        conn.close()
        
        return list(reversed(history))
    
    def update_personality_trait(self, user_id, trait_type, trait_value, confidence=1.0):
        """Update atau tambah personality trait"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO personality_traits 
            (user_id, trait_type, trait_value, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_id, trait_type, trait_value, confidence))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self, user_id):
        """Ambil profil lengkap pengguna"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ambil preferensi
        cursor.execute('''
            SELECT * FROM user_preferences WHERE user_id = ?
        ''', (user_id,))
        preferences = cursor.fetchone()
        
        # Ambil personality traits
        cursor.execute('''
            SELECT trait_type, trait_value, confidence 
            FROM personality_traits 
            WHERE user_id = ?
            ORDER BY confidence DESC
        ''', (user_id,))
        traits = cursor.fetchall()
        
        conn.close()
        
        return {
            'preferences': preferences,
            'traits': traits
        }

class QwenCharacterAI:
    def __init__(self, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = CharacterMemory(MEMORY_DB)
        self.conversation_cache = {}
        
        # Load model dengan optimasi
        self.load_model(model_name)
        
        # Character base personality
        self.base_personality = {
            "name": "Sari",
            "age": 24,
            "role": "istri muda yang baru menikah",
            "personality_traits": [
                "perhatian dan peduli",
                "suka memasak dan mengurus rumah",
                "sedikit pemalu tapi hangat",
                "masih belajar menjadi istri yang baik",
                "romantis dan penuh kasih sayang"
            ],
            "interests": ["memasak", "dekorasi rumah", "film romantis", "berkebun kecil"],
            "speaking_style": "hangat, penuh kasih sayang, kadang manja"
        }
    
    def load_model(self, model_name):
        """Load Qwen model dengan optimasi"""
        try:
            # Konfigurasi quantization untuk efisiensi
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logging.info(f"Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def analyze_sentiment(self, text):
        """Analisis sentimen sederhana"""
        positive_words = [
            'senang', 'bahagia', 'cinta', 'sayang', 'suka', 'bagus', 'cantik', 'indah',
            'happy', 'love', 'good', 'nice', 'beautiful', 'wonderful', 'amazing',
            'счастлив', 'любовь', 'хорошо', 'красиво', 'замечательно'
        ]
        
        negative_words = [
            'sedih', 'kecewa', 'marah', 'bosan', 'lelah', 'stress', 'buruk',
            'sad', 'angry', 'tired', 'bad', 'terrible', 'awful', 'hate',
            'грустно', 'злой', 'плохо', 'ужасно', 'ненавижу'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return min(1.0, positive_count * 0.3)
        elif negative_count > positive_count:
            return max(-1.0, -negative_count * 0.3)
        else:
            return 0.0
    
    def detect_language(self, text):
        """Enhanced language detection"""
        indonesian_words = ['aku', 'kamu', 'saya', 'anda', 'yang', 'dan', 'dengan', 'untuk', 'ini', 'itu']
        english_words = ['i', 'you', 'the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'this', 'that']
        russian_words = ['я', 'ты', 'вы', 'это', 'что', 'как', 'но', 'или', 'для', 'тот', 'этот']
        
        text_lower = text.lower()
        
        id_count = sum(1 for word in indonesian_words if word in text_lower)
        en_count = sum(1 for word in english_words if word in text_lower)
        ru_count = sum(1 for word in russian_words if word in text_lower)
        
        if id_count > en_count and id_count > ru_count:
            return "id"
        elif en_count > ru_count:
            return "en"
        else:
            return "ru"
    
    def build_context(self, user_id, current_message):
        """Build rich context dari history dan user profile"""
        # Ambil conversation history
        history = self.memory.get_conversation_history(user_id, limit=5)
        
        # Ambil user profile
        profile = self.memory.get_user_profile(user_id)
        
        # Build context string
        context_parts = []
        
        # Base personality
        context_parts.append(f"Kamu adalah {self.base_personality['name']}, {self.base_personality['role']}.")
        context_parts.append(f"Umurmu {self.base_personality['age']} tahun.")
        context_parts.append(f"Kepribadianmu: {', '.join(self.base_personality['personality_traits'])}.")
        
        # Learned traits dari percakapan sebelumnya
        if profile['traits']:
            learned_traits = [f"{trait[0]}: {trait[1]}" for trait in profile['traits'][:3]]
            context_parts.append(f"Yang sudah kamu pelajari tentang pengguna: {', '.join(learned_traits)}.")
        
        # Recent conversation history
        if history:
            context_parts.append("\nPercakapan sebelumnya:")
            for user_msg, bot_resp, timestamp in history[-3:]:
                context_parts.append(f"User: {user_msg}")
                context_parts.append(f"Sari: {bot_resp}")
        
        # Language instruction
        language = self.detect_language(current_message)
        if language == "en":
            context_parts.append("\nRespond in English naturally.")
        elif language == "ru":
            context_parts.append("\nОтвечай на русском языке естественно.")
        else:
            context_parts.append("\nJawab dalam bahasa Indonesia dengan natural.")
        
        context_parts.append(f"\nUser sekarang: {current_message}")
        context_parts.append("Sari:")
        
        return "\n".join(context_parts)
    
    def generate_response(self, user_id, user_message):
        """Generate response menggunakan Qwen dengan context learning"""
        try:
            # Detect language dan sentiment
            language = self.detect_language(user_message)
            sentiment = self.analyze_sentiment(user_message)
            
            # Build rich context
            full_context = self.build_context(user_id, user_message)
            
            # Tokenize input
            inputs = self.tokenizer(
                full_context,
                return_tensors="pt",
                max_length=MAX_CONTEXT_LENGTH,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract hanya bagian response baru
            response = full_response.split("Sari:")[-1].strip()
            
            # Clean up response
            response = self.clean_response(response)
            
            # Save conversation untuk learning
            self.memory.save_conversation(
                user_id, user_message, response, sentiment, language
            )
            
            # Update learned traits berdasarkan percakapan
            self.update_learned_traits(user_id, user_message, sentiment)
            
            return response
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return self.get_fallback_response(language)
    
    def clean_response(self, response):
        """Clean dan format response"""
        # Remove unwanted patterns
        response = response.split("User:")[0].strip()
        response = response.split("\n\n")[0].strip()
        
        # Limit length
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        return response
    
    def update_learned_traits(self, user_id, user_message, sentiment):
        """Update traits yang dipelajari dari user"""
        # Deteksi topik dan preferensi
        topics = {
            'makanan': ['makan', 'masak', 'resep', 'food', 'cook', 'recipe', 'еда', 'готовить'],
            'pekerjaan': ['kerja', 'kantor', 'boss', 'work', 'office', 'job', 'работа', 'офис'],
            'hobi': ['hobi', 'suka', 'hobby', 'like', 'enjoy', 'хобби', 'нравится'],
            'keluarga': ['keluarga', 'orangtua', 'family', 'parents', 'семья', 'родители']
        }
        
        user_message_lower = user_message.lower()
        
        for topic, keywords in topics.items():
            if any(keyword in user_message_lower for keyword in keywords):
                self.memory.update_personality_trait(
                    user_id, 'interest', topic, confidence=0.8
                )
        
        # Update sentiment patterns
        if abs(sentiment) > 0.5:
            mood = 'positive' if sentiment > 0 else 'negative'
            self.memory.update_personality_trait(
                user_id, 'typical_mood', mood, confidence=abs(sentiment)
            )
    
    def get_fallback_response(self, language="id"):
        """Fallback response jika ada error"""
        fallback_responses = {
            "id": "Maaf sayang, aku lagi bingung nih... Bisa cerita lagi? 🥺",
            "en": "Sorry honey, I'm a bit confused right now... Could you tell me again? 🥺",
            "ru": "Извини дорогой, я немного запуталась... Можешь рассказать еще раз? 🥺"
        }
        return fallback_responses.get(language, fallback_responses["id"])

# Global AI instance
ai_character = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    user_id = str(update.effective_user.id)
    
    welcome_messages = {
        "id": f"""
✨ Halo sayang! Aku Sari ✨

Aku istri muda yang baru menikah 3 bulan lalu. Aku senang banget bisa ngobrol sama kamu! 💕

Yang bisa kita lakukan bareng:
• Ngobrol tentang apa aja yang kamu mau 
• Cerita tentang hari-harimu
• Minta saran soal rumah tangga
• Atau sekedar curhat aja 😊

Aku bisa ngomong dalam bahasa Indonesia, English, atau русский язык!

Ayo mulai ngobrol! Cerita dong, gimana harimu? 🌸
        """,
        "en": f"""
✨ Hello darling! I'm Sari ✨

I'm a young wife who just got married 3 months ago. I'm so excited to chat with you! 💕

What we can do together:
• Talk about anything you want
• Share about your day
• Ask for advice about household stuff  
• Or just chat casually 😊

I can speak Indonesian, English, or русский язык!

Let's start chatting! Tell me, how was your day? 🌸
        """
    }
    
    # Detect user's language preference dari context atau default ke ID
    welcome_text = welcome_messages.get("id")
    await update.message.reply_text(welcome_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pesan biasa dengan AI response"""
    global ai_character
    
    user_id = str(update.effective_user.id)
    user_message = update.message.text
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        # Generate response menggunakan AI
        response = ai_character.generate_response(user_id, user_message)
        
        # Send response
        await update.message.reply_text(response)
        
    except Exception as e:
        logging.error(f"Error in handle_message: {e}")
        fallback = ai_character.get_fallback_response() if ai_character else "Maaf, ada error nih... 🥺"
        await update.message.reply_text(fallback)

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command untuk melihat statistik pembelajaran"""
    user_id = str(update.effective_user.id)
    
    if ai_character:
        profile = ai_character.memory.get_user_profile(user_id)
        history = ai_character.memory.get_conversation_history(user_id, limit=100)
        
        stats_text = f"""
📊 Statistik Pembelajaran Sari tentang kamu:

💬 Total percakapan: {len(history)}
🧠 Yang sudah dipelajari: {len(profile['traits'])} traits

🎯 Topik favorit yang terdeteksi:
        """
        
        # Tampilkan learned traits
        for trait in profile['traits'][:5]:
            stats_text += f"• {trait[0]}: {trait[1]} (confidence: {trait[2]:.1f})\n"
            
        await update.message.reply_text(stats_text)
    else:
        await update.message.reply_text("AI belum siap nih... 🤖")

def main():
    """Fungsi utama untuk menjalankan bot"""
    global ai_character
    
    try:
        # Initialize AI character
        logging.info("Loading AI character...")
        ai_character = QwenCharacterAI()
        logging.info("AI character loaded successfully!")
        
        # Create bot application
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Run bot
        print("🤖 Advanced Sari Bot dimulai dengan Qwen AI...")
        print(f"💾 Device: {ai_character.device}")
        print(f"🧠 Model: {MODEL_NAME}")
        
        application.run_polling()
        
    except Exception as e:
        logging.error(f"Error starting bot: {e}")
        raise

if __name__ == '__main__':
    main()
