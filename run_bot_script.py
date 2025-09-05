#!/usr/bin/env python3
"""
Production Bot Runner dengan Advanced Monitoring dan Error Recovery
(Rate Limiting Removed)
"""

import asyncio
import logging
import json
import sqlite3
import pickle
import time
import threading
import signal
import sys
import os
import psutil
from datetime import datetime, timedelta
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from collections import deque, defaultdict
import numpy as np

# Import modules dari bot utama
from telegram_bot import QwenCharacterAI, CharacterMemory

class AdvancedBotRunner:
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.ai_character = None
        self.application = None
        self.performance_metrics = {
            'start_time': time.time(),
            'total_messages': 0,
            'response_times': deque(maxlen=1000),
            'errors': deque(maxlen=100),
            'user_sessions': defaultdict(int),
            'memory_usage': deque(maxlen=100),
            'gpu_usage': deque(maxlen=100) if torch.cuda.is_available() else None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        # Start background tasks
        self.start_background_tasks()
    
    def load_config(self, config_path):
        """Load konfigurasi dari file JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['bot_token', 'model_config', 'character_config']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Required field '{field}' not found in config")
            
            return config
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            raise
    
    def setup_logging(self):
        """Setup advanced logging"""
        log_config = self.config.get('logging_config', {})
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup formatter
        formatter = logging.Formatter(
            log_config.get('log_format', 
                          '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Setup file handler
        if log_config.get('file_logging', True):
            file_handler = logging.handlers.RotatingFileHandler(
                f"logs/{log_config.get('log_file', 'bot.log')}",
                maxBytes=10*1024*1024,  # 10MB
                backupCount=log_config.get('backup_count', 3)
            )
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # Set level
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        logging.getLogger().setLevel(log_level)
    
    def start_background_tasks(self):
        """Start background monitoring dan maintenance tasks"""
        
        # Performance monitoring
        def performance_monitor():
            while True:
                try:
                    # Memory usage
                    process = psutil.Process()
                    memory_percent = process.memory_percent()
                    self.performance_metrics['memory_usage'].append(memory_percent)
                    
                    # GPU usage
                    if torch.cuda.is_available() and self.performance_metrics['gpu_usage'] is not None:
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                        self.performance_metrics['gpu_usage'].append(gpu_memory)
                    
                    time.sleep(30)  # Monitor every 30 seconds
                    
                except Exception as e:
                    logging.error(f"Performance monitoring error: {e}")
                    time.sleep(60)
        
        # Database backup
        def database_backup():
            while True:
                try:
                    backup_config = self.config.get('database_config', {})
                    if backup_config.get('backup_enabled', True):
                        self.backup_database()
                    
                    # Sleep for backup interval
                    interval = backup_config.get('backup_interval', 3600)
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"Database backup error: {e}")
                    time.sleep(3600)  # Retry in 1 hour
        
        # Memory consolidation
        def memory_consolidation():
            while True:
                try:
                    if self.ai_character and self.config.get('learning_config', {}).get('enable_online_learning', True):
                        # Perform memory consolidation
                        self.consolidate_memories()
                    
                    # Sleep for consolidation interval (24 hours default)
                    interval = self.config.get('database_config', {}).get('memory_consolidation_interval', 86400)
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"Memory consolidation error: {e}")
                    time.sleep(3600)
        
        # Start threads
        threading.Thread(target=performance_monitor, daemon=True).start()
        threading.Thread(target=database_backup, daemon=True).start()
        threading.Thread(target=memory_consolidation, daemon=True).start()
        
        logging.info("Background tasks started")
    
    def backup_database(self):
        """Backup database dengan rotasi"""
        try:
            backup_config = self.config.get('database_config', {})
            backup_dir = Path(backup_config.get('backup_path', 'backups'))
            backup_dir.mkdir(exist_ok=True)
            
            # Create backup filename dengan timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = backup_dir / f"character_memory_{timestamp}.db"
            
            # Copy database
            db_path = backup_config.get('db_path', 'character_memory.db')
            if Path(db_path).exists():
                import shutil
                shutil.copy2(db_path, backup_file)
                
                # Cleanup old backups
                max_backups = backup_config.get('max_backups', 24)
                backups = sorted(backup_dir.glob('character_memory_*.db'))
                if len(backups) > max_backups:
                    for old_backup in backups[:-max_backups]:
                        old_backup.unlink()
                
                logging.info(f"Database backed up to {backup_file}")
            
        except Exception as e:
            logging.error(f"Database backup failed: {e}")
    
    def consolidate_memories(self):
        """Consolidate memories untuk optimize storage dan improve learning"""
        try:
            if self.ai_character:
                # Run memory consolidation
                conn = sqlite3.connect(self.ai_character.memory.db_path)
                cursor = conn.cursor()
                
                # Analyze conversation patterns untuk each user
                cursor.execute('''
                    SELECT user_id, COUNT(*) as msg_count,
                           AVG(sentiment) as avg_sentiment,
                           MAX(timestamp) as last_interaction
                    FROM conversations 
                    WHERE timestamp > datetime('now', '-30 days')
                    GROUP BY user_id
                ''')
                
                user_stats = cursor.fetchall()
                
                for user_id, msg_count, avg_sentiment, last_interaction in user_stats:
                    # Update user activity level
                    if msg_count > 100:
                        activity_level = 'high'
                    elif msg_count > 20:
                        activity_level = 'medium'
                    else:
                        activity_level = 'low'
                    
                    # Update consolidated personality traits
                    cursor.execute('''
                        INSERT OR REPLACE INTO personality_traits
                        (user_id, trait_type, trait_value, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, 'activity_level', activity_level, 0.9))
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO personality_traits
                        (user_id, trait_type, trait_value, confidence)
                        VALUES (?, ?, ?, ?)
                    ''', (user_id, 'overall_sentiment', str(avg_sentiment), 0.8))
                
                conn.commit()
                conn.close()
                
                logging.info(f"Memory consolidation completed for {len(user_stats)} users")
                
        except Exception as e:
            logging.error(f"Memory consolidation failed: {e}")
    
    async def initialize_ai(self):
        """Initialize AI character dengan error handling"""
        try:
            logging.info("Initializing AI character...")
            
            # Load model dengan config
            model_config = self.config['model_config']
            self.ai_character = QwenCharacterAI(
                model_name=model_config['model_name']
            )
            
            # Update AI character config
            self.ai_character.base_personality.update(self.config['character_config'])
            
            logging.info("AI character initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize AI: {e}")
            return False
    
    async def enhanced_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command dengan user onboarding"""
        user_id = str(update.effective_user.id)
        
        # Create inline keyboard untuk language selection
        keyboard = [
            [
                InlineKeyboardButton("🇮🇩 Indonesia", callback_data='lang_id'),
                InlineKeyboardButton("🇺🇸 English", callback_data='lang_en'),
                InlineKeyboardButton("🇷🇺 Русский", callback_data='lang_ru')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = """
✨ Selamat datang! Welcome! Добро пожаловать! ✨

Aku Sari, istri muda yang baru menikah 3 bulan lalu! 💕

Pilih bahasa yang kamu suka untuk ngobrol:
Choose your preferred language:
Выберите предпочитаемый язык:
        """
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    
    async def language_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle language selection callback"""
        query = update.callback_query
        await query.answer()
        
        user_id = str(query.from_user.id)
        language = query.data.split('_')[1]
        
        # Save language preference
        if self.ai_character:
            conn = sqlite3.connect(self.ai_character.memory.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_preferences
                (user_id, preferred_language)
                VALUES (?, ?)
            ''', (user_id, language))
            
            conn.commit()
            conn.close()
        
        # Send welcome message dalam bahasa yang dipilih
        welcome_messages = {
            'id': "Senang banget bisa ngobrol sama kamu! Cerita dong, gimana harimu? 😊",
            'en': "I'm so excited to chat with you! Tell me, how was your day? 😊", 
            'ru': "Я так рада общаться с тобой! Расскажи, как прошел твой день? 😊"
        }
        
        await query.edit_message_text(welcome_messages.get(language, welcome_messages['id']))
    
    async def enhanced_handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced message handler without rate limiting"""
        user_id = str(update.effective_user.id)
        user_message = update.message.text
        
        start_time = time.time()
        
        try:
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
            
            # Generate response
            if self.ai_character:
                response = self.ai_character.generate_response(user_id, user_message)
            else:
                response = "Maaf sayang, aku lagi loading nih... Tunggu sebentar ya 🤖"
            
            # Send response
            await update.message.reply_text(response)
            
            # Update metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            self.performance_metrics['total_messages'] += 1
            self.performance_metrics['response_times'].append(response_time)
            self.performance_metrics['user_sessions'][user_id] += 1
            
        except Exception as e:
            # Log error
            logging.error(f"Error in message handler: {e}")
            self.performance_metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'user_id': user_id
            })
            
            # Send fallback response
            fallback_responses = [
                "Aduh maaf sayang, aku lagi bingung nih... 🥺",
                "Ups, ada yang error... Coba lagi ya? 😅", 
                "Maaf ya, aku lagi tidak fokus... Bisa ulangi? 🤔"
            ]
            
            import random
            await update.message.reply_text(random.choice(fallback_responses))
    
    async def admin_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command untuk melihat statistik detail (admin only)"""
        user_id = str(update.effective_user.id)
        
        # Simple admin check (you can enhance this)
        admin_users = self.config.get('admin_users', [])
        if user_id not in admin_users and user_id != "YOUR_ADMIN_USER_ID":
            await update.message.reply_text("Access denied 🚫")
            return
        
        # Generate detailed stats
        uptime = time.time() - self.performance_metrics['start_time']
        uptime_hours = uptime / 3600
        
        avg_response_time = np.mean(self.performance_metrics['response_times']) if self.performance_metrics['response_times'] else 0
        
        total_users = len(self.performance_metrics['user_sessions'])
        avg_memory = np.mean(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0
        
        stats_text = f"""
🤖 **Advanced Bot Statistics**

⏰ **Uptime:** {uptime_hours:.1f} hours
📊 **Performance:**
  • Total messages: {self.performance_metrics['total_messages']}
  • Avg response time: {avg_response_time:.2f}s
  • Active users: {total_users}
  • Errors: {len(self.performance_metrics['errors'])}

💾 **Memory Usage:** {avg_memory:.1f}%
🔥 **GPU Available:** {torch.cuda.is_available()}
"""
        
        if torch.cuda.is_available() and self.performance_metrics['gpu_usage']:
            avg_gpu = np.mean(self.performance_metrics['gpu_usage'])
            stats_text += f"🎮 **GPU Usage:** {avg_gpu:.1f}%\n"
        
        # Database stats
        if self.ai_character:
            conn = sqlite3.connect(self.ai_character.memory.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            unique_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM personality_traits")
            learned_traits = cursor.fetchone()[0]
            
            conn.close()
            
            stats_text += f"""
🧠 **Learning Statistics:**
  • Total conversations: {total_conversations}
  • Unique users: {unique_users}
  • Learned traits: {learned_traits}
"""
        
        await update.message.reply_text(stats_text, parse_mode='Markdown')
    
    async def user_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """User-friendly stats command"""
        user_id = str(update.effective_user.id)
        
        if not self.ai_character:
            await update.message.reply_text("AI belum siap nih... 🤖")
            return
        
        try:
            # Get user-specific data
            profile = self.ai_character.memory.get_user_profile(user_id)
            history = self.ai_character.memory.get_conversation_history(user_id, limit=50)
            
            # Calculate user stats
            total_messages = len(history)
            user_session_count = self.performance_metrics['user_sessions'].get(user_id, 0)
            
            stats_text = f"""
📊 **Statistik kamu dengan Sari:**

💬 **Percakapan:** {total_messages} pesan
🧠 **Yang Sari pelajari tentang kamu:** {len(profile.get('traits', []))} traits
⏰ **Session hari ini:** {user_session_count} pesan

🎯 **Topik yang sering dibahas:**
"""
            
            # Show learned traits
            traits = profile.get('traits', [])[:5]
            for trait in traits:
                stats_text += f"• {trait[0]}: {trait[1]}\n"
            
            if not traits:
                stats_text += "• Belum ada yang dipelajari, ayo ngobrol lebih banyak! 😊"
            
            # Add personality insights
            if total_messages > 10:
                stats_text += f"\n💕 **Sari's Note:** "
                if total_messages > 50:
                    stats_text += "Kita udah sering ngobrol ya! Sari makin kenal sama kamu nih 😊"
                elif total_messages > 20:
                    stats_text += "Sari mulai kenal kepribadian kamu loh! 😄"
                else:
                    stats_text += "Ayo ngobrol lebih banyak biar Sari makin kenal sama kamu! 🤗"
            
            await update.message.reply_text(stats_text)
            
        except Exception as e:
            logging.error(f"Error in user stats: {e}")
            await update.message.reply_text("Maaf, error waktu ngecek statistik... 😅")
    
    async def export_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Export conversation history untuk user"""
        user_id = str(update.effective_user.id)
        
        if not self.ai_character:
            await update.message.reply_text("AI belum siap nih... 🤖")
            return
        
        try:
            history = self.ai_character.memory.get_conversation_history(user_id, limit=500)
            
            if not history:
                await update.message.reply_text("Belum ada riwayat percakapan nih 😊")
                return
            
            # Create conversation export
            export_text = f"📝 Riwayat Percakapan dengan Sari\n"
            export_text += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            export_text += f"Total Messages: {len(history)}\n"
            export_text += "="*50 + "\n\n"
            
            for user_msg, bot_resp, timestamp in history:
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_time = timestamp
                
                export_text += f"[{formatted_time}]\n"
                export_text += f"You: {user_msg}\n"
                export_text += f"Sari: {bot_resp}\n\n"
            
            # Save to file
            export_dir = Path("exports")
            export_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = export_dir / f"conversation_{user_id}_{timestamp}.txt"
            
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(export_text)
            
            # Send file to user
            with open(export_file, 'rb') as f:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=f,
                    filename=f"conversation_with_sari_{timestamp}.txt",
                    caption="📄 Ini riwayat percakapan kita! Simpan ya sebagai kenangan 💕"
                )
            
            # Cleanup file
            export_file.unlink()
            
        except Exception as e:
            logging.error(f"Error in export conversation: {e}")
            await update.message.reply_text("Maaf, error waktu export percakapan... 😅")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced help command"""
        help_text = """
🌟 **Bantuan Bot Sari** 🌟

**Commands:**
/start - Mulai percakapan dengan Sari
/stats - Lihat statistik percakapan kamu
/export - Export riwayat percakapan
/help - Bantuan ini
/reset - Reset memori percakapan

**Features:**
🗣️ **Multi-bahasa:** Indonesia, English, Русский
🧠 **Smart Learning:** Sari belajar dari setiap percakapan
💕 **Adaptive Personality:** Sari menyesuaikan dengan karaktermu
📊 **Advanced Memory:** Ingat konteks dan preferensi
🎯 **Context Aware:** Memahami situasi dan emosi

**Tips:**
• Ngobrol santai aja, Sari akan belajar gaya komunikasimu
• Cerita tentang harimu, hobi, atau apapun yang kamu mau
• Sari bisa kasih saran tentang rumah tangga dan kehidupan sehari-hari
• Semakin sering ngobrol, semakin Sari kenal sama kamu

**Bahasa yang didukung:**
🇮🇩 Bahasa Indonesia
🇺🇸 English  
🇷🇺 Русский язык

Ayo mulai ngobrol! Sari siap jadi teman chatting terbaikmu! 💕
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset user's conversation memory"""
        user_id = str(update.effective_user.id)
        
        # Create confirmation keyboard
        keyboard = [
            [
                InlineKeyboardButton("✅ Ya, reset", callback_data=f'reset_confirm_{user_id}'),
                InlineKeyboardButton("❌ Batal", callback_data='reset_cancel')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤔 Yakin mau reset semua memori percakapan kita?\n"
            "Sari akan lupa semua yang udah dipelajari tentang kamu loh...",
            reply_markup=reply_markup
        )
    
    async def reset_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle reset confirmation callback"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'reset_cancel':
            await query.edit_message_text("Oke, memori kita tetap ada! Ayo lanjut ngobrol 😊")
            return
        
        if query.data.startswith('reset_confirm_'):
            user_id = query.data.split('_')[-1]
            
            try:
                if self.ai_character:
                    # Delete user's conversation history and traits
                    conn = sqlite3.connect(self.ai_character.memory.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                    cursor.execute("DELETE FROM personality_traits WHERE user_id = ?", (user_id,))
                    cursor.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
                    
                    conn.commit()
                    conn.close()
                
                await query.edit_message_text(
                    "✅ Reset berhasil! Sari akan berkenalan lagi dari awal.\n"
                    "Halo! Sepertinya kita belum pernah ngobrol sebelumnya ya? 😊"
                )
                
            except Exception as e:
                logging.error(f"Error in reset: {e}")
                await query.edit_message_text("❌ Error waktu reset... Coba lagi nanti ya 😅")
    
    def graceful_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logging.info("Received shutdown signal, shutting down gracefully...")
        
        # Save any pending data
        try:
            if self.ai_character:
                # Force backup before shutdown
                self.backup_database()
                logging.info("Final database backup completed")
        except Exception as e:
            logging.error(f"Error during shutdown backup: {e}")
        
        # Stop the application
        if self.application:
            logging.info("Stopping bot application...")
            self.application.stop_running()
        
        sys.exit(0)
    
    async def run_bot(self):
        """Main function to run the bot"""
        try:
            # Initialize AI
            if not await self.initialize_ai():
                logging.error("Failed to initialize AI, exiting...")
                return
            
            # Create application
            self.application = Application.builder().token(self.config['bot_token']).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.enhanced_start))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("stats", self.user_stats_command))
            self.application.add_handler(CommandHandler("export", self.export_conversation))
            self.application.add_handler(CommandHandler("reset", self.reset_command))
            self.application.add_handler(CommandHandler("admin", self.admin_stats))
            
            # Callback handlers
            self.application.add_handler(CallbackQueryHandler(self.language_callback, pattern=r'^lang_'))
            self.application.add_handler(CallbackQueryHandler(self.reset_callback, pattern=r'^reset_'))
            
            # Message handler (no rate limiting)
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.enhanced_handle_message))
            
            # Log startup info
            logging.info("🚀 Advanced Sari Bot starting (Rate Limiting Disabled)...")
            logging.info(f"🤖 Model: {self.config['model_config']['model_name']}")
            logging.info(f"💾 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
            logging.info(f"🧠 Learning enabled: {self.config.get('learning_config', {}).get('enable_online_learning', True)}")
            logging.info(f"📊 Performance monitoring: {self.config.get('performance_config', {}).get('enable_performance_monitoring', True)}")
            logging.info("⚠️  Security features disabled - No rate limiting")
            
            # Run bot
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            
            logging.info("✅ Bot is running! Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
        except Exception as e:
            logging.error(f"Fatal error: {e}")
        finally:
            if self.application:
                await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()

def main():
    """Entry point"""
    try:
        # Check if config exists
        if not Path("config.json").exists():
            print("❌ config.json not found!")
            print("💡 Run setup.py first or create config.json manually")
            return
        
        # Create and run bot
        bot_runner = AdvancedBotRunner()
        asyncio.run(bot_runner.run_bot())
        
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
        print(f"❌ Failed to start bot: {e}")

if __name__ == "__main__":
    main()