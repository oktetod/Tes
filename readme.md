# 🤖 Advanced Telegram Roleplay Bot with Qwen AI & PyTorch Learning

Bot Telegram roleplay yang canggih dengan karakter **Sari** - seorang istri muda yang baru menikah. Bot ini menggunakan model **Qwen 2.5** untuk AI yang powerful dan **PyTorch** untuk pembelajaran adaptif.

## ✨ Features

### 🧠 **Advanced AI Capabilities**
- **Model Qwen 2.5-7B/14B** untuk response yang natural dan contextual
- **Multi-language support** (Indonesia, English, Russian)
- **4-bit quantization** untuk optimasi memory
- **GPU acceleration** dengan CUDA support

### 📚 **Adaptive Learning System**
- **Memory consolidation** - Bot belajar dari setiap percakapan
- **Personality adaptation** - Menyesuaikan dengan gaya komunikasi user
- **Context awareness** - Memahami konteks percakapan sebelumnya
- **Sentiment analysis** - Memahami emosi dan mood user
- **Topic preference learning** - Mengingat topik favorit user

### 🛡️ **Production-Ready Features**
- **Automatic database backup** dengan rotasi file
- **Performance monitoring** dengan metrics lengkap
- **Error recovery** dan graceful shutdown
- **Docker support** untuk deployment
- **Logging system** yang comprehensive

### 💕 **Character: Sari**
- **Nama:** Sari, 24 tahun
- **Role:** Istri muda yang baru menikah 3 bulan
- **Personality:** Perhatian, peduli, hangat, kadang manja
- **Interests:** Memasak, dekorasi rumah, film romantis, berkebun
- **Speaking Style:** Natural, penuh kasih sayang, adaptif

## 🚀 Quick Start

### 1. **Setup Environment**

```bash
# Clone atau download semua files
git clone <repository-url>
cd telegram-roleplay-bot

# Install Python 3.8+ jika belum ada
python --version

# Run setup otomatis
python setup.py
```

### 2. **Manual Setup (Alternative)**

```bash
# Install dependencies
pip install -r requirements.txt

# Copy dan edit config
cp config.json.template config.json
# Edit config.json dengan Bot Token Anda
```

### 3. **Dapatkan Bot Token**

1. Chat dengan [@BotFather](https://t.me/BotFather) di Telegram
2. Buat bot baru: `/newbot`
3. Ikuti instruksi untuk mendapatkan token
4. Masukkan token ke `config.json`

### 4. **Run Bot**

```bash
# Method 1: Menggunakan production runner
python run_bot.py

# Method 2: Basic runner
python telegram_bot.py

# Method 3: Menggunakan startup script
./start_bot.sh

# Method 4: Docker (lihat Docker section)
docker-compose up -d
```

## ⚙️ Configuration

File `config.json` berisi semua konfigurasi bot:

### **Model Configuration**
```json
{
  "model_config": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_context_length": 8192,
    "temperature": 0.7,
    "use_4bit_quantization": true
  }
}
```

### **Character Configuration**
```json
{
  "character_config": {
    "name": "Sari",
    "age": 24,
    "personality_adaptation": true,
    "learning_rate": 0.8
  }
}
```

### **Learning Configuration**
```json
{
  "learning_config": {
    "enable_online_learning": true,
    "personality_evolution": true,
    "sentiment_learning": true,
    "memory_consolidation": true
  }
}
```

## 💻 System Requirements

### **Minimum Requirements**
- **OS:** Linux, Windows, macOS
- **Python:** 3.8+
- **RAM:** 8GB (untuk model 7B)
- **Storage:** 15GB untuk model dan cache
- **Internet:** Untuk download model pertama kali

### **Recommended Requirements**
- **GPU:** NVIDIA dengan 8GB+ VRAM
- **RAM:** 16GB+
- **CPU:** 8 cores+
- **Storage:** 50GB+ SSD

### **Model Options**
- **Qwen2.5-7B-Instruct:** ~13GB, cepat, cocok untuk sebagian besar use case
- **Qwen2.5-14B-Instruct:** ~25GB, lebih pintar, butuh GPU lebih powerful
- **Qwen2.5-32B-Instruct:** ~60GB, terpintar, butuh setup khusus

## 📊 Commands

### **User Commands**
- `/start` - Mulai percakapan dan pilih bahasa
- `/help` - Bantuan lengkap
- `/stats` - Lihat statistik percakapan personal
- `/export` - Export riwayat percakapan
- `/reset` - Reset memori percakapan (dengan konfirmasi)

### **Admin Commands**
- `/admin` - Statistik system lengkap (admin only)

## 🐳 Docker Deployment

### **Quick Docker Setup**
```bash
# Build dan run dengan docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

### **Manual Docker Build**
```bash
# Build image
docker build -t roleplay-bot .

# Run dengan GPU support
docker run --gpus all -d \
  --name roleplay-bot \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config.json:/app/config.json \
  roleplay-bot
```

## 🛠️ Advanced Features

### **Memory Consolidation**
Bot secara otomatis menganalisis dan mengkonsolidasi memori setiap 24 jam:
- Menganalisis pola percakapan
- Mengidentifikasi preferensi user
- Mengoptimalkan storage database
- Meningkatkan kualitas response

### **Performance Monitoring**
```python
# Check performance metrics
python -c "
from run_bot import AdvancedBotRunner
runner = AdvancedBotRunner()
# Monitor response times, memory usage, etc.
"
```

### **Database Backup**
- Automatic backup setiap jam
- Rotasi backup (keep 24 files)
- Manual backup: `python backup_db.py`

### **Custom Character Development**
Edit `config.json` untuk mengubah personality:
```json
{
  "character_config": {
    "name": "Nama_Custom",
    "personality_traits": [
      "trait 1",
      "trait 2"
    ],
    "interests": ["hobi 1", "hobi 2"]
  }
}
```

## 🔧 Troubleshooting

### **Common Issues**

1. **Out of Memory Error**
   ```bash
   # Reduce model size atau enable quantization
   "use_4bit_quantization": true
   ```

2. **Model Download Gagal**
   ```bash
   # Check internet connection dan disk space
   huggingface-cli download Qwen/Qwen2.5-7B-Instruct
   ```

3. **Bot Tidak Respond**
   ```bash
   # Check bot token dan connection
   python -c "import requests; print(requests.get('https://api.telegram.org/bot<TOKEN>/getMe').json())"
   ```

4. **Database Lock Error**
   ```bash
   # Stop bot dan check database
   sqlite3 character_memory.db ".timeout 30000"
   ```

### **Performance Optimization**

1. **GPU Optimization**
   ```python
   export CUDA_VISIBLE_DEVICES=0
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

2. **Memory Management**
   ```python
   # Enable gradient checkpointing
   "gradient_checkpointing": true
   
   # Reduce context length
   "max_context_length": 4096
   ```

3. **Response Speed**
   ```python
   # Reduce max_new_tokens
   "max_new_tokens": 150
   
   # Use smaller model
   "model_name": "Qwen/Qwen2.5-7B-Instruct"
   ```

## 📈 Monitoring & Analytics

### **Built-in Metrics**
- Response times per message
- Memory usage tracking
- GPU utilization (if available)
- User engagement metrics
- Error rate monitoring
- Database growth tracking

### **Custom Analytics**
```python
# Access metrics programmatically
from run_bot import AdvancedBotRunner
runner = AdvancedBotRunner()

# Get performance summary
metrics = runner.performance_metrics
print(f"Total messages: {metrics['total_messages']}")
print(f"Average response time: {np.mean(metrics['response_times']):.2f}s")
```

## 📚 Learning System Details

### **How the Bot Learns**

1. **Conversation Analysis**
   - Setiap percakapan dianalisis untuk sentiment, topics, patterns
   - Data disimpan dalam SQLite database
   - Preferensi user dipelajari secara incremental

2. **Personality Adaptation**
   - Bot menyesuaikan response style dengan user
   - Belajar dari feedback (implicit melalui conversation flow)
   - Mengingat konteks long-term

3. **Memory Consolidation**
   - Periodic analysis of conversation patterns
   - Optimization of stored memories
   - Removal of redundant data

### **Data Storage Structure**
```sql
-- Conversations table
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    user_message TEXT,
    bot_response TEXT,
    sentiment REAL,
    language TEXT,
    timestamp DATETIME
);

-- Learned personality traits
CREATE TABLE personality_traits (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    trait_type TEXT,
    trait_value TEXT,
    confidence REAL
);

-- User preferences
CREATE TABLE user_preferences (
    user_id TEXT PRIMARY KEY,
    preferred_language TEXT,
    conversation_style TEXT,
    topics_of_interest TEXT
);
```

## 🤝 Contributing

Kontribusi sangat welcome! Areas yang bisa diimprove:

1. **Additional Character Personalities**
2. **More Language Support** 
3. **Advanced Learning Algorithms**
4. **Better Performance Optimization**
5. **Web Dashboard untuk Analytics**

## 📄 License

MIT License - Feel free to use, modify, dan distribute.

## 🙏 Credits

- **Qwen Team** untuk amazing language model
- **Hugging Face Transformers** untuk easy model integration
- **python-telegram-bot** untuk Telegram API wrapper
- **PyTorch** untuk machine learning framework

## 📞 Support

Jika ada masalah atau pertanyaan:

1. Check troubleshooting section di atas
2. Create issue di repository
3. Join komunitas Telegram bot development

---

**Happy Chatting with Sari! 💕🤖**

> "Semakin sering kamu ngobrol, semakin Sari kenal sama kamu!"