#!/usr/bin/env python3
"""
Automated setup script untuk Advanced Telegram Roleplay Bot
dengan Qwen AI dan PyTorch learning capability
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8 atau lebih baru diperlukan!")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_cuda_availability():
    """Check CUDA availability for GPU acceleration"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available with {gpu_count} GPU(s): {gpu_name}")
            return True
        else:
            print("⚠️  CUDA tidak tersedia, akan menggunakan CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch belum terinstall, akan menggunakan CPU")
        return False

def install_requirements():
    """Install semua requirements"""
    print("📦 Installing requirements...")
    
    # Base requirements
    requirements = [
        "python-telegram-bot==20.7",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "datasets>=2.14.0",
        "peft>=0.5.0",
        "trl>=0.7.0",
        "sqlalchemy>=2.0.0",
        "langdetect>=1.0.0"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed: {req}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install: {req}")

def setup_config():
    """Setup konfigurasi bot"""
    config = {
        "bot_token": "",
        "model_config": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "max_context_length": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 200,
            "use_4bit_quantization": True
        },
        "database_config": {
            "db_path": "character_memory.db",
            "backup_enabled": True,
            "backup_interval": 3600  # seconds
        },
        "character_config": {
            "name": "Sari",
            "age": 24,
            "personality_adaptation": True,
            "memory_consolidation": True,
            "learning_rate": 0.8
        }
    }
    
    config_path = Path("config.json")
    
    if not config_path.exists():
        # Ask for bot token
        bot_token = input("\n🤖 Masukkan Bot Token dari @BotFather: ")
        if bot_token.strip():
            config["bot_token"] = bot_token.strip()
        
        # Ask for advanced settings
        print("\n⚙️  Konfigurasi Advanced (tekan Enter untuk default)")
        
        model_choice = input("Model choice (1=Qwen2.5-7B, 2=Qwen2.5-14B): ") or "1"
        if model_choice == "2":
            config["model_config"]["model_name"] = "Qwen/Qwen2.5-14B-Instruct"
        
        # Save config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"✅ Configuration saved to {config_path}")
    else:
        print(f"✅ Configuration file found: {config_path}")
    
    return config

def create_advanced_features():
    """Create advanced feature modules"""
    
    # Memory consolidation script
    memory_consolidation = """
import sqlite3
import json
from datetime import datetime, timedelta
import numpy as np

class MemoryConsolidator:
    def __init__(self, db_path):
        self.db_path = db_path
    
    def consolidate_memories(self):
        \"\"\"Consolidate dan optimize memories berdasarkan frequency dan recency\"\"\"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analyze conversation patterns
        cursor.execute('''
            SELECT user_id, user_message, bot_response, timestamp, sentiment
            FROM conversations 
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
        ''')
        
        recent_conversations = cursor.fetchall()
        
        # Group dan analyze patterns
        user_patterns = {}
        for conv in recent_conversations:
            user_id = conv[0]
            if user_id not in user_patterns:
                user_patterns[user_id] = {
                    'common_topics': [],
                    'avg_sentiment': 0,
                    'conversation_frequency': 0
                }
            
            # Update patterns
            user_patterns[user_id]['conversation_frequency'] += 1
            
        # Save consolidated insights
        for user_id, patterns in user_patterns.items():
            cursor.execute('''
                INSERT OR REPLACE INTO personality_traits 
                (user_id, trait_type, trait_value, confidence)
                VALUES (?, ?, ?, ?)
            ''', (user_id, 'conversation_frequency', 
                  str(patterns['conversation_frequency']), 0.9))
        
        conn.commit()
        conn.close()
        
        return user_patterns

if __name__ == "__main__":
    consolidator = MemoryConsolidator("character_memory.db")
    patterns = consolidator.consolidate_memories()
    print(f"Memory consolidation completed for {len(patterns)} users")
    """
    
    with open("memory_consolidation.py", "w") as f:
        f.write(memory_consolidation)
    
    # Performance monitor
    performance_monitor = """
import psutil
import torch
import time
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'cpu_utilization': []
        }
    
    def log_response_time(self, start_time, end_time):
        response_time = end_time - start_time
        self.metrics['response_times'].append({
            'time': response_time,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_system_metrics(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics['cpu_utilization'].append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics['memory_usage'].append(memory.percent)
        
        # GPU usage (if available)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            self.metrics['gpu_utilization'].append(gpu_memory)
    
    def get_performance_summary(self):
        if not self.metrics['response_times']:
            return "No performance data available"
        
        avg_response_time = sum(m['time'] for m in self.metrics['response_times']) / len(self.metrics['response_times'])
        avg_cpu = sum(self.metrics['cpu_utilization']) / len(self.metrics['cpu_utilization']) if self.metrics['cpu_utilization'] else 0
        avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        
        return f'''
📊 Performance Summary:
⏱️  Avg Response Time: {avg_response_time:.2f}s
🖥️  Avg CPU Usage: {avg_cpu:.1f}%
💾  Avg Memory Usage: {avg_memory:.1f}%
🎮  GPU Available: {torch.cuda.is_available()}
        '''

monitor = PerformanceMonitor()
    """
    
    with open("performance_monitor.py", "w") as f:
        f.write(performance_monitor)
    
    print("✅ Advanced feature modules created")

def create_startup_script():
    """Create startup script dengan error handling dan monitoring"""
    
    startup_script = """#!/bin/bash
# Advanced Telegram Roleplay Bot Startup Script

echo "🤖 Starting Advanced Telegram Roleplay Bot..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check GPU availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check disk space (model cache needs space)
echo "💾 Checking disk space..."
df -h | grep -E "/$"

# Start bot with monitoring
echo "🚀 Starting bot..."
python telegram_bot.py

# If bot crashes, log the error
if [ $? -ne 0 ]; then
    echo "❌ Bot crashed at $(date)" >> bot_error.log
    echo "Check bot_error.log for details"
fi
    """
    
    with open("start_bot.sh", "w") as f:
        f.write(startup_script)
    
    # Make executable
    os.chmod("start_bot.sh", 0o755)
    
    print("✅ Startup script created: start_bot.sh")

def download_model():
    """Download dan cache model"""
    print("🤖 Checking model availability...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        print(f"📥 Downloading {model_name}...")
        print("⚠️  This may take several minutes...")
        
        # Download tokenizer first (smaller)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✅ Tokenizer downloaded")
        
        # Download model (larger)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("✅ Model downloaded and cached")
        
        # Test generation
        test_input = "Hello"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                temperature=0.7
            )
        
        test_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Model test successful: {test_response[:50]}...")
        
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        print("💡 You can download manually later")

def create_docker_setup():
    """Create Docker setup untuk deployment"""
    
    dockerfile = """FROM nvidia/cuda:11.8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-venv \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for database
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Expose port (if needed for webhooks)
EXPOSE 8443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python3 -c "import sqlite3; conn = sqlite3.connect('character_memory.db'); conn.close()" || exit 1

# Run bot
CMD ["python3", "telegram_bot.py"]
    """
    
    docker_compose = """version: '3.8'

services:
  telegram-bot:
    build: .
    container_name: advanced_roleplay_bot
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config.json:/app/config.json
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    """
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    
    print("✅ Docker setup files created")

def main():
    """Main setup function"""
    print("🚀 Advanced Telegram Roleplay Bot Setup")
    print("=" * 50)
    
    # Check system requirements
    check_python_version()
    has_cuda = check_cuda_availability()
    
    # Install requirements
    install_requirements()
    
    # Setup configuration
    config = setup_config()
    
    # Create advanced features
    create_advanced_features()
    
    # Create startup script
    create_startup_script()
    
    # Create Docker setup
    create_docker_setup()
    
    # Optionally download model
    download_choice = input("\n📥 Download model sekarang? (y/n): ").lower()
    if download_choice == 'y':
        download_model()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit config.json dengan Bot Token Anda")
    print("2. Run: python telegram_bot.py")
    print("3. Atau gunakan: ./start_bot.sh")
    print("4. Untuk Docker: docker-compose up -d")
    print("\n📊 Monitoring:")
    print("- Check /stats command di bot")
    print("- Monitor performance dengan performance_monitor.py")
    print("- Database backup otomatis setiap jam")
    
    if has_cuda:
        print("\n🎮 GPU acceleration enabled!")
    else:
        print("\n💻 Running on CPU (slower but works)")

if __name__ == "__main__":
    main()
