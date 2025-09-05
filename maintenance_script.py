#!/usr/bin/env python3
"""
Database Maintenance & Backup Script untuk Advanced Telegram Roleplay Bot
"""

import sqlite3
import json
import shutil
import gzip
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

class DatabaseMaintenance:
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.db_path = self.config.get('database_config', {}).get('db_path', 'character_memory.db')
        self.backup_dir = Path(self.config.get('database_config', {}).get('backup_path', 'backups'))
        self.backup_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
    
    def create_backup(self, compress=True):
        """Create database backup"""
        try:
            if not Path(self.db_path).exists():
                logging.error(f"Database {self.db_path} not found!")
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"character_memory_{timestamp}.db"
            
            if compress:
                backup_name += ".gz"
                backup_path = self.backup_dir / backup_name
                
                # Create compressed backup
                with open(self.db_path, 'rb') as f_in:
                    with gzip.open(backup_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                backup_path = self.backup_dir / backup_name
                shutil.copy2(self.db_path, backup_path)
            
            logging.info(f"Backup created: {backup_path}")
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
    
    def cleanup_old_backups(self):
        """Remove old backup files"""
        try:
            max_backups = self.config.get('database_config', {}).get('max_backups', 24)
            
            # Get all backup files
            backup_files = sorted(self.backup_dir.glob('character_memory_*.db*'))
            
            if len(backup_files) > max_backups:
                files_to_remove = backup_files[:-max_backups]
                for file_path in files_to_remove:
                    file_path.unlink()
                    logging.info(f"Removed old backup: {file_path}")
                
                logging.info(f"Cleaned up {len(files_to_remove)} old backup files")
        
        except Exception as e:
            logging.error(f"Backup cleanup failed: {e}")
    
    def restore_backup(self, backup_file):
        """Restore database from backup"""
        try:
            backup_path = Path(backup_file)
            
            if not backup_path.exists():
                # Try to find in backup directory
                backup_path = self.backup_dir / backup_file
                
                if not backup_path.exists():
                    logging.error(f"Backup file not found: {backup_file}")
                    return False
            
            # Create backup of current database
            current_backup = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if Path(self.db_path).exists():
                shutil.copy2(self.db_path, current_backup)
                logging.info(f"Current database backed up to: {current_backup}")
            
            # Restore from backup
            if backup_path.suffix == '.gz':
                # Decompress and restore
                with gzip.open(backup_path, 'rb') as f_in:
                    with open(self.db_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(backup_path, self.db_path)
            
            logging.info(f"Database restored from: {backup_path}")
            
            # Verify restored database
            if self.verify_database():
                logging.info("Database verification successful")
                return True
            else:
                logging.error("Database verification failed")
                return False
                
        except Exception as e:
            logging.error(f"Restore failed: {e}")
            return False
    
    def verify_database(self):
        """Verify database integrity"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if main tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('conversations', 'personality_traits', 'user_preferences')
            """)
            
            tables = cursor.fetchall()
            expected_tables = {'conversations', 'personality_traits', 'user_preferences'}
            found_tables = {table[0] for table in tables}
            
            if not expected_tables.issubset(found_tables):
                missing = expected_tables - found_tables
                logging.error(f"Missing tables: {missing}")
                conn.close()
                return False
            
            # Check data integrity
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            
            if result[0] == 'ok':
                return True
            else:
                logging.error(f"Database integrity check failed: {result[0]}")
                return False
                
        except Exception as e:
            logging.error(f"Database verification failed: {e}")
            return False
    
    def optimize_database(self):
        """Optimize database performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            logging.info("Starting database optimization...")
            
            # Vacuum database
            cursor.execute("VACUUM")
            logging.info("Database vacuumed")
            
            # Analyze tables
            cursor.execute("ANALYZE")
            logging.info("Database analyzed")
            
            # Update statistics
            cursor.execute("PRAGMA optimize")
            logging.info("Statistics updated")
            
            conn.close()
            logging.info("Database optimization completed")
            
            return True
            
        except Exception as e:
            logging.error(f"Database optimization failed: {e}")
            return False
    
    def cleanup_old_data(self, days=90):
        """Remove old conversation data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff_date.isoformat()
            
            # Count records to be deleted
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE timestamp < ?", (cutoff_str,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                logging.info("No old conversations to clean up")
                conn.close()
                return True
            
            # Delete old conversations
            cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff_str,))
            
            # Clean up orphaned personality traits
            cursor.execute("""
                DELETE FROM personality_traits 
                WHERE user_id NOT IN (
                    SELECT DISTINCT user_id FROM conversations
                )
            """)
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up {count} old conversation records older than {days} days")
            
            # Optimize after cleanup
            self.optimize_database()
            
            return True
            
        except Exception as e:
            logging.error(f"Data cleanup failed: {e}")
            return False
    
    def get_database_stats(self):
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            stats['total_conversations'] = cursor.fetchone()[0]
            
            # Unique users
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
            stats['unique_users'] = cursor.fetchone()[0]
            
            # Total personality traits
            cursor.execute("SELECT COUNT(*) FROM personality_traits")
            stats['personality_traits'] = cursor.fetchone()[0]
            
            # Database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
            stats['database_size_mb'] = round(db_size, 2)
            
            # Most active users
            cursor.execute("""
                SELECT user_id, COUNT(*) as message_count 
                FROM conversations 
                GROUP BY user_id 
                ORDER BY message_count DESC 
                LIMIT 5
            """)
            stats['most_active_users'] = cursor.fetchall()
            
            # Recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE timestamp > ?", (week_ago,))
            stats['recent_conversations'] = cursor.fetchone()[0]
            
            # Language distribution
            cursor.execute("""
                SELECT language, COUNT(*) as count 
                FROM conversations 
                WHERE language IS NOT NULL 
                GROUP BY language 
                ORDER BY count DESC
            """)
            stats['language_distribution'] = cursor.fetchall()
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
            return None
    
    def export_user_data(self, user_id, output_file=None):
        """Export specific user's data"""
        try:
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"user_export_{user_id}_{timestamp}.json"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user conversations
            cursor.execute("""
                SELECT user_message, bot_response, timestamp, sentiment, language
                FROM conversations 
                WHERE user_id = ?
                ORDER BY timestamp
            """, (user_id,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'user_message': row[0],
                    'bot_response': row[1],
                    'timestamp': row[2],
                    'sentiment': row[3],
                    'language': row[4]
                })
            
            # Get personality traits
            cursor.execute("""
                SELECT trait_type, trait_value, confidence, last_updated
                FROM personality_traits 
                WHERE user_id = ?
            """, (user_id,))
            
            traits = []
            for row in cursor.fetchall():
                traits.append({
                    'trait_type': row[0],
                    'trait_value': row[1],
                    'confidence': row[2],
                    'last_updated': row[3]
                })
            
            # Get preferences
            cursor.execute("""
                SELECT preferred_language, conversation_style, topics_of_interest
                FROM user_preferences 
                WHERE user_id = ?
            """, (user_id,))
            
            pref_row = cursor.fetchone()
            preferences = None
            if pref_row:
                preferences = {
                    'preferred_language': pref_row[0],
                    'conversation_style': pref_row[1],
                    'topics_of_interest': pref_row[2]
                }
            
            conn.close()
            
            # Create export data
            export_data = {
                'user_id': user_id,
                'export_timestamp': datetime.now().isoformat(),
                'total_conversations': len(conversations),
                'conversations': conversations,
                'personality_traits': traits,
                'preferences': preferences
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"User data exported to: {output_file}")
            return output_file
            
        except Exception as e:
            logging.error(f"User data export failed: {e}")
            return None
    
    def list_backups(self):
        """List all available backups"""
        try:
            backup_files = sorted(self.backup_dir.glob('character_memory_*.db*'), reverse=True)
            
            if not backup_files:
                logging.info("No backup files found")
                return []
            
            backups = []
            for backup_file in backup_files:
                stat = backup_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                
                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_mb': round(size_mb, 2),
                    'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'compressed': backup_file.suffix == '.gz'
                })
            
            return backups
            
        except Exception as e:
            logging.error(f"Failed to list backups: {e}")
            return []

def main():
    """Main function dengan command line interface"""
    parser = argparse.ArgumentParser(description='Database Maintenance Tool')
    parser.add_argument('--config', default='config.json', help='Config file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create database backup')
    backup_parser.add_argument('--compress', action='store_true', help='Compress backup file')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup_file', help='Backup file to restore from')
    
    # Optimize command
    subparsers.add_parser('optimize', help='Optimize database')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Days to keep (default: 90)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export user data')
    export_parser.add_argument('user_id', help='User ID to export')
    export_parser.add_argument('--output', help='Output file path')
    
    # List backups command
    subparsers.add_parser('list-backups', help='List available backups')
    
    # Verify command
    subparsers.add_parser('verify', help='Verify database integrity')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize maintenance tool
    maintenance = DatabaseMaintenance(args.config)
    
    # Execute command
    if args.command == 'backup':
        success = maintenance.create_backup(compress=args.compress)
        exit(0 if success else 1)
    
    elif args.command == 'restore':
        success = maintenance.restore_backup(args.backup_file)
        exit(0 if success else 1)
    
    elif args.command == 'optimize':
        success = maintenance.optimize_database()
        exit(0 if success else 1)
    
    elif args.command == 'cleanup':
        success = maintenance.cleanup_old_data(args.days)
        exit(0 if success else 1)
    
    elif args.command == 'stats':
        stats = maintenance.get_database_stats()
        if stats:
            print("\n📊 Database Statistics:")
            print(f"Total conversations: {stats['total_conversations']}")
            print(f"Unique users: {stats['unique_users']}")
            print(f"Personality traits: {stats['personality_traits']}")
            print(f"Database size: {stats['database_size_mb']} MB")
            print(f"Recent conversations (7 days): {stats['recent_conversations']}")
            
            print(f"\n👥 Most active users:")
            for user_id, count in stats['most_active_users']:
                print(f"  {user_id}: {count} messages")
            
            print(f"\n🌍 Language distribution:")
            for lang, count in stats['language_distribution']:
                print(f"  {lang}: {count} messages")
        else:
            exit(1)
    
    elif args.command == 'export':
        output_file = maintenance.export_user_data(args.user_id, args.output)
        exit(0 if output_file else 1)
    
    elif args.command == 'list-backups':
        backups = maintenance.list_backups()
        if backups:
            print("\n💾 Available Backups:")
            for backup in backups:
                compressed = " (compressed)" if backup['compressed'] else ""
                print(f"  {backup['filename']} - {backup['size_mb']} MB - {backup['created']}{compressed}")
        else:
            print("No backups found")
    
    elif args.command == 'verify':
        success = maintenance.verify_database()
        if success:
            print("✅ Database verification successful")
        else:
            print("❌ Database verification failed")
        exit(0 if success else 1)

if __name__ == "__main__":
    main()