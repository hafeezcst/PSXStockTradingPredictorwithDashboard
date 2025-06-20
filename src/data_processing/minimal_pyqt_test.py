import sys
import os
import logging
from datetime import datetime
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSystemTrayIcon, QMenu, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread
from sqlalchemy import create_engine  # Test SQLAlchemy import

class TestThread(QThread):
    def run(self):
        logging.debug("Thread started")
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=20
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        while True:
            try:
                # Simulate HTTP request
                response = session.get("https://httpbin.org/delay/1", timeout=5)
                if response.status_code == 200:
                    logging.debug("Request successful")
                else:
                    logging.warning(f"Request failed: {response.status_code}")
                    
                # Simulate DB operation
                with engine.connect() as conn:
                    conn.execute("INSERT INTO test_data (value) VALUES (?)",
                                (str(datetime.now()),))
                    
                self.msleep(1000)
                
            except Exception as e:
                logging.error(f"Thread error: {str(e)}")
                self.msleep(5000)  # Wait longer after errors

def main():
    logging.basicConfig(level=logging.DEBUG)
    app = QApplication(sys.argv)
    
    try:
        # Test SQLAlchemy engine with connection pooling
        pool_config = {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
        
        # Create test database file
        db_path = os.path.join(os.getcwd(), 'test.db')
        if os.path.exists(db_path):
            os.remove(db_path)
            
        engine = create_engine(
            f'sqlite:///{db_path}',
            pool_size=pool_config['pool_size'],
            max_overflow=pool_config['max_overflow'],
            pool_timeout=pool_config['pool_timeout'],
            pool_recycle=pool_config['pool_recycle']
        )
        
        # Create a simple table
        with engine.connect() as conn:
            conn.execute("""
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("INSERT INTO test_data (value) VALUES ('test')")
            
        logging.debug("SQLAlchemy engine with file database created")

        # Test multiple thread creation (matching original app's max_threads=8)
        threads = []
        for i in range(8):
            thread = TestThread()
            thread.start()
            threads.append(thread)
            logging.debug(f"Test thread {i} started")

        # Setup cleanup on exit
        def cleanup():
            for thread in threads:
                thread.quit()
                thread.wait()
            engine.dispose()  # Cleanup connection pool
            logging.debug("All threads and engine stopped")
        app.aboutToQuit.connect(cleanup)

        sys.exit(app.exec_())
        
        logging.debug("QApplication created")
        
        window = QMainWindow()
        window.setWindowTitle("PyQt5 with System Tray")
        window.setGeometry(100, 100, 400, 300)
        label = QLabel("Testing System Tray", window)
        label.move(150, 150)
        
        # Add system tray
        tray_icon = QSystemTrayIcon(window)
        tray_icon.setIcon(QIcon.fromTheme("system-run"))
        tray_menu = QMenu()
        show_action = QAction("Show", window)
        tray_menu.addAction(show_action)
        tray_icon.setContextMenu(tray_menu)
        tray_icon.show()
        
        window.show()
        logging.debug("Window with system tray shown")
        
        logging.debug("Starting event loop")
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    main()