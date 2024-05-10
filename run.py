import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            print(f"Detected changes in {event.src_path}. Running Manim...")
            # Change 'your_manim_script.py' to the name of your Manim script file
            # Change 'output_directory' to your desired output directory for animations
            subprocess.run(["manim", "-pqh", event.src_path, "output_directory"])

if __name__ == "__main__":
    path = '.'  # the directory to watch, you can change to your specific path
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
