import time

from threading import Thread, Lock


class Listener:
    task_queue = []
    lock = Lock()
    thread = None
    
    @classmethod
    def _process_tasks(cls):
        while True:
            task = None
            with cls.lock:
                if cls.task_queue:
                    task = cls.task_queue.pop(0)
                    
            if task is None:
                time.sleep(0.001)
                continue
                
            func, args, kwargs = task
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in listener thread: {e}")
    
    @classmethod
    def add_task(cls, func, *args, **kwargs):
        with cls.lock:
            cls.task_queue.append((func, args, kwargs))
            if cls.thread is None:
                cls.thread = Thread(target=cls._process_tasks, daemon=True)
                cls.thread.start()


def async_run(func, *args, **kwargs):
    Listener.add_task(func, *args, **kwargs)


class FIFOQueue:
    def __init__(self):
        self.queue = []
        self.lock = Lock()

    def push(self, item):
        with self.lock:
            self.queue.append(item)

    def pop(self):
        with self.lock:
            if self.queue:
                return self.queue.pop(0)
            return None

    def top(self):
        with self.lock:
            if self.queue:
                return self.queue[0]
            return None

    def next(self):
        while True:
            with self.lock:
                if self.queue:
                    return self.queue.pop(0)

            time.sleep(0.001)


class AsyncStream:
    def __init__(self):
        self.input_queue = FIFOQueue()
        self.output_queue = FIFOQueue()
