import queue


class ObjectPool:
    def __init__(self, factory_method, param, max_size=10):
        self.factory_method = factory_method
        self.max_size = max_size
        self.objects = queue.Queue(maxsize=max_size)
        for i in range(max_size):
            self.objects.put(factory_method(param))

    def acquire(self):
        try:
            return self.objects.get_nowait()
        except queue.Empty:
            return self.factory_method()

    def release(self, obj):
        if self.objects.qsize() < self.max_size:
            self.objects.put(obj)
