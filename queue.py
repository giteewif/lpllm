import queue
import time
from threading import Thread
import random
def wait_queue(name, q):
    while True:
        time_start = time.time()
        a = q.get(block=True)
        print(f"queue wait cost {time.time() - time_start:.6f} seconds")
        # print(a)
def put_queue(name, q):
    q.put([random.randint(1,100) for _ in range(1024)])
def main():
    q = queue.Queue()
    q.put([random.randint(1,100) for _ in range(1024)])
    thread1=Thread(target=wait_queue, args=("w",q))
    thread2=Thread(target=put_queue, args=("p",q))
    
    thread1.start()
    thread2.start()
    
    thread2.join()
    thread1.join()
    
if __name__ == "__main__":
    main()