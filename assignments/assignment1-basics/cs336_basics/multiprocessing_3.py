import time
import multiprocessing as mp

def task():
    result = 0
    for _ in range(10**8):
        result += 1
    return result

if __name__ == "__main__":
    start= time.perf_counter()

    p1 = mp.Process(target=task)
    p2 = mp.Process(target=task)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
    # took 1.61 sec
    # task()
    # task()#3.05 sec
    finish = time.perf_counter()
    print(f"Task took {finish - start} seconds")


