import multiprocessing as mp
import time


def foo_pool(x):
    time.sleep(3)
    return x * x


result_list = []


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def apply_async_with_callback():
    pool = mp.Pool()
    for i in range(10):
        print(i)
        pool.apply_async(foo_pool, args=(i,), callback=log_result)
    pool.close()
    pool.join()
    print(result_list)

if __name__ == '__main__':
    apply_async_with_callback()

#%%
import multiprocessing as mp

def print_cube(num):
    print("PID:", mp.current_process)
    print("CUB: ", num * num * num)

def print_square(num):
    print("PID:", mp.current_process)
    print("Square: ", num * num)

if __name__ == "__main__":
    p1 = mp.Process(print_cube(5))
    p2 = mp.Process(print_square(5))

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(p1.is_alive())
    p2.is_alive()
    # p1.close()
    # p2.close()
    p1.terminate()
    p2.terminate()
    # p1.run()
    # p2.run()
    # p1.kill()
    # p2.kill()