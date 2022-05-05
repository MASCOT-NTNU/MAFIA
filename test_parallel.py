import time
from multiprocessing import Pool
import numpy as np

def run(fn):
    # fn: 函数参数是数据列表的一个元素
    time.sleep(.2)
    # print(fn * fn)
    return fn * fn

class mafia:

    def __init__(self):
        print("here comes mafia")

    def run(self):
        counter = 0

        while True:
            if counter % 100 == 0:
                print("Arrive!")

            time.sleep(.1)
            counter += 1


if __name__ == "__main__":
    testFL = np.arange(10)
    print('shunxu:')  # 顺序执行(也就是串行执行，单进程)
    s = time.time()
    rest = 0
    for fn in testFL:
        rest += run(fn)
    t1 = time.time()
    print("顺序执行时间：", t1 - s)
    print("res: ", rest)

    print('concurrent:')  # 创建多个进程，并行执行
    pool = Pool(1)  # 创建拥有3个进程数量的进程池
    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    res = [pool.apply_async(run, [i]) for i in testFL]
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出
    t2 = time.time()
    rest = 0
    for r in res:
        rest += r.get()
    print("并行执行时间：", t2 - t1)
    print("res: ", rest)

