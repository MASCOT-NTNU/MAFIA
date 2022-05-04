import time
from multiprocessing import Pool


def run(fn):
    # fn: 函数参数是数据列表的一个元素
    time.sleep(.3)
    # print(fn * fn)
    return fn * fn


if __name__ == "__main__":
    testFL = [1, 2, 3, 4, 5, 6, 7, 8]
    print('shunxu:')  # 顺序执行(也就是串行执行，单进程)
    s = time.time()
    for fn in testFL:
        print(run(fn))
    t1 = time.time()
    print("顺序执行时间：", int(t1 - s))

    print('concurrent:')  # 创建多个进程，并行执行
    pool = Pool(1)  # 创建拥有3个进程数量的进程池
    # testFL:要处理的数据列表，run：处理testFL列表中数据的函数
    res = [pool.apply_async(run, [i]) for i in testFL]
    # pool.close()  # 关闭进程池，不再接受新的进程
    # pool.join()  # 主进程阻塞等待子进程的退出
    t2 = time.time()
    for r in res:
        print(r.get())
    # print(res.get())
    print("并行执行时间：", int(t2 - t1))
