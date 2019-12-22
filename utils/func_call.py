from functools import wraps

import time


def func_time(f):
    """
    简单记录执行时间
    :param f:
    :return:
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, 'took', end - start, 'seconds')
        return result

    return wrapper

@func_time
def spent_time(n):
    s = 0
    for i in range(n):
        for ii in range(n):
            s += ii
    print(s)

spent_time(10000)