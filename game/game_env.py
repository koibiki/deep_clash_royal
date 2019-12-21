import ctypes
import os
import platform
import threading

if "Windows" in platform.platform():
    os.environ["path"] = \
        "F:\\opencv-4.1.2\\build\\install;F:\\opencv-4.1.2\\build\\install\\x64\\mingw\\bin;" \
        "F:\\opencv-4.1.2\\build\\install\\x64\\mingw\\lib;C:\\Windows\\system32;C:\\Windows;" \
        "C:\\Windows\\System32\\Wbem;C:\\Windows\\System32\\WindowsPowerShell\\v1.0;" \
        "C:\\Windows\\System32\\OpenSSH;C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common;" \
        "$HAVA_HOME\\bin;C:\\Program Files\\Git\\cmd;C:\\mingw64\\bin;F:\\adb;C:\\Program Files\\CMake\\bin;" \
        "D:\\Anaconda3;C:\\Users\\orient\\AppData\\Local\\Microsoft\\WindowsApps;C:\\Program Files (x86)\\CMake\\bin"

import sys
from utils.c_lib_utils import Result, convert2pymat

"""
仅支持 1920 * 1080 分辨率的屏幕
"""


class ClashRoyalEnv:
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(ClashRoyalEnv, "_instance"):
            with ClashRoyalEnv._instance_lock:
                if not hasattr(ClashRoyalEnv, "_instance"):
                    ClashRoyalEnv._instance = object.__new__(cls)
        return ClashRoyalEnv._instance

    def __init__(self, ):
        super().__init__()
        if sys.platform == 'win32':
            lib_path = "F:\\\\PyCharmProjects\\\\deep_clash_royal\\\\lib\\\\libc_opencv.dll"
        else:
            lib_path = "./lib/libc_opencv.so"

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.lib.detect_frame.restype = Result

    def init_game(self, game_id, agent_id):
        self.lib.init_game(game_id, agent_id)

    def detect_frame(self, img, agent_id):
        result = Result()
        pymat = convert2pymat(img)
        result = self.lib.detect_frame(pymat, result, agent_id)
        return result


if __name__ == '__main__':
    royal1 = ClashRoyalEnv()
    royal2 = ClashRoyalEnv()
    print(royal1)
    print(royal2)
