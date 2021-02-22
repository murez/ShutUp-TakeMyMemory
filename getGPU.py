import pynvml
import os
import argparse
import time
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Get Your GPU InTime')
parser.add_argument('--code', type=str, default = None, required=False, help=r'Write your exec code like "python train.py".')
parser.add_argument('--memory', type=int, default = 8000, required=False, help=r'The memory you need, 8000 by default.')
parser.add_argument('--get-all-resource', action='store_true', help=r'Just to get all resources without running anything or the code.')
ratio = 1024**2
args = parser.parse_args()

if __name__ == '__main__':
    
    pynvml.nvmlInit()
    while 1:
        deviceNum = pynvml.nvmlDeviceGetCount()
        for i in range(deviceNum):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total/ratio
            used = meminfo.used/ratio
            free = meminfo.free/ratio
            if free >= args.memory:
                if args.get_all_resource:
                    block_mem = int(float(free) * 0.9)
                    x = torch.cuda.FloatTensor(256,1024,block_mem)
                    del x
                    for _ in tqdm(range(60)):
                        time.sleep(1)
            
                localtime = time.asctime(time.localtime(time.time()))
                print(f"your code started at {localtime}")
                os.environ["CUDA_DEVICE_ORDER"]= f"{i}"
                os.system(args.code)
                localtime = time.asctime(time.localtime(time.time()))
                print(f"done at:{localtime}" )
                exit(0)
