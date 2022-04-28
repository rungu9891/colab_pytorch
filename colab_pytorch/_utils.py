import socket
from contextlib import closing
import os
import psutil
import string
import random
import shlex
import subprocess
import time
import signal
import cpuinfo
from colab_pytorch._command import run_command

def check_socket(port, host="127.0.0.1") -> str:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            return f"{host}:{port} is opened"
        else:
            raise Exception(f"{host}:{port} is not open")

def get_mem_gpu(idx):
    output = os.popen(f'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read()
    data = [gpu.split(",") for gpu in output.splitlines()]
    data = [(int(total), int(used)) for (total, used) in data]
    return data[idx]

def get_mem_cpu():
    return int(psutil.virtual_memory().total/1024/1024), int(psutil.virtual_memory().used/1024/1024)

def get_mem(idx=0, gpu=True):
    if gpu: return get_mem_gpu(idx)
    else: return get_mem_cpu()

def kill_port(port):
    run_command(f"kill $(lsof -t -i:{port})")

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def get_gpu_device():
    output = os.popen(f'nvidia-smi --query-gpu=gpu_name --format=csv,nounits,noheader').read()
    return output.strip()

def get_cpu_device():
    return cpuinfo.get_cpu_info()['brand_raw']

def get_device_name(gpu=True):
    if gpu: return get_gpu_device()
    else: return get_cpu_device()

def run_app(popen_command, port, prevent_interrupt=False, verbose=False):
    preexec_fn = None
    app = shlex.split(popen_command)[0]
    if prevent_interrupt:
        popen_command = 'nohup ' + popen_command
        preexec_fn = os.setpgrp
    popen_command = shlex.split(popen_command)
    # Initial sleep time
    sleep_time = 2.0

    # Create tunnel and retry if failed
    stdout = subprocess.PIPE if verbose else subprocess.DEVNULL
    for _ in range(10):
        proc = subprocess.Popen(popen_command, stdout=stdout, preexec_fn=preexec_fn)
        if verbose:
            print(f"DEBUG: {app} process: PID={proc.pid}")
        time.sleep(sleep_time)
        try:
            info = check_socket(port)
            break
        except Exception as e:
            os.kill(proc.pid, signal.SIGKILL)
            kill_port(port)
            if verbose:
                print(f"DEBUG: Exception: {e.args[0]}")
                print(f"DEBUG: Killing {proc.pid}. Retrying...")
        # Increase the sleep time and try again
        sleep_time *= 1.5

    if verbose:
        print("DEBUG:", info)

    return proc
