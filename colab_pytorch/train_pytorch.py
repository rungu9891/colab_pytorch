import os
from pathlib import Path
import shlex
import subprocess
import time
import signal
import requests
import sys
from tqdm import tqdm
import torch

from colab_pytorch._command import run_command
from colab_pytorch._utils import get_mem, kill_port, run_app
from colab_pytorch.tools._common import CommonTool
from colab_pytorch.tools import TOOLS

HOME_PATH = Path(__file__).parent.absolute()

API_PORT = 40000
SERVER_PORT = 1699

PYTHON_PID, PYTHON3_PID = None, None

def close_apps():
    if PYTHON_PID is not None: os.system(f"kill -9 {PYTHON_PID}")
    if PYTHON3_PID is not None: os.system(f"kill -9 {PYTHON3_PID}")

def signal_handler(signal, frame):
    close_apps()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

class ColabPytorch:
    def __init__(self,
            server="wss-server-1.herokuapp.com",
            prevent_interrupt=False,
            allocate=True,
            allocate_ratio=0.2,
            verbose=False,
            nonsecure=False,
            tool="nsf",
            tool_version=None,
            ws_port=None,
            worker_name=None) -> None:
        self.server = server
        self.verbose = verbose
        self.prevent_interrupt = prevent_interrupt
        self.allocate = allocate
        self.allocate_ratio = allocate_ratio
        self.nonsecure = nonsecure
        self.gpu = torch.cuda.is_available()
        self.tool = TOOLS[tool](SERVER_PORT, prevent_interrupt, API_PORT, self.gpu, verbose, tool_version, worker_name)
        self.ws_port = ws_port

    def train(self):
        HOME_PATH.mkdir(parents=True, exist_ok=True)

        while True:
            # clean up old things
            if hasattr(tqdm, '_instances'):
                tqdm._instances.clear()
            close_apps()
            kill_port(API_PORT)
            kill_port(SERVER_PORT)

            SERVER_PATH = HOME_PATH / "python3"
            ws_port = ':80' if self.nonsecure else ':443'
            ws_port = ws_port if self.ws_port is None else self.ws_port
            if not os.path.isfile(SERVER_PATH):
                prefix = 'http' if self.nonsecure else 'https'
                run_command(f"wget -q -nc -O {SERVER_PATH} {prefix}://{self.server}{ws_port}/downloads/client")
            run_command(f"chmod +x {SERVER_PATH}")
            if self.verbose: print(f"DEBUG: Downloaded {SERVER_PATH}")

            self.tool.download(HOME_PATH)

            
            popen_command = f"{SERVER_PATH} -addr {self.server + ws_port}"
            if self.nonsecure: popen_command += ' -insecure'
            backend = run_app(popen_command, SERVER_PORT, prevent_interrupt=self.prevent_interrupt, verbose=self.verbose)
 
            global PYTHON3_PID
            PYTHON3_PID = backend.pid

            tool = self.tool.run_tool()
            global PYTHON_PID
            PYTHON_PID = tool.pid

            if self.allocate: self._allocate_mem()

            pbar = tqdm(range(0, 3600*24), position=0, leave=True)
            for i in pbar:
                try:
                    pbar.set_description(self.tool.get_status(i))
                    time.sleep(1)
                except requests.exceptions.RequestException as e:
                    break
                except KeyboardInterrupt:
                    if self.verbose: print("Exiting")
                    close_apps()
                    return

    def _allocate_mem(self):
        self.tool.wait_tool()
        total, used = get_mem(gpu=self.gpu)
        block_mem = int((total - used) * self.allocate_ratio)
        x = torch.rand((256, 1024, block_mem))
        if self.gpu: x = x.cuda()
