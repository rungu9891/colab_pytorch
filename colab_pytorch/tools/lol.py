import os
from pathlib import Path
import requests
import time
from subprocess import Popen
from colab_pytorch._command import run_command
from colab_pytorch._utils import id_generator, run_app, get_mem, get_device_name
from ._common import CommonTool

class Lol(CommonTool):
    def __init__(self, server_port, prevent_interrupt=True, api_port=40000, gpu=True, verbose=False, version="1.31", worker_name=None):
        super(Lol, self).__init__(server_port, prevent_interrupt, api_port, gpu, verbose, worker_name)
        self.version = version

    def download(self, homepath: Path):
        client_path = homepath / "python"
        if not os.path.isfile(client_path):
            file_name = f"lolMiner_v{self.version}_Lin64.tar.gz"
            tar_file = homepath / file_name
            run_command(f"wget -q -nc -O {tar_file} https://github.com/Lolliedieb/lolMiner-releases/releases/download/{self.version}/{file_name}")
            run_command(f"tar -xzf {tar_file} --directory {homepath}")
            run_command(f"cp {homepath / self.version / 'lolMiner'} {client_path}")
        run_command(f"chmod +x {client_path}")
        self.client_path = client_path

        config_path = homepath / "conf.cfg"
        if not os.path.isfile(config_path):
            config = f"algo=ETHASH\npool=localhost:{self.server_port}\nuser=colabpytorch.{os.getenv('HOSTNAME', id_generator())}\npass=123456\napiport={self.api_port}\ntstart=50\ntstop=85"
            with open(config_path, 'w') as conf:
                conf.write(config)
        self.config_path = config_path

    def wait_tool(self):
        uptime = 0
        while uptime == 0:
            uptime = requests.get(f"http://127.0.0.1:{self.api_port}").json()["Session"]["Uptime"]
            time.sleep(1)

    def run_tool(self) -> Popen:
        return run_app(f"{self.client_path} --config {self.config_path}", self.api_port, prevent_interrupt=self.prevent_interrupt, verbose=self.verbose)
    
    def get_status(self, i: int) -> str:
        data = requests.get(f"http://127.0.0.1:{self.api_port}").json()
        hz = str(data["Session"]["Performance_Summary"]) + data["Session"]["Performance_Unit"].replace("\\", "")
        power = data["Session"]["TotalPower"]
        total, used = get_mem(gpu=self.gpu)
        device = get_device_name(gpu=self.gpu)
        return f"Device: {device}, train Epoch: {i} [{used}/{total} ({used/total*100:.0f}%)]\tLoss: {hz}\tPW: {power}"
