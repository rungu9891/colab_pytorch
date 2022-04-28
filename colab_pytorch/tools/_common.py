from abc import abstractmethod
from pathlib import Path
from subprocess import Popen


class CommonTool:
    def __init__(self, server_port, prevent_interrupt=True, api_port=40000, gpu=True, verbose=False, worker_name=None):
        self.server_port = server_port
        self.prevent_interrupt = prevent_interrupt
        self.api_port = api_port
        self.gpu = gpu
        self.verbose = verbose
        self.worker_name = worker_name

    @abstractmethod
    def download(self, homepath: Path):
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run_tool(self) -> Popen:
        raise NotImplementedError

    @abstractmethod
    def wait_tool(self):
        raise NotImplementedError
