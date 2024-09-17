from multiprocessing import Process
import time
import pathlib
from datetime import datetime
from .recorder import Recorder
import logging


class Ticker(Recorder):
    def __init__(self, TICKER_RATE, odir: pathlib.Path):
        self.TICKER_RATE = TICKER_RATE
        self.ticker_path = odir / "ticker.txt"
        self.process = None
        self.pid = None

    def start_process(self):
        self.process = Process(target=self.tick)
        self.process.start()
        self.pid = self.process.pid
        # TODO check if this continues when simulation is interrupted or terminated otherwise

    def stop_process(self):
        self.process.terminate()
        self.process.join()

    def tick(self):
        while True:
            self.write()
            time.sleep(self.TICKER_RATE)

    def write(self):
        """
        # OUTPUT SPECIFICATION
        path: /ticker.txt
        filetype: txt
        category: log
        description: A live file useful for determining whether the simulation is still running. It gets updated every TICKER_RATE seconds; if it is not updated, the simulation is not running.
        trait granularity: N/A
        time granularity: N/A
        frequency parameter: TICKER_RATE
        structure: A txt file with datetime stamp (%Y-%m-%d %H:%M:%S) in one line)
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        with open(self.ticker_path, "w") as file:
            file.write(timestamp)

    def read(self):
        if not self.ticker_path.exists():
            logging.error(f"{self.ticker_path} does not exist.")
            return
        with open(self.ticker_path, "r") as file:
            return file.read()

    def has_stopped(self):
        since_last = self.since_last()
        return since_last > self.TICKER_RATE

    def since_last(self):
        timestamp_recorded = self.read()
        if timestamp_recorded is None or timestamp_recorded == "":
            logging.info(f"timestamp_recorded is '{timestamp_recorded}'")
            return
        timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dt_recorded = datetime.strptime(timestamp_recorded, "%Y-%m-%d %H:%M:%S")
        dt_now = datetime.strptime(timestamp_now, "%Y-%m-%d %H:%M:%S")
        time_difference = (dt_now - dt_recorded).total_seconds()
        return time_difference
