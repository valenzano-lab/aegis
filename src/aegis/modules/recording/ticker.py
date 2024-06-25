from multiprocessing import Process
import time
import pathlib
from datetime import datetime


class Ticker:
    def __init__(self, TICKER_RATE, odir: pathlib.Path):
        self.TICKER_RATE = TICKER_RATE
        self.ticker_path = odir / "ticker.txt"
        self.process = None

    def start_process(self):
        self.process = Process(target=self.tick)
        self.process.start()

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
        filetype: txt
        domain: log
        short description:
        long description:
        content: date, time; this file gets updated every TICKER_RATE seconds; useful to determine if the simulation is still running (it is running while the ticker.txt file is updating)
        dtype:
        index:
        header:
        column:
        rows:
        path: /ticker.txt
        """
        with open(self.ticker_path, "w") as file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            file.write(timestamp)

    def read(self):
        with open(self.ticker_path, "r") as file:
            return file.read()

    def has_stopped(self):
        timestamp_recorded = self.read()
        timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        dt_recorded = datetime.strptime(timestamp_recorded, "%Y-%m-%d %H:%M:%S")
        dt_now = datetime.strptime(timestamp_now, "%Y-%m-%d %H:%M:%S")
        time_difference = (dt_now - dt_recorded).total_seconds()
        return time_difference > self.TICKER_RATE
