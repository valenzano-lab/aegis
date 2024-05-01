from multiprocessing import Process
import time


class Ticker:
    def __init__(self, TICKER_RATE, output_directory):
        self.TICKER_RATE = TICKER_RATE
        self.process = Process(target=self.tick)
        self.ticker_path = output_directory / "ticker.txt"

    def start_process(self):
        self.process.start()

    def stop_process(self):
        self.process.terminate()
        self.process.join()

    def tick(self):
        while True:
            print("tick")
            with open(self.ticker_path, "w") as file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                file.write(timestamp)
            time.sleep(self.TICKER_RATE)
