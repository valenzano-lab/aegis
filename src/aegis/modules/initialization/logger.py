# import numpy as np
# from aegis.modules.recording import recorder
# import logging


# class Logger:
#     def __init__(self, output_path):
#         self.output_path = output_path

#     def init_progress_log(self):
#         # Set up progress log
#         progress_path = self.output_path / "progress.log"
#         content = ("stage", "ETA", "t1M", "runtime", "stg/min", "popsize")
#         with open(progress_path, "wb") as f:
#             np.savetxt(f, [content], fmt="%-10s", delimiter="| ")

#     # def log_before_simulation(self):
#     #     recorder.record_input_summary()

#     # def log_after_siulation(self):
#     #     recorder.record_output_summary()
#     #     logging.info("Simulation finished")
