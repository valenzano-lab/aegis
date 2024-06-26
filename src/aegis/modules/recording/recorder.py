class Recorder:
    def init_odir(self):
        self.init_dir(self.odir)

    @staticmethod
    def init_dir(path):
        path.mkdir(exist_ok=True, parents=True)
