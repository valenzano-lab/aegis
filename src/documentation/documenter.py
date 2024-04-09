import pathlib


class Documenter:

    here = pathlib.Path(__file__).absolute().parent

    def __init__(self):
        pass

    @staticmethod
    def read(filename):
        with open(Documenter.here / filename, "r") as file_:
            md = file_.read()
        return md
