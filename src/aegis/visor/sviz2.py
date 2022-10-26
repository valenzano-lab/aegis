# python3 src/visor/sviz.py


from socketserver import DatagramRequestHandler
import tornado.ioloop
import tornado.web

import pathlib
import urllib.parse
import argparse

# import pandas as pd

from aegis.help.container import Container


settings = {"debug": True}

# Data to send
# - survivorship
# - mortality rates
# - reperoduction rates
# -


class MainHandler(tornado.web.RequestHandler):
    def get(self, path):

        basepath = pathlib.Path(".").absolute() / path

        print(basepath)

        container = Container(basepath)

        data = {"cumulative_ages": container.get_json("cumulative_ages")}

        print(type(DatagramRequestHandler))

        self.render("index.html", data=data["cumulative_ages"], name="nomen")


handlers = [(r"/visor/(.*)", MainHandler)]

app = tornado.web.Application(handlers, **settings)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Visualize results from AEGIS",
    )
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    url_path = urllib.parse.quote(args.path, safe="")

    url = "http://localhost:8080/visor/" + url_path

    print(url)
    app.listen(8080)
    tornado.ioloop.IOLoop.instance().start()
