# python3 src/visor/sviz.py


import tornado.ioloop
import tornado.web

# import pathlib
# import pandas as pd

from aegis.modules.container import Container


settings = {"debug": True}


class MainHandler(tornado.web.RequestHandler):
    def get(self, basepath):

        container = Container(basepath)

        data = {"cumulative_ages": container.get_json("cumulative_ages")}

        self.render("index.html", data=data)


handlers = [(r"/visor/(.*)", MainHandler)]

app = tornado.web.Application(handlers, **settings)

if __name__ == "__main__":
    app.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
