from http.server import HTTPServer, SimpleHTTPRequestHandler

# import BaseHTTPServer
# import SimpleHTTPServer

server_address = ("", 8888)
PUBLIC_RESOURCE_PREFIX = "/public"
PUBLIC_DIRECTORY = "/path/to/protected/public"


class MyRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if self.path.startswith(PUBLIC_RESOURCE_PREFIX):
            if (
                self.path == PUBLIC_RESOURCE_PREFIX
                or self.path == PUBLIC_RESOURCE_PREFIX + "/"
            ):
                return PUBLIC_DIRECTORY + "/index.html"
            else:
                return PUBLIC_DIRECTORY + path[len(PUBLIC_RESOURCE_PREFIX) :]
        else:
            return SimpleHTTPRequestHandler.translate_path(self, path)


httpd = HTTPServer(server_address, MyRequestHandler)
httpd.serve_forever()
