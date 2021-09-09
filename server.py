from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
import json
# import urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()

        message = "Hello, World!!"
        path, _, query_string = self.path.partition('?')
        query = parse_qs(query_string)
        details = query["details"][0]
        print(details)
        detailsJSON = json.loads(details)

        print(type(query))
        print(u"[START]: Received GET for %s with query: %s" % (path, detailsJSON))
        self.wfile.write(bytes(message, "utf8"))

with HTTPServer(('', 7800), handler) as server:
    server.serve_forever()