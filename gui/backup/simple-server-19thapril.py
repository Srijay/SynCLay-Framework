#!/usr/bin/env python3
import json
import os,sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

sys.path.insert(0,os.getcwd())

from collect_model_nuclei import get_model, json_to_img
#from collect_model_glands import get_model, json_to_img

model = get_model()

class StaticServer(BaseHTTPRequestHandler):

    def do_GET(self):
        root = os.path.dirname(os.path.realpath(__file__))
        path = unquote(self.path)
        if path == '/':
            filename = root + '/index.html'
        elif path.startswith('/get_data?'):
            image_path, mask_path = json_to_img(path.split('/get_data?data=')[1], model)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(str.encode(json.dumps({'img_pred': image_path,'layout_pred': mask_path})))
            return
        else:
            filename = root + self.path

        self.send_response(200)
        if filename[-4:] == '.css':
            self.send_header('Content-type', 'text/css')
        elif filename[-5:] == '.json':
            self.send_header('Content-type', 'application/javascript')
        elif filename[-3:] == '.js':
            self.send_header('Content-type', 'application/javascript')
        elif filename[-4:] == '.ico':
            return
            # self.send_header('Content-type', 'image/x-icon')
        else:
            self.send_header('Content-type', 'text/html')
        self.end_headers()
        with open(filename, 'rb') as fh:
            html = fh.read()
            # html = bytes(html, 'utf8')
            self.wfile.write(html)


def run(server_class=HTTPServer, handler_class=StaticServer, port=8080):
    print('Loading model')
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting https on port {}'.format(port))
    httpd.serve_forever()


run()
