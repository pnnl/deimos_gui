import time

import panel as pn
from run_app import app1
from playwright.sync_api import expect

CLICKS = 2

def test_component(page, port):

    # Given
    url = f"http://localhost:{port}"
    # When
    server = pn.serve(app1, port=port, threaded=True, show=False)
    time.sleep(0.2)
    page.goto(url)   
    server.stop()


