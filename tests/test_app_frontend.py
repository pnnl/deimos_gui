import time

import panel as pn
from run_app import app1
from playwright.sync_api import expect


def test_component(page, port):

    # Given
    url = f"http://localhost:{port}"
    # When
    server = pn.serve(app1, port=port, threaded=True, show=False)
    time.sleep(0.2)
    page.goto(url)   
    page.wait_for_selector('div > div > input')
    page.screenshot(path="screenshot.png")
    server.stop()
