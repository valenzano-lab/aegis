# test_app.py
import unittest
from aegis_gui import app as app_
from aegis_gui.guisettings.GuiSettings import gui_settings


class TestDashApp(unittest.TestCase):
    def setUp(self):

        gui_settings.set(environment="local", debug=False)

        app = app_.get_app()

        # Set up the test client
        self.client = app.server.test_client()
        self.app_context = app.server.app_context()
        self.app_context.push()

    def tearDown(self):
        # Clean up after tests
        self.app_context.pop()

    def test_app_runs(self):
        # Send a request to the root URL
        response = self.client.get("/aegis/")

        # Check that the response status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Optionally, check that the response contains expected content
        self.assertIn(b"AEGIS", response.data)


if __name__ == "__main__":
    unittest.main()
