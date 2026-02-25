import unittest
from aegis_gui import app as app_
from aegis_gui.guisettings.GuiSettings import gui_settings


class TestDashApp(unittest.TestCase):
    def setUp(self):
        gui_settings.set(environment="local", debug=False)
        app = app_.get_app()
        self.client = app.server.test_client()
        self.app_context = app.server.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_app_runs(self):
        response = self.client.get("/aegis/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AEGIS", response.data)

    # def test_404_response(self):
    #     # Send a request to a non-existent route
    #     response = self.client.get("/aegis/non-existent-route")
    #     # Check that the response status code is 404 (Not Found)
    #     self.assertEqual(response.status_code, 404)
    #     # Optionally, check for specific content in your 404 page
    #     # self.assertIn(b"Page not found", response.data)


if __name__ == "__main__":
    unittest.main()
