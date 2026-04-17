from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from dbx_model_planner.auth import (
    DatabricksCredentials,
    HuggingFaceCredentials,
    KeyringError,
    KeyringNotAvailableError,
    clear_stored_credentials,
    credential_exists,
    delete_credential,
    load_credential,
    save_credential,
)
from dbx_model_planner.auth.wizard import (
    DATABRICKS_CREDENTIAL_NAME,
    HUGGINGFACE_CREDENTIAL_NAME,
    _is_valid_databricks_url,
    _validate_databricks_connection,
    _validate_huggingface_token,
)


class DatabricksCredentialsTests(unittest.TestCase):
    def test_masked_token_short(self) -> None:
        creds = DatabricksCredentials(host="https://example.azuredatabricks.net", token="abc")
        self.assertEqual(creds.masked_token(), "***")

    def test_masked_token_long(self) -> None:
        creds = DatabricksCredentials(host="https://example.azuredatabricks.net", token="dapi1234567890abcdef")
        self.assertEqual(creds.masked_token(), "dapi...cdef")

    def test_host_accessible(self) -> None:
        creds = DatabricksCredentials(host="https://example.azuredatabricks.net", token="token123")
        self.assertEqual(creds.host, "https://example.azuredatabricks.net")


class HuggingFaceCredentialsTests(unittest.TestCase):
    def test_has_token_true(self) -> None:
        creds = HuggingFaceCredentials(token="hf_token123")
        self.assertTrue(creds.has_token)

    def test_has_token_false(self) -> None:
        creds = HuggingFaceCredentials(token=None)
        self.assertFalse(creds.has_token)

    def test_masked_token_none(self) -> None:
        creds = HuggingFaceCredentials(token=None)
        self.assertIsNone(creds.masked_token())

    def test_masked_token_short(self) -> None:
        creds = HuggingFaceCredentials(token="abc")
        self.assertEqual(creds.masked_token(), "***")

    def test_masked_token_long(self) -> None:
        creds = HuggingFaceCredentials(token="hf_1234567890abcdef")
        self.assertEqual(creds.masked_token(), "hf_1...cdef")


class URLValidationTests(unittest.TestCase):
    def test_valid_databricks_url(self) -> None:
        valid_urls = [
            "https://adb-1234567890123456.7.azuredatabricks.net",
            "https://adb-1234567890123456.8.azuredatabricks.net/",
            "http://adb-test.7.azuredatabricks.net",
        ]
        for url in valid_urls:
            self.assertTrue(_is_valid_databricks_url(url), f"Expected {url} to be valid")

    def test_invalid_databricks_url(self) -> None:
        invalid_urls = [
            "not-a-url",
            "adb-123.azuredatabricks.net",
            "https://example.com",
            "https://adb-123.azuredatabricks.net.extra.com",
            "",
        ]
        for url in invalid_urls:
            self.assertFalse(_is_valid_databricks_url(url), f"Expected {url} to be invalid")


class KeyringCredentialTests(unittest.TestCase):
    def test_credential_names_defined(self) -> None:
        self.assertEqual(DATABRICKS_CREDENTIAL_NAME, "databricks")
        self.assertEqual(HUGGINGFACE_CREDENTIAL_NAME, "huggingface")

    @patch("dbx_model_planner.auth.keyring.keyring")
    def test_save_and_load_credential(self, mock_keyring: MagicMock) -> None:
        mock_keyring.get_password.return_value = '{"host": "https://example.azuredatabricks.net", "token": "test-token"}'
        
        save_credential("test-cred", {"host": "https://example.azuredatabricks.net", "token": "test-token"})
        mock_keyring.set_password.assert_called_once()
        
        result = load_credential("test-cred")
        self.assertIsNotNone(result)
        self.assertEqual(result["host"], "https://example.azuredatabricks.net")

    @patch("dbx_model_planner.auth.keyring.keyring")
    def test_load_nonexistent_credential(self, mock_keyring: MagicMock) -> None:
        mock_keyring.get_password.return_value = None
        
        result = load_credential("nonexistent")
        self.assertIsNone(result)

    @patch("dbx_model_planner.auth.keyring.keyring")
    def test_delete_credential(self, mock_keyring: MagicMock) -> None:
        mock_keyring.get_password.return_value = '{"key": "value"}'
        
        result = delete_credential("test-cred")
        mock_keyring.delete_password.assert_called_once()
        self.assertTrue(result)

    @patch("dbx_model_planner.auth.keyring.keyring")
    def test_delete_nonexistent_credential(self, mock_keyring: MagicMock) -> None:
        mock_keyring.get_password.return_value = None
        
        result = delete_credential("nonexistent")
        self.assertFalse(result)

    @patch("dbx_model_planner.auth.keyring.keyring")
    def test_credential_exists(self, mock_keyring: MagicMock) -> None:
        mock_keyring.get_password.return_value = '{"key": "value"}'
        self.assertTrue(credential_exists("test-cred"))
        
        mock_keyring.get_password.return_value = None
        self.assertFalse(credential_exists("nonexistent"))


class ClearStoredCredentialsTests(unittest.TestCase):
    @patch("dbx_model_planner.auth.wizard.delete_credential")
    @patch("dbx_model_planner.auth.wizard.credential_exists")
    def test_clear_with_confirm(
        self,
        mock_exists: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        mock_exists.side_effect = [True, True, False]
        outputs: list[str] = []
        
        def input_fn(_: str) -> str:
            return "y"
        
        def output_fn(msg: str) -> None:
            outputs.append(msg)
        
        clear_stored_credentials(input_fn=input_fn, output_fn=output_fn)
        
        self.assertEqual(mock_delete.call_count, 2)
        self.assertIn("All credentials cleared.", outputs)

    @patch("dbx_model_planner.auth.wizard.delete_credential")
    @patch("dbx_model_planner.auth.wizard.credential_exists")
    def test_clear_with_cancel(
        self,
        mock_exists: MagicMock,
        mock_delete: MagicMock,
    ) -> None:
        mock_exists.return_value = True
        outputs: list[str] = []
        
        def input_fn(_: str) -> str:
            return "n"
        
        def output_fn(msg: str) -> None:
            outputs.append(msg)
        
        clear_stored_credentials(input_fn=input_fn, output_fn=output_fn)
        
        mock_delete.assert_not_called()
        self.assertIn("Cancelled.", outputs)


if __name__ == "__main__":
    unittest.main()
