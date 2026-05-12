import unittest

from pydantic import ValidationError

from tts_api.catalog import (
    UnsupportedLanguageError,
    UnsupportedSpeakerError,
    build_default_catalog,
)
from tts_api.schemas import TTSRequest, VoiceCloneRequest


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = build_default_catalog()

    def test_valid_language_and_speaker(self) -> None:
        speaker = self.catalog.get_speaker("hi", "Divya")
        self.assertEqual(speaker.name, "Divya")

    def test_invalid_speaker_for_language_fails(self) -> None:
        with self.assertRaises(UnsupportedSpeakerError):
            self.catalog.get_speaker("hi", "Mary")

    def test_unsupported_language_fails(self) -> None:
        with self.assertRaises(UnsupportedLanguageError):
            self.catalog.get_language("fr")


class SchemaTests(unittest.TestCase):
    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TTSRequest(text="", language="hi", speaker="Divya")

    def test_whitespace_only_text_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            TTSRequest(text="   ", language="hi", speaker="Divya")

    def test_language_is_normalized(self) -> None:
        request = TTSRequest(text="Hello", language=" EN ", speaker=" Mary ")
        self.assertEqual(request.language, "en")
        self.assertEqual(request.speaker, "Mary")

    def test_description_mode_accepts_custom_description(self) -> None:
        request = TTSRequest(
            text="Hello",
            language="EN",
            voice_mode="description",
            voice_description="  warm and steady voice  ",
        )
        self.assertEqual(request.voice_mode, "description")
        self.assertEqual(request.voice_description, "warm and steady voice")

    def test_speaker_mode_requires_speaker(self) -> None:
        with self.assertRaises(ValidationError):
            TTSRequest(text="Hello", language="hi", voice_mode="speaker")

    def test_description_mode_requires_description(self) -> None:
        with self.assertRaises(ValidationError):
            TTSRequest(text="Hello", language="hi", voice_mode="description")

    def test_clone_request_normalizes_language(self) -> None:
        request = VoiceCloneRequest(text="Hello", language=" EN ")
        self.assertEqual(request.language, "en")

    def test_clone_request_requires_text(self) -> None:
        with self.assertRaises(ValidationError):
            VoiceCloneRequest(text="   ", language="hi")
