from dataclasses import dataclass
from typing import Iterable


class CatalogValidationError(ValueError):
    """Raised when a request references unsupported catalog data."""


class UnsupportedLanguageError(CatalogValidationError):
    """Raised when the requested language is not available."""


class UnsupportedSpeakerError(CatalogValidationError):
    """Raised when the requested speaker is not available for a language."""


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    description: str
    recommended: bool


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    display_name: str
    speakers: tuple[SpeakerProfile, ...]


class LanguageCatalog:
    def __init__(self, languages: Iterable[LanguageProfile]) -> None:
        self._languages = tuple(languages)
        self._by_code = {language.code: language for language in self._languages}

    def list_languages(self) -> tuple[LanguageProfile, ...]:
        return self._languages

    def get_language(self, language_code: str) -> LanguageProfile:
        normalized_code = language_code.strip().lower()
        language = self._by_code.get(normalized_code)
        if language is None:
            raise UnsupportedLanguageError(
                f"Unsupported language '{language_code}'."
            )
        return language

    def get_speaker(self, language_code: str, speaker_name: str) -> SpeakerProfile:
        language = self.get_language(language_code)
        normalized_name = speaker_name.strip().casefold()
        for speaker in language.speakers:
            if speaker.name.casefold() == normalized_name:
                return speaker
        raise UnsupportedSpeakerError(
            f"Unsupported speaker '{speaker_name}' for language '{language.display_name}'."
        )

    def as_template_data(self) -> list[dict[str, object]]:
        return [
            {
                "code": language.code,
                "display_name": language.display_name,
                "speakers": [
                    {
                        "name": speaker.name,
                        "recommended": speaker.recommended,
                    }
                    for speaker in language.speakers
                ],
                "recommended_speakers": [
                    speaker.name for speaker in language.speakers if speaker.recommended
                ],
            }
            for language in self._languages
        ]


VOICE_TEXTURES = (
    "a clean studio recording with almost no background noise",
    "a clear near-field recording with steady loudness and minimal room echo",
    "a polished broadcast-style recording with crisp articulation and low noise",
    "a balanced indoor recording with natural resonance and soft ambience control",
)

VOICE_PACES = (
    "slightly brisk",
    "steady",
    "measured",
    "calm",
)

VOICE_TONES = (
    "warm and expressive",
    "confident and conversational",
    "gentle and composed",
    "bright and engaging",
    "grounded and natural",
)


def _build_description(name: str, language_name: str, index: int) -> str:
    tone = VOICE_TONES[index % len(VOICE_TONES)]
    pace = VOICE_PACES[index % len(VOICE_PACES)]
    texture = VOICE_TEXTURES[index % len(VOICE_TEXTURES)]
    return (
        f"{name}'s voice sounds {tone}, with a {pace} speaking pace and "
        f"{texture}. The delivery is natural for {language_name} speech, "
        "with clear pronunciation and smooth sentence flow."
    )


def _speaker(
    name: str,
    language_name: str,
    index: int,
    recommended_names: set[str],
) -> SpeakerProfile:
    return SpeakerProfile(
        name=name,
        description=_build_description(name, language_name, index),
        recommended=name in recommended_names,
    )


def _language(
    code: str,
    display_name: str,
    speaker_names: list[str],
    recommended_names: list[str],
) -> LanguageProfile:
    recommended_set = set(recommended_names)
    speakers = tuple(
        _speaker(name, display_name, index, recommended_set)
        for index, name in enumerate(speaker_names)
    )
    return LanguageProfile(code=code, display_name=display_name, speakers=speakers)


def build_default_catalog() -> LanguageCatalog:
    return LanguageCatalog(
        (
            _language("as", "Assamese", ["Amit", "Sita", "Poonam", "Rakesh"], ["Amit", "Sita"]),
            _language("bn", "Bengali", ["Arjun", "Aditi", "Tapan", "Rashmi", "Arnav", "Riya"], ["Arjun", "Aditi"]),
            _language("brx", "Bodo", ["Bikram", "Maya", "Kalpana"], ["Bikram", "Maya"]),
            _language("hne", "Chhattisgarhi", ["Bhanu", "Champa"], ["Bhanu", "Champa"]),
            _language("doi", "Dogri", ["Karan"], ["Karan"]),
            _language(
                "en",
                "English",
                [
                    "Thoma",
                    "Mary",
                    "Swapna",
                    "Dinesh",
                    "Meera",
                    "Jatin",
                    "Aakash",
                    "Sneha",
                    "Kabir",
                    "Tisha",
                    "Chingkhei",
                    "Thoiba",
                    "Priya",
                    "Tarun",
                    "Gauri",
                    "Nisha",
                    "Raghav",
                    "Kavya",
                    "Ravi",
                    "Vikas",
                    "Riya",
                ],
                ["Thoma", "Mary"],
            ),
            _language("gu", "Gujarati", ["Yash", "Neha"], ["Yash", "Neha"]),
            _language("hi", "Hindi", ["Rohit", "Divya", "Aman", "Rani"], ["Rohit", "Divya"]),
            _language("kn", "Kannada", ["Suresh", "Anu", "Chetan", "Vidya"], ["Suresh", "Anu"]),
            _language("ml", "Malayalam", ["Anjali", "Anju", "Harish"], ["Anjali", "Harish"]),
            _language("mni", "Manipuri", ["Laishram", "Ranjit"], ["Laishram", "Ranjit"]),
            _language("mr", "Marathi", ["Sanjay", "Sunita", "Nikhil", "Radha", "Varun", "Isha"], ["Sanjay", "Sunita"]),
            _language("ne", "Nepali", ["Amrita"], ["Amrita"]),
            _language("or", "Odia", ["Manas", "Debjani"], ["Manas", "Debjani"]),
            _language("pa", "Punjabi", ["Divjot", "Gurpreet"], ["Divjot", "Gurpreet"]),
            _language("sa", "Sanskrit", ["Aryan"], ["Aryan"]),
            _language("ta", "Tamil", ["Kavitha", "Jaya"], ["Jaya"]),
            _language("te", "Telugu", ["Prakash", "Lalitha", "Kiran"], ["Prakash", "Lalitha"]),
        )
    )
