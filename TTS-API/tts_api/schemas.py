from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class TTSRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    language: str
    voice_mode: Literal["speaker", "description"] = "speaker"
    speaker: str | None = None
    voice_description: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text must not be empty.")
        return value.strip()

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("voice_mode")
    @classmethod
    def normalize_voice_mode(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("speaker")
    @classmethod
    def normalize_speaker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("voice_description")
    @classmethod
    def normalize_voice_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_voice_inputs(self) -> "TTSRequest":
        if self.voice_mode == "speaker":
            if not self.speaker:
                raise ValueError("Speaker must be provided when voice_mode is 'speaker'.")
        elif self.voice_mode == "description":
            if not self.voice_description:
                raise ValueError(
                    "voice_description must be provided when voice_mode is 'description'."
                )
        return self


class VoiceCloneRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str
    language: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Text must not be empty.")
        return value.strip()

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        return value.strip().lower()
