from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtube_transcription.pipeline import (
    diarize,
    download_audio,
    get_video_metadata,
    result_to_dict,
    result_to_markdown,
    validate_url,
)


class TestDiarize:
    @patch("whisperx.diarize.assign_word_speakers")
    @patch("whisperx.diarize.DiarizationPipeline")
    def test_calls_pipeline_and_assigns_speakers(self, mock_pipeline_cls, mock_assign):
        diarizer = MagicMock()
        mock_pipeline_cls.return_value = diarizer
        diarizer.return_value = MagicMock()  # diarize dataframe
        result = {"segments": [{"start": 0.0, "end": 5.0, "text": "hello"}]}
        mock_assign.side_effect = lambda df, res: {**res, "segments": [{**res["segments"][0], "speaker": "SPEAKER_00"}]}

        out, model = diarize("/tmp/audio.wav", result, "cuda", hf_token="tok")

        mock_pipeline_cls.assert_called_once_with(device="cuda", token="tok")
        diarizer.assert_called_once_with("/tmp/audio.wav")
        mock_assign.assert_called_once()
        assert out["segments"][0]["speaker"] == "SPEAKER_00"
        assert model is diarizer


class TestDownloadAudio:
    @patch("youtube_transcription.pipeline.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="audio.wav\n", stderr="")
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_file", return_value=True):
            result = download_audio("https://youtube.com/watch?v=abc", Path("/tmp"))
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "--extract-audio" in cmd
        assert "wav" in cmd
        assert result is not None

    @patch("youtube_transcription.pipeline.subprocess.run")
    def test_failure_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = download_audio("https://youtube.com/watch?v=abc", Path("/tmp"))
        assert result is None


class TestGetVideoMetadata:
    @patch("youtube_transcription.pipeline.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="abc123\nTest Video\nTest Channel\n300\n1000\n", stderr=""
        )
        result = get_video_metadata("https://youtube.com/watch?v=abc123")
        assert result.video_id == "abc123"
        assert result.title == "Test Video"
        assert result.channel == "Test Channel"
        assert result.duration_seconds == 300.0
        assert result.view_count == 1000

    @patch("youtube_transcription.pipeline.subprocess.run")
    def test_failure_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        result = get_video_metadata("https://youtube.com/watch?v=abc123")
        assert result is None

    @patch("youtube_transcription.pipeline.subprocess.run")
    def test_na_values(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="abc\nTitle\nChannel\nNA\nNA\n", stderr="")
        result = get_video_metadata("https://youtube.com/watch?v=abc")
        assert result.duration_seconds is None
        assert result.view_count is None


class TestValidateUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42",
            "https://www.twitch.tv/ninoegyn/clip/WanderingGorgeousTireMingLee-udaOxVOvNij79pZS",
            "https://clips.twitch.tv/WanderingGorgeousTireMingLee-udaOxVOvNij79pZS",
            "https://www.twitch.tv/videos/1234567890",
        ],
    )
    def test_supported_urls_pass(self, url):
        validate_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "not_a_url",
            "https://vimeo.com/123456",
            "https://www.twitch.tv/ninoegyn",  # live stream, not a clip/VOD
            "dQw4w9WgXcQ",  # bare ID without a URL
        ],
    )
    def test_unsupported_urls_raise(self, url):
        with pytest.raises(ValueError, match="Unsupported URL"):
            validate_url(url)


class TestResultToDict:
    def test_basic_segments(self):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "  hello world  "},
                {"start": 5.0, "end": 10.0, "text": "foo bar"},
            ],
        }
        out = result_to_dict(result)
        assert out["language"] == "en"
        assert len(out["segments"]) == 2
        assert out["segments"][0]["text"] == "hello world"

    def test_with_speaker(self):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
            ],
        }
        out = result_to_dict(result)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_with_words(self):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 0.5}]},
            ],
        }
        out = result_to_dict(result)
        assert "words" in out["segments"][0]

    def test_no_speaker_omits_key(self):
        result = {
            "language": None,
            "segments": [{"start": 0.0, "end": 5.0, "text": "hello"}],
        }
        out = result_to_dict(result)
        assert "speaker" not in out["segments"][0]


class TestResultToMarkdown:
    def test_basic(self):
        result = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello world"},
            ],
        }
        md = result_to_markdown(result)
        assert "[00:00:00 - 00:00:05] hello world" in md

    def test_with_speaker(self):
        result = {
            "segments": [
                {"start": 10.0, "end": 20.0, "text": "hi", "speaker": "SPEAKER_01"},
            ],
        }
        md = result_to_markdown(result)
        assert "**SPEAKER_01**" in md
        assert "[00:00:10 - 00:00:20]" in md
