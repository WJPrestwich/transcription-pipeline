import json
import sys
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table

from youtube_transcription.settings import get_settings

# Transcription can take many minutes — don't time out mid-request
_LONG_TIMEOUT = httpx.Timeout(None)


def _request(method: str, path: str, timeout: httpx.Timeout | None = None, **kwargs) -> dict | str:
    """Call the REST API, exiting with a friendly message on connection or HTTP errors."""
    base_url = get_settings().api_url
    try:
        response = httpx.request(method, f"{base_url}{path}", timeout=timeout or 30.0, **kwargs)
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot reach API at {base_url}\n\nStart the server with: docker compose up",
            err=True,
        )
        sys.exit(1)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except json.JSONDecodeError:
            detail = response.text
        click.echo(f"Error ({response.status_code}): {detail}", err=True)
        sys.exit(1)
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.text


@click.group()
def cli() -> None:
    """CLI client for the youtube-transcription REST API.

    All commands call the running server (start with: docker compose up).
    The server URL is configured via API_URL (default http://localhost:8000).
    """


@cli.command()
@click.argument("url")
@click.option(
    "--isolate",
    is_flag=True,
    default=False,
    help="Isolate vocals with Demucs before transcribing (slower; helps on music-heavy audio)",
)
@click.option(
    "--diarize",
    is_flag=True,
    default=False,
    help="Identify speakers with pyannote diarization (requires HF_TOKEN; adds processing time)",
)
def transcribe(url: str, isolate: bool, diarize: bool) -> None:
    """Transcribe a YouTube video or Twitch clip/VOD (blocks until done)."""
    click.echo(f"Transcribing {url}...")
    result = _request(
        "POST",
        "/api/transcribe",
        json={"url": url, "isolate": isolate, "diarize": diarize},
        timeout=_LONG_TIMEOUT,
    )
    transcript_id = result.get("id", "ERROR")
    click.echo(f"Finished: {transcript_id=}")


def _format_duration(seconds: float | None) -> str:
    """Render a duration as H:MM:SS (or M:SS for short videos)."""
    if seconds is None:
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_created(iso_timestamp: str | None) -> str:
    """Trim an ISO 8601 timestamp to a compact 'YYYY-MM-DD HH:MM' form."""
    if not iso_timestamp:
        return "—"
    return iso_timestamp[:16].replace("T", " ")


@cli.command("list")
@click.option("--limit", default=20, help="Max transcripts to return")
@click.option("--offset", default=0, help="Number of transcripts to skip")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON instead of a table")
def list_cmd(limit: int, offset: int, as_json: bool) -> None:
    """List past transcripts."""
    result = _request("GET", "/api/transcripts", params={"limit": limit, "offset": offset})
    if as_json:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    transcripts = result.get("transcripts", [])
    total = result.get("total", 0)
    if not transcripts:
        click.echo("No transcripts found.")
        return

    table = Table(title="Transcripts", caption=f"Showing {offset + 1}-{offset + len(transcripts)} of {total}")
    table.add_column("ID", width=36, no_wrap=True)
    table.add_column("Title", overflow="ellipsis", ratio=3, no_wrap=True)
    table.add_column("Channel", overflow="ellipsis", ratio=1, no_wrap=True)
    table.add_column("Duration", width=8, justify="right", no_wrap=True)
    table.add_column("Lang", width=4, no_wrap=True)
    table.add_column("Speakers", width=8, justify="right", no_wrap=True)
    table.add_column("Created", width=16, no_wrap=True)

    for t in transcripts:
        table.add_row(
            t["id"],
            t.get("video_title") or t.get("video_url") or "—",
            t.get("channel") or "—",
            _format_duration(t.get("duration_seconds")),
            t.get("language") or "—",
            str(t["speaker_count"]) if t.get("speaker_count") is not None else "—",
            _format_created(t.get("created_at")),
        )

    Console().print(table)


@cli.command()
@click.argument("transcript_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    help="Output format",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Output file path (default: output/transcript.json or output/transcript.md)",
)
def get(transcript_id: str, output_format: str, output_path: str | None) -> None:
    """Retrieve a transcript by ID and write it to a file."""
    result = _request("GET", f"/api/transcripts/{transcript_id}", params={"format": output_format})
    if output_format == "markdown":
        content = result
        default_name = "transcript.md"
    else:
        content = json.dumps(result, indent=2, default=str)
        default_name = "transcript.json"

    path = Path(output_path) if output_path else Path("output") / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n")
    click.echo(f"Wrote {path}")


if __name__ == "__main__":
    cli()
