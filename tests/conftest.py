import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from youtube_transcription.db import init_engine, reset_engine
from youtube_transcription.models import Base
from youtube_transcription.settings import reset_settings

DOCKER_AVAILABLE = False
try:
    from docker import DockerClient

    DockerClient().ping()
    DOCKER_AVAILABLE = True
except Exception:
    pass


def pytest_collection_modifyitems(items):
    if not DOCKER_AVAILABLE:
        raise OSError("Docker is required to run the test suite but is not available. Start Docker and re-run tests.")


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("postgres:17") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def db_url(postgres_container):
    host = postgres_container.get_container_host_ip()
    port = postgres_container.get_exposed_port(5432)
    return f"postgresql+asyncpg://test:test@{host}:{port}/test"


@pytest_asyncio.fixture
async def db_session(db_url):
    init_engine(db_url)
    from youtube_transcription.db import _get_engine

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    reset_engine()


@pytest.fixture(autouse=True)
def _reset_settings():
    reset_settings()
    yield
    reset_settings()
