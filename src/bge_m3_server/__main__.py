# src/bge_m3_server/__main__.py
import logging
import sys


def main() -> None:
    """Run the BGE-M3 embedding server."""
    import uvicorn

    from .config import get_settings

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings = get_settings()

    uvicorn.run(
        "bge_m3_server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
