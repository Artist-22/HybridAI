"""Command-line interface for Hybrid4in1 AI"""

import click
import logging
from hybrid4in1 import HybridAI, HybridAIServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """🚀 Hybrid4in1 AI - Ultimate 4-in-1 Face Swap Engine"""
    pass


@cli.command()
@click.option('--video', required=True, help='Path to video file')
@click.option('--face', required=True, help='Path to face image')
@click.option('--output', required=True, help='Output video path')
@click.option('--quality', default='high', help='Quality: low/medium/high/ultra')
def swap(video, face, output, quality):
    """Swap faces in a video"""
    try:
        logger.info("Starting face swap...")
        ai = HybridAI()
        ai.swap_video(video, face, output, quality=quality)
        logger.info(f"✓ Done! Output: {output}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)


@cli.command()
@click.option('--face', required=True, help='Path to face image')
@click.option('--output', required=True, help='Output image path')
@click.option('--style', default='professional', help='Photo style')
def photo(face, output, style):
    """Generate AI-enhanced photo"""
    try:
        logger.info("Generating photo...")
        ai = HybridAI()
        ai.generate_photo(face, output, style=style)
        logger.info(f"✓ Done! Output: {output}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)


@cli.command()
@click.option('--port', default=5000, help='Port number')
@click.option('--host', default='0.0.0.0', help='Host address')
def server(port, host):
    """Start web server"""
    try:
        logger.info(f"Starting server on {host}:{port}")
        server = HybridAIServer(port=port, host=host)
        server.run()
    except Exception as e:
        logger.error(f"Failed: {e}")
        exit(1)


@cli.command()
def version():
    """Show version"""
    from hybrid4in1 import __version__
    click.echo(f"Hybrid4in1 AI v{__version__}")


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()
