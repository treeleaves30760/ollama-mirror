#!/usr/bin/env python3
"""
Command-line interface for Ollama Mirror Server

Provides easy commands to start, stop, configure, and manage the mirror server.
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

import uvicorn

from config import load_config, save_config, setup_logging, validate_config, get_sample_config
from monitoring import get_health_checker

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ollama Mirror Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                        # Start the server
  %(prog)s start --config config.json  # Start with config file
  %(prog)s config generate              # Generate sample config
  %(prog)s config validate             # Validate current config
  %(prog)s health                      # Check server health
  %(prog)s cache status                # Show cache status
  %(prog)s cache clear                 # Clear cache
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Start command
    start_parser = subparsers.add_parser(
        "start", help="Start the mirror server")
    start_parser.add_argument(
        "--host",
        help="Host to bind to (overrides config)"
    )
    start_parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (overrides config)"
    )
    start_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    # Config commands
    config_parser = subparsers.add_parser(
        "config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(
        dest="config_action", help="Config actions")

    config_subparsers.add_parser(
        "generate", help="Generate sample configuration file")
    config_subparsers.add_parser("validate", help="Validate configuration")
    config_subparsers.add_parser("show", help="Show current configuration")

    # Health command
    subparsers.add_parser("health", help="Check server health")

    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_action", help="Cache actions")

    cache_subparsers.add_parser("status", help="Show cache status")
    cache_subparsers.add_parser("clear", help="Clear all cached data")
    cache_subparsers.add_parser("cleanup", help="Clean up old cache entries")

    args = parser.parse_args()

    # Set up basic logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = load_config(args.config)

    # Override config with command line arguments
    if hasattr(args, 'host') and args.host:
        config.host = args.host
    if hasattr(args, 'port') and args.port:
        config.port = args.port
    if hasattr(args, 'reload') and args.reload:
        config.reload = args.reload

    # Set up logging from config
    setup_logging(config)

    # Execute command
    if args.command == "start":
        start_server(config)
    elif args.command == "config":
        handle_config_command(args, config)
    elif args.command == "health":
        asyncio.run(check_health(config))
    elif args.command == "cache":
        handle_cache_command(args, config)
    else:
        parser.print_help()


def start_server(config):
    """Start the mirror server"""
    logger.info("Starting Ollama Mirror Server")
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Cache directory: {config.cache_dir}")
    logger.info(f"Upstream URL: {config.upstream_url}")

    # Validate configuration
    issues = validate_config(config)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        sys.exit(1)

    # Create cache directory
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        uvicorn.run(
            "ollama_mirror:app",
            host=config.host,
            port=config.port,
            workers=config.workers,
            reload=config.reload,
            log_level=config.log_level.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


def handle_config_command(args, config):
    """Handle configuration-related commands"""
    if args.config_action == "generate":
        generate_sample_config()
    elif args.config_action == "validate":
        validate_current_config(config)
    elif args.config_action == "show":
        show_current_config(config)
    else:
        print("Available config actions: generate, validate, show")


def generate_sample_config():
    """Generate a sample configuration file"""
    sample_config = get_sample_config()

    config_file = Path("ollama-mirror-config.json")

    if config_file.exists():
        response = input(
            f"Config file {config_file} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    try:
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)

        print(f"Sample configuration generated: {config_file}")
        print("Edit the file and use: python cli.py start --config " + str(config_file))

    except Exception as e:
        print(f"Failed to generate config file: {e}")
        sys.exit(1)


def validate_current_config(config):
    """Validate the current configuration"""
    print("Validating configuration...")

    issues = validate_config(config)

    if not issues:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)


def show_current_config(config):
    """Show the current configuration"""
    print("Current configuration:")
    print(json.dumps(config.to_dict(), indent=2, default=str))


async def check_health(config):
    """Check server health"""
    print("Checking server health...")

    health_checker = get_health_checker(config.cache_dir, config.upstream_url)
    health_status = await health_checker.check_health()

    status = health_status['status']
    if status == "healthy":
        print("✅ Server is healthy")
    elif status == "degraded":
        print("⚠️  Server is degraded")
    else:
        print("❌ Server is unhealthy")

    print(f"\nHealth check details:")
    print(f"  Status: {status}")
    print(f"  Timestamp: {health_status['timestamp']}")

    print(f"\nChecks:")
    for check_name, result in health_status['checks'].items():
        status_icon = "✅" if result else "❌"
        print(f"  {status_icon} {check_name}")

    if health_status['issues']:
        print(f"\nIssues:")
        for issue in health_status['issues']:
            print(f"  - {issue}")


def handle_cache_command(args, config):
    """Handle cache-related commands"""
    if args.cache_action == "status":
        show_cache_status(config)
    elif args.cache_action == "clear":
        clear_cache(config)
    elif args.cache_action == "cleanup":
        cleanup_cache(config)
    else:
        print("Available cache actions: status, clear, cleanup")


def show_cache_status(config):
    """Show cache status"""
    cache_dir = config.cache_dir

    if not cache_dir.exists():
        print("❌ Cache directory does not exist")
        return

    print(f"Cache directory: {cache_dir}")

    # Check cache structure
    manifests_dir = cache_dir / "manifests"
    blobs_dir = cache_dir / "blobs"
    metadata_file = cache_dir / "metadata.json"

    print(f"\nCache structure:")
    print(f"  Manifests directory: {'✅' if manifests_dir.exists() else '❌'}")
    print(f"  Blobs directory: {'✅' if blobs_dir.exists() else '❌'}")
    print(f"  Metadata file: {'✅' if metadata_file.exists() else '❌'}")

    # Calculate sizes
    if cache_dir.exists():
        total_size = sum(
            f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        total_size_mb = total_size / (1024 * 1024)

        print(f"\nCache statistics:")
        print(f"  Total size: {total_size_mb:.1f} MB")

        if blobs_dir.exists():
            blob_count = len(list(blobs_dir.glob('*')))
            print(f"  Blob files: {blob_count}")

        if manifests_dir.exists():
            manifest_count = len(list(manifests_dir.rglob('*')))
            print(f"  Manifest files: {manifest_count}")

    # Load metadata if available
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print(f"\nCached models:")
            for model_name, info in metadata.get('models', {}).items():
                size_mb = info.get('size', 0) / (1024 * 1024)
                print(
                    f"  - {model_name} ({size_mb:.1f} MB) - {info.get('cached_at', 'unknown')}")

        except Exception as e:
            print(f"⚠️  Failed to read metadata: {e}")


def clear_cache(config):
    """Clear all cached data"""
    cache_dir = config.cache_dir

    if not cache_dir.exists():
        print("Cache directory does not exist")
        return

    response = input(
        f"This will delete all cached data in {cache_dir}. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return

    try:
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Cache cleared successfully")

    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")
        sys.exit(1)


def cleanup_cache(config):
    """Clean up old cache entries"""
    from ollama_mirror import ModelCache

    try:
        cache = ModelCache(config.cache_dir)
        cache.cleanup_old_cache(config.cache_retention_days)
        print("✅ Cache cleanup completed")

    except Exception as e:
        print(f"❌ Cache cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
