# Ollama Mirror Server

A Python-based mirror server for Ollama model pulling that provides local caching, bandwidth optimization, and offline capability for organizations and multiple users.

## Features

- 🚀 **Local Model Caching**: Automatically cache downloaded models for faster subsequent access
- 🌐 **API Compatibility**: Full compatibility with Ollama's REST API
- 📊 **Monitoring & Metrics**: Built-in metrics collection and health monitoring
- ⚙️ **Configurable**: Flexible configuration via environment variables or config files
- 🔒 **Security**: Optional API key authentication and model access controls
- 💾 **Bandwidth Optimization**: Reduces internet bandwidth usage for multiple users
- 🏥 **Health Checks**: Comprehensive health monitoring for production deployments
- 📈 **Performance**: Async/await architecture for high performance

## Quick Start

### Installation

1. Clone or download the repository:
```bash
git clone <repository-url>
cd ollama-mirror
```

2. Install dependencies:
```bash
conda create -n ollama_mirror python==3.11.10
conda activate ollama_mirror
pip install -r requirements.txt
```

3. Run the server:
```bash
python ollama_mirror.py
```

The server will start on `http://localhost:11434` by default.

### Using the Mirror

Point your Ollama client to the mirror server:

```bash
# Set Ollama to use the mirror
export OLLAMA_HOST=http://localhost:11434

# Pull a model (will be cached locally)
ollama pull llama3.2

# Subsequent pulls will be served from cache
ollama pull llama3.2  # Much faster!
```

## Configuration

### Environment Variables

The server can be configured using environment variables with the `OLLAMA_MIRROR_` prefix:

```bash
# Server settings
export OLLAMA_MIRROR_HOST=0.0.0.0
export OLLAMA_MIRROR_PORT=11434
export OLLAMA_MIRROR_WORKERS=1

# Cache settings
export OLLAMA_MIRROR_CACHE_DIR=./cache
export OLLAMA_MIRROR_CACHE_RETENTION_DAYS=30
export OLLAMA_MIRROR_CACHE_MAX_SIZE_GB=100

# Upstream settings
export OLLAMA_MIRROR_UPSTREAM_URL=https://registry.ollama.ai
export OLLAMA_MIRROR_UPSTREAM_TIMEOUT=300
export OLLAMA_MIRROR_MAX_CONCURRENT_DOWNLOADS=3

# Security settings
export OLLAMA_MIRROR_API_KEY=your-secret-key
export OLLAMA_MIRROR_ALLOWED_MODELS=llama3.2,codellama,mistral
export OLLAMA_MIRROR_RATE_LIMIT_REQUESTS=100

# Logging settings
export OLLAMA_MIRROR_LOG_LEVEL=INFO
export OLLAMA_MIRROR_LOG_FILE=/var/log/ollama-mirror.log
```

### Configuration File

Alternatively, create a `config.json` file:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 11434,
    "workers": 1
  },
  "cache": {
    "directory": "./cache",
    "retention_days": 30,
    "max_size_gb": 100,
    "cleanup_interval": 3600
  },
  "upstream": {
    "url": "https://registry.ollama.ai",
    "timeout": 300,
    "max_concurrent_downloads": 3,
    "retry_attempts": 3
  },
  "security": {
    "api_key": "your-secret-key",
    "allowed_models": ["llama3.2", "codellama", "mistral"],
    "rate_limit_requests": 100,
    "rate_limit_window": 60
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/ollama-mirror.log",
    "max_file_size": "10MB",
    "backup_count": 5
  },
  "features": {
    "enable_api_proxy": true,
    "enable_metrics": true,
    "enable_web_ui": false
  }
}
```

Then run with:
```bash
python ollama_mirror.py --config config.json
```

## API Endpoints

### Ollama Compatible Endpoints

- `POST /api/pull` - Pull and cache a model
- `GET /api/tags` - List cached models
- `POST /api/generate` - Generate text (proxied to local Ollama)
- `POST /api/chat` - Chat completion (proxied to local Ollama)
- `GET /api/version` - Get server version

### Registry Compatible Endpoints

- `GET /v2/library/{model}/manifests/{tag}` - Get model manifest
- `GET /v2/library/{model}/blobs/{digest}` - Get model blob

### Mirror-Specific Endpoints

- `GET /` - Health check and server info
- `GET /mirror/stats` - Mirror statistics and metrics
- `DELETE /mirror/cache` - Clear all cached data
- `GET /mirror/health` - Detailed health check

## Architecture

### Components

1. **FastAPI Server**: Main web server handling HTTP requests
2. **Model Cache**: Local storage and management of model files
3. **Upstream Client**: Handles communication with Ollama registry
4. **Metrics Collector**: Tracks usage and performance metrics
5. **Health Checker**: Monitors system health and connectivity

### File Structure

```
cache/
├── manifests/
│   └── registry.ollama.ai/
│       └── library/
│           └── {model_name}/
│               └── {tag}          # Model manifest files
├── blobs/
│   └── sha256-{digest}           # Model blob files
└── metadata.json                 # Cache metadata
```

## Deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./

# Create cache directory
RUN mkdir -p /app/cache

# Expose port
EXPOSE 11434

# Run the application
CMD ["python", "ollama_mirror.py"]
```

Build and run:

```bash
docker build -t ollama-mirror .
docker run -p 11434:11434 -v ./cache:/app/cache ollama-mirror
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  ollama-mirror:
    build: .
    ports:
      - "11434:11434"
    volumes:
      - ./cache:/app/cache
      - ./logs:/app/logs
    environment:
      - OLLAMA_MIRROR_CACHE_DIR=/app/cache
      - OLLAMA_MIRROR_LOG_FILE=/app/logs/mirror.log
      - OLLAMA_MIRROR_LOG_LEVEL=INFO
    restart: unless-stopped

  # Optional: Add Prometheus for metrics
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Systemd Service

Create `/etc/systemd/system/ollama-mirror.service`:

```ini
[Unit]
Description=Ollama Mirror Server
After=network.target

[Service]
Type=simple
User=ollama-mirror
WorkingDirectory=/opt/ollama-mirror
ExecStart=/usr/bin/python3 /opt/ollama-mirror/ollama_mirror.py
Restart=always
RestartSec=3

# Environment
Environment=OLLAMA_MIRROR_CACHE_DIR=/var/cache/ollama-mirror
Environment=OLLAMA_MIRROR_LOG_FILE=/var/log/ollama-mirror.log

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable ollama-mirror
sudo systemctl start ollama-mirror
```

## Monitoring

### Health Checks

The server provides comprehensive health checks:

```bash
# Basic health check
curl http://localhost:11434/

# Detailed health check
curl http://localhost:11434/mirror/health
```

### Metrics

View detailed metrics:

```bash
curl http://localhost:11434/mirror/stats
```

Example response:

```json
{
  "cached_models": 5,
  "total_cache_size": 15728640000,
  "cache_directory": "./cache",
  "upstream_url": "https://registry.ollama.ai",
  "models": [
    {
      "name": "llama3.2:latest",
      "cached_at": "2024-01-15T10:30:00",
      "size": 4500000000
    }
  ]
}
```

### Prometheus Integration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'ollama-mirror'
    static_configs:
      - targets: ['localhost:11434']
    metrics_path: '/metrics'
```

## Performance Tuning

### Cache Configuration

- **Retention**: Set `CACHE_RETENTION_DAYS` based on your storage capacity
- **Size Limit**: Use `CACHE_MAX_SIZE_GB` to prevent disk exhaustion
- **Cleanup Interval**: Adjust `CACHE_CLEANUP_INTERVAL` for automatic cleanup

### Network Configuration

- **Concurrent Downloads**: Increase `MAX_CONCURRENT_DOWNLOADS` for faster initial caching
- **Timeout**: Adjust `UPSTREAM_TIMEOUT` based on your network conditions
- **Retries**: Configure `RETRY_ATTEMPTS` and `RETRY_DELAY` for reliability

### System Resources

- **Workers**: Increase `WORKERS` for high-concurrency deployments
- **Memory**: Monitor memory usage, especially for large models
- **Disk I/O**: Use fast storage (SSD) for better performance

## Troubleshooting

### Common Issues

1. **Cache Directory Permissions**:
   ```bash
   sudo chown -R $(whoami) ./cache
   chmod -R 755 ./cache
   ```

2. **Port Already in Use**:
   ```bash
   export OLLAMA_MIRROR_PORT=11435
   ```

3. **Upstream Connection Issues**:
   ```bash
   curl -v https://registry.ollama.ai/v2/
   ```

4. **Disk Space**:
   ```bash
   df -h ./cache
   du -sh ./cache/*
   ```

### Debug Logging

Enable debug logging:

```bash
export OLLAMA_MIRROR_LOG_LEVEL=DEBUG
python ollama_mirror.py
```

### Health Checks

Check component health:

```bash
# Server health
curl http://localhost:11434/mirror/health

# Cache status
ls -la ./cache/

# Metrics
curl http://localhost:11434/mirror/stats | jq .
```

## Security

### API Key Authentication

Set an API key to restrict access:

```bash
export OLLAMA_MIRROR_API_KEY=your-secret-key
```

Include in requests:

```bash
curl -H "Authorization: Bearer your-secret-key" \
     http://localhost:11434/api/pull \
     -d '{"model": "llama3.2"}'
```

### Model Access Control

Restrict which models can be cached:

```bash
export OLLAMA_MIRROR_ALLOWED_MODELS=llama3.2,codellama,mistral
```

Block specific models:

```bash
export OLLAMA_MIRROR_BLOCKED_MODELS=dangerous-model,another-blocked-model
```

### Rate Limiting

Configure rate limiting:

```bash
export OLLAMA_MIRROR_RATE_LIMIT_REQUESTS=100
export OLLAMA_MIRROR_RATE_LIMIT_WINDOW=60
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black *.py

# Check linting
flake8 *.py

# Type checking
mypy *.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- 📖 Documentation: See this README and inline code comments
- 🐛 Issues: Report bugs via GitHub issues
- 💬 Discussions: Use GitHub discussions for questions
- 📧 Email: Contact the maintainers

## Roadmap

- [ ] Web UI for administration
- [ ] Multi-registry support
- [ ] Model compression/deduplication
- [ ] Distributed caching
- [ ] Advanced authentication (LDAP, OAuth)
- [ ] Model scanning and security
- [ ] Custom model repositories
- [ ] Statistics dashboard
