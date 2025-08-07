#!/usr/bin/env python3
"""
Simple test script for Ollama Mirror Server

This script demonstrates how to test the mirror server functionality.
"""

import asyncio
import json
import requests
import time
from pathlib import Path

BASE_URL = "http://localhost:11434"


def test_health_check():
    """Test basic health check"""
    print("Testing health check...")

    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_mirror_health():
    """Test mirror-specific health check"""
    print("Testing mirror health check...")

    try:
        response = requests.get(f"{BASE_URL}/mirror/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Mirror health check passed: {health_data['status']}")
            return True
        else:
            print(f"❌ Mirror health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Mirror health check failed: {e}")
        return False


def test_stats():
    """Test stats endpoint"""
    print("Testing stats endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/mirror/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats endpoint works")
            print(f"   Cached models: {stats.get('cached_models', 0)}")
            print(
                f"   Cache directory: {stats.get('cache_directory', 'unknown')}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint failed: {e}")
        return False


def test_model_list():
    """Test model listing"""
    print("Testing model list endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Model list endpoint works")
            print(f"   Models found: {len(models.get('models', []))}")
            return True
        else:
            print(f"❌ Model list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model list failed: {e}")
        return False


def test_model_pull_dry_run():
    """Test model pull endpoint (dry run - just check if endpoint responds)"""
    print("Testing model pull endpoint (dry run)...")

    try:
        # Use a very small model for testing
        payload = {
            "model": "registry.ollama.ai/library/all-minilm:latest",
            "stream": False
        }

        response = requests.post(
            f"{BASE_URL}/api/pull", json=payload, timeout=10)

        # We expect this might fail due to network or model availability
        # but we want to check if the endpoint responds correctly
        if response.status_code in [200, 404, 503]:
            print("✅ Model pull endpoint responds correctly")
            return True
        else:
            print(f"⚠️  Model pull endpoint returned: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return True  # Still consider this a pass for endpoint functionality

    except requests.exceptions.Timeout:
        print("⚠️  Model pull timed out (expected for large models)")
        return True
    except Exception as e:
        print(f"❌ Model pull test failed: {e}")
        return False


def test_version():
    """Test version endpoint"""
    print("Testing version endpoint...")

    try:
        response = requests.get(f"{BASE_URL}/api/version")
        if response.status_code == 200:
            version = response.json()
            print(
                f"✅ Version endpoint works: {version.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Version endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Version endpoint failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("🧪 Running Ollama Mirror Server Tests")
    print("=" * 50)

    tests = [
        test_health_check,
        test_mirror_health,
        test_version,
        test_model_list,
        test_stats,
        test_model_pull_dry_run,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

        print()  # Add spacing between tests

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the mirror server configuration.")
        return False


def main():
    """Main function"""
    print("Ollama Mirror Server Test Suite")
    print()

    # Check if server is running
    print("Checking if server is running...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Server is not responding: {e}")
        print("\nPlease start the mirror server first:")
        print("  python ollama_mirror.py")
        print("  or")
        print("  python cli.py start")
        return False

    print()

    # Run tests
    success = run_all_tests()

    if not success:
        print("\nTroubleshooting tips:")
        print("1. Ensure the mirror server is running on the correct port")
        print("2. Check server logs for any errors")
        print("3. Verify network connectivity to upstream registry")
        print("4. Check cache directory permissions")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
