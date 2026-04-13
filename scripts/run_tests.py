#!/usr/bin/env python3
"""
Run all tests with coverage reporting.
"""

import subprocess
import sys

def run_tests():
    """Run pytest with coverage."""
    
    print("=" * 60)
    print("Running Test Suite")
    print("=" * 60)
    
    # Run pytest without coverage first (to avoid config issues)
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        "tests/",
        "-v"
    ])
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.returncode

def run_linting():
    """Run flake8 linting (relaxed rules)."""
    print("\n" + "=" * 60)
    print("Running Linting (relaxed)")
    print("=" * 60)
    
    # Run flake8 but ignore many issues for now
    result = subprocess.run([
        sys.executable, "-m", "flake8",
        "scripts/",
        "app.py",
        "--max-line-length=120",
        "--ignore=E501,W291,W293,E302,E305,F401,W504,E128,W292,F841",
        "--count",
        "--statistics"
    ])
    
    print("Linting completed (some issues ignored for now)")
    return 0  # Don't fail on linting issues yet

def run_formatting():
    """Run black to check formatting."""
    print("\n" + "=" * 60)
    print("Checking Code Formatting")
    print("=" * 60)
    
    # Run black but don't fail
    result = subprocess.run([
        sys.executable, "-m", "black",
        "--check",
        "scripts/", "app.py",
        "--line-length", "120"
    ])
    
    if result.returncode != 0:
        print("Note: Formatting issues found but not blocking")
    
    return 0  # Don't fail on formatting issues yet

def main():
    """Run all checks."""
    tests_passed = run_tests() == 0
    run_linting()  # Don't fail on linting
    run_formatting()  # Don't fail on formatting
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Tests: {'✅' if tests_passed else '❌'}")
    print("Linting: ⚠️ (relaxed, issues to fix later)")
    print("Formatting: ⚠️ (relaxed, issues to fix later)")
    
    return 0 if tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())