#!/usr/bin/env python3
"""
Script that prints the location of a specific user
using the GitHub API.
"""

import requests
import sys
import time


def main(url):
    """
    Prints:
    - User's location if found
    - 'Not found' if user doesn't exist (404)
    - 'Reset in X min' if rate limit reached (403)
    """

    response = requests.get(url)

    # Handle user not found
    if response.status_code == 404:
        print("Not found")
        return

    # Handle rate limit exceeded
    elif response.status_code == 403:
        reset_time = response.headers.get("X-RateLimit-Reset")
        if reset_time:
            reset_timestamp = int(reset_time)
            current_time = int(time.time())
            minutes_left = (reset_timestamp - current_time) // 60
            print(f"Reset in {minutes_left} min")
        else:
            print("Reset in unknown time")
        return

    # Handle other successful responses
    elif response.status_code == 200:
        data = response.json()
        location = data.get("location")
        if location:
            print(location)
        else:
            print("Not found")
        return

    # Handle other unexpected cases
    else:
        print("Error: Unexpected response")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./2-user_location.py <GitHub API user URL>")
    else:
        main(sys.argv[1])