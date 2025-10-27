#!/usr/bin/env python3
from pymongo import MongoClient

def main():
    # Connect to the MongoDB server
    client = MongoClient('localhost', 27017)
    db = client.logs  # Select the logs database
    collection = db.nginx  # Select the nginx collection

    # Get the total number of logs
    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    # Count methods
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {method: collection.count_documents({"method": method}) for method in methods}

    # Print method counts
    print("Methods:")
    for method in methods:
        print(f"\tmethod {method}: {method_counts[method]}")

    # Count specific condition: method=GET and path=/status
    status_check_count = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check_count} status check")

if __name__ == "__main__":
    main()