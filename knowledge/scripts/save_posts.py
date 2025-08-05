"""
Save posts to a JSON file.

Example:

{
  "id": 1,
  "title": "His mother had always taught him",
  "body": "His mother had always taught him not to ever think of himself as better than others. He'd tried to live by this motto. He never looked down on those who were less fortunate or who had less money than him. But the stupidity of the group of people he was talking to made him change his mind.",
  "tags": [
    "history",
    "american",
    "crime"
  ],
  "reactions": {
    "likes": 192,
    "dislikes": 25
  },
  "views": 305,
  "userId": 121
}

"""

import os
import requests
import json


def fetch_and_save_posts(output_dir="knowledge/files"):
    resp = requests.get("https://dummyjson.com/posts")
    data = resp.json()
    posts = data.get("posts", [])

    for post in posts:
        post_id = post.get("id")
        filename = os.path.join(output_dir, f"post_{post_id}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(post, f, ensure_ascii=False, indent=2)
        print(f"Saved post {post_id} → {filename}")


if __name__ == "__main__":
    fetch_and_save_posts()
