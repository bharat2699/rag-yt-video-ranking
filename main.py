import os
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
import csv


class YouTubeRanker:
    def __init__(self):
        load_dotenv()
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")

        if not self.youtube_api_key:
            raise ValueError(
                "Missing API keys. Please set YOUTUBE_API_KEY in .env file"
            )
        self.youtube = build("youtube", "v3", developerKey=self.youtube_api_key)

    def search_videos(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """Search YouTube videos using the YouTube Data API."""
        request = self.youtube.search().list(
            part="snippet", q=query, type="video", maxResults=max_results
        )
        response = request.execute()

        videos = []
        for item in response["items"]:
            video_id = item["id"]["videoId"]
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([entry["text"] for entry in transcript])

                videos.append(
                    {
                        "video_id": video_id,
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "transcript": transcript_text,
                        "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                    }
                )
            except Exception as e:
                print(f"Could not get transcript for video {video_id}: {str(e)}")
                continue

        return videos

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Sentence Transformers."""
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model
        return model.encode(text).tolist()

    def rank_videos(
        self, videos: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Rank videos based on semantic similarity to query."""
        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Get video embeddings
        video_embeddings = []
        for video in videos:
            # Combine title, description, and transcript for embedding
            combined_text = (
                f"{video['title']} {video['description']} {video['transcript'][:1000]}"
            )
            embedding = self.get_embedding(combined_text)
            video_embeddings.append(embedding)

        # Convert to numpy arrays
        query_embedding = np.array(query_embedding).reshape(1, -1)
        video_embeddings = np.array(video_embeddings)

        # Create FAISS index
        dimension = len(query_embedding[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(video_embeddings)

        # Search
        distances, indices = index.search(query_embedding, len(videos))

        # Rank videos
        ranked_videos = []
        for idx in indices[0]:
            video = videos[idx].copy()
            video["similarity_score"] = float(
                1 / (1 + distances[0][idx])
            )  # Convert distance to similarity score
            ranked_videos.append(video)

        return ranked_videos

    def search_and_rank(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search and rank videos based on query."""
        # Search videos
        videos = self.search_videos(query, max_results=max_results)

        if not videos:
            return []

        # Rank videos
        ranked_videos = self.rank_videos(videos, query)

        # Sort by similarity score
        ranked_videos.sort(key=lambda x: x["similarity_score"], reverse=True)

        return ranked_videos

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: str):
        """Save ranked video results to a CSV file."""
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Rank", "Title", "Video ID", "Similarity Score", "URL"])
            # Write video data
            for rank, video in enumerate(results, 1):
                writer.writerow(
                    [
                        rank,
                        video["title"],
                        video["video_id"],
                        video["similarity_score"],
                        f"https://www.youtube.com/watch?v={video['video_id']}",
                    ]
                )


def main():
    parser = argparse.ArgumentParser(
        description="Search and rank YouTube videos based on semantic relevance"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing content",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=10,
        help="Maximum number of results to return",
    )
    args = parser.parse_args()

    # Load JSON content
    with open(args.json_file, "r") as file:
        json_content = json.load(file)

    # Derive query strictly from the 'main_content' key
    main_content = json_content.get("main_content", [])
    if not main_content:
        raise ValueError(
            "The JSON file must contain a 'main_content' field with relevant content."
        )

    query = " ".join(
        main_content
    )  # Combine all main content into a single query string

    # Initialize the ranker and search
    ranker = YouTubeRanker()
    results = ranker.search_and_rank(query, args.max_results)

    # Print results
    for i, video in enumerate(results, 1):
        print(f"\n{i}. {video['title']}")
        print(f"Video ID: {video['video_id']}")
        print(f"Similarity Score: {video['similarity_score']:.4f}")
        print(f"URL: https://www.youtube.com/watch?v={video['video_id']}")
        print("-" * 80)

    # Save results to CSV
    output_csv = "ranked_videos.csv"
    ranker.save_results_to_csv(results, output_csv)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()
