import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import csv
from flask import Flask, request, jsonify


class YouTubeRanker:
    def __init__(self):
        load_dotenv()
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")

        if not self.youtube_api_key:
            raise ValueError(
                "Missing API keys. Please set YOUTUBE_API_KEY in .env file"
            )
        self.youtube = build("youtube", "v3", developerKey=self.youtube_api_key)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search_videos(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
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
                stats_request = self.youtube.videos().list(
                    part="statistics", id=video_id
                )
                stats_response = stats_request.execute()
                stats = stats_response["items"][0]["statistics"]
                duration_request = self.youtube.videos().list(
                    part="contentDetails", id=video_id
                )
                duration_response = duration_request.execute()
                duration = duration_response["items"][0]["contentDetails"]["duration"]
                duration_seconds = self.iso8601_duration_to_seconds(duration)
                videos.append(
                    {
                        "video_id": video_id,
                        "title": item["snippet"]["title"],
                        "description": item["snippet"]["description"],
                        "transcript": transcript_text,
                        "view_count": stats.get("viewCount"),
                        "like_count": stats.get("likeCount"),
                        "comment_count": stats.get("commentCount"),
                        "duration": duration_seconds,
                    }
                )
            except Exception as e:
                print(
                    f"Could not get transcript or stats for video {video_id}: {str(e)}"
                )
                continue
        return videos

    def iso8601_duration_to_seconds(self, iso8601_duration: str) -> int:
        import isodate

        duration = isodate.parse_duration(iso8601_duration)
        return int(duration.total_seconds())

    def get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def enhance_score(self, video: Dict[str, Any]) -> float:
        base_score = video.get("similarity_score", 0)
        view_count = int(video.get("view_count", 0))
        duration_minutes = video.get("duration", 0) / 60
        view_score = min(view_count / 1_000_000, 1)
        duration_penalty = 1 if duration_minutes <= 20 else 0.5
        return base_score * (1 + view_score) * duration_penalty

    def rank_videos(
        self, videos: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        query_embedding = self.get_embedding(query).reshape(1, -1)
        video_embeddings = []
        for video in videos:
            combined_text = (
                f"{video['title']} {video['description']} {video['transcript'][:1000]}"
            )
            video_embeddings.append(self.get_embedding(combined_text))
        video_embeddings = np.array(video_embeddings)
        dimension = query_embedding.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(video_embeddings)
        distances, indices = index.search(query_embedding, len(videos))
        ranked_videos = []
        for i, idx in enumerate(indices[0]):
            video = videos[idx]
            video["similarity_score"] = float(1 / (1 + distances[0][i]))
            video["enhanced_score"] = self.enhance_score(video)
            ranked_videos.append(video)
        return sorted(ranked_videos, key=lambda x: x["enhanced_score"], reverse=True)

    def filter_videos(self, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_videos = [
            video
            for video in videos
            if int(video.get("view_count", 0)) >= 10 and video.get("duration", 0) >= 120
        ]
        return filtered_videos

    def search_and_rank(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        videos = self.search_videos(query, max_results=max_results)
        if not videos:
            return []
        videos = self.filter_videos(videos)
        if not videos:
            return []
        ranked_videos = self.rank_videos(videos, query)
        return sorted(ranked_videos, key=lambda x: x["similarity_score"], reverse=True)

    def save_results_to_csv(self, results: List[Dict[str, Any]], output_file: str):
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Rank",
                    "Title",
                    "Video ID",
                    "Similarity Score",
                    "Views",
                    "Duration",
                    "URL",
                ]
            )
            for rank, video in enumerate(results, 1):
                humanized_duration = self.humanize_duration(video["duration"])
                writer.writerow(
                    [
                        rank,
                        video["title"],
                        video["video_id"],
                        video["similarity_score"],
                        video["view_count"],
                        humanized_duration,
                        f"https://www.youtube.com/watch?v={video['video_id']}",
                    ]
                )

    def humanize_duration(self, duration_seconds: int) -> str:
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"


app = Flask(__name__)


@app.route("/ranked_videos", methods=["POST"])
def ranked_videos():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400
        main_content = data.get("main_content", [])
        if not main_content:
            return jsonify(
                {"error": "'main_content' field is required and cannot be empty"}
            ), 400
        query = " ".join(main_content)
        max_results = data.get("max_results", 10)
        ranker = YouTubeRanker()
        results = ranker.search_and_rank(query, max_results)
        output_csv = "ranked_videos.csv"
        ranker.save_results_to_csv(results, output_csv)
        video_links = [
            f"https://www.youtube.com/watch?v={video['video_id']}" for video in results
        ]
        return jsonify(video_links)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
