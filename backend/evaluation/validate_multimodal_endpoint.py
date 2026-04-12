from argparse import ArgumentParser
import json
import os
import urllib.request


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--user-id", default="validation_user")
    parser.add_argument("--patient-text", required=True)
    parser.add_argument("--relative-text", default="")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--audio-path", default=None)
    parser.add_argument(
        "--url",
        default=os.environ.get("APP_BASE_URL", "http://127.0.0.1:8000").rstrip("/") + "/multimodal/assess",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    payload = {
        "user_id": args.user_id,
        "patient_text": args.patient_text,
        "relative_text": args.relative_text,
        "video_path": args.video_path,
    }
    if args.audio_path:
        payload["audio_path"] = args.audio_path

    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        print(response.status)
        print(response.read().decode("utf-8"))
