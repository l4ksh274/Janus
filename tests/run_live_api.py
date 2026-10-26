import argparse
import os
import json

import httpx


MODEL = "openai/gpt-oss-20b"
PROMPT = "What is the integral of x^2?"
DEFAULT_PROXY_URL = "http://127.0.0.1:8000"


def main():
    parser = argparse.ArgumentParser(
        description="Send a real request to a running Janus proxy."
    )
    parser.add_argument(
        "--proxy-url",
        default=os.getenv("JANUS_PROXY_URL", DEFAULT_PROXY_URL),
        help="Janus server URL (default: %(default)s; or set JANUS_PROXY_URL)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="test streaming instead of the default non-streaming request",
    )
    args = parser.parse_args()

    with httpx.Client(
        base_url=args.proxy_url,
        timeout=httpx.Timeout(130.0, connect=10.0),
    ) as client:
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": MODEL, "input": PROMPT, "stream": args.stream},
        ) as response:
            response.raise_for_status()

            if args.stream:
                if not response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    raise AssertionError(
                        "The streaming response did not use text/event-stream."
                    )
                
                received_any = False
                for line in response.iter_lines():
                    received_any = True
                    if line.startswith("data: "):
                        data_str = line[len("data: "):].strip()
                        if data_str and data_str != "[DONE]":
                            event_data = json.loads(data_str)
                            # Print only the text delta tokens as they arrive
                            if event_data.get("type") == "response.output_text.delta":
                                print(event_data.get("delta", ""), end="", flush=True)

                if not received_any:
                    raise AssertionError("The streaming API response was empty.")
                print(f"\nStreaming request succeeded (HTTP {response.status_code}).")


if __name__ == "__main__":
    main()
