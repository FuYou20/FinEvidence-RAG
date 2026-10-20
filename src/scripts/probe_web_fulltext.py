"""One small real search plus page-read probe; never prints credentials."""

import os
from pathlib import Path

from dotenv import load_dotenv

from errors import RequestError
from src.web_page_reader import create_web_page_reader
from src.web_search import create_bocha_web_search


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    search = create_bocha_web_search(
        api_key=os.getenv("BOCHA_API_KEY"),
        count=2,
        timeout=15.0,
    )
    reader = create_web_page_reader(timeout=10.0)
    try:
        results = search(
            question="查找公开的公司资料",
            query="传音控股 官方 年度报告",
            evidence=[],
        )
    except RequestError as exc:
        print(
            f"BOCHA failed: status={exc.status_code}, "
            f"retryable={exc.retryable}"
        )
        raise SystemExit(1) from None

    print(f"BOCHA ok: count={len(results)}")
    for index, result in enumerate(results, start=1):
        try:
            page = reader(url=result["url"])
        except RequestError as exc:
            print(
                f"[{index}] summary fallback: "
                f"status={exc.status_code}, retryable={exc.retryable}, "
                f"url={result['url']}"
            )
            continue
        print(
            f"[{index}] fulltext ok: chars={len(page['text'])}, "
            f"title={page.get('title')!r}, url={page['url']}"
        )


if __name__ == "__main__":
    main()
