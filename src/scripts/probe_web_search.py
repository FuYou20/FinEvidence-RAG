"""小额真实连通性探测；不打印密钥或服务端原始错误正文。"""

import os
from pathlib import Path

from dotenv import load_dotenv

from errors import RequestError
from src.web_search import create_bocha_web_search


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    search = create_bocha_web_search(
        api_key=os.getenv("BOCHA_API_KEY"), count=2, timeout=15.0,
    )
    try:
        results = search(
            question="查找公开的公司资料",
            query="传音控股 官方 年度报告",
            evidence=[],
        )
    except RequestError as exc:
        print(
            f"BOCHA failed: status={exc.status_code}, "
            f"retryable={exc.retryable}",
            flush=True,
        )
        raise SystemExit(1) from None
    print(f"BOCHA ok: count={len(results)}", flush=True)
    for result in results:
        print(result["url"], flush=True)


if __name__ == "__main__":
    main()
