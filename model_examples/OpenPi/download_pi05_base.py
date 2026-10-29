import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/"
DEST = Path("/workspace/openpi_TurboPhysAI/static_data/pi05_base")
DEST.mkdir(parents=True, exist_ok=True)
ROOT = DEST.resolve()

# 获取对象列表
objects = []
token = None
while True:
    query = {"prefix": PREFIX, "maxResults": "1000"}
    if token:
        query["pageToken"] = token
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        page = json.load(response)
    objects.extend(page.get("items", []))
    token = page.get("nextPageToken")
    if not token:
        break

if not objects:
    raise RuntimeError(f"GCS 中没有找到对象：{PREFIX}")

def paths_for(obj):
    name = obj["name"]
    rel = name[len(PREFIX):]
    target = (ROOT / rel).resolve()
    target.relative_to(ROOT)  # 确认目标仍在下载目录内
    part = Path(str(target) + ".part")
    return target, part

def local_bytes():
    total = 0
    for obj in objects:
        expected = int(obj["size"])
        target, part = paths_for(obj)
        if target.exists() and target.stat().st_size == expected:
            total += expected
        elif part.exists():
            total += min(part.stat().st_size, expected)
    return total

def publish_part(part, target, expected):
    if target.exists():
        if target.stat().st_size == expected:
            return
        raise RuntimeError(f"正式文件已存在但大小不符，未覆盖：{target}")
    os.link(part, target)

total_bytes = sum(int(obj["size"]) for obj in objects)
initial_bytes = local_bytes()
started = time.monotonic()
print(f"{len(objects)} 个对象，共 {total_bytes / 1024**3:.2f} GiB；已有数据 {initial_bytes / 1024**3:.2f} GiB")

for index, obj in enumerate(objects, 1):
    expected = int(obj["size"])
    target, part = paths_for(obj)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        actual = target.stat().st_size
        if actual == expected:
            print(f"跳过 {index}/{len(objects)}：{obj['name']}（已完整）")
            continue
        raise RuntimeError(f"正式文件大小不符，为避免覆盖而停止：{target}")

    if part.exists() and part.stat().st_size > expected:
        raise RuntimeError(f".part 大于 GCS 对象大小，为避免覆盖而停止：{part}")

    if part.exists() and part.stat().st_size == expected:
        publish_part(part, target, expected)
        print(f"完成 {index}/{len(objects)}：{obj['name']}（已有完整 .part）")
        continue

    object_url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
        f"{urllib.parse.quote(obj['name'], safe='')}"
    )
    params = {"alt": "media"}
    if obj.get("generation"):
        params["generation"] = obj["generation"]
    object_url += "?" + urllib.parse.urlencode(params)

    print(f"下载 {index}/{len(objects)}：{obj['name']}；"
          f"续传位置 {part.stat().st_size if part.exists() else 0} 字节")

    for attempt in range(1, 9):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(object_url, headers=headers)
        last_report = time.monotonic()
        bytes_since_report = 0

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", 200)
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes (\d+)-\d+/", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise RuntimeError(f"服务器返回了不匹配的续传范围：{content_range}")
                elif offset:
                    raise RuntimeError(f"服务器未接受 Range 续传（HTTP {status}），未覆盖已有部分文件")

                mode = "ab" if offset else "wb"
                with open(part, mode, buffering=0) as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_since_report += len(chunk)

                        now = time.monotonic()
                        if bytes_since_report >= 64 * 1024**2 or now - last_report >= 10:
                            done = local_bytes()
                            elapsed = max(now - started, 0.001)
                            speed = max(done - initial_bytes, 0) / elapsed / 1024**2
                            remaining_mib = max(total_bytes - done, 0) / 1024**2
                            eta = (
                                f"{remaining_mib / (speed * 60):.1f} 分钟"
                                if speed >= 0.1 else "估算中"
                            )
                            print(f"总进度 {done / total_bytes * 100:.1f}%  "
                                  f"{speed:.1f} MiB/s  剩余约 {eta}")
                            last_report = now
                            bytes_since_report = 0

            actual = part.stat().st_size
            if actual != expected:
                raise EOFError(f"对象不完整：本地 {actual} 字节，预期 {expected} 字节")

            publish_part(part, target, expected)
            print(f"完成 {index}/{len(objects)}：{obj['name']}")
            break

        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (TimeoutError, socket.timeout, urllib.error.URLError,
                http.client.HTTPException, ConnectionError, EOFError) as error:
            failure = error

        if attempt == 8:
            raise RuntimeError(
                f"重试 8 次仍失败：{obj['name']}；部分数据保留在 {part}"
            ) from failure

        wait = min(60, 2 ** (attempt - 1))
        offset = part.stat().st_size if part.exists() else 0
        print(f"读取失败：{failure}；{wait}s 后从 {offset} 字节处重试")
        time.sleep(wait)

print(f"\n下载完成：{ROOT}")
