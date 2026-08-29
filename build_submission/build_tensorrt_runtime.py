"""Build a compact TensorRT runtime payload from NVIDIA's Linux wheel."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import struct
import tarfile
import tempfile
import urllib.request
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import zstandard


VERSION = "10.9.0.34"
WHEEL_URL = (
    "https://pypi.nvidia.com/tensorrt-cu12-libs/"
    "tensorrt_cu12_libs-10.9.0.34-py2.py3-none-manylinux_2_28_x86_64.whl"
)
WHEEL_SIZE = 3_103_291_777
LIBRARIES = {
    "libnvinfer.so.10": (
        "366342e2c2da994d281237449a16c23a44cbfb5a4806d3de6a8c68d995a7c5de"
    ),
    "libnvinfer_plugin.so.10": None,
    "libnvonnxparser.so.10": (
        "b067692867444727381976dddad62012c2922ee90e799ec74a17c3e955b13bcd"
    ),
    "libnvinfer_builder_resource.so.10.9.0": (
        "70dc6615a7634e0fef346e285b942c20812879873c7112ffcff90a86c638a0a6"
    ),
}


class _RemoteWheel(io.RawIOBase):
    """Minimal seekable reader used only for ZIP central-directory metadata."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.size
        if offset < 0:
            raise ValueError("negative seek position")
        self.position = offset
        return offset

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.size, self.position + size) - 1
        data = _request_range(self.url, self.position, end).read()
        self.position += len(data)
        return data


def _request_range(url: str, start: int, end: int):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206:
        response.close()
        raise RuntimeError(f"server ignored byte range {start}-{end}")
    return response


def _member_data_offset(url: str, member: ZipInfo) -> int:
    with _request_range(url, member.header_offset, member.header_offset + 29) as response:
        header = response.read()
    fields = struct.unpack("<IHHHHHIIIHH", header)
    if fields[0] != 0x04034B50:
        raise RuntimeError(f"invalid ZIP local header for {member.filename}")
    return member.header_offset + 30 + fields[-2] + fields[-1]


def _extract_member(url: str, member: ZipInfo, target: Path) -> str:
    start = _member_data_offset(url, member)
    end = start + member.compress_size - 1
    digest = hashlib.sha256()
    checksum = 0
    decompressor = (
        zlib.decompressobj(-zlib.MAX_WBITS)
        if member.compress_type == ZIP_DEFLATED
        else None
    )
    if member.compress_type not in {ZIP_DEFLATED, ZIP_STORED}:
        raise RuntimeError(
            f"unsupported ZIP compression {member.compress_type}: {member.filename}"
        )
    written = 0
    with _request_range(url, start, end) as response, target.open("wb") as output:
        while chunk := response.read(8 * 1024**2):
            data = decompressor.decompress(chunk) if decompressor else chunk
            output.write(data)
            digest.update(data)
            checksum = zlib.crc32(data, checksum)
            written += len(data)
        if decompressor:
            data = decompressor.flush()
            output.write(data)
            digest.update(data)
            checksum = zlib.crc32(data, checksum)
            written += len(data)
    if written != member.file_size:
        raise RuntimeError(
            f"wrong extracted size for {member.filename}: {written} != {member.file_size}"
        )
    if checksum != member.CRC:
        raise RuntimeError(
            f"CRC-32 mismatch for {member.filename}: {checksum:08x} != {member.CRC:08x}"
        )
    return digest.hexdigest()


def _select_members(url: str) -> tuple[dict[str, ZipInfo], ZipInfo]:
    with ZipFile(_RemoteWheel(url, WHEEL_SIZE)) as archive:
        members = archive.infolist()
    selected: dict[str, ZipInfo] = {}
    license_member = None
    for member in members:
        basename = Path(member.filename).name
        if basename in LIBRARIES:
            selected[basename] = member
        if member.filename.endswith(".dist-info/LICENSE.txt"):
            license_member = member
    missing = sorted(set(LIBRARIES) - set(selected))
    if missing or license_member is None:
        raise RuntimeError(
            f"TensorRT wheel layout changed; missing libraries={missing}, "
            f"license={license_member is not None}"
        )
    return selected, license_member


def build(output_directory: Path, *, level: int = 22) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = output_directory / f"tensorrt-runtime-{VERSION}-linux-x86_64.tar.zst"
    selected, license_member = _select_members(WHEEL_URL)
    with tempfile.TemporaryDirectory(prefix="twin2attr_trt_build_") as directory:
        staging = Path(directory)
        for name in LIBRARIES:
            member = selected[name]
            print(
                f"Downloading {name}: {member.compress_size / 1e6:.1f} MB compressed",
                flush=True,
            )
            digest = _extract_member(WHEEL_URL, member, staging / name)
            expected_digest = LIBRARIES[name]
            if expected_digest is not None and digest != expected_digest:
                raise RuntimeError(f"SHA-256 mismatch for {name}: {digest}")
        license_path = output_directory / "LICENSE.txt"
        _extract_member(WHEEL_URL, license_member, license_path)

        temporary = payload.with_suffix(payload.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        compressor = zstandard.ZstdCompressor(level=level, threads=-1)
        with temporary.open("wb") as compressed:
            with compressor.stream_writer(compressed, closefd=False) as writer:
                with tarfile.open(fileobj=writer, mode="w|") as archive:
                    for name in LIBRARIES:
                        source = staging / name
                        info = archive.gettarinfo(str(source), arcname=name)
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)
        temporary.replace(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "tensorrt_runtime",
    )
    parser.add_argument("--level", type=int, default=22)
    args = parser.parse_args()
    result = build(args.output_directory, level=args.level)
    print(f"Created {result} ({result.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
