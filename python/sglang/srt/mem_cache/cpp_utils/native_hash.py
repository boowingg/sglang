import hashlib
import logging
import os
import platform
import sys
from array import array
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _cpu_supports_avx2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64"):
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            return "avx2" in f.read().lower()
    except OSError:
        return False


@lru_cache(maxsize=1)
def _load_native_hash_module() -> Any:
    if sys.byteorder != "little" or not sys.platform.startswith("linux"):
        logger.warning(
            "HiCache native hash is unsupported on this platform; using the "
            "Python SHA-256 fallback."
        )
        return None

    try:
        from torch.utils.cpp_extension import load

        abs_path = os.path.dirname(os.path.abspath(__file__))
        extra_cflags = ["-O3", "-std=c++17", "-DNDEBUG"]
        if _cpu_supports_avx2():
            extra_cflags.append("-mavx2")
        return load(
            name="hicache_hash_cpp",
            sources=[f"{abs_path}/hash_binding.cpp"],
            extra_cflags=extra_cflags,
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load HiCache native hash extension; using the Python "
            "SHA-256 fallback. Native build error: %s",
            exc,
        )
        return None


def _native_hash_input(token_ids: Any) -> tuple[array, int, int, bool]:
    raw_token_ids = getattr(token_ids, "raw_token_ids", None)
    raw = (
        raw_token_ids()
        if raw_token_ids is not None
        else getattr(token_ids, "token_ids", token_ids)
    )

    logical_len = len(token_ids)
    is_bigram = getattr(token_ids, "is_bigram", False)

    if isinstance(raw, array) and raw.typecode in ("I", "q", "Q", "L"):
        if is_bigram and logical_len > 0 and len(raw) < logical_len + 1:
            raise ValueError("bigram token buffer is shorter than logical length")
        return raw, logical_len, 2 if is_bigram else 1, is_bigram

    if is_bigram:
        return array("I", raw[: logical_len + 1]), logical_len, 2, is_bigram

    if logical_len == 0:
        return array("I"), logical_len, 1, is_bigram

    first_token = raw[0]
    if isinstance(first_token, tuple):
        unit_width = len(first_token)
        return (
            array("I", (elem for token in raw[:logical_len] for elem in token)),
            logical_len,
            unit_width,
            is_bigram,
        )

    return array("I", raw[:logical_len]), logical_len, 1, is_bigram


def get_native_hash(
    token_ids: Any, prior_digest: Optional[bytes], page_size: Optional[int] = None
) -> str | list[str]:
    raw, logical_len, unit_width, is_bigram = _native_hash_input(token_ids)
    native_module = _load_native_hash_module()
    if native_module is not None:
        return native_module.get_hash(
            raw, logical_len, unit_width, is_bigram, prior_digest, page_size
        )
    return _python_hash(
        raw, logical_len, unit_width, is_bigram, prior_digest, page_size
    )


def _python_hash(
    raw: array,
    logical_len: int,
    unit_width: int,
    is_bigram: bool,
    prior_digest: Optional[bytes],
    page_size: Optional[int],
) -> str | list[str]:
    """Byte-compatible fallback for ``hash_binding.cpp``."""

    if prior_digest is not None and len(prior_digest) != hashlib.sha256().digest_size:
        raise ValueError("prior_digest must be exactly 32 bytes")

    def page_bytes(start: int, count: int) -> bytes:
        words = bytearray()
        if is_bigram:
            for index in range(start, start + count):
                for value in (raw[index], raw[index + 1]):
                    words.extend(_u32_bytes(value))
        else:
            word_start = start * unit_width
            word_end = word_start + count * unit_width
            for value in raw[word_start:word_end]:
                words.extend(_u32_bytes(value))
        return bytes(words)

    def digest_page(data: bytes, previous: Optional[bytes]) -> bytes:
        hasher = hashlib.sha256()
        if previous is not None:
            hasher.update(previous)
        hasher.update(data)
        return hasher.digest()

    if page_size is None:
        return digest_page(page_bytes(0, logical_len), prior_digest).hex()
    if page_size < 0:
        raise ValueError("page_size must be non-negative")
    if page_size == 0:
        return []

    hashes = []
    running_digest = prior_digest
    for start in range(0, logical_len, page_size):
        count = min(page_size, logical_len - start)
        running_digest = digest_page(page_bytes(start, count), running_digest)
        hashes.append(running_digest.hex())
    return hashes


def _u32_bytes(value: int) -> bytes:
    value = int(value)
    if value < 0 or value > 0xFFFFFFFF:
        raise OverflowError("token id does not fit in uint32")
    return value.to_bytes(4, byteorder="little", signed=False)
