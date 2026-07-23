"""Small Linux io_uring O_DIRECT reader implemented with ctypes and mmap.

It intentionally supports only the operation V3 needs: batched, aligned READs.
Keeping this boundary small makes syscall/ring behavior auditable.
"""
from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path

SYS_IO_URING_SETUP = 425
SYS_IO_URING_ENTER = 426
IORING_OFF_SQ_RING = 0
IORING_OFF_CQ_RING = 0x8000000
IORING_OFF_SQES = 0x10000000
IORING_ENTER_GETEVENTS = 1
IORING_OP_READ = 22


class SqOffsets(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("head", "tail", "ring_mask", "ring_entries", "flags", "dropped", "array", "resv1")] + \
               [("resv2", ctypes.c_uint64)]


class CqOffsets(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("head", "tail", "ring_mask", "ring_entries", "overflow", "cqes", "flags", "resv1")] + \
               [("resv2", ctypes.c_uint64)]


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("sq_entries", "cq_entries", "flags", "sq_thread_cpu", "sq_thread_idle", "features", "wq_fd")]+ \
               [("resv", ctypes.c_uint32*3), ("sq_off", SqOffsets), ("cq_off", CqOffsets)]


class Sqe(ctypes.Structure):
    _fields_ = [("opcode", ctypes.c_uint8), ("flags", ctypes.c_uint8),
                ("ioprio", ctypes.c_uint16), ("fd", ctypes.c_int32),
                ("off", ctypes.c_uint64), ("addr", ctypes.c_uint64),
                ("length", ctypes.c_uint32), ("rw_flags", ctypes.c_uint32),
                ("user_data", ctypes.c_uint64), ("buf_index", ctypes.c_uint16),
                ("personality", ctypes.c_uint16), ("splice_fd_in", ctypes.c_int32),
                ("pad2", ctypes.c_uint64*2)]


class Cqe(ctypes.Structure):
    _fields_ = [("user_data", ctypes.c_uint64), ("res", ctypes.c_int32),
                ("flags", ctypes.c_uint32)]


def _u32(buffer, offset):
    return ctypes.c_uint32.from_buffer(buffer, offset)


class IoUringDirectReader:
    def __init__(self, path: str | Path, entries: int = 32):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.params = Params()
        self.ring_fd = self.libc.syscall(SYS_IO_URING_SETUP, entries, ctypes.byref(self.params))
        if self.ring_fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), "io_uring_setup")
        self.file_fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        sq_len = self.params.sq_off.array + self.params.sq_entries*4
        cq_len = self.params.cq_off.cqes + self.params.cq_entries*ctypes.sizeof(Cqe)
        self.sq_ring = mmap.mmap(self.ring_fd, sq_len, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_SQ_RING)
        self.cq_ring = mmap.mmap(self.ring_fd, cq_len, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_CQ_RING)
        self.sqes = mmap.mmap(self.ring_fd, self.params.sq_entries*ctypes.sizeof(Sqe),
                              flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                              prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=IORING_OFF_SQES)
        self.sq_tail = _u32(self.sq_ring, self.params.sq_off.tail)
        self.sq_mask = _u32(self.sq_ring, self.params.sq_off.ring_mask).value
        self.sq_array_offset = self.params.sq_off.array
        self.cq_head = _u32(self.cq_ring, self.params.cq_off.head)
        self.cq_tail = _u32(self.cq_ring, self.params.cq_off.tail)
        self.cq_mask = _u32(self.cq_ring, self.params.cq_off.ring_mask).value

    def _submit(self, info: dict, block_ids: list[int], base: int, stride: int,
                slots: list[int]) -> None:
        if not block_ids:
            return
        bytes_per_token = 2 * info["kv_heads"] * info["head_dim"] * 2
        block_bytes = info["block_tokens"] * bytes_per_token
        if block_bytes % 512 or info["offset"] % 512:
            raise ValueError("O_DIRECT request is not 512-byte aligned")
        tail = self.sq_tail.value
        for index, block in enumerate(block_ids):
            slot = (tail + index) & self.sq_mask
            sqe = Sqe.from_buffer(self.sqes, slot*ctypes.sizeof(Sqe))
            ctypes.memset(ctypes.addressof(sqe), 0, ctypes.sizeof(Sqe))
            sqe.opcode, sqe.fd = IORING_OP_READ, self.file_fd
            sqe.off = info["offset"] + block*block_bytes
            sqe.addr, sqe.length, sqe.user_data = base + slots[index]*stride, block_bytes, index
            _u32(self.sq_ring, self.sq_array_offset + slot*4).value = slot
        self.sq_tail.value = tail + len(block_ids)
        result = self.libc.syscall(SYS_IO_URING_ENTER, self.ring_fd, len(block_ids), len(block_ids),
                                   IORING_ENTER_GETEVENTS, 0, 0)
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), "io_uring_enter")
        completed = 0
        head = self.cq_head.value
        while completed < len(block_ids):
            if head == self.cq_tail.value:
                self.libc.syscall(SYS_IO_URING_ENTER, self.ring_fd, 0, 1,
                                  IORING_ENTER_GETEVENTS, 0, 0)
                continue
            cqe = Cqe.from_buffer(self.cq_ring, self.params.cq_off.cqes +
                                  (head & self.cq_mask)*ctypes.sizeof(Cqe))
            index = int(cqe.user_data)
            if cqe.res != block_bytes:
                error = -cqe.res if cqe.res < 0 else 5
                raise OSError(error, f"short direct read: {cqe.res}/{block_bytes}")
            completed += 1
            head += 1
        self.cq_head.value = head
        # ctypes.from_buffer exports must be released before mmap.close().
        try:
            del sqe, cqe
        except UnboundLocalError:
            pass

    def read_blocks_into(self, info: dict, block_ids: list[int], tensor,
                         slots: list[int] | None = None) -> None:
        """DMA blocks directly into rows of a contiguous pinned uint8 tensor."""
        if tensor.device.type != "cpu" or not tensor.is_pinned() or str(tensor.dtype) != "torch.uint8":
            raise ValueError("target must be a pinned CPU uint8 tensor")
        if tensor.ndim != 2 or not tensor.is_contiguous():
            raise ValueError("target must be a contiguous [slots, block_bytes] tensor")
        slots = list(range(len(block_ids))) if slots is None else slots
        self._submit(info, block_ids, tensor.data_ptr(), tensor.stride(0), slots)

    def read_blocks(self, info: dict, block_ids: list[int]) -> dict[int, bytes]:
        if not block_ids:
            return {}
        bytes_per_token = 2 * info["kv_heads"] * info["head_dim"] * 2
        block_bytes = info["block_tokens"] * bytes_per_token
        buffer = mmap.mmap(-1, block_bytes*len(block_ids), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
        base = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        self._submit(info, block_ids, base, block_bytes, list(range(len(block_ids))))
        result = {block: bytes(buffer[index*block_bytes:(index+1)*block_bytes])
                  for index, block in enumerate(block_ids)}
        buffer.close()
        return result

    def close(self):
        del self.sq_tail, self.cq_head, self.cq_tail
        self.sqes.close(); self.sq_ring.close(); self.cq_ring.close()
        os.close(self.file_fd); os.close(self.ring_fd)

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
