# 02 — Storage Engine, Buffer Pool & Object Store Specification

## 1. Universal 8,192-Byte Page Geometry (`src/storage/page_header.py`)

Every relational page (Row or Column) and every B+ Tree index page is **strictly 8,192 bytes (`PAGE_SIZE = 8192`)** and begins with an identical **32-byte header**.

### 1.1 Binary Header Layout (`<IBBHQIIQ`)
* `offset 0..3` (`I`, `uint32`): `page_id` — Zero-based physical page index within its file.
* `offset 4` (`B`, `uint8`): `page_type` — `0` = `FREE_OR_FSM`, `1` = `SLOTTED_ROW`, `2` = `COLUMN_CHUNK`, `3` = `BPLUS_INTERNAL`, `4` = `BPLUS_LEAF`, `5` = `OBJECT_DIR`.
* `offset 5` (`B`, `uint8`): `flags` — Bitmask (`1` = `IS_ROOT`, `2` = `IS_LEAF`, `4` = `COMPRESSED`, `8` = `ENCRYPTED`).
* `offset 6..7` (`H`, `uint16`): `slot_count` — Number of active slots (`SlottedPage`), values (`ColumnPage`), or keys (`BPlusTree`).
* `offset 8..15` (`Q`, `uint64`): `page_lsn` — Log Sequence Number of last modification (initialized to `0`; used by Milestone 4 ARIES recovery).
* `offset 16..19` (`I`, `uint32`): `free_space_ptr` — Byte offset where tuple payload starts (`SlottedPage`, initialized to `8192`) or byte offset where column data ends (`ColumnPage`, initialized to `32`).
* `offset 20..23` (`I`, `uint32`): `next_page_id` — Right-sibling page ID for B+ Tree leaf range scans or chained pages (`0xFFFFFFFF` = null).
* `offset 24..31` (`Q`, `uint64`): `checksum` — 64-bit integrity checksum computed over bytes `0..23` concatenated with bytes `32..8191` (using `zlib.crc32` combined or `hashlib.blake2b(digest_size=8)`).

### 1.2 Checksum Verification Contract
* Whenever `DiskManager.write_page(page_id, buffer)` writes a page, it computes the checksum over `[0:24] + [32:8192]` and writes it into `[24:32]` prior to disk write.
* Whenever `DiskManager.read_page(page_id)` reads a page from disk, it recomputes the checksum and compares it against `[24:32]`. If they differ (and the page is not an uninitialized zero-filled page), raise `PageCorruptedError`.

---

## 2. Disk Manager (`src/storage/disk_manager.py`)

`DiskManager` manages a single binary file on disk (`*.db`, `*.col`, or `*.idx`).

### 2.1 File Layout & Free Space Map (FSM)
* **`page_id = 0` (FSM Page):** Reserved metadata/bitmap page tracking allocated vs. freed pages. Bit $i$ in the FSM payload (`bytes 32..8191`) tracks whether `page_id = i` is allocated (`1`) or free (`0`). Supports up to $8,160 \times 8 = 65,280$ pages ($510\text{ MB}$) per file in a single FSM page.
* **`allocate_page() -> int`:** Under `self._lock`, finds the lowest `page_id >= 1` with bit `0` in the FSM, sets it to `1`, extends the file on disk with `8,192` zero bytes if needed, flushes the FSM page, and returns the allocated `page_id`.
* **`deallocate_page(page_id: int) -> None`:** Clears bit `page_id` in the FSM and zeroes the page on disk.
* **`read_page(page_id: int) -> bytearray`:** Validates `0 <= page_id < num_pages`, reads exactly `8,192` bytes at offset `page_id * 8192`, verifies checksum, increments `disk_reads` counter, and returns a mutable `bytearray(8192)`.
* **`write_page(page_id: int, data: bytes | bytearray | memoryview) -> None`:** Validates `len(data) == 8192`, stamps checksum, writes at offset `page_id * 8192`, increments `disk_writes` counter.
* **`sync() -> None`:** Calls `os.fsync(self._fd)` to force OS kernel buffers to physical storage.

---

## 3. Row-Oriented Storage: `SlottedPage` (`src/storage/slotted_page.py`)

`SlottedPage` wraps an 8,192-byte `memoryview` / `bytearray` (`page_type = 1`).

### 3.1 Internal Byte Layout
```text
+--------------------+-----------------------------------+--------------------+
| PageHeader (32 B)  | Slot Directory (4 B * slot_count) |   Free Space ...   |
+--------------------+-----------------------------------+--------------------+
0                   32                               32+4N              free_space_ptr
                                                         +--------------------+
                                                         | ... Packed Tuples  |
                                                         +--------------------+
                                                       free_space_ptr        8192
```
* **Slot Entry (`<HH`, 4 bytes):** `(tuple_offset: uint16, tuple_length: uint16)`.
  * If a tuple is deleted, set `tuple_length = 0` (`tuple_offset = 0`) in its slot entry. **Never remove or shift slot entries**, because external indexes hold immutable `RID(page_id, slot_id)` pointers.
* **Tuple Binary Format:**
  * Bytes `0..15`: **16-byte MVCC System Header** (`<QQ`: `xmin: uint64 = 1`, `xmax: uint64 = 0`) reserved for Milestone 4.
  * Bytes `16..end`: Serialized user column values packed sequentially according to `Schema`.
* **Free Space Calculation:**
  $$\text{contiguous\_free} = \text{free\_space\_ptr} - (32 + 4 \times \text{slot\_count})$$
* **Defragmentation (`compact()`)**:
  * If `contiguous_free < required_bytes` (where `required_bytes = tuple_len + 4` for a new slot), compute total fragmented free space across deleted/shrunk tuples.
  * If `total_free >= required_bytes`, pack all live tuples contiguously toward byte `8192`, update `tuple_offset` in their respective slot entries, and update `free_space_ptr`.
* **Required Methods:**
  * `insert_tuple(tuple_bytes: bytes) -> int | None`: Returns `slot_id` if inserted, or `None` if insufficient space even after compaction.
  * `get_tuple(slot_id: int) -> bytes | None`: Returns raw tuple bytes (including or stripping the 16-byte MVCC header via helper) or `None` if deleted.
  * `update_tuple(slot_id: int, new_tuple_bytes: bytes) -> bool`: Updates in place if `len(new_tuple_bytes) <= old_len`; otherwise moves tuple if page has space (updating slot offset) or returns `False` if page overflow occurs (triggering relocation in `RowTableHeap`).
  * `delete_tuple(slot_id: int) -> bool`: Marks slot `(0, 0)` and returns `True`.

---

## 4. Column-Oriented Storage: `ColumnPage` (`src/storage/column_page.py`)

`ColumnPage` stores values for a **single column** of a columnar table (`page_type = 2`), enabling contiguous memory layout and zero-copy NumPy execution.

### 4.1 Internal Byte Layout
* **Fixed-Width Columns (`INT32`, `INT64`, `UINT64`, `FLOAT64`):**
  * Header (32 bytes): `slot_count` holds the number of stored values $N$; `free_space_ptr` holds `32 + N * item_size`.
  * Payload (`bytes 32 .. 32 + N * item_size`): Raw C-contiguous little-endian elements.
  * Capacity per page: $\lfloor (8192 - 32) / 8 \rfloor = 1,020$ values for 8-byte types (`INT64`, `FLOAT64`), or $2,040$ values for 4-byte types (`INT32`).
  * **Zero-Copy NumPy Access (`as_numpy_array(dtype) -> np.ndarray`)**:
    ```python
    end = 32 + self.slot_count * dtype.itemsize
    return np.frombuffer(self.buffer, dtype=dtype, count=self.slot_count, offset=32)
    ```
* **Variable-Length Columns (`VARCHAR`):**
  * Stores a 4-byte offset directory growing forward from byte `32` and UTF-8 bytes growing backward from `8192`.
* **`ColumnTableHeap` Coordination:**
  * Manages one `DiskManager` file per column (`order_items.order_id.col`, `order_items.qty.col`, etc.).
  * Aligns row ordinals across all columns so logical row index $r$ maps to identical `(page_idx, slot_id)` coordinates across every column file when all columns use the same chunk capacity (`CHUNK_ROWS = 1020`).

---

## 5. Buffer Pool Manager & Replacement Policies (`src/storage/buffer_pool.py`, `replacer.py`)

### 5.1 `ReplacementPolicy` Interface (`src/storage/replacer.py`)
```python
class ReplacementPolicy(ABC):
    @abstractmethod
    def victim(self) -> int | None: ...
    @abstractmethod
    def pin(self, frame_id: int) -> None: ...
    @abstractmethod
    def unpin(self, frame_id: int) -> None: ...
    @abstractmethod
    def size(self) -> int: ...
```
* **`LRUReplacer`:** Uses `OrderedDict[int, None]`. `unpin(frame_id)` adds `frame_id` to the most-recently-used end; `pin(frame_id)` removes `frame_id` if present; `victim()` pops and returns `next(iter(self._od))` (the least-recently-used unpinned frame).
* **`FIFOReplacer`:** Uses `OrderedDict[int, None]`. `unpin(frame_id)` inserts `frame_id` **only if not already present** (preserving original insertion order); `pin(frame_id)` removes `frame_id`; `victim()` pops the oldest inserted frame via `popitem(last=False)`.

### 5.2 `BufferPoolManager` State & Telemetry
* **Frame Metadata (`FrameHeader`):** `frame_id: int`, `page_id: int | None`, `pin_count: int`, `is_dirty: bool`, `data: bytearray(8192)`.
* **Core Operations (all protected by `self._lock = threading.RLock()`):**
  1. `fetch_page(page_id: int) -> memoryview`:
     * **Hit:** Increment `self.hit_count`, increment `frame.pin_count`, call `replacer.pin(frame_id)`, return `memoryview(frame.data)`.
     * **Miss:** Increment `self.miss_count`. Find a free frame from `free_list`, or call `replacer.victim()`. If both return `None`, raise `BufferPoolFullError`. If victim has `is_dirty == True`, call `disk_manager.write_page(victim.page_id, victim.data)` and increment `eviction_write_count`. Read `page_id` from `disk_manager`, set `pin_count = 1`, `is_dirty = False`, update `page_table`, and return `memoryview(frame.data)`.
  2. `unpin_page(page_id: int, is_dirty: bool = False) -> bool`:
     * Decrement `pin_count` (must not drop below `0`). Set `frame.is_dirty |= is_dirty`. When `pin_count == 0`, call `replacer.unpin(frame_id)`.
  3. `new_page() -> tuple[int, memoryview]`:
     * Allocates `page_id = disk_manager.allocate_page()`, claims a frame (`pin_count = 1`, `is_dirty = True`), initializes 8,192 zero bytes, and returns `(page_id, memoryview(frame.data))`.
  4. `flush_page(page_id: int) -> bool` & `flush_all_pages() -> None`:
     * Writes dirty frames to `disk_manager` and calls `disk_manager.sync()`.

---

## 6. Encrypted Object Store (`src/storage/object_store.py`, `cipher_hooks.py`)

### 6.1 Pluggable Encryption Hooks (`src/storage/cipher_hooks.py`)
```python
@dataclass(frozen=True)
class CipherPayload:
    nonce: bytes       # 12 bytes (zero-filled for NoOpCipher)
    tag: bytes         # 16 bytes (zero-filled for NoOpCipher)
    ciphertext: bytes  # Encrypted payload

class CipherHook(ABC):
    @property
    @abstractmethod
    def algorithm_id(self) -> int: ...
    @abstractmethod
    def encrypt(self, plaintext: bytes, aad: bytes) -> CipherPayload: ...
    @abstractmethod
    def decrypt(self, payload: CipherPayload, aad: bytes) -> bytes: ...
```
* **`NoOpCipher` (`algorithm_id = 0`):** Returns `CipherPayload(nonce=b"\x00"*12, tag=b"\x00"*16, ciphertext=plaintext)`.
* **`AES256GCMHook(key: bytes)` (`algorithm_id = 1`):** Validates `len(key) == 32`. Uses `AESGCM(key)` with `os.urandom(12)` nonce and `aad = struct.pack("<Q", oid)` to bind ciphertext to its object ID. Raises `SecurityViolationError` if `InvalidTag` is caught on decrypt.

### 6.2 Extent Binary Format (`objects.dat` + `objects.dir`)
* **`objects.dat` Record Envelope:**
  * Header (`<4sQII12s16s`, 48 bytes): `magic = b"OBJ1"`, `oid: uint64`, `raw_len: uint32`, `cipher_len: uint32`, `nonce: 12B`, `tag: 16B`.
  * Body: `ciphertext` (`cipher_len` bytes).
  * Trailer (`<I`, 4 bytes): `crc32` computed over header + ciphertext.
* **`objects.dir` Directory Mapping:**
  * Maintained in memory as `dict[int, ObjectMeta]` and persisted across restarts in a structured binary directory file (`objects.dir`) with 24-byte entries (`<QQII`: `oid`, `file_offset`, `cipher_len`, `raw_len`) and an 8-byte header (`<4sI`: `b"ODIR"`, `entry_count`).

---

## 7. Unified `StorageEngine` Facade (`src/storage/storage_engine.py`)

`StorageEngine` wraps `DiskManager`, `BufferPoolManager`, `RowTableHeap`, `ColumnTableHeap`, and `ObjectStore` behind one lifecycle interface (`open()`, `flush_all()`, `close()`, `get_stats()`) so M1 and M2 components share a single storage entry point.