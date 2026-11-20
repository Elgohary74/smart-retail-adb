# 01 — Retail Orders Schema & Deterministic Data Specification

## 1. Domain Overview
The demonstration workload models an e-commerce **Retail Orders and Product Search** platform. It exercises both Online Transaction Processing (OLTP) point lookups/updates and Online Analytical Processing (OLAP) column scans, alongside encrypted binary storage for product images.

---

## 2. Supported Data Types & Binary Encoding (`src/common/types.py`)

Every column in a table schema uses one of five canonical data types:

| `DataType` Enum | Size (Bytes) | `struct` Format | Encoding & Representation Rules |
| :--- | :--- | :--- | :--- |
| `INT32` | 4 | `<i` | Signed 32-bit little-endian integer ($-2^{31}$ to $2^{31}-1$). |
| `INT64` | 8 | `<q` | Signed 64-bit little-endian integer (used for IDs and Unix epoch timestamps in ms). |
| `UINT64` | 8 | `<Q` | Unsigned 64-bit little-endian integer (used for `image_oid` and LSNs). |
| `FLOAT64` | 8 | `<d` | IEEE 754 double-precision floating-point number (prices, amounts). |
| `VARCHAR` | Variable ($2 + L$) | `<H` + `Ls` | In Row pages: 2-byte unsigned length $L$ followed by $L$ UTF-8 bytes ($L \le \text{max\_len}$). In fixed-width index keys: zero-padded to `max_len` bytes. |

### `RID` (Record Identifier) Representation
* Represented in Python as `@dataclass(frozen=True, order=True) class RID: page_id: int; slot_id: int`.
* Packed on disk as **8 bytes**: `struct.Struct("<II")` (`page_id: uint32`, `slot_id: uint32`).
* Null / Tombstone sentinel: `RID(0xFFFFFFFF, 0xFFFFFFFF)`.

---

## 3. Retail Relational & Object Schemas

### 3.1 Table 1: `customers` (OLTP Row Layout)
* **Storage File:** `data/tables/customers.db` (`LayoutType.ROW`)
* **Primary Key:** `customer_id` (B+ Tree index `idx_customers_pk`)
* **Columns:**
  1. `customer_id`: `INT64` (Primary Key, Not Null)
  2. `name`: `VARCHAR(64)` (Not Null)
  3. `region`: `VARCHAR(16)` (Values: `"EGYPT"`, `"MENA"`, `"EUROPE"`, `"NA"`, `"APAC"`)
  4. `tier`: `INT32` (Loyalty tier: `1`, `2`, `3`)

### 3.2 Table 2: `products` (OLTP Row Layout + Object Reference)
* **Storage File:** `data/tables/products.db` (`LayoutType.ROW`)
* **Primary Key:** `product_id` (B+ Tree index `idx_products_pk`)
* **Secondary Index:** `price` (B+ Tree index `idx_products_price` for range queries)
* **Columns:**
  1. `product_id`: `INT64` (Primary Key, Not Null)
  2. `category`: `VARCHAR(32)` (Values: `"Electronics"`, `"Apparel"`, `"Home"`, `"Books"`, `"Sports"`, `"Beauty"`)
  3. `price`: `FLOAT64` (Range: `5.00` to `2500.00`)
  4. `image_oid`: `UINT64` (Foreign reference to `ObjectStore`; `0` if no image attached)

### 3.3 Table 3: `inventory` (High-Write OLTP Row Layout)
* **Storage File:** `data/tables/inventory.db` (`LayoutType.ROW`)
* **Primary Key:** `product_id` (LSM Tree index `idx_inventory_pk` — optimized for frequent stock updates)
* **Columns:**
  1. `product_id`: `INT64` (Primary Key, Not Null)
  2. `stock_qty`: `INT32` (Non-negative stock count)
  3. `warehouse_id`: `INT32` (Range: `1` to `5`)
  4. `updated_ts`: `INT64` (Unix timestamp in milliseconds)

### 3.4 Table 4: `orders` (OLTP Row Layout)
* **Storage File:** `data/tables/orders.db` (`LayoutType.ROW`)
* **Primary Key:** `order_id` (B+ Tree index `idx_orders_pk`)
* **Secondary Index:** `order_date` (B+ Tree index `idx_orders_date` for date-range analytics)
* **Columns:**
  1. `order_id`: `INT64` (Primary Key, Not Null)
  2. `customer_id`: `INT64` (Foreign key referencing `customers.customer_id`)
  3. `order_date`: `INT64` (Unix timestamp in milliseconds)
  4. `total_amt`: `FLOAT64` (Order total in USD)

### 3.5 Table 5: `order_items` (OLAP Columnar Layout)
* **Storage Files:** `data/tables/order_items.<col_name>.col` (`LayoutType.COLUMN`)
* **Index:** `order_id` (LSM Tree index `idx_order_items_order_id` — append-heavy ingestion)
* **Columns:**
  1. `order_id`: `INT64`
  2. `item_seq`: `INT32`
  3. `product_id`: `INT64`
  4. `qty`: `INT32`
  5. `unit_price`: `FLOAT64`

### 3.6 Object Store Entity: `product_images`
* **Storage Files:** `data/objects/objects.dat` and `data/objects/objects.dir`
* **Key:** `oid: uint64` stored in `products.image_oid`.
* **Payload:** Synthetic binary image payload (`10 KB` to `250 KB` per product image) encrypted via `AES256GCMHook`.

---

## 4. Persistent Catalog Metadata (`catalog.json`)

The `CatalogManager` persists table definitions, file paths, row counts, page counts, and active indexes to `data/catalog.json` using atomic file replacement (`write temp -> fsync -> os.replace`).

```json
{
  "version": 1,
  "tables": {
    "products": {
      "layout": "ROW",
      "file_path": "data/tables/products.db",
      "primary_key": "product_id",
      "row_count": 5000,
      "page_count": 48,
      "columns": [
        {"name": "product_id", "dtype": "INT64", "max_len": 8, "nullable": false},
        {"name": "category", "dtype": "VARCHAR", "max_len": 32, "nullable": false},
        {"name": "price", "dtype": "FLOAT64", "max_len": 8, "nullable": false},
        {"name": "image_oid", "dtype": "UINT64", "max_len": 8, "nullable": false}
      ],
      "indexes": {
        "idx_products_pk": {"column": "product_id", "type": "BPLUS_TREE", "unique": true},
        "idx_products_price": {"column": "price", "type": "BPLUS_TREE", "unique": false}
      }
    }
  }
}
```

---

## 5. Deterministic Dataset Generator (`bench/retail_loader.py`)

Implement `RetailDataGenerator(seed: int = 42)` using `random.Random(seed)` and `numpy.random.default_rng(seed)` so every test and benchmark run is 100% reproducible.

### 5.1 Standard Dataset Scale (`SCALE = "standard"`)
* **`customers`:** `10,000` records (`customer_id` from `1` to `10,000`).
* **`products`:** `5,000` records (`product_id` from `1` to `5,000`).
* **`inventory`:** `5,000` records (`1:1` matching `product_id` `1` to `5,000`, initial `stock_qty` between `50` and `500`).
* **`orders`:** `50,000` records (`order_id` from `1` to `50,000`, `customer_id` sampled using Zipfian skew $\theta = 0.9$ so 20% of customers generate ~80% of orders, `order_date` uniformly distributed across 365 days of 2026).
* **`order_items`:** `200,000` records (`4` items per order on average, `product_id` sampled with Zipfian skew $\theta = 0.9$).
* **`product_images`:** `500` synthetic binary payloads (`16 KB` to `128 KB` each, prefixed with valid PNG magic header `\x89PNG\r\n\x1a\n` followed by deterministic pseudo-random bytes) linked to `product_id` `1` through `500`.

### 5.2 Fast Unit-Test Scale (`SCALE = "tiny"`)
Used inside `pytest` for sub-second execution:
* `500` customers, `250` products, `250` inventory rows, `2,000` orders, `8,000` order items, `25` product images.