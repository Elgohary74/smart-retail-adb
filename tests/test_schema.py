import unittest

from smartdb.schema import (
    ColumnSpec,
    DataType,
    TableSchema,
    decode_key,
    deserialize_row,
    encode_key,
    serialize_row,
)


class SchemaTests(unittest.TestCase):
    def test_schema_rejects_bool_as_int_and_orders_distinctly(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("flag", DataType.BOOL),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("vec", DataType.VECTOR, dim=3),
        ])
        with self.assertRaises(TypeError):
            schema.validate_record({"id": True, "flag": True, "price": 1.0, "vec": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            schema.validate_record({"id": 1, "flag": True, "price": float("nan"), "vec": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            schema.validate_record({"id": 1, "flag": True, "price": 1.0, "vec": [1.0, 2.0]})

        self.assertLess(encode_key(DataType.ANY, None), encode_key(DataType.ANY, False))
        self.assertLess(encode_key(DataType.ANY, False), encode_key(DataType.ANY, True))
        self.assertNotEqual(encode_key(DataType.ANY, True), encode_key(DataType.ANY, 1))
        self.assertEqual(encode_key(DataType.ANY, 1), encode_key(DataType.ANY, 1.0))

    def test_lossless_int64_and_float64_key_ordering(self):
        vals = [-2**62, -10, -1.5, -1, -0.25, 0, 0.25, 1, 1.5, 2**53, 2**53 + 1, 2**62]
        encoded = [encode_key(DataType.ANY, v) for v in vals]
        self.assertEqual(encoded, sorted(encoded))
        self.assertEqual(len(set(encoded)), len(vals))
        for v, raw in zip(vals, encoded):
            self.assertEqual(decode_key(DataType.ANY, raw), v)

    def test_binary_row_round_trip_and_projection(self):
        schema = TableSchema([
            ColumnSpec("id", DataType.INT, nullable=False),
            ColumnSpec("name", DataType.STR),
            ColumnSpec("price", DataType.FLOAT),
            ColumnSpec("active", DataType.BOOL),
            ColumnSpec("emb", DataType.VECTOR, dim=2),
        ])
        row = {"id": 42, "name": "Keyboard ⌨️", "price": 29.99, "active": False, "emb": [0.5, -1.25]}
        blob = serialize_row(schema, schema.validate_record(row))
        decoded = deserialize_row(schema, blob)
        self.assertEqual(decoded["id"], 42)
        self.assertEqual(decoded["name"], "Keyboard ⌨️")
        self.assertAlmostEqual(decoded["price"], 29.99)
        self.assertIs(decoded["active"], False)
        self.assertEqual(decoded["emb"], [0.5, -1.25])
        proj = deserialize_row(schema, blob, projected=["name", "active"])
        self.assertEqual(proj, {"name": "Keyboard ⌨️", "active": False})


if __name__ == "__main__":
    unittest.main()
