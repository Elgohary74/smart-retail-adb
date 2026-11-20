"""
Module: src/__init__.py
Layer: Package Root
Lock Domain: Stateless

Purpose:
    Marks ``src`` as the root Python package for the ``csai302_db`` custom
    relational and object database management system.

Key Invariants:
    1. Only approved Python standard library modules, ``numpy``, and
       ``cryptography`` may be imported anywhere under ``src/``; external
       database engines and object serialization shortcuts are prohibited.
"""

__version__: str = "0.1.0"
