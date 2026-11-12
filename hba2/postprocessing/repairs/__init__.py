"""
Fixes for data recorded or processed before a bug was found.

Each is a separate step run **after** post-processing, on sessions exactly as it left
them, so the normal pipeline never has to know a bug existed. Every repair records what
it did beside the files it changed and keeps the originals.
"""
