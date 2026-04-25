# CNLIB Train Data Snapshot

This folder contains the original CNLIB training parquet files used by the UI
and by `scripts/generate_unseen_test_data.py`.

Keeping this snapshot in the repository prevents runtime failures when Python
loads a `cnlib` installation that is missing its bundled `data/` directory.
