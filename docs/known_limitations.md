# Known limitations

- GPU-backed XGBoost and PyTorch behavior may vary across environments, especially across CUDA, driver, and library combinations.
- Parser parity with older branches is not claimed automatically; it should only be claimed after representative replay-audit validation.
- The upstream replay source is now documented, but exact reconstruction of the locally used corpus still depends on replaypack-level inventories, parser-side exclusions, and project filtering decisions.
- Some public-facing repository polish may still remain outside file content alone, including GitHub About metadata, topics, and any final license choice for repository-authored code and documentation.
