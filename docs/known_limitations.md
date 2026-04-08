# Known limitations

- GPU-backed XGBoost and PyTorch behavior may vary across environments, especially across CUDA, driver, and library combinations.
- Parser parity with older branches is not claimed automatically; it should only be claimed after representative replay-audit validation.
- The upstream replay source is now documented, but exact reconstruction of the locally used corpus still depends on replaypack-level inventories, parser-side exclusions, and project filtering decisions.
- Final repository polish is not fully captured by committed files alone: GitHub About metadata, topics, and the final chosen license still need to be set on the hosting platform if not already completed.
