# Known Limitations

- The remake now contains final datasets, final tables, and final runner scripts, but several numbered scripts still reflect historical experimentation stages and are not yet reduced to a minimal public-facing command surface.
- Replay-group-aware methodology is part of the canonical design, but a fresh freeze claim should still be tied to rerunning the explicit smoke/audit/check scripts before each new tagged release.
- GPU-backed XGBoost and PyTorch behavior may vary across environments, especially across CUDA, driver, and library combinations.
- Parser parity with older branches is not claimed automatically; it should only be claimed after representative replay-audit validation.
- Some public-facing repository polish remains outside file content alone, including GitHub About metadata, topics, and an explicit license decision.
