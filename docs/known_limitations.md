# Known Limitations

- The clean-room remake is a framework skeleton plus baseline implementations; some experimental scripts may still require adaptation to the final frozen dataset schema.
- Matchup analysis depends on explicit matchup metadata being present in the processed dataset.
- GPU-backed XGBoost and PyTorch behavior may vary across environments.
- Some scripts are intentionally conservative and optimized for reproducibility over raw speed.
- The parser logic is newly rewritten and should be validated on representative replay subsets before claiming parity with any previous project branch.
