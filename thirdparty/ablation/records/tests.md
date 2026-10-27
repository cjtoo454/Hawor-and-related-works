# TWM-only fixture test record

Date: 2026-10-20

Command:

```bash
.env/hawor/bin/python -m unittest thirdparty/ablation/twm_only/test_twm_only.py
```

Observed output summary:

```text
........
Ran 8 tests in 5.276s
OK
```

The fixture suite covered restricted loading of real-format `save_checkpoint` files for both supported stages, a short synthetic TWM sequence, one-call-per-frame and raw-history behavior, zero-length history, frozen-protocol onset matching, prediction/details hash binding, formal table-row guards, and `--help` for all three entry points. This is the captured unittest output summary, not a reconstructed full log. No test was rerun to create this record.
