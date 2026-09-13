# Baseline A: fixed-MaleCNS teacher-gap trajectory

Same 256 validation bytes and frozen teacher at every checkpoint. Single-seed feasibility; no topology-superiority claim.

| Additional updates | Total bytes seen | Val nats/byte | Teacher gap | KL (T=1) | Teacher argmax agreement | Zero-edge loss | Last gradient norm | Hidden RMS | Saturation | Training seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 3.1754 | 2.8504 | 2.8762 | 0.098 | 3.0815 | 2.822 | 0.1597 | 0.001394 | 191.9 |
| 64 | 2560 | 2.8628 | 2.5379 | 2.5925 | 0.137 | 2.9072 | 2.167 | 0.2179 | 0.002696 | 591.1 |
| 128 | 4608 | 2.5221 | 2.1971 | 2.2487 | 0.191 | 2.9045 | 1.880 | 0.2009 | 0.002751 | 939.3 |
| 256 | 8704 | 1.8719 | 1.5469 | 1.6012 | 0.395 | 2.9569 | 1.688 | 0.2141 | 0.011143 | 1599.9 |

Gradient norms are measured before clipping to 1.0. Saturation is abs(h)>0.95, sampled at byte boundaries. KL is forward KL(teacher || student), in nats per byte at T=1; the T=2 training-scaled KL is also retained in JSON. Training wall time excludes these separate diagnostic processes but can include resource contention.
