### Results (batch size 512)

| Arm | N | Recall | Precision | F1 | Best F1 | s/query | docs scored |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANN only | 500 | 0.983 | 0.982 | 0.982 | 0.991 | 0.00 | 0 |
| Classify (full scan) | 500 | 1.000 | 1.000 | 1.000 | 1.000 | 0.01 | 500 |
| Hybrid (ANN + classify) | 500 | 0.983 | 1.000 | 0.991 | 0.991 | 0.00 | 100 |
| ANN only | 1,000 | 0.501 | 1.000 | 0.667 | 0.667 | 0.00 | 0 |
| Classify (full scan) | 1,000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.02 | 1,000 |
| Hybrid (ANN + classify) | 1,000 | 0.501 | 1.000 | 0.667 | 0.667 | 0.00 | 100 |
| ANN only | 2,000 | 0.250 | 1.000 | 0.400 | 0.400 | 0.00 | 0 |
| Classify (full scan) | 2,000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.03 | 2,000 |
| Hybrid (ANN + classify) | 2,000 | 0.250 | 1.000 | 0.400 | 0.400 | 0.00 | 100 |

### Throughput against batch size (classify)

| Batch | docs/sec | prefill tok/sec |
|---:|---:|---:|
| 32 | 44,298.8 | 8,208,550 |
| 128 | 52,667.1 | 9,759,189 |
| 512 | 55,300.2 | 10,247,088 |

### Hybrid retrieval ceiling

- mean fraction of gold inside top-k: **0.578**
- mean hybrid recall: **0.578**
- queries where gold exceeds k: **80%**
