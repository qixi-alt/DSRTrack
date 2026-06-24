# DSRTrack

<p align="center">
  <b>Disturbance-Aware State Recovery and Volatility Regulation for RGB-T Tracking</b>
</p>

<p align="center">
  <img alt="Task" src="https://img.shields.io/badge/Task-RGB--T%20Tracking-blue">
  <img alt="Results" src="https://img.shields.io/badge/Results-Available-brightgreen">
  <img alt="Code" src="https://img.shields.io/badge/Code-Coming%20Soon-orange">
  <img alt="Benchmarks" src="https://img.shields.io/badge/Benchmarks-4-lightgrey">
</p>

📌 Official repository for **DSRTrack: Disturbance-Aware State Recovery and Volatility Regulation for RGB-T Tracking**.

🔓 The source code and pretrained models will be released upon paper acceptance. This repository currently provides tracking results and evaluation curves for reproducibility.

## ✨ Highlights

- Disturbance-aware state recovery for degraded online states.
- Volatility regulation to reduce the persistent influence of unreliable historical states.
- Evaluation results on four public RGB-T tracking benchmarks.

## 📊 Performance

| Dataset | PR | NPR | SR |
|---|---:|---:|---:|
| LasHeR | 79.1 | 75.2 | 62.8 |
| GTOT | 94.2 | - | 80.0 |
| RGBT210 | 92.5 | - | 68.2 |
| RGBT234 | 93.0 | - | 70.3 |

## 📈 Evaluation Curves

### LasHeR

![LasHeR evaluation curves](assets/LasHeR_curve.png)

### RGBT234

![RGBT234 evaluation curves](assets/RGBT234_curve.png)

## 📁 Tracking Results

Tracking results are provided on four public RGB-T tracking benchmarks:

```text
.
├── LasHeR/
├── GTOT/
├── RGBT210/
└── RGBT234/
```

## 🚀 Code

The source code and pretrained models will be released upon paper acceptance.

## 📝 Citation

Citation information will be updated after publication.
