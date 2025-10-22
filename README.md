# Stripe Rust DAB Quant – 显微镜下 H₂O₂-DAB 染色视野面积测量

本仓库提供一条**可复现**的 DAB 染色面积量化流水线：  
灰世界白平衡 → 轻量照明校正 → **HED 颜色去卷积（取 DAB 通道）** → **Otsu 自适应阈值** → 形态学清噪 → 面积百分比与叠图导出。

- 方法依据：`rgb2hed`（Hematoxylin–Eosin–DAB 去卷积）与 Otsu 阈值分割，详见 scikit-image 官方文档与阈值指南。  
- 一键分享：nbconvert 导出 HTML/PDF；Binder/Colab 在线运行；Streamlit 网页交互端。  
参考：scikit-image `rgb2hed`、Otsu 阈值、nbconvert、Binder、Streamlit、README/CITATION 规范。[^refs]

---

## 快速开始

### Conda
```bash
conda env create -f environment.yml
conda activate stripe-rust-dab
python -m src.dab_analysis --input sample_data --out results_demo
# 单张图：python -m src.dab_analysis --input "D:\你的图片.jpg" --out results_demo
```

### Pip（可选）
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.dab_analysis --input sample_data --out results_demo
```

### Streamlit 网页
```bash
streamlit run app/streamlit_app.py
```

### 导出 HTML/PDF 报告
```bash
jupyter nbconvert --to html --execute notebooks/DAB_demo.ipynb
# 或导出 PDF（本地需 LaTeX 依赖）
```

## 命令行用法
```bash
python -m src.dab_analysis \
  --input path/to/image_or_dir \
  --out path/to/outdir \
  --border-frac 0.05 \
  --manual-thr None \
  --min-obj-frac 1e-4 \
  --open-radius 2
```

- `--input`：单张图片或目录（自动遍历 png/jpg/jpeg/tif/bmp）
- `--border-frac`：忽略四周边框比例，避开载玻片/标尺
- `--manual-thr`：覆盖自动阈值（0~1），默认使用 Otsu
- `--min-obj-frac`：最小连通域占比阈值，过滤噪声
- `--open-radius`：形态学开运算半径

输出：`*_overlay.png`（绿=ROI，红=阳性区域）、`results.csv`（面积百分比、阈值等）

## 方法说明（极简）

- 颜色去卷积：`rgb2hed` 将 RGB 分离成 H/E/DAB 通道；直接使用 DAB 通道做后续阈值。
- 阈值：Otsu 在 DAB 直方图上自动找阈值（最大类间方差）。
- 形态学清噪：开运算 + 去小连通域，获得干净掩膜。
- 计量：对 ROI（默认去除四周 5%）统计阳性像素比例，得到视野面积百分比。

## 目录说明

- `src/dab_analysis.py`：核心算法 + 命令行接口
- `app/streamlit_app.py`：网页交互（上传图片→返回叠图与百分比）
- `sample_data/`：示例图（请自行放 1–3 张，避免隐私）
- `results_demo/`：输出示例目录
- `notebooks/`：放演示/报告 Notebook（可 nbconvert 导出 HTML/PDF）

## 引用本仓库

请使用 `CITATION.cff` 的条目（在 GitHub 会显示“Cite this repository”按钮）。

## 许可证

MIT（见 `LICENSE`）。

---

### 在线分享（可选）

- **Binder**：到 https://mybinder.org 填你的仓库链接（使用本仓库 `environment.yml` 就行），生成一个徽章粘到 README。
- **Colab**：把 `.ipynb` 放仓库/Drive，首格 `!pip install scikit-image matplotlib`，分享链接任何人可运行。
- **Streamlit Community Cloud**：直接连 GitHub 仓库部署，免费生成访问 URL。

---

[^refs]: scikit-image `rgb2hed`（Hematoxylin–Eosin–DAB）与颜色空间；scikit-image 阈值分割 & Otsu 方法；Jupyter nbconvert 导出 HTML/PDF/Markdown；Binder 在线运行指南；Streamlit Community Cloud 部署；GitHub README/CITATION 最佳实践。
