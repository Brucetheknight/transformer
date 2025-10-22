# Notebooks

- 把你的演示/报告 Notebook 放在这里，并用 nbconvert 导出 HTML/PDF。
- 命令：
  - `jupyter nbconvert --to html --execute DAB_demo.ipynb`
  - `jupyter nbconvert --to pdf DAB_demo.ipynb`
- nbconvert 文档：支持 HTML、PDF、Markdown 等多种格式。

## 如何推送到 GitHub

```bash
cd stripe-rust-dab
git init
git add .
git commit -m "Initial commit: DAB quant pipeline"
git branch -M main
git remote add origin https://github.com/<yourname>/stripe-rust-dab.git
git push -u origin main
```

写好 README 是官方推荐的最佳实践；加上 LICENSE、CITATION 更专业。

## 在线分享（可选）

- Binder：到 https://mybinder.org 填写仓库链接并使用 `environment.yml`。
- Colab：Notebook 首格添加 `!pip install scikit-image matplotlib`。
- Streamlit Cloud：直接连接 GitHub 仓库部署网页端。
