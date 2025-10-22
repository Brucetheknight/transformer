import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

from src.dab_analysis import (
    dab_mask_from_hed,
    gray_world_white_balance,
    illumination_flatten,
    make_roi_mask,
    overlay,
)

st.set_page_config(page_title="DAB Quant", layout="centered")
st.title("DAB 染色面积测量（显微视野）")

uploaded = st.file_uploader("上传一张图片（png/jpg/tif）", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
border = st.slider("边框忽略比例", 0.0, 0.2, 0.05, 0.01)
manual = st.text_input("手动阈值(0~1，可留空使用 Otsu)", "")
min_obj = st.number_input("最小连通域占比", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001, format="%.4f")
open_radius = st.slider("开运算半径", 0, 5, 2)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    img_u8 = img_np if img_np.dtype == np.uint8 else (img_np / img_np.max() * 255).astype(np.uint8)
    img_wb = gray_world_white_balance(img_u8)
    img_fl = illumination_flatten(img_wb)
    roi = make_roi_mask(img_fl.shape, border)
    thr = None if manual.strip() == "" else float(manual)
    mask, _, thr_used = dab_mask_from_hed(
        img_fl,
        roi,
        manual_thr=thr,
        min_obj_frac=min_obj,
        open_radius=open_radius,
    )
    vis = overlay(overlay(img_fl, roi, (0, 255, 0), 0.25), mask, (255, 0, 0), 0.60)
    st.image(vis, caption=f"阳性面积 = {mask.mean()*100:.2f}%  (thr={thr_used:.3f})")

    buf = BytesIO()
    Image.fromarray(vis).save(buf, format="PNG")
    st.download_button("下载叠图 PNG", data=buf.getvalue(), file_name="overlay.png", mime="image/png")
