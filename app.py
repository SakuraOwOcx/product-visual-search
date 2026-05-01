from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.product_search.clip_engine import encode_pil_image, load_clip_model
from src.product_search.config import (
    CLIP_FULL_INDEX_PATH,
    DEVICE,
    FIGURE_DIR,
    FULL_SPLIT_CSV,
    INDEX_PATH,
    LOCAL_CLIP_CHECKPOINT,
    MODEL_DISPLAY_NAME,
    REPORT_DIR,
    RESNET18_FULL_BEST_CHECKPOINT,
    RESNET18_FULL_INDEX_PATH,
    VIT_SUPERVISED_BEST_CHECKPOINT,
    VIT_SUPERVISED_INDEX_PATH,
)
from src.product_search.data_utils import get_full_gallery_query_dataframes, get_gallery_query_dataframes, load_full_split_dataframe
from src.product_search.index_utils import build_clip_full_gallery_index, build_gallery_index, load_visual_search_index
from src.product_search.path_utils import resolve_project_path, to_project_relative
from src.product_search.resnet_engine import encode_resnet_pil_image, load_resnet18_full_model
from src.product_search.search_engine import cosine_top_k
from src.product_search.vit_engine import encode_vit_pil_image, load_vit_supervised_model


st.set_page_config(page_title="Product Visual Search Demo", layout="wide")


@st.cache_resource(show_spinner=False)
def get_clip_resources():
    return load_clip_model()


@st.cache_resource(show_spinner=False)
def get_resnet_resources():
    return load_resnet18_full_model()


@st.cache_resource(show_spinner=False)
def get_vit_resources():
    return load_vit_supervised_model()


@st.cache_data(show_spinner=False)
def get_dataframes():
    return get_gallery_query_dataframes()


@st.cache_data(show_spinner=False)
def get_cached_visual_index(index_path):
    return load_visual_search_index(Path(index_path))


def reset_cached_index():
    get_cached_visual_index.clear()


def load_image_from_path(path):
    return Image.open(resolve_project_path(path)).convert("RGB")


def show_model_comparison():
    summary_path = REPORT_DIR / "experiment_summary.csv"
    recall_fig = FIGURE_DIR / "model_recall_comparison.png"

    st.subheader("Model Comparison")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        display_cols = [
            "model_name",
            "num_classes",
            "train_size",
            "test_size",
            "top1_acc",
            "top5_acc",
            "recall@1",
            "recall@5",
            "recall@10",
        ]
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No experiment summary CSV found yet.")

    if recall_fig.exists():
        st.image(str(recall_fig), caption="ResNet18 vs CLIP Recall@K comparison")


def render_results(results_df, query_class=None, query_image_id=None):
    if results_df.empty:
        st.warning("No retrieved products were returned.")
        return

    cards_per_row = 5
    for start in range(0, len(results_df), cards_per_row):
        cols = st.columns(cards_per_row)
        for col, (_, row) in zip(cols, results_df.iloc[start:start + cards_per_row].iterrows()):
            with col:
                image_path = resolve_project_path(row["image_path"])
                st.image(str(image_path), use_container_width=True)
                match_text = ""
                if query_class is not None:
                    match_text = "match" if row["class_name"] == query_class else "different"
                    match_text = f" | {match_text}"
                same_id = ""
                if query_image_id is not None:
                    same_id = " | exact same image" if str(row.get("image_id")) == str(query_image_id) else ""
                st.markdown(f"**Rank {int(row['rank'])}**")
                st.caption(f"{row['class_name']}{match_text}{same_id}")
                st.caption(f"Similarity: {row['similarity']:.4f}")
                st.caption(f"ID: {row.get('image_id', image_path.stem)}")


def get_resnet_query_dataframe():
    if not FULL_SPLIT_CSV.exists():
        return pd.DataFrame()
    split_df = load_full_split_dataframe()
    query_df = split_df[split_df["split"] == "test"].copy()
    query_df = query_df.rename(columns={"articleType": "class_name"})
    return query_df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_full_query_dataframe():
    _, query_df, _ = get_full_gallery_query_dataframes()
    return query_df.reset_index(drop=True)


def resnet_full_ready(index_info):
    if index_info is None:
        return False, "ResNet18 full index is missing."
    scope = str(index_info.get("index_scope", ""))
    if scope.startswith("smoke"):
        return False, "ResNet18 index is only a smoke-test index, not the completed full train gallery index."
    if not RESNET18_FULL_BEST_CHECKPOINT.exists():
        return False, "ResNet18 full checkpoint is missing."
    try:
        import torch

        ckpt = torch.load(RESNET18_FULL_BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        if ckpt.get("is_smoke_test") or ckpt.get("train_config", {}).get("is_smoke_test"):
            return False, "ResNet18 checkpoint is a smoke-test checkpoint, not a completed full model."
    except Exception as exc:
        return False, f"Could not validate ResNet18 checkpoint: {exc}"
    return True, ""


def vit_supervised_ready(index_info):
    if index_info is None:
        return False, "ViT supervised index is missing."
    scope = str(index_info.get("index_scope", ""))
    if scope.startswith("smoke"):
        return False, "ViT index is only a smoke-test index, not a completed full train gallery index."
    if not VIT_SUPERVISED_BEST_CHECKPOINT.exists():
        return False, "ViT supervised checkpoint is missing."
    try:
        import torch

        ckpt = torch.load(VIT_SUPERVISED_BEST_CHECKPOINT, map_location="cpu", weights_only=False)
        if ckpt.get("is_smoke_test") or ckpt.get("train_config", {}).get("is_smoke_test"):
            return False, "ViT checkpoint is a smoke-test checkpoint, not a completed supervised model."
    except Exception as exc:
        return False, f"Could not validate ViT checkpoint: {exc}"
    return True, ""


def main():
    st.title("Product Visual Search Demo")
    st.write(
        "Upload a product image or sample an existing query image. "
        "The app uses frozen CLIP embeddings and cosine similarity to retrieve visually similar products."
    )

    with st.sidebar:
        st.header("Search Settings")
        model_choice = st.selectbox(
            "Retrieval model",
            ["CLIP ViT-B/32 frozen", "ResNet18 supervised", "ViT-B/16 supervised"],
            index=0,
        )
        top_k = st.selectbox("Top-K results", [5, 10, 20], index=0)

        st.divider()
        st.write("Index status")
        if model_choice.startswith("CLIP"):
            selected_index_path = CLIP_FULL_INDEX_PATH
        elif model_choice.startswith("ResNet18"):
            selected_index_path = RESNET18_FULL_INDEX_PATH
        else:
            selected_index_path = VIT_SUPERVISED_INDEX_PATH
        st.code(to_project_relative(selected_index_path), language="text")
        current_index_info = load_visual_search_index(selected_index_path)
        if current_index_info is not None:
            st.success("Cached index found.")
            st.caption(f"Scope: {current_index_info['index_scope']}")
            st.caption(f"Gallery size: {current_index_info['gallery_size']}")
            st.caption(f"Embedding dim: {current_index_info['embedding_dim']}")
        else:
            st.warning("No cached index found for the selected model.")

        if model_choice.startswith("CLIP") and st.button("Build / refresh full CLIP index"):
            if not LOCAL_CLIP_CHECKPOINT.exists():
                st.error(f"Local CLIP checkpoint not found: {to_project_relative(LOCAL_CLIP_CHECKPOINT)}")
            else:
                with st.spinner("Building CLIP full train-gallery index. No CLIP training, no model download."):
                    summary = build_clip_full_gallery_index()
                    reset_cached_index()
                st.success("CLIP full train-gallery index built.")
                st.json(summary)

        if model_choice.startswith("ResNet18"):
            st.info(
                "ResNet18 Full Dataset requires the full training and index scripts. "
                "The app will not train or build the full index automatically."
            )
        if model_choice.startswith("ViT"):
            st.info(
                "ViT-B/16 supervised requires local fine-tuning and index building first. "
                "The app will not train ViT automatically."
            )

        show_comparison = st.checkbox("Show model comparison", value=False)

    if model_choice.startswith("CLIP") and not LOCAL_CLIP_CHECKPOINT.exists():
        st.error(
            "Local CLIP checkpoint is missing. This app does not download models automatically. "
            f"Expected checkpoint: {to_project_relative(LOCAL_CLIP_CHECKPOINT)}"
        )
        return

    if show_comparison:
        show_model_comparison()

    if model_choice.startswith("CLIP"):
        try:
            query_df = get_full_query_dataframe()
            label_column = "articleType"
        except Exception as exc:
            st.error(f"Could not load full shared query split: {exc}")
            return
        selected_index_path = CLIP_FULL_INDEX_PATH
        model_note = "Using shared full test query split. CLIP searches the full train gallery if the full CLIP index exists."
    elif model_choice.startswith("ResNet18"):
        query_df = get_resnet_query_dataframe()
        gallery_df = pd.DataFrame()
        label_column = "articleType"
        selected_index_path = RESNET18_FULL_INDEX_PATH
        model_note = "Using ResNet18 Full Dataset mode."
    else:
        query_df = get_resnet_query_dataframe()
        gallery_df = pd.DataFrame()
        label_column = "articleType"
        selected_index_path = VIT_SUPERVISED_INDEX_PATH
        model_note = "Using supervised ViT-B/16 mode."

    st.caption(
        f"{model_note} Query candidates: {len(query_df)} | Label column: {label_column} | Device: {DEVICE}"
    )

    upload_col, sample_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    with sample_col:
        if st.button("Random sample query"):
            if query_df.empty:
                st.warning("No query dataframe is available yet. Run the split script first.")
            else:
                st.session_state["sample_query"] = query_df.sample(1).iloc[0].to_dict()

    query_image = None
    query_class = None
    query_source = None

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert("RGB")
        query_source = uploaded_file.name
        query_image_id = None
        st.session_state.pop("sample_query", None)
    elif "sample_query" in st.session_state:
        sample = st.session_state["sample_query"]
        query_image = load_image_from_path(sample["image_path"])
        query_class = sample["class_name"]
        query_source = sample["image_path"]
        query_image_id = sample.get("image_id")
    else:
        query_image_id = None

    search_clicked = st.button("Search", type="primary")

    if query_image is None:
        st.info("Upload an image or click Random sample query to begin.")
        return

    left, right = st.columns([1, 3])
    with left:
        st.subheader("Query Image")
        st.image(query_image, use_container_width=True)
        if query_class:
            st.caption(f"articleType: {query_class}")
        if query_source:
            st.caption(f"source: {query_source}")

    with right:
        st.subheader(f"Top-{top_k} Retrieved Products")
        if not search_clicked:
            st.info("Click Search to run retrieval.")
            return
        index_info = load_visual_search_index(selected_index_path)
        if index_info is None:
            if model_choice.startswith("ResNet18"):
                st.warning(
                    "ResNet18 full model or index is not available yet. Please run the full training pipeline first."
                )
                st.code(
                    """
python scripts/prepare_full_dataset_split.py
python scripts/train_resnet18_full.py --epochs 10
python scripts/evaluate_resnet18_full.py
python scripts/build_resnet18_full_index.py
python scripts/evaluate_resnet18_full_retrieval.py
                    """.strip(),
                    language="text",
                )
            elif model_choice.startswith("ViT"):
                st.warning("ViT supervised checkpoint or index is not available yet. Please run the ViT pipeline first.")
                st.code(
                    """
python scripts/train_vit_supervised.py --epochs 3 --batch-size 16
python scripts/evaluate_vit_supervised.py --update-summary
python scripts/build_vit_supervised_index.py
python scripts/evaluate_vit_supervised_retrieval.py --update-summary
                    """.strip(),
                    language="text",
                )
            else:
                st.warning("Please build the CLIP full gallery index from the sidebar first.")
                st.code(
                    "python .\\scripts\\build_clip_full_index.py",
                    language="powershell",
                )
            return

        if model_choice.startswith("ResNet18"):
            ready, message = resnet_full_ready(index_info)
            if not ready:
                st.warning(
                    "ResNet18 Full Dataset is not ready for product demo search yet. "
                    f"{message}"
                )
                st.code(
                    "python .\\scripts\\train_resnet18_full.py --epochs 10\n"
                    "python .\\scripts\\build_resnet18_full_index.py",
                    language="powershell",
                )
                return
        if model_choice.startswith("ViT"):
            ready, message = vit_supervised_ready(index_info)
            if not ready:
                st.warning(f"ViT-B/16 supervised is not ready for product demo search yet. {message}")
                st.code(
                    "python .\\scripts\\train_vit_supervised.py --epochs 3 --batch-size 16\n"
                    "python .\\scripts\\build_vit_supervised_index.py",
                    language="powershell",
                )
                return

        with st.spinner("Encoding query image and searching cached gallery index..."):
            visual_index = get_cached_visual_index(str(selected_index_path))
            gallery_embeddings = visual_index["embeddings"]
            gallery_metadata = visual_index["metadata"]
            if model_choice.startswith("CLIP"):
                model, preprocess = get_clip_resources()
                query_embedding = encode_pil_image(model, preprocess, query_image)
            elif model_choice.startswith("ResNet18"):
                model, transform, _ = get_resnet_resources()
                query_embedding = encode_resnet_pil_image(model, transform, query_image)
            else:
                model, transform, _ = get_vit_resources()
                query_embedding = encode_vit_pil_image(model, transform, query_image)
            results_df = cosine_top_k(query_embedding, gallery_embeddings, gallery_metadata, top_k=top_k)

        st.caption(
            f"Model: {model_choice} | Index scope: {visual_index['index_scope']} | "
            f"Gallery size: {visual_index['gallery_size']} | Embedding dim: {visual_index['embedding_dim']}"
        )
        render_results(results_df, query_class=query_class, query_image_id=query_image_id)
        return

    st.divider()
    st.markdown(
        """
        **Model notes.** CLIP and ResNet18 now use the same full train/test split in the product demo.
        Random sample queries come from the shared full test split, while both retrieval models search train-gallery indexes.
        This prevents the same random query image from being a query for one model and a gallery item for another model.
        """
    )


if __name__ == "__main__":
    main()
