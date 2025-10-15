import io
import os
from datetime import datetime
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shortcuts import add_shortcuts, shortcut_button

# --------- Helpers: Data Loading ---------
def _standardize_sample_to_xy(sample: Union[np.ndarray, Dict, List]) -> Optional[np.ndarray]:
    """
    Convert a single sample into an (N, 2) float64 numpy array [x, y].

    Accepts common shapes:
    - (N, 2)
    - dict with keys 'x' and 'y' (same length)
    - list/tuple of length-2 iterables
    Returns None if the sample cannot be interpreted.
    """
    if sample is None:
        return None

    # dict-like with 'x' and 'y'
    if isinstance(sample, dict):
        if 'x' in sample and 'y' in sample:
            x = np.asarray(sample['x']).reshape(-1)
            y = np.asarray(sample['y']).reshape(-1)
            if x.shape[0] == y.shape[0] and x.shape[0] > 0:
                return np.stack([x, y], axis=1).astype(np.float64)
            return None

    # numpy array
    if isinstance(sample, np.ndarray):
        arr = sample
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0:
            return arr.astype(np.float64)
        # maybe it's (2, N)
        if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] > 0:
            return arr.T.astype(np.float64)
        # maybe it's a list of pairs but as object array
        if arr.ndim == 1 and arr.dtype == object:
            try:
                stacked = np.vstack([np.asarray(p, dtype=np.float64).reshape(2) for p in arr])
                if stacked.ndim == 2 and stacked.shape[1] == 2:
                    return stacked
            except Exception:
                return None
        return None

    # list/tuple of pairs
    if isinstance(sample, (list, tuple)):
        try:
            stacked = np.vstack([np.asarray(p, dtype=np.float64).reshape(2) for p in sample])
            if stacked.ndim == 2 and stacked.shape[1] == 2:
                return stacked
        except Exception:
            return None

    return None


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> List[np.ndarray]:
    """
    Load dataset from uploaded bytes based on extension.
    Supported: .npy, .npz, .jsonl, .json, .csv, .txt

    Returns list of samples, each as (N, 2) np.ndarray.
    """
    name_lower = filename.lower()
    buf = io.BytesIO(file_bytes)

    samples: List[np.ndarray] = []

    if name_lower.endswith('.npy'):
        arr = np.load(buf, allow_pickle=True)
        # Accept a list/array of samples
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            for s in arr:
                xy = _standardize_sample_to_xy(s)
                if xy is not None:
                    samples.append(xy)
        elif isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[2] == 2:
            for i in range(arr.shape[0]):
                xy = _standardize_sample_to_xy(arr[i])
                if xy is not None:
                    samples.append(xy)
        else:
            # Try treat as a single sample
            xy = _standardize_sample_to_xy(arr)
            if xy is not None:
                samples.append(xy)

    elif name_lower.endswith('.npz'):
        with np.load(buf, allow_pickle=True) as data:
            # Heuristic: if 'samples' key exists, prefer it
            if 'samples' in data:
                arr = data['samples']
                if isinstance(arr, np.ndarray) and arr.dtype == object:
                    for s in arr:
                        xy = _standardize_sample_to_xy(s)
                        if xy is not None:
                            samples.append(xy)
                elif arr.ndim == 3 and arr.shape[2] == 2:
                    for i in range(arr.shape[0]):
                        xy = _standardize_sample_to_xy(arr[i])
                        if xy is not None:
                            samples.append(xy)
            else:
                # Try all arrays
                for key in data.files:
                    arr = data[key]
                    xy = _standardize_sample_to_xy(arr)
                    if xy is not None:
                        samples.append(xy)

    elif name_lower.endswith('.jsonl'):
        text = buf.read().decode('utf-8')
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = pd.read_json(io.StringIO(line), typ='series')
                xy = _standardize_sample_to_xy(obj.to_dict())
            except ValueError:
                # Fallback: try eval-like with pandas json reader
                try:
                    df = pd.read_json(io.StringIO(line), typ='series')
                    xy = _standardize_sample_to_xy(df.to_dict())
                except Exception:
                    xy = None
            if xy is not None:
                samples.append(xy)

    elif name_lower.endswith('.json'):
        df = pd.read_json(buf)
        # Could be list of dict samples or a dict of arrays
        if isinstance(df, pd.DataFrame) and {'x', 'y'}.issubset(df.columns):
            xy = _standardize_sample_to_xy({'x': df['x'].to_numpy(), 'y': df['y'].to_numpy()})
            if xy is not None:
                samples.append(xy)
        else:
            try:
                records = df.to_dict(orient='records')
                for rec in records:
                    xy = _standardize_sample_to_xy(rec)
                    if xy is not None:
                        samples.append(xy)
            except Exception:
                pass

    elif name_lower.endswith('.csv') or name_lower.endswith('.txt'):
        # Expect either two columns x,y or a JSON-in-cell per row
        df = pd.read_csv(buf)
        if {'x', 'y'}.issubset(df.columns):
            xy = _standardize_sample_to_xy({'x': df['x'].to_numpy(), 'y': df['y'].to_numpy()})
            if xy is not None:
                samples.append(xy)
        else:
            # Try per-row object
            for _, row in df.iterrows():
                rec = row.to_dict()
                xy = _standardize_sample_to_xy(rec)
                if xy is not None:
                    samples.append(xy)

    # Deduplicate any Nones and ensure list
    return samples


# --------- Helpers: Labels Persistence ---------
def derive_default_labels_path(dataset_name: str) -> str:
    base = os.path.splitext(os.path.basename(dataset_name))[0]
    return f"{base}_labels.csv"


def load_existing_labels(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Normalize expected columns
            expected = {'index', 'label'}
            if not expected.issubset(set(df.columns)):
                return pd.DataFrame(columns=['index', 'label', 'timestamp', 'notes'])
            df = df[['index', 'label'] + [c for c in ['timestamp', 'notes'] if c in df.columns]]
            df['index'] = df['index'].astype(int)
            df['label'] = df['label'].astype(int)
            return df
        except Exception:
            return pd.DataFrame(columns=['index', 'label', 'timestamp', 'notes'])
    return pd.DataFrame(columns=['index', 'label', 'timestamp', 'notes'])


def upsert_label(path: str, sample_index: int, label: int, notes: str = "") -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df = load_existing_labels(path)
    ts = datetime.now().isoformat(timespec='seconds')
    if (df['index'] == sample_index).any():
        df.loc[df['index'] == sample_index, ['label', 'timestamp', 'notes']] = [label, ts, notes]
    else:
        df = pd.concat([
            df,
            pd.DataFrame({'index': [sample_index], 'label': [label], 'timestamp': [ts], 'notes': [notes]})
        ], ignore_index=True)
    df.sort_values('index', inplace=True)
    df.to_csv(path, index=False)


# --------- UI: Plotting ---------
def plot_sample(xy: np.ndarray, title: str, plot_type: str = "scatter") -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    
    if plot_type == "scatter":
        ax.scatter(xy[:, 0], xy[:, 1], color='tab:blue', s=10, alpha=0.6)
    elif plot_type == "histogram":
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(xy[:, 0], xy[:, 1], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(hist.T, extent=extent, origin='lower', cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Count')
    
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim(4.65, 4.85)
    ax.grid(True, linestyle='--', alpha=0.3)
    st.pyplot(fig, clear_figure=True)


# --------- App ---------
st.set_page_config(page_title="Streaking Tinder", page_icon="✨", layout="wide")
st.title("Streaking Tinder")
st.caption("Label each sample as 'streaked' (looks sinusoidal) or 'not streaked'.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload dataset (.npy/.npz/.jsonl/.json/.csv)", type=["npy", "npz", "jsonl", "json", "csv", "txt"]) 

    labels_dir_default = os.getcwd()
    st.divider()
    st.header("Labels Output")
    labels_directory = st.text_input("Labels directory", value=labels_dir_default, help="Folder to store labels CSV")

    if uploaded is not None:
        default_labels_path = os.path.join(labels_directory, derive_default_labels_path(uploaded.name))
    else:
        default_labels_path = os.path.join(labels_directory, "labels.csv")

    labels_path = st.text_input("Labels file path", value=default_labels_path)

    st.divider()
    st.header("Visualization")
    plot_type = st.selectbox(
        "Plot type",
        options=["scatter", "histogram"],
        index=0,
        help="Scatter: individual points, Histogram: 2D density heatmap"
    )
    
    st.divider()
    st.header("Controls")
    jump_to = st.number_input("Jump to index", min_value=0, step=1, value=0)
    apply_jump = st.button("Go")


# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = []
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = ''
if 'index' not in st.session_state:
    st.session_state.index = 0
if 'labels_path' not in st.session_state:
    st.session_state.labels_path = ''


# Load dataset
if uploaded is not None:
    try:
        data = load_dataset(uploaded.getvalue(), uploaded.name)
        if data:
            st.session_state.dataset = data
            st.session_state.dataset_name = uploaded.name
            # Initialize labels path on first load
            if not st.session_state.labels_path:
                st.session_state.labels_path = labels_path
        else:
            st.warning("No valid samples detected in the uploaded file.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")


# Respond to jump control
if apply_jump:
    st.session_state.index = int(jump_to)


dataset: List[np.ndarray] = st.session_state.dataset
num_samples = len(dataset)

if num_samples == 0:
    st.info("Upload a dataset to begin.")
    st.stop()



# Ensure labels path
if labels_path != st.session_state.labels_path and labels_path:
    st.session_state.labels_path = labels_path

existing_labels_df = load_existing_labels(st.session_state.labels_path)
done_indices = set(existing_labels_df['index'].tolist())


# Allow navigation to any sample, including already labeled ones


current_index = max(0, min(st.session_state.index, num_samples - 1))


xy = dataset[current_index]

# Create a 3x3 grid with centered navigation
col_left, col_center, col_right = st.columns([1, 2, 1])

# Left column - Skip button (centered vertically, right-aligned)
with col_left:
    if shortcut_button("Skip ⬇️", "arrowdown", key="skip_top", help="Skip this sample"):
        st.session_state.index = min(num_samples - 1, current_index + 1)
        st.rerun()
   
# Center column - Image
with col_center:
    plot_sample(xy, title=f"Sample {current_index + 1} / {num_samples}", plot_type=plot_type)

# Right column - Back button (centered vertically)
with col_right:
    if shortcut_button("Back ⬆️", "arrowup", key="back_bottom", help="Go to previous sample"):
        st.session_state.index = max(0, current_index - 1)
        st.rerun()

# Bottom row - Decision buttons
col_decision_left, col_decision_center, col_decision_right = st.columns([1, 2, 1])

with col_decision_left:
    if shortcut_button("Not Streaked ❌", "arrowleft", key="not_streaked_left"):
        upsert_label(st.session_state.labels_path, current_index, 0, "")
        st.session_state.index = min(num_samples - 1, current_index + 1)
        st.rerun()

with col_decision_center:
    st.markdown("")  # Empty space for alignment

with col_decision_right:
    if shortcut_button("Streaked ✅", "arrowright", key="streaked_right"):
        upsert_label(st.session_state.labels_path, current_index, 1, "")
        st.session_state.index = min(num_samples - 1, current_index + 1)
        st.rerun()

# Progress and info section
st.divider()
col_info1, col_info2 = st.columns([1, 1])

with col_info1:
    st.subheader("Progress")
    labeled_count = len(done_indices)
    st.progress(labeled_count / num_samples)
    st.write(f"Labeled: {labeled_count} / {num_samples}")
    if current_index in done_indices:
        existing_label = int(existing_labels_df.loc[existing_labels_df['index'] == current_index, 'label'].iloc[0])
        st.success(f"This sample is already labeled as: {'streaked' if existing_label == 1 else 'not streaked'}")

with col_info2:
    st.subheader("Info")
    st.caption(f"Labels file: {st.session_state.labels_path}")
    notes = st.text_input("Notes (optional)", value="", key="notes_input")

st.divider()
with st.expander("View/Edit Existing Labels"):
    df = load_existing_labels(st.session_state.labels_path)
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        st.download_button(
            label="Download labels CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=os.path.basename(st.session_state.labels_path) or 'labels.csv',
            mime='text/csv',
        )


