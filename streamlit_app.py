import io
import os
from datetime import datetime
from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_shortcuts import add_shortcuts, shortcut_button

import gspread
from google.oauth2.service_account import Credentials

from streamlit.logger import get_logger

logger = get_logger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
gc = gspread.authorize(creds)
sh = gc.open_by_url(st.secrets["sheet_url"])

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
def get_user_id() -> str:
    """Get or create a unique user ID for this session"""
    if 'user_id' not in st.session_state:
        import uuid
        st.session_state.user_id = str(uuid.uuid4())[:8]  # Short unique ID
    return st.session_state.user_id

def get_username() -> str:
    """Get or create a username for this session"""
    if 'username' not in st.session_state:
        st.session_state.username = f"User_{get_user_id()}"
    return st.session_state.username


def get_labels_path(dataset_name: str, user_id: str, username: str = "") -> str:
    """Generate labels path for this user and dataset"""
    base = os.path.splitext(os.path.basename(dataset_name))[0]

    return f"labels/{base}_{username}_labels.csv"

def load_existing_labels_from_drive(dataset_name: str, username: str) -> pd.DataFrame:
    path = get_labels_path(dataset_name, '', username)
    base = os.path.splitext(os.path.basename(dataset_name))[0]

    try:
        ws = sh.worksheet(base)
    except gspread.WorksheetNotFound:
        df = pd.DataFrame(columns=['index', 'label', 'username'])
        df.to_csv(path, index=False)
        return df
    existing_users = ws.row_values(1)

    if username not in existing_users:
        df = pd.DataFrame(columns=['index', 'label', 'username'])
    else:
        ind = existing_users.index(username)
        ws_labels = ws.col_values(ind+1)
        logger.info(ws_labels)
        inds, labels = [], []
        for i, label in enumerate(ws_labels[1:]):
            logger.info(label)
            if label == '':
                pass
            elif label == '0':
                inds.append(i)
                labels.append(0)
            elif label == '1':
                inds.append(i)
                labels.append(1)
        df = pd.DataFrame({'index': inds, 'label': labels, 'username': [username for i in range(len(inds))]})

    df.to_csv(path, index=False)
    return df

def load_existing_labels(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Normalize expected columns
            expected = {'index', 'label'}
            if not expected.issubset(set(df.columns)):
                return pd.DataFrame(columns=['index', 'label', 'username'])
            df = df[['index', 'label'] + [c for c in ['username'] if c in df.columns]]
            df['index'] = df['index'].astype(int)
            df['label'] = df['label'].astype(int)
            return df
        except Exception:
            return pd.DataFrame(columns=['index', 'label', 'username'])
    return pd.DataFrame(columns=['index', 'label', 'username'])


def upsert_label(path: str, sample_index: int, label: int, notes: str = "", user_id: str = "", username: str = "") -> None:
    logger.info('Upserting label')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df = load_existing_labels(path)
    logger.info('Current df is', df)
    ts = datetime.now().isoformat(timespec='seconds')
    if (df['index'] == sample_index).any():
        df.loc[df['index'] == sample_index, ['label', 'username']] = [label, username]
    else:
        df = pd.concat([
            df,
            pd.DataFrame({'index': [sample_index], 'label': [label], 'username': [username]})
        ], ignore_index=True)
    df.sort_values('index', inplace=True)
    df.to_csv(path, index=False)

def write_labels_to_drive(dataset_name: str, username: str, num_samples: int):
    path = get_labels_path(dataset_name, user_id, username)
    base = os.path.splitext(os.path.basename(dataset_name))[0]
    
    try:
        ws = sh.worksheet(base)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=base, rows=num_samples, cols=100)
        ws.update_cell(1, 1, 'Index')

    existing_users = ws.row_values(1)

    if username not in existing_users:
        ws.update_cell(1, len(existing_users) + 1, username)
        existing_users = ws.row_values(1)

    ind = existing_users.index(username)
    labels_df = load_existing_labels(path)

    cells_to_update = ws.range(2, ind+1, num_samples+1, ind+1)
    for i, index in enumerate(labels_df['index'].values):
        cells_to_update[index].value = str(labels_df['label'][i])
    ws.update_cells(cells_to_update)


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
st.set_page_config(page_title="Streak Finder", page_icon="✨", layout="wide")
st.image('StreakFinder_logo.png', width=600)

# User identification
user_id = get_user_id()
username = get_username()

# Username input
st.sidebar.header("User")
new_username = st.sidebar.text_input(
    "Choose your username:", 
    value=username,
    help="This will be used to identify your labels"
)

# Update username if changed
if new_username != username and new_username.strip():
    st.session_state.username = new_username.strip()
    username = st.session_state.username
    # Regenerate labels path with new username if dataset is loaded
    if st.session_state.dataset_name:
        st.session_state.labels_path = get_labels_path(st.session_state.dataset_name, user_id, username)

st.sidebar.info(f"👤 Username: `{username}`")
st.sidebar.caption(f"ID: `{user_id}`")

with st.sidebar:
    st.header("Dataset")
    
    # Preloaded datasets
    st.subheader("Preloaded Datasets")
    preloaded_options = {
        "Run 96": "datasets/shots.npy"
    }
    
    selected_preloaded = st.selectbox("Choose a preloaded dataset:", ["None"] + list(preloaded_options.keys()))
    
    st.divider()
    st.subheader("Or Upload Your Own")
    uploaded = st.file_uploader("Upload dataset (.npy/.npz/.jsonl/.json/.csv)", type=["npy", "npz", "jsonl", "json", "csv", "txt"])
    
    # Determine which dataset to use
    if selected_preloaded != "None" and selected_preloaded in preloaded_options:
        dataset_path = preloaded_options[selected_preloaded]
        dataset_name = selected_preloaded
        st.success(f"Selected: {selected_preloaded}")
    elif uploaded is not None:
        dataset_path = uploaded.name
        dataset_name = uploaded.name
        st.success(f"Uploaded: {uploaded.name}")
    else:
        dataset_path = None
        dataset_name = None

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
if dataset_path and dataset_name:
    try:
        if uploaded is not None:
            # Load from uploaded file
            data = load_dataset(uploaded.getvalue(), uploaded.name)
        else:
            # Load from preloaded dataset
            if os.path.exists(dataset_path):
                with open(dataset_path, 'rb') as f:
                    data = load_dataset(f.read(), dataset_path)
            else:
                st.error(f"Preloaded dataset not found: {dataset_path}")
                data = []
        
        if data:
            st.session_state.dataset = data
            st.session_state.dataset_name = dataset_name
            # Generate labels path for this user and dataset
            st.session_state.labels_path = get_labels_path(dataset_name, user_id, username)
        else:
            st.warning("No valid samples detected in the dataset.")
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

# Load existing labels if we have a dataset
if st.session_state.labels_path:
    logger.info('Loading from local file')
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
        upsert_label(st.session_state.labels_path, current_index, 0, "", user_id, username)
        st.session_state.index = min(num_samples - 1, current_index + 1)
        st.rerun()

with col_decision_center:
    st.markdown("")  # Empty space for alignment

with col_decision_right:
    if shortcut_button("Streaked ✅", "arrowright", key="streaked_right"):
        upsert_label(st.session_state.labels_path, current_index, 1, "", user_id, username)
        st.session_state.index = min(num_samples - 1, current_index + 1)
        st.rerun()

st.divider()

if st.button("Load labels from drive"):
    existing_labels_df = load_existing_labels_from_drive(st.session_state.dataset_name, st.session_state.username)

if st.button("Upload labels to drive"):
    write_labels_to_drive(st.session_state.dataset_name, username, num_samples)

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
    st.dataframe(df, width='stretch')
    if not df.empty:
        st.download_button(
            label="Download labels CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=os.path.basename(st.session_state.labels_path) or 'labels.csv',
            mime='text/csv',
        )


