# Streaking Tinder - Multi-User Data Labeling App

A Streamlit app for labeling 2D coordinate data as "streaked" (sinusoidal) or "not streaked" with Tinder-style interface.

## Features
- **Multi-user support** - Each user gets a unique ID and separate labels
- **Preloaded datasets** - Choose from sample datasets or upload your own
- **Keyboard shortcuts** - Arrow keys for quick labeling
- **User tracking** - All labels include user ID and timestamp
- **Progress tracking** - See how many samples you've labeled
- **Data visualization** - Switch between scatter plot and histogram views

## Deployment

### Option 1: Streamlit Cloud (Recommended)
1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Option 2: Heroku
1. Create a `Procfile` with: `web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
2. Deploy to Heroku

### Option 3: Railway/Render
1. Connect your GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run streamlit_app.py`

## Adding Preloaded Datasets
1. Create a `datasets/` folder
2. Add your `.npy` files (e.g., `sample1.npy`, `sample2.npy`)
3. Update the `preloaded_options` dictionary in `streamlit_app.py`

## Viewing User Labels
All labels are saved in the `labels/` folder with format:
- `{dataset_name}_{user_id}_labels.csv`
- Columns: `index`, `label`, `timestamp`, `notes`, `user_id`

## Keyboard Shortcuts
- **← (Left Arrow)**: Not Streaked
- **→ (Right Arrow)**: Streaked
- **↓ (Down Arrow)**: Skip
- **↑ (Up Arrow)**: Back

## Requirements
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- streamlit-shortcuts