import streamlit as st
import numpy as np
import pandas as pd
import io
import json
import time
import zipfile
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Fixed password
CORRECT_PASSWORD = "NeuroCrix2024!"

def check_password():
    """Simple password check"""
    
    def password_entered():
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Enter Password", type="password", on_change=password_entered, key="password")
        st.info("Password: NeuroCrix2024!")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Enter Password", type="password", on_change=password_entered, key="password")
        st.error("❌ Incorrect. Password is: NeuroCrix2024!")
        return False
    else:
        return True

# Check for scipy
try:
    import scipy.signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class DataLoader:
    """Load EEG data from Google Drive and GitHub"""
    
    def __init__(self):
        # YOUR GITHUB USERNAME HERE
        self.github_user = "YOUR_GITHUB_USERNAME"  # <-- CHANGE THIS!
        self.repo = "neurocrix-eeg-platform"
        
        # YOUR GOOGLE DRIVE FILE ID HERE
        self.google_drive_file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # <-- CHANGE THIS!
        
        # Dataset configuration
        self.datasets = {
            "alcohol": {
                "name": "EEG Alcoholism Dataset (300MB)",
                "filename": "EEG-Alcohol-Kag.zip",
                "description": "Large dataset comparing alcoholic vs control subjects",
                "source": "google_drive"  # Uses Google Drive
            },
            "psychiatric": {
                "name": "Psychiatric Disorders EEG",
                "filename": "Psidis-Kag.zip",
                "description": "EEG data for psychiatric disorder analysis",
                "source": "github"  # Uses GitHub
            }
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_dataset(_self, dataset_key: str) -> pd.DataFrame:
        """Load dataset from appropriate source"""
        
        if dataset_key not in _self.datasets:
            st.error(f"Unknown dataset: {dataset_key}")
            return None
        
        dataset = _self.datasets[dataset_key]
        
        # Choose source based on dataset
        if dataset["source"] == "google_drive":
            return _self.load_from_google_drive(dataset)
        else:
            return _self.load_from_github(dataset)
    
    def load_from_google_drive(self, dataset: Dict) -> pd.DataFrame:
        """Load large file from Google Drive"""
        
        st.info(f"📥 Downloading {dataset['name']} from Google Drive...")
        
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?export=download&id={self.google_drive_file_id}"
        
        try:
            # Download with confirmation handling for large files
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Check if we got a virus scan warning
            if b"virus scan warning" in response.content[:1000]:
                st.warning("Large file detected, bypassing Google Drive warning...")
                
                # Extract confirmation token
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        url = f"https://drive.google.com/uc?export=download&confirm={value}&id={self.google_drive_file_id}"
                        response = session.get(url, stream=True)
                        break
            
            # Download the file
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                downloaded = 0
                content = b""
                
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    downloaded += len(chunk)
                    progress = downloaded / total_size
                    progress_bar.progress(progress)
                    progress_text.text(f"Downloaded: {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB")
                
                progress_bar.empty()
                progress_text.empty()
            else:
                # No content length, download all at once
                content = response.content
            
            # Extract ZIP
            return self.extract_zip_data(content, dataset['name'])
            
        except Exception as e:
            st.error(f"❌ Error downloading from Google Drive: {str(e)}")
            st.info("Make sure: 1) File ID is correct, 2) File is shared with 'Anyone with link'")
            return None
    
    def load_from_github(self, dataset: Dict) -> pd.DataFrame:
        """Load smaller file from GitHub"""
        
        st.info(f"📥 Loading {dataset['name']} from GitHub...")
        
        # GitHub raw URL
        url = f"https://github.com/{self.github_user}/{self.repo}/raw/main/{dataset['filename']}"
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                st.error(f"❌ Failed to download from GitHub (HTTP {response.status_code})")
                st.info("Make sure: 1) GitHub username is correct, 2) File exists in repo")
                return None
            
            # Extract ZIP
            return self.extract_zip_data(response.content, dataset['name'])
            
        except Exception as e:
            st.error(f"❌ Error downloading from GitHub: {str(e)}")
            return None
    
    def extract_zip_data(self, content: bytes, dataset_name: str) -> pd.DataFrame:
        """Extract data from ZIP content"""
        
        try:
            zip_buffer = io.BytesIO(content)
            
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.success(f"✅ Extracted {len(file_list)} files from {dataset_name}")
                
                # Find CSV files
                csv_files = [f for f in file_list if f.lower().endswith('.csv')]
                
                if csv_files:
                    # Read first CSV
                    with zip_ref.open(csv_files[0]) as f:
                        df = pd.read_csv(f)
                        st.success(f"✅ Loaded {len(df):,} records × {len(df.columns)} columns")
                        return df
                else:
                    st.error("❌ No CSV files found in ZIP")
                    return None
                    
        except zipfile.BadZipFile:
            st.error("❌ Invalid ZIP file")
            return None
        except Exception as e:
            st.error(f"❌ Error extracting ZIP: {str(e)}")
            return None

class EEGAnalyzer:
    """Analyze EEG data for criticality"""
    
    def __init__(self):
        self.fs = 256  # Default sampling rate
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare data for analysis"""
        
        info = {
            "total_records": len(df),
            "columns": len(df.columns),
            "has_labels": False
        }
        
        # Find label column
        label_cols = ['y', 'label', 'class', 'target', 'condition']
        label_col = None
        
        for col in label_cols:
            if col in df.columns:
                label_col = col
                info["has_labels"] = True
                info["label_column"] = col
                break
        
        if label_col:
            # Extract features and labels
            feature_cols = [c for c in df.columns if c != label_col]
            signals = df[feature_cols].values.T
            
            labels = df[label_col].values
            unique_labels = np.unique(labels)
            info["labels"] = labels
            info["unique_labels"] = unique_labels.tolist()
            info["n_classes"] = len(unique_labels)
            
            # Check for seizures
            if 1 in unique_labels:
                seizure_count = np.sum(labels == 1)
                info["has_seizures"] = True
                info["seizure_count"] = int(seizure_count)
                info["seizure_percentage"] = seizure_count / len(labels) * 100
        else:
            # All columns are features
            signals = df.values.T
        
        # Ensure correct shape
        if signals.shape[0] > signals.shape[1]:
            signals = signals.T
        
        info["n_channels"] = signals.shape[0]
        info["n_samples"] = signals.shape[1]
        
        return signals, info
    
    def extract_features(self, signals: np.ndarray, window_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frequency features"""
        
        n_channels, n_samples = signals.shape
        fs = self.fs
        
        # Window parameters
        window = min(int(window_s * fs), n_samples // 2)
        step = window // 2
        
        if n_samples < window:
            n_windows = 1
            window = n_samples
        else:
            n_windows = (n_samples - window) // step + 1
        
        times = np.arange(n_windows) * (step / fs)
        
        # 5 frequency bands
        features = np.zeros((n_windows, n_channels, 5))
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        
        for w in range(n_windows):
            start = w * step if n_windows > 1 else 0
            end = min(start + window, n_samples)
            
            for ch in range(n_channels):
                segment = signals[ch, start:end]
                
                # FFT
                fft = np.fft.fft(segment)
                freqs = np.fft.fftfreq(len(segment), 1/fs)
                power = np.abs(fft) ** 2
                
                # Positive frequencies
                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                power_pos = power[pos_mask]
                
                # Band powers
                for b, (low, high) in enumerate(bands):
                    band_mask = (freqs_pos >= low) & (freqs_pos <= high)
                    if np.any(band_mask):
                        features[w, ch, b] = np.sqrt(np.mean(power_pos[band_mask]))
        
        return times, features
    
    def compute_criticality(self, features: np.ndarray, times: np.ndarray, 
                           threshold: float = 0.3, has_seizures: bool = False) -> Dict:
        """Compute criticality using logistic map"""
        
        n_windows, n_channels, n_bands = features.shape
        
        # Normalize
        features_norm = np.zeros_like(features)
        for b in range(n_bands):
            band_data = features[:, :, b]
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            if std_val > 0:
                features_norm[:, :, b] = (band_data - mean_val) / std_val
        
        # Logistic map
        r_params = np.full(n_bands, 3.0)
        r_evolution = []
        state_evolution = []
        critical_windows = []
        
        sensitivity = 0.15 if has_seizures else 0.1
        chaos_threshold = 3.5 if has_seizures else 3.57
        
        for w in range(n_windows):
            for b in range(n_bands):
                activity = np.mean(np.abs(features_norm[w, :, b]))
                
                if activity > threshold:
                    r_params[b] = min(3.99, r_params[b] + sensitivity)
                else:
                    r_params[b] = max(2.5, r_params[b] - 0.02)
            
            r_mean = np.mean(r_params)
            r_evolution.append(r_mean)
            
            if r_mean > chaos_threshold:
                state = "critical"
                critical_windows.append(w)
            elif r_mean > 3.2:
                state = "transitional"
            else:
                state = "stable"
            
            state_evolution.append(state)
        
        criticality_ratio = len(critical_windows) / max(1, n_windows)
        
        if criticality_ratio > 0.4:
            final_state = "HIGHLY CRITICAL"
        elif criticality_ratio > 0.2:
            final_state = "MODERATELY CRITICAL"
        elif criticality_ratio > 0.1:
            final_state = "TRANSITIONAL"
        else:
            final_state = "STABLE"
        
        # Band statistics
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_stats = {}
        for i, name in enumerate(band_names):
            band_stats[name] = {
                'mean': float(np.mean(features[:, :, i])),
                'std': float(np.std(features[:, :, i]))
            }
        
        return {
            'criticality_ratio': criticality_ratio,
            'final_state': final_state,
            'critical_windows': len(critical_windows),
            'total_windows': n_windows,
            'r_evolution': r_evolution,
            'state_evolution': state_evolution,
            'times': times.tolist(),
            'band_statistics': band_stats,
            'mean_r': float(np.mean(r_evolution)),
            'chaos_percentage': sum(1 for r in r_evolution if r > chaos_threshold) / len(r_evolution) * 100
        }

def main():
    st.set_page_config(
        page_title="🧠 EEG Analysis Platform",
        page_icon="🧠",
        layout="wide"
    )
    
    # Password check
    if not check_password():
        st.stop()
    
    # Header
    st.title("🧠 EEG Criticality Analysis Platform")
    st.markdown("**Analyze Real EEG Data from Kaggle Datasets**")
    
    # Initialize
    loader = DataLoader()
    analyzer = EEGAnalyzer()
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Dataset selection
        dataset_key = st.selectbox(
            "📊 Select Dataset",
            options=list(loader.datasets.keys()),
            format_func=lambda x: loader.datasets[x]["name"]
        )
        
        if dataset_key:
            dataset = loader.datasets[dataset_key]
            
            # Show source
            if dataset["source"] == "google_drive":
                st.info(f"📁 **{dataset['name']}** - Stored on Google Drive (Large file)")
            else:
                st.info(f"📁 **{dataset['name']}** - Stored on GitHub")
            
            st.caption(dataset["description"])
        
        # Analysis parameters
        col1a, col1b = st.columns(2)
        with col1a:
            window_size = st.slider("Window Size (seconds)", 0.5, 5.0, 2.0)
        with col1b:
            threshold = st.slider("Threshold", 0.1, 0.5, 0.3)
        
        # Analyze button
        if st.button("🚀 Load & Analyze Dataset", type="primary", use_container_width=True):
            
            # Load data
            df = loader.load_dataset(dataset_key)
            
            if df is not None:
                # Overview
                st.subheader("📊 Dataset Overview")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Records", f"{len(df):,}")
                with c2:
                    st.metric("Features", len(df.columns))
                with c3:
                    st.metric("Size (MB)", f"{df.memory_usage().sum() / 1024**2:.1f}")
                
                # Prepare data
                with st.spinner("🔬 Processing EEG data..."):
                    signals, data_info = analyzer.prepare_data(df)
                
                # Show data info
                if data_info.get("has_labels"):
                    st.success(f"""
                    ✅ **Labeled Data Detected**
                    - Classes: {data_info.get('n_classes')}
                    - Labels: {data_info.get('unique_labels')}
                    """)
                    
                    if data_info.get("has_seizures"):
                        st.warning(f"""
                        ⚡ **Seizure Data Found!**
                        - Seizure Records: {data_info.get('seizure_count'):,}
                        - Percentage: {data_info.get('seizure_percentage'):.1f}%
                        """)
                
                # Analyze
                with st.spinner("🧠 Computing brain criticality..."):
                    times, features = analyzer.extract_features(signals, window_size)
                    results = analyzer.compute_criticality(
                        features, times, threshold, 
                        data_info.get("has_seizures", False)
                    )
                
                # Results
                st.success("✅ Analysis Complete!")
                
                st.subheader("📈 Results")
                r1, r2, r3, r4 = st.columns(4)
                
                with r1:
                    crit = results['criticality_ratio']
                    if crit > 0.4:
                        st.metric("🔴 Criticality", f"{crit:.1%}")
                    elif crit > 0.2:
                        st.metric("🟡 Criticality", f"{crit:.1%}")
                    else:
                        st.metric("🟢 Criticality", f"{crit:.1%}")
                
                with r2:
                    st.metric("State", results['final_state'])
                
                with r3:
                    st.metric("Critical", f"{results['critical_windows']}/{results['total_windows']}")
                
                with r4:
                    st.metric("Mean R", f"{results['mean_r']:.3f}")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # R evolution
                colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red'
                         for s in results['state_evolution']]
                ax1.scatter(results['times'], results['r_evolution'], c=colors, s=50)
                ax1.axhline(y=3.57, color='red', linestyle='--', alpha=0.5)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('R Parameter')
                ax1.set_title('Brain State Evolution')
                ax1.grid(True, alpha=0.3)
                
                # Frequency bands
                bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                values = [results['band_statistics'][b.lower()]['mean'] for b in bands]
                bars = ax2.bar(bands, values, color=['purple', 'blue', 'green', 'orange', 'red'], alpha=0.7)
                ax2.set_ylabel('Power (μV)')
                ax2.set_title('Frequency Bands')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for bar, val in zip(bars, values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Report
                report = f"""
## EEG Analysis Report
**Dataset:** {dataset['name']}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Results
- Criticality: {results['criticality_ratio']:.1%}
- State: {results['final_state']}
- Critical Episodes: {results['critical_windows']}/{results['total_windows']}
- Mean R: {results['mean_r']:.3f}
- Chaos: {results['chaos_percentage']:.1f}%

### Frequency Bands
- Delta: {results['band_statistics']['delta']['mean']:.1f} μV
- Theta: {results['band_statistics']['theta']['mean']:.1f} μV
- Alpha: {results['band_statistics']['alpha']['mean']:.1f} μV
- Beta: {results['band_statistics']['beta']['mean']:.1f} μV
- Gamma: {results['band_statistics']['gamma']['mean']:.1f} μV
"""
                
                if data_info.get("has_seizures"):
                    report += f"""
### Seizure Analysis
- Seizure Records: {data_info.get('seizure_count'):,}
- Percentage: {data_info.get('seizure_percentage'):.1f}%
"""
                
                st.download_button(
                    "📄 Download Report",
                    data=report,
                    file_name=f"eeg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with col2:
        st.header("⚙️ Setup")
        
        with st.expander("🔧 Configuration Required", expanded=True):
            st.error("""
            **MUST UPDATE:**
            
            1. **Line 46 - GitHub Username:**
            ```python
            self.github_user = "YOUR_GITHUB_USERNAME"
            ```
            
            2. **Line 50 - Google Drive File ID:**
            ```python
            self.google_drive_file_id = "YOUR_FILE_ID"
            ```
            
            **How to get File ID:**
            - Google Drive link: 
            `drive.google.com/file/d/ABC123/view`
            - File ID is: `ABC123`
            """)
        
        with st.expander("📊 Dataset Info"):
            st.markdown("""
            ### Files Required:
            
            **Google Drive:**
            - `EEG-Alcohol-Kag.zip` (300MB)
            - Too large for GitHub
            
            **GitHub Repo:**
            - `Psidis-Kag.zip` (<25MB)
            - Small enough for GitHub
            """)
        
        with st.expander("❓ Troubleshooting"):
            st.markdown("""
            ### Common Issues:
            
            **Google Drive Error:**
            - Make sure file is shared
            - Set to "Anyone with link"
            - Check File ID is correct
            
            **GitHub Error:**
            - Check username is correct
            - Ensure repo is public
            - File must be in main branch
            
            **Password:**
            - `NeuroCrix2024!`
            - Case sensitive!
            """)

if __name__ == "__main__":
    main()
