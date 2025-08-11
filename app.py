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

# Fixed password - no secrets needed
CORRECT_PASSWORD = "NeuroCrix2024!"

def check_password():
    """Simple password check with hardcoded password"""
    
    def password_entered():
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Enter Password", type="password", on_change=password_entered, key="password")
        st.write("*Password: NeuroCrix2024!*")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Enter Password", type="password", on_change=password_entered, key="password")
        st.error("😞 Password incorrect. Try: NeuroCrix2024!")
        return False
    else:
        return True

# Check for scipy
try:
    import scipy.signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    st.warning("scipy not installed - using basic FFT for frequency analysis")

class ZIPDataLoader:
    """Load EEG data from ZIP files in GitHub repo"""
    
    def __init__(self):
        # UPDATE THIS WITH YOUR ACTUAL GITHUB USERNAME
        self.github_user = "YOUR_GITHUB_USERNAME"  # <-- CHANGE THIS!
        self.repo = "neurocrix-eeg-platform"
        self.branch = "main"
        
        # Your ZIP files
        self.datasets = {
            "alcohol": {
                "name": "EEG Alcoholism Dataset",
                "filename": "EEG-Alcohol-Kag.zip",
                "description": "EEG recordings comparing alcoholic vs control subjects"
            },
            "psychiatric": {
                "name": "Psychiatric Disorders EEG",
                "filename": "Psidis-Kag.zip",
                "description": "EEG data for psychiatric disorder analysis"
            }
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_zip_from_github(_self, dataset_key: str) -> pd.DataFrame:
        """Download and extract ZIP file from GitHub"""
        
        if dataset_key not in _self.datasets:
            st.error(f"Unknown dataset: {dataset_key}")
            return None
        
        dataset = _self.datasets[dataset_key]
        filename = dataset["filename"]
        
        # GitHub raw URL
        url = f"https://github.com/{_self.github_user}/{_self.repo}/raw/{_self.branch}/{filename}"
        
        st.info(f"📥 Loading {dataset['name']}...")
        
        try:
            # Download the ZIP file
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                st.error(f"❌ Failed to download from GitHub (HTTP {response.status_code})")
                st.info(f"URL attempted: {url}")
                st.warning("Please check: 1) GitHub username is correct, 2) File exists in repo, 3) Repo is public")
                return None
            
            # Open ZIP file
            try:
                zip_buffer = io.BytesIO(response.content)
                with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                    # List all files
                    file_list = zip_ref.namelist()
                    st.success(f"✅ Found {len(file_list)} files in {filename}")
                    
                    # Find CSV files
                    csv_files = [f for f in file_list if f.lower().endswith('.csv')]
                    
                    if csv_files:
                        # Read the first CSV file
                        csv_file = csv_files[0]
                        st.info(f"📄 Reading: {csv_file}")
                        
                        with zip_ref.open(csv_file) as f:
                            df = pd.read_csv(f)
                            st.success(f"✅ Loaded {len(df):,} records × {len(df.columns)} columns")
                            return df
                    else:
                        st.error("❌ No CSV files found in ZIP")
                        st.info(f"Files found: {file_list[:5]}...")
                        return None
                        
            except zipfile.BadZipFile:
                st.error("❌ Invalid ZIP file - might be corrupted or not a ZIP")
                return None
                
        except requests.RequestException as e:
            st.error(f"❌ Network error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return None

class EEGProcessor:
    """Process EEG data for criticality analysis"""
    
    def __init__(self):
        self.fs = 256  # Default sampling rate
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare DataFrame for analysis"""
        
        info = {
            "total_records": len(df),
            "columns": len(df.columns),
            "has_labels": False
        }
        
        # Check for label columns
        label_cols = ['y', 'label', 'class', 'target', 'condition', 'diagnosis']
        label_col = None
        
        for col in label_cols:
            if col in df.columns:
                label_col = col
                info["has_labels"] = True
                info["label_column"] = col
                break
        
        if label_col:
            # Separate features and labels
            feature_cols = [c for c in df.columns if c != label_col]
            signals = df[feature_cols].values.T
            
            # Get label info
            labels = df[label_col].values
            unique_labels = np.unique(labels)
            info["labels"] = labels
            info["unique_labels"] = unique_labels.tolist()
            info["n_classes"] = len(unique_labels)
            
            # Check for seizures (usually label=1)
            if 1 in unique_labels:
                seizure_count = np.sum(labels == 1)
                info["has_seizures"] = True
                info["seizure_count"] = int(seizure_count)
                info["seizure_percentage"] = seizure_count / len(labels) * 100
        else:
            # No labels - all columns are features
            signals = df.values.T
        
        # Ensure correct shape (channels × samples)
        if signals.shape[0] > signals.shape[1]:
            signals = signals.T
        
        info["n_channels"] = signals.shape[0]
        info["n_samples"] = signals.shape[1]
        
        return signals, info
    
    def extract_features(self, signals: np.ndarray, window_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frequency band features"""
        
        n_channels, n_samples = signals.shape
        fs = self.fs
        
        # Adjust window for short signals
        max_window = n_samples // 2
        window = min(int(window_s * fs), max_window)
        step = window // 2
        
        # Number of windows
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
                signal_segment = signals[ch, start:end]
                
                # FFT for frequency analysis
                fft = np.fft.fft(signal_segment)
                freqs = np.fft.fftfreq(len(signal_segment), 1/fs)
                power = np.abs(fft) ** 2
                
                # Positive frequencies only
                pos_mask = freqs > 0
                freqs_pos = freqs[pos_mask]
                power_pos = power[pos_mask]
                
                # Extract band powers
                for b, (low, high) in enumerate(bands):
                    band_mask = (freqs_pos >= low) & (freqs_pos <= high)
                    if np.any(band_mask):
                        features[w, ch, b] = np.sqrt(np.mean(power_pos[band_mask]))
        
        return times, features
    
    def compute_criticality(self, features: np.ndarray, times: np.ndarray, 
                           threshold: float = 0.3, has_seizures: bool = False) -> Dict:
        """Compute brain criticality using logistic map"""
        
        n_windows, n_channels, n_bands = features.shape
        
        # Normalize features
        features_norm = np.zeros_like(features)
        for b in range(n_bands):
            band_data = features[:, :, b]
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            if std_val > 0:
                features_norm[:, :, b] = (band_data - mean_val) / std_val
        
        # Initialize logistic map
        r_params = np.full(n_bands, 3.0)
        r_evolution = []
        state_evolution = []
        critical_windows = []
        
        # Sensitivity based on data type
        sensitivity = 0.15 if has_seizures else 0.1
        chaos_threshold = 3.5 if has_seizures else 3.57
        
        # Process each window
        for w in range(n_windows):
            # Calculate band activities
            for b in range(n_bands):
                activity = np.mean(np.abs(features_norm[w, :, b]))
                
                if activity > threshold:
                    r_params[b] = min(3.99, r_params[b] + sensitivity)
                else:
                    r_params[b] = max(2.5, r_params[b] - 0.02)
            
            # Mean R parameter
            r_mean = np.mean(r_params)
            r_evolution.append(r_mean)
            
            # Determine state
            if r_mean > chaos_threshold:
                state = "critical"
                critical_windows.append(w)
            elif r_mean > 3.2:
                state = "transitional"
            else:
                state = "stable"
            
            state_evolution.append(state)
        
        # Calculate metrics
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
        page_title="🧠 EEG Criticality Analysis",
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
    loader = ZIPDataLoader()
    processor = EEGProcessor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        dataset_key = st.selectbox(
            "📊 Select Dataset",
            options=list(loader.datasets.keys()),
            format_func=lambda x: loader.datasets[x]["name"]
        )
        
        if dataset_key:
            st.info(loader.datasets[dataset_key]["description"])
        
        # Parameters
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (sec)", 0.5, 5.0, 2.0)
        threshold = st.slider("Threshold", 0.1, 0.5, 0.3)
        
        # Display options
        st.subheader("📈 Display Options")
        show_raw = st.checkbox("Show Raw Data", False)
        show_evolution = st.checkbox("Show R Evolution", True)
        show_bands = st.checkbox("Show Frequency Bands", True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Analysis")
        
        if st.button("🚀 Load & Analyze Dataset", type="primary", use_container_width=True):
            
            # Load data
            df = loader.load_zip_from_github(dataset_key)
            
            if df is not None:
                # Data overview
                st.subheader("📋 Dataset Overview")
                col1a, col1b, col1c = st.columns(3)
                
                with col1a:
                    st.metric("Records", f"{len(df):,}")
                with col1b:
                    st.metric("Features", len(df.columns))
                with col1c:
                    st.metric("Size (MB)", f"{df.memory_usage().sum() / 1024**2:.1f}")
                
                # Show raw data
                if show_raw:
                    with st.expander("View Raw Data"):
                        st.dataframe(df.head(100))
                
                # Prepare data
                with st.spinner("🔬 Processing EEG data..."):
                    signals, data_info = processor.prepare_data(df)
                
                # Display data info
                if data_info.get("has_labels"):
                    st.info(f"""
                    **Labeled Data Detected:**
                    - Label Column: `{data_info.get('label_column')}`
                    - Classes: {data_info.get('n_classes')}
                    - Labels: {data_info.get('unique_labels')}
                    """)
                    
                    if data_info.get("has_seizures"):
                        st.warning(f"""
                        ⚡ **Seizure Data Detected!**
                        - Seizure Records: {data_info.get('seizure_count'):,}
                        - Percentage: {data_info.get('seizure_percentage'):.1f}%
                        """)
                
                # Analyze
                with st.spinner("🧠 Computing brain criticality..."):
                    times, features = processor.extract_features(signals, window_size)
                    results = processor.compute_criticality(
                        features, times, threshold, 
                        data_info.get("has_seizures", False)
                    )
                
                # Results
                st.success("✅ Analysis Complete!")
                
                # Metrics
                st.subheader("📊 Results")
                col2a, col2b, col2c, col2d = st.columns(4)
                
                with col2a:
                    crit = results['criticality_ratio']
                    color = "🔴" if crit > 0.4 else "🟡" if crit > 0.2 else "🟢"
                    st.metric(f"{color} Criticality", f"{crit:.1%}")
                
                with col2b:
                    st.metric("State", results['final_state'])
                
                with col2c:
                    st.metric("Critical", f"{results['critical_windows']}/{results['total_windows']}")
                
                with col2d:
                    st.metric("Mean R", f"{results['mean_r']:.3f}")
                
                # Visualizations
                if show_evolution or show_bands:
                    fig, axes = plt.subplots(
                        1 + show_bands, 1,
                        figsize=(12, 4 * (1 + show_bands))
                    )
                    
                    if not show_bands:
                        axes = [axes]
                    
                    # R parameter evolution
                    if show_evolution:
                        ax = axes[0]
                        colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red'
                                 for s in results['state_evolution']]
                        ax.scatter(results['times'], results['r_evolution'], c=colors, s=50)
                        ax.axhline(y=3.57, color='red', linestyle='--', alpha=0.5, label='Chaos')
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('R Parameter')
                        ax.set_title('Brain State Evolution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Frequency bands
                    if show_bands:
                        ax = axes[-1]
                        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                        values = [results['band_statistics'][b.lower()]['mean'] for b in bands]
                        colors = ['purple', 'blue', 'green', 'orange', 'red']
                        bars = ax.bar(bands, values, color=colors, alpha=0.7)
                        ax.set_ylabel('Power (μV)')
                        ax.set_title('Frequency Band Analysis')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        for bar, val in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'{val:.1f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Generate report
                report = f"""
## Analysis Report
**Dataset:** {loader.datasets[dataset_key]['name']}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Results
- **Criticality:** {results['criticality_ratio']:.1%}
- **State:** {results['final_state']}
- **Critical Episodes:** {results['critical_windows']}/{results['total_windows']}
- **Mean R:** {results['mean_r']:.3f}
- **Chaos:** {results['chaos_percentage']:.1f}%

### Frequency Bands (μV)
- Delta: {results['band_statistics']['delta']['mean']:.1f}
- Theta: {results['band_statistics']['theta']['mean']:.1f}
- Alpha: {results['band_statistics']['alpha']['mean']:.1f}
- Beta: {results['band_statistics']['beta']['mean']:.1f}
- Gamma: {results['band_statistics']['gamma']['mean']:.1f}
"""
                
                if data_info.get("has_seizures"):
                    report += f"""
### Seizure Analysis
- **Seizure Records:** {data_info.get('seizure_count'):,}
- **Percentage:** {data_info.get('seizure_percentage'):.1f}%
- **Correlation:** High criticality {'detected' if results['criticality_ratio'] > 0.3 else 'not detected'}
"""
                
                # Download button
                st.download_button(
                    "📄 Download Report",
                    data=report,
                    file_name=f"eeg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with col2:
        st.header("ℹ️ Information")
        
        with st.expander("Setup Instructions", expanded=True):
            st.markdown("""
            ### ⚠️ IMPORTANT: Update GitHub Username
            
            1. **Edit line 46 in the code:**
               ```python
               self.github_user = "YOUR_GITHUB_USERNAME"
               ```
               Replace with your actual GitHub username!
            
            2. **Ensure ZIP files are in your repo:**
               - `EEG-Alcohol-Kag.zip`
               - `Psidis-Kag.zip`
            
            3. **Repository must be public**
            
            ### Password
            - Password is: **NeuroCrix2024!**
            """)
        
        with st.expander("About the Datasets"):
            st.markdown("""
            ### EEG Alcoholism Dataset
            - Compares alcoholic vs control subjects
            - Multiple EEG recordings per subject
            - Useful for addiction studies
            
            ### Psychiatric Disorders EEG
            - Various psychiatric conditions
            - Labeled data for classification
            - Useful for mental health research
            """)
        
        with st.expander("Understanding Results"):
            st.markdown("""
            ### Criticality Levels
            - 🟢 **<20%**: Stable brain state
            - 🟡 **20-40%**: Transitional state
            - 🔴 **>40%**: Critical/unstable state
            
            ### R Parameter
            - **<3.0**: Very stable
            - **3.0-3.57**: Edge of chaos
            - **>3.57**: Chaotic dynamics
            
            ### Frequency Bands
            - **Delta (0.5-4 Hz)**: Deep sleep, pathology
            - **Theta (4-8 Hz)**: Drowsiness
            - **Alpha (8-13 Hz)**: Relaxed, eyes closed
            - **Beta (13-30 Hz)**: Active thinking
            - **Gamma (30-50 Hz)**: Cognitive processing
            """)

if __name__ == "__main__":
    main()
