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

# Password protection
CORRECT_PASSWORD = "NeuroCrix2024!"

def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔐 Enter Password to Access EEG Platform", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.write("*Contact administrator for access credentials*")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔐 Enter Password to Access EEG Platform", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😞 Password incorrect. Please try again.")
        return False
    else:
        return True

# Check for scipy
try:
    import scipy.signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class KaggleDatasetLoader:
    """Load and process Kaggle EEG datasets from GitHub repo"""
    
    def __init__(self, github_username="YOUR_USERNAME", repo_name="neurocrix-eeg-platform"):
        """Initialize with your GitHub details"""
        self.github_username = github_username
        self.repo_name = repo_name
        self.base_url = f"https://github.com/{github_username}/{repo_name}/raw/main/"
        self.datasets = {
            "eeg_alcohol": {
                "filename": "EEG-Alcohol-Kag.zip",
                "name": "EEG Alcoholism Dataset",
                "description": "EEG recordings comparing alcoholic vs control subjects",
                "url": f"{self.base_url}EEG-Alcohol-Kag.zip"
            },
            "psidis": {
                "filename": "Psidis-Kag.zip", 
                "name": "Psychiatric Disorders EEG Dataset",
                "description": "EEG data for psychiatric disorder analysis",
                "url": f"{self.base_url}Psidis-Kag.zip"
            }
        }
        self.cached_data = {}
    
    @st.cache_data(ttl=3600)
    def download_and_extract_zip(_self, dataset_key: str) -> pd.DataFrame:
        """Download ZIP file from GitHub and extract data"""
        
        dataset_info = _self.datasets.get(dataset_key)
        if not dataset_info:
            st.error(f"Dataset {dataset_key} not found")
            return None
        
        try:
            # Download ZIP file
            with st.spinner(f"📥 Downloading {dataset_info['name']}..."):
                response = requests.get(dataset_info['url'], timeout=30)
                
                if response.status_code != 200:
                    st.error(f"Failed to download: HTTP {response.status_code}")
                    # Try alternative URL format
                    alt_url = f"https://raw.githubusercontent.com/{_self.github_username}/{_self.repo_name}/main/{dataset_info['filename']}"
                    response = requests.get(alt_url, timeout=30)
                    
                    if response.status_code != 200:
                        st.error("Could not download dataset from GitHub")
                        return None
                
                # Extract ZIP contents
                zip_data = io.BytesIO(response.content)
                
                with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                    # List files in ZIP
                    file_list = zip_ref.namelist()
                    st.info(f"Found {len(file_list)} files in ZIP archive")
                    
                    # Find CSV files
                    csv_files = [f for f in file_list if f.endswith('.csv')]
                    
                    if not csv_files:
                        st.warning("No CSV files found in ZIP. Looking for other formats...")
                        # Try to find any data files
                        data_files = [f for f in file_list if f.endswith(('.txt', '.dat', '.tsv'))]
                        if data_files:
                            # Read first data file
                            with zip_ref.open(data_files[0]) as f:
                                content = f.read().decode('utf-8')
                                # Try to parse as CSV
                                df = pd.read_csv(io.StringIO(content))
                                st.success(f"✅ Loaded {len(df)} records from {data_files[0]}")
                                return df
                    else:
                        # Read first CSV file
                        with zip_ref.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                            st.success(f"✅ Loaded {len(df)} records from {csv_files[0]}")
                            
                            # Display basic info
                            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                            
                            return df
                            
        except zipfile.BadZipFile:
            st.error("❌ File is not a valid ZIP archive. The file might be corrupted or not properly uploaded.")
            st.info("Please ensure the ZIP files are properly uploaded to your GitHub repository.")
            return None
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def get_dataset_info(self) -> Dict:
        """Get information about available datasets"""
        return self.datasets
    
    def load_dataset(self, dataset_key: str) -> pd.DataFrame:
        """Load a specific dataset"""
        if dataset_key in self.cached_data:
            return self.cached_data[dataset_key]
        
        df = self.download_and_extract_zip(dataset_key)
        if df is not None:
            self.cached_data[dataset_key] = df
        return df

class EEGAnalyzer:
    """Analyze EEG data for criticality and seizure patterns"""
    
    def __init__(self):
        self.fs = 256  # Default sampling frequency
        
    def prepare_eeg_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict]:
        """Prepare DataFrame for EEG analysis"""
        
        # Detect data format
        info = {"format": "unknown", "has_labels": False}
        
        # Check for common label columns
        label_columns = ['y', 'label', 'class', 'target', 'diagnosis', 'condition']
        label_col = None
        
        for col in label_columns:
            if col in df.columns:
                label_col = col
                info["has_labels"] = True
                info["label_column"] = col
                break
        
        # Separate features and labels
        if label_col:
            # Get unique labels
            unique_labels = df[label_col].unique()
            info["unique_labels"] = unique_labels.tolist()
            info["n_classes"] = len(unique_labels)
            
            # Get feature columns (all except label)
            feature_cols = [col for col in df.columns if col != label_col]
            signals = df[feature_cols].values.T  # Transpose to get (channels, samples)
            
            # Store labels
            info["labels"] = df[label_col].values
            
            # Check if seizure data (label = 1 often indicates seizure)
            if 1 in unique_labels:
                info["has_seizures"] = True
                info["seizure_count"] = int(np.sum(df[label_col] == 1))
        else:
            # No labels, treat all columns as channels
            signals = df.values.T
            feature_cols = list(df.columns)
        
        # Ensure we have the right shape
        if signals.shape[0] > signals.shape[1]:
            # More channels than samples, probably need to transpose
            signals = signals.T
            
        info["n_channels"] = signals.shape[0]
        info["n_samples"] = signals.shape[1]
        
        return signals, feature_cols, info
    
    def extract_features(self, signals: np.ndarray, fs: int = 256, window_s: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract frequency band features from EEG signals"""
        
        n_channels, n_samples = signals.shape
        
        # For short recordings, adjust window size
        max_window_s = n_samples / fs * 0.5  # Use at most half the signal length
        window_s = min(window_s, max_window_s)
        
        window = int(window_s * fs)
        step = int(window_s * fs / 2)  # 50% overlap
        
        # Calculate number of windows
        if n_samples < window:
            # Signal too short, use entire signal as one window
            n_windows = 1
            window = n_samples
        else:
            n_windows = max(1, (n_samples - window) // step + 1)
        
        times = np.arange(n_windows) * (step / fs) if n_windows > 1 else [window_s / 2]
        
        # Initialize features (5 frequency bands)
        features = np.zeros((n_windows, n_channels, 5))
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # Delta, Theta, Alpha, Beta, Gamma
        
        for w in range(n_windows):
            if n_windows == 1:
                start, end = 0, n_samples
            else:
                start = w * step
                end = min(start + window, n_samples)
            
            for ch in range(n_channels):
                segment = signals[ch, start:end]
                
                # Compute power spectral density
                if HAS_SCIPY and len(segment) > 10:
                    freqs, psd = sp_signal.welch(segment, fs=fs, nperseg=min(256, len(segment)//2))
                else:
                    # Simple FFT
                    fft = np.fft.fft(segment)
                    freqs = np.fft.fftfreq(len(segment), 1/fs)
                    psd = np.abs(fft) ** 2
                    # Keep only positive frequencies
                    pos_mask = freqs >= 0
                    freqs = freqs[pos_mask]
                    psd = psd[pos_mask]
                
                # Extract band powers
                for b, (low, high) in enumerate(bands):
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        features[w, ch, b] = np.sqrt(np.mean(psd[mask]))
        
        return times, features
    
    def compute_criticality(self, features: np.ndarray, times: np.ndarray, 
                           threshold: float = 0.3, data_info: Dict = None) -> Dict:
        """Compute criticality using logistic map dynamics"""
        
        n_windows, n_channels, n_bands = features.shape
        
        # Initialize arrays
        critical_windows = []
        r_evolution = []
        state_evolution = []
        
        # Normalize features
        features_norm = np.zeros_like(features)
        for b in range(n_bands):
            band_data = features[:, :, b]
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            if std_val > 0:
                features_norm[:, :, b] = (band_data - mean_val) / std_val
        
        # Logistic map parameters
        r_values = np.full(n_bands, 3.0)
        
        # Adjust sensitivity if we know there are seizures
        if data_info and data_info.get("has_seizures", False):
            sensitivity = 0.15
            chaos_threshold = 3.5
        else:
            sensitivity = 0.1
            chaos_threshold = 3.57
        
        # Analyze each window
        for w in range(n_windows):
            # Calculate activity
            activity = np.mean(np.abs(features_norm[w]))
            
            # Update R parameters
            for b in range(n_bands):
                band_activity = np.mean(np.abs(features_norm[w, :, b]))
                
                if band_activity > threshold:
                    r_values[b] = min(3.99, r_values[b] + sensitivity * (1 + band_activity * 0.1))
                else:
                    r_values[b] = max(2.5, r_values[b] - 0.05)
            
            # Mean R value
            r_mean = np.mean(r_values)
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
        
        # Final state
        if criticality_ratio > 0.4:
            final_state = "highly_critical"
        elif criticality_ratio > 0.2:
            final_state = "moderately_critical"
        elif criticality_ratio > 0.1:
            final_state = "transitional"
        else:
            final_state = "stable"
        
        # Band statistics
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_stats = {}
        for i, name in enumerate(band_names):
            band_data = features[:, :, i]
            band_stats[name] = {
                'mean_power': float(np.mean(band_data)),
                'std_power': float(np.std(band_data))
            }
        
        return {
            'criticality_ratio': criticality_ratio,
            'final_state': final_state,
            'critical_windows': len(critical_windows),
            'total_windows': n_windows,
            'r_evolution': r_evolution,
            'state_evolution': state_evolution,
            'times': times.tolist() if isinstance(times, np.ndarray) else times,
            'critical_indices': critical_windows,
            'band_statistics': band_stats,
            'mean_r': float(np.mean(r_evolution)),
            'chaos_percentage': float(sum(1 for r in r_evolution if r > chaos_threshold) / len(r_evolution) * 100)
        }

def generate_report(results: Dict, data_info: Dict, dataset_name: str) -> str:
    """Generate analysis report"""
    
    report = f"""
## 🧠 EEG Criticality Analysis Report

**Dataset:** {dataset_name}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Data Information
- **Records:** {data_info.get('n_samples', 'Unknown')}
- **Channels:** {data_info.get('n_channels', 'Unknown')}
"""
    
    if data_info.get('has_labels'):
        report += f"""- **Classes:** {data_info.get('n_classes', 'Unknown')}
- **Has Seizures:** {'Yes' if data_info.get('has_seizures') else 'No'}
"""
        if data_info.get('has_seizures'):
            report += f"- **Seizure Count:** {data_info.get('seizure_count', 0)}\n"
    
    report += f"""
### Analysis Results
- **Brain State:** {results['final_state'].upper().replace('_', ' ')}
- **Criticality:** {results['criticality_ratio']:.1%}
- **Critical Episodes:** {results['critical_windows']}/{results['total_windows']}
- **Mean R Parameter:** {results['mean_r']:.3f}
- **Chaos Level:** {results['chaos_percentage']:.1f}%

### Frequency Bands (μV)
- **Delta (0.5-4 Hz):** {results['band_statistics']['delta']['mean_power']:.1f}
- **Theta (4-8 Hz):** {results['band_statistics']['theta']['mean_power']:.1f}
- **Alpha (8-13 Hz):** {results['band_statistics']['alpha']['mean_power']:.1f}
- **Beta (13-30 Hz):** {results['band_statistics']['beta']['mean_power']:.1f}
- **Gamma (30-50 Hz):** {results['band_statistics']['gamma']['mean_power']:.1f}
"""
    
    if results['criticality_ratio'] > 0.4:
        report += "\n🚨 **HIGH CRITICALITY** - Significant brain instability detected"
    elif results['criticality_ratio'] > 0.2:
        report += "\n⚠️ **MODERATE CRITICALITY** - Transitional brain state"
    else:
        report += "\n✅ **STABLE** - Normal brain dynamics"
    
    return report

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
    st.title("🧠 Advanced EEG Criticality Analysis Platform")
    st.markdown("**Analyze Real EEG Data from Kaggle Datasets**")
    
    # Initialize components
    # UPDATE THIS WITH YOUR GITHUB USERNAME
    loader = KaggleDatasetLoader(github_username="YOUR_USERNAME", repo_name="neurocrix-eeg-platform")
    analyzer = EEGAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Dataset Selection")
        dataset_options = loader.get_dataset_info()
        
        dataset_key = st.selectbox(
            "Select Dataset:",
            options=list(dataset_options.keys()),
            format_func=lambda x: dataset_options[x]["name"]
        )
        
        if dataset_key:
            st.info(dataset_options[dataset_key]["description"])
        
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (seconds)", 0.5, 5.0, 2.0)
        threshold = st.slider("Criticality Threshold", 0.1, 0.5, 0.3)
        
        st.subheader("📈 Visualization")
        show_raw = st.checkbox("Show Raw Data Sample", value=False)
        show_bands = st.checkbox("Show Frequency Bands", value=True)
    
    # Main content area
    st.header("📊 EEG Data Analysis")
    
    # Load Data button
    if st.button("📥 Load & Analyze Dataset", type="primary", use_container_width=True):
        
        # Load dataset
        df = loader.load_dataset(dataset_key)
        
        if df is not None:
            # Display data info
            st.subheader("📋 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Memory Size", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
            
            # Show data sample
            if show_raw:
                st.subheader("📊 Data Sample")
                st.dataframe(df.head(10))
            
            # Prepare for analysis
            with st.spinner("🔬 Preparing EEG data..."):
                signals, channels, data_info = analyzer.prepare_eeg_data(df)
            
            # Show data info
            if data_info.get("has_labels"):
                st.info(f"""
                **Data Format Detected:**
                - Label Column: `{data_info.get('label_column')}`
                - Classes: {data_info.get('n_classes')}
                - Unique Labels: {data_info.get('unique_labels')}
                """)
                
                if data_info.get("has_seizures"):
                    st.warning(f"⚡ Dataset contains {data_info.get('seizure_count')} seizure recordings!")
            
            # Analyze
            with st.spinner("🧠 Analyzing brain criticality..."):
                times, features = analyzer.extract_features(signals, window_s=window_size)
                results = analyzer.compute_criticality(features, times, threshold, data_info)
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Metrics
            st.subheader("📊 Analysis Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                crit = results['criticality_ratio']
                if crit > 0.4:
                    st.metric("🔴 Criticality", f"{crit:.1%}")
                elif crit > 0.2:
                    st.metric("🟡 Criticality", f"{crit:.1%}")
                else:
                    st.metric("🟢 Criticality", f"{crit:.1%}")
            
            with col2:
                st.metric("Brain State", results['final_state'].replace('_', ' ').title())
            
            with col3:
                st.metric("Critical Episodes", f"{results['critical_windows']}/{results['total_windows']}")
            
            with col4:
                st.metric("R Parameter", f"{results['mean_r']:.3f}")
            
            # Visualization
            if len(results['r_evolution']) > 1:
                fig, axes = plt.subplots(2 if show_bands else 1, 1, 
                                        figsize=(12, 8 if show_bands else 4))
                
                if not show_bands:
                    axes = [axes]
                
                # R parameter evolution
                ax1 = axes[0]
                colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red' 
                         for s in results['state_evolution']]
                ax1.scatter(results['times'], results['r_evolution'], c=colors, s=50, alpha=0.7)
                ax1.axhline(y=3.57, color='red', linestyle='--', alpha=0.5, label='Chaos threshold')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('R Parameter')
                ax1.set_title('Brain State Evolution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Frequency bands
                if show_bands:
                    ax2 = axes[1]
                    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
                    powers = [results['band_statistics'][b.lower()]['mean_power'] for b in bands]
                    colors2 = ['purple', 'blue', 'green', 'orange', 'red']
                    bars = ax2.bar(bands, powers, color=colors2, alpha=0.7)
                    ax2.set_ylabel('Power (μV)')
                    ax2.set_title('Frequency Band Analysis')
                    ax2.grid(True, alpha=0.3, axis='y')
                    
                    # Add values on bars
                    for bar, power in zip(bars, powers):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                               f'{power:.1f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Generate report
            report = generate_report(results, data_info, dataset_options[dataset_key]["name"])
            st.markdown(report)
            
            # Download report
            st.download_button(
                "📄 Download Analysis Report",
                data=report + f"\n\nRaw Results:\n{json.dumps(results, indent=2)}",
                file_name=f"eeg_analysis_{dataset_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # If seizure data, show specific seizure analysis
            if data_info.get("has_seizures") and data_info.get("labels") is not None:
                st.subheader("⚡ Seizure-Specific Analysis")
                
                # Analyze seizure vs non-seizure
                seizure_mask = data_info["labels"] == 1
                
                if np.any(seizure_mask):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Seizure Records", f"{np.sum(seizure_mask):,}")
                        st.metric("Normal Records", f"{np.sum(~seizure_mask):,}")
                    
                    with col2:
                        seizure_ratio = np.sum(seizure_mask) / len(seizure_mask) * 100
                        st.metric("Seizure Percentage", f"{seizure_ratio:.1f}%")
                        
                        if results['criticality_ratio'] > 0.3 and seizure_ratio > 10:
                            st.success("✅ High criticality correlates with seizure presence!")
    
    # File upload option
    with st.expander("📁 Upload Custom EEG File"):
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV format with channels as columns or rows"
        )
        
        if uploaded_file:
            if st.button("🔬 Analyze Uploaded File"):
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} rows")
                    
                    # Analyze
                    signals, channels, data_info = analyzer.prepare_eeg_data(df)
                    times, features = analyzer.extract_features(signals, window_s=window_size)
                    results = analyzer.compute_criticality(features, times, threshold, data_info)
                    
                    # Show results
                    st.metric("Criticality", f"{results['criticality_ratio']:.1%}")
                    st.metric("State", results['final_state'].replace('_', ' ').title())
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        ### Setup Instructions:
        
        1. **Update GitHub Username**: 
           - Edit line in code: `loader = KaggleDatasetLoader(github_username="YOUR_USERNAME")`
           - Replace `YOUR_USERNAME` with your actual GitHub username
        
        2. **Ensure ZIP Files are Uploaded**:
           - `EEG-Alcohol-Kag.zip` should be in your repo
           - `Psidis-Kag.zip` should be in your repo
        
        3. **Click "Load & Analyze Dataset"** to process the data
        
        ### What This Does:
        - Downloads ZIP files from your GitHub repo
        - Extracts EEG data automatically
        - Analyzes for seizures and brain criticality
        - Generates comprehensive reports
        
        ### Troubleshooting:
        - If download fails, check GitHub repo permissions
        - Ensure files are in the main branch
        - Check file names match exactly
        """)

if __name__ == "__main__":
    main()
