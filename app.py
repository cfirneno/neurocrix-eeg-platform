import streamlit as st
import numpy as np
import pandas as pd
import io
import json
import time
import requests
import re
import zipfile
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib
import base64
import warnings
warnings.filterwarnings('ignore')

# ======================== AUTHENTICATION ========================
USERS = {
    "admin": {
        "password": "neurocrix2024",
        "access_level": "full"
    },
    "researcher": {
        "password": "eeg2024",
        "access_level": "full"
    },
    "demo": {
        "password": "demo123",
        "access_level": "limited"
    }
}

def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'access_level' not in st.session_state:
        st.session_state['access_level'] = None
    if 'login_attempts' not in st.session_state:
        st.session_state['login_attempts'] = 0
    # Data storage
    if 'downloaded_data' not in st.session_state:
        st.session_state['downloaded_data'] = None
    if 'parsed_data' not in st.session_state:
        st.session_state['parsed_data'] = None
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None

def authenticate_user():
    """Handle user authentication with username and password"""
    init_session_state()
    
    if st.session_state.get('authenticated'):
        return True
    
    st.markdown("<h1 style='text-align: center;'>🧠 EEG Analysis Platform</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Secure Login</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### Enter Your Credentials")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                submit = st.form_submit_button("🔓 Login", use_container_width=True, type="primary")
            with col_b:
                demo = st.form_submit_button("📖 Demo Access", use_container_width=True)
            
            if demo:
                st.info("💡 Use username: **demo** and password: **demo123** for demo access")
            
            if submit:
                if username in USERS and USERS[username]["password"] == password:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['access_level'] = USERS[username]["access_level"]
                    st.session_state['login_attempts'] = 0
                    st.success(f"✅ Welcome, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state['login_attempts'] += 1
                    if st.session_state['login_attempts'] >= 3:
                        st.error(f"❌ Too many failed attempts. Please contact administrator.")
                        time.sleep(3)
                    else:
                        remaining = 3 - st.session_state['login_attempts']
                        st.error(f"❌ Invalid credentials. {remaining} attempts remaining.")
        
        with st.expander("🔑 Test Accounts"):
            st.markdown("""
            | Username | Password | Access |
            |----------|----------|--------|
            | admin | neurocrix2024 | Full |
            | researcher | eeg2024 | Full |
            | demo | demo123 | Limited |
            """)
    
    return False

# ======================== DATABASE CONFIGURATION ========================
REAL_DATABASES = {
    "psi_database": {
        "name": "🍄 PSI Psychedelic EEG Database",
        "description": "Real EEG recordings from psychedelic/psychiatric studies",
        "url": "https://github.com/cfirneno/neurocrix-eeg-platform/raw/main/psydis-Kag.zip",
        "type": "github",
        "format": "zip"
    },
    "alcohol_database": {
        "name": "🍺 Alcohol Effects EEG Database", 
        "description": "EEG recordings studying alcohol effects on brain activity",
        "url": "1loM7-BHPwboU64Tsb10baO0vpbI4LMX0",
        "type": "gdrive",
        "format": "zip"
    },
    "demo_database": {
        "name": "🎮 Demo EEG Database",
        "description": "Generated demonstration EEG patterns for testing",
        "url": "demo",
        "type": "demo",
        "format": "generated"
    }
}

# Optional imports
try:
    import scipy.signal as sp_signal
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# ======================== STEP 1: DATA DOWNLOAD ========================
class DataDownloader:
    """Handle downloading from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 EEG-Platform/2.0'
        })
    
    def download_from_github(self, url: str) -> Tuple[bool, Optional[bytes], str]:
        """Download from GitHub"""
        try:
            if "/blob/" in url:
                url = url.replace("/blob/", "/raw/")
            
            response = self.session.get(url, timeout=60)
            if response.status_code == 200:
                return True, response.content, "Successfully downloaded from GitHub"
            else:
                return False, None, f"GitHub download failed: HTTP {response.status_code}"
        except Exception as e:
            return False, None, f"GitHub error: {str(e)}"
    
    def download_from_gdrive(self, file_id: str) -> Tuple[bool, Optional[bytes], str]:
        """Download from Google Drive"""
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = self.session.get(download_url, timeout=60)
            
            if response.status_code == 200:
                return True, response.content, "Successfully downloaded from Google Drive"
            else:
                return False, None, f"Google Drive download failed"
        except Exception as e:
            return False, None, f"Google Drive error: {str(e)}"
    
    def generate_demo_data(self) -> Tuple[bool, Optional[bytes], str]:
        """Generate demo data as zip"""
        try:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                # Generate multiple demo files
                for i in range(3):
                    fs = 256
                    duration = 30
                    n_channels = 8
                    n_samples = fs * duration
                    time = np.linspace(0, duration, n_samples)
                    
                    data = {'Time': time}
                    for ch in range(n_channels):
                        signal = 30 * np.sin(2 * np.pi * (10 + ch) * time + np.random.random() * 2 * np.pi)
                        signal += 20 * np.sin(2 * np.pi * 20 * time)
                        signal += np.random.normal(0, 5, n_samples)
                        data[f'CH_{ch+1}'] = signal
                    
                    df = pd.DataFrame(data)
                    csv_content = df.to_csv(index=False)
                    zf.writestr(f'demo_subject_{i+1}.csv', csv_content)
            
            return True, zip_buffer.getvalue(), "Demo data generated"
        except Exception as e:
            return False, None, f"Demo generation failed: {str(e)}"
    
    def download_dataset(self, database_id: str) -> Tuple[bool, Optional[bytes], str]:
        """Download dataset based on type"""
        if database_id not in REAL_DATABASES:
            return False, None, "Database not found"
        
        db_config = REAL_DATABASES[database_id]
        
        if db_config["type"] == "demo":
            return self.generate_demo_data()
        elif db_config["type"] == "github":
            return self.download_from_github(db_config["url"])
        elif db_config["type"] == "gdrive":
            return self.download_from_gdrive(db_config["url"])
        else:
            return False, None, "Unknown database type"

# ======================== STEP 2: DATA PARSING ========================
class DataParser:
    """Parse and validate downloaded data"""
    
    def extract_zip_contents(self, zip_content: bytes) -> Dict:
        """Extract contents from zip file"""
        extracted = {
            "files": [],
            "csv_files": [],
            "txt_files": [],
            "json_files": [],
            "other_files": []
        }
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
                for filename in zf.namelist():
                    if filename.endswith('/') or filename.startswith('__'):
                        continue
                    
                    extracted["files"].append(filename)
                    
                    if filename.endswith('.csv'):
                        extracted["csv_files"].append(filename)
                    elif filename.endswith('.txt'):
                        extracted["txt_files"].append(filename)
                    elif filename.endswith('.json'):
                        extracted["json_files"].append(filename)
                    else:
                        extracted["other_files"].append(filename)
            
            return extracted
        except Exception as e:
            st.error(f"Failed to extract zip: {str(e)}")
            return extracted
    
    def parse_csv_file(self, zip_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Parse a specific CSV file from zip"""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
                with zf.open(filename) as f:
                    # Try different encodings
                    content = f.read()
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                            return df
                        except:
                            continue
            return None
        except Exception as e:
            st.error(f"Failed to parse CSV {filename}: {str(e)}")
            return None
    
    def validate_eeg_data(self, df: pd.DataFrame) -> Tuple[bool, List[str], str]:
        """Validate if DataFrame contains valid EEG data"""
        # Find numeric columns
        numeric_cols = []
        for col in df.columns:
            try:
                # Check if column can be converted to numeric
                test_series = pd.to_numeric(df[col], errors='coerce')
                # If more than 80% of values are numeric, consider it valid
                if test_series.notna().sum() / len(test_series) > 0.8:
                    numeric_cols.append(col)
            except:
                pass
        
        # Check if we have enough numeric columns (at least 1 channel)
        if len(numeric_cols) == 0:
            return False, [], "No numeric columns found"
        
        # Filter out time/index columns if present
        eeg_cols = []
        for col in numeric_cols:
            if not any(keyword in str(col).lower() for keyword in ['time', 'index', 'timestamp', 'sample']):
                eeg_cols.append(col)
        
        # If all columns were time/index, use all numeric columns
        if len(eeg_cols) == 0:
            eeg_cols = numeric_cols
        
        # Check if we have enough samples
        if len(df) < 100:
            return False, eeg_cols, "Not enough samples (minimum 100 required)"
        
        return True, eeg_cols, "Valid EEG data"
    
    def prepare_eeg_array(self, df: pd.DataFrame, eeg_columns: List[str]) -> np.ndarray:
        """Convert DataFrame to EEG array format"""
        # Extract only EEG columns
        eeg_df = df[eeg_columns].copy()
        
        # Convert to numeric and handle NaN
        for col in eeg_columns:
            eeg_df[col] = pd.to_numeric(eeg_df[col], errors='coerce')
        
        # Fill NaN with 0
        eeg_df = eeg_df.fillna(0)
        
        # Convert to numpy array
        data = eeg_df.values
        
        # Ensure channels x samples format
        if data.shape[0] > data.shape[1]:
            # More rows than columns - transpose
            data = data.T
        
        return data.astype(float)

# ======================== STEP 3: EEG ANALYSIS ========================
class EEGAnalyzer:
    """Perform EEG analysis with chaos theory metrics"""
    
    def compute_band_powers(self, signals: np.ndarray, fs: int, window_s: float = 2.0) -> Dict:
        """Compute frequency band powers"""
        n_channels, n_samples = signals.shape
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            powers = []
            for ch in range(n_channels):
                if HAS_SCIPY:
                    freqs, psd = sp_signal.welch(signals[ch], fs=fs, nperseg=min(256, n_samples))
                    mask = (freqs >= low) & (freqs <= high)
                    power = np.trapz(psd[mask], freqs[mask])
                else:
                    # FFT fallback
                    fft = np.fft.fft(signals[ch])
                    freqs = np.fft.fftfreq(len(fft), 1/fs)
                    psd = np.abs(fft) ** 2
                    mask = (freqs >= low) & (freqs <= high) & (freqs >= 0)
                    power = np.sum(psd[mask])
                powers.append(power)
            band_powers[band_name] = np.mean(powers)
        
        return band_powers
    
    def compute_lyapunov_exponent(self, signal: np.ndarray) -> float:
        """Simplified Lyapunov exponent calculation"""
        n = len(signal)
        if n < 100:
            return 0.0
        
        # Simple divergence measure
        divergences = []
        for i in range(n - 10):
            if i + 20 < n:
                d1 = abs(signal[i] - signal[i+1])
                d2 = abs(signal[i+10] - signal[i+11])
                if d1 > 0:
                    divergences.append(d2 / d1)
        
        if divergences:
            return np.mean(np.log(np.array(divergences) + 1e-10))
        return 0.0
    
    def compute_logistic_map_r(self, signal: np.ndarray) -> float:
        """Estimate R parameter from signal dynamics"""
        # Normalize signal
        sig_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Calculate variability measure
        variability = np.std(np.diff(sig_norm))
        
        # Map to R parameter range (2.5 to 4.0)
        r_param = 2.5 + variability * 1.5
        r_param = np.clip(r_param, 2.5, 4.0)
        
        return r_param
    
    def analyze_eeg(self, signals: np.ndarray, fs: int) -> Dict:
        """Perform complete EEG analysis"""
        n_channels, n_samples = signals.shape
        
        # Compute band powers
        band_powers = self.compute_band_powers(signals, fs)
        
        # Compute chaos metrics for each channel
        lyapunov_exps = []
        r_params = []
        
        for ch in range(n_channels):
            lyap = self.compute_lyapunov_exponent(signals[ch])
            r_param = self.compute_logistic_map_r(signals[ch])
            lyapunov_exps.append(lyap)
            r_params.append(r_param)
        
        # Average metrics
        mean_lyapunov = np.mean(lyapunov_exps)
        mean_r = np.mean(r_params)
        
        # Determine criticality
        chaos_threshold = 3.57
        if mean_r > chaos_threshold or mean_lyapunov > 0.1:
            criticality = "HIGH"
            criticality_score = 0.8
        elif mean_r > 3.3 or mean_lyapunov > 0.05:
            criticality = "MODERATE"
            criticality_score = 0.5
        else:
            criticality = "LOW"
            criticality_score = 0.2
        
        return {
            'n_channels': n_channels,
            'n_samples': n_samples,
            'sampling_rate': fs,
            'duration_seconds': n_samples / fs,
            'band_powers': band_powers,
            'mean_lyapunov': mean_lyapunov,
            'mean_r_parameter': mean_r,
            'criticality': criticality,
            'criticality_score': criticality_score,
            'chaos_percentage': (mean_r - 2.5) / 1.5 * 100
        }

# ======================== MAIN APPLICATION ========================
def main():
    st.set_page_config(
        page_title="🧠 EEG Analysis Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Authentication
    if not authenticate_user():
        st.stop()
    
    # Header
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🧠 Advanced EEG Criticality Analysis Platform")
        st.caption("Modular Architecture: Download → Parse → Analyze")
    with col2:
        st.markdown(f"**User:** {st.session_state['username']}")
        if st.button("🚪 Logout"):
            for key in ['authenticated', 'username', 'access_level']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Initialize components
    downloader = DataDownloader()
    parser = DataParser()
    analyzer = EEGAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection
        st.subheader("📊 Select Database")
        
        # List databases explicitly
        database_options = ["psi_database", "alcohol_database", "demo_database"]
        database_names = {
            "psi_database": "🍄 PSI Psychedelic EEG",
            "alcohol_database": "🍺 Alcohol Effects EEG",
            "demo_database": "🎮 Demo EEG"
        }
        
        database_id = st.selectbox(
            "Choose Dataset:",
            options=database_options,
            format_func=lambda x: database_names.get(x, x)
        )
        
        if database_id and database_id in REAL_DATABASES:
            db_info = REAL_DATABASES[database_id]
            st.info(f"""
            **Type:** {db_info['type'].upper()}
            
            **Format:** {db_info['format'].upper()}
            
            **Description:** {db_info['description']}
            """)
        
        st.subheader("🔧 Analysis Settings")
        sampling_rate = st.number_input("Sampling Rate (Hz)", 100, 1000, 256)
        window_size = st.slider("Window Size (seconds)", 1.0, 5.0, 2.0)
    
    # Main content - Three step process
    st.header("📊 EEG Analysis Workflow")
    
    # Create three columns for the workflow
    step1, step2, step3 = st.columns(3)
    
    # STEP 1: DOWNLOAD
    with step1:
        st.subheader("Step 1️⃣: Download Data")
        
        if database_id:
            st.write(f"**Selected:** {database_names.get(database_id, database_id)}")
            
            if st.button("📥 Download Dataset", use_container_width=True, key="download_btn"):
                with st.spinner("Downloading..."):
                    success, content, message = downloader.download_dataset(database_id)
                    
                    if success and content:
                        st.session_state['downloaded_data'] = {
                            'content': content,
                            'database': database_id,
                            'timestamp': datetime.now()
                        }
                        st.success(f"✅ {message}")
                        st.info(f"Size: {len(content) / 1024:.1f} KB")
                    else:
                        st.error(f"❌ {message}")
        
        # Show download status
        if st.session_state.get('downloaded_data'):
            st.success("✅ Data Downloaded")
            st.caption(f"Downloaded at {st.session_state['downloaded_data']['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("⏳ No data downloaded yet")
    
    # STEP 2: PARSE
    with step2:
        st.subheader("Step 2️⃣: Parse & Validate")
        
        if st.session_state.get('downloaded_data'):
            if st.button("🔍 Parse Data", use_container_width=True, key="parse_btn"):
                with st.spinner("Parsing..."):
                    zip_content = st.session_state['downloaded_data']['content']
                    
                    # Extract zip contents
                    extracted = parser.extract_zip_contents(zip_content)
                    
                    st.write(f"**Files found:** {len(extracted['files'])}")
                    st.write(f"- CSV files: {len(extracted['csv_files'])}")
                    st.write(f"- TXT files: {len(extracted['txt_files'])}")
                    
                    # Try to parse CSV files
                    valid_data = None
                    for csv_file in extracted['csv_files']:
                        df = parser.parse_csv_file(zip_content, csv_file)
                        if df is not None:
                            is_valid, eeg_cols, msg = parser.validate_eeg_data(df)
                            if is_valid:
                                st.success(f"✅ Valid EEG in {csv_file}")
                                st.write(f"Channels: {len(eeg_cols)}")
                                st.write(f"Samples: {len(df)}")
                                
                                # Prepare data array
                                eeg_array = parser.prepare_eeg_array(df, eeg_cols)
                                
                                st.session_state['parsed_data'] = {
                                    'signals': eeg_array,
                                    'channels': eeg_cols[:eeg_array.shape[0]],
                                    'filename': csv_file,
                                    'sampling_rate': sampling_rate
                                }
                                valid_data = True
                                break
                            else:
                                st.warning(f"⚠️ {csv_file}: {msg}")
                    
                    if not valid_data:
                        st.error("❌ No valid EEG data found")
        else:
            st.info("⏳ Download data first")
        
        # Show parse status
        if st.session_state.get('parsed_data'):
            st.success("✅ Data Parsed")
            data = st.session_state['parsed_data']
            st.caption(f"{data['signals'].shape[0]} channels × {data['signals'].shape[1]} samples")
        else:
            st.info("⏳ No data parsed yet")
    
    # STEP 3: ANALYZE
    with step3:
        st.subheader("Step 3️⃣: Analyze EEG")
        
        if st.session_state.get('parsed_data'):
            if st.button("🧠 Analyze", use_container_width=True, key="analyze_btn"):
                with st.spinner("Analyzing..."):
                    data = st.session_state['parsed_data']
                    
                    # Perform analysis
                    results = analyzer.analyze_eeg(
                        data['signals'],
                        data['sampling_rate']
                    )
                    
                    st.session_state['analysis_results'] = results
                    
                    # Show key metrics
                    st.metric("Criticality", results['criticality'])
                    st.metric("Chaos %", f"{results['chaos_percentage']:.1f}%")
                    st.metric("Lyapunov", f"{results['mean_lyapunov']:.3f}")
        else:
            st.info("⏳ Parse data first")
        
        # Show analysis status
        if st.session_state.get('analysis_results'):
            st.success("✅ Analysis Complete")
        else:
            st.info("⏳ No analysis yet")
    
    # Results Section
    if st.session_state.get('analysis_results'):
        st.header("📈 Analysis Results")
        
        results = st.session_state['analysis_results']
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            color = "🔴" if results['criticality'] == "HIGH" else "🟡" if results['criticality'] == "MODERATE" else "🟢"
            st.metric(f"{color} Criticality", results['criticality'])
        
        with col2:
            st.metric("R Parameter", f"{results['mean_r_parameter']:.3f}")
        
        with col3:
            st.metric("Lyapunov", f"{results['mean_lyapunov']:.4f}")
        
        with col4:
            st.metric("Chaos %", f"{results['chaos_percentage']:.1f}%")
        
        with col5:
            st.metric("Duration", f"{results['duration_seconds']:.1f}s")
        
        # Frequency bands
        st.subheader("🌊 Frequency Band Powers")
        band_df = pd.DataFrame([results['band_powers']])
        st.dataframe(band_df, use_container_width=True)
        
        # Visualization
        if HAS_MATPLOTLIB:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Band powers bar chart
            bands = list(results['band_powers'].keys())
            powers = list(results['band_powers'].values())
            colors = ['#4B0082', '#0000FF', '#00FF00', '#FFA500', '#FF0000']
            
            ax1.bar(bands, powers, color=colors)
            ax1.set_ylabel("Power (µV²)")
            ax1.set_title("Frequency Band Analysis")
            ax1.grid(True, alpha=0.3)
            
            # Signal preview (first channel, first 1000 samples)
            if st.session_state.get('parsed_data'):
                signal = st.session_state['parsed_data']['signals'][0][:1000]
                time = np.arange(len(signal)) / results['sampling_rate']
                ax2.plot(time, signal, 'b-', alpha=0.7)
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Amplitude (µV)")
                ax2.set_title("EEG Signal Preview (Channel 1)")
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Report
        st.subheader("📄 Analysis Report")
        report = f"""
        ## EEG Analysis Report
        
        **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Database:** {st.session_state['downloaded_data']['database']}
        **File:** {st.session_state['parsed_data']['filename']}
        
        ### Signal Properties
        - Channels: {results['n_channels']}
        - Samples: {results['n_samples']}
        - Sampling Rate: {results['sampling_rate']} Hz
        - Duration: {results['duration_seconds']:.2f} seconds
        
        ### Chaos Metrics
        - Mean R Parameter: {results['mean_r_parameter']:.4f}
        - Mean Lyapunov Exponent: {results['mean_lyapunov']:.4f}
        - Chaos Percentage: {results['chaos_percentage']:.2f}%
        - Criticality Level: {results['criticality']}
        
        ### Frequency Analysis
        - Delta (0.5-4 Hz): {results['band_powers']['delta']:.2f}
        - Theta (4-8 Hz): {results['band_powers']['theta']:.2f}
        - Alpha (8-13 Hz): {results['band_powers']['alpha']:.2f}
        - Beta (13-30 Hz): {results['band_powers']['beta']:.2f}
        - Gamma (30-50 Hz): {results['band_powers']['gamma']:.2f}
        """
        
        st.text_area("Report", report, height=400)
        
        # Download button
        st.download_button(
            "📥 Download Report",
            data=report,
            file_name=f"eeg_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
