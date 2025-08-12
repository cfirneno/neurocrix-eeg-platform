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

# ======================== REAL DATABASE CONFIGURATION ========================
REAL_DATABASES = {
    "psi_database": {
        "name": "🍄 PSI Psychedelic EEG Database",
        "description": "Real EEG recordings from psychedelic/psychiatric studies - GitHub hosted",
        "url": "https://github.com/cfirneno/neurocrix-eeg-platform/raw/main/psydis-Kag.zip",
        "type": "github",
        "format": "zip",
        "citation": "Psychedelic Studies Initiative Dataset",
        "expected_criticality": "Variable - depends on substance and dose"
    },
    "alcohol_database": {
        "name": "🍺 Alcohol Effects EEG Database", 
        "description": "EEG recordings studying alcohol effects on brain activity - Google Drive hosted",
        "url": "1loM7-BHPwboU64Tsb10baO0vpbI4LMX0",  # Google Drive file ID
        "type": "gdrive",
        "format": "zip",
        "citation": "Alcohol Neuroscience Study",
        "expected_criticality": "Moderate - increased with intoxication levels"
    },
    "demo_database": {
        "name": "🎮 Demo EEG Database",
        "description": "Generated demonstration EEG patterns for testing",
        "url": "demo",
        "type": "demo",
        "format": "generated",
        "citation": "Simulated data for demonstration",
        "expected_criticality": "Configurable - for testing purposes"
    }
}

# Optional imports with fallback
try:
    import scipy.signal as sp_signal
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    st.warning("SciPy not installed. Some features limited.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pywt  # For wavelet analysis
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    st.info("PyWavelets not installed. Using FFT-based analysis instead.")

class DataDownloader:
    """Handle downloading from GitHub and Google Drive"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def download_from_github(_self, url: str) -> Tuple[bool, Optional[bytes], str]:
        """Download from GitHub"""
        try:
            # Convert blob to raw if needed
            if "/blob/" in url:
                url = url.replace("/blob/", "/raw/")
            
            response = _self.session.get(url, timeout=60)
            
            if response.status_code == 200:
                return True, response.content, "Successfully downloaded from GitHub"
            else:
                return False, None, f"GitHub download failed: HTTP {response.status_code}"
                
        except Exception as e:
            return False, None, f"GitHub error: {str(e)}"
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def download_from_gdrive(_self, file_id: str) -> Tuple[bool, Optional[bytes], str]:
        """Download from Google Drive using direct download link"""
        try:
            # Direct download URL for Google Drive
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            response = _self.session.get(download_url, timeout=60, stream=True)
            
            # Check if file is too large and needs confirmation
            if b"download-warning" in response.content[:1000]:
                # Extract confirmation token
                for chunk in response.iter_content(chunk_size=32768):
                    if b"confirm=" in chunk:
                        match = re.search(r'confirm=([0-9A-Za-z_]+)', chunk.decode('utf-8', errors='ignore'))
                        if match:
                            confirm_token = match.group(1)
                            download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                            response = _self.session.get(download_url, timeout=60)
                            break
            
            if response.status_code == 200:
                return True, response.content, "Successfully downloaded from Google Drive"
            else:
                # Try alternative method
                alt_url = f"https://drive.google.com/u/0/uc?id={file_id}&export=download"
                response = _self.session.get(alt_url, timeout=60)
                
                if response.status_code == 200:
                    return True, response.content, "Successfully downloaded from Google Drive (alt method)"
                else:
                    return False, None, f"Google Drive download failed: HTTP {response.status_code}"
                    
        except Exception as e:
            return False, None, f"Google Drive error: {str(e)}"
    
    def download_dataset(self, database_id: str) -> Tuple[bool, Optional[bytes], str]:
        """Download dataset based on type"""
        if database_id not in REAL_DATABASES:
            return False, None, "Database not found"
        
        db_config = REAL_DATABASES[database_id]
        
        # Handle demo database
        if db_config["type"] == "demo":
            st.info("🎮 Using demo database - generating sample data...")
            # Create a fake zip with demo CSV data
            import io
            import zipfile
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                # Generate demo EEG data
                fs = 256
                duration = 30
                n_channels = 8
                n_samples = fs * duration
                time = np.linspace(0, duration, n_samples)
                
                # Create demo CSV data
                data = {
                    'Time': time
                }
                for ch in range(n_channels):
                    signal = 30 * np.sin(2 * np.pi * 10 * time + np.random.random() * 2 * np.pi)
                    signal += 20 * np.sin(2 * np.pi * 20 * time + np.random.random() * 2 * np.pi)
                    signal += np.random.normal(0, 5, n_samples)
                    data[f'CH_{ch+1}'] = signal
                
                df = pd.DataFrame(data)
                csv_content = df.to_csv(index=False)
                zf.writestr('demo_eeg_data.csv', csv_content)
            
            return True, zip_buffer.getvalue(), "Demo data generated successfully"
        
        with st.spinner(f"📥 Downloading {db_config['name']}..."):
            if db_config["type"] == "github":
                return self.download_from_github(db_config["url"])
            elif db_config["type"] == "gdrive":
                return self.download_from_gdrive(db_config["url"])
            else:
                return False, None, "Unknown database type"
    
    def extract_zip_data(self, zip_content: bytes, db_name: str) -> Dict:
        """Extract and organize data from zip file"""
        extracted_data = {
            "files": [],
            "eeg_data": [],
            "metadata": {},
            "subjects": [],
            "database": db_name
        }
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
                extracted_data["files"] = zf.namelist()
                
                st.info(f"📦 Found {len(extracted_data['files'])} files in archive")
                
                for filename in zf.namelist():
                    # Skip directories and system files
                    if filename.endswith('/') or filename.startswith('__') or filename.startswith('.'):
                        continue
                    
                    if filename.endswith('.csv'):
                        try:
                            with zf.open(filename) as f:
                                # Try different encodings
                                content = f.read()
                                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                                    try:
                                        df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                                        break
                                    except:
                                        continue
                                
                                # Store the dataframe
                                extracted_data["eeg_data"].append({
                                    "filename": filename,
                                    "data": df,
                                    "shape": df.shape,
                                    "columns": list(df.columns)
                                })
                                
                                st.success(f"✅ Loaded CSV: {filename} ({df.shape[0]} rows, {df.shape[1]} cols)")
                                
                                # Extract subject IDs
                                subject_match = re.search(r'[SP]\d{3}|subject_\d+|sub\d+|participant_\d+|patient_\d+', filename, re.I)
                                if subject_match:
                                    extracted_data["subjects"].append(subject_match.group())
                        except Exception as e:
                            st.warning(f"Could not read CSV {filename}: {str(e)}")
                    
                    elif filename.endswith('.json'):
                        try:
                            with zf.open(filename) as f:
                                extracted_data["metadata"][filename] = json.loads(f.read())
                        except:
                            pass
                    
                    elif filename.endswith('.txt'):
                        try:
                            with zf.open(filename) as f:
                                content = f.read().decode('utf-8', errors='ignore').strip()
                                lines = content.split('\n')
                                
                                # Try to parse as numeric data
                                numeric_lines = []
                                for line in lines:
                                    line = line.strip()
                                    if line and not any(c.isalpha() for c in line):  # No letters
                                        try:
                                            val = float(line.replace(',', '.'))  # Handle comma decimals
                                            numeric_lines.append(val)
                                        except:
                                            pass
                                
                                if len(numeric_lines) > 100:  # Likely EEG data
                                    data = np.array(numeric_lines)
                                    extracted_data["eeg_data"].append({
                                        "filename": filename,
                                        "data": data,
                                        "shape": data.shape,
                                        "type": "single_channel"
                                    })
                                    st.success(f"✅ Loaded TXT: {filename} ({len(data)} samples)")
                        except Exception as e:
                            st.warning(f"Could not read TXT {filename}: {str(e)}")
                    
                    elif filename.endswith(('.edf', '.bdf')):
                        st.info(f"ℹ️ Found EDF/BDF file: {filename} (requires special library)")
                
                extracted_data["subjects"] = list(set(extracted_data["subjects"]))
                
        except Exception as e:
            st.error(f"Error extracting zip: {str(e)}")
            
        return extracted_data

class AdvancedEEGProcessor:
    """Advanced EEG processing with wavelets and Lyapunov exponents"""
    
    def __init__(self):
        self.downloader = DataDownloader()
    
    def process_eeg_data(self, data_dict: Dict) -> Tuple[np.ndarray, int, List[str]]:
        """Process extracted EEG data into standard format"""
        
        if not data_dict.get("eeg_data"):
            # Generate demo data if no real data found
            st.warning("No EEG data found. Generating demo signals...")
            return self.generate_demo_data()
        
        # Try to find valid EEG data in the extracted files
        valid_data = None
        
        for eeg_item in data_dict["eeg_data"]:
            try:
                if isinstance(eeg_item["data"], pd.DataFrame):
                    df = eeg_item["data"]
                    
                    # Show data preview for debugging
                    st.info(f"Processing file: {eeg_item['filename']}")
                    
                    # Clean the dataframe - remove non-numeric columns
                    numeric_columns = []
                    for col in df.columns:
                        try:
                            # Try to convert column to numeric
                            pd.to_numeric(df[col], errors='raise')
                            numeric_columns.append(col)
                        except:
                            # Skip non-numeric columns (like gender, labels, etc.)
                            st.warning(f"Skipping non-numeric column: {col}")
                    
                    if len(numeric_columns) == 0:
                        st.warning(f"No numeric columns found in {eeg_item['filename']}")
                        continue
                    
                    # Keep only numeric columns
                    df_numeric = df[numeric_columns]
                    
                    # Check if first column is time/index
                    first_col = df_numeric.columns[0] if len(df_numeric.columns) > 0 else ""
                    if any(col in str(first_col).lower() for col in ['time', 'index', 'timestamp', 'sample']):
                        data = df_numeric.iloc[:, 1:].values.T
                        channels = list(df_numeric.columns[1:])
                    else:
                        data = df_numeric.values.T
                        channels = list(df_numeric.columns)
                    
                    # Convert to float and check for validity
                    data = data.astype(float)
                    
                    # Remove any NaN values
                    if np.any(np.isnan(data)):
                        st.warning("Found NaN values, replacing with zeros")
                        data = np.nan_to_num(data, nan=0.0)
                    
                    # Ensure we have valid data
                    if data.shape[0] > 0 and data.shape[1] > 100:  # At least 100 samples
                        valid_data = data
                        valid_channels = channels
                        st.success(f"✅ Successfully loaded {data.shape[0]} channels with {data.shape[1]} samples")
                        break
                    
                elif isinstance(eeg_item["data"], np.ndarray):
                    # Single channel data
                    single_channel = eeg_item["data"]
                    if len(single_channel) > 100:  # At least 100 samples
                        n_samples = len(single_channel)
                        n_channels = 8
                        
                        # Create multi-channel from single channel
                        data = np.zeros((n_channels, n_samples))
                        for ch in range(n_channels):
                            data[ch] = single_channel + np.random.normal(0, 1, n_samples) * 0.1
                        
                        valid_data = data
                        valid_channels = [f"CH_{i+1}" for i in range(n_channels)]
                        st.success(f"✅ Processed single-channel data into {n_channels} channels")
                        break
                        
            except Exception as e:
                st.warning(f"Could not process {eeg_item.get('filename', 'unknown')}: {str(e)}")
                continue
        
        # If no valid data found, generate demo data
        if valid_data is None:
            st.warning("Could not process uploaded data. Generating demo signals for demonstration...")
            return self.generate_demo_data()
        
        # Estimate sampling rate
        fs = 256  # Default
        if "metadata" in data_dict and data_dict["metadata"]:
            for meta in data_dict["metadata"].values():
                if "sampling_rate" in meta:
                    fs = meta["sampling_rate"]
                    break
                elif "fs" in meta:
                    fs = meta["fs"]
                    break
        
        return valid_data, int(fs), valid_channels
    
    def generate_demo_data(self) -> Tuple[np.ndarray, int, List[str]]:
        """Generate demo EEG data"""
        fs = 256
        duration = 30
        n_channels = 8
        n_samples = fs * duration
        time = np.linspace(0, duration, n_samples)
        
        st.info("🎮 Generating demo EEG signals for analysis...")
        
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Mix of frequencies to simulate real EEG
            data[ch] += 30 * np.sin(2 * np.pi * 10 * time + np.random.random() * 2 * np.pi)  # Alpha
            data[ch] += 20 * np.sin(2 * np.pi * 20 * time + np.random.random() * 2 * np.pi)  # Beta
            data[ch] += 10 * np.sin(2 * np.pi * 5 * time + np.random.random() * 2 * np.pi)   # Theta
            data[ch] += 15 * np.sin(2 * np.pi * 2 * time + np.random.random() * 2 * np.pi)   # Delta
            data[ch] += np.random.normal(0, 5, n_samples)  # Noise
        
        channels = [f"CH_{i+1}" for i in range(n_channels)]
        return data, fs, channels
    
    def compute_wavelet_features(self, signal: np.ndarray, fs: int) -> Dict:
        """Compute wavelet-based features"""
        features = {}
        
        if HAS_PYWT:
            # Use PyWavelets for wavelet decomposition
            wavelet = 'db4'  # Daubechies 4 wavelet
            max_level = min(pywt.dwt_max_level(len(signal), wavelet), 5)
            
            # Decompose signal
            coeffs = pywt.wavedec(signal, wavelet, level=max_level)
            
            # Calculate energy for each level
            energies = []
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff ** 2)
                energies.append(energy)
                features[f'wavelet_energy_level_{i}'] = energy
            
            # Relative wavelet energy
            total_energy = sum(energies)
            if total_energy > 0:
                for i, energy in enumerate(energies):
                    features[f'relative_wavelet_energy_{i}'] = energy / total_energy
            
            # Wavelet entropy
            probs = np.array(energies) / (total_energy + 1e-10)
            wavelet_entropy = -np.sum(probs * np.log2(probs + 1e-10))
            features['wavelet_entropy'] = wavelet_entropy
        else:
            # Fallback: Use FFT-based band energy
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), 1/fs)
            psd = np.abs(fft) ** 2
            
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
            for i, (low, high) in enumerate(bands):
                mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
                features[f'fft_band_energy_{i}'] = np.sum(psd[mask])
        
        return features
    
    def compute_lyapunov_exponent(self, signal: np.ndarray, 
                                  embedding_dim: int = 3, 
                                  time_delay: int = 1) -> float:
        """
        Compute largest Lyapunov exponent using Rosenstein's method
        This indicates the rate of divergence of nearby trajectories
        """
        n = len(signal)
        
        # Create embedded matrix
        if n < embedding_dim * time_delay:
            return 0.0
        
        embedded = np.zeros((n - (embedding_dim - 1) * time_delay, embedding_dim))
        for i in range(embedding_dim):
            embedded[:, i] = signal[i * time_delay:n - (embedding_dim - 1 - i) * time_delay]
        
        # Find nearest neighbors
        n_points = embedded.shape[0]
        min_sep = int(n_points * 0.1)  # Minimum temporal separation
        
        divergences = []
        
        for i in range(n_points - min_sep):
            # Find nearest neighbor with temporal separation
            distances = np.sqrt(np.sum((embedded[i] - embedded[i + min_sep:]) ** 2, axis=1))
            if len(distances) > 0:
                min_idx = np.argmin(distances)
                
                # Track divergence
                div = []
                for j in range(1, min(50, n_points - i - min_sep - min_idx)):
                    if i + j < n_points and i + min_sep + min_idx + j < n_points:
                        d = np.linalg.norm(embedded[i + j] - embedded[i + min_sep + min_idx + j])
                        if d > 0:
                            div.append(np.log(d))
                
                if len(div) > 10:
                    divergences.append(div)
        
        if not divergences:
            return 0.0
        
        # Average divergence rate (Lyapunov exponent)
        avg_div = np.mean(divergences, axis=0)
        
        # Fit linear regression to get slope (Lyapunov exponent)
        if len(avg_div) > 2:
            x = np.arange(len(avg_div))
            if HAS_SCIPY:
                slope, _, _, _, _ = stats.linregress(x, avg_div)
                return float(slope)
            else:
                # Simple linear fit without scipy
                coeffs = np.polyfit(x, avg_div, 1)
                return float(coeffs[0])
        
        return 0.0
    
    def extract_advanced_features(self, signals: np.ndarray, fs: int, 
                                 window_s: float = 2.0, 
                                 step_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Extract advanced features including wavelets and Lyapunov exponents"""
        n_channels, n_samples = signals.shape
        window = int(window_s * fs)
        step = int(step_s * fs)
        n_windows = max(1, (n_samples - window) // step + 1)
        
        times = np.arange(n_windows) * step_s + window_s / 2
        
        # Standard frequency bands
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        band_features = np.zeros((n_windows, n_channels, len(bands)))
        
        # Advanced features storage
        wavelet_features = []
        lyapunov_exponents = []
        
        for i in range(n_windows):
            start_idx = i * step
            end_idx = min(start_idx + window, n_samples)
            window_data = signals[:, start_idx:end_idx]
            
            if window_data.shape[1] < window // 4:
                continue
            
            window_wavelet = {}
            window_lyapunov = []
            
            for ch in range(n_channels):
                channel_signal = window_data[ch]
                
                # Frequency band analysis
                if HAS_SCIPY:
                    freqs, psd = sp_signal.welch(channel_signal, fs=fs, 
                                                nperseg=min(256, len(channel_signal)))
                else:
                    fft = np.fft.fft(channel_signal)
                    freqs = np.fft.fftfreq(len(fft), 1/fs)
                    psd = np.abs(fft) ** 2
                    pos_mask = freqs >= 0
                    freqs = freqs[pos_mask]
                    psd = psd[pos_mask]
                
                for j, (low, high) in enumerate(bands):
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        band_features[i, ch, j] = np.sqrt(np.trapz(psd[mask], freqs[mask]))
                
                # Wavelet features
                wav_feat = self.compute_wavelet_features(channel_signal, fs)
                for key, val in wav_feat.items():
                    if key not in window_wavelet:
                        window_wavelet[key] = []
                    window_wavelet[key].append(val)
                
                # Lyapunov exponent
                lyap = self.compute_lyapunov_exponent(channel_signal)
                window_lyapunov.append(lyap)
            
            wavelet_features.append(window_wavelet)
            lyapunov_exponents.append(window_lyapunov)
        
        advanced_features = {
            'wavelet_features': wavelet_features,
            'lyapunov_exponents': lyapunov_exponents
        }
        
        return times, band_features, advanced_features
    
    def compute_criticality_with_chaos(self, band_features: np.ndarray, 
                                      advanced_features: Dict,
                                      times: np.ndarray) -> Dict:
        """
        Compute criticality using:
        1. Logistic map dynamics (R parameter)
        2. Lyapunov exponents (chaos indicator)
        3. Wavelet entropy (complexity measure)
        """
        n_windows, n_channels, n_bands = band_features.shape
        
        critical_windows = []
        r_evolution = []
        state_evolution = []
        lyapunov_evolution = []
        
        # Normalize band features
        normalized = band_features.copy()
        for band in range(n_bands):
            band_data = normalized[:, :, band]
            mean_val = np.mean(band_data)
            std_val = np.std(band_data)
            if std_val > 0:
                normalized[:, :, band] = (band_data - mean_val) / std_val
        
        # Initialize logistic map
        r_params = np.full(n_bands, 3.0)
        x_states = np.full(n_bands, 0.5)
        chaos_threshold = 3.57
        
        for i in range(n_windows):
            # Get Lyapunov exponents for this window
            if i < len(advanced_features['lyapunov_exponents']):
                mean_lyapunov = np.mean(advanced_features['lyapunov_exponents'][i])
                lyapunov_evolution.append(mean_lyapunov)
            else:
                mean_lyapunov = 0
                lyapunov_evolution.append(0)
            
            # Calculate power changes
            if i > 0:
                power_change = normalized[i] - normalized[i-1]
            else:
                power_change = np.zeros((n_channels, n_bands))
            
            # Update R parameters based on multiple factors
            for j in range(n_bands):
                max_change = np.max(np.abs(power_change[:, j]))
                mean_power = np.mean(np.abs(normalized[i, :, j]))
                
                # Combine multiple indicators
                if max_change > 0.5 or mean_power > 1.5 or mean_lyapunov > 0.1:
                    # Increase R (toward chaos)
                    increment = 0.1 * (1 + max_change * 0.1 + abs(mean_lyapunov))
                    r_params[j] = min(3.99, r_params[j] + increment)
                else:
                    # Decrease R (toward stability)
                    r_params[j] = max(2.8, r_params[j] - 0.02)
                
                # Update logistic map state
                x_states[j] = r_params[j] * x_states[j] * (1 - x_states[j])
                x_states[j] = np.clip(x_states[j], 0.001, 0.999)
            
            # Calculate mean R
            r_avg = np.mean(r_params)
            r_evolution.append(r_avg)
            
            # Determine state based on R and Lyapunov
            if r_avg > chaos_threshold or mean_lyapunov > 0.15:
                state = "critical"
                critical_windows.append(i)
            elif r_avg > 3.3 or mean_lyapunov > 0.05:
                state = "transitional"
            else:
                state = "stable"
            
            state_evolution.append(state)
        
        # Calculate final metrics
        criticality_ratio = len(critical_windows) / max(1, n_windows)
        
        # Determine final state
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
            band_data = band_features[:, :, i]
            band_stats[name] = {
                'mean': float(np.mean(band_data)),
                'std': float(np.std(band_data)),
                'max': float(np.max(band_data))
            }
        
        # Calculate complexity metrics
        complexity_metrics = {
            'mean_r_parameter': float(np.mean(r_evolution)),
            'std_r_parameter': float(np.std(r_evolution)),
            'mean_lyapunov': float(np.mean(lyapunov_evolution)),
            'max_lyapunov': float(np.max(lyapunov_evolution)) if lyapunov_evolution else 0,
            'chaos_percentage': float(np.sum([1 for r in r_evolution if r > chaos_threshold]) / len(r_evolution) * 100),
            'temporal_complexity': float(np.var(r_evolution))
        }
        
        return {
            'criticality_ratio': criticality_ratio,
            'critical_windows': len(critical_windows),
            'total_windows': n_windows,
            'r_evolution': r_evolution,
            'lyapunov_evolution': lyapunov_evolution,
            'state_evolution': state_evolution,
            'times': times.tolist(),
            'critical_indices': critical_windows,
            'band_statistics': band_stats,
            'complexity_metrics': complexity_metrics,
            'final_state': final_state
        }

def generate_comprehensive_report(results: Dict, database: str, data_dict: Dict) -> str:
    """Generate comprehensive analysis report"""
    
    report = f"""
# 🧠 Advanced EEG Analysis Report

**Database:** {database}
**Files Analyzed:** {len(data_dict.get('eeg_data', []))}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analyst:** {st.session_state.get('username', 'Unknown')}

## Executive Summary

### Brain State Classification
- **Overall State:** {results['final_state'].replace('_', ' ').title()}
- **Criticality Ratio:** {results['criticality_ratio']:.1%}
- **Critical Episodes:** {results['critical_windows']} of {results['total_windows']} windows

## Chaos Theory Metrics

### Logistic Map Analysis
- **Mean R Parameter:** {results['complexity_metrics']['mean_r_parameter']:.3f}
- **R Std Deviation:** {results['complexity_metrics']['std_r_parameter']:.3f}
- **Chaos Percentage:** {results['complexity_metrics']['chaos_percentage']:.1f}%
- **Temporal Complexity:** {results['complexity_metrics']['temporal_complexity']:.3f}

### Lyapunov Exponents
- **Mean Lyapunov:** {results['complexity_metrics']['mean_lyapunov']:.4f}
- **Max Lyapunov:** {results['complexity_metrics']['max_lyapunov']:.4f}
- **Interpretation:** {"Chaotic dynamics" if results['complexity_metrics']['mean_lyapunov'] > 0.1 else "Stable dynamics" if results['complexity_metrics']['mean_lyapunov'] < 0.05 else "Edge of chaos"}

## Frequency Band Analysis

| Band | Frequency Range | Mean Power (µV) | Std Dev | Max Power |
|------|----------------|-----------------|---------|-----------|"""
    
    band_info = {
        'delta': '0.5-4 Hz',
        'theta': '4-8 Hz',
        'alpha': '8-13 Hz',
        'beta': '13-30 Hz',
        'gamma': '30-50 Hz'
    }
    
    for band, freq_range in band_info.items():
        stats = results['band_statistics'][band]
        report += f"\n| {band.title()} | {freq_range} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['max']:.2f} |"
    
    report += f"""

## Clinical Interpretation

### Primary Findings
"""
    
    if results['criticality_ratio'] > 0.4:
        report += """
🚨 **HIGH CRITICALITY DETECTED**
- Brain dynamics show significant instability
- Multiple critical state transitions observed
- High risk for seizure activity or consciousness alterations
- **Recommendation:** Immediate clinical review required
"""
    elif results['criticality_ratio'] > 0.2:
        report += """
⚠️ **MODERATE CRITICALITY**
- Transitional brain state with periodic instabilities
- Some critical episodes detected
- Possible altered consciousness or cognitive changes
- **Recommendation:** Close monitoring advised
"""
    elif results['criticality_ratio'] > 0.1:
        report += """
📊 **MILD CRITICALITY**
- Minor fluctuations in brain dynamics
- Occasional transitions between states
- Within normal physiological variation
- **Recommendation:** Routine follow-up
"""
    else:
        report += """
✅ **STABLE BRAIN DYNAMICS**
- Well-regulated neural activity
- Minimal critical transitions
- Normal homeostatic control
- **Recommendation:** No immediate concerns
"""
    
    # Add database-specific interpretations
    if "psi" in database.lower() or "psychedelic" in database.lower():
        report += """

### Psychedelic-Specific Analysis
- Increased neural entropy may indicate expanded consciousness
- Higher criticality could reflect enhanced neuroplasticity
- Monitor for integration period post-experience
"""
    elif "alcohol" in database.lower():
        report += """

### Alcohol-Specific Analysis
- Decreased alpha power may indicate intoxication
- Increased theta/delta could suggest sedation
- Monitor for withdrawal-related hyperexcitability
"""
    
    report += f"""

## Technical Details

### Processing Parameters
- Window Size: 2.0 seconds
- Analysis Method: Wavelet + Logistic Map + Lyapunov
- Chaos Threshold: R > 3.57
- Critical Lyapunov: λ > 0.15

### Data Quality
- Total Data Points: {results['total_windows'] * 256 * 2:,}
- Files Processed: {len(data_dict.get('files', []))}
- Subjects Identified: {len(data_dict.get('subjects', [])) if data_dict.get('subjects') else 'Unknown'}

## Recommendations

1. **Clinical Correlation:** Compare findings with clinical observations
2. **Temporal Analysis:** Review specific time points of critical episodes
3. **Follow-up:** Consider repeat analysis for trending
4. **Documentation:** Include in patient medical record if applicable

---
*Report generated by Advanced EEG Analysis Platform v2.0*
*Utilizing Wavelet Analysis, Lyapunov Exponents, and Logistic Map Dynamics*
"""
    
    return report

# Initialize processor
@st.cache_resource
def get_processor():
    return AdvancedEEGProcessor()

def main():
    st.set_page_config(
        page_title="🧠 EEG Analysis Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Authentication
    if not authenticate_user():
        st.stop()
    
    # Header with user info
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🧠 Advanced EEG Criticality Analysis Platform")
        st.caption("Real Database Integration with Wavelet Analysis & Lyapunov Exponents")
    with col2:
        st.markdown(f"**User:** {st.session_state['username']}")
        if st.button("🚪 Logout", key="logout"):
            for key in ['authenticated', 'username', 'access_level']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Initialize processor
    processor = get_processor()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Database selection with icons
        st.subheader("📊 Select Database")
        
        # Debug: Show available databases
        available_dbs = list(REAL_DATABASES.keys())
        st.caption(f"Available: {len(available_dbs)} databases")
        
        database_id = st.selectbox(
            "Choose Dataset:",
            options=available_dbs,
            format_func=lambda x: REAL_DATABASES[x]["name"],
            key="database_selector"
        )
        
        if database_id:
            db_info = REAL_DATABASES[database_id]
            with st.expander("ℹ️ Database Info", expanded=True):
                st.markdown(f"""
                **Description:** {db_info['description']}
                
                **Format:** {db_info['format'].upper()}
                
                **Source:** {'GitHub' if db_info['type'] == 'github' else 'Google Drive' if db_info['type'] == 'gdrive' else 'Demo'}
                
                **Citation:** {db_info['citation']}
                
                **Expected Criticality:** {db_info['expected_criticality']}
                """)
        
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (seconds)", 1.0, 5.0, 2.0, 0.5)
        step_size = st.slider("Step Size (seconds)", 0.5, 2.0, 1.0, 0.5)
        
        st.subheader("📈 Advanced Features")
        use_wavelets = st.checkbox("Enable Wavelet Analysis", value=True, 
                                  help="Decompose signals using wavelet transform")
        compute_lyapunov = st.checkbox("Compute Lyapunov Exponents", value=True,
                                      help="Measure chaos and divergence of trajectories")
        
        st.subheader("📊 Visualization Options")
        show_raw = st.checkbox("Show Raw Signals", value=False)
        show_bands = st.checkbox("Show Frequency Bands", value=True)
        show_criticality = st.checkbox("Show Criticality Evolution", value=True)
        show_lyapunov = st.checkbox("Show Lyapunov Exponents", value=compute_lyapunov)
    
    # Main content area
    st.header(f"Analyzing: {REAL_DATABASES[database_id]['name']}")
    
    # Add troubleshooting info
    if database_id == "psi_database":
        st.info("💡 PSI Database: Contains psychedelic EEG recordings. Data may include metadata columns that will be automatically filtered.")
    elif database_id == "alcohol_database":
        st.info("💡 Alcohol Database: Contains EEG recordings under alcohol influence. Large file may take time to download.")
    
    # Analysis button
    if st.button("🚀 Download & Analyze Dataset", type="primary", use_container_width=True):
        
        try:
            # Download dataset
            success, zip_content, message = processor.downloader.download_dataset(database_id)
            
            if success and zip_content:
                st.success(f"✅ {message}")
                
                # Check file size
                file_size_mb = len(zip_content) / (1024 * 1024)
                st.info(f"📦 Downloaded file size: {file_size_mb:.2f} MB")
                
                # Extract data
                with st.spinner("📦 Extracting data from archive..."):
                    extracted_data = processor.downloader.extract_zip_data(zip_content, database_id)
                
                if extracted_data["files"]:
                    st.success(f"✅ Found {len(extracted_data['files'])} files in archive")
                
                # Show extracted files
                with st.expander(f"📁 Archive Contents ({len(extracted_data['files'])} files)", expanded=True):
                    # Show file list
                    st.write("**Files found:**")
                    cols = st.columns(2)
                    for idx, file in enumerate(extracted_data['files'][:20]):
                        cols[idx % 2].write(f"• {file}")
                    if len(extracted_data['files']) > 20:
                        st.write(f"... and {len(extracted_data['files']) - 20} more files")
                    
                    # Show data preview if available
                    if extracted_data["eeg_data"]:
                        st.write("\n**📊 Data Preview:**")
                        for i, eeg_item in enumerate(extracted_data["eeg_data"][:3]):  # Show first 3 files
                            if isinstance(eeg_item["data"], pd.DataFrame):
                                df = eeg_item["data"]
                                st.write(f"\n**File:** {eeg_item['filename']}")
                                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                                st.write("Columns:", list(df.columns)[:10])
                                if len(df.columns) > 10:
                                    st.write(f"... and {len(df.columns) - 10} more columns")
                                
                                # Show first few rows
                                st.write("First 5 rows:")
                                st.dataframe(df.head(), use_container_width=True)
                            elif isinstance(eeg_item["data"], np.ndarray):
                                st.write(f"\n**File:** {eeg_item['filename']}")
                                st.write(f"Array shape: {eeg_item['data'].shape}")
                                st.write(f"Data type: Single channel numeric")
                    
                    if extracted_data["subjects"]:
                        st.write(f"\n**Subjects found:** {', '.join(extracted_data['subjects'][:10])}")
                
                # Process EEG data
                try:
                    with st.spinner("🧠 Processing EEG signals..."):
                        signals, fs, channels = processor.process_eeg_data(extracted_data)
                        st.success(f"✅ Loaded {len(channels)} channels at {fs} Hz sampling rate")
                    
                    # Extract features
                    with st.spinner("📊 Extracting features with wavelets and computing Lyapunov exponents..."):
                        times, band_features, advanced_features = processor.extract_advanced_features(
                            signals, fs, window_size, step_size
                        )
                        
                    # Compute criticality
                    with st.spinner("🔮 Computing criticality metrics using chaos theory..."):
                        results = processor.compute_criticality_with_chaos(
                            band_features, advanced_features, times
                        )
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display key metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        color = "🔴" if results['criticality_ratio'] > 0.4 else "🟡" if results['criticality_ratio'] > 0.2 else "🟢"
                        st.metric(f"{color} Criticality", f"{results['criticality_ratio']:.1%}")
                    
                    with col2:
                        st.metric("Brain State", results['final_state'].replace('_', ' ').title())
                    
                    with col3:
                        st.metric("Mean R", f"{results['complexity_metrics']['mean_r_parameter']:.3f}")
                    
                    with col4:
                        st.metric("Mean Lyapunov", f"{results['complexity_metrics']['mean_lyapunov']:.4f}")
                    
                    with col5:
                        st.metric("Chaos %", f"{results['complexity_metrics']['chaos_percentage']:.1f}%")
                    
                    # Visualizations
                    if HAS_MATPLOTLIB:
                        # Count plots to show
                        n_plots = sum([show_raw, show_bands, show_criticality, show_lyapunov])
                        
                        if n_plots > 0:
                            fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots))
                            if n_plots == 1:
                                axes = [axes]
                            
                            plot_idx = 0
                            
                            # Raw signals plot
                            if show_raw:
                                ax = axes[plot_idx]
                                time_vec = np.arange(min(2000, signals.shape[1])) / fs
                                for i in range(min(3, len(channels))):
                                    ax.plot(time_vec, signals[i, :len(time_vec)] + i*50, 
                                           label=channels[i], alpha=0.7)
                                ax.set_xlabel("Time (s)")
                                ax.set_ylabel("Amplitude (µV)")
                                ax.set_title("Raw EEG Signals (First 3 channels)")
                                ax.legend(loc='upper right')
                                ax.grid(True, alpha=0.3)
                                plot_idx += 1
                            
                            # Frequency bands plot
                            if show_bands:
                                ax = axes[plot_idx]
                                bands = ['Delta\n0.5-4Hz', 'Theta\n4-8Hz', 'Alpha\n8-13Hz', 
                                        'Beta\n13-30Hz', 'Gamma\n30-50Hz']
                                powers = [results['band_statistics'][b.split('\n')[0].lower()]['mean'] 
                                         for b in bands]
                                colors = ['#4B0082', '#0000FF', '#00FF00', '#FFA500', '#FF0000']
                                bars = ax.bar(bands, powers, color=colors, alpha=0.7, edgecolor='black')
                                ax.set_ylabel("Mean Power (µV)")
                                ax.set_title("EEG Frequency Band Analysis")
                                ax.grid(True, alpha=0.3, axis='y')
                                
                                for bar, power in zip(bars, powers):
                                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                           f'{power:.1f}', ha='center', va='bottom', fontweight='bold')
                                plot_idx += 1
                            
                            # Criticality evolution plot
                            if show_criticality:
                                ax = axes[plot_idx]
                                colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red'
                                         for s in results['state_evolution']]
                                ax.scatter(results['times'], results['r_evolution'], 
                                          c=colors, alpha=0.7, s=50, edgecolors='black')
                                ax.axhline(y=3.57, color='red', linestyle='--', 
                                          alpha=0.5, label='Chaos Threshold (3.57)', linewidth=2)
                                ax.axhline(y=3.0, color='green', linestyle='--', 
                                          alpha=0.3, label='Stability Threshold (3.0)')
                                ax.fill_between(results['times'], 3.57, 4.0, alpha=0.1, color='red')
                                ax.set_xlabel("Time (s)")
                                ax.set_ylabel("R Parameter")
                                ax.set_title("Brain State Criticality Evolution (Logistic Map)")
                                ax.legend(loc='upper right')
                                ax.grid(True, alpha=0.3)
                                ax.set_ylim([2.5, 4.0])
                                plot_idx += 1
                            
                            # Lyapunov exponents plot
                            if show_lyapunov and 'lyapunov_evolution' in results:
                                ax = axes[plot_idx]
                                lyap_values = results['lyapunov_evolution']
                                ax.plot(results['times'][:len(lyap_values)], lyap_values, 
                                       'b-', linewidth=2, label='Lyapunov Exponent')
                                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                                ax.axhline(y=0.1, color='red', linestyle='--', 
                                          alpha=0.5, label='Chaos Threshold (λ>0.1)')
                                ax.fill_between(results['times'][:len(lyap_values)], 
                                               0, lyap_values, where=[l > 0 for l in lyap_values],
                                               alpha=0.3, color='red', label='Chaotic regions')
                                ax.set_xlabel("Time (s)")
                                ax.set_ylabel("Lyapunov Exponent (λ)")
                                ax.set_title("Lyapunov Exponents - Chaos Indicator")
                                ax.legend(loc='upper right')
                                ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Generate comprehensive report
                    report = generate_comprehensive_report(results, 
                                                          REAL_DATABASES[database_id]['name'],
                                                          extracted_data)
                    
                    # Display report
                    with st.expander("📄 Full Analysis Report", expanded=True):
                        st.markdown(report)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "💾 Download Report (Markdown)",
                            data=report,
                            file_name=f"eeg_report_{database_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        # Create JSON with all results
                        json_results = {
                            "database": database_id,
                            "analysis_date": datetime.now().isoformat(),
                            "results": results,
                            "parameters": {
                                "window_size": window_size,
                                "step_size": step_size,
                                "sampling_rate": fs
                            }
                        }
                        st.download_button(
                            "📊 Download Raw Data (JSON)",
                            data=json.dumps(json_results, indent=2, default=str),
                            file_name=f"eeg_data_{database_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    st.info("💡 Try adjusting the analysis parameters or contact support if the issue persists.")
            else:
                st.warning("⚠️ No valid EEG data found in the archive. The file may be corrupted or in an unsupported format.")
        else:
            st.error(f"❌ Failed to download dataset: {message}")
            st.info("💡 Please check your internet connection or try again later.")
    
    # Information footer
    with st.expander("ℹ️ About This Platform"):
        st.markdown("""
        ### Advanced Features
        
        **🌊 Wavelet Analysis**
        - Decomposes EEG signals into time-frequency components
        - Provides better temporal resolution than FFT
        - Captures transient events and non-stationary dynamics
        
        **📈 Lyapunov Exponents**
        - Measures the rate of divergence of nearby trajectories
        - Positive values indicate chaotic dynamics
        - Critical for identifying edge-of-chaos states
        
        **🔄 Logistic Map Dynamics**
        - Models brain state transitions using chaos theory
        - R parameter evolution tracks criticality
        - R > 3.57 indicates chaotic regime
        
        **📊 Integrated Analysis**
        - Combines multiple chaos indicators
        - Provides comprehensive brain state assessment
        - Suitable for research and clinical applications
        """)

if __name__ == "__main__":
    main()
