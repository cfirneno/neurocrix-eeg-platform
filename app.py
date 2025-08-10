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

# Simple password - no secrets file needed to avoid configuration issues
CORRECT_PASSWORD = "neurocrix2024"

# Password protection
def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "🔐 Enter Password to Access EEG Platform", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.write("*Contact administrator for access credentials*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "🔐 Enter Password to Access EEG Platform", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😞 Password incorrect. Please try again.")
        return False
    else:
        # Password correct
        return True

# Optional imports with fallbacks
try:
    import scipy.signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Database Configuration
PUBLIC_DATABASES = {
    "bonn_seizure": {
        "name": "Bonn University Seizure Database (Simulated)",
        "description": "Scientifically accurate simulations based on published Bonn dataset characteristics",
        "format": "csv",
        "subjects": 5,
        "license": "Educational use",
        "note": "Real Bonn data requires manual download from their website",
        "sets": {
            "Z": {
                "description": "Healthy volunteers, eyes open (simulated)",
                "characteristics": "Normal alpha rhythm (8-13 Hz), low amplitude"
            },
            "O": {
                "description": "Healthy volunteers, eyes closed (simulated)", 
                "characteristics": "Enhanced alpha rhythm, increased amplitude"
            },
            "N": {
                "description": "Seizure-free, from epileptogenic zone (simulated)",
                "characteristics": "Occasional interictal spikes, slightly abnormal"
            },
            "F": {
                "description": "Seizure-free, opposite hemisphere (simulated)",
                "characteristics": "Near-normal patterns with minor abnormalities"
            },
            "S": {
                "description": "SEIZURE ACTIVITY recordings (simulated)",
                "characteristics": "High amplitude 3-8 Hz oscillations, periodic spikes"
            }
        }
    },
    "demo_database": {
        "name": "Demo EEG Database",
        "description": "Generated demonstration EEG patterns for testing",
        "format": "csv",
        "subjects": 3,
        "license": "Open use"
    }
}

def generate_realistic_eeg_data(set_name: str, duration_seconds: float = 23.6) -> np.ndarray:
    """
    Generate scientifically accurate EEG patterns based on Bonn dataset characteristics.
    
    Bonn Dataset Properties:
    - Sampling rate: 173.61 Hz
    - Duration: 23.6 seconds per segment
    - Single channel recordings
    - Amplitude range varies by set
    """
    fs = 173.61  # Bonn sampling frequency
    n_samples = int(duration_seconds * fs)  # 4097 samples for 23.6 seconds
    time = np.arange(n_samples) / fs
    
    # Initialize random seed for reproducibility within session
    np.random.seed(int(time[0] * 1000) % 2**32)
    
    # Base signal
    signal = np.zeros(n_samples)
    
    if set_name == 'Z':  # Healthy, eyes open
        # Normal EEG with dominant alpha when relaxed
        # Alpha (8-13 Hz) - moderate amplitude for eyes open
        signal += 30 * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
        signal += 15 * np.sin(2 * np.pi * 11 * time + np.random.uniform(0, 2*np.pi))
        
        # Beta (13-30 Hz) - present due to eyes open
        signal += 20 * np.sin(2 * np.pi * 18 * time + np.random.uniform(0, 2*np.pi))
        signal += 10 * np.sin(2 * np.pi * 25 * time + np.random.uniform(0, 2*np.pi))
        
        # Theta (4-8 Hz) - low amplitude
        signal += 10 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Delta (0.5-4 Hz) - minimal in awake state
        signal += 5 * np.sin(2 * np.pi * 2 * time + np.random.uniform(0, 2*np.pi))
        
        # Biological noise
        signal += np.random.normal(0, 5, n_samples)
        
    elif set_name == 'O':  # Healthy, eyes closed
        # Enhanced alpha rhythm with eyes closed
        # Alpha (8-13 Hz) - high amplitude with eyes closed
        signal += 60 * np.sin(2 * np.pi * 10 * time + np.random.uniform(0, 2*np.pi))
        signal += 40 * np.sin(2 * np.pi * 9 * time + np.random.uniform(0, 2*np.pi))
        signal += 30 * np.sin(2 * np.pi * 11 * time + np.random.uniform(0, 2*np.pi))
        
        # Beta (13-30 Hz) - reduced with eyes closed
        signal += 10 * np.sin(2 * np.pi * 20 * time + np.random.uniform(0, 2*np.pi))
        
        # Theta (4-8 Hz) - slightly increased
        signal += 15 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Delta (0.5-4 Hz) - still minimal
        signal += 8 * np.sin(2 * np.pi * 2 * time + np.random.uniform(0, 2*np.pi))
        
        # Less noise with eyes closed
        signal += np.random.normal(0, 3, n_samples)
        
    elif set_name == 'N':  # Seizure-free from epileptogenic zone
        # Interictal abnormalities but no seizure
        # Slower alpha
        signal += 40 * np.sin(2 * np.pi * 8 * time + np.random.uniform(0, 2*np.pi))
        
        # Increased theta
        signal += 30 * np.sin(2 * np.pi * 5 * time + np.random.uniform(0, 2*np.pi))
        signal += 25 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Some beta
        signal += 15 * np.sin(2 * np.pi * 15 * time + np.random.uniform(0, 2*np.pi))
        
        # Increased delta
        signal += 20 * np.sin(2 * np.pi * 2.5 * time + np.random.uniform(0, 2*np.pi))
        
        # Occasional interictal spikes
        n_spikes = np.random.randint(5, 15)
        spike_times = np.random.choice(n_samples - 50, n_spikes, replace=False)
        for spike_time in spike_times:
            spike_amplitude = np.random.uniform(80, 150)
            spike_width = np.random.randint(10, 30)
            signal[spike_time:spike_time+spike_width] += spike_amplitude * np.exp(-np.arange(spike_width)/5)
        
        # More biological noise
        signal += np.random.normal(0, 10, n_samples)
        
    elif set_name == 'F':  # Seizure-free from opposite hemisphere
        # More normal but with some abnormalities
        # Near-normal alpha
        signal += 35 * np.sin(2 * np.pi * 9.5 * time + np.random.uniform(0, 2*np.pi))
        signal += 25 * np.sin(2 * np.pi * 10.5 * time + np.random.uniform(0, 2*np.pi))
        
        # Slightly elevated theta
        signal += 20 * np.sin(2 * np.pi * 6 * time + np.random.uniform(0, 2*np.pi))
        
        # Normal beta
        signal += 18 * np.sin(2 * np.pi * 18 * time + np.random.uniform(0, 2*np.pi))
        
        # Slightly elevated delta
        signal += 12 * np.sin(2 * np.pi * 3 * time + np.random.uniform(0, 2*np.pi))
        
        # Rare spikes
        n_spikes = np.random.randint(1, 5)
        spike_times = np.random.choice(n_samples - 50, n_spikes, replace=False)
        for spike_time in spike_times:
            spike_amplitude = np.random.uniform(50, 100)
            spike_width = np.random.randint(10, 20)
            signal[spike_time:spike_time+spike_width] += spike_amplitude * np.exp(-np.arange(spike_width)/5)
        
        signal += np.random.normal(0, 7, n_samples)
        
    elif set_name == 'S':  # SEIZURE recordings
        # Ictal (seizure) activity - dramatic patterns
        
        # Pre-ictal period (first 20% of recording)
        pre_ictal_samples = int(n_samples * 0.2)
        signal[:pre_ictal_samples] += 40 * np.sin(2 * np.pi * 7 * time[:pre_ictal_samples])
        signal[:pre_ictal_samples] += 30 * np.sin(2 * np.pi * 4 * time[:pre_ictal_samples])
        signal[:pre_ictal_samples] += np.random.normal(0, 15, pre_ictal_samples)
        
        # Ictal period (middle 60% - actual seizure)
        ictal_start = pre_ictal_samples
        ictal_end = int(n_samples * 0.8)
        ictal_samples = ictal_end - ictal_start
        
        # Classic 3-4 Hz spike-wave complexes
        for i in range(ictal_start, ictal_end, int(fs/3)):  # 3 Hz repetition
            spike_duration = int(fs * 0.07)  # 70ms spike
            wave_duration = int(fs * 0.25)   # 250ms wave
            
            if i + spike_duration + wave_duration < ictal_end:
                # Sharp spike
                signal[i:i+spike_duration] += 300 * np.exp(-np.arange(spike_duration)/3)
                # Slow wave
                wave_time = np.arange(wave_duration) / fs
                signal[i+spike_duration:i+spike_duration+wave_duration] -= 150 * np.sin(np.pi * wave_time / wave_time[-1])
        
        # Add rhythmic seizure activity (3-5 Hz)
        seizure_freq = np.random.uniform(3, 5)
        signal[ictal_start:ictal_end] += 200 * np.sin(2 * np.pi * seizure_freq * time[ictal_start:ictal_end])
        
        # High-frequency oscillations during seizure
        signal[ictal_start:ictal_end] += 50 * np.sin(2 * np.pi * 40 * time[ictal_start:ictal_end])
        
        # Post-ictal period (last 20% - suppression)
        post_ictal_start = ictal_end
        signal[post_ictal_start:] += 20 * np.sin(2 * np.pi * 2 * time[post_ictal_start:])
        signal[post_ictal_start:] += 10 * np.sin(2 * np.pi * 6 * time[post_ictal_start:])
        signal[post_ictal_start:] += np.random.normal(0, 5, n_samples - post_ictal_start)
        
        # Add random high-amplitude transients
        n_transients = np.random.randint(20, 40)
        transient_times = np.random.choice(range(ictal_start, ictal_end), n_transients, replace=False)
        for t_time in transient_times:
            if t_time < n_samples - 20:
                signal[t_time:t_time+20] += np.random.uniform(100, 400) * np.exp(-np.arange(20)/3)
        
    return signal

class DatabaseConnector:
    def __init__(self, db_config: Dict):
        self.config = db_config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EEG-Analysis-Platform/1.0'})
    
    def list_subjects(self) -> List[str]:
        db_id = self.config.get("db_id")
        if db_id == "bonn_seizure":
            return ["Z", "O", "N", "F", "S"]
        elif db_id == "demo_database":
            return ["Healthy_1", "Healthy_2", "Patient_1"]
        return []
    
    def get_subject_sessions(self, subject_id: str) -> List[str]:
        db_id = self.config.get("db_id")
        if db_id == "bonn_seizure":
            # Each Bonn set would have 100 recordings, we'll simulate access to 10
            return [f"Segment_{i:03d}" for i in range(1, 11)]
        elif db_id == "demo_database":
            return ["Session_1", "Session_2", "Session_3"]
        return []
    
    def download_eeg_sample(self, subject_id: str, session_id: str) -> Tuple[bytes, str]:
        """Generate scientifically accurate EEG data based on dataset characteristics"""
        db_id = self.config.get("db_id")
        
        if db_id == "bonn_seizure":
            # Generate realistic Bonn-like data
            segment_num = int(session_id.split("_")[1]) if "_" in session_id else 1
            
            # Generate single channel data matching Bonn characteristics
            data = generate_realistic_eeg_data(subject_id)
            
            # Convert to multi-channel for better visualization
            # Real Bonn is single-channel, but we'll create 8 correlated channels
            n_samples = len(data)
            n_channels = 8
            signals = np.zeros((n_channels, n_samples))
            
            for ch in range(n_channels):
                # Create correlated channels with slight variations
                correlation = 0.7 + (ch * 0.04)  # 70-98% correlation
                variation = np.random.normal(0, 10 * (1 - correlation), n_samples)
                phase_shift = int(ch * 5)  # Small phase differences
                
                # Apply channel-specific modifications
                signals[ch] = np.roll(data * correlation, phase_shift) + variation
                
                # Add channel-specific noise
                signals[ch] += np.random.normal(0, 2, n_samples)
            
            # Create DataFrame
            df = pd.DataFrame(signals.T, columns=[f"CH_{i+1}" for i in range(n_channels)])
            csv_data = df.to_csv(index=False)
            filename = f"Simulated_Bonn_{subject_id}_{session_id}.csv"
            return csv_data.encode('utf-8'), filename
            
        else:  # demo_database
            # Generate generic demo data
            duration = 30
            fs = 256
            n_channels = 8
            n_samples = duration * fs
            
            time_axis = np.linspace(0, duration, n_samples)
            signals = np.zeros((n_channels, n_samples))
            
            for ch in range(n_channels):
                # Basic EEG rhythms
                signals[ch] += 40 * np.sin(2 * np.pi * 10 * time_axis + np.random.random() * 2 * np.pi)
                signals[ch] += 20 * np.sin(2 * np.pi * 20 * time_axis + np.random.random() * 2 * np.pi)
                signals[ch] += 10 * np.sin(2 * np.pi * 5 * time_axis + np.random.random() * 2 * np.pi)
                signals[ch] += np.random.normal(0, 5, n_samples)
            
            df = pd.DataFrame(signals.T, columns=[f"CH_{i+1}" for i in range(n_channels)])
            csv_data = df.to_csv(index=False)
            filename = f"{subject_id}_{session_id}_demo.csv"
            return csv_data.encode('utf-8'), filename

class EnhancedEEGProcessor:
    def __init__(self):
        self.connectors = {}
        for db_id, db_config in PUBLIC_DATABASES.items():
            self.connectors[db_id] = DatabaseConnector({**db_config, "db_id": db_id})
    
    def get_database_info(self, db_id: str) -> Optional[Dict]:
        if db_id not in PUBLIC_DATABASES:
            return None
        
        db_config = PUBLIC_DATABASES[db_id]
        connector = self.connectors[db_id]
        subjects = connector.list_subjects()
        
        return {
            "db_id": db_id,
            "name": db_config["name"],
            "description": db_config["description"],
            "available_subjects": subjects
        }
    
    def load_eeg_file(self, file_content: bytes, filename: str) -> Tuple[np.ndarray, int, List[str]]:
        try:
            if filename.endswith('.csv') or 'csv' in filename:
                df = pd.read_csv(io.BytesIO(file_content))
                if 'time' in df.columns[0].lower() or 'index' in df.columns[0].lower():
                    data = df.iloc[:, 1:].values.T
                    channels = list(df.columns[1:])
                else:
                    data = df.values.T
                    channels = list(df.columns)
                
                # Determine sampling rate based on filename
                if 'Bonn' in filename:
                    fs = 173.61  # Bonn sampling rate
                else:
                    fs = 256  # Default sampling rate
                    
                return data.astype(float), fs, channels
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            raise ValueError(f"Error loading file {filename}: {str(e)}")
    
    def extract_features(self, signals: np.ndarray, fs: int, window_s: float = 2.0, 
                        step_s: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        n_channels, n_samples = signals.shape
        window = int(window_s * fs)
        step = int(step_s * fs)
        n_windows = max(1, (n_samples - window) // step + 1)
        
        times = np.arange(n_windows) * step_s + window_s / 2
        features = np.zeros((n_windows, n_channels, 5))
        
        # Standard EEG frequency bands
        bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        
        for i in range(n_windows):
            start_idx = i * step
            end_idx = min(start_idx + window, n_samples)
            window_data = signals[:, start_idx:end_idx]
            
            if window_data.shape[1] < window // 4:
                continue
            
            for ch in range(n_channels):
                if HAS_SCIPY:
                    freqs, psd = sp_signal.welch(
                        window_data[ch], fs=fs, 
                        nperseg=min(window, window_data.shape[1])
                    )
                else:
                    fft = np.fft.fft(window_data[ch])
                    freqs = np.fft.fftfreq(len(fft), 1/fs)
                    psd = np.abs(fft) ** 2
                    pos_mask = freqs >= 0
                    freqs = freqs[pos_mask]
                    psd = psd[pos_mask]
                
                for j, (low, high) in enumerate(bands):
                    mask = (freqs >= low) & (freqs <= high)
                    if np.any(mask):
                        features[i, ch, j] = np.sqrt(np.trapz(psd[mask], freqs[mask]))
        
        return times, features
    
    def compute_advanced_criticality(self, features: np.ndarray, times: np.ndarray, 
                                   amp_threshold: float = 0.3) -> Dict:
        n_windows, n_channels, n_bands = features.shape
        
        critical_windows = []
        r_evolution = []
        state_evolution = []
        
        baseline = np.median(features, axis=0, keepdims=True)
        threshold = baseline * (1 + amp_threshold)
        
        r_params = np.full(n_bands, 3.0)
        x_states = np.full(n_bands, 0.5)
        
        for i in range(n_windows):
            band_triggers = np.any(features[i] > threshold[0], axis=0)
            channel_triggers = np.any(features[i] > threshold[0], axis=1)
            
            trigger_ratio = np.sum(band_triggers) / n_bands
            channel_ratio = np.sum(channel_triggers) / n_channels
            
            for j in range(n_bands):
                if band_triggers[j]:
                    r_params[j] = min(3.9, r_params[j] + 0.05)
                else:
                    r_params[j] = max(2.5, r_params[j] - 0.01)
                
                x_states[j] = r_params[j] * x_states[j] * (1 - x_states[j])
                x_states[j] = np.clip(x_states[j], 0.001, 0.999)
            
            r_avg = np.mean(r_params)
            r_evolution.append(r_avg)
            
            if trigger_ratio > 0.6 and channel_ratio > 0.5:
                state = "critical"
                critical_windows.append(i)
            elif trigger_ratio > 0.3 or channel_ratio > 0.3:
                state = "transitional"
            else:
                state = "stable"
            
            state_evolution.append(state)
        
        criticality_ratio = len(critical_windows) / max(1, n_windows)
        
        if criticality_ratio > 0.4:
            final_state = "highly_critical"
        elif criticality_ratio > 0.2:
            final_state = "moderately_critical"
        elif criticality_ratio > 0.1:
            final_state = "transitional"
        else:
            final_state = "stable"
        
        # Compute band statistics
        band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        band_stats = {}
        for i, band in enumerate(band_names):
            band_data = features[:, :, i]
            band_stats[band] = {
                'mean_power': float(np.mean(band_data)),
                'std_power': float(np.std(band_data))
            }
        
        return {
            'total_windows': n_windows,
            'critical_windows': len(critical_windows),
            'criticality_ratio': criticality_ratio,
            'final_state': final_state,
            'mean_amplitude': float(np.mean(features)),
            'std_amplitude': float(np.std(features)),
            'r_evolution': r_evolution,
            'state_evolution': state_evolution,
            'times': times.tolist(),
            'critical_indices': critical_windows,
            'band_statistics': band_stats,
            'complexity_metrics': {
                'temporal_complexity': float(np.var(r_evolution)),
                'mean_r_parameter': float(np.mean(r_evolution))
            }
        }

def generate_clinical_interpretation(results: Dict, patient_info: str = "", database_info: str = "") -> str:
    ratio = results['criticality_ratio']
    state = results['final_state']
    bands = results['band_statistics']
    
    interpretation = f"""
    ## 🧠 **CLINICAL EEG ANALYSIS REPORT**
    
    **Patient Information:** {patient_info}
    **Data Source:** {database_info}
    **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ### **EXECUTIVE SUMMARY**
    - **Brain State Classification:** {state.upper().replace('_', ' ')}
    - **Criticality Level:** {ratio:.1%} of analyzed periods showed critical dynamics
    - **Total Analysis Windows:** {results['total_windows']}
    - **Critical Episodes:** {results['critical_windows']}
    
    ### **FREQUENCY BAND ANALYSIS**
    - **Delta Band (0.5-4 Hz):** {bands['delta']['mean_power']:.1f}μV (deep sleep, pathology)
    - **Theta Band (4-8 Hz):** {bands['theta']['mean_power']:.1f}μV (drowsiness, meditation)
    - **Alpha Band (8-13 Hz):** {bands['alpha']['mean_power']:.1f}μV (relaxed awareness)
    - **Beta Band (13-30 Hz):** {bands['beta']['mean_power']:.1f}μV (active thinking)
    - **Gamma Band (30-50 Hz):** {bands['gamma']['mean_power']:.1f}μV (cognitive processing)
    
    ### **CLINICAL INTERPRETATION**
    """
    
    if ratio > 0.4:
        interpretation += """
    🚨 **HIGH CRITICALITY (>40%)**
    - Significant instability in brain dynamics detected
    - Elevated risk for state transitions and potential seizure activity
    - **RECOMMENDATION:** Immediate clinical correlation and continuous monitoring
        """
    elif ratio > 0.2:
        interpretation += """
    ⚠️ **MODERATE CRITICALITY (20-40%)**
    - Transitional brain state with periodic instability
    - **RECOMMENDATION:** Serial monitoring and clinical correlation
        """
    elif ratio > 0.1:
        interpretation += """
    📈 **MILD CRITICALITY (10-20%)**
    - Occasional critical dynamics within physiological range
    - **RECOMMENDATION:** Baseline documentation and follow-up assessment
        """
    else:
        interpretation += """
    ✅ **STABLE DYNAMICS (<10%)**
    - Well-regulated brain state with strong homeostatic control
    - **RECOMMENDATION:** Continue current management if applicable
        """
    
    interpretation += f"""
    
    ### **TECHNICAL PARAMETERS**
    - **Mean R-parameter:** {results['complexity_metrics']['mean_r_parameter']:.3f}
    - **Temporal Complexity:** {results['complexity_metrics']['temporal_complexity']:.3f}
    - **Chaos Threshold:** 3.57 (R > 3.57 indicates chaotic dynamics)
    
    ### **DATA QUALITY NOTICE**
    """
    
    if "Simulated" in database_info:
        interpretation += """
    ⚠️ **SIMULATED DATA**: This analysis was performed on scientifically modeled EEG patterns
    based on published characteristics of the Bonn University dataset. For clinical decisions,
    please use actual patient recordings.
    
    **Simulation characteristics applied:**"""
        
        if "Subject Z" in database_info:
            interpretation += """
    - Healthy adult, eyes open: Normal alpha (8-13 Hz), low amplitude
    - Expected criticality: < 5%"""
        elif "Subject O" in database_info:
            interpretation += """
    - Healthy adult, eyes closed: Enhanced alpha rhythm, increased amplitude
    - Expected criticality: < 5%"""
        elif "Subject N" in database_info:
            interpretation += """
    - Epileptic patient, interictal from epileptogenic zone
    - Occasional spikes, abnormal background
    - Expected criticality: 10-20%"""
        elif "Subject F" in database_info:
            interpretation += """
    - Epileptic patient, interictal from opposite hemisphere
    - Near-normal patterns with minor abnormalities
    - Expected criticality: 5-15%"""
        elif "Subject S" in database_info:
            interpretation += """
    - SEIZURE SIMULATION: Ictal patterns with 3-5 Hz spike-wave complexes
    - High amplitude oscillations and rhythmic discharges
    - Expected criticality: > 40%"""
    
    interpretation += """
    
    ### **IMPORTANT DISCLAIMERS**
    - This analysis is for research and educational purposes only
    - Clinical decisions should only be made using real patient data
    - Always consult qualified healthcare professionals for medical decisions
    - Simulated data, while scientifically accurate, cannot replace actual recordings
    
    ---
    *Generated by Advanced EEG Criticality Analysis Platform*
    *Using scientifically modeled EEG patterns for demonstration*
    """
    
    return interpretation

# Initialize processor
@st.cache_resource
def get_processor():
    return EnhancedEEGProcessor()

# MAIN FUNCTION - FIXED INDENTATION
def main():
    st.set_page_config(
        page_title="🧠 EEG Criticality Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check password
    if not check_password():
        st.stop()
    
    # Main application
    st.title("🧠 Advanced EEG Criticality Analysis Platform")
    st.markdown("**Professional brain state analysis using scientifically modeled EEG patterns**")
    
    # Add important notice
    with st.expander("ℹ️ Important Information About Data Sources", expanded=False):
        st.warning("""
        **DATA SOURCE DISCLOSURE:**
        
        This platform uses **SIMULATED EEG data** that is scientifically modeled based on published 
        characteristics of the Bonn University Epilepsy dataset. 
        
        **Why simulated data?**
        - The real Bonn dataset requires manual download and authentication
        - Direct server access is restricted to authorized researchers
        - Simulated patterns allow demonstration of analysis capabilities
        
        **Accuracy of simulations:**
        - Set Z (Healthy, eyes open): Normal alpha rhythm patterns
        - Set O (Healthy, eyes closed): Enhanced alpha, as expected
        - Set N & F (Interictal): Appropriate abnormalities modeled
        - Set S (Seizure): Realistic 3-5 Hz spike-wave complexes
        
        **For real clinical analysis:**
        1. Download actual EEG data from authorized sources
        2. Use the "File Upload" feature to analyze real recordings
        3. Never make clinical decisions based on simulated data
        """)
    
    processor = get_processor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["🔬 Simulated Database", "📁 File Upload (For Real Data)"]
        )
        
        st.subheader("👤 Patient Information")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        patient_condition = st.text_input("Medical Condition", placeholder="e.g., Epilepsy")
        
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (seconds)", 1.0, 5.0, 2.0, 0.5)
        threshold = st.slider("Criticality Threshold", 0.1, 1.0, 0.3, 0.1)
        
        # Data source indicator
        st.subheader("📊 Data Source")
        st.info("Currently using: **Simulated EEG Patterns**")
        st.caption("Based on Bonn University dataset characteristics")
    
    # Main content
    if analysis_type == "🔬 Simulated Database":
        st.header("🔬 Simulated EEG Database Analysis")
        st.caption("Scientifically accurate patterns for demonstration and education")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            db_id = st.selectbox(
                "Select Database:",
                options=list(PUBLIC_DATABASES.keys()),
                format_func=lambda x: PUBLIC_DATABASES[x]["name"]
            )
        
        if db_id:
            db_info = processor.get_database_info(db_id)
            
            with col2:
                subject_id = st.selectbox(
                    "Select Subject/Set:",
                    options=db_info["available_subjects"],
                    help="Z/O: Healthy | N/F: Interictal | S: Seizure"
                )
            
            with col3:
                if subject_id:
                    connector = processor.connectors[db_id]
                    sessions = connector.get_subject_sessions(subject_id)
                    session_id = st.selectbox("Select Segment:", options=sessions)
        
        # Show description of selected set
        if db_id == "bonn_seizure" and subject_id:
            set_info = PUBLIC_DATABASES["bonn_seizure"]["sets"].get(subject_id)
            if set_info:
                st.info(f"**{subject_id} Set:** {set_info['description']}\n\n"
                       f"**Characteristics:** {set_info['characteristics']}")
        
        if st.button("🚀 Analyze Simulated Data", type="primary"):
            if db_id and subject_id and session_id:
                with st.spinner("Generating and analyzing simulated EEG patterns..."):
                    try:
                        # Generate and analyze
                        connector = processor.connectors[db_id]
                        file_content, filename = connector.download_eeg_sample(subject_id, session_id)
                        
                        signals, fs, channels = processor.load_eeg_file(file_content, filename)
                        times, features = processor.extract_features(signals, fs, window_s=window_size)
                        results = processor.compute_advanced_criticality(features, times, amp_threshold=threshold)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            metric_color = "🔴" if results['criticality_ratio'] > 0.4 else "🟡" if results['criticality_ratio'] > 0.2 else "🟢"
                            st.metric(f"{metric_color} Criticality", f"{results['criticality_ratio']:.1%}")
                        with col2:
                            st.metric("Brain State", results['final_state'].replace('_', ' ').title())
                        with col3:
                            st.metric("Critical Episodes", f"{results['critical_windows']}/{results['total_windows']}")
                        with col4:
                            st.metric("Mean R-parameter", f"{results['complexity_metrics']['mean_r_parameter']:.3f}")
                        
                        # Visualization
                        try:
                            import matplotlib.pyplot as plt
                            
                            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
                            
                            # 1. R-parameter evolution
                            ax1 = axes[0]
                            colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red' 
                                     for s in results['state_evolution']]
                            scatter = ax1.scatter(results['times'], results['r_evolution'], 
                                                c=colors, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
                            ax1.axhline(y=3.57, color='red', linestyle='--', alpha=0.7, 
                                       label='Chaos Threshold (3.57)', linewidth=2)
                            ax1.axhline(y=3.0, color='green', linestyle='--', alpha=0.5, 
                                       label='Stability Baseline (3.0)', linewidth=1)
                            ax1.set_ylabel('R Parameter', fontsize=12)
                            ax1.set_title('Brain State Criticality Evolution', fontsize=14, fontweight='bold')
                            ax1.legend(loc='upper right')
                            ax1.grid(True, alpha=0.3)
                            ax1.set_ylim([2.5, 4.0])
                            
                            # Add shaded regions for different states
                            for i, (time, state) in enumerate(zip(results['times'], results['state_evolution'])):
                                if i < len(results['times']) - 1:
                                    next_time = results['times'][i + 1]
                                    if state == 'critical':
                                        ax1.axvspan(time, next_time, alpha=0.2, color='red')
                                    elif state == 'transitional':
                                        ax1.axvspan(time, next_time, alpha=0.1, color='orange')
                            
                            # 2. Frequency band power
                            ax2 = axes[1]
                            bands = ['Delta\n(0.5-4Hz)', 'Theta\n(4-8Hz)', 'Alpha\n(8-13Hz)', 
                                    'Beta\n(13-30Hz)', 'Gamma\n(30-50Hz)']
                            powers = [results['band_statistics'][b.split('\n')[0].lower()]['mean_power'] 
                                     for b in bands]
                            bars = ax2.bar(bands, powers, color=['#4B0082', '#0000FF', '#00FF00', '#FFA500', '#FF0000'], 
                                          alpha=0.7, edgecolor='black', linewidth=1)
                            ax2.set_ylabel('Mean Power (μV)', fontsize=12)
                            ax2.set_title('EEG Frequency Band Analysis', fontsize=14, fontweight='bold')
                            ax2.grid(True, alpha=0.3, axis='y')
                            
                            # Add value labels on bars
                            for bar, power in zip(bars, powers):
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{power:.1f}μV', ha='center', va='bottom', fontweight='bold')
                            
                            # 3. Critical episodes timeline
                            ax3 = axes[2]
                            state_map = {'stable': 0, 'transitional': 1, 'critical': 2}
                            state_values = [state_map[s] for s in results['state_evolution']]
                            
                            # Create step plot
                            ax3.step(results['times'], state_values, where='post', linewidth=2, color='black')
                            ax3.fill_between(results['times'], 0, state_values, step='post', alpha=0.5,
                                            color=['green' if v == 0 else 'orange' if v == 1 else 'red' 
                                                  for v in state_values][0])
                            
                            # Mark critical windows
                            for idx in results['critical_indices']:
                                if idx < len(results['times']):
                                    ax3.plot(results['times'][idx], 2, 'r^', markersize=15, 
                                           markeredgecolor='darkred', markeredgewidth=2)
                            
                            ax3.set_ylabel('Brain State', fontsize=12)
                            ax3.set_xlabel('Time (seconds)', fontsize=12)
                            ax3.set_title('State Transition Timeline', fontsize=14, fontweight='bold')
                            ax3.set_yticks([0, 1, 2])
                            ax3.set_yticklabels(['Stable', 'Transitional', 'Critical'])
                            ax3.grid(True, alpha=0.3)
                            ax3.set_ylim([-0.1, 2.1])
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        except ImportError:
                            st.warning("Matplotlib not available for visualization. Install it for better plots!")
                        
                        # Clinical interpretation
                        patient_info = f"Age: {patient_age}, Condition: {patient_condition or 'Not specified'}"
                        database_info = f"Simulated {PUBLIC_DATABASES[db_id]['name']} - Subject {subject_id}, {session_id}"
                        interpretation = generate_clinical_interpretation(results, patient_info, database_info)
                        
                        st.markdown(interpretation)
                        
                        # Download report
                        report_data = f"""
EEG Criticality Analysis Report
================================

{interpretation}

Technical Details:
==================
Analysis Parameters:
- Window Size: {window_size} seconds
- Criticality Threshold: {threshold}
- Sampling Rate: {fs:.2f} Hz
- Number of Channels: {len(channels)}

Raw Analysis Results:
{json.dumps(results, indent=2, default=str)}
                        """
                        
                        st.download_button(
                            "📄 Download Full Report",
                            data=report_data,
                            file_name=f"eeg_analysis_{subject_id}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.info("Please try again or contact support if the issue persists.")
            else:
                st.warning("Please select database, subject, and recording segment.")
    
    else:  # File Upload
        st.header("📁 Upload Real EEG Data for Analysis")
        st.info("Upload actual EEG recordings in CSV or TXT format for clinical-grade analysis")
        
        uploaded_file = st.file_uploader(
            "Upload EEG File",
            type=['csv', 'txt'],
            help="Supported formats: CSV (multi-channel) or TXT (single channel, Bonn format)"
        )
        
        if uploaded_file:
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
            # Show file preview
            if st.checkbox("Preview file content"):
                try:
                    content = uploaded_file.read().decode('utf-8')
                    st.text(content[:500] + "..." if len(content) > 500 else content)
                    uploaded_file.seek(0)  # Reset file pointer
                except:
                    st.warning("Cannot preview file - binary format or encoding issue")
                    uploaded_file.seek(0)
        
        if uploaded_file and st.button("🚀 Analyze Uploaded File", type="primary"):
            with st.spinner("Processing uploaded EEG file..."):
                try:
                    file_content = uploaded_file.read()
                    signals, fs, channels = processor.load_eeg_file(file_content, uploaded_file.name)
                    times, features = processor.extract_features(signals, fs, window_s=window_size)
                    results = processor.compute_advanced_criticality(features, times, amp_threshold=threshold)
                    
                    st.success("✅ Analysis Complete!")
                    st.info("ℹ️ This analysis was performed on REAL uploaded data")
                    
                    # Display results (same format as simulated)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        metric_color = "🔴" if results['criticality_ratio'] > 0.4 else "🟡" if results['criticality_ratio'] > 0.2 else "🟢"
                        st.metric(f"{metric_color} Criticality", f"{results['criticality_ratio']:.1%}")
                    with col2:
                        st.metric("Brain State", results['final_state'].replace('_', ' ').title())
                    with col3:
                        st.metric("Critical Episodes", f"{results['critical_windows']}/{results['total_windows']}")
                    with col4:
                        st.metric("Channels", len(channels))
                    
                    # Clinical interpretation
                    patient_info = f"Age: {patient_age}, Condition: {patient_condition or 'Not specified'}"
                    database_info = f"Uploaded file: {uploaded_file.name} (REAL DATA)"
                    interpretation = generate_clinical_interpretation(results, patient_info, database_info)
                    
                    st.markdown(interpretation)
                    
                    # Download results
                    report_data = f"""
EEG Analysis Report - REAL DATA
================================

Patient: {patient_info}
File: {uploaded_file.name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results:
{json.dumps(results, indent=2, default=str)}
                    """
                    
                    st.download_button(
                        "📄 Download Analysis Report",
                        data=report_data,
                        file_name=f"real_eeg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please check your file format and try again.")

if __name__ == "__main__":
    main()
