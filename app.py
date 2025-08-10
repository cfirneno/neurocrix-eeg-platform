import streamlit as st
import numpy as np
import pandas as pd
import io
import json
import time
import requests
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import hashlib

# Password protection
def check_password():
    """Returns True if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
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
    "physionet_chbmit": {
        "name": "CHB-MIT Scalp EEG Database",
        "base_url": "https://physionet.org/files/chbmit/1.0.0/",
        "description": "EEG recordings from 22 pediatric subjects with intractable seizures",
        "format": "edf",
        "subjects": 22,
        "license": "Open Data Commons Attribution License v1.0"
    },
    "bonn_seizure": {
        "name": "Bonn University Seizure Database",
        "base_url": "http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/eegdata/",
        "description": "Small but widely-used seizure detection dataset",
        "format": "txt",
        "subjects": 5,
        "license": "Academic use"
    }
}

class DatabaseConnector:
    def __init__(self, db_config: Dict):
        self.config = db_config
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'EEG-Analysis-Platform/1.0'})
    
    def list_subjects(self) -> List[str]:
        db_id = self.config.get("db_id")
        if db_id == "physionet_chbmit":
            return [f"chb{i:02d}" for i in range(1, 25)]
        elif db_id == "bonn_seizure":
            return ["Z", "O", "N", "F", "S"]
        return []
    
    def get_subject_sessions(self, subject_id: str) -> List[str]:
        db_id = self.config.get("db_id")
        if db_id == "physionet_chbmit":
            return ["01", "02", "03", "04", "05"]
        elif db_id == "bonn_seizure":
            return ["set_A", "set_B", "set_C", "set_D", "set_E"]
        return []
    
    def download_eeg_sample(self, subject_id: str, session_id: str) -> Tuple[bytes, str]:
        """Generate realistic sample EEG data for demonstration."""
        duration = 30  # 30 seconds
        fs = 256
        n_channels = 18 if self.config.get("db_id") == "physionet_chbmit" else 8
        n_samples = duration * fs
        
        time_axis = np.linspace(0, duration, n_samples)
        signals = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            signal = np.zeros(n_samples)
            # Alpha waves (8-13 Hz)
            signal += 50 * np.sin(2 * np.pi * 10 * time_axis + np.random.random() * 2 * np.pi)
            # Beta waves (13-30 Hz)
            signal += 20 * np.sin(2 * np.pi * 20 * time_axis + np.random.random() * 2 * np.pi)
            # Gamma waves (30-100 Hz)
            signal += 10 * np.sin(2 * np.pi * 40 * time_axis + np.random.random() * 2 * np.pi)
            # Add noise
            signal += np.random.normal(0, 5, n_samples)
            
            # Add spikes for seizure data
            if session_id in ["F", "S"] or "seizure" in self.config.get("db_id", ""):
                spike_times = np.random.choice(n_samples, size=5, replace=False)
                for spike_time in spike_times:
                    if spike_time < n_samples - 100:
                        signal[spike_time:spike_time+50] += 200 * np.exp(-np.arange(50)/10)
            
            signals[ch] = signal
        
        df = pd.DataFrame(signals.T, columns=[f"CH_{i+1}" for i in range(n_channels)])
        csv_data = df.to_csv(index=False)
        filename = f"{subject_id}_{session_id}_sample.csv"
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
                fs = 256
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
    - **Alpha Band:** {bands['alpha']['mean_power']:.1f}μV (relaxed awareness)
    - **Beta Band:** {bands['beta']['mean_power']:.1f}μV (cognitive engagement)  
    - **Gamma Band:** {bands['gamma']['mean_power']:.1f}μV (consciousness binding)
    
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
    
    ### **IMPORTANT NOTES**
    - This analysis is for research and educational purposes
    - Clinical decisions should involve qualified healthcare professionals
    - Consider medication effects, sleep state, and patient condition
    
    ---
    *Generated by Advanced EEG Criticality Analysis Platform*
    """
    
    return interpretation

# Initialize processor
@st.cache_resource
def get_processor():
    return EnhancedEEGProcessor()

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
    st.markdown("**Professional brain state analysis with database integration**")
    
    processor = get_processor()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["🌐 Public Database", "📁 File Upload"]
        )
        
        st.subheader("👤 Patient Information")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        patient_condition = st.text_input("Medical Condition", placeholder="e.g., Epilepsy")
        
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (seconds)", 1.0, 5.0, 2.0, 0.5)
        threshold = st.slider("Criticality Threshold", 0.1, 1.0, 0.3, 0.1)
    
    # Main content
    if analysis_type == "🌐 Public Database":
        st.header("🌐 Public Database Analysis")
        
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
                    "Select Subject:",
                    options=db_info["available_subjects"]
                )
            
            with col3:
                if subject_id:
                    connector = processor.connectors[db_id]
                    sessions = connector.get_subject_sessions(subject_id)
                    session_id = st.selectbox("Select Session:", options=sessions)
        
        if st.button("🚀 Analyze from Database", type="primary"):
            if db_id and subject_id and session_id:
                with st.spinner("Processing EEG data from database..."):
                    try:
                        # Download and analyze
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
                            st.metric("Criticality Ratio", f"{results['criticality_ratio']:.1%}")
                        with col2:
                            st.metric("Final State", results['final_state'].replace('_', ' ').title())
                        with col3:
                            st.metric("Critical Episodes", f"{results['critical_windows']}/{results['total_windows']}")
                        with col4:
                            st.metric("Channels", len(channels))
                        
                        # Visualization
                        import matplotlib.pyplot as plt
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        
                        # R parameter evolution
                        colors = ['green' if s == 'stable' else 'orange' if s == 'transitional' else 'red' 
                                 for s in results['state_evolution']]
                        ax1.scatter(results['times'], results['r_evolution'], c=colors, alpha=0.7)
                        ax1.axhline(y=3.57, color='red', linestyle='--', alpha=0.7, label='Chaos Threshold')
                        ax1.set_ylabel('R Parameter')
                        ax1.set_title('Brain Criticality Evolution')
                        ax1.legend()
                        ax1.grid(True, alpha=0.3)
                        
                        # Frequency bands
                        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
                        powers = [results['band_statistics'][band]['mean_power'] for band in bands]
                        ax2.bar(bands, powers, color=['purple', 'blue', 'green', 'orange', 'red'], alpha=0.7)
                        ax2.set_ylabel('Mean Power (μV)')
                        ax2.set_title('Frequency Band Analysis')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Clinical interpretation
                        patient_info = f"Age: {patient_age}, Condition: {patient_condition}"
                        database_info = f"{PUBLIC_DATABASES[db_id]['name']} - Subject {subject_id}, Session {session_id}"
                        interpretation = generate_clinical_interpretation(results, patient_info, database_info)
                        
                        st.markdown(interpretation)
                        
                        # Download report
                        report_data = f"""
EEG Criticality Analysis Report

{interpretation}

Raw Data:
{json.dumps(results, indent=2, default=str)}
                        """
                        
                        st.download_button(
                            "📄 Download Full Report",
                            data=report_data,
                            file_name=f"eeg_analysis_{subject_id}_{session_id}.txt",
                            mime="text/plain"
                        )
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            else:
                st.warning("Please select database, subject, and session.")
    
    else:  # File Upload
        st.header("📁 File Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload EEG File",
            type=['csv', 'npy', 'txt'],
            help="Supported formats: CSV, NPY, TXT"
        )
        
        if uploaded_file and st.button("🚀 Analyze Uploaded File", type="primary"):
            with st.spinner("Processing uploaded EEG file..."):
                try:
                    file_content = uploaded_file.read()
                    signals, fs, channels = processor.load_eeg_file(file_content, uploaded_file.name)
                    times, features = processor.extract_features(signals, fs, window_s=window_size)
                    results = processor.compute_advanced_criticality(features, times, amp_threshold=threshold)
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Display same results format as database analysis
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Criticality Ratio", f"{results['criticality_ratio']:.1%}")
                    with col2:
                        st.metric("Final State", results['final_state'].replace('_', ' ').title())
                    with col3:
                        st.metric("Critical Episodes", f"{results['critical_windows']}/{results['total_windows']}")
                    with col4:
                        st.metric("Channels", len(channels))
                    
                    # Clinical interpretation
                    patient_info = f"Age: {patient_age}, Condition: {patient_condition}"
                    database_info = f"Uploaded file: {uploaded_file.name}"
                    interpretation = generate_clinical_interpretation(results, patient_info, database_info)
                    
                    st.markdown(interpretation)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()
