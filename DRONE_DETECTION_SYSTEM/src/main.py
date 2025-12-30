import numpy as np
import pyaudio
import librosa
import joblib
import os
import sys
import math
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
MODEL_FILE = "drone_brain_v2.pkl"
SCALER_FILE = "feature_scaler.pkl"
RATE = 16000
CHUNK = 2048
CHANNELS = 6
FORMAT = pyaudio.paInt16
DEVICE_INDEX = 1
BUFFER_SECONDS = 3.0

# 🛠️ "RAW MODE" TUNING
SMOOTH_FACTOR = 0.3      # 0.1 = Slow/Laggy, 1.0 = Instant/Jittery. 0.3 is "Silky".
CONFIDENCE_GATE = 45.0   # Show detection if AI is > 45% sure (More sensitive)

# ==========================================
# 🛠️ CALIBRATION
# ==========================================
MIRROR_X = True
MIRROR_Y = False
ROTATION_OFFSET = 0
# ==========================================

# --- GLOBALS ---
latest_audio = np.zeros(CHUNK)
latest_conf = 0.0
latest_angle = 0.0
latest_vol = 0.0
is_drone = False
running = True

# --- 1. LOAD AI ---
print("🚀 LOADING RAW FAST SYSTEM...")
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    print("❌ ERROR: Files missing!")
    print(f"   Looking for: {MODEL_FILE} and {SCALER_FILE}")
    sys.exit()

try:
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("✅ SYSTEM READY: RAW MODE ACTIVE")
except Exception as e:
    print(f"❌ LOAD ERROR: {e}")
    sys.exit()

# --- 2. FEATURE EXTRACTOR (Advanced) ---
def extract_live_features(audio_data, sr=16000):
    try:
        audio = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
        features = []
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfccs.T, axis=0))
        features.extend(np.std(mfccs.T, axis=0))
        
        features.extend([
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
            np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
        ])
        
        features.append(np.mean(librosa.feature.zero_crossing_rate(audio)))
        features.extend(np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1))
        features.append(np.mean(librosa.feature.rms(y=audio)))
        
        return np.array(features).reshape(1, -1)
    except:
        return None

# --- 3. AUDIO THREAD (RAW SPEED) ---
def audio_thread():
    global latest_conf, is_drone, latest_angle, latest_audio, latest_vol
    
    p = pyaudio.PyAudio()
    dev_index = DEVICE_INDEX
    
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        input_device_index=dev_index, frames_per_buffer=CHUNK)
    except:
        # Fallback: Search for "Seeed" device if index 1 fails
        info = p.get_host_api_info_by_index(0)
        for i in range(info.get('deviceCount')):
            if "Seeed" in p.get_device_info_by_host_api_device_index(0, i).get('name'):
                dev_index = i
                break
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        input_device_index=dev_index, frames_per_buffer=CHUNK)

    print("🛡️ LISTENING (RAW MODE)...")
    
    audio_buffer = []
    frames_needed = int(RATE / CHUNK * BUFFER_SECONDS)
    
    # Variables for smooth interpolation
    smooth_x, smooth_y = 0, 0

    while running:
        try:
            raw_data = stream.read(CHUNK, exception_on_overflow=False)
            if len(raw_data) != CHUNK * 2 * CHANNELS: continue
            
            data_int = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, CHANNELS)
            latest_audio = data_int[:, 0]
            
            # Volume (No Gating! We calculate angle even for quiet sounds)
            vol_calc = np.sqrt(np.mean(data_int[:, 0]**2))
            latest_vol = 0 if np.isnan(vol_calc) else vol_calc

            # --- A. INSTANT DIRECTION (DOA) ---
            # Mapping based on ReSpeaker 4-Mic Array
            vectors = [
                (data_int[:, 1], 135), (data_int[:, 2], 225),
                (data_int[:, 3], 315), (data_int[:, 4], 45)
            ]
            sum_x, sum_y = 0, 0
            for mic_data, angle in vectors:
                rms = np.sqrt(np.mean(mic_data.astype(float)**2))
                if np.isnan(rms): rms = 0
                rad = math.radians(angle)
                sum_x += rms * math.cos(rad)
                sum_y += rms * math.sin(rad)
            
            if MIRROR_X: sum_x = -sum_x
            if MIRROR_Y: sum_y = -sum_y

            # --- B. SILKY SMOOTHING (Interpolation) ---
            # Blend new value with old for smooth needle movement
            smooth_x = (smooth_x * (1 - SMOOTH_FACTOR)) + (sum_x * SMOOTH_FACTOR)
            smooth_y = (smooth_y * (1 - SMOOTH_FACTOR)) + (sum_y * SMOOTH_FACTOR)
            
            if smooth_x != 0 or smooth_y != 0:
                deg = math.degrees(math.atan2(smooth_y, smooth_x))
                latest_angle = (deg + ROTATION_OFFSET) % 360

            # --- C. AI INFERENCE ---
            audio_buffer.append(data_int[:, 0])
            if len(audio_buffer) >= frames_needed:
                full_clip = np.concatenate(audio_buffer).astype(np.float32)
                
                # Run heavy feature extraction
                feat = extract_live_features(full_clip)
                
                if feat is not None:
                    probs = clf.predict_proba(scaler.transform(feat))[0]
                    drone_prob = probs[1] * 100
                    latest_conf = drone_prob # Raw update
                    is_drone = latest_conf > CONFIDENCE_GATE
                
                # Overlap logic: Keep the last 1 second, discard the rest
                overlap = int(RATE / CHUNK * 1.0)
                audio_buffer = audio_buffer[-overlap:]

        except Exception as e:
            pass

t = threading.Thread(target=audio_thread)
t.daemon = True
t.start()

# --- 4. GUI (FAST REFRESH) ---
plt.style.use('dark_background')
fig = plt.figure(figsize=(10, 6))
gs = GridSpec(2, 2, figure=fig)
fig.canvas.manager.set_window_title('SKY-WATCH: RAW')

ax_radar = fig.add_subplot(gs[:, 0], projection='polar')
ax_radar.set_theta_zero_location("N")
ax_radar.set_theta_direction(-1)
ax_radar.set_ylim(0, 1)
ax_radar.set_yticks([])
ax_radar.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
ax_radar.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], color='lime')
ax_radar.set_facecolor('black')

radar_dot, = ax_radar.plot([], [], 'ro', markersize=25, markeredgecolor='white', zorder=10)
sweep_line, = ax_radar.plot([], [], color='lime', lw=2, alpha=0.5)

ax_bar = fig.add_subplot(gs[0, 1])
ax_bar.set_title("CONFIDENCE", color='lime')
ax_bar.set_ylim(0, 100)
bar = ax_bar.bar(['Prob'], [0], color='green', width=0.4)

status_txt = fig.text(0.5, 0.92, 'INITIALIZING...', ha='center', fontsize=18, color='lime', weight='bold')

sweep_angle = 0

def update_gui(frame):
    global sweep_angle
    
    safe_conf = 0 if np.isnan(latest_conf) else latest_conf
    safe_angle = 0 if np.isnan(latest_angle) else int(latest_angle)

    bar[0].set_height(safe_conf)
    
    sweep_angle = (sweep_angle + 8) % 360 
    sweep_line.set_data([np.deg2rad(sweep_angle)]*2, [0, 1])
    
    if is_drone and safe_conf > CONFIDENCE_GATE:
        fig.patch.set_facecolor('#1a0000')
        ax_radar.set_facecolor('#1a0000')
        
        rad_target = np.deg2rad(safe_angle)
        radar_dot.set_data([rad_target], [0.8])
        radar_dot.set_alpha(1.0)
        
        status_txt.set_text(f"🚨 TARGET LOCKED: {safe_angle}°")
        status_txt.set_color('red')
        bar[0].set_color('red')
    else:
        fig.patch.set_facecolor('black')
        ax_radar.set_facecolor('black')
        radar_dot.set_alpha(0)
        
        status_txt.set_text(f"SCANNING... (Vol: {int(latest_vol)})")
        status_txt.set_color('lime')
        bar[0].set_color('green')

    return radar_dot, sweep_line, bar[0], status_txt

# ⚡ FAST UPDATE (40ms = 25 FPS)
ani = FuncAnimation(fig, update_gui, interval=40, blit=False)
plt.show()