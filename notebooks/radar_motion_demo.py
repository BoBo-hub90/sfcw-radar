# Standalone simulation demonstrating Doppler velocity estimation
# and micro-Doppler spectrogram for a walking person.
# Uses synthetic data at f0=24 GHz, fs=2000 Hz.
# See src/processing/pipeline.py doppler_process() for the real
# implementation adapted for SFCW at 1-4 GHz, 1.5s sweep period.

"""
Radar ile HAREKET algilama
===========================
radar_vitals.py sabit bir hedefin minik gogus titresimlerini buluyordu.
Bu betik, diyagramdaki "Hareket -> Doppler / mikro-Doppler" kutusunu
gerceklestirir:

  BOLUM A  Yaklasan bir kisinin HIZINI, Doppler fazinin turevinden geri
           cikarir.        v(t) = -(lambda / 4*pi) * dphi/dt

  BOLUM B  Yuruyen bir insanin MIKRO-DOPPLER imzasi: govde sabit hizda
           gelirken kol/bacaklar salinir. Kisa-zamanli Fourier donusumu
           (STFT) ile govde cizgisi + salinan uzuv yan bantlari gorulur.
           Bu imza, insani katı bir nesneden (tek temiz Doppler cizgisi)
           ayirir.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)

c   = 3e8
f0  = 24e9
lam = c / f0                       # ~12.5 mm
fs  = 2000.0                       # yuksek hiz: Doppler frekanslarini yakalamak icin


# ============================================================
# BOLUM A — Hareket eden hedefin hizini bulma
# ============================================================
T  = 6.0
t  = np.arange(0, T, 1 / fs)

# GERCEK hiz profili: ~1 m/s yaklasan, hafif degisen bir kisi
v_true = 1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)       # m/s (radar yonunde)
R0 = 5.0
R  = R0 - np.cumsum(v_true) / fs                        # mesafe = R0 - integral(v)

s = np.exp(1j * 4 * np.pi * R / lam)                    # alinan IQ sinyali
# gurultu (30 dB SNR)
np_ = np.mean(np.abs(s) ** 2) / 10 ** (30 / 10)
s = s + np.sqrt(np_ / 2) * (np.random.randn(t.size) + 1j * np.random.randn(t.size))

# --- HIZ KESTIRIMI: fazin turevi ---
phase   = np.unwrap(np.angle(s))
dphidt  = np.gradient(phase, 1 / fs)
v_est   = -(lam / (4 * np.pi)) * dphidt
v_est   = uniform_filter1d(v_est, size=201)             # turev gurultusunu yumusat

print("=== BOLUM A: Hiz kestirimi ===")
print(f"  Ortalama gercek hiz : {v_true.mean():.2f} m/s")
print(f"  Ortalama kestirim   : {v_est[200:-200].mean():.2f} m/s\n")


# ============================================================
# BOLUM B — Yuruyen insanin mikro-Doppler imzasi
# ============================================================
t2 = np.arange(0, 5.0, 1 / fs)
f_step = 1.0                                            # adim/salinim frekansi (Hz)
v_torso = 1.0                                           # govde sabit hizda yaklasiyor

# her sacici: (genlik, salinim_genligi[m], faz)  govde + 2 bacak + 2 kol
scatterers = [
    (1.0, 0.00, 0.0),          # govde (salinmaz)
    (0.4, 0.35, 0.0),          # bacak 1
    (0.4, 0.35, np.pi),        # bacak 2 (ters faz)
    (0.3, 0.25, np.pi),        # kol 1
    (0.3, 0.25, 0.0),          # kol 2
]

R0b = 4.0
sig = np.zeros_like(t2, dtype=complex)
for amp, swing, ph in scatterers:
    Rk = R0b - v_torso * t2 + swing * np.sin(2 * np.pi * f_step * t2 + ph)
    sig += amp * np.exp(1j * 4 * np.pi * Rk / lam)

# --- elle yazilmis STFT (kompleks giris -> cift yonlu Doppler) ---
def stft(x, nperseg=256, hop=48):
    win = np.hanning(nperseg)
    cols = range(0, len(x) - nperseg, hop)
    S = np.array([np.fft.fftshift(np.fft.fft(x[i:i + nperseg] * win)) for i in cols]).T
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / fs))
    tt = (np.array(list(cols)) + nperseg / 2) / fs
    return f, tt, np.abs(S)

f, tt, S = stft(sig)
v_axis = (-lam * f / 2)[::-1]                           # Doppler frekansi -> radyal hiz
S = S[::-1, :]
Sdb = 20 * np.log10(S / S.max() + 1e-6)

print("=== BOLUM B: Mikro-Doppler ===")
print(f"  Govde hizi    : {v_torso:.1f} m/s  (spektrogramda sabit parlak cizgi)")
print(f"  Adim frekansi : {f_step:.1f} Hz   (uzuv yan bantlarinin salinim hizi)")


# ============================================================
# Grafikler
# ============================================================
fig, ax = plt.subplots(2, 1, figsize=(9, 8))

ax[0].plot(t, v_true, lw=2, label="gercek hiz")
ax[0].plot(t, v_est, lw=1, alpha=0.85, label="Doppler fazindan kestirim")
ax[0].set_xlabel("zaman (s)"); ax[0].set_ylabel("radyal hiz (m/s)")
ax[0].set_title("BOLUM A — Hareket eden kisinin hizi geri cikariliyor")
ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3)

pm = ax[1].pcolormesh(tt, v_axis, Sdb, cmap="viridis", vmin=-40, vmax=0, shading="auto")
ax[1].set_ylim(-2, 4)
ax[1].axhline(v_torso, color="w", ls="--", lw=1, alpha=0.7)
ax[1].set_xlabel("zaman (s)"); ax[1].set_ylabel("radyal hiz (m/s)")
ax[1].set_title("BOLUM B — Yuruyen insanin mikro-Doppler imzasi "
                "(govde cizgisi + uzuv yan bantlari)")
fig.colorbar(pm, ax=ax[1], label="guc (dB)")

plt.tight_layout()
plt.savefig("/home/claude/radar_motion.png", dpi=130)
print("\nGrafik kaydedildi: radar_motion.png")
