import io
import numpy as np
import streamlit as st

try:
    import control as ctl
except Exception as e:
    st.error("The 'control' library is required. Install with: pip install control")
    raise

from dataclasses import dataclass
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bode Plot Tool", layout="wide")

st.title("Bode Plot Tool")
st.caption("Bode plots, margins/bandwidth, time delays, compensators.")

# ---------- Helpers ----------
@dataclass
class SpecConfig:
    show_specs: bool
    min_phase_margin_deg: float | None
    min_gain_margin_db: float | None
    target_bw_rad: float | None
    weight_sensitivity_peak: float | None

def parse_tf(mode: str):
    if mode == "Numerator / Denominator":
        num = st.text_input("Numerator coefficients (highest power first)", value="1")
        den = st.text_input("Denominator coefficients (highest power first)", value="1, 1")
        try:
            num_c = [float(x) for x in num.replace(";",",").split(",") if x.strip()!=""]
            den_c = [float(x) for x in den.replace(";",",").split(",") if x.strip()!=""]
            sys = ctl.tf(num_c, den_c)
            return sys, None
        except Exception as e:
            return None, f"Failed to parse numerator/denominator: {e}"
    elif mode == "Zeros / Poles / Gain":
        z = st.text_input("Zeros (comma-separated)", value="")
        p = st.text_input("Poles (comma-separated)", value="-1, -2")
        k = st.number_input("Gain k", value=1.0, step=0.1, format="%.6g")
        try:
            zeros = [float(x) for x in z.replace(";",",").split(",") if x.strip()!=""]
            poles = [float(x) for x in p.replace(";",",").split(",") if x.strip()!=""]
            sys = ctl.zpk(zeros, poles, k)
            return sys, None
        except Exception as e:
            return None, f"Failed to parse zeros/poles/gain: {e}"
    else:
        return None, "Unknown TF input mode"

def apply_compensator(sys):
    """Add optional compensator via checkbox."""
    add_comp = st.checkbox("Add compensator C(s)", value=False)

    # Wenn Checkbox ausgeschaltet → Reset und Identity zurückgeben
    if not add_comp:
        st.session_state.pop("pid_params", None)
        st.session_state.pop("last_ctype", None)
        return sys, ctl.tf([1], [1]), None

    # --- Auswahl Kompensatortyp ---
    ctype = st.selectbox(
        "Compensator type",
        ["PID", "Lead", "Lag", "Lead-Lag", "Dynamic (num/den)"],
        index=0,
    )

    # --- Prüfe, ob Typ gewechselt wurde ---
    last_ctype = st.session_state.get("last_ctype")
    if last_ctype != ctype:
        # Reset aller Kompensator-spezifischen States, wenn Typ wechselt
        st.session_state.pop("pid_params", None)
    st.session_state["last_ctype"] = ctype
    
    if ctype == "PID":
        Kp = st.number_input("Kp", value=1.0, step=0.1, format="%.6g")
        Ki = st.number_input("Ki", value=1.0, step=0.1, format="%.6g")
        Kd = st.number_input("Kd", value=1.0, step=0.1, format="%.6g")
        N  = st.number_input("Derivative filter N", value=10.0, step=1.0, format="%.6g")
        C = ctl.tf([Kd*N, Kp*N, Ki*N], [1, N, 0])
        st.session_state["pid_params"] = {"Kp": Kp, "Ki": Ki, "Kd": Kd, "N": N}
        return C*sys, C, None
    
    if ctype in ["Lead", "Lag", "Lead-Lag"]:
        if ctype == "Lead":
            z = st.number_input("Lead zero (rad/s)", value=1.0, step=0.1, format="%.6g")
            p = st.number_input("Lead pole (rad/s)", value=10.0, step=0.1, format="%.6g")
            k = st.number_input("Lead gain", value=1.0, step=0.1, format="%.6g")
            C = k * ctl.tf([1/z, 1], [1/p, 1])
        elif ctype == "Lag":
            z = st.number_input("Lag zero (rad/s)", value=0.1, step=0.1, format="%.6g")
            p = st.number_input("Lag pole (rad/s)", value=0.01, step=0.01, format="%.6g")
            k = st.number_input("Lag gain", value=1.0, step=0.1, format="%.6g")
            C = k * ctl.tf([1/z, 1], [1/p, 1])
        else:
            z1 = st.number_input("Lead zero (rad/s)", value=1.0, key="llz1", step=0.1, format="%.6g")
            p1 = st.number_input("Lead pole (rad/s)", value=10.0, key="llp1", step=0.1, format="%.6g")
            z2 = st.number_input("Lag  zero (rad/s)", value=0.1, key="llz2", step=0.1, format="%.6g")
            p2 = st.number_input("Lag  pole (rad/s)", value=0.01, key="llp2", step=0.01, format="%.6g")
            k  = st.number_input("Overall gain", value=1.0, key="llk", step=0.1, format="%.6g")
            C = k * ctl.tf([1/z1, 1], [1/p1, 1]) * ctl.tf([1/z2, 1], [1/p2, 1])
        return C*sys, C, None
    
    if ctype == "Dynamic (num/den)":
        num = st.text_input("Compensator numerator coefficients", value="1")
        den = st.text_input("Compensator denominator coefficients", value="1, 1")
        try:
            num_c = [float(x) for x in num.replace(";",",").split(",") if x.strip()!=""]
            den_c = [float(x) for x in den.replace(";",",").split(",") if x.strip()!=""]
            C = ctl.tf(num_c, den_c)
            return C*sys, C, None
        except Exception as e:
            return sys, ctl.tf([1],[1]), f"Failed to parse compensator: {e}"
    return sys, ctl.tf([1],[1]), None

def apply_delay(sys):
    add_delay = st.checkbox("Add time delay τ (Padé or exact)", value=False)
    if not add_delay:
        return sys, None, None
    tau = st.number_input("Time delay τ (seconds)", value=1.0, step=0.1, format="%.6g")
    method = st.radio("Delay model", ["Padé approximation", "Exact e^{-sτ} (frequency-domain only)"],
                      index=0, horizontal=True)
    if method == "Padé approximation":
        order = st.slider("Padé order", min_value=1, max_value=4, value=1, step=1)
        num_d, den_d = ctl.pade(tau, order)
        D = ctl.tf(num_d, den_d)
        # ⬅️ Gib sowohl sys*D als auch D selbst zurück
        return sys * D, ("pade", tau, order, D), None
    else:
        return sys, ("exact", tau, None, None), None

def bode_np(sys, w):
    mag, phase, omega = ctl.freqresp(sys, w)
    mag = np.squeeze(mag)
    phase = np.squeeze(phase) * 180/np.pi
    return mag, phase

def compute_margins_and_bw(L, w):
    try:
        gm, pm, wgc, wpc = ctl.margin(L)
    except Exception:
        gm = pm = wgc = wpc = np.nan
    T = ctl.feedback(L, 1)
    try:
        magT, phaseT = bode_np(T, w)
        magT_db = 20*np.log10(magT)
        idx = np.where(magT_db <= -3.0)[0]
        bw = w[idx[0]] if len(idx) > 0 else np.nan
    except Exception:
        bw = np.nan
    return gm, pm, wgc, wpc, bw

def bode_with_optional_exact_delay(L, delay_info, w):
    mag, phase = bode_np(L, w)
    if delay_info is None or delay_info[0] != "exact":
        return mag, phase, None, None
    tau = delay_info[1]
    phase_delayed = phase - w * tau * 180/np.pi
    return mag, phase, mag, phase_delayed

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Transfer Function")
    mode = st.selectbox("Input form", ["Numerator / Denominator", "Zeros / Poles / Gain"])
    P, err = parse_tf(mode)
    if err:
        st.error(err)
    K = st.number_input("Scalar gain multiplier", value=1.0, step=0.1, format="%.6g")
    if P is not None:
        P = K*P

    st.header("Compensator (optional)")
    if P is not None:
        L_base, C, comp_err = apply_compensator(P)
        if comp_err:
            st.warning(comp_err)
    else:
        L_base, C = None, ctl.tf([1],[1])

    st.header("Time Delay (optional)")
    if L_base is not None:
        L_delay, delay_info, delay_err = apply_delay(L_base)
    else:
        L_delay, delay_info, delay_err = None, None, None

    st.header("Frequency range")
    wmin = st.number_input("ω min (rad/s)", value=1e-2, step=0.1, format="%.6g")
    wmax = st.number_input("ω max (rad/s)", value=1e2, step=10.0, format="%.6g")
    pts  = 750 #sets a fixed value for the frequency points
    # FIXED: Removed max(wmax, wmin*10) - now uses wmin and wmax directly
    w = np.logspace(np.log10(max(wmin, 1e-8)), np.log10(wmax), pts)

    st.header("Specs (\"Bode obstacle course\")")
    show_specs = st.checkbox("Show design specs overlay", value=False)
    min_pm = st.number_input("Min phase margin (deg)", value=45.0, step=1.0, format="%.6g") if show_specs else None
    min_gm_db = st.number_input("Min gain margin (dB)", value=6.0, step=1.0, format="%.6g") if show_specs else None
    target_bw = st.number_input("Target closed-loop BW ω_B (rad/s)", value=10.0, step=0.5, format="%.6g") if show_specs else None
    Ms = st.number_input("Sensitivity peak M_S ≤", value=2.0, step=0.1, format="%.6g") if show_specs else None
    
    # Low frequency (disturbance attenuation) bounds
    show_freq_bounds = st.checkbox("Show frequency-dependent bounds", value=False) if show_specs else False
    if show_freq_bounds:
        st.write("**Low frequency (disturbance attenuation):**")
        freq_low_max = st.number_input("ω_low_max (rad/s)", value=0.1, step=0.01, format="%.6g")
        mag_low_min = st.number_input("Min magnitude at low freq (dB)", value=-20.0, step=1.0, format="%.6g")
        
        st.write("**High frequency (measurement noise):**")
        freq_high_min = st.number_input("ω_high_min (rad/s)", value=100.0, step=10.0, format="%.6g")
        mag_high_max = st.number_input("Max magnitude at high freq (dB)", value=-40.0, step=1.0, format="%.6g")
        
        freq_bounds = {
            "show": True,
            "freq_low_max": freq_low_max,
            "mag_low_min": mag_low_min,
            "freq_high_min": freq_high_min,
            "mag_high_max": mag_high_max
        }
    else:
        freq_bounds = {"show": False}
    
    specs = SpecConfig(show_specs, min_pm, min_gm_db, target_bw, Ms)

# ---------- Main ----------
if P is None:
    st.stop()

# --- Combined Transfer Function Display (with optional delay) ---
if P is not None:
    st.subheader("Resulting Transfer Functions")

    # helper formatting
    def clean_num(x):
        if abs(x - round(x)) < 1e-10:
            return str(int(round(x)))
        else:
            return f"{x:.4g}"

    def poly_to_latex(coeffs):
        terms = []
        n = len(coeffs)
        for i, c in enumerate(coeffs):
            power = n - i - 1
            if abs(c) < 1e-12:
                continue
            c_str = clean_num(c)
            if power == 0:
                terms.append(f"{c_str}")
            elif power == 1:
                terms.append(f"{c_str} s")
            else:
                terms.append(f"{c_str} s^{{{power}}}")
        return " + ".join(terms) if terms else "0"

    def tf_to_latex(tf_obj):
        num, den = ctl.tfdata(tf_obj)
        num = [float(n) for n in num[0][0]]
        den = [float(d) for d in den[0][0]]
        num_str = poly_to_latex(num)
        den_str = poly_to_latex(den)
        return r"\frac{" + num_str + "}{" + den_str + "}"

    # Detect if compensator active
    comp_active = "C" in locals() and C is not None and not (
        len(ctl.tfdata(C)[0][0][0]) == 1 and ctl.tfdata(C)[0][0][0][0] == 1
    )

    # Choose which system to show (with delay or base)
    L_display = L_delay if delay_info is not None else L_base

    # --- Case 1: No compensator ---
    if not comp_active:
        if delay_info is None:
            st.latex(r"L(s) = P(s) = " + tf_to_latex(P))
        else:
            if delay_info[0] == "exact":
                tau = delay_info[1]
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[6pt]
                    L(s) &= {tf_to_latex(P)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )

            elif delay_info[0] == "pade":
                tau, order = delay_info[1], delay_info[2]
                D = delay_info[3]
                num_d, den_d = ctl.pade(tau, order)
                num_str = poly_to_latex(num_d)
                den_str = poly_to_latex(den_d)

                # Saubere LaTeX-Strings mit Formatkorrektur
                pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"

                st.latex(
                    rf"""\begin{{aligned}}
                    &{pade_formula}\\[6pt]
                    L(s) &= P(s)\,D_{{\text{{Padé}}}}(s)
                    = \big({tf_to_latex(P)}\big)\,\big({tf_to_latex(D)}\big)\\[8pt]
                    L(s) &= {tf_to_latex(L_delay)}
                    \end{{aligned}}"""
                )

    # --- Case 2: With compensator (PID and non-PID) ---
    else:
        L = C * P
        pid = st.session_state.get("pid_params", None)

        if pid is not None:  # PID custom formatting
            def cn(x):
                return str(int(round(x))) if abs(x - round(x)) < 1e-10 else f"{x:.4g}"
            Kp, Ki, Kd, N = pid["Kp"], pid["Ki"], pid["Kd"], pid["N"]

            terms = []
            if abs(Kp) > 1e-12:
                terms.append(cn(Kp))
            if abs(Ki) > 1e-12:
                terms.append(rf"\frac{{{cn(Ki)}}}{{s}}")
            if abs(Kd) > 1e-12:
                terms.append(rf"\frac{{{cn(Kd)}\,\cdot {cn(N)}\,s}}{{s+{cn(N)}}}")
            if not terms:
                terms = ["0"]
            pid_expr = " + ".join(terms)

            if delay_info is None:
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s) = \big({pid_expr}\big)\cdot{tf_to_latex(P)}\\[10pt]
                    L(s) &= {tf_to_latex(L)}
                    \end{{aligned}}"""
                )

            elif delay_info[0] == "exact":
                tau = delay_info[1]
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )

            elif delay_info[0] == "pade":
                tau, order = delay_info[1], delay_info[2]
                D = delay_info[3]
                num_d, den_d = ctl.pade(tau, order)
                num_str = poly_to_latex(num_d)
                den_str = poly_to_latex(den_d)

                pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"

                st.latex(
                    rf"""\begin{{aligned}}
                    &{pade_formula}\\[6pt]
                    L(s) &= C(s)\,P(s)\,D_{{\text{{Padé}}}}(s)
                    = \big({pid_expr}\big)\,\big({tf_to_latex(P)}\big)\,\big({tf_to_latex(D)}\big)\\[10pt]
                    L(s) &= {tf_to_latex(L_delay)}
                    \end{{aligned}}"""
                )

        else:
            # Non-PID compensator
            if delay_info is None:
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s) = {tf_to_latex(C)}\cdot{tf_to_latex(P)}\\[10pt]
                    L(s) &= {tf_to_latex(L)}
                    \end{{aligned}}"""
                )

            elif delay_info[0] == "exact":
                tau = delay_info[1]
                st.latex(
                    rf"""\begin{{aligned}}
                    L(s) &= C(s)\,P(s)\,e^{{-s\tau}},\quad \tau={tau:.3g}\\[8pt]
                    L(s) &= {tf_to_latex(L)}\,e^{{-{tau:.3g}s}}
                    \end{{aligned}}"""
                )

            elif delay_info[0] == "pade":
                tau, order = delay_info[1], delay_info[2]
                D = delay_info[3]
                num_d, den_d = ctl.pade(tau, order)
                num_str = poly_to_latex(num_d)
                den_str = poly_to_latex(den_d)
                pade_formula = rf"e^{{-s\tau}} \approx \frac{{{num_str}}}{{{den_str}}},\quad n={order},\;\tau={tau:.3g}"

                st.latex(
                    rf"""\begin{{aligned}}
                    &{pade_formula}\\[6pt]
                    L(s) &= C(s)\,P(s)\,D_{{\text{{Padé}}}}(s)
                    = \big({tf_to_latex(C)}\big)\,\big({tf_to_latex(P)}\big)\,\big({tf_to_latex(D)}\big)\\[10pt]
                    L(s) &= {tf_to_latex(L_delay)}
                    \end{{aligned}}"""
                )




col1, col2 = st.columns(2)

# ---------- Margins and Bandwidth ----------
gm, pm, wgc, wpc, bw = compute_margins_and_bw(L_base, w)

# Initialize session state
if "show_margins" not in st.session_state:
    st.session_state.show_margins = False

# Toggle Button BEFORE plots
if st.button("Show / Hide margins"):
    st.session_state.show_margins = not st.session_state.show_margins

show_margins = st.session_state.show_margins

# ---------- Plot: Magnitude ----------
with col1:
    st.markdown("**Magnitude**")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    mag_base, phase_base = bode_np(L_base, w)
    mag_db = 20 * np.log10(mag_base)
    ax1.semilogx(w, mag_db, label="No delay", lw=2)

    # optional delay curve
    if delay_info is not None:
        if delay_info[0] == "pade":
            mag_d, _ = bode_np(L_delay, w)
            ax1.semilogx(w, 20*np.log10(mag_d), "--", label=f"With delay (Padé n={delay_info[2]})", lw=2)
        else:
            mag_d, _, _, _ = bode_with_optional_exact_delay(L_base, delay_info, w)
            ax1.semilogx(w, 20*np.log10(mag_d), "--", label=f"With delay (exact τ={delay_info[1]:.3g}s)", lw=2)

    # Reference line (0 dB)
    ax1.axhline(0, color="black", lw=0.8, ls="-", alpha=0.7)

    # --- Gain Margin ---
    if show_margins and np.isfinite(wgc) and np.isfinite(gm):
        mag_at_wgc = 20 * np.log10(np.abs(ctl.freqresp(L_base, [wgc])[0][0]))
        ax1.axvline(wgc, color="b", ls=":", alpha=0.8)
        ax1.plot(wgc, mag_at_wgc, "bo", markersize=8)
        ax1.vlines(wgc, mag_at_wgc, 0, color="b", lw=1.2)
        ax1.text(wgc, (mag_at_wgc + 0) / 2, f"GM\n{20*np.log10(gm):.2f} dB", color="b", fontsize=9, ha="center", va="center", 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Specs overlay: Min Gain Margin ---
    if specs.show_specs and specs.min_gain_margin_db is not None:
        min_gm_db = specs.min_gain_margin_db
        ax1.axhline(min_gm_db, color="orange", ls="--", lw=2, alpha=0.7, label=f"Min GM = {min_gm_db} dB")
        ax1.fill_between(w, -150, min_gm_db, alpha=0.1, color="red")
    
    # --- Frequency-dependent bounds ---
    if freq_bounds["show"]:
        # Low frequency region (disturbance attenuation)
        freq_low_max = freq_bounds["freq_low_max"]
        mag_low_min = freq_bounds["mag_low_min"]
        ax1.fill_between(w[w <= freq_low_max], mag_low_min, -150, alpha=0.2, color="red", label="Disturbance attenuation")
        ax1.axvline(freq_low_max, color="gray", ls="--", lw=1, alpha=0.5)
        
        # High frequency region (measurement noise)
        freq_high_min = freq_bounds["freq_high_min"]
        mag_high_max = freq_bounds["mag_high_max"]
        ax1.fill_between(w[w >= freq_high_min], mag_high_max, 150, alpha=0.2, color="yellow", label="Measurement noise")
        ax1.axvline(freq_high_min, color="gray", ls="--", lw=1, alpha=0.5)

    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlabel("ω (rad/s)")
    ax1.set_xlim([w[0], w[-1]])  # FIXED: Force proper x-range
    ax1.grid(True, which="both", ls=":")
    ax1.legend(fontsize=9)
    st.pyplot(fig1)

# ---------- Plot: Phase ----------
with col2:
    st.markdown("**Phase**")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    
    # --- Phase plot: NO unwrapping to avoid artifacts ---
    ax2.semilogx(w, phase_base, label="No delay", lw=2)

    if delay_info is not None:
        if delay_info[0] == "pade":
            _, phase_d = bode_np(L_delay, w)
            ax2.semilogx(w, phase_d, "--", label=f"With delay (Padé n={delay_info[2]})", lw=2)
        else:
            _, phase_d_delayed, _, _ = bode_with_optional_exact_delay(L_base, delay_info, w)
            ax2.semilogx(w, phase_d_delayed, "--", label=f"With delay (exact τ={delay_info[1]:.3g}s)", lw=2)

    # Reference line (−180°)
    ax2.axhline(-180, color="black", lw=0.8, ls="-", alpha=0.7)

    # Bandwidth markieren
    if np.isfinite(bw):
        ax2.axvline(bw, color="g", ls="--", alpha=0.6, lw=1.5, label=f"BW ω_bw={bw:.3g}")
    
    # --- Phase Margin ---
    if show_margins and np.isfinite(wpc) and np.isfinite(pm):
        phase_at_wpc = np.interp(wpc, w, phase_base)
        ax2.axvline(wpc, color="r", ls=":", alpha=0.8)
        ax2.plot(wpc, phase_at_wpc, "ro", markersize=8)
        ax2.vlines(wpc, -180, phase_at_wpc, color="r", lw=1.2)
        y_mid = (phase_at_wpc - 180) / 2
        ax2.text(wpc, y_mid, f"PM\n{pm:.2f}°", color="r", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Specs overlay: Min Phase Margin und Target Bandwidth ---
    if specs.show_specs:
        if specs.min_phase_margin_deg is not None:
            min_pm = specs.min_phase_margin_deg
            spec_phase_line = -180 + min_pm
            ax2.axhline(spec_phase_line, color="orange", ls="--", lw=2, alpha=0.7, label=f"Min PM = {min_pm}°")
            ax2.fill_between(w, -500, spec_phase_line, alpha=0.1, color="red")
        
        if specs.target_bw_rad is not None:
            target_bw = specs.target_bw_rad
            ax2.axvline(target_bw, color="purple", ls="--", lw=2, alpha=0.7, label=f"Target BW = {target_bw} rad/s")

    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("ω (rad/s)")
    ax2.set_xlim([w[0], w[-1]])  # FIXED: Force same x-range as magnitude plot
    ax2.grid(True, which="both", ls=":")
    ax2.legend(fontsize=9)
    st.pyplot(fig2)

# ---------- Margins & Bandwidth ----------
st.subheader("Margins & Bandwidth")

colA, colB, colC, colD, colE = st.columns(5)

if show_margins:
    colA.metric("Gain margin (×)", f"{gm:.3g}" if np.isfinite(gm) else "–")
    colB.metric("Gain margin (dB)", f"{20*np.log10(gm):.3g}" if np.isfinite(gm) else "–")
    colC.metric("Phase margin (deg)", f"{pm:.3g}" if np.isfinite(pm) else "–")
    colD.metric("ω_gc (rad/s)", f"{wgc:.3g}" if np.isfinite(wgc) else "–")
    colE.metric("ω_pc (rad/s)", f"{wpc:.3g}" if np.isfinite(wpc) else "–")
else:
    colA.metric("Gain margin (×)", "–")
    colB.metric("Gain margin (dB)", "–")
    colC.metric("Phase margin (deg)", "–")
    colD.metric("ω_gc (rad/s)", "–")
    colE.metric("ω_pc (rad/s)", "–")


# ---------- Export ----------
st.subheader("Export")
export_mode = st.radio("Select export mode:",["Magnitude", "Phase", "Combined"],index=0,horizontal=True)
st.caption("After changing the export mode, please wait a few seconds until the download button updates.")

buf = io.BytesIO()

if export_mode == "Magnitude":
    fig, ax = plt.subplots()
    ax.semilogx(w, 20*np.log10(mag_base))
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlabel("ω (rad/s)")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    st.download_button(
        "Download magnitude plot (PNG)",
        data=buf.getvalue(),
        file_name="bode_magnitude.png",
        mime="image/png"
    )

elif export_mode == "Phase":
    fig, ax = plt.subplots()
    ax.semilogx(w, phase_base)
    ax.set_ylabel("Phase (deg)")
    ax.set_xlabel("ω (rad/s)")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    st.download_button(
        "Download phase plot (PNG)",
        data=buf.getvalue(),
        file_name="bode_phase.png",
        mime="image/png"
    )

else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.semilogx(w, 20*np.log10(mag_base))
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xlabel("ω (rad/s)")
    ax1.grid(True, which="both", ls=":")
    ax2.semilogx(w, phase_base)
    ax2.set_ylabel("Phase (deg)")
    ax2.set_xlabel("ω (rad/s)")
    ax2.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    st.download_button(
        "Download combined (PNG)",
        data=buf.getvalue(),
        file_name="bode_combined.png",
        mime="image/png"
    )