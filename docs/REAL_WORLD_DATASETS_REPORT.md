# Real-World Datasets and Benchmarks Used in SINDy Papers

**Research Report - February 9, 2026**

This report catalogs real-world datasets and experimental validations used across SINDy papers to identify gaps in our current validation coverage.

---

## Executive Summary

| Category | Datasets Used | Our Coverage |
|----------|--------------|--------------|
| **Ecology/Population** | Lynx-Hare (1845-1935) | ✅ Covered |
| **Fluid Dynamics** | Cylinder wake, Sea surface temperature | ❌ Not covered |
| **Robotics** | UR10 robot arm, Rotary flexible joint | ❌ Not covered |
| **Video/Vision** | Pendulum video, Turbulent flow video | ❌ Not covered |
| **Climate/Atmosphere** | ERA5 reanalysis, Simulated atmospheric data | ❌ Not covered |
| **Biological** | Gene expression time series | ❌ Not covered |
| **Control Systems** | F8 aircraft, UAV flight data | ❌ Not covered |

---

## 1. Ecological/Population Dynamics

### Lynx-Hare Population Data (Most Common)
- **Source:** Hudson Bay Company fur trading records (1845-1935 or 1900-1920 subset)
- **Size:** 91 annual measurements
- **Expected Model:** Lotka-Volterra predator-prey dynamics
- **Used By:** Original SINDy (Brunton 2016), E-SINDy (Fasel 2022), many others
- **Our Status:** ✅ Currently used

### Recommendation
This is the standard real-world benchmark. We have it covered but should ensure our results are reproducible and competitive.

---

## 2. Fluid Dynamics

### Cylinder Wake (DNS/Experimental)
- **Source:** Direct numerical simulation or PIV measurements
- **Systems:** Single cylinder at Re=100, two-parallel cylinders, transient flows
- **Methods:** CNN-SINDy (autoencoder + SINDy), SINDy-SHRED
- **Used By:** Fukami et al. (JFM 2021), Loiseau & Brunton
- **Challenge:** High-dimensional → requires dimensionality reduction first
- **Data:** https://github.com/kfukami/CNN-SINDy-MLROM

### Sea Surface Temperature
- **Source:** Real-world oceanographic sensor measurements
- **Used By:** SINDy-SHRED (2025)
- **Application:** Mesoscale ocean closures, climate modeling

### Turbulent Flows
- **Source:** DNS of turbulence, nine-shear flow model
- **Challenge:** Many spatial modes required
- **Used By:** CNN-SINDy for turbulence closures

### Recommendation
Fluid dynamics is a major application area but requires high-dimensional handling (autoencoders). Consider adding a 2D reduced-order fluid example if feasible.

---

## 3. Robotics and Control

### UR10 Industrial Robot Arm
- **Source:** Experimental measurements from 6-DOF robot
- **Data:** Joint angles, velocities, torques
- **Application:** Dynamic model identification
- **Used By:** Springer Nonlinear Dynamics (2024)
- **Result:** SINDy successfully identified robot dynamics including payload and friction

### Rotary Flexible Joint
- **Source:** Laboratory experimental setup
- **Application:** Nonlinear dynamics identification
- **Used By:** SINDY-MPC validation

### UAV Flight Data
- **Source:** Real unmanned aerial vehicle experiments
- **Application:** Dynamics identification for MPC control
- **Used By:** MDPI Drones (January 2025)
- **Result:** SINDy achieved superior tracking accuracy vs other methods

### Recommendation
Robotics data would be a strong addition—it's practical, reproducible, and directly relevant to control applications.

---

## 4. Video and Vision Data

### Pendulum Video (Real Experimental)
- **Source:** 14-second recording of real pendulum
- **Application:** Discover equations + estimate gravity constant g
- **Used By:** Bayesian SINDy Autoencoders (2023)
- **Code:** https://github.com/gaoliyao/BayesianSindyAutoencoder (examples/pendulum_real_video)
- **Significance:** Demonstrates discovery from raw video, not processed trajectories

### Turbulent Flow Video
- **Source:** Direct video data of fluid flows
- **Used By:** SINDy-SHRED
- **Application:** Long-term video prediction with interpretable dynamics

### Recommendation
Video-based discovery is cutting-edge but requires autoencoder architectures (out of scope for our 2D focus).

---

## 5. Climate and Atmospheric Science

### ERA5 Reanalysis Data
- **Source:** ECMWF atmospheric reanalysis
- **Application:** Weather and climate PDE discovery
- **Used By:** WSINDy for Atmospheric Models (2025)

### Simulated Atmospheric Regimes
- **Types:** Barotropic vorticity, quasi-geostrophic, shallow water equations
- **Application:** Interpretable geophysics models
- **Used By:** WSINDy (Messenger & Bortz)

### Recommendation
Climate data could be interesting but may require PDE-SINDy capabilities beyond our current ODE focus.

---

## 6. Biological Systems

### Gene Expression Time Series
- **Source:** Single-cell RNA sequencing
- **Challenge:** Noisy, low-sample, high-dimensional
- **Status:** SINDy has had "limited" application to biological data
- **Used By:** Limited—mostly other GRN inference methods

### Biological Oscillator Data
- **Source:** Various experimental biological recordings
- **Challenges:** "insufficient resolution, noise, dimensionality, and limited prior knowledge"
- **Used By:** PMC paper on SINDy for biological oscillators (2024)

### Recommendation
Biological data is challenging but could be impactful. Gene regulatory networks typically require different approaches than standard SINDy.

---

## 7. Chemical and Industrial Systems

### Belousov-Zhabotinsky Reaction
- **Source:** Chemical oscillation experiments
- **Application:** Chaotic reaction dynamics
- **Used By:** SINDy-PI

### Continuous Direct Compression Tableting
- **Source:** Pharmaceutical manufacturing process
- **Application:** Process modeling
- **Used By:** 35th European Symposium on Computer Aided Process Engineering (2025)

### Diesel Engine Airpath
- **Source:** Engine sensor data
- **Application:** Discovery + multi-step simulation
- **Used By:** Yahagi et al. (March 2025)

---

## 8. Standard Synthetic Benchmarks

Most papers also validate on these synthetic systems:

| System | Dimension | Key Challenge |
|--------|-----------|---------------|
| Lorenz | 3D | Chaos, sensitivity |
| Van der Pol | 2D | Limit cycle |
| Duffing Oscillator | 2D | Nonlinearity |
| Double Pendulum | 4D | Chaos, multi-body |
| Hopf Bifurcation | 2D | Bifurcation |
| Lotka-Volterra | 2D | Ecological dynamics |
| Kuramoto-Sivashinsky | PDE | Spatiotemporal chaos |
| Reaction-Diffusion | PDE | Pattern formation |

---

## 9. Noise Robustness Benchmarks

Papers typically test with:
- **Noise levels:** 0%, 1%, 5%, 10%, 20%, sometimes up to 50%
- **Signal-to-noise ratios:** Down to SNR ≈ 1 (equal noise and signal)
- **Data amounts:** Low-data limit (tens to hundreds of points)
- **Weak SINDy:** Demonstrated recovery with noise ratios > 0.1

---

## 10. Gap Analysis for Our Project

### What We Have
- ✅ Lynx-Hare (standard ecology benchmark)
- ✅ Synthetic 2D systems (VanDerPol, Duffing, Lotka-Volterra, etc.)
- ✅ Noise robustness testing

### What We're Missing
1. **Second real-world dataset** - Most papers use at least 2 real datasets
2. **Experimental physics data** - Pendulum, oscillator, etc.
3. **Higher-dimensional validation** - 3D Lorenz is standard
4. **PDE validation** - Though out of scope for our 2D focus

### Recommended Additions (Priority Order)

1. **Damped Harmonic Oscillator (Experimental)**
   - Blog post: https://bea.stollnitz.com/blog/oscillator-pysindy/
   - Easy to collect with smartphone accelerometer
   - Ground truth known

2. **Double Pendulum Video**
   - Chaotic, challenging
   - Multiple existing datasets available
   - Tests robustness to chaos

3. **NOAA Climate Data (simplified)**
   - Temperature time series
   - Well-documented, freely available

4. **Electrocardiogram (ECG)**
   - Biological oscillator
   - PhysioNet has free datasets
   - Tests generalization to periodic biological signals

---

## References

1. [E-SINDy Paper (Fasel et al., 2022)](https://royalsocietypublishing.org/doi/10.1098/rspa.2021.0904)
2. [SINDy-SHRED (2025)](https://arxiv.org/abs/2501.13329)
3. [WSINDy (Messenger & Bortz, 2021)](https://epubs.siam.org/doi/10.1137/20M1343166)
4. [CNN-SINDy for Fluid Flows (Fukami et al., 2021)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/sparse-identification-of-nonlinear-dynamics-with-lowdimensionalized-flow-representations/B0A6BC75E087EE8F7B8100CF1185F29A)
5. [Bayesian SINDy Autoencoders](https://royalsocietypublishing.org/doi/10.1098/rspa.2023.0506)
6. [SINDy-MPC (Kaiser et al.)](https://royalsocietypublishing.org/doi/10.1098/rspa.2018.0335)
7. [PySINDy Documentation](https://pysindy.readthedocs.io/en/latest/examples/index.html)
8. [PySINDy Experimental Data Tutorial](https://bea.stollnitz.com/blog/oscillator-pysindy/)
9. [SINDy for Robot Dynamics (2024)](https://link.springer.com/article/10.1007/s11071-024-09526-7)
10. [WSINDy for Atmospheric Models (2025)](https://arxiv.org/html/2501.00738)
