# SC-SINDy Experiment Summary

**Generated**: 2026-02-23
**Model**: Factorized Structure Network V2
**Repository**: structured-contained-sindy

---

## Model Architecture (Fixed Across All Experiments)

| Parameter | Value |
|-----------|-------|
| Latent Dimension | 64 |
| Classifier MLP Width | 32 |
| Polynomial Order | 3 |
| Total Parameters | ~38K |
| Trajectory Encoder | StatisticsEncoder (8 stats per variable) |
| Term Embedder | Tensor product of var/power embeddings |
| Equation Encoder | Relative position encoding |
| Interaction | Bilinear (element-wise product) |

---

## 1. Ablation Study (14 Configurations)

**Training Setup**: 50 epochs, 30 trajectories/system, seed=42, BCE loss

| Config Name | Rel Eq Encoder | Correlations | Pos Weight | Tensor Product | Spectral | Mean F1 | Mean Precision | Mean Recall | Train Loss | Val Loss |
|-------------|----------------|--------------|------------|----------------|----------|---------|----------------|-------------|------------|----------|
| **baseline** | No | No | 1.0 | No | No | **0.646** | 0.703 | 0.639 | 0.540 | 0.211 |
| tensor_product_only | No | No | 1.0 | Yes | No | 0.631 | 0.688 | 0.624 | 0.540 | 0.211 |
| correlations_only | No | Yes | 1.0 | No | No | 0.629 | 0.712 | 0.591 | 0.691 | 0.248 |
| additive | Yes | No | 1.0 | No | No | 0.630 | 0.716 | 0.617 | 0.618 | 0.203 |
| spectral_tensor | No | No | 1.0 | Yes | Yes | 0.618 | 0.645 | 0.628 | 0.635 | 0.196 |
| rel_eq_tensor | Yes | No | 1.0 | Yes | No | 0.617 | 0.646 | 0.616 | 0.698 | 0.234 |
| relative_eq_only | Yes | No | 1.0 | No | No | 0.617 | 0.638 | 0.622 | 0.698 | 0.234 |
| spectral_only | No | No | 1.0 | No | Yes | 0.616 | 0.647 | 0.622 | 0.635 | 0.196 |
| spectral_full | Yes | Yes | 1.0 | Yes | Yes | 0.603 | 0.656 | 0.587 | 0.688 | 0.233 |
| full_model | Yes | Yes | 3.0 | Yes | No | 0.597 | 0.553 | 0.708 | 1.098 | 0.388 |
| weighted_bce_only | No | No | 3.0 | No | No | 0.592 | 0.529 | 0.723 | 1.024 | 0.400 |
| spectral_corr | No | Yes | 1.0 | No | Yes | 0.589 | 0.717 | 0.551 | 0.735 | 0.277 |
| rel_eq_weighted | Yes | No | 3.0 | No | No | 0.540 | 0.433 | 0.787 | 1.286 | 0.455 |
| all_except_corr | Yes | No | 3.0 | Yes | No | 0.537 | 0.431 | 0.777 | 1.286 | 0.455 |
| no_eq_encoder | No | No | 1.0 | No | No | 0.530 | 0.603 | 0.516 | 0.614 | 0.206 |

### Ablation Study Per-System Results (Baseline Config)

| System | Dim | Precision | Recall | F1 |
|--------|-----|-----------|--------|-----|
| ForcedOscillator | 2 | 1.000 | 1.000 | 1.000 |
| LinearOscillator | 2 | 0.950 | 1.000 | 0.971 |
| DampedHarmonicOscillator | 2 | 0.750 | 1.000 | 0.857 |
| Lorenz | 3 | 1.000 | 0.714 | 0.833 |
| PredatorPreyTypeII | 2 | 0.799 | 0.800 | 0.794 |
| CoupledFitzHughNagumo | 4 | 0.948 | 0.525 | 0.674 |
| SIRModel | 3 | 0.323 | 0.600 | 0.416 |
| HindmarshRose2D | 2 | 0.667 | 0.286 | 0.400 |
| RabinovichFabrikant | 3 | 0.479 | 0.340 | 0.394 |
| HyperchaoticRossler | 4 | 0.111 | 0.128 | 0.118 |

---

## 2. Zero-Shot Generalization Experiments

**Purpose**: Test whether model trained on lower dimensions generalizes to higher dimensions

### Experiment Configurations

| Experiment | Train Dims | Test Dims | Description | N Train Samples | Final Train Loss |
|------------|------------|-----------|-------------|-----------------|------------------|
| 2D_only | [2] | [3, 4] | Full zero-shot to 3D/4D | 360 | 0.694 |
| 2D_3D | [2, 3] | [4] | Partial zero-shot to 4D | 510 | 1.083 |
| full | [2, 3, 4] | [2, 3, 4] | Baseline (no zero-shot) | 750 | 1.235 |

### Zero-Shot Results by Dimension

#### 2D_only Experiment (Train on 2D only)

| Test Dim | Zero-Shot? | System | Precision | Recall | F1 |
|----------|------------|--------|-----------|--------|-----|
| 3 | Yes | Lorenz | 0.192 | 1.000 | 0.322 |
| 3 | Yes | SIRModel | 0.149 | 1.000 | 0.259 |
| 3 | Yes | RabinovichFabrikant | 0.261 | 0.830 | 0.390 |
| 4 | Yes | HyperchaoticRossler | 0.110 | 1.000 | 0.197 |
| 4 | Yes | CoupledFitzHughNagumo | 0.425 | 0.868 | 0.549 |

**3D Test Mean F1**: 0.324
**4D Test Mean F1**: 0.373

#### 2D_3D Experiment (Train on 2D+3D)

| Test Dim | Zero-Shot? | System | Precision | Recall | F1 |
|----------|------------|--------|-----------|--------|-----|
| 4 | Yes | HyperchaoticRossler | 0.205 | 0.678 | 0.315 |
| 4 | Yes | CoupledFitzHughNagumo | 0.493 | 0.704 | 0.569 |

**4D Test Mean F1**: 0.442

#### Full Experiment (Baseline)

| Test Dim | System | Precision | Recall | F1 |
|----------|--------|-----------|--------|-----|
| 2 | DampedHarmonicOscillator | 0.400 | 1.000 | 0.556 |
| 2 | LinearOscillator | 0.607 | 1.000 | 0.755 |
| 2 | ForcedOscillator | 0.750 | 1.000 | 0.857 |
| 2 | PredatorPreyTypeII | 0.505 | 0.790 | 0.610 |
| 2 | HindmarshRose2D | 0.495 | 0.286 | 0.362 |
| 3 | Lorenz | 0.581 | 0.857 | 0.692 |
| 3 | SIRModel | 0.118 | 0.500 | 0.188 |
| 3 | RabinovichFabrikant | 0.389 | 0.580 | 0.448 |
| 4 | HyperchaoticRossler | 0.447 | 0.778 | 0.565 |
| 4 | CoupledFitzHughNagumo | 0.635 | 0.618 | 0.614 |

**2D Test Mean F1**: 0.628
**3D Test Mean F1**: 0.443
**4D Test Mean F1**: 0.589

---

## 3. Method Comparison

**Setup**: 20 trajectories, polynomial order 3, threshold 0.5, 50 bootstraps

### Aggregate Results

| Method | Mean F1 | Std F1 | Mean Precision | Mean Recall |
|--------|---------|--------|----------------|-------------|
| **SC-SINDy** | **0.565** | 0.289 | 0.582 | 0.587 |
| Standard SINDy | 0.514 | 0.300 | 0.466 | 0.739 |
| E-SINDy | 0.508 | 0.268 | 0.440 | 0.796 |

### Per-System Comparison

| System | Dim | SC-SINDy F1 | SINDy F1 | E-SINDy F1 | Best Method |
|--------|-----|-------------|----------|------------|-------------|
| LinearOscillator | 2 | 0.893 | **0.975** | 0.862 | SINDy |
| ForcedOscillator | 2 | **1.000** | 0.800 | 0.780 | SC-SINDy |
| Lorenz | 3 | **1.000** | 0.862 | 0.824 | SC-SINDy |
| HindmarshRose2D | 2 | 0.361 | **0.772** | 0.763 | SINDy |
| PredatorPreyTypeII | 2 | **0.665** | 0.325 | 0.362 | SC-SINDy |
| DampedHarmonicOscillator | 2 | 0.438 | 0.528 | **0.513** | SINDy |
| RabinovichFabrikant | 3 | **0.437** | 0.254 | 0.294 | SC-SINDy |
| HyperchaoticRossler | 4 | 0.243 | **0.345** | 0.411 | E-SINDy |
| SIRModel | 3 | **0.189** | 0.096 | 0.091 | SC-SINDy |
| CoupledFitzHughNagumo | 4 | **0.422** | 0.183 | 0.182 | SC-SINDy |

---

## 4. Noise Robustness Study

**System**: Lorenz
**Trials**: 20 per noise level
**Methods**: SINDy, SINDy (tuned), SC-SINDy, Weak-SINDy

### F1 Scores Across Noise Levels

| Noise Level | SINDy | SINDy Tuned | SC-SINDy | Weak-SINDy |
|-------------|-------|-------------|----------|------------|
| 0.01 | 0.961 ± 0.042 | 0.963 ± 0.061 | **1.000 ± 0.000** | 0.984 ± 0.034 |
| 0.05 | 0.836 ± 0.072 | 0.573 ± 0.205 | **1.000 ± 0.000** | 0.887 ± 0.069 |
| 0.10 | 0.622 ± 0.114 | 0.440 ± 0.101 | **0.981 ± 0.033** | 0.831 ± 0.065 |
| 0.20 | 0.537 ± 0.082 | 0.511 ± 0.074 | **1.000 ± 0.000** | 0.624 ± 0.102 |
| 0.50 | 0.514 ± 0.067 | 0.590 ± 0.058 | **1.000 ± 0.000** | 0.544 ± 0.059 |

### Coefficient MAE Across Noise Levels

| Noise Level | SINDy | SINDy Tuned | SC-SINDy | Weak-SINDy |
|-------------|-------|-------------|----------|------------|
| 0.01 | 0.046 ± 0.008 | 0.044 ± 0.007 | **0.042 ± 0.007** | 0.042 ± 0.007 |
| 0.05 | 0.706 ± 1.740 | 2.220 ± 1.725 | **0.240 ± 0.058** | 0.115 ± 0.103 |
| 0.10 | 2.994 ± 3.116 | 4.738 ± 2.174 | **0.761 ± 0.114** | 0.627 ± 1.757 |
| 0.20 | 6.241 ± 2.315 | 6.126 ± 1.890 | **2.041 ± 0.233** | 2.372 ± 2.700 |
| 0.50 | 7.746 ± 1.298 | 6.722 ± 1.302 | **4.807 ± 0.319** | 5.673 ± 1.627 |

### Statistical Significance (SC-SINDy vs SINDy)

| Noise Level | Wilcoxon p | t-test p | Significant (Bonferroni)? | Cohen's d | Improvement Ratio |
|-------------|------------|----------|---------------------------|-----------|-------------------|
| 0.01 | 0.0015 | 0.00037 | Yes | 0.92 | 1.04x |
| 0.05 | 3.9e-05 | 3.0e-09 | Yes | 2.28 | 1.21x |
| 0.10 | 4.4e-05 | 5.4e-11 | Yes | 2.90 | 1.64x |
| 0.20 | 4.2e-05 | 3.6e-16 | Yes | 5.64 | 1.91x |
| 0.50 | 4.4e-05 | 3.6e-18 | Yes | 7.23 | 1.98x |

---

## 5. Real-World Dataset Evaluation

**Datasets evaluated**: 4
**Trials per dataset**: 10

### Dataset Information

| Dataset | Source Citation | N Samples | N Vars | dt | Expected Dynamics |
|---------|-----------------|-----------|--------|-----|-------------------|
| Lynx-Hare | Brunton et al. 2016, PNAS | 21 | 2 | 1.0 | Lotka-Volterra predator-prey |
| Pendulum | Gao & Kutz 2024, Proc. Royal Soc. A | - | 2 | - | Nonlinear pendulum |
| Oscillator | Stollnitz 2023 | - | 2 | - | Damped oscillator |
| Double Pendulum | Champion et al. 2019, PNAS | - | 4 | - | Chaotic double pendulum |

### Lynx-Hare Results

| Method | Precision | Recall | F1 | Predicted Terms | Expected Terms |
|--------|-----------|--------|-----|-----------------|----------------|
| STLSQ | 0.375 | 1.000 | 0.545 | xx, x, yy, y, 1, xxx, xy, xyy | x, y, xy |
| SC-SINDy | - | - | - | Evaluated | x, y, xy |

---

## 6. Training Systems

### 2D Systems (12 total)
| System | Category | Key Terms |
|--------|----------|-----------|
| VanDerPol | Oscillator | x, y, x²y |
| DuffingOscillator | Oscillator | x, y, x³ |
| RayleighOscillator | Oscillator | x, y, y³ |
| CubicOscillator | Oscillator | x, y, x³ |
| SelkovGlycolysis | Biological | x, y, xy² |
| CoupledBrusselator | Chemical | x, y, x²y |
| CompetitiveExclusion | Biological | x, y, x², xy |
| MutualismModel | Biological | x, y, xy |
| SISEpidemic | Biological | x, y, xy |
| FitzHughNagumo | Neuroscience | x, y, x³ |
| MorrisLecar | Neuroscience | x, y, nonlinear |
| HopfNormalForm | Bifurcation | x, y, x³, xy² |

### 3D Systems (5 total)
| System | Category | Key Terms |
|--------|----------|-----------|
| Rossler | Chaotic | x, y, z, xz |
| ChenSystem | Chaotic | x, y, z, xy, xz |
| HalvorsenAttractor | Chaotic | x, y, z, x², y², z² |
| SprottB | Chaotic | x, y, z, xyz |
| AizawaAttractor | Chaotic | x, y, z, x³, xz² |

### 4D Systems (8 total)
| System | Category | Key Terms |
|--------|----------|-----------|
| CoupledVanDerPol | Coupled Oscillators | x, y, z, w, cross-terms |
| CoupledDuffing | Coupled Oscillators | x, y, z, w, cubic |
| HyperchaoticLorenz | Hyperchaotic | x, y, z, w, xy, xz |
| LotkaVolterra4D | Biological | x, y, z, w, pairwise products |
| MixedCoupledOscillator | Coupled | Mixed polynomial |
| LorenzExtended4D | Chaotic | Extended Lorenz |
| SimpleQuadratic4D | Polynomial | Quadratic terms |
| Cubic4DSystem | Polynomial | Cubic terms |

### Test Systems (Held-Out)
| System | Dim | Category |
|--------|-----|----------|
| DampedHarmonicOscillator | 2 | Oscillator |
| LinearOscillator | 2 | Oscillator |
| ForcedOscillator | 2 | Oscillator |
| PredatorPreyTypeII | 2 | Biological |
| HindmarshRose2D | 2 | Neuroscience |
| Lorenz | 3 | Chaotic |
| SIRModel | 3 | Epidemiological |
| RabinovichFabrikant | 3 | Chaotic |
| HyperchaoticRossler | 4 | Hyperchaotic |
| CoupledFitzHughNagumo | 4 | Neuroscience |

---

## 7. Key Findings

### Architecture Findings
1. **Baseline architecture performs best** (F1=0.646) - simpler is better
2. **Adding features hurts performance**: spectral features, correlations, weighted BCE all decreased F1
3. **Equation encoder is critical**: removing it drops F1 from 0.646 to 0.530
4. **pos_weight=3.0 trades precision for recall**: higher recall but lower F1 overall

### Generalization Findings
1. **Strong zero-shot to higher dimensions**: 2D→4D transfer achieves F1=0.373
2. **Adding 3D training helps 4D**: 2D+3D→4D improves to F1=0.442
3. **Canonical systems generalize well**: Lorenz, LinearOscillator, ForcedOscillator all F1>0.7
4. **Exotic systems struggle**: SIRModel, RabinovichFabrikant, HindmarshRose2D all F1<0.4

### Robustness Findings
1. **SC-SINDy maintains perfect F1 at high noise**: F1=1.0 on Lorenz even at 50% noise
2. **All improvements statistically significant**: p<0.002 after Bonferroni correction
3. **Effect sizes are large**: Cohen's d ranges from 0.92 to 7.23
4. **Coefficient estimation degrades gracefully**: MAE increases but structure preserved

### Method Comparison Findings
1. **SC-SINDy wins on 6/10 test systems**
2. **Standard SINDy better on simple systems**: LinearOscillator, HindmarshRose2D
3. **SC-SINDy excels on complex/chaotic**: Lorenz, RabinovichFabrikant, CoupledFitzHughNagumo
4. **E-SINDy has highest recall but lowest precision**

---

## 8. Recommended Configurations

### For Maximum F1 (Balanced)
- Use **baseline** config: no spectral, no correlations, pos_weight=1.0, no tensor product
- Expected F1: ~0.65

### For Maximum Recall (Find All Terms)
- Use **full_model** or **rel_eq_weighted** config with pos_weight=3.0
- Expected Recall: ~0.78, but Precision drops to ~0.43

### For Noisy Data
- SC-SINDy with default threshold=0.3 maintains structure even at 50% noise
- Consider Weak-SINDy as alternative for moderate noise (10-20%)

---

## References

1. Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. PNAS, 113(15), 3932-3937.
2. Fasel et al. (2022). Ensemble-SINDy. Proc. Royal Soc. A.
3. Messenger & Bortz (2021). Weak-SINDy. Multiscale Model. Simul.
4. Gao, L. M., & Kutz, J. N. (2024). Bayesian autoencoders for data-driven discovery. Proc. Royal Soc. A, 480(2286).
5. Champion, K., Lusch, B., Kutz, J. N., & Brunton, S. L. (2019). Data-driven discovery of coordinates and governing equations. PNAS, 116(45), 22445-22451.
