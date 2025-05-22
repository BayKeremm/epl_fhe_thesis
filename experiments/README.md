# Experimental Study

This directory contains the implementation of all experiments described in **Chapter 5** of the thesis. The experiments are organized into two main categories based on the evaluation domain:

- **Plaintext Domain**: Evaluates the preprocessing pipeline before encryption.
- **Ciphertext Domain**: Evaluates the same pipeline under homomorphic encryption using the Concrete framework.

Performance is reported using the **True Positive Rate (TPR)** at fixed **False Positive Rate (FPR)** thresholds: **0.01%, 0.1%, and 1%**.

---

## Plaintext Domain

These experiments analyze how various preprocessing steps affect matching performance on unencrypted data. All evaluations use 10-fold cross-validation on the LFW dataset.

- `e0_baseline`  
  Establishes the baseline performance of 512-dimensional FaceNet embeddings without any preprocessing.  
  → Refer to Section 5.1.1.

- `e1_elbow_scree`  
  Computes and plots Elbow and Scree curves to analyze the intrinsic dimensionality of embeddings.  
  → Refer to Section 5.1.2.

- `e1_pca_eval`  
  Evaluates matching performance for various PCA-reduced dimensions, using 10-fold CV.  
  → Refer to Section 5.1.2.

- `e1_pca_trend_plot`  
  Visualizes the performance trend as a function of retained PCA components.  
  → Refer to Section 5.1.2.

- `e2_minmax_eval`  
  Adds min-max normalization after PCA. Compares **global** and **per-feature** strategies via 10-fold CV.  
  → Refer to Section 5.1.3.

- `e3_full`  
  Completes the plaintext pipeline by adding quantization on top of PCA and normalization. Reports final metrics.  
  → Refer to Section 5.1.4.

---

## Ciphertext Domain

These scripts evaluate the encrypted pipeline using the Concrete (TFHE) framework. Results include both accuracy and system performance characteristics.

- `enc_eval`  
  Performs 10-fold CV over various combinations of preprocessing strategies in the encrypted domain.
  → Refer to Section 5.2.2.

- `enc_timing`  
  Measures runtime performance of various preprocessing configurations, as reported in timing tables.  
  → See Section 5.2.3.

- `enc_no_minmax`  
  Evaluates a signed-integer pipeline that skips min-max normalization and applies global quantization directly.
  → Refer to Section 5.2.4.

- `memory_costs`  
  Computes memory consumption (e.g., ciphertext sizes) for different configurations.  
  → See Section 5.2.5.

- `enc_eval_p_error`  
  Benchmarks the impact of different programmable bootstrapping error probabilities (`p_error`) for the global-global pipeline.
  → Refer to Section 5.3.
