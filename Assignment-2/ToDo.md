# Assignment 2 To-Do List

.\run_all.ps1
python PA2_code/check_status.py
python PA2_code/generate_plots.py
python PA2_code/generate_report.py
python PA2_code/package_submission.py


This document outlines the steps you need to take to complete and submit Programming Assignment 2.

## 1. Monitor Training Runs
I am currently running the required experiments (A1-A5, B1-B2). You can monitor the progress in the terminal.
- Each run will generate a `<name>_final_state.npz` file and a log in the console.
- **Action:** Ensure all runs complete successfully. Do not close the terminal until they are done.

## 2. Generate Plots
Once the `.npz` files are generated, you need to use the provided plotting snippet (mentioned in the assignment PDF/ReadMe) to create the visualization for your report.
- **Required Plots for each run:**
    - Histogram of $\log_{10}(g_t)$ (Gradient through time)
    - Saturation histograms (Hidden state $d(h)$ and GRU gates $d(v)$)
    - Validation error curves
- **Action:** Create a Python script (e.g., `plot_results.py`) to generate these figures.

## 3. Write the Report (PDF)
This is the most critical part of your submission. You need to analyze the training dynamics.
- **A1 vs A2/A3:** How does gradient clipping affect the stability and convergence?
- **RNN vs GRU (A1 vs A4):** Compare the vanishing gradient behavior. Look at the $\log_{10}(g_t)$ histograms.
- **Saturation Analysis:** Identify if the units are saturating and how it affects learning.
- **Spectral Radius:** Discuss the behavior of $\rho(W_{hh})$ over training.
- **Action:** Compile your analysis and plots into a PDF.

## 4. AI Tool Disclosure (Mandatory)
The assignment has a strict policy regarding AI.
- **Action:** Document any conceptual help you received. Since I assisted you in the implementation, you must include the link to this chat session in your submission as required by Section 10.6 of the ReadMe.
- **Note:** The ReadMe states "You may NOT generate code using AI". You should review the code I wrote in `model.py` and `train.py` and ensure you fully understand it as your own work.

## 5. Submission Checklist
Prepare a ZIP file containing:
- [ ] `model.py` and `train.py`
- [ ] All `.npz` files (A1 to A5, B1 and B2)
- [ ] Raw training logs (copy-pasted from terminal or redirected to files)
- [ ] `commands.txt` (listing exact commands used)
- [ ] Your Report PDF
- [ ] AI Disclosure link/file

---
**Current Status:** Running Experiment A1.
