# Causality Final Project

This repository contains code for running all experiments from **Skills, Signals, or Spuriousness?** 

### 📁 IV Directory
- **`IV.ipynb`**: Complete instrumental variables analysis

### 📁 MediationAnalysis Directory
- **`1989.csv`**: cleaned NLSY79 dataset with variables:

- **`nls.py`**: Basic mediation analysis

- **`nls_adjustment.py`**: Mediation analysis with adjustment for ability

### Running the Analysis

1. **Instrumental Variables**:
   ```bash
   cd IV/
   jupyter notebook IV.ipynb       
   ```

2. **Mediation Analysis**:
   ```bash
   cd MediationAnalysis/
   python nls.py                    
   python nls_adjustment.py         
   ```

