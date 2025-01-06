# Special Topics in Finance - Assignments Repository

This repository contains two distinct assignments completed as part of the **Special Topics in Finance** course. Each assignment explores unique aspects of financial analysis and econometrics, employing advanced statistical and econometric techniques. Below is a summary of each assignment and their respective contents.

---

## Assignment 1: Analyzing Oil and Natural Gas Volatility  
**Directory:** `Assignment_1/`  
This assignment is divided into two parts, each contained in its respective subdirectory: `Part_A` and `Part_B`.  

### **Part A: Relationship Between Oil and Natural Gas Volatility**
In this section, we investigate the interplay between oil and natural gas prices, with a specific focus on how natural gas price volatility influences oil price volatility. Key highlights include:  
- **Data**: 10 years of daily price data for oil and natural gas.  
- **Analysis**: Computation of monthly annualized volatilities for both commodities.  
- **Model**: An ARMA(1,1) model that captures both autoregressive and moving average dynamics.  
- **Findings**: Insights into volatility interactions, providing valuable information for risk management strategies in the energy sector.  
- **Limitations**: Results are to be interpreted as general guidelines rather than definitive conclusions due to the scope and purpose of the assignment.  

### **Part B: Forecasting Oil Price Volatility**
In this part, we forecast oil price volatility using different statistical models:  
- **Data Division**: Training (first 100 observations) and out-of-sample forecasting (last 20 observations).  
- **Models**:  
  - Autoregressive (AR(1)) model.  
  - Autoregressive with Exogenous Variables (AR-X(1)) model, incorporating natural gas volatility as a predictor.  
  - No-Change (NC) model for baseline comparison.  
- **Evaluation**: Mean Absolute Error (MAE) for 1-step, 2-step, and 3-step ahead predictions.  
- **Results**: The AR(1) model demonstrates the best overall performance, particularly at the 1-step prediction horizon, with a strong visual alignment between predictions and actual values.  

**Contents in `Part_A` and `Part_B` directories:**  
- **Datasets**: Raw data used for analysis.  
- **Code**: Scripts implementing the models and producing the results.  
- **PDFs**: Comprehensive reports detailing the methodology, results, and conclusions.

---

## Assignment 2: Macroeconomic Determinants of Default Risk and Term Structure  
**Directory:** `Assignment_2/`  

This assignment examines the effects of macroeconomic variables on Greece’s default risk premium and term structure over the period 2013-2024. The analysis is conducted across two distinct periods to capture variations during financial crises and non-crisis times.  

### Key Highlights:  
- **Framework**: Inspired by the three latent factors of the yield curve (level, slope, curvature).  
- **Variables**: Macro determinants drawn from prior research, such as Evans and Marshall (2007) and Paccagnini (2016).  
- **Modeling Approach**:  
  - Estimation of regime-dependent effects of macroeconomic variables.  
  - Use of a linear proxy for the term structure, regressing macro determinants directly on the term premium.  
- **Structure**: Includes a literature review, data presentation, model construction, result estimation, and concluding remarks.  
- **Findings**: Validation of prior literature regarding the influence of macro-variables on Greece’s default risk premium, with distinctions observed across different financial regimes.  

**Contents in `Assignment_2` directory:**  
- **Datasets**: Relevant data used for model estimation.  
- **Code**: Implementation of the term structure and default risk models.  
- **PDF**: Full report outlining the methodology, analysis, and key results.  

---

This repository showcases the application of econometric techniques to real-world financial problems, highlighting the interplay between macroeconomic indicators and financial asset dynamics. It is structured for clarity, with all necessary files organized within their respective directories.

---  
