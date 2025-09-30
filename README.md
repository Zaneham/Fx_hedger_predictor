#  Currency Hedger Predictor

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen)](https://currency-hedger-predictor-npmezsnuu5qvepbesbjunn.streamlit.app/).

This repository contains my experiments with **Long Short-Term Memory (LSTM)** networks for foreign exchange (FX) prediction.  
The goal is to explore how deep learning can be applied to time series forecasting in financial markets, and to build reproducible results that can be extended to live trading or hedging strategies.  

I’m currently using this as part of a side case study analysis I’ll be publishing. I’m constantly learning, so any feedback is appreciated!

---

##  Overview
This project applies LSTM models to FX data with the aim of predicting exchange rate movements.  
It focuses on:
- **Reproducibility** of experiments  
- **Automation** of workflows  
- **Integration** of external features such as sentiment and macroeconomic indicators  

---

##  Features
- Multivariate LSTM model for FX rate prediction  
- Integration of price data with external features (e.g., sentiment, macro indicators)  
- Automated logging and evaluation of predictions vs. actual outcomes  
- Configurable workflow using `.env` and YAML for reproducibility  
- Backtesting framework to benchmark model performance  
- Interactive **Streamlit dashboard** for visualization and exploration  

---

##  Tech Stack
- **Python** (TensorFlow / Keras, NumPy, Pandas, Scikit-learn)  
- **YAML** & `.env` for configuration management  
- **Matplotlib** for plotting results  
- **Streamlit** for a user-friendly app  

---

##  Live App
--> Try the interactive dashboard here:  
**[NZD-USD Hedger Predictor on Streamlit](https://nzd-usd-hedger-predictor.streamlit.app/)**  

---

##  Contributing
Feedback and collaboration are welcome!  
If you have ideas for improving the model or dashboard, feel free to open an issue or submit a pull request.  

---

##  Contact
- **GitHub**: [Zaneham](https://github.com/Zaneham)  
- **LinkedIn**: [Zane Hambly](https://www.linkedin.com/in/zane-hambly-11a69b166/)  
