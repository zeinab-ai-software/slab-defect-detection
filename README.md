# Slab Defect Detection using Multi-label Multi-class Classification

This project implements a machine learning model to **detect surface and internal defects in steel slabs**, based on process data collected from three key production stages in a steel factory:

- 🔥 **EAF** – Electric Arc Furnace  
- 🧪 **LF** – Ladle Furnace  
- 🌊 **CCM** – Continuous Casting Machine  

---

## 🎯 Project Objective

The goal is to **predict potential defects** in each slab using tabular features from production process logs.  
Since a single slab can exhibit **multiple defects simultaneously**, this is framed as a:

> ✅ **Multi-class Multi-label Classification problem**

---

## 🧱 Project Structure
slab-defect-detection/
│
├── main.py # Main entry point (data loading, training, evaluation)
├── model.py # MultiLabelClassifier class definition
├── utils.py # Data splitting and metric utilities
├── README.md # Project overview (this file)
├── requirements.txt # Required Python packages
└── data/ # CSV files (not included)

---

## ⚙️ How to Run

1. First, install dependencies:

```bash
pip install -r requirements.txt
```
2. Run the project:

```bash
python main.py
```
---

## 📊 Evaluation Metrics
This project uses the following metrics tailored for multi-label classification:

F1 Score (sample-based)
accuracy (specialized for multi-output models)
These metrics consider partial correctness and overlapping label sets.

---

## 🔒 Data Notice
⚠️ Due to the company's privacy policy, the original CSV files (Total_Table_Slabs.csv, SlabLabels.csv) are not shared in this repository.
If you need it, you can use dummy data or contact the author for collaboration.

---

## 👩‍💻 Author
Zeinab Sedighi
AI Engineer | Steel Industry | Industrial Predictive Analytics

📧 sedighi.63@gmail.com

📄 License
This project is intended for educational and research purposes. Unauthorized commercial use or redistribution of company data is prohibited. 
