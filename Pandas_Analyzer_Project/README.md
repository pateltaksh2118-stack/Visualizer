# Pandas Analyzer & Data Visualization

A menu-driven Python project for analyzing sales data stored in CSV files. It uses **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**.

## Features

- Load sales data from a CSV file
- Explore rows, columns, data types, and descriptive information
- Handle missing values
- Perform NumPy indexing, slicing, and numeric operations
- Search, filter, and sort records
- Combine and split DataFrames
- Calculate group summaries and statistics
- Create pivot tables
- Visualize data with bar, line, scatter, pie, histogram, stack, heatmap, and box plots
- Save summaries and visualizations to the `outputs/` folder
- Use object-oriented programming through the `SalesDataAnalyzer` class

## Project structure

```text
Pandas_Analyzer_Project/
├── main.py
├── requirements.txt
├── README.md
├── data/
│   └── sales_data.csv
└── outputs/
```

## Requirements

- Python 3.9 or newer
- pip

## Installation

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
python main.py
```

Choose an option from the main menu. The sample CSV is loaded by selecting option 1 and pressing Enter at the file-path prompt. You can also enter the path to your own CSV file.

## Expected CSV columns

The included sample file contains:

- `SalesID`
- `Date`
- `Product`
- `Region`
- `Sales`
- `Profit`
- `Year`

Your own dataset may use different column names. When using operations or charts, select column names that actually exist in your CSV. Numeric operations require numeric columns.

## Output

Generated CSV summaries, split datasets, and saved chart images are placed in the `outputs/` folder.

## Important note

The project modifies the dataset in memory during cleaning and DataFrame operations. Keep a backup of your original CSV if you need to preserve it. The included dataset is synthetic sample data for learning and demonstration.
