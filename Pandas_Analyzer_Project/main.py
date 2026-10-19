"""
Pandas Analyzer & Data Visualization
A menu-driven sales data analysis project using Pandas, NumPy,
Matplotlib, and Seaborn.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "sales_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


class SalesDataAnalyzer:
    """Load, clean, analyze, combine, split, and visualize sales data."""

    def __init__(self, file_path=None):
        self.data = pd.DataFrame()
        self.file_path = None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if file_path:
            self.load_data(file_path)

    def __del__(self):
        """No external resources need to be closed for this project."""
        pass

    def load_data(self, file_path):
        """Load a CSV file into a Pandas DataFrame."""
        try:
            path = Path(file_path).expanduser()
            self.data = pd.read_csv(path)
            self.file_path = path
            print(f"Dataset loaded successfully: {path}")
            print(f"Rows: {len(self.data)}, Columns: {len(self.data.columns)}")
            return True
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            print(f"Could not read the CSV file: {error}")
        except Exception as error:
            print(f"Error loading dataset: {error}")
        return False

    def _require_data(self):
        if self.data.empty:
            print("Please load a dataset first (menu option 1).")
            return False
        return True

    def explore_data(self):
        """Display common information about the dataset."""
        if not self._require_data():
            return

        while True:
            print("""
--- Explore Data ---
1. Display first 5 rows
2. Display last 5 rows
3. Display column names
4. Display data types
5. Display basic information
6. Display descriptive statistics
7. Return to main menu
""")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                print(self.data.head())
            elif choice == "2":
                print(self.data.tail())
            elif choice == "3":
                print(list(self.data.columns))
            elif choice == "4":
                print(self.data.dtypes)
            elif choice == "5":
                self.data.info()
            elif choice == "6":
                print(self.data.describe(include="all"))
            elif choice == "7":
                break
            else:
                print("Invalid choice. Please enter 1-7.")

    def clean_data(self):
        """Show missing values and offer common ways to handle them."""
        if not self._require_data():
            return

        while True:
            print("""
--- Handle Missing Data ---
1. Display rows containing missing values
2. Fill numeric missing values with each column's mean
3. Drop rows containing missing values
4. Replace missing values with a chosen value
5. Return to main menu
""")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                missing_rows = self.data[self.data.isna().any(axis=1)]
                if missing_rows.empty:
                    print("No missing values found in the dataset.")
                else:
                    print(missing_rows)
                    print("\nMissing values per column:\n", self.data.isna().sum())
            elif choice == "2":
                numeric_columns = self.data.select_dtypes(include=np.number).columns
                if len(numeric_columns) == 0:
                    print("There are no numeric columns to fill.")
                else:
                    for column in numeric_columns:
                        self.data[column] = self.data[column].fillna(self.data[column].mean())
                    print("Filled missing numeric values with column means.")
            elif choice == "3":
                before = len(self.data)
                self.data = self.data.dropna().reset_index(drop=True)
                print(f"Dropped {before - len(self.data)} row(s) containing missing values.")
            elif choice == "4":
                value = input("Enter the replacement value (text or number): ").strip()
                try:
                    replacement = float(value)
                    if replacement.is_integer():
                        replacement = int(replacement)
                except ValueError:
                    replacement = value
                self.data = self.data.fillna(replacement)
                print(f"Replaced missing values with {replacement!r}.")
            elif choice == "5":
                break
            else:
                print("Invalid choice. Please enter 1-5.")

    def mathematical_operations(self):
        """Perform simple numeric calculations and NumPy indexing/slicing."""
        if not self._require_data():
            return

        numeric_columns = list(self.data.select_dtypes(include=np.number).columns)
        if not numeric_columns:
            print("No numeric columns found.")
            return

        print("Numeric columns:", ", ".join(numeric_columns))
        column = input("Enter a numeric column: ").strip()
        if column not in numeric_columns:
            print("Column not found or is not numeric.")
            return

        values = self.data[column].dropna().to_numpy()
        if values.size == 0:
            print("The selected column has no numeric values.")
            return

        print(f"\nFirst 10 values as NumPy array:\n{values[:10]}")
        print(f"First value (index 0): {values[0]}")
        print(f"First five values (slice [0:5]): {values[0:5]}")
        print(f"Sum: {np.sum(values):.2f}")
        print(f"Mean: {np.mean(values):.2f}")
        print(f"Minimum: {np.min(values):.2f}")
        print(f"Maximum: {np.max(values):.2f}")

        try:
            factor = float(input("Enter a number to multiply this column by (or 1 to leave unchanged): "))
            new_column = f"{column}_adjusted"
            self.data[new_column] = self.data[column] * factor
            print(f"Created new column '{new_column}'.")
        except ValueError:
            print("Invalid number. No adjusted column was created.")

    def combine_data(self, other_dataframe=None):
        """Combine the current DataFrame with another CSV or DataFrame."""
        if not self._require_data():
            return
        if other_dataframe is None:
            path = input("Enter path to another CSV file: ").strip()
            try:
                other_dataframe = pd.read_csv(Path(path).expanduser())
            except Exception as error:
                print(f"Could not load second dataset: {error}")
                return

        print("""
1. Concatenate rows (same/similar columns)
2. Merge on a common column
""")
        choice = input("Choose combine method: ").strip()
        if choice == "1":
            self.data = pd.concat([self.data, other_dataframe], ignore_index=True, sort=False)
            print("Datasets concatenated successfully.")
        elif choice == "2":
            common = [column for column in self.data.columns if column in other_dataframe.columns]
            if not common:
                print("No common columns are available for merging.")
                return
            print("Common columns:", ", ".join(common))
            key = input("Enter the column to merge on: ").strip()
            if key not in common:
                print("Invalid merge column.")
                return
            try:
                self.data = pd.merge(self.data, other_dataframe, on=key, how="outer", suffixes=("_left", "_right"))
                print("Datasets merged successfully.")
            except Exception as error:
                print(f"Merge failed: {error}")
        else:
            print("Invalid choice.")

    def split_data(self):
        """Split data using a column value and save each group as a CSV."""
        if not self._require_data():
            return
        print("Available columns:", ", ".join(map(str, self.data.columns)))
        column = input("Enter the column to split by (e.g. Region or Product): ").strip()
        if column not in self.data.columns:
            print("Column not found.")
            return

        groups = self.data.groupby(column, dropna=False)
        for value, group in groups:
            safe_value = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value))
            output_path = OUTPUT_DIR / f"{column}_{safe_value}.csv"
            group.to_csv(output_path, index=False)
            print(f"Saved group {value!r}: {output_path}")
        print(f"Created {groups.ngroups} split file(s) in the outputs folder.")

    def search_sort_filter(self):
        """Search text, filter rows, and sort data."""
        if not self._require_data():
            return

        while True:
            print("""
--- Search, Sort, and Filter ---
1. Search for text in a column
2. Filter numeric column by minimum value
3. Sort by a column
4. Return to main menu
""")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                print("Columns:", ", ".join(map(str, self.data.columns)))
                column = input("Column to search: ").strip()
                if column not in self.data.columns:
                    print("Column not found.")
                    continue
                term = input("Text to find: ").strip()
                mask = self.data[column].astype("string").str.contains(term, case=False, na=False, regex=False)
                print(self.data[mask])
                print(f"Matching rows: {int(mask.sum())}")
            elif choice == "2":
                numeric_columns = list(self.data.select_dtypes(include=np.number).columns)
                print("Numeric columns:", ", ".join(numeric_columns) if numeric_columns else "None")
                column = input("Numeric column to filter: ").strip()
                if column not in numeric_columns:
                    print("Column not found or is not numeric.")
                    continue
                try:
                    minimum = float(input("Show rows where value is at least: "))
                    print(self.data[self.data[column] >= minimum])
                except ValueError:
                    print("Please enter a valid number.")
            elif choice == "3":
                print("Columns:", ", ".join(map(str, self.data.columns)))
                column = input("Column to sort by: ").strip()
                if column not in self.data.columns:
                    print("Column not found.")
                    continue
                order = input("Ascending order? (y/n): ").strip().lower()
                print(self.data.sort_values(column, ascending=(order != "n")).to_string(index=False))
            elif choice == "4":
                break
            else:
                print("Invalid choice. Please enter 1-4.")

    def aggregate_functions(self):
        """Group by a category and calculate sum, mean, and count."""
        if not self._require_data():
            return
        numeric_columns = list(self.data.select_dtypes(include=np.number).columns)
        if not numeric_columns:
            print("No numeric columns found.")
            return
        print("Columns:", ", ".join(map(str, self.data.columns)))
        group_column = input("Group by which column (e.g. Region or Product): ").strip()
        if group_column not in self.data.columns:
            print("Column not found.")
            return
        print("Numeric columns:", ", ".join(numeric_columns))
        value_column = input("Numeric column to summarize: ").strip()
        if value_column not in numeric_columns:
            print("Numeric column not found.")
            return
        summary = self.data.groupby(group_column, dropna=False)[value_column].agg(
            Sum="sum", Mean="mean", Count="count", Minimum="min", Maximum="max"
        )
        print(summary)
        save = input("Save summary to CSV? (y/n): ").strip().lower()
        if save == "y":
            path = OUTPUT_DIR / "aggregation_summary.csv"
            summary.to_csv(path)
            print(f"Summary saved to {path}")

    def statistical_analysis(self):
        """Calculate descriptive statistics, variance, standard deviation, and percentiles."""
        if not self._require_data():
            return
        numeric = self.data.select_dtypes(include=np.number)
        if numeric.empty:
            print("No numeric columns found.")
            return
        print("\nDescriptive statistics:\n", numeric.describe())
        print("\nVariance:\n", numeric.var())
        print("\nStandard deviation:\n", numeric.std())
        print("\n25th, 50th, and 75th percentiles:\n", numeric.quantile([0.25, 0.50, 0.75]))

    def create_pivot_table(self):
        """Create a pivot table using user-selected columns."""
        if not self._require_data():
            return
        print("Columns:", ", ".join(map(str, self.data.columns)))
        index_column = input("Pivot index column (e.g. Region): ").strip()
        value_column = input("Numeric values column (e.g. Sales): ").strip()
        if index_column not in self.data.columns:
            print("Index column not found.")
            return
        if value_column not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[value_column]):
            print("Values column must exist and be numeric.")
            return
        columns_input = input("Optional columns column (press Enter to skip): ").strip()
        columns_column = columns_input if columns_input else None
        if columns_column and columns_column not in self.data.columns:
            print("Columns field not found.")
            return
        pivot = pd.pivot_table(
            self.data,
            index=index_column,
            columns=columns_column,
            values=value_column,
            aggfunc="sum",
            fill_value=0,
        )
        print(pivot)
        if input("Save pivot table to CSV? (y/n): ").strip().lower() == "y":
            path = OUTPUT_DIR / "pivot_table.csv"
            pivot.to_csv(path)
            print(f"Pivot table saved to {path}")

    def visualize_data(self):
        """Create a chart using Matplotlib and Seaborn."""
        if not self._require_data():
            return
        numeric_columns = list(self.data.select_dtypes(include=np.number).columns)
        if not numeric_columns:
            print("No numeric columns found for plotting.")
            return

        while True:
            print("""
--- Data Visualization ---
1. Bar plot
2. Line plot
3. Scatter plot
4. Pie chart
5. Histogram
6. Stack plot
7. Seaborn heatmap
8. Seaborn box plot
9. Return to main menu
""")
            choice = input("Enter your choice: ").strip()
            try:
                if choice in {"1", "2"}:
                    print("Columns:", ", ".join(map(str, self.data.columns)))
                    x_column = input("Category/date column for x-axis: ").strip()
                    y_column = input("Numeric column for y-axis: ").strip()
                    if x_column not in self.data.columns or y_column not in numeric_columns:
                        print("Please select valid columns.")
                        continue
                    grouped = self.data.groupby(x_column, dropna=False)[y_column].sum().sort_index()
                    plt.figure(figsize=(10, 6))
                    if choice == "1":
                        plt.bar(grouped.index.astype(str), grouped.values)
                        title = f"{y_column} by {x_column} (Bar Plot)"
                    else:
                        plt.plot(grouped.index.astype(str), grouped.values, marker="o")
                        title = f"{y_column} by {x_column} (Line Plot)"
                    plt.xticks(rotation=45, ha="right")
                    plt.ylabel(y_column)
                    plt.title(title)
                elif choice == "3":
                    print("Numeric columns:", ", ".join(numeric_columns))
                    x_column = input("X-axis numeric column: ").strip()
                    y_column = input("Y-axis numeric column: ").strip()
                    if x_column not in numeric_columns or y_column not in numeric_columns:
                        print("Please select valid numeric columns.")
                        continue
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=self.data, x=x_column, y=y_column)
                    plt.title(f"{y_column} vs {x_column}")
                elif choice == "4":
                    print("Columns:", ", ".join(map(str, self.data.columns)))
                    category = input("Category column: ").strip()
                    value_column = input("Numeric column to total: ").strip()
                    if category not in self.data.columns or value_column not in numeric_columns:
                        print("Please select valid columns.")
                        continue
                    grouped = self.data.groupby(category)[value_column].sum()
                    if (grouped < 0).any() or grouped.sum() <= 0:
                        print("Pie charts require non-negative values with a positive total.")
                        continue
                    plt.figure(figsize=(8, 8))
                    plt.pie(grouped.values, labels=grouped.index.astype(str), autopct="%1.1f%%")
                    plt.title(f"{value_column} share by {category}")
                elif choice == "5":
                    print("Numeric columns:", ", ".join(numeric_columns))
                    column = input("Column for histogram: ").strip()
                    if column not in numeric_columns:
                        print("Please select a valid numeric column.")
                        continue
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data=self.data, x=column, kde=True)
                    plt.title(f"Distribution of {column}")
                elif choice == "6":
                    print("Numeric columns:", ", ".join(numeric_columns))
                    first = input("First numeric column: ").strip()
                    second = input("Second numeric column: ").strip()
                    if first not in numeric_columns or second not in numeric_columns:
                        print("Please select valid numeric columns.")
                        continue
                    x_values = np.arange(len(self.data))
                    plt.figure(figsize=(10, 6))
                    plt.stackplot(
                        x_values,
                        self.data[first].fillna(0).to_numpy(),
                        self.data[second].fillna(0).to_numpy(),
                        labels=[first, second],
                        alpha=0.8,
                    )
                    plt.legend(loc="upper left")
                    plt.xlabel("Row number")
                    plt.ylabel("Value")
                    plt.title(f"Stack Plot: {first} and {second}")
                elif choice == "7":
                    plt.figure(figsize=(10, 7))
                    sns.heatmap(self.data[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
                    plt.title("Correlation Heatmap")
                elif choice == "8":
                    print("Numeric columns:", ", ".join(numeric_columns))
                    column = input("Column for box plot: ").strip()
                    if column not in numeric_columns:
                        print("Please select a valid numeric column.")
                        continue
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(data=self.data, y=column)
                    plt.title(f"Box Plot of {column}")
                elif choice == "9":
                    break
                else:
                    print("Invalid choice. Please enter 1-9.")
                    continue

                plt.tight_layout()
                plt.show()
                plt.close()
            except (ValueError, TypeError, KeyError) as error:
                print(f"Could not create chart: {error}")
            except Exception as error:
                print(f"Unexpected plotting error: {error}")

    def save_visualization(self):
        """Save the current dataset's numeric correlation heatmap as a PNG."""
        if not self._require_data():
            return
        numeric = self.data.select_dtypes(include=np.number)
        if numeric.shape[1] < 1:
            print("No numeric columns available to visualize.")
            return
        filename = input("Enter output image filename (e.g. sales_heatmap.png): ").strip()
        if not filename:
            filename = "sales_heatmap.png"
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename += ".png"
        # Keep output inside the project's outputs folder.
        filename = Path(filename).name
        path = OUTPUT_DIR / filename

        try:
            plt.figure(figsize=(10, 7))
            if numeric.shape[1] >= 2:
                sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Sales Data Correlation Heatmap")
            else:
                sns.histplot(data=numeric, x=numeric.columns[0], kde=True)
                plt.title(f"Distribution of {numeric.columns[0]}")
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Visualization saved successfully: {path}")
        except Exception as error:
            plt.close("all")
            print(f"Could not save visualization: {error}")

    def main_menu(self):
        """Run the menu-driven program."""
        while True:
            print("""
========== Pandas Analyzer & Data Visualization ==========
1. Load Dataset
2. Explore Data
3. Perform DataFrame / NumPy Operations
4. Handle Missing Data
5. Search, Sort, and Filter
6. Aggregation Functions
7. Statistical Analysis
8. Create Pivot Table
9. Combine DataFrames
10. Split DataFrame
11. Data Visualization
12. Save Visualization
13. Exit
==========================================================
""")
            choice = input("Enter your choice: ").strip()
            if choice == "1":
                path = input(f"Enter CSV path (press Enter for {DEFAULT_CSV}): ").strip()
                self.load_data(path or DEFAULT_CSV)
            elif choice == "2":
                self.explore_data()
            elif choice == "3":
                self.mathematical_operations()
            elif choice == "4":
                self.clean_data()
            elif choice == "5":
                self.search_sort_filter()
            elif choice == "6":
                self.aggregate_functions()
            elif choice == "7":
                self.statistical_analysis()
            elif choice == "8":
                self.create_pivot_table()
            elif choice == "9":
                self.combine_data()
            elif choice == "10":
                self.split_data()
            elif choice == "11":
                self.visualize_data()
            elif choice == "12":
                self.save_visualization()
            elif choice == "13":
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 13.")


if __name__ == "__main__":
    analyzer = SalesDataAnalyzer()
    print("Welcome to the Pandas Analyzer & Data Visualization Project!")
    print(f"Sample dataset location: {DEFAULT_CSV}")
    analyzer.main_menu()
