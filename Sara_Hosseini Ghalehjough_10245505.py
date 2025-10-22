# ==============================================================================
# DLMDSPWP01 Final Project – Full Fixed Version (mocked unit tests)
# Author: Sara Hosseini Ghalehjough
# Matriculation: 10245505
# ==============================================================================

import os
import io
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, String, Integer
from sqlalchemy.orm import declarative_base
from bokeh.plotting import figure, output_file, save
from bokeh.models import Legend
import math
import unittest

# -------------------------------------------------------------------------
# SQLAlchemy ORM
# -------------------------------------------------------------------------
Base = declarative_base()


class TrainingData(Base):
    """ORM model for training data (y1-y4)."""
    __tablename__ = 'training_data'
    x = Column(Float, primary_key=True)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)


class IdealFunctions(Base):
    """ORM model for ideal functions (y1-y50)."""
    __tablename__ = 'ideal_functions'
    x = Column(Float, primary_key=True)

for i in range(1, 51):
    setattr(IdealFunctions, f'y{i}', Column(Float))


class TestResults(Base):
    """ORM model for mapping results."""
    __tablename__ = 'test_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float)
    y = Column(Float)
    deviation = Column(Float)
    ideal_func_no = Column(String)


# -------------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------------
class DataProcessingError(Exception):
    """Base exception for data processing."""
    pass


class DataLoadingError(DataProcessingError):
    """Raised when CSV loading fails."""
    pass


# -------------------------------------------------------------------------
# Core classes
# -------------------------------------------------------------------------
class DataProcessor:
    """Loads CSV files with validation."""
    def __init__(self, train_path, ideal_path, test_path):
        self.train_path = train_path
        self.ideal_path = ideal_path
        self.test_path = test_path
        self.df_train = None
        self.df_ideal = None
        self.df_test = None

    def load_csv_data(self):
        """Read the three CSV files."""
        try:
            self.df_train = pd.read_csv(self.train_path, sep=None, engine='python')
            self.df_ideal = pd.read_csv(self.ideal_path, sep=None, engine='python')
            self.df_test = pd.read_csv(self.test_path, sep=None, engine='python')
        except Exception as e:
            raise DataLoadingError(f"Error loading CSVs: {e}")


class ModelSelector(DataProcessor):
    """Selects ideal functions and maps test points."""
    def __init__(self, train_path, ideal_path, test_path):
        super().__init__(train_path, ideal_path, test_path)
        self.selected_models = {}
        self.mapping_details = {}
        self.df_results = None

    # ---------- ideal function selection ----------
    def select_ideal_functions(self):
        """Choose the 4 ideal functions with minimal SSE."""
        df_ideal = self.df_ideal[self.df_ideal['x'].isin(self.df_train['x'])].set_index('x')
        df_train = self.df_train.set_index('x')

        for col in [f'y{i}' for i in range(1, 5)]:
            best_err = math.inf
            best_ideal = None
            for i in range(1, 51):
                ideal_col = f'y{i}'
                if ideal_col not in df_ideal.columns:
                    continue
                err = np.sum((df_train[col] - df_ideal[ideal_col]) ** 2)
                if err < best_err:
                    best_err = err
                    best_ideal = ideal_col
            self.selected_models[col] = {'ideal_col': best_ideal, 'sq_error': best_err}

    # ---------- tolerance ----------
    def calculate_tolerance(self):
        """Tolerance = √2 × max deviation for each chosen ideal function."""
        df_ideal = self.df_ideal[self.df_ideal['x'].isin(self.df_train['x'])].set_index('x')
        df_train = self.df_train.set_index('x')
        for train_col, info in self.selected_models.items():
            ideal_col = info['ideal_col']
            max_dev = np.abs(df_train[train_col] - df_ideal[ideal_col]).max()
            tol = math.sqrt(2) * max_dev
            self.mapping_details[ideal_col] = {'train_col': train_col, 'tolerance': tol}

    # ---------- mapping ----------
    def map_test_data(self):
        """Map each test point to the best ideal function (if inside tolerance)."""
        df_res = self.df_test[['x', 'y']].copy()
        df_res['ideal_func_no'] = 'No match'
        df_res['deviation'] = np.nan
        df_res['id'] = range(1, len(df_res) + 1)

        ideal_idx = self.df_ideal.set_index('x')
        for i, row in df_res.iterrows():
            x, y = row['x'], row['y']
            if x not in ideal_idx.index:
                continue
            best_dev = math.inf
            best_func = None
            for ideal_col, info in self.mapping_details.items():
                if ideal_col not in ideal_idx.columns:
                    continue
                dev = abs(y - ideal_idx.at[x, ideal_col])
                if dev <= info['tolerance'] and dev < best_dev:
                    best_dev = dev
                    best_func = ideal_col
            if best_func:
                df_res.at[i, 'ideal_func_no'] = best_func
                df_res.at[i, 'deviation'] = best_dev

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result.csv')
        df_res.to_csv(out_path, index=False)
        print(f"Result file saved: {out_path}")
        self.df_results = df_res


class DatabaseHandler:
    """Writes DataFrames to SQLite."""
    def __init__(self, db_path='assignment_db.sqlite'):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)

    def insert_data(self, df, table_name):
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)


class Visualization:
    """Bokeh interactive plot."""
    def __init__(self, df_train, df_ideal, df_results, selected_models):
        self.df_train = df_train
        self.df_ideal = df_ideal
        self.df_results = df_results
        self.selected_models = selected_models

    def plot_data(self, outfile='visualization.html'):
        output_file(outfile, title="Data Mapping Visualization")
        p = figure(title="Training vs Ideal vs Test",
                   x_axis_label='X', y_axis_label='Y',
                   width=900, height=500)

        # distinctive colors for training sets
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        legend_items = []

        for i, col in enumerate([f'y{i}' for i in range(1, 5)]):
            if col in self.df_train.columns:
                r = p.scatter(self.df_train['x'], self.df_train[col],
                              size=5, color=colors[i], alpha=0.6)
                legend_items.append((f"Train {col}", [r]))

        # ideal functions (black line)
        for train_col, info in self.selected_models.items():
            ic = info['ideal_col']
            if ic in self.df_ideal.columns:
                r = p.line(self.df_ideal['x'], self.df_ideal[ic],
                           line_width=2, color="black")
                legend_items.append((f"Ideal {ic} (for {train_col})", [r]))

        # mapped test points
        mapped = self.df_results[self.df_results['ideal_func_no'] != 'No match']
        if not mapped.empty:
            r = p.scatter(mapped['x'], mapped['y'], size=7,
                          marker="x", color="purple")
            legend_items.append(("Mapped Test", [r]))

        # unmapped test points
        unmapped = self.df_results[self.df_results['ideal_func_no'] == 'No match']
        if not unmapped.empty:
            r = p.scatter(unmapped['x'], unmapped['y'], size=5,
                          marker="circle", alpha=0.4, color="gray")
            legend_items.append(("Unmapped Test", [r]))

        legend = Legend(items=legend_items)
        p.add_layout(legend, 'right')
        p.legend.click_policy = "hide"
        save(p)
        print(f"Visualization saved to {outfile}")


# -------------------------------------------------------------------------
# UNIT TESTS – work with **in-memory** CSV strings (no file needed)
# -------------------------------------------------------------------------
class TestModelSelector(unittest.TestCase):
    """All tests run on tiny synthetic data."""

    def setUp(self):
        # ---- synthetic train.csv ----
        train_csv = """x,y1,y2,y3,y4
0,0,0,0,0
1,1,2,3,4
2,4,3,2,1
"""
        # ---- synthetic ideal.csv (only 4 ideal functions for simplicity) ----
        ideal_csv = """x,y1,y2,y3,y4
0,0,0,0,0
1,1,2,3,4
2,4,3,2,1
"""
        # ---- synthetic test.csv ----
        test_csv = """x,y
0,0.1
1,1.2
2,4.0
"""
        # create ModelSelector that reads from StringIO
        self.model = ModelSelector(
            train_path=io.StringIO(train_csv),
            ideal_path=io.StringIO(ideal_csv),
            test_path=io.StringIO(test_csv)
        )
        # overload load_csv_data to read from StringIO
        def mocked_load():
            self.model.df_train = pd.read_csv(self.model.train_path)
            self.model.df_ideal = pd.read_csv(self.model.ideal_path)
            self.model.df_test  = pd.read_csv(self.model.test_path)
        self.model.load_csv_data = mocked_load
        self.model.load_csv_data()

    def test_load_csv(self):
        self.assertFalse(self.model.df_train.empty)
        self.assertFalse(self.model.df_ideal.empty)
        self.assertFalse(self.model.df_test.empty)

    def test_select_ideal_functions(self):
        self.model.select_ideal_functions()
        self.assertEqual(len(self.model.selected_models), 4)

    def test_map_test_data(self):
        self.model.select_ideal_functions()
        self.model.calculate_tolerance()
        self.model.map_test_data()
        self.assertFalse(self.model.df_results.empty)
        self.assertIn('ideal_func_no', self.model.df_results.columns)


# -------------------------------------------------------------------------
# MAIN EXECUTION (uses real files in the script folder)
# -------------------------------------------------------------------------
def main_execution():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN = os.path.join(BASE_DIR, 'train.csv')
    IDEAL = os.path.join(BASE_DIR, 'ideal.csv')
    TEST  = os.path.join(BASE_DIR, 'test.csv')

    try:
        model = ModelSelector(TRAIN, IDEAL, TEST)
        model.load_csv_data()
        model.select_ideal_functions()
        model.calculate_tolerance()
        model.map_test_data()

        db = DatabaseHandler(os.path.join(BASE_DIR, 'assignment_db.sqlite'))
        db.insert_data(model.df_train, 'training_data')
        db.insert_data(model.df_ideal, 'ideal_functions')
        db.insert_data(model.df_results, 'test_results')

        viz = Visualization(model.df_train, model.df_ideal,
                            model.df_results, model.selected_models)
        viz.plot_data(os.path.join(BASE_DIR, 'visualization.html'))

        # ----- run unit tests -----
        print("\n=== Running unit tests ===")
        unittest.main(module=__name__, exit=False, verbosity=2)

        print("\nAll steps completed successfully.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main_execution()