#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch aggregator for .prep files -> Final regression -> Coeff output -> Plot
Uses numpy.linalg.lstsq (no sklearn).
"""

import glob
import os
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot 

class STCMBatchRegressor:
    """
    Reads multiple .prep files from a folder, aggregates them, 
    then performs linear regression (dh -> rpyf).

    .prep files have 7 columns: [Hx, Hy, Hz, 1, rpyX, rpyY, rpyZ].
        - The first 4 columns (Hx, Hy, Hz, 1) are the 'features' (X data),
        - The last 3 columns (rpyf) are the 'targets' (Y data).

    Attributes
    ----------
    folder_path : str
        Path to the folder containing .prep files.
    outname : str
        Base name for saving output (e.g. 'GS24').
    data_aggregate : np.ndarray
        Aggregated data from all .prep files, shape (N,7).
    dh : np.ndarray
        The feature matrix (N,4).
    rpyf : np.ndarray
        The target matrix (N,3).
    coeff : np.ndarray
        The fitted coefficients (shape (3,3)) if you separate out intercept.
    intercept : np.ndarray
        The fitted intercept (shape (3,)) if you separate it out from Beta.
    score : float
        Mean R^2 across the 3 outputs (same logic as in the earlier example).
    y_fit : np.ndarray
        The fitted values for rpyf (shape (N,3)) after applying the model.
    """

    def __init__(self, folder_path, outname="GS24"):
        self.folder_path = folder_path
        self.outname = outname

        # Data placeholders
        self.data_aggregate = None
        self.dh = None  # shape (N,4)
        self.rpyf = None  # shape (N,3)

        # Regression results
        self.coeff = None        # shape (3,3)
        self.intercept = None    # shape (3,)
        self.score = None
        self.y_fit = None        # shape (N,3)

    def load_prep_files(self):
        """
        Load all .prep files in self.folder_path, aggregate into a single array.
        .prep has 7 columns: [Hx, Hy, Hz, 1, rpyX, rpyY, rpyZ].
        """
        # Collect .prep files
        pattern = os.path.join(self.folder_path, "*.prep")
        files = glob.glob(pattern)

        if not files:
            print("No .prep files found in", self.folder_path)
            return

        data_list = []
        for file in files:
            print("Reading:", file)
            df = pd.read_csv(file, header=None, sep=' ', engine='python')
            # shape is (#rows, 7)
            data_arr = df.values.astype(float)
            data_list.append(data_arr)

        # Concatenate all
        if data_list:
            self.data_aggregate = np.vstack(data_list)
            print("Aggregate shape:", self.data_aggregate.shape)
        else:
            print("No data aggregated.")

        # Slice into dh (cols 0:4) and rpyf (cols 4:7)
        if self.data_aggregate is not None and self.data_aggregate.shape[1] == 7:
            self.dh = self.data_aggregate[:, 0:4]  # (Hx, Hy, Hz, 1)
            self.rpyf = self.data_aggregate[:, 4:7]  # (rpyX, rpyY, rpyZ)

    def fit_regression(self):
        """
        Fit a linear model rpyf ~ dh using numpy.linalg.lstsq.
        That means Y = X*Beta  => Beta = (X^T X)^(-1) (X^T Y).
        
        We'll also compute the R^2 (coefficient of determination).
        """
        if self.dh is None or self.rpyf is None:
            raise ValueError("No data to fit. Did you call load_prep_files()?")

        # X => shape (N,4)
        # Y => shape (N,3)
        X = self.dh
        Y = self.rpyf

        # Solve
        Beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        slope = Beta[0:3, :].T      # shape(3,3)
        intercept = Beta[3, :]      # shape(3,)

        self.coeff = slope
        self.intercept = intercept

        # Compute fitted Y
        Yhat = X @ Beta  # shape (N,3)
        self.y_fit = Yhat

        # Compute R^2 for each output, then average
        r2_list = []
        for j in range(Y.shape[1]):
            y_true = Y[:, j]
            y_pred = Yhat[:, j]
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
            r2_list.append(r2)
        self.score = np.mean(r2_list)
        print(f"Fit done. R^2 = {self.score:.4f}")

    def save_coefficients(self):
        """
        Save the resulting slope (coeff) and intercept to .npy files 
        or any desired format.
        """
        if self.coeff is None or self.intercept is None:
            print("No coefficients to save (did you run fit_regression?)")
            return

        # Construct output paths
        out_coeff = os.path.join(self.folder_path, self.outname + "coeff.npy")
        out_intercept = os.path.join(self.folder_path, self.outname + "coeff_intercept.npy")

        np.save(out_coeff, self.coeff)
        np.save(out_intercept, self.intercept)

        print("Saved coeff to:", out_coeff)
        print("Saved intercept to:", out_intercept)

    def print_results(self):
        """
        Print final fitting coefficients and R^2, etc.
        """
        if self.coeff is not None:
            print("Coefficients (shape 3x3):\n", self.coeff)
        if self.intercept is not None:
            print("Intercept (3,):\n", self.intercept)
        if self.score is not None:
            print("R^2 score:", self.score)

    def make_plot(self):
        """
        Create and save an offline Plotly figure comparing the fitted values vs the original.
        Also includes total field Ht vs HtIGRF.
        """
        if self.y_fit is None or self.rpyf is None:
            print("No fitted data to plot. Run fit_model() first.")
            return

        tr = np.arange(self.rpyf.shape[0])
        yyy = self.y_fit
        rpyf = self.rpyf

        Ht = np.sqrt(np.sum(yyy**2, axis=1))
        HtIGRF = np.sqrt(np.sum(rpyf**2, axis=1))

        fig = make_subplots(rows=4, cols=1)

        fig.add_trace(go.Scatter(x=tr, y=yyy[:, 0], mode='lines',
                                name='Hx with coeff', opacity=0.4),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=tr, y=rpyf[:, 0], mode='lines',
                                name='Hx-IGRF', opacity=0.4),
                    row=1, col=1)

        fig.add_trace(go.Scatter(x=tr, y=yyy[:, 1], mode='lines',
                                name='Hy with coeff', opacity=0.4),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=tr, y=rpyf[:, 1], mode='lines',
                                name='Hy-IGRF', opacity=0.4),
                    row=2, col=1)

        fig.add_trace(go.Scatter(x=tr, y=yyy[:, 2], mode='lines',
                                name='Hz with coeff', opacity=0.4),
                    row=3, col=1)
        fig.add_trace(go.Scatter(x=tr, y=rpyf[:, 2], mode='lines',
                                name='Hz-IGRF', opacity=0.4),
                    row=3, col=1)

        fig.add_trace(go.Scatter(x=tr, y=Ht, mode='lines',
                                name='Ht with coeff', opacity=0.4),
                    row=4, col=1)
        fig.add_trace(go.Scatter(x=tr, y=HtIGRF, mode='lines',
                                name='Ht-IGRF', opacity=0.4),
                    row=4, col=1)

        fig.update_layout(height=1000, title="Comparison: Fitted vs IGRF (Aggregated)")

        if hasattr(self, "folder_path") and hasattr(self, "outname"):
            out_html = os.path.join(self.folder_path, self.outname + "_coeff_plot.html")
        else:
            out_html = "coeff_plot.html"  # fallback

        plotly_plot(fig, filename=out_html, auto_open=False)
        print(f"Saved plot to: {out_html}")

    def get_output_paths(self):
        """
        Return absolute paths of the saved coefficient and intercept .npy files.
        Useful for chaining into STCMConverter.
        """
        out_coeff = os.path.join(self.folder_path, self.outname + "coeff.npy")
        out_intercept = os.path.join(self.folder_path, self.outname + "coeff_intercept.npy")
        return out_coeff, out_intercept



#
# Example usage (if you run this script directly):
#
# if __name__ == "__main__":
#     folder = "/content/drive/MyDrive/stcmtest"
#     outname = "GS24"

#     regressor = STCMBatchRegressor(folder, outname)
#     regressor.load_prep_files()      # read .prep files
#     regressor.fit_regression()       # do least-squares
#     regressor.print_results()        # show console output
#     regressor.save_coefficients()    # save .npy
#     regressor.make_plot()            # show Plotly figure
