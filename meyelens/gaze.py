import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ScreenPositions:
    """
    Generate a shuffled sequence of calibration target positions on a screen.

    The sequence always includes 5 fixed points:

    - center
    - top-left, top-right
    - bottom-left, bottom-right

    Optionally, additional random points can be added within the screen bounds.

    Parameters
    ----------
    screen_width : float
        Screen width in degrees (or in any coordinate unit you consistently use).
    screen_height : float
        Screen height in degrees (or in any coordinate unit you consistently use).
    random_points : bool, optional
        If ``True``, add random positions in addition to the fixed ones.
    num_points : int, optional
        Total number of points in the returned sequence (including the 5 fixed points).
        If smaller than 5, only the 5 fixed points will be used.

    Notes
    -----
    - Positions are shuffled once at initialization.
    - Coordinates are returned as (x, y) where the origin is the center.
    """

    def __init__(self, screen_width, screen_height, random_points: bool = False, num_points: int = 5):
        self.screen_width_deg = screen_width
        self.screen_height_deg = screen_height
        self.random_points = random_points
        self.num_points = num_points

        self.positions = self._generate_positions()
        self.current_index = -1

    def _generate_positions(self):
        """
        Build and shuffle the list of target positions.

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) target positions.
        """
        fixed_points = [
            (0, 0),  # center
            (-self.screen_width_deg / 2, self.screen_height_deg / 2),   # top-left
            (self.screen_width_deg / 2, self.screen_height_deg / 2),    # top-right
            (-self.screen_width_deg / 2, -self.screen_height_deg / 2),  # bottom-left
            (self.screen_width_deg / 2, -self.screen_height_deg / 2),   # bottom-right
        ]

        # Decide how many random points to generate (never negative).
        n_random = max(0, self.num_points - len(fixed_points))

        if self.random_points and n_random > 0:
            random_points = [
                (
                    np.random.uniform(-self.screen_width_deg / 2, self.screen_width_deg / 2),
                    np.random.uniform(-self.screen_height_deg / 2, self.screen_height_deg / 2),
                )
                for _ in range(n_random)
            ]
        else:
            random_points = []

        positions = fixed_points + random_points
        np.random.shuffle(positions)
        return positions

    def next(self):
        """
        Return the next target position in the shuffled sequence.

        Returns
        -------
        tuple[float, float] or None
            Next (x, y) position, or ``None`` when the sequence is exhausted.
        """
        self.current_index += 1
        if self.current_index < len(self.positions):
            return self.positions[self.current_index]
        return None


class GazeData:
    """
    Load, preprocess, and visualize gaze calibration recordings.

    This class expects recordings in a folder as ``*.txt`` (or any extension) with
    semicolon-separated columns, including at least:

    - ``x``, ``y``: gaze coordinates
    - ``trg1``: target identifier (used for plotting groups)
    - ``trg2``, ``trg3``: screen target coordinates (x, y) associated with each sample

    By default, if no folder is provided, data is stored under:

    ``~/Documents/GazeData``

    Notes
    -----
    - Missing ``x``/``y`` samples are linearly interpolated in both directions.
    - File discovery is based on a filename pattern used by your acquisition
      pipeline: the second dash-separated token must equal ``track_cal.txt``.
    """

    def __init__(self, folder=None):
        """
        Parameters
        ----------
        folder : str or pathlib.Path or None, optional
            Folder containing gaze data files. If ``None``, uses
            ``~/Documents/GazeData`` (creating it if needed).
        """
        if folder is None:
            documents = Path.home() / "Documents"
            documents.mkdir(parents=True, exist_ok=True)
            folder = documents / "GazeData"

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        self.data_folder = folder

        # Discover recordings following the naming convention:
        # <prefix>-track_cal.txt-<suffix> ... OR ... <prefix>-track_cal.txt
        recs = os.listdir(folder)
        self.recs = [r for r in recs if len(r.split("-")) > 1 and r.split("-")[1] == "track_cal.txt"]

    def list(self) -> None:
        """
        Print the available recordings and their indices.

        Returns
        -------
        None
        """
        for i, fn in enumerate(self.recs):
            print(i, fn)

    def get_last(self):
        """
        Convenience method to load the most recent recording in the list.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(gaze_points, screen_positions)`` where each has shape (n_samples, 2).

        Notes
        -----
        This simply calls :meth:`get` with ``i=-1`` (last element in list).
        """
        return self.get(-1)

    def get(self, i: int = -1):
        """
        Load a specific recording by index.

        Parameters
        ----------
        i : int, optional
            Recording index (supports negative indexing).

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(gaze_points, screen_positions)`` where each has shape (n_samples, 2).

        Raises
        ------
        IndexError
            If ``i`` is out of range for the available recordings.
        """
        print(f"### GAZEDATA ### Loading rec: {self.recs[i]}")
        pupil = pd.read_csv(os.path.join(self.data_folder, self.recs[i]), sep=";")

        # Interpolate missing gaze samples.
        pupil[["x", "y"]] = pupil[["x", "y"]].interpolate(
            method="linear",
            limit_direction="both",
            inplace=False,
        )

        screen_positions = pupil[["trg2", "trg3"]].values
        gaze_points = pupil[["x", "y"]].values
        return gaze_points, screen_positions

    def get_all(self):
        """
        Load and concatenate all recordings.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(all_gaze_points, all_screen_positions)`` concatenated across
            recordings. Each has shape (n_total_samples, 2).

        Notes
        -----
        Uses the same interpolation strategy as :meth:`get`.
        """
        all_screen_positions = []
        all_gaze_points = []

        for r in self.recs:
            pupil = pd.read_csv(os.path.join(self.data_folder, r), sep=";")
            pupil[["x", "y"]] = pupil[["x", "y"]].interpolate(
                method="linear",
                limit_direction="both",
                inplace=False,
            )

            all_screen_positions.append(pupil[["trg2", "trg3"]].values)
            all_gaze_points.append(pupil[["x", "y"]].values)

        return np.vstack(all_gaze_points), np.vstack(all_screen_positions)

    def plot(self, i: int = -1, skip_samples: int = 20) -> None:
        """
        Plot gaze traces grouped by target identifier for a given recording.

        Parameters
        ----------
        i : int, optional
            Recording index (supports negative indexing).
        skip_samples : int, optional
            Number of samples to skip at the beginning of each target segment.
            This is useful to ignore the initial transient after a target switch.

        Returns
        -------
        None

        Notes
        -----
        - Uses ``trg1`` to group samples by target ID.
        - Plots ``y`` on the x-axis and ``x`` on the y-axis (matching your current
          convention), then inverts the y-axis.
        """
        pupil = pd.read_csv(os.path.join(self.data_folder, self.recs[i]), sep=";")

        fig, ax = plt.subplots()
        colors = plt.cm.tab10.colors

        for j, pos in enumerate(pupil["trg1"].unique()):
            data = pupil[pupil["trg1"] == pos].iloc[skip_samples:]
            ax.plot(
                data["y"],
                data["x"],
                color=colors[j % len(colors)],
                marker="o",
                label=f"Target {pos}",
            )

        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()


class GazeModelPoly:
    """
    Polynomial regression gaze calibration model.

    This model learns a mapping from raw gaze coordinates (e.g., pupil/eye
    coordinates) to screen target coordinates using polynomial feature expansion
    and linear regression.

    Workflow
    --------
    1. Call :meth:`train` with paired ``(gaze_points, screen_positions)``.
    2. Call :meth:`predict` to map new gaze points to calibrated screen positions.
    3. Optionally call :meth:`save` / :meth:`load` to persist the trained model.

    Notes
    -----
    - Both inputs and targets are standardized using :class:`sklearn.preprocessing.StandardScaler`.
    - Training metrics (MSE and R²) are printed to stdout (no logging).
    """

    def __init__(self):
        """
        Initialize an untrained model with feature/target scalers.

        Attributes
        ----------
        scaler_features : sklearn.preprocessing.StandardScaler
            Scaler fitted on gaze features.
        scaler_target : sklearn.preprocessing.StandardScaler
            Scaler fitted on screen target coordinates.
        poly_features : sklearn.preprocessing.PolynomialFeatures or None
            Polynomial feature expander created during training.
        calibration_model : sklearn.linear_model.LinearRegression or None
            Linear regression model fitted during training.
        """
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.poly_features = None
        self.calibration_model = None

    def train(self, gaze_points, screen_positions, degree: int = 2) -> None:
        """
        Train the calibration model.

        Parameters
        ----------
        gaze_points : numpy.ndarray
            Raw gaze points of shape (n_samples, 2).
        screen_positions : numpy.ndarray
            Screen target positions of shape (n_samples, 2).
        degree : int, optional
            Polynomial degree used by :class:`sklearn.preprocessing.PolynomialFeatures`.

        Returns
        -------
        None

        Notes
        -----
        Training is performed in standardized space (both X and y). Reported MSE
        and R² are therefore in standardized units.
        """
        # Standardize targets and features.
        screen_positions_std = self.scaler_target.fit_transform(screen_positions)
        gaze_points_std = self.scaler_features.fit_transform(gaze_points)

        # Polynomial expansion then linear regression.
        self.poly_features = PolynomialFeatures(degree=degree)
        transformed_gaze = self.poly_features.fit_transform(gaze_points_std)

        self.calibration_model = LinearRegression()
        self.calibration_model.fit(transformed_gaze, screen_positions_std)

        # Quick training-set diagnostics.
        calibration_predictions = self.calibration_model.predict(transformed_gaze)
        calibration_mse = mean_squared_error(screen_positions_std, calibration_predictions)
        calibration_r2 = r2_score(screen_positions_std, calibration_predictions)
        print(f"Calibration Model - MSE: {calibration_mse:.4f}, R2: {calibration_r2:.4f}")

    def predict(self, new_eye_position):
        """
        Predict calibrated screen coordinates for new gaze points.

        Parameters
        ----------
        new_eye_position : numpy.ndarray
            New gaze points of shape (n_samples, 2).

        Returns
        -------
        numpy.ndarray
            Predicted screen positions of shape (n_samples, 2), returned in the
            original (inverse-transformed) coordinate space.

        Raises
        ------
        ValueError
            If the model has not been trained (or loaded) yet.
        """
        if self.calibration_model is None or self.poly_features is None:
            raise ValueError("The model has not been trained yet.")

        normalized_input = self.scaler_features.transform(new_eye_position)
        transformed_input = self.poly_features.transform(normalized_input)
        calibrated_prediction_std = self.calibration_model.predict(transformed_input)

        return self.scaler_target.inverse_transform(calibrated_prediction_std)

    def save(self, model_path: str | None = None) -> None:
        """
        Save the trained model and preprocessing objects to disk using joblib.

        Parameters
        ----------
        model_path : str or None, optional
            Output path. If ``None``, defaults to ``gaze_models/gazemodel_poly.pkl``
            and creates the ``gaze_models`` folder if needed.

        Returns
        -------
        None

        Notes
        -----
        The saved bundle includes:

        - ``calibration_model``
        - ``scaler_features``
        - ``scaler_target``
        - ``poly_features``
        """
        if model_path is None:
            os.makedirs("gaze_models", exist_ok=True)
            model_path = "gaze_models/gazemodel_poly.pkl"

        joblib.dump(
            {
                "calibration_model": self.calibration_model,
                "scaler_features": self.scaler_features,
                "scaler_target": self.scaler_target,
                "poly_features": self.poly_features,
            },
            model_path,
        )

    @staticmethod
    def load(model_path: str | None = None):
        """
        Load a saved model bundle from disk.

        Parameters
        ----------
        model_path : str or None, optional
            Path to the saved ``.pkl``. If ``None``, defaults to
            ``gaze_models/gazemodel_poly.pkl``.

        Returns
        -------
        GazeModelPoly
            An instance populated with the loaded model and preprocessing objects.
        """
        if model_path is None:
            model_path = "gaze_models/gazemodel_poly.pkl"

        data = joblib.load(model_path)

        instance = GazeModelPoly()
        instance.calibration_model = data["calibration_model"]
        instance.scaler_features = data["scaler_features"]
        instance.scaler_target = data["scaler_target"]
        instance.poly_features = data["poly_features"]
        return instance


if __name__ == "__main__":
    # Minimal example: print the generated calibration positions.
    poss = ScreenPositions(10, 10)

    while True:
        point = poss.next()
        if point:
            print(point)
        else:
            break
