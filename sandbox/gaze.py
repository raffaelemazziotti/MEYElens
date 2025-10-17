import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class ScreenPositions:
    """
    Generates a sequence of screen positions for gaze tracking experiments.
    """
    def __init__(self, screen_width_deg, screen_height_deg, num_points):
        """
        Initialize the position generator.

        Args:
            screen_width_deg (float): Total width of the screen in degrees.
            screen_height_deg (float): Total height of the screen in degrees.
            num_points (int): Total number of points to generate.
        """
        self.screen_width_deg = screen_width_deg
        self.screen_height_deg = screen_height_deg
        self.num_points = num_points
        self.positions = self._generate_positions()
        self.current_index = -1

    def _generate_positions(self):
        """
        Generate positions with 5 fixed points (corners and center) and additional random points.

        Returns:
            List[Tuple[float, float]]: A list of positions (x, y).
        """
        fixed_points = [
            (0, 0),  # Center
            (-self.screen_width_deg / 2, self.screen_height_deg / 2),  # Top-left
            (self.screen_width_deg / 2, self.screen_height_deg / 2),  # Top-right
            (-self.screen_width_deg / 2, -self.screen_height_deg / 2),  # Bottom-left
            (self.screen_width_deg / 2, -self.screen_height_deg / 2),  # Bottom-right
        ]

        random_points = [
            (
                np.random.uniform(-self.screen_width_deg / 2, self.screen_width_deg / 2),
                np.random.uniform(-self.screen_height_deg / 2, self.screen_height_deg / 2),
            )
            for _ in range(self.num_points - len(fixed_points))
        ]

        positions = fixed_points + random_points
        np.random.shuffle(positions)
        return positions

    def next(self):
        """
        Get the next position in the sequence.

        Returns:
            Tuple[float, float]: The next position (x, y), or None if all positions are exhausted.
        """
        self.current_index += 1
        if self.current_index < len(self.positions):
            return self.positions[self.current_index]
        return None

class GazeData:
    """
    Handles reading, preprocessing, and managing gaze data from files.
    """

    def __init__(self, folder=None):
        """
        Args:
            folder (str or None): Folder containing gaze data files.
                                  If None, creates/uses '~/Documents/GazeData'.
        """
        if folder is None:
            documents = Path.home() / 'Documents'
            documents.mkdir(parents=True, exist_ok=True)
            folder = documents / 'GazeData'

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        self.data_folder = folder
        recs = os.listdir(folder)
        self.recs = [r for r in recs if len(r.split('-')) > 1 and r.split('-')[1] == 'track_cal.txt']

    def list(self):
        """
        List all available recordings.

        Returns:
            None
        """
        for i, fn in enumerate(self.recs):
            print(i, fn)

    def get_last(self):
        """
        Get gaze points and screen positions from the last recording.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gaze points and screen positions.
        """

        return self.get(-1)

    def get(self, i=-1):
        """
        Get gaze points and screen positions from a specific recording.

        Args:
            i (int): Index of the recording.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gaze points and screen positions.
        """
        print(f'### GAZEDATA ### Loading rec: {self.recs[i]}')
        pupil = pd.read_csv(os.path.join(self.data_folder, self.recs[i]), sep=';')
        pupil[['x', 'y']] = pupil[['x', 'y']].interpolate(method='linear', limit_direction='both', inplace=False)
        screen_positions = pupil[['trg2', 'trg3']].values
        gaze_points = pupil[['x', 'y']].values
        return gaze_points, screen_positions

    def get_all(self):
        """
        Get gaze points and screen positions from all recordings.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Combined gaze points and screen positions.
        """
        all_screen_positions = []
        all_gaze_points = []
        for r in self.recs:
            pupil = pd.read_csv(os.path.join(self.data_folder, r), sep=';')
            pupil[['x', 'y']] = pupil[['x', 'y']].interpolate(method='linear', limit_direction='both', inplace=False)
            screen_positions = pupil[['trg2', 'trg3']].values
            gaze_points = pupil[['x', 'y']].values
            all_screen_positions.append(screen_positions)
            all_gaze_points.append(gaze_points)
        return np.vstack(all_gaze_points), np.vstack(all_screen_positions)

    def plot(self, i=-1, skip_samples=20):
        """
        Plot gaze points for a specific recording.

        Args:
            i (int): Index of the recording.
            skip_samples (int): Number of samples to skip at the start.

        Returns:
            None
        """
        pupil = pd.read_csv(os.path.join(self.data_folder, self.recs[i]), sep=';')
        fig, ax = plt.subplots()
        colors = plt.cm.tab10.colors
        for j, pos in enumerate(pupil['trg1'].unique()):
            data = pupil[pupil['trg1'] == pos].iloc[skip_samples:]
            ax.plot(data['y'], data['x'], color=colors[j % len(colors)], marker='o', label=f'Target {pos}')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

class GazeModelPoly:
    """
    GazeModelPoly: A class for calibrating and predicting gaze positions using polynomial regression.
    """
    def __init__(self):
        """
        Initialize the GazeModelPoly class with scalers and placeholders for the polynomial model.
        """
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.poly_features = None
        self.calibration_model = None

    def train(self, gaze_points, screen_positions, degree=2):
        """
        Train the polynomial regression model for gaze calibration.

        Args:
            gaze_points (ndarray): Array of gaze points with shape (n_samples, 2).
            screen_positions (ndarray): Array of corresponding screen positions with shape (n_samples, 2).
            degree (int): Degree of the polynomial fit (default is 2).

        Returns:
            None
        """
        screen_positions = self.scaler_target.fit_transform(screen_positions)
        gaze_points = self.scaler_features.fit_transform(gaze_points)

        self.poly_features = PolynomialFeatures(degree=degree)
        transformed_gaze = self.poly_features.fit_transform(gaze_points)
        self.calibration_model = LinearRegression()
        self.calibration_model.fit(transformed_gaze, screen_positions)
        calibration_predictions = self.calibration_model.predict(transformed_gaze)
        calibration_mse = mean_squared_error(screen_positions, calibration_predictions)
        calibration_r2 = r2_score(screen_positions, calibration_predictions)
        print(f"Calibration Model - MSE: {calibration_mse:.4f}, R2: {calibration_r2:.4f}")

    def predict(self, new_eye_position):
        """
        Predict screen positions for new gaze points using the trained calibration model.

        Args:
            new_eye_position (ndarray): Array of new gaze points with shape (n_samples, 2).

        Returns:
            ndarray: Predicted screen positions with shape (n_samples, 2).

        Raises:
            ValueError: If the model has not been trained yet.
        """
        normalized_input = self.scaler_features.transform(new_eye_position)

        if self.calibration_model is not None and self.poly_features is not None:
            transformed_input = self.poly_features.transform(normalized_input)
            calibrated_prediction = self.calibration_model.predict(transformed_input)
        else:
            raise ValueError("The model has not been trained yet.")

        return self.scaler_target.inverse_transform(calibrated_prediction)

    def save(self, model_path=None):
        """
        Save the trained model and scalers to disk.

        Args:
            model_path (str): Path to save the model. Defaults to 'gaze_models/gazemodel_poly.pkl'.

        Returns:
            None
        """
        if model_path is None:
            os.makedirs('gaze_models', exist_ok=True)
            model_path = 'gaze_models/gazemodel_poly.pkl'

        joblib.dump({
            'calibration_model': self.calibration_model,
            'scaler_features': self.scaler_features,
            'scaler_target': self.scaler_target,
            'poly_features': self.poly_features
        }, model_path)

    @staticmethod
    def load(model_path=None):
        """
        Load a trained model and scalers from disk.

        Args:
            model_path (str): Path to load the model from. Defaults to 'gaze_models/gazemodel_poly.pkl'.

        Returns:
            GazeModelPoly: An instance of the class with the loaded model and scalers.
        """
        if model_path is None:
            model_path = 'gaze_models/gazemodel_poly.pkl'

        data = joblib.load(model_path)
        instance = GazeModelPoly()
        instance.calibration_model = data['calibration_model']
        instance.scaler_features = data['scaler_features']
        instance.scaler_target = data['scaler_target']
        instance.poly_features = data['poly_features']
        return instance

if __name__ == '__main__':
    """
    Main function for training, saving, and predicting gaze positions.
    """
    gdata = GazeData()
    gdata.list()
    #gaze_points, positions = gdata.get_all()
    #gmodel = GazeModelPoly()
    #gmodel.train(gaze_points, positions)
    #gmodel.save()
    #print(gmodel.predict(np.array((21, 30)).reshape(1, -1)))