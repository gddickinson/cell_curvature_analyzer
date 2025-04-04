# New file: modules/ml_analysis.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

class MLAnalyzer:
    """
    Class for machine learning analysis of cell curvature and intensity patterns
    """

    def __init__(self):
        """Initialize MLAnalyzer"""
        self.model = None
        self.scaler = None
        self.pipeline = None
        self.trained = False
        self.feature_names = ['curvature_sign', 'curvature_magnitude',
                             'curvature_normalized', 'intensity']

    def prepare_training_data(self, results):
        """
        Extract features and labels from results for training

        Parameters:
        -----------
        results : dict
            Analysis results

        Returns:
        --------
        features : ndarray
            Feature matrix
        labels : ndarray
            Labels for classification (0=low interest, 1=high interest)
        """
        features_list = []
        labels_list = []

        # Extract features from all frames
        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):

                # Get curvatures
                sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

                # Get intensities
                intensities = frame_results['intensities']

                # Get valid points
                valid_points = frame_results['valid_points']

                # For each valid point, create a feature vector
                for i in range(len(valid_points)):
                    if valid_points[i]:
                        features = [
                            sign_curvatures[i],
                            magnitude_curvatures[i],
                            normalized_curvatures[i],
                            intensities[i]
                        ]

                        # Simple rule for initial labeling:
                        # Points with high curvature magnitude and high intensity are of high interest
                        # This can be replaced with manual labeling or more sophisticated rules
                        label = 1 if (magnitude_curvatures[i] > 0.5 and
                                      intensities[i] > np.percentile(intensities[valid_points], 75)) else 0

                        features_list.append(features)
                        labels_list.append(label)

        return np.array(features_list), np.array(labels_list)

    def train_model(self, features, labels):
        """
        Train a random forest classifier on the provided data

        Parameters:
        -----------
        features : ndarray
            Feature matrix
        labels : ndarray
            Labels for classification

        Returns:
        --------
        accuracy : float
            Training accuracy
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        # Create and train model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.pipeline.fit(X_train, y_train)
        self.trained = True

        # Evaluate on test set
        accuracy = self.pipeline.score(X_test, y_test)

        return accuracy

    def classify_points(self, frame_results):
        """
        Classify points in a frame as high or low interest

        Parameters:
        -----------
        frame_results : dict
            Results for a single frame

        Returns:
        --------
        classifications : ndarray
            Binary classification for each point (0=low interest, 1=high interest)
        probabilities : ndarray
            Probability of high interest for each point
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model first.")

        if ('curvatures' not in frame_results or
            'intensities' not in frame_results or
            'valid_points' not in frame_results):
            raise ValueError("Frame results missing required data.")

        # Get curvatures
        sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']

        # Get intensities
        intensities = frame_results['intensities']

        # Get valid points
        valid_points = frame_results['valid_points']

        # Prepare feature matrix
        features = []
        valid_indices = []

        for i in range(len(valid_points)):
            if valid_points[i]:
                features.append([
                    sign_curvatures[i],
                    magnitude_curvatures[i],
                    normalized_curvatures[i],
                    intensities[i]
                ])
                valid_indices.append(i)

        # Skip if no valid points
        if not features:
            return np.array([]), np.array([])

        # Classify points
        features = np.array(features)
        probabilities = self.pipeline.predict_proba(features)[:, 1]  # Probability of class 1
        classifications = (probabilities > 0.5).astype(int)

        # Create full arrays with NaN for invalid points
        full_classifications = np.zeros(len(valid_points)) - 1  # -1 for invalid points
        full_probabilities = np.zeros(len(valid_points)) - 1

        # Fill in values for valid points
        for i, idx in enumerate(valid_indices):
            full_classifications[idx] = classifications[i]
            full_probabilities[idx] = probabilities[i]

        return full_classifications, full_probabilities

    def save_model(self, file_path):
        """
        Save trained model to disk

        Parameters:
        -----------
        file_path : str
            Path to save model
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save pipeline
        joblib.dump(self.pipeline, file_path)

    def load_model(self, file_path):
        """
        Load trained model from disk

        Parameters:
        -----------
        file_path : str
            Path to load model from
        """
        # Load pipeline
        self.pipeline = joblib.load(file_path)
        self.trained = True

    def get_feature_importances(self):
        """
        Get feature importances from trained model

        Returns:
        --------
        importances : dict
            Dictionary mapping feature names to importance scores
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train_model first.")

        # Get feature importances
        importances = self.pipeline.named_steps['classifier'].feature_importances_

        # Map to feature names
        return dict(zip(self.feature_names, importances))
