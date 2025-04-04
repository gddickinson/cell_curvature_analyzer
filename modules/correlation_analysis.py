import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class CorrelationAnalyzer:
    """
    Class for analyzing correlations between cell curvature and intensity measurements
    """
    
    def __init__(self):
        """Initialize CorrelationAnalyzer"""
        pass
    
    def analyze_frame_correlation(self, frame_results, curvature_type):
        """
        Analyze correlation between curvature and intensity for a single frame
        
        Parameters:
        -----------
        frame_results : dict
            Results for a single frame
        curvature_type : str
            Type of curvature to analyze ("Sign Curvature", "Magnitude Curvature", "Normalized Curvature")
            
        Returns:
        --------
        correlation_stats : dict
            Dictionary of correlation statistics
        """
        # Check if required data is available
        if ('curvatures' not in frame_results or 
            'intensities' not in frame_results or 
            'valid_points' not in frame_results):
            return None
        
        # Get data
        if curvature_type == "Sign Curvature":
            curvatures = frame_results['curvatures'][0]  # Sign curvature
        elif curvature_type == "Magnitude Curvature":
            curvatures = frame_results['curvatures'][1]  # Magnitude curvature
        elif curvature_type == "Normalized Curvature":
            curvatures = frame_results['curvatures'][2]  # Normalized curvature
        else:
            return None
        
        intensities = frame_results['intensities']
        valid_points = frame_results['valid_points']
        
        # Filter for valid points
        valid_curvatures = curvatures[valid_points]
        valid_intensities = intensities[valid_points]
        
        # Skip if not enough valid points
        if len(valid_curvatures) < 2:
            return {
                "r_squared": None,
                "p_value": None,
                "slope": None,
                "intercept": None,
                "sample_size": len(valid_curvatures)
            }
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_curvatures, valid_intensities)
        
        # Return statistics
        return {
            "r_squared": r_value**2,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "sample_size": len(valid_curvatures)
        }
    
    def analyze_all_frames_correlation(self, results, curvature_type):
        """
        Analyze correlation between curvature and intensity across all frames
        
        Parameters:
        -----------
        results : dict
            Analysis results
        curvature_type : str
            Type of curvature to analyze ("Sign Curvature", "Magnitude Curvature", "Normalized Curvature")
            
        Returns:
        --------
        correlation_stats : dict
            Dictionary of correlation statistics
        """
        # Combine data from all frames
        combined_curvatures = []
        combined_intensities = []
        
        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue
            
            # Check if required data is available
            if ('curvatures' not in frame_results or 
                'intensities' not in frame_results or 
                'valid_points' not in frame_results):
                continue
            
            # Get data
            if curvature_type == "Sign Curvature":
                curvatures = frame_results['curvatures'][0]  # Sign curvature
            elif curvature_type == "Magnitude Curvature":
                curvatures = frame_results['curvatures'][1]  # Magnitude curvature
            elif curvature_type == "Normalized Curvature":
                curvatures = frame_results['curvatures'][2]  # Normalized curvature
            else:
                continue
            
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']
            
            # Filter for valid points
            valid_curvatures = curvatures[valid_points]
            valid_intensities = intensities[valid_points]
            
            # Add to combined data
            combined_curvatures.extend(valid_curvatures)
            combined_intensities.extend(valid_intensities)
        
        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)
        
        # Skip if not enough valid points
        if len(combined_curvatures) < 2:
            return {
                "r_squared": None,
                "p_value": None,
                "slope": None,
                "intercept": None,
                "sample_size": len(combined_curvatures)
            }
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_curvatures, combined_intensities)
        
        # Return statistics
        return {
            "r_squared": r_value**2,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "sample_size": len(combined_curvatures)
        }
    
    def analyze_movement_grouped_correlation(self, results, curvature_type):
        """
        Analyze correlation between curvature and intensity grouped by movement type
        
        Parameters:
        -----------
        results : dict
            Analysis results
        curvature_type : str
            Type of curvature to analyze ("Sign Curvature", "Magnitude Curvature", "Normalized Curvature")
            
        Returns:
        --------
        correlation_stats : dict
            Dictionary of correlation statistics by movement type
        """
        # Check if movement data is available
        if 'movement_types' not in results:
            return {}
        
        movement_types = results['movement_types']
        
        # Group frames by movement type
        extending_frames = []
        retracting_frames = []
        stable_frames = []
        
        for i, movement_type in enumerate(movement_types):
            # Movement type is for transitions, so the frame index is i+1
            frame = i + 1
            if movement_type == 'extending':
                extending_frames.append(frame)
            elif movement_type == 'retracting':
                retracting_frames.append(frame)
            elif movement_type == 'stable':
                stable_frames.append(frame)
        
        # Analyze correlation for each movement type
        movement_stats = {}
        
        # Extending frames
        extending_stats = self.analyze_frames_by_indices(
            results, extending_frames, curvature_type)
        if extending_stats:
            movement_stats['extending'] = extending_stats
        
        # Retracting frames
        retracting_stats = self.analyze_frames_by_indices(
            results, retracting_frames, curvature_type)
        if retracting_stats:
            movement_stats['retracting'] = retracting_stats
        
        # Stable frames
        stable_stats = self.analyze_frames_by_indices(
            results, stable_frames, curvature_type)
        if stable_stats:
            movement_stats['stable'] = stable_stats
        
        return movement_stats
    
    def analyze_frames_by_indices(self, results, frame_indices, curvature_type):
        """
        Analyze correlation for a specific group of frames
        
        Parameters:
        -----------
        results : dict
            Analysis results
        frame_indices : list
            List of frame indices to analyze
        curvature_type : str
            Type of curvature to analyze ("Sign Curvature", "Magnitude Curvature", "Normalized Curvature")
            
        Returns:
        --------
        correlation_stats : dict
            Dictionary of correlation statistics
        """
        # Combine data from specified frames
        combined_curvatures = []
        combined_intensities = []
        
        for frame_idx in frame_indices:
            if frame_idx not in results:
                continue
            
            frame_results = results[frame_idx]
            
            # Check if required data is available
            if ('curvatures' not in frame_results or 
                'intensities' not in frame_results or 
                'valid_points' not in frame_results):
                continue
            
            # Get data
            if curvature_type == "Sign Curvature":
                curvatures = frame_results['curvatures'][0]  # Sign curvature
            elif curvature_type == "Magnitude Curvature":
                curvatures = frame_results['curvatures'][1]  # Magnitude curvature
            elif curvature_type == "Normalized Curvature":
                curvatures = frame_results['curvatures'][2]  # Normalized curvature
            else:
                continue
            
            intensities = frame_results['intensities']
            valid_points = frame_results['valid_points']
            
            # Filter for valid points
            valid_curvatures = curvatures[valid_points]
            valid_intensities = intensities[valid_points]
            
            # Add to combined data
            combined_curvatures.extend(valid_curvatures)
            combined_intensities.extend(valid_intensities)
        
        # Convert to numpy arrays
        combined_curvatures = np.array(combined_curvatures)
        combined_intensities = np.array(combined_intensities)
        
        # Skip if not enough valid points
        if len(combined_curvatures) < 2:
            return {
                "r_squared": None,
                "p_value": None,
                "slope": None,
                "intercept": None,
                "frame_count": len(frame_indices),
                "sample_size": len(combined_curvatures)
            }
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            combined_curvatures, combined_intensities)
        
        # Return statistics
        return {
            "r_squared": r_value**2,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
            "std_err": std_err,
            "frame_count": len(frame_indices),
            "sample_size": len(combined_curvatures)
        }
    
    def compare_movement_correlations(self, results, curvature_type):
        """
        Compare correlation statistics between different movement types
        
        Parameters:
        -----------
        results : dict
            Analysis results
        curvature_type : str
            Type of curvature to analyze ("Sign Curvature", "Magnitude Curvature", "Normalized Curvature")
            
        Returns:
        --------
        comparison_stats : dict
            Dictionary of comparison statistics
        """
        # Get movement-grouped correlation statistics
        movement_stats = self.analyze_movement_grouped_correlation(results, curvature_type)
        
        # Skip if insufficient data
        if len(movement_stats) < 2:
            return {
                "significant_difference": False,
                "f_statistic": None,
                "p_value": None,
                "movement_stats": movement_stats
            }
        
        # Compare R-squared values
        movement_r2 = {
            movement: stats["r_squared"]
            for movement, stats in movement_stats.items()
            if stats["r_squared"] is not None
        }
        
        # Skip if insufficient data
        if len(movement_r2) < 2:
            return {
                "significant_difference": False,
                "f_statistic": None,
                "p_value": None,
                "movement_stats": movement_stats
            }
        
        # Perform ANOVA test
        from scipy.stats import f_oneway
        
        # Prepare data for ANOVA
        groups = []
        for movement, stats in movement_stats.items():
            if stats["r_squared"] is not None:
                # Create a group with r_squared repeated sample_size times
                group = [stats["r_squared"]] * stats["sample_size"]
                groups.append(group)
        
        # Skip if insufficient data
        if len(groups) < 2:
            return {
                "significant_difference": False,
                "f_statistic": None,
                "p_value": None,
                "movement_stats": movement_stats
            }
        
        # Perform ANOVA
        f_statistic, p_value = f_oneway(*groups)
        
        # Return comparison statistics
        return {
            "significant_difference": p_value < 0.05,
            "f_statistic": f_statistic,
            "p_value": p_value,
            "movement_stats": movement_stats
        }
