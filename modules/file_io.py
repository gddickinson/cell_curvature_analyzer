import numpy as np
import tifffile
import os
import json
import pandas as pd

class FileManager:
    """
    Class for handling file input/output operations
    """
    
    def __init__(self):
        """Initialize FileManager"""
        pass
    
    def load_image_stack(self, file_path):
        """
        Load a microscope image stack
        
        Parameters:
        -----------
        file_path : str
            Path to TIFF image stack
            
        Returns:
        --------
        images : ndarray
            Array of microscope images
        """
        # Load images
        images = tifffile.imread(file_path)
        
        # Handle single image case
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        
        return images
    
    def load_mask_stack(self, file_path):
        """
        Load a binary mask stack
        
        Parameters:
        -----------
        file_path : str
            Path to TIFF mask stack
            
        Returns:
        --------
        masks : ndarray
            Array of binary masks
        """
        # Load masks
        masks = tifffile.imread(file_path)
        
        # Ensure masks are binary (0 = background, 1 = cell)
        masks = (masks > 0).astype(np.uint8)
        
        # Handle single image case
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]
        
        return masks
    
    def save_image(self, image, file_path):
        """
        Save a single image
        
        Parameters:
        -----------
        image : ndarray
            Image to save
        file_path : str
            Path to save image
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image
        tifffile.imwrite(file_path, image)
    
    def save_image_stack(self, images, file_path):
        """
        Save an image stack
        
        Parameters:
        -----------
        images : ndarray
            Array of images to save
        file_path : str
            Path to save image stack
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image stack
        tifffile.imwrite(file_path, images)
    
    def save_results_json(self, results, file_path):
        """
        Save analysis results as JSON
        
        Parameters:
        -----------
        results : dict
            Analysis results
        file_path : str
            Path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert numpy types to Python types
        def numpy_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_python(item) for item in obj]
            elif isinstance(obj, tuple):
                return [numpy_to_python(item) for item in obj]
            else:
                return obj
        
        # Convert results
        serializable_results = numpy_to_python(results)
        
        # Save JSON
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def save_results_csv(self, results, file_path):
        """
        Save analysis results as CSV
        
        Parameters:
        -----------
        results : dict
            Analysis results
        file_path : str
            Path to save results
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Extract frame data and combine into a DataFrame
        frames_data = []
        
        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int):
                continue
            
            # Extract key data from frame results
            if ('points' in frame_results and 
                'curvatures' in frame_results and 
                'intensities' in frame_results and
                'valid_points' in frame_results):
                
                # Get curvatures
                sign_curvatures, magnitude_curvatures, normalized_curvatures = frame_results['curvatures']
                
                # Get data
                points = frame_results['points']
                intensities = frame_results['intensities']
                valid_points = frame_results['valid_points']
                
                # Add to frames data
                for i in range(len(points)):
                    frames_data.append({
                        'frame': frame_idx,
                        'point_index': i,
                        'x': points[i, 1],  # x coordinate
                        'y': points[i, 0],  # y coordinate
                        'valid': valid_points[i],
                        'curvature_sign': sign_curvatures[i],
                        'curvature_magnitude': magnitude_curvatures[i],
                        'curvature_normalized': normalized_curvatures[i],
                        'intensity': intensities[i]
                    })
        
        # Convert to DataFrame
        if frames_data:
            df = pd.DataFrame(frames_data)
            
            # Save CSV
            df.to_csv(file_path, index=False)
    
    def load_results_json(self, file_path):
        """
        Load analysis results from JSON
        
        Parameters:
        -----------
        file_path : str
            Path to results file
            
        Returns:
        --------
        results : dict
            Analysis results
        """
        # Load JSON
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays for key data
        for frame_idx, frame_results in results.items():
            if not isinstance(frame_idx, int) or not isinstance(frame_results, dict):
                continue
            
            # Convert points to numpy array
            if 'points' in frame_results:
                frame_results['points'] = np.array(frame_results['points'])
            
            # Convert curvatures to numpy arrays
            if 'curvatures' in frame_results:
                curvatures = frame_results['curvatures']
                if isinstance(curvatures, list) and len(curvatures) == 3:
                    frame_results['curvatures'] = (
                        np.array(curvatures[0]),
                        np.array(curvatures[1]),
                        np.array(curvatures[2])
                    )
            
            # Convert intensities to numpy array
            if 'intensities' in frame_results:
                frame_results['intensities'] = np.array(frame_results['intensities'])
            
            # Convert valid_points to numpy array
            if 'valid_points' in frame_results:
                frame_results['valid_points'] = np.array(frame_results['valid_points'])
            
            # Convert normals to numpy array
            if 'normals' in frame_results:
                frame_results['normals'] = np.array(frame_results['normals'])
            
            # Convert sampling_regions to list of numpy arrays
            if 'sampling_regions' in frame_results:
                sampling_regions = frame_results['sampling_regions']
                if isinstance(sampling_regions, list):
                    frame_results['sampling_regions'] = [np.array(region) for region in sampling_regions]
        
        return results
