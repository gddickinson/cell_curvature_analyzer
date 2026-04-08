#!/usr/bin/env python3
"""
Configuration Management Module

Centralized configuration for cell edge movement analysis.
Handles parameter validation and provides default configurations.

Author: George Dickinson
Date: November 4, 2025
"""

import json
from pathlib import Path
from typing import Dict, Any
import copy


# Default configuration parameters
DEFAULT_CONFIG = {
    # Sampling parameters
    'n_points': 12,
    'depth': 200,
    'width': 75,
    'min_cell_coverage': 0.8,
    'try_rotation': True,
    
    # Movement detection parameters
    'movement_threshold': 0.1,
    'min_movement_pixels': 5,
    'exclude_endpoints': True,
    'temporal_direction': 'past',  # 'past' or 'future'
}

DEFAULT_OUTPUT_CONFIG = {
    'save_detailed_csv': True,
    'save_summary_json': True,
    'save_correlation_plots': True,
    'save_frame_plots': True,
}

DEFAULT_VIZ_CONFIG = {
    'plot_every_nth_frame': 10,
    'plot_every_nth_sampling_point': 1,
    'save_sampling_plots': True,
    'save_intensity_plots': True,
    'save_movement_plots': True,
    'save_edge_transition_plots': True,
    'save_movement_overlay_plots': True,
    'save_intensity_overlay_plots': True,
    'save_movement_type_correlation': True,
    'movement_cmap': 'RdBu_r',
    'intensity_cmap': 'viridis',
    'marker_size': 30,
    'figure_dpi': 150,
}


class Config:
    """Configuration manager for analysis pipeline."""
    
    def __init__(self, custom_config: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        custom_config : dict, optional
            Custom configuration parameters to override defaults
        """
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.output_config = copy.deepcopy(DEFAULT_OUTPUT_CONFIG)
        self.viz_config = copy.deepcopy(DEFAULT_VIZ_CONFIG)
        
        if custom_config:
            self.update(custom_config)
    
    def update(self, custom_config: Dict[str, Any]):
        """
        Update configuration with custom parameters.
        
        Parameters
        ----------
        custom_config : dict
            Dictionary with configuration updates
        """
        if 'analysis' in custom_config:
            self.config.update(custom_config['analysis'])
        if 'output' in custom_config:
            self.output_config.update(custom_config['output'])
        if 'visualization' in custom_config:
            self.viz_config.update(custom_config['visualization'])
        
        # Also support flat updates
        for key, value in custom_config.items():
            if key in self.config:
                self.config[key] = value
            elif key in self.output_config:
                self.output_config[key] = value
            elif key in self.viz_config:
                self.viz_config[key] = value
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns
        -------
        bool
            True if configuration is valid
        
        Raises
        ------
        ValueError
            If configuration parameters are invalid
        """
        # Validate sampling parameters
        if self.config['n_points'] < 1:
            raise ValueError("n_points must be >= 1")
        if self.config['depth'] < 1:
            raise ValueError("depth must be >= 1")
        if self.config['width'] < 1:
            raise ValueError("width must be >= 1")
        if not 0 <= self.config['min_cell_coverage'] <= 1:
            raise ValueError("min_cell_coverage must be between 0 and 1")
        
        # Validate movement parameters
        if self.config['movement_threshold'] < 0:
            raise ValueError("movement_threshold must be >= 0")
        if self.config['min_movement_pixels'] < 1:
            raise ValueError("min_movement_pixels must be >= 1")
        
        # Validate temporal direction
        if self.config['temporal_direction'] not in ['past', 'future']:
            raise ValueError("temporal_direction must be 'past' or 'future'")
        
        return True
    
    def save(self, filepath: Path):
        """
        Save configuration to JSON file.
        
        Parameters
        ----------
        filepath : Path
            Path to save configuration
        """
        config_dict = {
            'analysis': self.config,
            'output': self.output_config,
            'visualization': self.viz_config,
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path):
        """
        Load configuration from JSON file.
        
        Parameters
        ----------
        filepath : Path
            Path to configuration file
        
        Returns
        -------
        Config
            Configuration object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def get_dict(self) -> Dict[str, Any]:
        """
        Get complete configuration as dictionary.
        
        Returns
        -------
        dict
            Complete configuration dictionary
        """
        return {
            'analysis': self.config,
            'output': self.output_config,
            'visualization': self.viz_config,
        }
    
    def __repr__(self):
        """String representation of configuration."""
        return f"Config(n_points={self.config['n_points']}, " \
               f"depth={self.config['depth']}, " \
               f"width={self.config['width']}, " \
               f"temporal_direction={self.config['temporal_direction']})"


# =============================================================================
# UNIT TEST
# =============================================================================

def run_unit_test(visualize: bool = False, test_data_path: str = None):
    """
    Unit test for configuration module.
    
    Parameters
    ----------
    visualize : bool
        Whether to print configuration details
    test_data_path : str, optional
        Path to test configuration file (not used in this module)
    """
    print("="*60)
    print("CONFIGURATION MODULE UNIT TEST")
    print("="*60)
    
    # Test 1: Default configuration
    print("\n[Test 1] Creating default configuration...")
    config = Config()
    assert config.config['n_points'] == 12
    assert config.config['temporal_direction'] == 'past'
    print("✓ Default configuration created successfully")
    
    # Test 2: Custom configuration
    print("\n[Test 2] Creating custom configuration...")
    custom = {
        'analysis': {
            'n_points': 20,
            'depth': 300,
        },
        'output': {
            'save_detailed_csv': False,
        }
    }
    config = Config(custom)
    assert config.config['n_points'] == 20
    assert config.config['depth'] == 300
    assert config.output_config['save_detailed_csv'] == False
    print("✓ Custom configuration applied successfully")
    
    # Test 3: Validation
    print("\n[Test 3] Testing configuration validation...")
    valid_config = Config()
    assert valid_config.validate() == True
    print("✓ Valid configuration passes validation")
    
    # Test invalid configuration
    try:
        invalid_config = Config({'analysis': {'n_points': -1}})
        invalid_config.validate()
        print("✗ Validation should have failed for invalid n_points")
        assert False
    except ValueError as e:
        print(f"✓ Validation correctly rejects invalid config: {e}")
    
    # Test 4: Save and load
    print("\n[Test 4] Testing save and load...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    config = Config({'analysis': {'n_points': 15}})
    config.save(temp_path)
    loaded_config = Config.load(temp_path)
    
    assert loaded_config.config['n_points'] == 15
    print("✓ Configuration saved and loaded successfully")
    
    # Cleanup
    import os
    os.remove(temp_path)
    
    # Test 5: Update configuration
    print("\n[Test 5] Testing configuration updates...")
    config = Config()
    config.update({'n_points': 25, 'movement_threshold': 0.5})
    assert config.config['n_points'] == 25
    assert config.config['movement_threshold'] == 0.5
    print("✓ Configuration updated successfully")
    
    if visualize:
        print("\n" + "="*60)
        print("CONFIGURATION DETAILS")
        print("="*60)
        print("\nDefault Analysis Configuration:")
        for key, value in DEFAULT_CONFIG.items():
            print(f"  {key}: {value}")
        
        print("\nDefault Output Configuration:")
        for key, value in DEFAULT_OUTPUT_CONFIG.items():
            print(f"  {key}: {value}")
        
        print("\nDefault Visualization Configuration:")
        for key, value in DEFAULT_VIZ_CONFIG.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    visualize = '--visualize' in sys.argv or '-v' in sys.argv
    test_data = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--test-data' and i + 1 < len(sys.argv):
            test_data = sys.argv[i + 1]
    
    run_unit_test(visualize=visualize, test_data_path=test_data)
