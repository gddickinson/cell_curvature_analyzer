"""Image and overlay display, random-control state and the results display."""
import numpy as np


class DisplayMixin:
    """MainWindow mixin: image and overlay display, random-control state and the results display."""

    def update_image_display(self):
        """Update the image display based on selected type and brightness settings"""
        # Check if data is loaded
        if not self.loaded_data:
            return

        # Get current selection
        image_type = self.image_type_combo.currentText()

        # Get current frame
        frame = self.current_frame

        # Get brightness settings
        min_brightness = self.min_brightness_slider.value() / 100.0
        max_brightness = self.max_brightness_slider.value() / 100.0
        mask_opacity = self.mask_opacity_slider.value() / 100.0

        # Update based on type
        if image_type == "Original Image" and 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if frame < images.shape[0]:
                # Copy the image but preserve original intensity values
                image = images[frame].copy().astype(float)

                # Skip normalization, but still apply brightness windowing
                # Calculate actual min/max values of the image
                actual_min = np.min(image)
                actual_max = np.max(image)

                # Calculate brightness window - scale between min and max based on slider settings
                window_min = actual_min + min_brightness * (actual_max - actual_min)
                window_max = actual_min + max_brightness * (actual_max - actual_min)

                # Apply brightness window without normalization
                image = np.clip(image, window_min, window_max)

                self.image_viewer.set_image(image)
                self.image_viewer.set_title(f"Original Image (Frame {frame+1}/{images.shape[0]})")

        elif image_type == "Binary Mask" and 'masks' in self.loaded_data and self.loaded_data['masks'] is not None:
            masks = self.loaded_data['masks']
            if frame < masks.shape[0]:
                self.image_viewer.set_image(masks[frame])
                self.image_viewer.set_title(f"Binary Mask (Frame {frame+1}/{masks.shape[0]})")

        elif image_type == "Overlay":
            if ('images' in self.loaded_data and self.loaded_data['images'] is not None and
                'masks' in self.loaded_data and self.loaded_data['masks'] is not None):

                images = self.loaded_data['images']
                masks = self.loaded_data['masks']

                if frame < images.shape[0] and frame < masks.shape[0]:
                    # Get raw image without normalization
                    image = images[frame].copy().astype(float)
                    mask = masks[frame]

                    # Calculate actual min/max values of the image
                    actual_min = np.min(image)
                    actual_max = np.max(image)

                    # Calculate brightness window
                    window_min = actual_min + min_brightness * (actual_max - actual_min)
                    window_max = actual_min + max_brightness * (actual_max - actual_min)

                    # Apply brightness window without normalization
                    image_windowed = np.clip(image, window_min, window_max)

                    # To create overlay, we need to normalize for the RGB array
                    # This is only for display purposes - raw values are preserved in the image_data property
                    image_range = window_max - window_min
                    if image_range > 0:
                        image_norm = (image_windowed - window_min) / image_range
                    else:
                        image_norm = np.zeros_like(image_windowed)

                    # Create RGB overlay
                    overlay = np.stack([image_norm, image_norm, image_norm], axis=2)

                    # Add red tint to masked areas with opacity control
                    red_mask = np.zeros_like(overlay)
                    red_mask[mask > 0, 0] = 1.0  # Red channel

                    # Blend using mask opacity
                    overlay = overlay * (1 - mask_opacity * mask[:, :, np.newaxis]) + red_mask * mask_opacity * mask[:, :, np.newaxis]

                    # Display the overlay, but store original data
                    self.image_viewer.set_image(overlay)
                    self.image_viewer.image_data = image  # Store original intensity data for mouse hover
                    self.image_viewer.set_title(f"Overlay (Frame {frame+1}/{images.shape[0]})")

    def update_overlay_display(self):
        """Update the overlay display based on selected options"""
        # Check if data is loaded
        if not self.loaded_data:
            return

        # Get current frame
        frame = self.current_frame

        # Check if we have results for this frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            frame not in self.loaded_data['results']):
            return

        # Get frame results
        frame_results = self.loaded_data['results'][frame]

        # Clear overlay canvas
        self.overlay_canvas.clear()

        # Add background image if available
        if 'images' in self.loaded_data and self.loaded_data['images'] is not None:
            images = self.loaded_data['images']
            if frame < images.shape[0]:
                # Pass the raw image data to the overlay canvas without normalization
                image = images[frame].copy().astype(float)
                self.overlay_canvas.add_background(image)

                # Store brightness settings for display purposes
                self.overlay_canvas.min_brightness = self.overlay_min_brightness_slider.value() / 100.0
                self.overlay_canvas.max_brightness = self.overlay_max_brightness_slider.value() / 100.0

        # Add mask if available
        if 'masks' in self.loaded_data and self.loaded_data['masks'] is not None:
            masks = self.loaded_data['masks']
            if frame < masks.shape[0]:
                mask = masks[frame]
                self.overlay_canvas.add_mask(mask, opacity=self.overlay_mask_opacity_slider.value() / 100.0)

        # Add contour if available and checkbox is checked
        if (self.show_contour_cb.isChecked() and
            'contour_data' in frame_results):
            self.overlay_canvas.add_contour(
                frame_results['contour_data'],
                color='r',  # Changed from 'y' to 'r' for red
                linewidth=2.0  # Increased from default value for better visibility
            )

        # Add curvature points if available and checkbox is checked
        if (self.show_curvature_cb.isChecked() and
            'points' in frame_results and
            'curvatures' in frame_results):
            # Determine which curvature to use
            if self.show_curvature_sign_cb.isChecked():
                # Use sign curvature
                curvature_index = 0
            else:
                # Use normalized curvature
                curvature_index = 2

            self.overlay_canvas.add_curvature_points(
                frame_results['points'],
                frame_results['curvatures'][curvature_index],  # Use selected curvature type
                cmap=self.curvature_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add intensity points if available and checkbox is checked
        if (self.show_intensity_cb.isChecked() and
            'points' in frame_results and
            'intensities' in frame_results):
            self.overlay_canvas.add_intensity_points(
                frame_results['points'],
                frame_results['intensities'],
                cmap=self.intensity_cmap_combo.currentText(),
                marker_size=self.marker_size_spin.value()
            )

        # Add sampling regions if available and checkbox is checked
        if (self.show_sampling_cb.isChecked() and
            'sampling_regions' in frame_results and
            'valid_points' in frame_results):
            self.overlay_canvas.add_sampling_regions(
                frame_results['sampling_regions'],
                frame_results['valid_points'],
                frame_results.get('recovered_by_rotation'),
                frame_results.get('exclude_endpoints', False)
            )


        # Add movement visualization if checkbox is checked and data exists
        try:
            if (self.show_movement_cb.isChecked()):
                # Try to generate movement data if it doesn't exist
                if 'movement_data' not in frame_results:
                    self.generate_frame_movement_data(self.current_frame)

                # If we now have movement data, display it
                if 'movement_data' in frame_results and 'points' in frame_results:
                    movement_scores = frame_results['movement_data']['scores']
                    movement_types = frame_results['movement_data']['types']

                    # Create color-coded points based on movement types
                    colors = []
                    for mtype in movement_types:
                        if mtype == 'extending':
                            colors.append('blue')
                        elif mtype == 'retracting':
                            colors.append('red')
                        else:
                            colors.append('gray')

                    self.overlay_canvas.add_movement_visualization(
                        frame_results['points'],
                        movement_scores,
                        colors,
                        marker_size=self.marker_size_spin.value()
                    )
        except Exception as e:
            # Log error but continue
            print(f"Error updating movement overlay: {str(e)}")


        # Update canvas
        self.overlay_canvas.update_canvas()

        # Set title
        self.overlay_canvas.set_title(f"Analysis Overlay (Frame {frame+1})")

    def update_random_control_state(self):
        """Enable or disable the random control analysis option based on settings"""
        enabled = self.enable_random_cb.isChecked()

        # Get current temporal mode
        current_mode = self.parameter_panel.temporal_mode_combo.currentText()

        # Block signals to prevent unnecessary updates
        self.parameter_panel.temporal_mode_combo.blockSignals(True)

        # Clear and rebuild the combo items
        self.parameter_panel.temporal_mode_combo.clear()
        self.parameter_panel.temporal_mode_combo.addItem("Current Frame")
        self.parameter_panel.temporal_mode_combo.addItem("Current-Previous Frames")

        if enabled:
            self.parameter_panel.temporal_mode_combo.addItem("Current-Random Frames")

        # Restore current selection if possible
        index = self.parameter_panel.temporal_mode_combo.findText(current_mode)
        if index >= 0:
            self.parameter_panel.temporal_mode_combo.setCurrentIndex(index)

        # Re-enable signals
        self.parameter_panel.temporal_mode_combo.blockSignals(False)

        # Update reference frame visibility
        self.parameter_panel.update_reference_frame_visibility()

    def update_results_display(self):
        """Update the results display based on current frame and temporal mode"""
        # Check if results are available for current frame
        if ('results' not in self.loaded_data or
            self.loaded_data['results'] is None or
            self.current_frame not in self.loaded_data['results']):
            return

        # Get frame results for current frame
        frame_results = self.loaded_data['results'][self.current_frame]

        # Get temporal mode from parameter panel
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Update frame info
        self.update_frame_info(frame_results)

        # Clear all plots first to ensure proper updates
        self.curvature_figure.clear()
        self.intensity_figure.clear()
        self.correlation_figure.clear()

        # Process based on temporal mode
        if temporal_mode == "Current Frame":
            # Standard single frame analysis
            if 'curvatures' in frame_results:
                self.update_curvature_plot(frame_results)

            if 'intensities' in frame_results and 'valid_points' in frame_results:
                self.update_intensity_plot(frame_results)

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):
                self.update_correlation_plot(frame_results)

        elif temporal_mode == "Current-Previous Frames" and self.current_frame > 0:
            # Get previous frame
            previous_frame = self.current_frame - 1

            if previous_frame in self.loaded_data['results']:
                previous_frame_results = self.loaded_data['results'][previous_frame]

                # Update plots with temporal comparison
                self.update_temporal_comparison_plots(frame_results, previous_frame_results,
                                                    "Previous Frame")

                # Log that we're using temporal comparison
                self.log_console.log(f"Using temporal comparison with previous frame {previous_frame+1}")

        elif temporal_mode == "Current-Random Frames" and reference_frame != self.current_frame:
            # Ensure reference frame exists in results
            if reference_frame in self.loaded_data['results']:
                reference_frame_results = self.loaded_data['results'][reference_frame]

                # Update plots with temporal comparison
                self.update_temporal_comparison_plots(frame_results, reference_frame_results,
                                                    f"Reference Frame {reference_frame+1}")

                # Log that we're using temporal comparison
                self.log_console.log(f"Using temporal comparison with reference frame {reference_frame+1}")
        else:
            # Fallback to standard display if temporal mode doesn't apply
            if 'curvatures' in frame_results:
                self.update_curvature_plot(frame_results)

            if 'intensities' in frame_results and 'valid_points' in frame_results:
                self.update_intensity_plot(frame_results)

            if ('curvatures' in frame_results and
                'intensities' in frame_results and
                'valid_points' in frame_results):
                self.update_correlation_plot(frame_results)

            # Log that we fell back to standard display
            self.log_console.log(f"Falling back to standard display (temporal mode {temporal_mode} not applicable)")

        # Update frame results table regardless of mode
        self.update_frame_results_table(frame_results)

        # Refresh all canvases
        self.curvature_canvas.draw()
        self.intensity_canvas.draw()
        self.correlation_canvas.draw()
