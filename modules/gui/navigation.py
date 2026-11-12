"""Frame navigation, temporal and movement tab displays, parameter and visualisation-setting changes."""



class NavigationMixin:
    """MainWindow mixin: frame navigation, temporal and movement tab displays, parameter and visualisation-setting changes."""

    def update_temporal_display(self):
        """Update the temporal analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None:
            return

        # Get current selections
        visualization_type = self.temporal_type_combo.currentText()
        measure_type = self.temporal_measure_combo.currentText()

        # Update display based on visualization type
        if visualization_type == "Heatmap Visualization":
            self.temporal_analyzer.plot_temporal_heatmap(
                results, measure_type,
                self.temporal_plot_figure, self.temporal_plot_canvas,
                self.temporal_summary_figure, self.temporal_summary_canvas
            )

    def update_movement_display(self):
        """Update the movement analysis tab display"""
        results = self.loaded_data.get('results')

        if results is None or 'movement_types' not in results:
            return

        # Get current selections
        display_type = self.movement_display_combo.currentText()
        frame = self.movement_frame_spin.value()

        # Update display based on display type
        if display_type == "Movement Map":
            self.visualization_manager.plot_movement_map(
                results, frame,
                self.loaded_data.get('images'),
                self.loaded_data.get('masks'),
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_stats_table
            )

        elif display_type == "Edge Comparison":
            self.visualization_manager.plot_edge_comparison(
                results, frame,
                self.loaded_data.get('images'),
                self.loaded_data.get('masks'),
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_stats_table
            )

        elif display_type == "Movement Over Time":
            self.visualization_manager.plot_movement_over_time(
                results,
                self.movement_plot_figure,
                self.movement_plot_canvas,
                self.movement_summary_figure,
                self.movement_summary_canvas,
                self.movement_stats_table
            )

    def set_current_frame(self, frame_index):
        """Set the current frame index and update displays"""
        # Ensure valid index
        images = self.loaded_data.get('images')
        if images is not None:
            max_index = images.shape[0] - 1
            frame_index = max(0, min(frame_index, max_index))

        # Update current frame
        self.current_frame = frame_index

        # Update UI
        if images is not None:
            self.frame_label.setText(f"Frame: {frame_index+1}/{images.shape[0]}")

            # Update reference frame spinner max value
            self.parameter_panel.update_frame_range(max_index)

            # Update movement frame spinner max value (minimum is always 1)
            self.movement_frame_spin.setMaximum(max(1, max_index))

            # Set a reasonable default reference frame that's different from current
            if max_index > 0:
                ref_frame = (frame_index + max_index // 2) % (max_index + 1)
                self.parameter_panel.reference_frame_spin.setValue(ref_frame)

        # Update displays
        self.update_image_display()
        self.update_overlay_display()
        self.update_results_display()

    def next_frame(self):
        """Go to next frame"""
        images = self.loaded_data.get('images')
        if images is not None:
            max_index = images.shape[0] - 1
            new_frame = min(self.current_frame + 1, max_index)
            self.set_current_frame(new_frame)

    def previous_frame(self):
        """Go to previous frame"""
        new_frame = max(self.current_frame - 1, 0)
        self.set_current_frame(new_frame)

    def on_parameters_changed(self):
        """Handle changes to analysis parameters"""
        # Update parameter panel values in main settings if needed
        if 'n_points' in self.parameter_panel.parameters:
            self.n_points_spin.setValue(self.parameter_panel.parameters['n_points'])
        if 'depth' in self.parameter_panel.parameters:
            self.depth_spin.setValue(self.parameter_panel.parameters['depth'])
        if 'width' in self.parameter_panel.parameters:
            self.width_spin.setValue(self.parameter_panel.parameters['width'])
        if 'min_cell_coverage' in self.parameter_panel.parameters:
            self.min_coverage_spin.setValue(self.parameter_panel.parameters['min_cell_coverage'])

        # Get the temporal mode
        temporal_mode = self.parameter_panel.parameters.get('temporal_mode', "Current Frame")
        reference_frame = self.parameter_panel.parameters.get('reference_frame', 0)

        # Log parameter changes
        self.log_console.log(f"Analysis parameters updated: n_points={self.n_points_spin.value()}, "
                           f"depth={self.depth_spin.value()}, width={self.width_spin.value()}, "
                           f"min_cell_coverage={self.min_coverage_spin.value()}, "
                           f"temporal_mode={temporal_mode}, reference_frame={reference_frame}")

        # Force an update of the display
        self.update_results_display()

    def update_visualization_settings(self):
        """Update visualization settings"""
        # Update displays
        self.update_overlay_display()

        # Log settings changes
        self.log_console.log(f"Visualization settings updated: curvature_cmap={self.curvature_cmap_combo.currentText()}, "
                           f"intensity_cmap={self.intensity_cmap_combo.currentText()}, "
                           f"marker_size={self.marker_size_spin.value()}, "
                           f"contour_width={self.contour_width_spin.value()}")
