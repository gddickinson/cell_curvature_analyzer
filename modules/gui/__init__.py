"""MainWindow mixins and workers for the desktop app (MainWindow itself is in modules/cell_curvature_analyzer.py).

tab_visualization, tab_analysis, tab_export_settings: building the tabs; frame_plots: per-frame plots and tables;
correlation_display, correlation_pooled, correlation_temporal, correlation_movement_temporal: the Correlation tab;
navigation, display: frame navigation and image / overlay / results display; data_actions: loading stacks and
running the analysis; export_actions, settings_actions, ml_tab, popup_plots; workers: QThread workers.
"""
