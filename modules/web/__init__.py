"""Web interface for Cell Curvature Analyzer (FastAPI + plain ES-module front end).

- app.py       FastAPI app: GET /api/info, /api/demo; POST /api/analyze; STATIC_ENDPOINTS for the static site
- analysis.py  analyze_stack (stack-level analysis on top of modules.edge_analysis), array encoding helpers
- static/      front end; *_engine.js are the browser ports tested against Python (edge_engine.test.mjs)
"""
