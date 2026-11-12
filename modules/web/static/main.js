// Entry point: registers lazily loaded pages, sets the footer mode text and starts the router.
import { start, lazy, route, app, isStatic, $ } from './core.js';

const PAGES = { 'home.js': ['home'], 'analyze.js': ['analyze'], 'convention.js': ['convention'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic() ? 'static site: analysis runs in your browser (JS port of the Python core)' : 'local app: analysis runs in Python';
start();
