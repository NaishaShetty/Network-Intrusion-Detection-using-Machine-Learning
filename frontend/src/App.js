import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Overview from './pages/Overview';
import Performance from './pages/Performance';
import Prediction from './pages/Prediction';
import Insights from './pages/Insights';
import Simulation from './pages/Simulation';
import Monitoring from './pages/Monitoring';

function App() {
    return (
        <Router>
            <div className="app">
                <Navbar />
                <main>
                    <Routes>
                        <Route path="/" element={<Overview />} />
                        <Route path="/performance" element={<Performance />} />
                        <Route path="/prediction" element={<Prediction />} />
                        <Route path="/insights" element={<Insights />} />
                        <Route path="/simulation" element={<Simulation />} />
                        <Route path="/monitoring" element={<Monitoring />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}

export default App;
