import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Clock, AlertTriangle, AlertCircle, Database, RefreshCcw } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './Monitoring.css';

const Monitoring = () => {
    const [driftData, setDriftData] = useState(null);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchMonitoringData();
    }, []);

    const fetchMonitoringData = async () => {
        setLoading(true);
        try {
            const [driftRes, statsRes] = await Promise.all([
                axios.get('/api/monitoring/drift'),
                axios.get('/api/monitoring/stats')
            ]);
            setDriftData(driftRes.data);
            setStats(statsRes.data);
        } catch (error) {
            console.error('Error fetching monitoring data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="page-container">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Analyzing system health...</p>
                </div>
            </div>
        );
    }

    const latencyData = stats?.latencies && Object.keys(stats.latencies).length > 0 ?
        Object.entries(stats.latencies).map(([name, val]) => ({
            name: name.toUpperCase().replace('_', ' '),
            latency: parseFloat(val.toFixed(4))
        })) : [];

    const driftScores = driftData?.feature_drift ?
        Object.entries(driftData.feature_drift).map(([f, score]) => ({
            feature: f,
            score: parseFloat(score.toFixed(4))
        })).sort((a, b) => b.score - a.score).slice(0, 10) : [];

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <h1>Operational Monitoring</h1>
                    <p className="page-description">
                        Track model inference speed and data distribution shifts to ensure long-term accuracy.
                    </p>
                </header>

                <div className="monitoring-grid">
                    {/* Data Drift Section */}
                    <div className="card full-width">
                        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                                <Database size={24} color="#00f2fe" />
                                <h3>Concept Drift Analysis</h3>
                            </div>
                            <button className="btn-small" onClick={fetchMonitoringData}><RefreshCcw size={14} /> Refresh</button>
                        </div>

                        {driftData?.status === 'no_data' || driftData?.status === 'error' ? (
                            <div className="empty-monitoring">
                                <AlertTriangle size={40} color="#f59e0b" />
                                <p>{driftData?.message || "No recent traffic data found to compare against baseline. Use the Prediction page to upload traffic."}</p>
                            </div>
                        ) : (
                            <div className="drift-content">
                                <div className="drift-summary">
                                    <div className={`drift-badge ${driftData?.avg_drift > 0.3 ? 'warn' : 'stable'}`}>
                                        {driftData?.avg_drift > 0.3 ? 'MILD DRIFT DETECTED' : 'DATA DISTRIBUTION STABLE'}
                                    </div>
                                    <p>The system compares current input feature statistics against the training baseline (KDD-99).</p>
                                </div>
                                <div style={{ height: '300px', marginTop: '1.5rem' }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={driftScores} layout="vertical">
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                            <XAxis type="number" stroke="#64748b" label={{ value: 'Drift Score (normalized Δ)', position: 'insideBottom', offset: -5 }} />
                                            <YAxis dataKey="feature" type="category" stroke="#64748b" width={100} />
                                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                                            <Bar dataKey="score">
                                                {driftScores.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.score > 0.2 ? '#ef4444' : '#3b82f6'} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Latency Section */}
                    <div className="card">
                        <div className="card-header">
                            <Clock size={24} color="#10b981" />
                            <h3>Inference Latency</h3>
                        </div>
                        <div style={{ height: '250px', marginTop: '1rem' }}>
                            {latencyData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={latencyData}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                        <XAxis dataKey="name" stroke="#64748b" />
                                        <YAxis stroke="#64748b" label={{ value: 'ms / sample', angle: -90, position: 'insideLeft' }} />
                                        <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                                        <Bar dataKey="latency" fill="#10b981" />
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="empty-monitoring">
                                    <Clock size={40} color="#64748b" />
                                    <p>No latency data. Perform predictions to see metrics.</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* System Health */}
                    <div className="card">
                        <div className="card-header">
                            <Activity size={24} color="#8b5cf6" />
                            <h3>Calibration Status</h3>
                        </div>
                        <div className="health-details">
                            <div className="health-row">
                                <span>Probability Calibration</span>
                                <span className={stats?.is_calibrated ? 'text-success' : 'text-danger'}>
                                    {stats?.is_calibrated ? 'ACTIVE' : 'INACTIVE'}
                                </span>
                            </div>
                            <div className="health-row">
                                <span>Active Models</span>
                                <span>{stats?.active_models?.length || 0}</span>
                            </div>
                            <div className="health-row">
                                <span>Fallback Mode</span>
                                <span className="text-success">DISABLED</span>
                            </div>
                        </div>
                        <div className="calibration-tip">
                            <AlertCircle size={16} />
                            <p>Calibration ensures that a predicted probability of 0.8 actually corresponds to an 80% likelihood of an attack.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Monitoring;
