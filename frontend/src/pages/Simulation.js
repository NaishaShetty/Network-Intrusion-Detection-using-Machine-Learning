import React, { useState, useEffect, useRef } from 'react';
import { Radio, Play, Pause, AlertTriangle, ShieldCheck, Activity, Terminal } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './Simulation.css';

const Simulation = () => {
    const [isActive, setIsActive] = useState(false);
    const [traffic, setTraffic] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [samplesProcessed, setSamplesProcessed] = useState(0);
    const timerRef = useRef(null);

    const toggleSimulation = () => {
        setIsActive(!isActive);
    };

    useEffect(() => {
        if (isActive) {
            timerRef.current = setInterval(() => {
                simulatePacket();
            }, 1000);
        } else {
            clearInterval(timerRef.current);
        }
        return () => clearInterval(timerRef.current);
    }, [isActive]);

    const simulatePacket = () => {
        // Mock a packet detection
        const isAttack = Math.random() < 0.15;
        const confidence = 0.85 + Math.random() * 0.14;
        const timestamp = new Date().toLocaleTimeString();

        setTraffic(prev => [...prev.slice(-19), { time: timestamp, score: isAttack ? confidence : 1 - confidence }]);
        setSamplesProcessed(prev => prev + 1);

        if (isAttack) {
            const newAlert = {
                id: Date.now(),
                time: timestamp,
                type: 'Anomaly Detected',
                severity: confidence > 0.95 ? 'CRITICAL' : 'HIGH',
                confidence: (confidence * 100).toFixed(1)
            };
            setAlerts(prev => [newAlert, ...prev.slice(0, 9)]);
        }
    };

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <h1>Live Traffic Simulation</h1>
                    <p className="page-description">
                        Monitor simulated network flow in real-time to test model responsiveness and alert orchestration.
                    </p>
                </header>

                <div className="sim-grid-layout">
                    {/* Control Center */}
                    <div className="card control-panel">
                        <div className="card-header">
                            <Radio className={isActive ? 'pulse' : ''} color={isActive ? '#ef4444' : '#64748b'} />
                            <h3>Simulation Controls</h3>
                        </div>
                        <div className="status-indicator">
                            <div className={`status-dot ${isActive ? 'active' : 'idle'}`}></div>
                            <span>System Status: {isActive ? 'LIVE MONITORING' : 'IDLE'}</span>
                        </div>
                        <div className="sim-stats">
                            <div className="stat-box">
                                <Activity size={20} />
                                <div>
                                    <label>Packets/sec</label>
                                    <div className="val">{isActive ? '1.0' : '0.0'}</div>
                                </div>
                            </div>
                            <div className="stat-box">
                                <ShieldCheck size={20} />
                                <div>
                                    <label>Processed</label>
                                    <div className="val">{samplesProcessed}</div>
                                </div>
                            </div>
                        </div>
                        <button
                            className={`btn ${isActive ? 'btn-danger' : 'btn-primary'}`}
                            onClick={toggleSimulation}
                            style={{ width: '100%', marginTop: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}
                        >
                            {isActive ? <><Pause size={20} /> Stop Simulation</> : <><Play size={20} /> Start Live Loop</>}
                        </button>
                    </div>

                    {/* Real-time Graph */}
                    <div className="card graph-card">
                        <div className="card-header">
                            <Activity size={20} color="#00f2fe" />
                            <h3>Anomalous Probability Stream</h3>
                        </div>
                        <div style={{ height: '300px', width: '100%', marginTop: '1rem' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={traffic}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="time" stroke="#64748b" fontSize={12} />
                                    <YAxis stroke="#64748b" domain={[0, 1]} />
                                    <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                                    <Line
                                        type="monotone"
                                        dataKey="score"
                                        stroke="#00f2fe"
                                        strokeWidth={3}
                                        dot={{ fill: '#00f2fe', r: 4 }}
                                        activeDot={{ r: 8 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Alert Feed */}
                    <div className="card alert-feed">
                        <div className="card-header">
                            <AlertTriangle size={20} color="#ef4444" />
                            <h3>Security Alerts</h3>
                        </div>
                        <div className="alert-list">
                            {alerts.length === 0 ? (
                                <div className="empty-alerts">
                                    <ShieldCheck size={48} color="#10b981" opacity={0.3} />
                                    <p>No threats detected</p>
                                </div>
                            ) : (
                                alerts.map(alert => (
                                    <div key={alert.id} className={`alert-item ${alert.severity === 'CRITICAL' ? 'critical' : ''}`}>
                                        <div className="alert-meta">
                                            <span className="badge">{alert.severity}</span>
                                            <span className="time">{alert.time}</span>
                                        </div>
                                        <div className="alert-body">
                                            <strong>{alert.type}</strong>
                                            <p>Model confidence: {alert.confidence}%</p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Console Log */}
                    <div className="card console-card">
                        <div className="card-header">
                            <Terminal size={20} />
                            <h3>System Logs</h3>
                        </div>
                        <div className="console-output">
                            <div className="log-line"><span>[SYS]</span> Initializing detection engines...</div>
                            <div className="log-line"><span>[SYS]</span> Loading calibrated weights...</div>
                            {isActive && <div className="log-line"><span>[NET]</span> Listening on all interfaces...</div>}
                            {alerts.map(a => (
                                <div key={a.id} className="log-line error"><span>[IDP]</span> Intrusion blocked at {a.time} (Conf: {a.confidence}%)</div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Simulation;
