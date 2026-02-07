import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, AlertCircle, Clock, BarChart as BarChartIcon, Search, ShieldAlert } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell } from 'recharts';
import './Prediction.css';

const Prediction = () => {
    const [file, setFile] = useState(null);
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [threshold, setThreshold] = useState(0.5);
    const [isPredicting, setIsPredicting] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await axios.get('/api/models');
            setModels(response.data.models || []);
            if (response.data.models?.length > 0) {
                setSelectedModel(response.data.models[0]);
            }
        } catch (err) {
            console.error('Error fetching models:', err);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files.length > 0) {
            setFile(e.target.files[0]);
            setResults(null);
            setError(null);
        }
    };

    const runPrediction = async () => {
        if (!file) {
            setError('Please select a CSV file first.');
            return;
        }

        setIsPredicting(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(`/api/predict?model_name=${selectedModel}&threshold=${threshold}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            setResults(response.data);
        } catch (err) {
            console.error('Prediction error:', err);
            setError(err.response?.data?.detail || 'An error occurred during prediction.');
        } finally {
            setIsPredicting(false);
        }
    };

    const reset = () => {
        setFile(null);
        setResults(null);
        setError(null);
        setThreshold(0.5);
    };

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <h1>Real-time Prediction Engine</h1>
                    <p className="page-description">
                        Analyze network traffic batches and adjust sensitivity for balanced detection.
                    </p>
                </header>

                <div className="prediction-grid">
                    {/* Control Panel */}
                    <div className="card">
                        <div className="card-header">
                            <Upload className="section-icon" size={24} />
                            <h3>Upload Traffic Data</h3>
                        </div>

                        <div className="form-group">
                            <label>Model selection</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="form-select"
                                disabled={isPredicting}
                            >
                                {models.map(m => (
                                    <option key={m} value={m}>{m.replace('_', ' ').toUpperCase()}</option>
                                ))}
                            </select>
                        </div>

                        <div className="form-group">
                            <div className="slider-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                <label>Attack Threshold: <strong>{(threshold * 100).toFixed(0)}%</strong></label>
                                <span className={`status-pill ${threshold < 0.3 ? 'danger' : threshold > 0.7 ? 'success' : 'medium'}`}>
                                    {threshold < 0.3 ? 'Aggressive' : threshold > 0.7 ? 'Conservative' : 'Standard'}
                                </span>
                            </div>
                            <input
                                type="range"
                                min="0.01"
                                max="0.99"
                                step="0.01"
                                value={threshold}
                                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                                className="form-range"
                                disabled={isPredicting}
                            />
                            <p className="help-text" style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.5rem' }}>
                                Lower threshold = Fewer False Negatives (Missed Attacks).<br />
                                Higher threshold = Fewer False Positives (False Alarms).
                            </p>
                        </div>

                        <div className="file-input-wrapper" style={{ margin: '1.5rem 0' }}>
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                id="file-upload"
                                style={{ display: 'none' }}
                            />
                            <label htmlFor="file-upload" className="file-upload-label" style={{ padding: '2rem', border: '2px dashed #334155', borderRadius: '12px', display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
                                <FileText size={40} color={file ? '#10b981' : '#475569'} />
                                <span style={{ marginTop: '0.5rem' }}>{file ? file.name : 'Select Capture File...'}</span>
                            </label>
                        </div>

                        <div className="action-buttons">
                            <button
                                className="btn btn-primary"
                                onClick={runPrediction}
                                disabled={!file || isPredicting}
                                style={{ width: '100%' }}
                            >
                                {isPredicting ? 'Analyzing Packets...' : 'Start Detection'}
                            </button>
                            <button className="btn btn-outline" onClick={reset} style={{ width: '100%', marginTop: '0.5rem' }}>Clear</button>
                        </div>

                        {error && (
                            <div className="error-box" style={{ marginTop: '1rem', color: '#f43f5e', background: 'rgba(244, 63, 94, 0.1)', padding: '0.75rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                <AlertCircle size={20} />
                                <span>{error}</span>
                            </div>
                        )}
                    </div>

                    {/* Experimental Results */}
                    <div className="prediction-results">
                        {results ? (
                            <div className="results-stack" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                {/* Summary Card */}
                                <div className="card result-card">
                                    <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <ShieldAlert size={24} color={results.summary.attack_count > 0 ? '#ef4444' : '#10b981'} />
                                            <h3>Detection Overview</h3>
                                        </div>
                                        <div className="latency-info" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: '#94a3b8', fontSize: '0.85rem' }}>
                                            <Clock size={16} />
                                            <span>{results.latency_ms.toFixed(3)} ms / sample</span>
                                        </div>
                                    </div>
                                    <div className="summary-stats" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                                        <div className="stat-item">
                                            <div className="stat-label">Total Traffic</div>
                                            <div className="stat-value">{results.total_samples} samples</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">Attacks Detected</div>
                                            <div className="stat-value text-danger" style={{ color: '#ef4444', fontWeight: 'bold' }}>{results.summary.attack_count}</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">Normal Patterns</div>
                                            <div className="stat-value text-success" style={{ color: '#10b981' }}>{results.summary.normal_count}</div>
                                        </div>
                                        <div className="stat-item">
                                            <div className="stat-label">Mean Confidence</div>
                                            <div className="stat-value">{(results.summary.avg_confidence * 100).toFixed(1)}%</div>
                                        </div>
                                    </div>
                                </div>

                                {/* Histogram */}
                                <div className="card">
                                    <div className="card-header" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <BarChartIcon size={20} />
                                        <h3>Probability Distribution</h3>
                                    </div>
                                    <div style={{ height: '220px', width: '100%', marginTop: '1rem' }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={results.probability_distribution}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                                <XAxis
                                                    dataKey="bin"
                                                    stroke="#cbd5e1"
                                                    tickFormatter={(v) => v.toFixed(1)}
                                                />
                                                <YAxis stroke="#cbd5e1" />
                                                <Tooltip
                                                    contentStyle={{ background: '#1e293b', border: '1px solid #334155' }}
                                                />
                                                <Bar dataKey="count" name="Samples">
                                                    {results.probability_distribution.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.bin >= threshold ? '#ef4444' : '#10b981'} />
                                                    ))}
                                                </Bar>
                                                <ReferenceLine
                                                    x={threshold}
                                                    stroke="#00f2fe"
                                                    strokeWidth={2}
                                                    label={{ value: 'Sensitivity', position: 'top', fill: '#00f2fe' }}
                                                />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <p style={{ fontSize: '0.75rem', color: '#64748b', textAlign: 'center', marginTop: '0.5rem' }}>
                                        Red bars indicate samples classified as <strong>Attack</strong> at current threshold ({threshold}).
                                    </p>
                                </div>

                                {/* Table Preview */}
                                <div className="card">
                                    <div className="card-header">
                                        <Search size={20} />
                                        <h3>Result Samples</h3>
                                    </div>
                                    <div className="table-responsive" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                        <table className="results-table" style={{ width: '100%', borderCollapse: 'collapse', marginTop: '0.5rem' }}>
                                            <thead style={{ position: 'sticky', top: 0, background: '#1e293b' }}>
                                                <tr>
                                                    <th style={{ textAlign: 'left', padding: '0.75rem' }}>#</th>
                                                    <th style={{ textAlign: 'left', padding: '0.75rem' }}>Classification</th>
                                                    <th style={{ textAlign: 'left', padding: '0.75rem' }}>Att. Prob</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {results.predictions.slice(0, 10).map((pred, i) => (
                                                    <tr key={i} style={{ borderBottom: '1px solid #334155' }}>
                                                        <td style={{ padding: '0.75rem' }}>{i + 1}</td>
                                                        <td style={{ padding: '0.75rem' }}>
                                                            <span className={`status-pill ${pred.prediction === 'attack' ? 'danger' : 'success'}`}>
                                                                {pred.prediction.toUpperCase()}
                                                            </span>
                                                        </td>
                                                        <td style={{ padding: '0.75rem' }}>{(pred.attack_probability * 100).toFixed(1)}%</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        {results.total_samples > 10 && (
                                            <p style={{ padding: '1rem', textAlign: 'center', fontSize: '0.85rem', color: '#94a3b8' }}>
                                                Showing top 10 of {results.total_samples} results.
                                            </p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div className="empty-results" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', opacity: 0.5 }}>
                                <ShieldAlert size={64} />
                                <p style={{ marginTop: '1rem' }}>Awaiting traffic capture file...</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Prediction;
