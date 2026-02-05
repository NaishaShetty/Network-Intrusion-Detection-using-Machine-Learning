import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';
import { TrendingUp, Award, AlertCircle, ShieldAlert, Zap, Target } from 'lucide-react';
import './Performance.css';

const Performance = () => {
    const [dashboardData, setDashboardData] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const fetchDashboardData = async () => {
        try {
            const response = await axios.get('/api/results/dashboard');
            if (response.data && response.data.comparison && response.data.comparison.model_names) {
                setDashboardData(response.data);
                if (response.data.comparison.model_names.length > 0) {
                    setSelectedModel(response.data.comparison.model_names[0]);
                }
            } else {
                setDashboardData(null);
            }
        } catch (error) {
            console.error('Error fetching dashboard data:', error);
            setDashboardData(null);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="page-container">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading performance data...</p>
                </div>
            </div>
        );
    }

    if (!dashboardData || !dashboardData.comparison?.model_names?.length) {
        return (
            <div className="page-container">
                <div className="container">
                    <div className="empty-state">
                        <AlertCircle size={64} />
                        <h2>No Performance Data Available</h2>
                        <p>Please train the models first from the Overview page.</p>
                    </div>
                </div>
            </div>
        );
    }

    const { comparison } = dashboardData;
    const modelNames = comparison?.model_names || [];

    // Advanced comparison data
    const comparisonData = modelNames.map((name, idx) => ({
        name: name?.replace('_', ' ').toUpperCase() || 'UNKNOWN',
        accuracy: (comparison.accuracy?.[idx] * 100 || 0).toFixed(1),
        balanced_accuracy: (comparison.balanced_accuracy?.[idx] * 100 || 0).toFixed(1),
        f1_score: (comparison.f1_score?.[idx] * 100 || 0).toFixed(1),
        roc_auc: (comparison.roc_auc?.[idx] * 100 || 0).toFixed(1),
        fnr: (comparison.fnr?.[idx] * 100 || 0).toFixed(2),
        brier: comparison.brier_score?.[idx] || 0
    }));

    // Find models with lowest False Negative Rate (Best for security)
    const fnrValues = comparison?.fnr || [];
    const minFNR = fnrValues.length > 0 ? Math.min(...fnrValues) : 1;
    const bestSecurityModelIndex = fnrValues.indexOf(minFNR);
    const bestSecurityModel = modelNames[bestSecurityModelIndex] || 'N/A';

    // Find models with highest Balanced Accuracy (Best for imbalanced data)
    const bAccValues = comparison?.balanced_accuracy || [];
    const maxBAcc = bAccValues.length > 0 ? Math.max(...bAccValues) : 0;
    const bestBalancedModelIndex = bAccValues.indexOf(maxBAcc);
    const bestBalancedModel = modelNames[bestBalancedModelIndex] || 'N/A';

    const selectedModelData = selectedModel ? dashboardData.models?.[selectedModel] : null;

    const confusionMatrixData = selectedModelData?.confusion_matrix ? [
        { name: 'True Normal', predicted_normal: selectedModelData.confusion_matrix[0]?.[0] || 0, predicted_attack: selectedModelData.confusion_matrix[0]?.[1] || 0 },
        { name: 'True Attack', predicted_normal: selectedModelData.confusion_matrix[1]?.[0] || 0, predicted_attack: selectedModelData.confusion_matrix[1]?.[1] || 0 }
    ] : [];

    // Radar chart data for selected model
    const radarData = selectedModelData?.metrics ? [
        { subject: 'Accuracy', A: (selectedModelData.metrics.accuracy || 0) * 100, fullMark: 100 },
        { subject: 'Precision', A: (selectedModelData.metrics.precision || 0) * 100, fullMark: 100 },
        { subject: 'Recall', A: (selectedModelData.metrics.recall || 0) * 100, fullMark: 100 },
        { subject: 'F1 Score', A: (selectedModelData.metrics.f1_score || 0) * 100, fullMark: 100 },
        { subject: 'Balanced Acc', A: (selectedModelData.metrics.balanced_accuracy || 0) * 100, fullMark: 100 }
    ] : [];

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <h1>Network Intrusion Detection Metrics</h1>
                    <p className="page-description">
                        Advanced evaluation including imbalanced class awareness and probability calibration reliability.
                    </p>
                </header>

                {/* Performance Summary Cards */}
                <div className="summary-grid" style={{ marginBottom: '2rem' }}>
                    <div className="summary-card gold">
                        <Award size={24} />
                        <h4>Best Accuracy</h4>
                        <p>{(comparison?.accuracy?.length > 0) ? modelNames[comparison.accuracy.indexOf(Math.max(...comparison.accuracy))].replace('_', ' ').toUpperCase() : 'N/A'}</p>
                    </div>
                    <div className="summary-card blue">
                        <ShieldAlert size={24} />
                        <h4>Lowest False Negative Rate</h4>
                        <p>{bestSecurityModel.replace('_', ' ').toUpperCase()}</p>
                    </div>
                    <div className="summary-card cyan">
                        <Zap size={24} />
                        <h4>Optimal Balanced Score</h4>
                        <p>{bestBalancedModel.replace('_', ' ').toUpperCase()}</p>
                    </div>
                </div>

                {/* Primary Metrics Grid */}
                <section className="metrics-grid" style={{ marginBottom: '3rem' }}>
                    {modelNames.map((name, idx) => (
                        <div key={name} className={`metric-card-v2 ${selectedModel === name ? 'active' : ''}`} onClick={() => setSelectedModel(name)}>
                            <div className="metric-header">
                                <h3>{name.replace('_', ' ').toUpperCase()}</h3>
                                <span className={`status-pill ${(comparison.accuracy?.[idx] || 0) > 0.98 ? 'high' : 'medium'}`}>
                                    {(comparison.accuracy?.[idx] * 100 || 0).toFixed(1)}%
                                </span>
                            </div>
                            <div className="metric-body">
                                <div className="sub-metric">
                                    <span>Balanced Accuracy</span>
                                    <strong>{(comparison.balanced_accuracy?.[idx] * 100 || 0).toFixed(1)}%</strong>
                                </div>
                                <div className="sub-metric">
                                    <span>False Negative Rate</span>
                                    <strong style={{ color: (comparison.fnr?.[idx] || 0) < 0.05 ? '#10b981' : '#f43f5e' }}>
                                        {(comparison.fnr?.[idx] * 100 || 0).toFixed(2)}%
                                    </strong>
                                </div>
                                <div className="sub-metric">
                                    <span>Brier Score (Calibration)</span>
                                    <strong>{comparison.brier_score?.[idx]?.toFixed(4) || 'N/A'}</strong>
                                </div>
                            </div>
                        </div>
                    ))}
                </section>

                {/* Comparison Charts */}
                <div className="charts-double-grid">
                    <section className="chart-section card">
                        <h3>Model Comparison by Security Metrics</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={comparisonData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis dataKey="name" stroke="#cbd5e1" />
                                <YAxis yAxisId="left" stroke="#3b82f6" orientation="left" domain={[90, 100]} />
                                <YAxis yAxisId="right" stroke="#ef4444" orientation="right" domain={[0, 10]} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                                <Legend />
                                <Bar yAxisId="left" dataKey="balanced_accuracy" fill="#3b82f6" name="Balanced Accuracy %" />
                                <Bar yAxisId="right" dataKey="fnr" fill="#ef4444" name="False Negative Rate %" />
                            </BarChart>
                        </ResponsiveContainer>
                    </section>

                    <section className="chart-section card">
                        <h3>Performance Radar - {selectedModel?.toUpperCase()}</h3>
                        <ResponsiveContainer width="100%" height={350}>
                            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                                <PolarGrid stroke="#334155" />
                                <PolarAngleAxis dataKey="subject" stroke="#cbd5e1" />
                                <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#475569" />
                                <Radar name={selectedModel} dataKey="A" stroke="#00f2fe" fill="#00f2fe" fillOpacity={0.6} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </section>
                </div>

                {/* Confusion Matrix and Detailed Table */}
                {selectedModelData && (
                    <section className="detailed-analysis">
                        <div className="card">
                            <div className="card-header">
                                <Target size={20} style={{ color: '#00f2fe' }} />
                                <h3>Decision Breakdown: {selectedModel.toUpperCase()}</h3>
                            </div>
                            <div className="table-responsive">
                                <table className="metrics-table">
                                    <thead>
                                        <tr>
                                            <th>Metric</th>
                                            <th>Value</th>
                                            <th>Significance in NIDS</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Balanced Accuracy</td>
                                            <td>{(selectedModelData.metrics.balanced_accuracy * 100).toFixed(2)}%</td>
                                            <td>Handles class imbalance (Attacks vs Normal) effectively.</td>
                                        </tr>
                                        <tr>
                                            <td>False Negative Rate (FNR)</td>
                                            <td style={{ color: '#f43f5e', fontWeight: 'bold' }}>{(selectedModelData.metrics.fnr * 100).toFixed(4)}%</td>
                                            <td>Critical. Percentage of actual attacks that went undetected.</td>
                                        </tr>
                                        <tr>
                                            <td>Detection Rate</td>
                                            <td>{(selectedModelData.metrics.detection_rate * 100).toFixed(2)}%</td>
                                            <td>Percentage of attacks successfully correctly identified.</td>
                                        </tr>
                                        <tr>
                                            <td>Brier Score</td>
                                            <td>{selectedModelData.metrics.brier_score.toFixed(6)}</td>
                                            <td>Calibration quality. Lower means probability estimates are reliable.</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </section>
                )}
            </div>
        </div>
    );
};

export default Performance;
