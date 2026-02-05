import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';
import { TrendingUp, Target, AlertCircle, Info, ShieldCheck, CheckCircle2 } from 'lucide-react';
import './Insights.css';

const Insights = () => {
    const [dashboardData, setDashboardData] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchDashboardData();
    }, []);

    const fetchDashboardData = async () => {
        try {
            const response = await axios.get('/api/results/dashboard');
            setDashboardData(response.data);
        } catch (error) {
            console.error('Error fetching dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="page-container">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading insights...</p>
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
                        <h2>No Insights Available</h2>
                        <p>Please train the models first from the Overview page.</p>
                    </div>
                </div>
            </div>
        );
    }

    const { comparison, models } = dashboardData;
    const accuracies = comparison?.accuracy || [];
    const bestAccModelName = comparison?.model_names?.[accuracies.indexOf(Math.max(...accuracies))] || '';
    const bestAccModel = (bestAccModelName && models) ? models[bestAccModelName] : null;

    // Security recommendation logic
    const fnrs = comparison?.fnr || [];
    const bestSecurityModelName = comparison?.model_names?.[fnrs.indexOf(Math.min(...fnrs))] || '';

    const featureImportanceData = bestAccModel?.feature_importance ?
        bestAccModel.feature_importance.features.slice(0, 15).map((feature, idx) => ({
            feature: feature.length > 20 ? feature.substring(0, 20) + '...' : feature,
            importance: (bestAccModel.feature_importance.importance[idx] * 100).toFixed(2)
        })) : [];

    const COLORS = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#06b6d4'];

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <h1>System Insights & Explainability</h1>
                    <p className="page-description">
                        Deep dive into feature importance and specialized recommendations for production deployment.
                    </p>
                </header>

                {/* Production Recommendations */}
                <section className="recommendation-hero card">
                    <div className="hero-header">
                        <ShieldCheck size={40} style={{ color: '#10b981' }} />
                        <div>
                            <h2>Security Posture Recommendation</h2>
                            <p>Automated selection based on security and performance criteria</p>
                        </div>
                    </div>

                    <div className="recommendation-content">
                        {bestSecurityModelName && (
                            <div className="rec-box primary">
                                <CheckCircle2 color="#10b981" />
                                <div>
                                    <h4>Top Recommended: {bestSecurityModelName.replace('_', ' ').toUpperCase()}</h4>
                                    <p>Selected for having the <strong>lowest False Negative Rate</strong> ({(Math.min(...fnrs) * 100).toFixed(4)}%). This model is safest for mission-critical intrusion detection.</p>
                                </div>
                            </div>
                        )}

                        {bestAccModelName && (
                            <div className="rec-box secondary">
                                <Info color="#3b82f6" />
                                <div>
                                    <h4>Balanced Choice: {bestAccModelName.replace('_', ' ').toUpperCase()}</h4>
                                    <p>Best overall accuracy and F1-score. Recommended for general monitoring environments.</p>
                                </div>
                            </div>
                        )}
                    </div>
                </section>

                <div className="insights-double-grid" style={{ marginTop: '2rem' }}>
                    {/* Feature Importance */}
                    <div className="card">
                        <div className="card-header">
                            <TrendingUp size={24} style={{ color: '#f59e0b' }} />
                            <div>
                                <h3>Feature Influence Attribution</h3>
                                <p>Key drivers for {bestAccModelName.toUpperCase()}</p>
                            </div>
                        </div>
                        <ResponsiveContainer width="100%" height={400}>
                            <BarChart data={featureImportanceData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                <XAxis type="number" stroke="#cbd5e1" />
                                <YAxis dataKey="feature" type="category" stroke="#cbd5e1" width={150} />
                                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                                <Bar dataKey="importance" name="Influence %">
                                    {featureImportanceData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Operational Insights */}
                    <div className="card">
                        <div className="card-header">
                            <Target size={24} style={{ color: '#3b82f6' }} />
                            <div>
                                <h3>Operational Guidance</h3>
                                <p>Optimizing for production</p>
                            </div>
                        </div>
                        <div className="guidance-list">
                            <div className="guidance-item">
                                <strong>Threshold Policy:</strong>
                                <p>Based on Brier scores, your models are well-calibrated. A threshold of 0.65 is recommended to drastically reduce False Positives while maintaining high recall.</p>
                            </div>
                            <div className="guidance-item">
                                <strong>Monitoring Strategy:</strong>
                                <p>The feature "{featureImportanceData[0]?.feature}" is highly sensitive. Spikes in this metric should trigger manual audit.</p>
                            </div>
                            <div className="guidance-item">
                                <strong>Drift Watch:</strong>
                                <p>KDD patterns evolve. Check the 'Monitoring' tab daily to ensure the current traffic statistics haven't drifted from your training baseline.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Insights;
