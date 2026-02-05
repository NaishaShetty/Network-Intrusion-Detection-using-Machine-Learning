import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Shield, Database, Activity, AlertTriangle, Play, CheckCircle } from 'lucide-react';
import './Overview.css';

const Overview = () => {
    const [datasetInfo, setDatasetInfo] = useState(null);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [isTraining, setIsTraining] = useState(false);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchDatasetInfo();
        checkTrainingStatus();
    }, []);

    useEffect(() => {
        let interval;
        if (isTraining) {
            interval = setInterval(checkTrainingStatus, 2000);
        }
        return () => clearInterval(interval);
    }, [isTraining]);

    const fetchDatasetInfo = async () => {
        try {
            const response = await axios.get('/api/dataset/info');
            setDatasetInfo(response.data);
        } catch (error) {
            console.error('Error fetching dataset info:', error);
        } finally {
            setLoading(false);
        }
    };

    const checkTrainingStatus = async () => {
        try {
            const response = await axios.get('/api/training/status');
            setTrainingStatus(response.data);
            setIsTraining(response.data.is_training);
        } catch (error) {
            console.error('Error checking training status:', error);
        }
    };

    const startTraining = async () => {
        try {
            await axios.post('/api/train', { use_tuning: true });
            setIsTraining(true);
        } catch (error) {
            console.error('Error starting training:', error);
            alert('Failed to start training: ' + error.message);
        }
    };

    if (loading) {
        return (
            <div className="page-container">
                <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <p>Loading...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="page-container fade-in">
            <div className="container">
                <header className="page-header">
                    <div>
                        <h1>Network Intrusion Detection System</h1>
                        <p className="page-description">
                            Advanced machine learning system for detecting network intrusions and cyber threats in real-time.
                            Utilizing multiple ML algorithms to provide comprehensive security analysis.
                        </p>
                    </div>
                </header>

                <section className="info-section">
                    <div className="card">
                        <div className="card-header">
                            <Shield className="section-icon" size={24} />
                            <div>
                                <h3 className="card-title">What is Network Intrusion Detection?</h3>
                                <p className="card-description">Understanding the technology</p>
                            </div>
                        </div>
                        <div className="info-content">
                            <p>
                                Network Intrusion Detection Systems (NIDS) monitor network traffic for suspicious activity and potential threats.
                                Our ML-powered system analyzes network packets in real-time to identify:
                            </p>
                            <ul className="feature-list">
                                <li><AlertTriangle size={16} /> Unauthorized access attempts</li>
                                <li><AlertTriangle size={16} /> Denial of Service (DoS) attacks</li>
                                <li><AlertTriangle size={16} /> Port scanning and reconnaissance</li>
                                <li><AlertTriangle size={16} /> Data exfiltration attempts</li>
                                <li><AlertTriangle size={16} /> Malware communication patterns</li>
                            </ul>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <Database className="section-icon" size={24} />
                            <div>
                                <h3 className="card-title">Dataset Information</h3>
                                <p className="card-description">KDD Cup 1999 - Industry standard benchmark</p>
                            </div>
                        </div>
                        {datasetInfo && (
                            <div className="dataset-stats">
                                <div className="stat-item">
                                    <div className="stat-label">Dataset</div>
                                    <div className="stat-value">{datasetInfo.name}</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Total Features</div>
                                    <div className="stat-value">{datasetInfo.total_features}</div>
                                </div>
                                <div className="stat-item">
                                    <div className="stat-label">Sample Size</div>
                                    <div className="stat-value">{datasetInfo.sample_size.toLocaleString()}</div>
                                </div>
                            </div>
                        )}
                    </div>
                </section>

                <section className="models-section">
                    <div className="card">
                        <div className="card-header">
                            <Activity className="section-icon" size={24} />
                            <div>
                                <h3 className="card-title">Machine Learning Models</h3>
                                <p className="card-description">Multiple algorithms for comprehensive detection</p>
                            </div>
                        </div>
                        <div className="models-grid">
                            <div className="model-card">
                                <h4>Decision Tree</h4>
                                <p><strong>Purpose:</strong> Fast, interpretable classification</p>
                                <p><strong>Strength:</strong> Easy to understand feature importance</p>
                                <p><strong>Trade-off:</strong> May overfit on complex patterns</p>
                            </div>
                            <div className="model-card">
                                <h4>SGD Classifier</h4>
                                <p><strong>Purpose:</strong> Efficient online learning</p>
                                <p><strong>Strength:</strong> Fast training on large datasets</p>
                                <p><strong>Trade-off:</strong> Requires feature scaling</p>
                            </div>
                            <div className="model-card">
                                <h4>Random Forest</h4>
                                <p><strong>Purpose:</strong> Ensemble learning for robustness</p>
                                <p><strong>Strength:</strong> Excellent accuracy, handles noise well</p>
                                <p><strong>Trade-off:</strong> Slower prediction, more memory</p>
                            </div>
                            <div className="model-card">
                                <h4>XGBoost</h4>
                                <p><strong>Purpose:</strong> State-of-the-art gradient boosting</p>
                                <p><strong>Strength:</strong> Highest accuracy, handles imbalanced data</p>
                                <p><strong>Trade-off:</strong> Longer training time</p>
                            </div>
                            <div className="model-card">
                                <h4>LightGBM</h4>
                                <p><strong>Purpose:</strong> Fast gradient boosting</p>
                                <p><strong>Strength:</strong> Memory efficient, very fast</p>
                                <p><strong>Trade-off:</strong> May overfit on small datasets</p>
                            </div>
                        </div>
                    </div>
                </section>

                <section className="training-section">
                    <div className="card">
                        <div className="card-header">
                            <Play className="section-icon" size={24} />
                            <div>
                                <h3 className="card-title">Model Training</h3>
                                <p className="card-description">Train all models with hyperparameter tuning</p>
                            </div>
                        </div>

                        {trainingStatus && trainingStatus.completed && (
                            <div className="success-message">
                                <CheckCircle size={24} />
                                <span>Models trained successfully! View results in the Performance tab.</span>
                            </div>
                        )}

                        {isTraining ? (
                            <div className="training-progress">
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${trainingStatus?.progress || 0}%` }}
                                    ></div>
                                </div>
                                <p className="progress-text">
                                    {trainingStatus?.message || 'Training in progress...'}
                                </p>
                                <p className="progress-percentage">{trainingStatus?.progress || 0}%</p>
                            </div>
                        ) : (
                            <div className="training-actions">
                                <button
                                    className="btn btn-primary btn-large"
                                    onClick={startTraining}
                                    disabled={isTraining}
                                >
                                    <Play size={20} />
                                    Start Training
                                </button>
                                <p className="training-note">
                                    Training will take several minutes depending on your hardware.
                                    All models will be trained with hyperparameter tuning for optimal performance.
                                </p>
                            </div>
                        )}
                    </div>
                </section>
            </div>
        </div>
    );
};

export default Overview;
