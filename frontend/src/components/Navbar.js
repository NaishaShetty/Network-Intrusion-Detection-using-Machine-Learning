import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, BarChart3, Upload, TrendingUp, Radio, Activity } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
    const location = useLocation();

    const navItems = [
        { path: '/', label: 'Overview', icon: Shield },
        { path: '/performance', label: 'Performance', icon: BarChart3 },
        { path: '/insights', label: 'Insights', icon: TrendingUp },
        { path: '/prediction', label: 'Prediction', icon: Upload },
        { path: '/simulation', label: 'Simulation', icon: Radio },
        { path: '/monitoring', label: 'Monitoring', icon: Activity }
    ];

    return (
        <nav className="navbar">
            <div className="navbar-container">
                <div className="navbar-brand">
                    <Shield className="brand-icon" size={32} />
                    <div className="brand-text">
                        <h1 className="brand-title">NIDS</h1>
                        <p className="brand-subtitle">Network Intrusion Detection</p>
                    </div>
                </div>

                <div className="navbar-links">
                    {navItems.map(({ path, label, icon: Icon }) => (
                        <Link
                            key={path}
                            to={path}
                            className={`nav-link ${location.pathname === path ? 'active' : ''}`}
                        >
                            <Icon size={20} />
                            <span>{label}</span>
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
