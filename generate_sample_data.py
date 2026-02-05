"""
Sample Data Generator
Creates a small sample CSV file for testing predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Sample data (first 100 rows from typical KDD data)
# This creates realistic-looking network traffic data for testing

def generate_sample_data(output_path: str = "sample_data.csv", num_samples: int = 100):
    """
    Generate sample network traffic data for testing
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of samples to generate
    """
    
    np.random.seed(42)
    
    # Column names (without label for prediction)
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate"
    ]
    
    data = []
    
    for i in range(num_samples):
        # Generate realistic network traffic features
        if i % 5 == 0:  # ~20% attack traffic
            # Attack pattern (e.g., DoS)
            row = [
                0,  # duration
                np.random.choice([0, 1, 2]),  # protocol_type (encoded)
                np.random.choice(range(70)),  # service (encoded)
                np.random.choice(range(11)),  # flag (encoded)
                np.random.randint(0, 10000),  # src_bytes
                0,  # dst_bytes (typical for DoS)
                0,  # land
                0,  # wrong_fragment
                0,  # urgent
                0,  # hot
                0,  # num_failed_logins
                0,  # logged_in
                0,  # num_compromised
                0,  # root_shell
                0,  # su_attempted
                0,  # num_root
                0,  # num_file_creations
                0,  # num_shells
                0,  # num_access_files
                0,  # num_outbound_cmds
                0,  # is_host_login
                0,  # is_guest_login
                np.random.randint(100, 500),  # count (high for DoS)
                np.random.randint(100, 500),  # srv_count
                np.random.uniform(0, 1),  # serror_rate
                np.random.uniform(0, 1),  # srv_serror_rate
                0,  # rerror_rate
                0,  # srv_rerror_rate
                np.random.uniform(0, 1),  # same_srv_rate
                np.random.uniform(0, 1),  # diff_srv_rate
                0,  # srv_diff_host_rate
                np.random.randint(0, 255),  # dst_host_count
                np.random.randint(0, 255),  # dst_host_srv_count
                np.random.uniform(0, 1),  # dst_host_same_srv_rate
                np.random.uniform(0, 1),  # dst_host_diff_srv_rate
                np.random.uniform(0, 1),  # dst_host_same_src_port_rate
                0,  # dst_host_srv_diff_host_rate
                np.random.uniform(0, 1),  # dst_host_serror_rate
                np.random.uniform(0, 1),  # dst_host_srv_serror_rate
                0,  # dst_host_rerror_rate
                0   # dst_host_srv_rerror_rate
            ]
        else:
            # Normal traffic pattern
            row = [
                np.random.randint(0, 100),  # duration
                np.random.choice([0, 1, 2]),  # protocol_type
                np.random.choice(range(70)),  # service
                np.random.choice(range(11)),  # flag
                np.random.randint(100, 5000),  # src_bytes
                np.random.randint(100, 5000),  # dst_bytes
                0,  # land
                0,  # wrong_fragment
                0,  # urgent
                0,  # hot
                0,  # num_failed_logins
                1,  # logged_in
                0,  # num_compromised
                0,  # root_shell
                0,  # su_attempted
                0,  # num_root
                0,  # num_file_creations
                0,  # num_shells
                0,  # num_access_files
                0,  # num_outbound_cmds
                0,  # is_host_login
                0,  # is_guest_login
                np.random.randint(1, 50),  # count
                np.random.randint(1, 50),  # srv_count
                0,  # serror_rate
                0,  # srv_serror_rate
                0,  # rerror_rate
                0,  # srv_rerror_rate
                np.random.uniform(0.8, 1.0),  # same_srv_rate
                np.random.uniform(0, 0.2),  # diff_srv_rate
                0,  # srv_diff_host_rate
                np.random.randint(0, 255),  # dst_host_count
                np.random.randint(0, 255),  # dst_host_srv_count
                np.random.uniform(0.8, 1.0),  # dst_host_same_srv_rate
                np.random.uniform(0, 0.2),  # dst_host_diff_srv_rate
                np.random.uniform(0.8, 1.0),  # dst_host_same_src_port_rate
                0,  # dst_host_srv_diff_host_rate
                0,  # dst_host_serror_rate
                0,  # dst_host_srv_serror_rate
                0,  # dst_host_rerror_rate
                0   # dst_host_srv_rerror_rate
            ]
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {num_samples} sample records")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Expected ~{num_samples // 5} attacks and ~{num_samples - num_samples // 5} normal traffic")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    generate_sample_data("sample_data.csv", num_samples=100)
    
    print("\n🎯 You can now use this file to test predictions!")
    print("   Upload 'sample_data.csv' in the Prediction page")
