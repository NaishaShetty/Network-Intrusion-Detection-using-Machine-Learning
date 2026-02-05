
import pandas as pd
import numpy as np

# Define columns (41 features)
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

# Create dummy user data (Normal traffic)
# Based on typical KDD normal traffic patterns
data = []

# Row 1: Normal HTTP traffic
data.append([
    0, 'tcp', 'http', 'SF', 215, 45076, 
    0, 0, 0, 0, 0, 1, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 1, 1, 0.00, 
    0.00, 0.00, 0.00, 1.00, 
    0.00, 0.00, 0, 
    0, 0.00, 
    0.00, 0.00, 
    0.00, 0.00, 
    0.00, 0.00, 
    0.00
])

# Row 2: Another Normal HTTP
data.append([
    0, 'tcp', 'http', 'SF', 162, 4528, 
    0, 0, 0, 0, 0, 1, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 2, 2, 0.00, 
    0.00, 0.00, 0.00, 1.00, 
    0.00, 0.00, 1, 
    1, 1.00, 
    0.00, 1.00, 
    0.00, 0.00, 
    0.00, 0.00, 
    0.00
])

# Row 3: Possible Attack (Neptune/Syn Flood pattern - high count, serror)
data.append([
    0, 'tcp', 'private', 'REJ', 0, 0, 
    0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    0, 0, 123, 6, 0.00, 
    0.00, 1.00, 1.00, 0.05, 
    0.07, 0.00, 255, 
    6, 0.02, 
    0.07, 0.00, 
    0.00, 0.00, 
    0.00, 1.00, 
    1.00
])

df = pd.DataFrame(data, columns=columns)

# Save WITHOUT header to match KDD format expectation for the API
# API expects: 41 columns, no header
df.to_csv("sample_network_traffic.csv", index=False, header=False)

print("Created sample_network_traffic.csv with 3 rows (41 columns, no header)")
print("File size:", len(df.to_csv(index=False, header=False).encode('utf-8')), "bytes")
