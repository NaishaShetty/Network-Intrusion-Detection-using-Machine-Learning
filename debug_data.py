import pandas as pd
from sklearn.preprocessing import LabelEncoder

COLUMN_NAMES = [
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
    "dst_host_srv_rerror_rate", "label"
]

print("Loading first 10 rows...")
df = pd.read_csv("KDDTrain+.txt", header=None, names=COLUMN_NAMES, nrows=10)

print("\nDataFrame info:")
print(df.info())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nColumn dtypes:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print("\nChecking categorical columns:")
print(f"  protocol_type values: {df['protocol_type'].unique()}")
print(f"  protocol_type dtype: {df['protocol_type'].dtype}")

print("\nSeparating features...")
X = df.drop("label", axis=1).copy()

print("\nX dtypes BEFORE encoding:")
print(X.dtypes)

print("\nEncoding protocol_type...")
le = LabelEncoder()
X['protocol_type'] = le.fit_transform(X['protocol_type'].astype(str))

print(f"\nAfter encoding protocol_type:")
print(f"  Values: {X['protocol_type'].values}")
print(f"  Dtype: {X['protocol_type'].dtype}")

print("\nAll X dtypes AFTER encoding protocol_type:")
print(X.dtypes)

print("\nChecking for any remaining object columns:")
object_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"  Object columns: {object_cols}")
if object_cols:
    for col in object_cols:
        print(f"    {col}: {X[col].unique()[:5]}")
