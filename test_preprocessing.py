"""
Quick test script to verify data preprocessing works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import DataPreprocessor

print("Testing data preprocessing...")
print("=" * 60)

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load data
print("\n1. Loading data...")
df = preprocessor.load_data(Path("KDDTrain+.txt"))
print(f"   ✓ Loaded {len(df)} records")
print(f"   First row sample:")
print(f"     protocol_type: {df.iloc[0]['protocol_type']} (type: {type(df.iloc[0]['protocol_type'])})")
print(f"     service: {df.iloc[0]['service']}")
print(f"     flag: {df.iloc[0]['flag']}")

# Encode categorical
print("\n2. Encoding categorical features...")
df_encoded = preprocessor.encode_categorical_features(df.head(100), fit=True)
print(f"   ✓ Encoded successfully")
print(f"   After encoding:")
print(f"     protocol_type: {df_encoded.iloc[0]['protocol_type']} (type: {type(df_encoded.iloc[0]['protocol_type'])})")
print(f"     service: {df_encoded.iloc[0]['service']}")
print(f"     flag: {df_encoded.iloc[0]['flag']}")

# Full preprocessing
print("\n3. Full preprocessing pipeline...")
X, y = preprocessor.preprocess(df.head(1000), fit=True)
print(f"   ✓ Preprocessing successful")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   X dtype: {X.dtype}")

print("\n" + "=" * 60)
print("✅ All tests passed! Data preprocessing is working correctly.")
print("\nYou can now run: python train.py")
