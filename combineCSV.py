import pandas as pd

# Load both CSV files
vpn_df = pd.read_csv("vpn_features.csv")
nonvpn_df = pd.read_csv("nonVpn_features.csv")

# Add labels
vpn_df["label"] = "VPN"
nonvpn_df["label"] = "NonVPN"

# Combine into one dataframe
combined_df = pd.concat([vpn_df, nonvpn_df], ignore_index=True)

# Save as new CSV
combined_df.to_csv("combined_features.csv", index=False)

print(combined_df["label"].value_counts())
print("✅ Combined CSV saved as all_features.csv")
