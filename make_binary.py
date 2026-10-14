import sys
 
import pandas as pd
 
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
 
df["binary"] = (df["disease"] != "NO").astype(int)
df["binary_name"] = df["binary"].map({0: "normal", 1: "lesion"})
 
df.to_csv(dst, index=False)
print(df["binary_name"].value_counts())
 
