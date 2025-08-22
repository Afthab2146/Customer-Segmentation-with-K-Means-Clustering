
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

df=pd.read_csv('Mall_Customers.csv')
print(df.head())

X=df[["Age","Annual Income (k$)","Spending Score (1-100)",]].values

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

inertias=[]
sil_scores=[]

for k in range(2,11):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    labels=km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled,labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(2,11),inertias,marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")


plt.subplot(1,2,2)
plt.plot(range(2,11), sil_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")

plt.show()

kmeans=KMeans(n_clusters=6,random_state=42,n_init=10)
labels=kmeans.fit_predict(X_scaled)
df["cluster"]=labels
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

# Plot clusters
ax.scatter(X_scaled[:,0], X_scaled[:,1], X_scaled[:,2], 
           c=labels, cmap="viridis", s=50)

# Plot centroids
centers = kmeans.cluster_centers_
ax.scatter(centers[:,0], centers[:,1], centers[:,2], 
           c="red", marker="X", s=300, label="Centroids")

ax.set_xlabel("Age (scaled)")
ax.set_ylabel("Annual Income (scaled)")
ax.set_zlabel("Spending Score (scaled)")
ax.set_title("Customer Segmentation (3D K-Means)")
ax.legend()
plt.show()

figg=px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="cluster",
    symbol="cluster",
    opacity=0.8,
    size_max=10,
    title="Customer Segmentation with K-Means (3D Interactive)"
)

centers = kmeans.cluster_centers_
centers_df = pd.DataFrame(
    scaler.inverse_transform(centers),
    columns=["Age", "Annual Income (k$)", "Spending Score (1-100)"]
)
centers_df["Cluster"] = "Centroid"

figg.add_scatter3d(
    x=centers_df["Age"],
    y=centers_df["Annual Income (k$)"],
    z=centers_df["Spending Score (1-100)"],
    mode="markers",
    marker=dict(size=10, color="red", symbol="x"),
    text=[f"C{i}" for i in range(len(centers_df))],
    name="Centroids"
)

figg.show()