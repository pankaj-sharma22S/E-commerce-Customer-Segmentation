
import os
import time
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



mlflow.set_tracking_uri("sqlite:///mlflow.db")

OUTPUT_DIR = "experiment1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_PATH = "data/ecommerce_customers.csv"
EXPERIMENT_NAME = "Ecommerce_Customer_Segmentation_Normal_v2"

RANDOM_STATE = 42
N_INIT = 20
K_VALUES = range(2, 9)



df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())


# SELECT FEATURES

numeric_features = [
    "age", "annual_income", "spending_score",
    "purchase_frequency", "avg_order_value",
    "total_purchases", "browsing_time_minutes",
    "last_purchase_days", "return_rate",
    "social_media_engagement", "avg_review_rating",
    "cart_abandonment_rate", "discount_usage_pct",
    "customer_satisfaction_score", "preferred_shopping_hour",
    "wishlist_items", "customer_since_months",
    "email_open_rate", "click_through_rate",
    "cross_category_purchases", "repeat_purchase_rate",
    "avg_session_duration", "pages_per_session",
    "support_interactions", "product_reviews_count",
    "social_shares", "coupon_redemptions"
]

categorical_features = [
    "gender", "product_category_preference",
    "device_type", "payment_method",
    "newsletter_subscribed", "mobile_app_user",
    "loyalty_tier", "referral_source",
    "geographic_region", "preferred_shipping"
]

feature_columns = numeric_features + categorical_features

missing_columns = [
    col for col in feature_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing these required columns: {missing_columns}"
    )

X = df[feature_columns].copy()


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_features:
    mode = X[col].mode()
    fill_value = mode.iloc[0] if not mode.empty else "Unknown"
    X[col] = X[col].fillna(fill_value)

print("Missing values after cleaning:", X.isnull().sum().sum())


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_array = encoder.fit_transform(X[categorical_features])

X_encoded = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_features),
    index=X.index
)



scaler = StandardScaler()

scaled_array = scaler.fit_transform(X[numeric_features])

X_scaled = pd.DataFrame(
    scaled_array,
    columns=numeric_features,
    index=X.index
)


# Combine numeric and categorical features
X_final = pd.concat([X_scaled, X_encoded], axis=1)

print("Processed feature shape:", X_final.shape)
print("Processed feature count:", X_final.shape[1])


# ============================================================
# 7. NORMAL K-MEANS EXPERIMENTS WITH MLFLOW
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)

normal_results = []
inertias = []

for k in K_VALUES:

    with mlflow.start_run(run_name=f"Normal_KMeans_k{k}"):

        start_time = time.time()

        normal_model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT
        )

        normal_clusters = normal_model.fit_predict(X_final)

        training_time = time.time() - start_time

        silhouette = silhouette_score(
            X_final,
            normal_clusters
        )

        inertia = normal_model.inertia_
        inertias.append(inertia)

        # Log parameters
        mlflow.log_param("experiment_type", "Normal_KMeans")
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("n_init", N_INIT)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_features", X_final.shape[1])
        mlflow.log_param("n_samples", X_final.shape[0])

        # Log metrics
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("training_time_seconds", training_time)

        # Log trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            normal_model,
            name="kmeans_model"
        )

        normal_results.append({
            "K": k,
            "Inertia": inertia,
            "Silhouette Score": silhouette,
            "Training Time (seconds)": training_time
        })

        


normal_results_df = pd.DataFrame(normal_results)

print(normal_results_df.to_string(index=False))

results_path = os.path.join(
    OUTPUT_DIR,
    "normal_experiment_results.csv"
)

normal_results_df.to_csv(results_path, index=False)

# Visulaization

plt.figure(figsize=(8, 5))

plt.plot(
    normal_results_df["K"],
    normal_results_df["Inertia"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Normal K-Means Elbow Curve")
plt.xticks(list(K_VALUES))
plt.grid(True)

elbow_path = os.path.join(
    OUTPUT_DIR,
    "normal_elbow.png"
)

plt.savefig(elbow_path, dpi=300, bbox_inches="tight")
plt.close()


# SILHOUETTE SCORE GRAPH


plt.figure(figsize=(8, 5))

plt.plot(
    normal_results_df["K"],
    normal_results_df["Silhouette Score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Normal K-Means Silhouette Scores")
plt.xticks(list(K_VALUES))
plt.grid(True)

silhouette_path = os.path.join(
    OUTPUT_DIR,
    "normal_silhouette.png"
)

plt.savefig(silhouette_path, dpi=300, bbox_inches="tight")
plt.close()


best_row = normal_results_df.loc[
    normal_results_df["Silhouette Score"].idxmax()
]

best_k = int(best_row["K"])

print("\nBest K based on silhouette score:", best_k)
print("Best silhouette score:", best_row["Silhouette Score"])


# 12. TRAIN FINAL MODEL AND GENERATE CLUSTERS


final_model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=N_INIT
)

cluster_labels = final_model.fit_predict(X_final)


pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_final)

plot_df = pd.DataFrame({
    "PC1": X_2d[:, 0],
    "PC2": X_2d[:, 1],
    "Cluster": cluster_labels
})

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=plot_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="Set1",
    s=100
)

plt.title(f"Normal K-Means Clusters (K={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")

pca_plot_path = os.path.join(
    OUTPUT_DIR,
    "normal_clusters_pca.png"
)

plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
plt.close()


# Mlflow logging


with mlflow.start_run(run_name="Normal_KMeans_Summary"):

    mlflow.log_param("experiment_type", "Normal_KMeans_Summary")
    mlflow.log_param("selected_k", best_k)
    mlflow.log_param("selection_method", "Highest silhouette score")

    mlflow.log_metric(
        "best_silhouette_score",
        float(best_row["Silhouette Score"])
    )

    mlflow.log_artifact(elbow_path)
    mlflow.log_artifact(silhouette_path)
    mlflow.log_artifact(pca_plot_path)
    mlflow.log_artifact(results_path)

#prediction

customer_index = 1

original_customer = df.iloc[[customer_index]]
customer = X_final.iloc[[customer_index]]

prediction = final_model.predict(customer)

print("\nSelected customer:")
print(original_customer)

print("Customer belongs to Cluster:", prediction[0])


print("\nAll experiments completed successfully.")
print("Tracking URI:", mlflow.get_tracking_uri())


