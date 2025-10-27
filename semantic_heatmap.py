# -*- coding: utf-8 -*-

import sys
print(sys.executable)

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity


#######################################################################

# Step 1: Load the Sentence-BERT model
model = SentenceTransformer('all-MPNet-base-v2')#('all-MiniLM-L6-v2')
#all-MiniLM-L12-v2		Slightly better than L6, slower
#all-MPNet-base-v2

# Step 2: Define a list of texts (sentences, documents, etc.)
# To read a specific sheet, use the `sheet_name` parameter.
df = pd.read_excel('traits.xlsx', sheet_name='list_for_model')

# Use the column header to select the data and convert it to a list
traits_list = df['Traits'].tolist()

# creare embeddings from list of words
embeddings = model.encode(traits_list)

# calculate teh cosine similarity
similarity_matrix_traits = cosine_similarity(embeddings)



# Print the similarity matrix
print("Semantic Similarity Matrix:")
np.set_printoptions(precision=3, suppress=True) 
print(similarity_matrix_traits)
#############################################################################################################
#
# PLOT SEMANTIC SIMILARITY MATRIX
#
#########################################################
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Visualizing the similarity matrix
sns.heatmap(similarity_matrix_traits, annot=False, cmap="YlOrBr", xticklabels= traits_list, yticklabels= traits_list, ax=ax)

# Use tick_params to set the font size of the tick labels on both axes
ax.tick_params(axis='both', which='major', labelsize=4) # Change '8' to your desired font size

# Optional: Adjust the figure size to prevent labels from overlapping
fig.set_size_inches(6, 6)

plt.title("Semantic Similarity Heatmap")
#plt.savefig('semantic_similarity/similarity_matrix_heatmap.pdf', bbox_inches='tight')

plt.show()

np.save("similarity_matrix_traits.npy",similarity_matrix_traits )
similarity_matrix_rounded = np.round(similarity_matrix_traits, 3)

################################################################################ 
###################################################################
# 
# align by order of eigenvalues 
#
##################################################################################
###################################################################

#eigenvalues, eigenvectors = np.linalg.eig(similarity_matrix_traits)
eigenvalues, eigenvectors = np.linalg.eigh(similarity_matrix_traits)


sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order

sorted_eigenvectors = eigenvectors[:, sorted_indices] # sort eigenvecrors

#first non trivial eigenvector
first_non_trivial_eigenvector = sorted_eigenvectors[:, 1]  # Corresponding to the second largest eigenvalue

# Option 1: sort by ascending (negative to positive)
#sorted_indices_by_non_trivial = np.argsort(first_non_trivial_eigenvector)# ???

# Option 2: sort by absolute value (strongest loadings, ignoring sign)
sorted_indices_by_non_trivial = np.argsort(np.abs(first_non_trivial_eigenvector))[::-1]

aligned_matrix = similarity_matrix_traits[sorted_indices_by_non_trivial, :][:, sorted_indices_by_non_trivial]
#The result is a matrix where traits with similar loadings on the eigenvector are placed close together, making clusters or patterns easier to spot when visualizing.

traits_reordered_non_trivial = [traits_list [i] for i in sorted_indices_by_non_trivial]


# 5.  plot before/after
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
im0 = axs[0].imshow(similarity_matrix_traits, cmap="YlOrBr")
axs[0].set_title("Original Matrix")
axs[0].set_xticks(np.arange(len(traits_list)))
axs[0].set_yticks(np.arange(len(traits_list)))
axs[0].set_xticklabels(traits_list, rotation=90)
axs[0].set_yticklabels(traits_list, rotation=0)
axs[0].tick_params(axis='x', labelsize=5)
axs[0].tick_params(axis='y', labelsize=5)
################
im1 = axs[1].imshow(aligned_matrix, cmap="YlOrBr")
axs[1].set_title("Aligned by Eigenvector - first non trivial")
axs[1].set_xticks(np.arange(len(traits_reordered_non_trivial)))
axs[1].set_yticks(np.arange(len(traits_reordered_non_trivial)))
axs[1].set_xticklabels(traits_reordered_non_trivial, rotation=90)
axs[1].set_yticklabels(traits_reordered_non_trivial, rotation=0)
axs[1].tick_params(axis='x', labelsize=4)
axs[1].tick_params(axis='y', labelsize=4)
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
plt.tight_layout()
#plt.savefig('semantic_similarity/aligned_byeigenvector_firstnontrivial_2.pdf', bbox_inches='tight')
#plt.show()

###############################################
# plot eigenvalues
#
############################
#list 0 to 58
#arr = list(range(0, len(traits_list)))
# plot eigenvalues aligned
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))



#axs[0].scatter(arr, eigenvalues[sorted_indices])
#axs[0].set_title("Eigenvalues aligned")
#axs[0].set_ylabel("Eigenvalues")     
#axs[0].set_xticks(np.arange(len(traits_list)))       # label for the y-axis
#axs[0].tick_params(axis='x', labelsize=5)
#axs[0].set_xticklabels([traits_list[x] for x in arr], rotation=90)

# reordered this makes no sense1!!
#axs[1].scatter(arr, eigenvalues[sorted_indices_by_non_trivial])
#axs[1].set_title("Eigenvalues aligned - by non trivial")
#axs[1].set_ylabel("Eigenvalues")     
#axs[1].set_xticks(np.arange(len(traits_list)))       # label for the y-axis
#axs[1].tick_params(axis='x', labelsize=4)
#axs[1].set_xticklabels([traits_reordered_non_trivial[x] for x in arr], rotation=90)
#plt.savefig('semantic_similarity/eigenvalues.pdf', bbox_inches='tight')
#plt.show()

#####################################################################
#####################

########################################################################################################################
###########################################################################
#
# Hierarchical Clustering
#
################################################################################################


import scipy.cluster.hierarchy as sch
#from sch import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist, squareform

#calculate distance matrix
#distance_matrix  = 1- similarity_matrix_traits

# ✅ Step 2: Set diagonal to 0 (must be exactly zero)
#np.fill_diagonal(distance_matrix, 0)

# Step 2: Condense the distance matrix (for linkage)
#condensed_distance = squareform(distance_matrix)
#
# Step 3: Perform hierarchical clustering (e.g., with 'average' linkage)
#linkage_matrix = linkage(condensed_distance, method='average')  # 
#linkage_matrix = sch.linkage(condensed_distance, method='single')  # average, complete single ward
#method="average" #can be replaced with 'ward', 'single', 'complete', etc., depending on clustering style.

#linkage_matrix = sch.linkage(distance_matrix, method='ward')  # average, complete single ward
Z = sch.linkage(first_non_trivial_eigenvector[sorted_indices_by_non_trivial].reshape(-1,1), method='ward')


#linkage_matrix = sch.linkage(condensed_distance, method='single')  # average, complete single ward








from scipy.cluster.hierarchy import fcluster

num_clusters = 3  # or pick threshold with 't' param
clusters = fcluster(Z, num_clusters, criterion='maxclust')


cluster_boundaries = []
prev_cluster = clusters[0]
for i in range(1, len(clusters)):
    if clusters[i] != prev_cluster:
        cluster_boundaries.append(i)
    prev_cluster = clusters[i]
    
    
for boundary in cluster_boundaries:
    axs[1].axhline(boundary - 0.5, color='black', linewidth=1)
    axs[1].axvline(boundary - 0.5, color='black', linewidth=1)
    

cluster_colors = ['red', 'blue', 'green']  # extend if needed

for tick_label, cluster_idx in zip(axs[1].get_xticklabels(), clusters):
    tick_label.set_color(cluster_colors[cluster_idx-1])
for tick_label, cluster_idx in zip(axs[1].get_yticklabels(), clusters):
    tick_label.set_color(cluster_colors[cluster_idx-1])
    
    
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('semantic_similarity/aligned_byeigenvector_firstnontrivial_2_clusters.pdf', bbox_inches='tight')
plt.show()



# Get the cluster ordering

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(Z, no_plot=False,   labels=traits_list, leaf_rotation=90, leaf_font_size=5) #color_threshold=2.2

#dendrogram = sch.dendrogram(linkage_matrix, no_plot=False, color_threshold=2.2,  labels=traits_list, leaf_rotation=90, leaf_font_size=5) #color_threshold=2.2
plt.title('Dendrogram - ward method thrshold 2.2')
plt.xlabel('Samples')
plt.ylabel('Distance')
#plt.savefig('semantic_similarity/dendrogram_method_ward_threshold.pdf', format='pdf')
plt.show()


order = dendrogram['leaves']  # This gives the row/column order to reorder the matrix


# Step 5: Reorder the matrix and traits
reordered_matrix = similarity_matrix_traits[order][:, order]

traits_reordered = [traits_list[i] for i in order]

# Plot the reordered matrix as a heatmap

# Set the tick locations for both axes to be from 0 to num_labels-1
reordered_matric_rounded = np.round(reordered_matrix, 3)



fig, ax = plt.subplots()


sns.heatmap(reordered_matrix, annot=False, cmap="YlOrBr", xticklabels= traits_reordered, yticklabels= traits_reordered, ax=ax)
plt.title("Reordered Cosine Similarity Matrix")

# Use tick_params to set the font size of the tick labels on both axes
ax.tick_params(axis='both', which='major', labelsize=5) # Change '8' to your desired font size
fig.set_size_inches(6, 6)

# Optional: Adjust the figure size to prevent labels from overlapping

plt.title("Reordered Semantic Similarity Heatmap")
#plt.savefig('semantic_similarity/similarity_matrix_heatmap_reordered.pdf', bbox_inches='tight')

plt.show()

num_clusters = 4
clusters = fcluster(Z, num_clusters, criterion='maxclust')


"""

##########################################################################
#
#   Spectral clustering
#
#
#####################################
"""
from sklearn.cluster import SpectralClustering
import numpy as np

# similarity_matrix: your precomputed semantic similarity matrix

# Run spectral clustering

n_clusters = 3  # Change this based on your expected clusters



sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed') #, assign_labels='kmeans'
labels = sc.fit_predict(similarity_matrix_traits)



# Step 4: Show results
print("Cluster labels:", labels)


from sklearn.decomposition import PCA
reduced = PCA(n_components=2).fit_transform(similarity_matrix_traits)
plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='rainbow')
for i in range(len(reduced)):
    plt.text(reduced[i, 0], reduced[i, 1], traits_list[i], fontsize=4, ha='right')
plt.title("Spectral Clustering Results (PCA Visualization) 4 clusters 2 comp")
#plt.savefig('semantic_similarity/Spectral_clusters_PCA_2comp_4clusters.pdf', bbox_inches='tight')



plt.show()


"""

#####
###############
"""
# 4. Get cluster labels (3 clusters)
cluster_labels = fcluster(linkage_matrix, t=3, criterion='maxclust')

# 5. Order the matrix by cluster (optional: within-cluster sort)
# First sort by cluster, then by hierarchical structure within each
cluster_order = np.argsort(cluster_labels)
ordered_indices = cluster_order

reordered_matrix = similarity_matrix_rounded[ordered_indices][:, ordered_indices]
traits_reordered = [traits_list[i] for i in ordered_indices]
clusters_reordered = [cluster_labels[i] for i in ordered_indices]


fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.imshow(reordered_matrix, cmap='Blues')

# Set tick labels
ax.set_xticks(np.arange(len(traits_reordered)))
ax.set_yticks(np.arange(len(traits_reordered)))
ax.set_xticklabels(traits_reordered, rotation=90, fontsize=6)
ax.set_yticklabels(traits_reordered, fontsize=6)

# Show cluster labels (color bar for clusters)
for i, label in enumerate(clusters_reordered):
    ax.get_xticklabels()[i].set_color(['red', 'green', 'blue'][label - 1])
    ax.get_yticklabels()[i].set_color(['red', 'green', 'blue'][label - 1])

plt.title("Similarity Matrix with 3 Clusters")
plt.colorbar(cax)
plt.tight_layout()
plt.show()
#Your similarity matrix might reflect one large, loosely connected group and two smaller, tightly connected groups. Hierarchical clustering is sensitive to that structure — it will often:

#Group tight mini-clusters early

#Leave the rest as a big "leftover" cluster

##################################
# spectral clustering implementation
##################################################
#assert 
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np



assert np.allclose(similarity_matrix_traits, similarity_matrix_traits.T)

S = similarity_matrix_traits.copy()
threshold = 0.7  # only keep similarities above this
S[S < threshold] = 0

# ------------------------------------
# 2. Build the normalized Laplacian
# ------------------------------------
# Degree matrix
D = np.diag(similarity_matrix_traits.sum(axis=1))
# Normalized Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(similarity_matrix_traits, axis=1)))
L_sym = np.eye(len(similarity_matrix_traits)) - D_inv_sqrt @ similarity_matrix_traits @ D_inv_sqrt

# ------------------------------------
# 3. Compute eigenvalues and eigenvectors
# ------------------------------------
eigenvals, eigenvecs = np.linalg.eigh(L_sym)

# Sort by eigenvalue
idx = np.argsort(eigenvals)
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# ------------------------------------
# 4. Plot eigenvalue spectrum (the "eigengap" plot)
# ------------------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, len(eigenvals)+1), eigenvals, marker='o')
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (Laplacian)")
plt.title("Eigenvalue spectrum (find the 'elbow')")
plt.grid(True)

plt.savefig('semantic_similarity/automatic_numberclusters_fund the elbow.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------
# 5. Automatically estimate k (number of clusters)
# ------------------------------------
# Find the largest gap between consecutive eigenvalues
diffs = np.diff(eigenvals)
k_est = np.argmax(diffs[:59]) + 1   # Only consider first 20 typically
print(f"Estimated number of clusters (eigengap heuristic): k = {k_est}")

# ------------------------------------
# 6. Cluster in eigenvector space
# ------------------------------------
U = eigenvecs[:, :k_est]          # take the first k_est eigenvectors
U_norm = normalize(U)             # row-normalize
kmeans = KMeans(n_clusters=k_est, n_init=20, random_state=42)
labels = kmeans.fit_predict(U_norm)

# ------------------------------------
# 7. Inspect clusters
# ------------------------------------
#print("Cluster labels:", labels)

S = similarity_matrix_traits.copy()
threshold = 0.35  # only keep similarities above this
S[S < threshold] = 0


D = np.diag(S.sum(axis=1))
# Normalized Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(S, axis=1)))
L_sym = np.eye(len(    S)) - D_inv_sqrt @ S @ D_inv_sqrt

# ------------------------------------
# 3. Compute eigenvalues and eigenvectors
# ------------------------------------
eigenvals, eigenvecs = np.linalg.eigh(L_sym)

# Sort by eigenvalue
idx = np.argsort(eigenvals)
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

# ------------------------------------
# 4. Plot eigenvalue spectrum (the "eigengap" plot)
# ------------------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, len(eigenvals)+1), eigenvals, marker='o')
plt.xlabel("Eigenvalue index")
plt.ylabel("Eigenvalue (Laplacian)")
plt.title("Eigenvalue spectrum (find the 'elbow')")
plt.grid(True)

plt.savefig('semantic_similarity/automatic_numberclusters_fund the elbow.pdf', bbox_inches='tight')
plt.show()

# ------------------------------------
# 5. Automatically estimate k (number of clusters)
# ------------------------------------
# Find the largest gap between consecutive eigenvalues
diffs = np.diff(eigenvals)
k_est = np.argmax(diffs[:58]) + 1   # Only consider first 20 typically
print(f"Estimated number of clusters (eigengap heuristic): k = {k_est}")

U = eigenvecs[:, :k_est]          # take the first k_est eigenvectors
U_norm = normalize(U)             # row-normalize
kmeans = KMeans(n_clusters=k_est, n_init=20, random_state=42)
labels = kmeans.fit_predict(U_norm)
