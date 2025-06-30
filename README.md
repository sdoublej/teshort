teshort
teshort is a Python class designed for item reduction (short-form development) using transformer-based embeddings and KMeans clustering. It helps generate semantically representative short forms of psychological or survey items by selecting items closest to cluster centroids.

Key Features
Embedding generation
Generates embeddings for text items using SentenceTransformer (default: sentence-t5-xxl).

Optimal cluster search visualization
Provides an elbow plot using average distances of nearest neighbors to help select an appropriate number of clusters.

Clustering
Performs KMeans clustering to group items based on semantic similarity.

Short form selection
Selects a specified number of items, choosing those closest to the cluster centroids from each cluster.

Save results
Saves the selected short form items to a CSV file.

Class Structure
python
--
teshort(
    file_path='result.csv',         # File path to save results
    model_name='sentence-t5-xxl'    # Transformer model to use
)
--

Main Methods
Method	Description
embedding(df)	Generates and stores embeddings from the first column of a DataFrame.
find_nclustes()	Plots an elbow chart to assist in determining the optimal number of clusters.
cluster_only(n_clusters)	Performs KMeans clustering and returns DataFrame with cluster labels.
short(n_clusters, n_items)	Selects a specified number of items (closest to centroids) to create a short form.
sav()	Saves the selected short form items to a CSV file specified by file_path.

Input Data Requirements
The DataFrame passed to embedding(df) and short() must:

Have the first column containing the text items to be embedded.

Have a proper index to uniquely identify each item (item number, ID, etc.).

Example Usage
python
from teshort_module import teshort  # Assuming you saved it as teshort.py

# Initialize class
model = teshort(file_path='short_form.csv')

# Load data
import pandas as pd
df = pd.read_csv('items.csv', index_col=0)

# Generate embeddings
model.embedding(df)

# Visualize optimal cluster number
model.find_nclustes()

# Perform clustering (e.g., 5 clusters)
clustered_df = model.cluster_only(n_clusters=5)

# Select short form items (e.g., total 15 items across 5 clusters)
short_df = model.short(n_clusters=5, n_items=15)

# Save short form to file
model.sav()
Notes
Be sure to call embedding(df) before using cluster_only() or short().

find_nclustes() only visualizes average distance trends; it does not automatically select the cluster number.

The short() method distributes selected items approximately evenly across clusters.

Dependencies
Install required packages:

pip install sentence-transformers scikit-learn matplotlib pandas numpy
License
This code is provided for research and educational purposes. Commercial use requires permission from the author.
If you'd like, I can generate this as a properly formatted .md file or add badges (e.g., Python version, license) at the top. Let me know!


--
Sample Data
Data used in previous studies has been collected and organized separately; you may check and review them if needed.

Sample data is provided for your convenience. We recommend using the 50positive dataset in particular, as it is suitable for generating text embeddings.
