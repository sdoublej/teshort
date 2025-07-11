
# teshort
--

'teshort' is a Python class designed for item reduction (short-form development) using transformer-based embeddings and reduction, KMeans clustering. It helps generate semantically representative short forms of psychological or survey items by selecting items closest to cluster centroids.

---
🚩 Why is this important?
teshort is a Python tool for reducing questionnaire items without any response data. While traditional methods rely on large-scale data, teshort selects key items based only on semantic meaning using transformer-based language models.

---
# How to Use
### install
```python
pip install teshort
```    
### import
```python
from teshort import teshort
import pandas as pd
import numpy as np
```    
### load df
```python
df = pd.read_csv('Your file path/sample_filesample_file/full_50_items_positive_form_sample.csv', index_col = 0)
``` 
⚠️ Note: Your file must follow the same format as the sample file.
The column containing the item texts must be in the first column
Otherwise, the embedding and clustering process may fail.
### model
```python
model = teshort.teshort(file_path = 'result.csv', model_name='sentence-t5-xxl')
```
🔄  The file_path parameter is the path to your item file (usually a .csv file) that contains the full set of items to be reduced.
🔄  The model_name specifies the pretrained sentence embedding model to be used for transforming items into semantic vectors (ex: sentence-t5-xxl, xl, aall-MiniLM-L6-v2.)  
---
# First_step. Embedding
```python
model.embedding(df)
```
if you want to see the results of embedding, try 'model.ebeddings'

---
# Second_step. Reduction
```python
model.reduction( n_components = 32,  metric='cosine', random_state=42 )
```
### 🔧 Parameters
- n_components: Number of dimensions to reduce to (default: 32)

- metric: Distance metric used (e.g., 'cosine', 'euclidean')

- random_state: Set a seed for reproducibility (important for consistent clustering results)
- 
⭐ This step uses UMAP internally. Dimensionality reduction helps group semantically similar items more clearly.

---
# Third_step. cluster
Now, get cluster
```python

clusterd_df = model.cluster( n_clusters = cluster number what you want )
clusterd_df
```
🔍 Not sure how many clusters to choose? do 'model.find_nclusters()'
📊 This will generate a visualization (e.g., silhouette scores) to help you decide the optimal number of clusters.
---
# Final! make shortForm
```ptyhon
selected_df = model.short(n_items = number_what you want)
selected_df
```
📌 Notes
- Items are selected to best represent each semantic cluster.

- n_items can be any number ≤ total number of items.
** but may differ slightly because items are selected per cluster
---
💾 Save Selected Items
You can save the selected short-form items to the same file path you initially provided.
```python
model.sav()
```
This will save selected_df as a CSV file to the path specified in file_path.
It overwrites the original file, so make sure to back up if needed.




----
# 🎯 Summary
teshort allows you to reduce psychological or questionnaire items without response data, using transformer-based semantic embeddings.
It may be enable theory-driven short form development that is fast, efficient, and scalable.

---
### Q. Do I need a high-end GPU?

Not necessarily.

While the default model (`sentence-t5-xxl`) is large, `teshort` uses it **only for embedding** (not training).  
This makes it runnable on most **home computers**, even without a GPU.

> ✅ If your system struggles, try:  
> - A lighter model like `'intfloat/e5-base'`  
> - **Google Colab** (tested: works fine for free on google.colab)
---
