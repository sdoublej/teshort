
class teshort:
    #n_cluster = 클러스터를 몇 개 설정할 것인가?
    #n_items = 내가 원하는 아이템 숫자가 몇개인가?

    def __init__(self, file_path = 'result.csv', model_name='sentence-t5-xxl'):
        self.file_path = file_path
        self.model = SentenceTransformer(model_name) # 트랜스포머

        self.df = None
        self.embeddings = None
        self.kmeans = None
        self.centroids = None
        self.labels = None
        self.selected_df = None

    def embedding(self, df):
        self.df = df
        self.embeddings = self.model.encode(df.iloc[:,0])

    def find_nclustes(self):
        find_distances = []
        K = range(1, 10)  # n_neighbors 값 범위 설정

        for k in K:
            knn = NearestNeighbors(n_neighbors=k, metric='cosine')
            knn.fit(self.embeddings)
            avg_dist = np.mean(knn.kneighbors(self.embeddings)[0])
            find_distances.append(avg_dist)

        plt.figure(figsize=(10, 6))
        plt.plot(K, find_distances, 'bx-')
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Average Distance')
        plt.title('Elbow Method for Optimal K')
        plt.show()

    def cluster_only(self, n_clusters):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call 'embedding(df)' first.")

        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.kmeans.fit(self.embeddings)
        self.labels = self.kmeans.labels_
        self.centroids = self.kmeans.cluster_centers_
        self.df_cluster = self.df

        self.df_cluster['cluster'] = self.labels
        return self.df_cluster # 또는 df.columns 확인 후 원하는 열만 선택

    #반드시 인덱스, 제목, 문항 의 컬럼을 갖고 있어야함.
    def short(self, n_clusters, n_items):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call 'embedding(df)' first.")

        else:
            self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            self.kmeans.fit(self.embeddings)
            self.labels = self.kmeans.labels_
            self.centroids = self.kmeans.cluster_centers_
            self.df['labels'] = self.labels
            self.sort_df = self.df.sort_values('labels')

        self.distances = []
        for i in range(n_clusters):
            centroid = self.centroids[i]
            cluster_points = self.embeddings[self.labels == i]
            distance = np.linalg.norm(cluster_points - centroid, axis=1)
            self.distances += distance.tolist()

        self.sort_df['distances'] = self.distances

        self.n_clusters = n_clusters
        self.n_items = n_items

        # n클러스터로 나눠서 빼낸다
        self.n_lower_items_by_lables = round(self.n_items / self.n_clusters)
        self.label_numbers = self.sort_df.labels.unique()

        self.final_short_items_index = []
        for label in self.label_numbers:
            item_names = (self.sort_df[self.sort_df.labels == label].sort_values(by='distances').head(self.n_lower_items_by_lables).index).tolist()
            self.final_short_items_index += item_names

        selected_df = self.sort_df.loc[self.final_short_items_index].sort_index()
        self.selected_df = selected_df

        return selected_df


    def sav(self):
        self.selected_df.to_csv(self.file_path , index = True)