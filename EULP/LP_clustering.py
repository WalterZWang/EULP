from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

class LoadProfileCluster:
    def __init__(self):
        self.name = 'Clustering'

    @staticmethod
    def prepare_kmeans_data(df_features, standardize_features=True):
        scaler = preprocessing.StandardScaler()
        if standardize_features:
            out = scaler.fit_transform(df_features)
        else:
            out = df_features.to_numpy()
        return out
    
    @staticmethod
    def kmeans_elbow_plot(cluster_data, k_max=25, metric='distortion', dir_save=None):
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2,k_max))
        visualizer.fit(cluster_data) # Fit the data to the visualizer
        if dir_save != None:
            visualizer.poof(outpath=dir_save)
        visualizer.show()        # Finalize and render the figure
        return visualizer.elbow_value_, visualizer.elbow_score_
        
    @staticmethod
    def kmeans_silhouette_plot(cluster_data, k, dir_save=None):
        # Use the quick method and immediately show the figure
        model = KMeans(k, random_state=42)
        visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
        visualizer.fit(cluster_data)        # Fit the data to the visualizer
        if dir_save != None:
            visualizer.poof(outpath=dir_save)
        visualizer.show()        # Finalize and render the figure
        
    @staticmethod
    def find_optimal_k(cluster_data):
        return 42
        
        
class TimeDomainCluster(LoadProfileCluster):
    '''
    Child class of LoadProfileCluster
    '''
    def method_1(self):
        return 42

    
class FrequencyDomainCluster(LoadProfileCluster):
    '''
    Child class of LoadProfileCluster
    '''
    def method_1():
        return 42
    
