""" 
author: My Gia Nguyen (my.ngngia@gmail.com)
"""

import numpy as np
import random
import math
import joblib

class KMeansModel():
    def __init__(self, data, k, max_iters=100, tol=1e-4):
        self.data = data
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centers = []
        self.labels = []
        
    def initialize_centers(self):
        indices = np.random.choice(len(self.data), self.k, replace=False)
        self.centers = self.data[indices]
        return np.array(self.centers)
    
    def predict(self):
        self.labels = [] 
        for point in self.data:
            distance_to_cluster = [math.dist(point, c) for c in self.centers]            
            self.labels.append(distance_to_cluster.index(min(distance_to_cluster)))
        return self.labels
    
    def update_centers(self):
        new_centers = []
        for i in range(self.k):
            cluster_points = self.data[np.array(self.labels) == i]
            if len(cluster_points) > 0:
                new_centers.append(np.mean(cluster_points, axis=0))
            else:
                new_centers.append(self.centers[i])
        
        return np.array(new_centers)
    
    def fit(self):
        self.centers = self.initialize_centers()
        prev_centers = np.copy(self.centers)

        for i in range(self.max_iters):
            self.labels = self.predict()
            self.centers = self.update_centers()
            center_shifts = np.linalg.norm(self.centers - prev_centers, axis=1)
            if np.max(center_shifts) < self.tol:
                print(f"Converged at iteration {i+1}")
                break
            prev_centers = np.copy(self.centers)
    
    def compute_inertia(self):
        inertia = 0
        for i in range(len(self.centers)):
            cluster_points = self.data[np.array(self.labels) == i]
            inertia += np.sum(np.linalg.norm(cluster_points - self.centers[i], axis=1) ** 2)
        return inertia

    def save_model(model, file_name):
        joblib.dump(model, file_name)
    
    def load_model(file_name):
        return joblib.load(file_name)


