import pandas as pd
import numpy as np
from kmeans_model import KMeansModel
from datetime import datetime 

class ProcessingModel:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
    
    def rfm(self):
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'], "%Y-%m-%d %H:%M:%S")
        max_date = self.data['InvoiceDate'].max()
        recency_data = self.data.groupby('Customer ID')['InvoiceDate'].max().reset_index()
        recency_data['Recency'] = (max_date - recency_data['InvoiceDate']).days
        recency_data = recency_data[['Customer ID', 'Recency']]

        frequency_data = self.data.groupby('Customer ID')['Invoice'].nunique().reset_index()
        frequency_data.rename(columns={'Invoice': 'Frequency'}, inplace=True)

        self.data['MonetaryValue'] = (self.data['Price'] * self.data['Quantity']).round(2)
        monetary_data = self.data.groupby('Customer ID')['MonetaryValue'].sum().round(2).reset_index()
        monetary_data.rename(columns={'MonetaryValue': 'Monetary'}, inplace=True)

        customer_info = self.data.groupby('Customer ID').first()[['Gender', 'Location', 'Age']].reset_index()

        rfm_data = recency_data.merge(frequency_data, on='Customer ID') \
                        .merge(monetary_data, on='Customer ID') \
                        .merge(customer_info, on='Customer ID')
        # self.rfm = pd.DataFrame(rfm_data)
        return rfm_data.head(10)
    
    def elbow(self):
        rfm_data = self.rfm[["Recency", "Frequency", "Monetary"]]
        sse = []
        for k in range(1, 11):
            kmeans = KMeansModel(rfm_data, k)
            kmeans.fit()
            sse.append(kmeans.compute_inertia())
        return sse
    
    def clustering(self, k):
        rfm_data = self.rfm()[["Recency", "Frequency", "Monetary"]]
        kmeans = KMeansModel(rfm_data, k)
        kmeans.fit()
        labels = kmeans.predict()
        rfm_data["Cluster"] = labels
        return rfm_data
        
    def get_top_customers(self, isDesc):
        top_customers = self.data.groupby('Customer ID')['Invoice'].nunique().reset_index()
        top_customers = top_customers.sort_values(by='Invoice', ascending=isDesc)
        return top_customers.head(10)
    
    def get_top_location(self, isDesc):
        top_locations = self.data.groupby('Location')['Invoice'].nunique().reset_index()
        top_locations = top_locations.sort_values(by='Invoice', ascending=isDesc)
        return top_locations.head(10)

    def get_top_products(self, isDesc):
        top_purchased = self.data.groupby('StockCode')['Quantity'].sum().reset_index()
        top_purchased = top_purchased.sort_values(by='Quantity', ascending=isDesc)
        descriptions = self.data[['StockCode', 'Description']].drop_duplicates()
        result = pd.merge(top_purchased, descriptions, on='StockCode', how='left')
        return result.head(10)
    
