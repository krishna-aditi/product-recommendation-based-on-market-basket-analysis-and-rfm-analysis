# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:20:47 2022

@author: krish
"""

# %% Libraries
import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import calendar
import json

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Other viz
import seaborn as sns
import matplotlib.pyplot as plt

# Runtime Configuration Parameters for Matplotlib
plt.rcParams['font.family'] = 'Verdana'
plt.style.use('ggplot')

# Warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For Customer Segmentation
# Clustering
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

def main():
    # Set page config
    st.set_page_config(layout="wide")

    # st. set_page_config(layout="wide")
    page = st.sidebar.radio("Navigation Pane:", ["Product Recommendation using Market Basket Analysis", "Customer Segmentation based on RFM Analysis", "Dashboard"])
   
    #Add sidebar to the app
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("##### Made by: :computer:")
    st.sidebar.markdown("### Aditi Krishna :dog:")
    
    # Read clean dataset
    retail = pd.read_csv('CleanRetailData.csv') 
    
    # List of all countries in dataset
    country_list = list(dict(retail['Country'].value_counts()).keys())
    
    # Subsetting retail dataframe based on country
    def choose_country(country = "all", data = retail):
        if country == "all":
            return data
        else:
            temp_df = data[data["Country"] == country]
            temp_df.reset_index(drop= True, inplace= True)
            return temp_df
             
    # For United Kingdom, since it contains majority of data
    uk_retail = choose_country("United Kingdom")
        
    def cluster_plot(data_frame):
        fig = px.scatter_3d(data_frame, x = 'Recency', y='Frequency', z='Monetary',
                  color='Clusters', opacity = 0.8, width=600, height=600, template="plotly_dark")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True, height=600)
        
    def kmeans_on_df():
        # Scaling Recency, Frequency, Monetary and RFM_Score columns
        scaler = StandardScaler()
        # Subset
        rfm_scaled = rfm_new[['Recency','Frequency','Monetary','RFM_Score']]
        rfm_scaled = scaler.fit_transform(rfm_scaled)
        rfm_scaled = pd.DataFrame(rfm_scaled, columns = ['Recency','Frequency','Monetary','RFM_Score'])
        
        # Fit Kmeans at n_clusters = 4
        kmeans = KMeans(n_clusters=4, init='k-means++',n_init=10,max_iter=50,verbose=0)
        kmeans.fit(rfm_scaled)
        
        # Assigning Clusters
        rfm_new['Clusters'] = kmeans.labels_
        
        return rfm_new
    
    def plot_pcts(df, string):
        # https://sakizo-blog.com/en/607/
        fig_target = go.Figure(data=[go.Pie(labels=df.index,
                                    values=df[string],
                                    hole=.3)])
        fig_target.update_layout(showlegend=False,
                                 height=500,
                                 margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
        fig_target.update_traces(textposition='inside', textinfo='label+percent')
        fig_target.update_traces(marker=dict(colors=['lightcyan', 'cyan', 'royalblue', 'darkblue']))
        return st.plotly_chart(fig_target, use_container_width=True)
    
    # Function to group on Month/Date/Day of the Week/Week of the Year/Time of the Day
    def group_sales_quantity(df, feature):
        df = df[[f'{feature}','Quantity','Sales Revenue']].groupby([f'{feature}']).sum().sort_values(by= 'Sales Revenue', ascending = False).reset_index()
        return df
# %%    
    # First page
    if page == "Product Recommendation using Market Basket Analysis":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Product Recommendation using Market Basket Analysis</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("")

        # Pick country 
        st.markdown('### Choose Country:')
        option = st.selectbox('', country_list)
        country_retail = choose_country(option)
        
        # Display
        AgGrid(country_retail, theme='blue', height = 200, width = 150)
                    
        # List of all products
        product_catalog = list(country_retail['Description'].unique())
        
        #### Need to have a drop down to choose country and then filter dataset based on that
        st.markdown('### Choose Product:')
        prod_option = st.selectbox('', product_catalog)
        
        # Opening JSON file
        with open('item_sets.json') as json_file:
            data = json.load(json_file)
       
        # Display    
        if len(data[prod_option]) == 0:
            st.error("Oops! No product recommendations available yet! Please select a different item.")
        else:
            st.markdown("####")
            st.success("##### People also bought...")
            for d in data[prod_option]:
                if d:
                    st.markdown("- " + d)
# %%
    if page == "Customer Segmentation based on RFM Analysis":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Customer Segmentation based on RFM Analysis</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("")

        try:                   
            col1, col2, col3= st.columns([5, 1, 10])
        
            with col1:
                # About RFM
                st.markdown('## What is RFM Analysis?')
                st.markdown('It is a customer segmentation technique that uses past purchase behavior to segment customers. To perform RFM analysis, we divide customers into four equal groups according to the distribution of values for recency, frequency, and monetary value.')
                st.markdown('**1. Recency (R)**: Time since last purchase')
                st.markdown('**2. Frequency (F)**: Total number of purchases')
                st.markdown('**3. Monetary Value (M)**: Total monetary value')
                
            with col3:
                # Pick country 
                st.markdown('## Choose Country:')
                rfm_country = st.selectbox('', country_list)
                rfm_country_df = choose_country(rfm_country)
               
                # We need a reference day to perform the RFM Analysis
                # In this case the day after the last recorded date in the dataset plus a day
                rfm_country_df['InvoiceDate'] = pd.to_datetime(rfm_country_df['InvoiceDate'])
                ref_date = rfm_country_df['InvoiceDate'].max() + dt.timedelta(days=1)
                
                # Remove 'Guest Customer' 
                rfm_country_df = rfm_country_df[rfm_country_df['CustomerID'] != "Guest Customer"]
                
                # Aggregating over CustomerID
                rfm_new = rfm_country_df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (ref_date - x.max()).days,
                                            'InvoiceNo': lambda x: x.nunique(),
                                            'Sales Revenue': lambda x: x.sum()})
                # Calculate quantiles
                rfm_new.columns = ['Recency', 'Frequency', 'Monetary']
                rfm_new["R"] = pd.qcut(rfm_new['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
                rfm_new["F"] = pd.qcut(rfm_new['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
                rfm_new["M"] = pd.qcut(rfm_new['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
                
                # Calculate RFM Score
                rfm_new['RFM_Score'] = (rfm_new['R'].astype(int) + rfm_new['F'].astype(int) + rfm_new['M'].astype(int))
                
                # New RFM Dataframe
                rfm_new.reset_index(inplace=True)
                
                # K-means
                df = kmeans_on_df()
                
                # Display merged dataframes   
                AgGrid(df, theme='blue', height = 200, width = 150)
            
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">Identified Clusters</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            cluster_plot(df)
            
            # Pie charts
            rfm_stats = df[["Clusters","RFM_Score", "Recency", "Frequency", "Monetary"]].groupby("Clusters").agg(["mean"])
            rfm_stats.columns = ["RFM_Score_Mean", "Recency_Mean", "Frequency_Mean", "Monetary_Mean"]
            
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">Pie-plot Distribution of Clusters Based on RFM Analysis</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown('###')
            col1, col2, col3, col4, col5 = st.columns([5, 1, 5, 1, 5])
            with col1:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Recency</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Recency_Mean')
            with col3:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Frequency</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Frequency_Mean')
            with col5:
                html_temp_title = """
                <div style="background-color:lightblue;padding:4px">
                <h5 style="color:white;text-align:center;">Monetary</h5>
                </div>
                """
                st.markdown(html_temp_title, unsafe_allow_html=True)
                plot_pcts(rfm_stats, 'Monetary_Mean')
            
        except:
            st.error("Oops! Error performing operation! Please select another country.")
# %%
    if page == "Dashboard":
        # Title
        html_temp_title = """
            <div style="background-color:#154360;padding:2px">
            <h2 style="color:white;text-align:center;">Dashboard</h2>
            </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        
        # Pick country 
        st.markdown('#### Choose Country:')
        option = st.selectbox('', country_list)
        country_retail = choose_country(option)
        
        # Top 10 customers without 'Guest Customer'
        top_customers = country_retail[country_retail["CustomerID"] != "Guest Customer"].groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
        html_temp_title = """
        <div style="background-color:#ABBAEA;padding:4px">
        <h3 style="color:white;text-align:center;">Top Customers without 'Guest Customer'</h3>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        fig = px.bar(top_customers, x ="CustomerID", y = "InvoiceNo", color= 'InvoiceNo')
        fig.update_layout(showlegend=False,
                                 height=250, width = 500,
                                 margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
        fig.update(layout_coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Top 10 performing products 
        col1, col2, col3= st.columns([10, 1, 10])
        with col1:
            html_temp_title = """
            <div style="background-color:#ABBAEA;padding:4px">
            <h3 style="color:white;text-align:center;">Top Products by Sold Quantity</h3>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            top_products_qty = group_sales_quantity(country_retail, 'Description').sort_values(ascending=False, by = "Quantity").reset_index(drop=True)
            top_products_qty.drop('Sales Revenue', axis=1, inplace=True)
            # Display merged dataframes   
            AgGrid(top_products_qty, theme='blue', height = 200, width = 150)
        with col3:
            html_temp_title = """
            <div style="background-color:#ABBAEA;padding:4px">
            <h3 style="color:white;text-align:center;">Top Products by Gross Sales Revenue</h3>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Top 10 Product Description by Sales Revenue
            top_products_revenue = group_sales_quantity(country_retail, 'Description').sort_values(ascending=False, by = "Sales Revenue").reset_index(drop=True)
            top_products_revenue.drop('Quantity', axis=1, inplace=True)
            # Display merged dataframes   
            AgGrid(top_products_revenue, theme='blue', height = 200, width = 150)
        
        
        html_temp_title = """
        <div style="background-color:#ABBAEA;padding:4px">
        <h3 style="color:white;text-align:center;">Country Level Statistics</h3>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        # Top 10 Countries and their total order counts (without U.K.)
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">1. Top 10 Countries and their total order counts (without U.K.)</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Aggregating on Countries without United Kingdom
            country_data_wo_uk = retail[retail['Country']!='United Kingdom'].groupby('Country')['InvoiceNo'].nunique().sort_values(ascending = False).reset_index()
            country_data_wo_uk_subset = country_data_wo_uk.head(10)
            
            # Plotting
            fig = px.bar(country_data_wo_uk_subset, x ="Country", y = "InvoiceNo", color= 'InvoiceNo')
            fig.update_layout(showlegend=False,
                                     height=400, width = 650,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown('- The above charts show that the UK by far has more invoices with the count surpassing 16000')
            st.markdown('- Germany in in second place, with approximately 30 time less invoices (414 count)')
            st.markdown('- The retail store management can start posing question of why this is the case, especially when this is a Online retail store')
            st.markdown('- They need a process to improve the quality and quantity of website traffic to a website')
            
        # Total Sales Revenue for Countries (except UK)
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">2. Total Sales Revenue for Countries (without U.K.)</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            sales_wo_uk = retail[retail['Country'] != 'United Kingdom'].groupby('Country').sum().sort_values(by = 'Sales Revenue', ascending = False).reset_index()
            
            # Plotting
            fig = px.bar(sales_wo_uk, x ="Sales Revenue", y = "Country", color= 'Sales Revenue')
            fig.update_layout(showlegend=False,
                                     height=400, width = 650,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown('- Sales Revenue for Netherlands and Ireland(EIRE) are quite close ($270K)')
            
        # Total Quantity of Items Sold for Countries (without U.K.)
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">3. Total Quantity of Items Sold for Countries (without U.K.)</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Plotting
            fig = px.bar(sales_wo_uk, x ="Quantity", y = "Country", color= 'Quantity')
            fig.update_layout(showlegend=False,
                                     height=400, width = 650,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig)
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown('- Total Quantity of items sold for Netherlands and Ireland(EIRE) are at the top with values 190K and 140K')
            
        # Monthly Stats
        html_temp_title = """
        <div style="background-color:#ABBAEA;padding:4px">
        <h3 style="color:white;text-align:center;">Monthly Statistics</h3>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">1. Monthly Stats by Sales Revenue</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Get monthly stats dataframe
            monthly_stats = group_sales_quantity(retail, 'Month')
            
            # Plotting
            fig = px.bar(monthly_stats, x ="Sales Revenue", y = "Month", color= 'Sales Revenue')
            fig.update_layout(showlegend=False,
                                     height=400, width = 650,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig.update(layout_coloraxis_showscale=False)
            st.plotly_chart(fig)
            
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown('- As expected, the Monthly statistics for November is the highest due to Thanksgiving Holiday/Black Friday/Cyber Monday sale')
            
        # Percentage Pie Chart for Quantity per Month
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">2. Percentage Pie Chart for Quantity Sold per Month</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            # Plotting
            fig_target = go.Figure(data=[go.Pie(labels=monthly_stats["Month"],
                                        values=monthly_stats["Quantity"],
                                        hole=.3)])
            fig_target.update_layout(showlegend=False,
                                     height=500,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig_target.update_traces(textposition='inside', textinfo='label+percent')
            fig_target.update_traces(marker=dict(colors=['lightcyan', 'cyan', 'royalblue', 'darkblue']))
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown("- Highest sales (by revenue and quantity) has been noted during the Fall season, which are the month of September, October, November")
            st.markdown('- Lowest percentage of sales (by revenue and quantity) has been noted during the Winter season (January, February, March, April) where people are unable to leave the house due to harsh weather')
        
        # Daily Statistics
        html_temp_title = """
        <div style="background-color:#ABBAEA;padding:4px">
        <h3 style="color:white;text-align:center;">Daily Statistics</h3>
        </div>
        """
        st.markdown(html_temp_title, unsafe_allow_html=True)
        st.markdown("###")
        
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">1. Percentage Pie Chart for Gross Sales Revenue per Day of Week</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            
            # Get daily stats dataframe
            daily_stats = group_sales_quantity(retail, 'Day of Week')
            # Plotting
            fig_target = go.Figure(data=[go.Pie(labels=daily_stats["Day of Week"],
                                        values=daily_stats["Sales Revenue"],
                                        hole=.3)])
            fig_target.update_layout(showlegend=False,
                                     height=500,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig_target.update_traces(textposition='inside', textinfo='label+percent')
            fig_target.update_traces(marker=dict(colors=['lightcyan', 'cyan', 'royalblue', 'darkblue']))
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown("- People buy more on Thursdays to prepare for the weekend, followed by Tuesday to prepare for the week ahead")
            st.markdown('- Least sale is noted for Sunday')
        
        col1, col2, col3= st.columns([10, 1, 5])
        with col1:
            html_temp_title = """
            <div style="background-color:SteelBlue;padding:4px">
            <h4 style="color:white;text-align:center;">2. Percentage Pie Chart for Quantity Sold per Time of Day</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            timeofday_stats = group_sales_quantity(retail, 'Time of Day')

            # Plotting
            fig_target = go.Figure(data=[go.Pie(labels=timeofday_stats["Time of Day"],
                                        values=timeofday_stats["Quantity"],
                                        hole=.3)])
            fig_target.update_layout(showlegend=False,
                                     height=500,
                                     margin={'l': 10, 'r': 10, 't': 0, 'b': 0})
            fig_target.update_traces(textposition='inside', textinfo='label+percent')
            fig_target.update_traces(marker=dict(colors=['lightcyan', 'cyan', 'royalblue', 'darkblue']))
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col3:
            html_temp_title = """
            <div style="background-color:lightblue;padding:4px">
            <h4 style="color:white;text-align:center;">Observations</h4>
            </div>
            """
            st.markdown(html_temp_title, unsafe_allow_html=True)
            st.markdown("###")
            st.markdown("- People tend to buy more during the morning period than at night")
            
        
# %%
if __name__ == "__main__":
    main()