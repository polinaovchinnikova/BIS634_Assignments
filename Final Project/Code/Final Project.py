#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import datetime as dt
import warnings


# Data upload 

# In[78]:


df = pd.read_csv('/Users/polina/Desktop/Life Expectancy Data.csv')
df


# Data Info

# In[79]:


df.info()


# In[80]:


df.shape


# In[81]:


df.describe()


# Data Cleaning 

# In[82]:


# Looking for null value in the data
df.isnull().sum()


# In[83]:


# Replacing the Null Values with mean values of the data
from sklearn.impute import SimpleImputer
#reference: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value=None)
df['Life expectancy ']=imputer.fit_transform(df[['Life expectancy ']])
df['Adult Mortality']=imputer.fit_transform(df[['Adult Mortality']])
df['Alcohol']=imputer.fit_transform(df[['Alcohol']])
df['Hepatitis B']=imputer.fit_transform(df[['Hepatitis B']])
df[' BMI ']=imputer.fit_transform(df[[' BMI ']])
df['Polio']=imputer.fit_transform(df[['Polio']])
df['Total expenditure']=imputer.fit_transform(df[['Total expenditure']])
df['Diphtheria ']=imputer.fit_transform(df[['Diphtheria ']])
df['GDP']=imputer.fit_transform(df[['GDP']])
df['Population']=imputer.fit_transform(df[['Population']])
df[' thinness  1-19 years']=imputer.fit_transform(df[[' thinness  1-19 years']])
df[' thinness 5-9 years']=imputer.fit_transform(df[[' thinness 5-9 years']])
df['Income composition of resources']=imputer.fit_transform(df[['Income composition of resources']])
df['Schooling']=imputer.fit_transform(df[['Schooling']])


# In[84]:


# Looking for null value in the data after fitting
df.isnull().sum()


# In[85]:


# Changing/Renaming the columns for easy access.
df = df.rename(columns={'Country': 'country', 'Year': 'year', 'Status': 'status', 'Life expectancy ': 'life_expectancy', 'Adult Mortality': 'adult_mortality',
       'infant deaths':'infant_death', 'Alcohol':'alcohol', 'percentage expenditure': 'percentage_expenditure', 'Hepatitis B':'Hepatitis_b',
       'Measles ':'measles', ' BMI ':'bmi', 'under-five deaths ':'under_five_deaths', 'Polio':'polio', 'Total expenditure': 'total_expenditure','Diphtheria ':'diphtheria', ' HIV/AIDS':'hiv_Aids', 'GDP':'gdp', 'Population':'population',
       ' thinness  1-19 years':'thinness_1_to_19', ' thinness 5-9 years':'thinness_5_to_9',
       'Income composition of resources':'income_composition_of_resources', 'Schooling': 'schooling'})


# In[86]:


# Looking for columns after rename
df.columns


# In[87]:


#remove empty space 
orig_cols = list(df.columns)
new_cols = []
for col in orig_cols:
    new_cols.append(col.strip().replace('  ', ' ').replace(' ', '_').lower())
df.columns = new_cols


# In[ ]:


# save the clean data into a CSV file 
df.to_csv('clean_df.csv', index=False)


# In[ ]:


clean_df = pd.read_csv("clean_df.csv")
clean_df


# Data Describe 

# In[12]:


df.describe(include='object')


# In[13]:


#Top 10 Countries
print("Top 10 Countries with Most Life Expectancy")
print("="*50)
print(df.groupby("country").life_expectancy.mean().sort_values(ascending =False).head(10))
print("="*50)
print("Top 10 Countries with Least Life Expectancy")
print("="*50)
print(df.groupby("country").life_expectancy.mean().sort_values(ascending =True).head(10))


# In[14]:


# Countries with Highest Life Expectancy
country_vs_life = df.groupby('country', as_index=False)['life_expectancy'].mean()
country_vs_life.sort_values(by = 'life_expectancy', ascending=False).head(10)


# In[15]:


# Countries with Lowest Life Expectancy
country_vs_life.sort_values(by = 'life_expectancy', ascending = True).head(10)


# In[16]:


df.corr()['life_expectancy'].abs().sort_values(ascending=False)[1:]


# In[17]:


df["status"].value_counts()


# In[19]:


df.groupby(['status'])[["life_expectancy"]].mean()


# In[20]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')


# Outliers 

# In[21]:


cont_vars = list(df.columns)[3:]
def outliers_visual(data):
    plt.figure(figsize=(15, 40))
    i = 0
    for col in cont_vars:
        i += 1
        plt.subplot(9, 4, i)
        plt.boxplot(data[col])
        plt.title('{} boxplot'.format(col))
        i += 1
        plt.subplot(9, 4, i)
        plt.hist(data[col])
        plt.title('{} histogram'.format(col))
    plt.show()
outliers_visual(df)


# Visually, it is plain to see that there are a number of outliers for all of these variables - including the target variable, life expectancy.

# Uisng Tukey's method below - outliers being considered anything outside of 1.5 times the IQR

# In[22]:


def outlier_count(col, data=df):
    print(15*'-' + col + 15*'-')
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))


# In[23]:


for col in cont_vars:
    outlier_count(col)


# Data Viz

# In[24]:


# Visual Distributions
plt.figure(figsize=(15, 20))
for i, col in enumerate(cont_vars, 1):
    plt.subplot(5, 4, i)
    plt.hist(df[col])
    plt.title(col)


# In[25]:


#. Values of rows for status 
plt.figure(figsize=(10, 5))
plt.subplot(121)
df.status.value_counts().plot(kind='bar')
plt.title('Count of Rows by Country Status')
plt.xlabel('Country Status')
plt.ylabel('Count of Rows')
plt.xticks(rotation=0)

plt.subplot(122)
df.status.value_counts().plot(kind='pie', autopct='%.2f')
plt.ylabel('')
plt.title('Country Status Pie Chart')

plt.show()


# The above displays that the majority of our data comes from countries listed as 'Developing' - 82.57% to be exact. It is likely that any model used will more accurately depict results for 'Developing' countries over 'Developed' countries as the majority of the data lies within countries that are 'Developing' rather than 'Developed'.

# In[26]:


#Adult mortality
sns.barplot(df["status"],df['adult_mortality'])


# In[27]:


#Distribution of Life Expectancy according to the age
fig = px.histogram(df, x = 'life_expectancy', template = 'plotly_dark')
fig.show()


# In[28]:


#Comparing the life expectancy of Developing and Developed Countries
fig = px.violin(df, x= 'status', y= 'life_expectancy',
                color = 'status',template = 'plotly_dark', box = True,title='Life Expectancy on the Basis of Country Status')
fig.show()


# In[29]:


#Country Wise Life Expectancy over the years
fig = px.line((df.sort_values(by = 'year')), x = 'year', y = 'life_expectancy',
    animation_frame= 'country',template = 'plotly_dark', animation_group='year',color='country',
    markers=True,title='Country Wise Life Expectancy over the years')
fig.show()


# In[30]:


country_df = px.data.gapminder()
country_df.tail()


# In[31]:


#Life Expectancy over the World Map
map_fig = px.scatter_geo(country_df,locations = 'iso_alpha', projection = 'orthographic', 
                         opacity = 0.8, color = 'country', hover_name = 'country', 
                         hover_data = ['lifeExp', 'year'],template = 'plotly_dark',title = 'Life Expectancy over the World Map')
map_fig.show()


# In[32]:


# Life Expectancy versus the adult Mortality in different countries every year¶
px.scatter(df, x = 'life_expectancy', y = 'adult_mortality',
           color = 'country', template = 'plotly_dark', size = 'life_expectancy',opacity = 0.6, 
           title = '<b>Life Expectancy Vs Adult Mortality in Countries')


# In[33]:


#Life Expectancy Versus GDP of Counntries all over the World¶
px.scatter(df.sort_values(by='year'), x = 'life_expectancy', y = 'gdp', color = 'country',
          size = 'year',animation_frame = 'year', animation_group = 'country',template = 'plotly_dark',
           title = '<b>Life Expectancy Vs GDP in Countries')


# In[34]:


# country and the sum population 
fig=px.histogram(df,x='country',y = 'population', template='seaborn')
fig.show()


# In[35]:


#correlation plot
plt.figure(figsize=(20,20)) 
sns.heatmap(df.corr(),annot = True,cmap = "YlGnBu")
plt.show()


# In[36]:


# plot by. status
y = df["life_expectancy"]
df_clean2 = df.drop("life_expectancy",axis=1)

for feature in df_clean2.select_dtypes(exclude="O").columns:
    plt.figure(figsize=(12,5))
    sns.scatterplot(x=df_clean2[feature],y=y,hue=df_clean2["status"])


# In[94]:


# Let's made an interactive plot with the help of plotly
numerical_features = df.copy()


# In[95]:


countries = numerical_features["country"].unique()   
def make_scatter_plot_country_wise(feature):
    first_title = countries[0]
    traces = []
    buttons = []
    frame = []

    for index,country in enumerate(countries):
        visible = [False] * len(countries)
        visible[index] = True
        name = country
        # Get the dataFrame curresponding to that country
        country_data = numerical_features.query('country == @country')
        
        traces.append(
            px.scatter(country_data, x=feature, y="life_expectancy", color="year").update_traces(visible=True if index==0 else False).data[0]
        )
        
        buttons.append(dict(label=name,
                            method="update",
                            args=[{"visible":visible}, {"title":f"{name}"}]))

    fig = go.Figure(data=traces)
    fig.update_layout( xaxis_title=f"<b>{feature}</b>",
                        yaxis_title="<b>life_expectancy</b>",
                        legend_title="Year",
                        updatemenus=[go.layout.Updatemenu(
                            active=0,
                            buttons=buttons
                            )
                        ])
    fig.update_traces(marker_size=40)

    fig.update_layout(title=f"<b>{feature}</b>")
    fig.show()


# In[96]:


numerical_features.columns.values


# In[97]:


import plotly.graph_objs as go


# In[98]:


for i in numerical_features.columns.values:
    if i == "Country" or i == "Status" or i == "Life expectancy ":
        continue
    make_scatter_plot_country_wise(i)


# In[104]:


# worldmap plotly 
from plotly.offline import init_notebook_mode, iplot
count = [ dict(
        # set the map type is choropleth  
        type = 'choropleth',
        locations = df['country'],
        locationmode='country names',
        z = df['life_expectancy'],
        text = df['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        # set the plotly gragh color
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        # add a color bar 
        colorbar = dict(
            autotick =False,
            title = 'Life Expectancy Country-based'),
      ) ]
# create layout for gragh
layout = dict(
    title = 'Life Expectancy across the Global',
    # 
    geo = dict(
        showframe = True,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)
# prepare the fig parameter
fig = dict( data=count, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )


# Dashboard with pandas-profiling

# In[ ]:


pip install pandas-profiling


# In[37]:


import pandas_profiling
report = df.profile_report(
    sort=None, html={"style": {"full_width": True}}, progress_bar=False
)
report


# Linear Regression

# In[38]:


df_status = df.replace({"Developed":1,"Developing":0})


# In[40]:


#check status 
df_status['status']
df_status


# In[41]:


from sklearn.linear_model import LinearRegression

Y = df_status["life_expectancy"]
X = df_status.drop(["life_expectancy","country"], axis=1)
lrm = LinearRegression()
lrm.fit(X,Y)


# In[42]:


import statsmodels.api as sm

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
results.summary()


# KNN

# In[43]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

X_test = sm.add_constant(X_test)
y_preds = results.predict(X_test)


# In[44]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train,y_train)


# In[45]:


from sklearn.model_selection import GridSearchCV

knn_parameters = {"n_neighbors":range(1,10),
                  "weights":["uniform","distance"],
                }

grid_knn = GridSearchCV(estimator=knn,
                       param_grid = knn_parameters,
                       cv = 10)

grid_knn.fit(X, Y)


# In[46]:


print("Best R-squared score::{}".format(grid_knn.best_score_))
print("Best parameters::\n{}".format(grid_knn.best_params_))


# PCA

# PCA is used to preprocess the data to perform K-Means Clustering

# In[47]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[48]:


df.iloc[:,3:].apply(pd.to_numeric,errors='coerce')


# In[49]:


drop_list = ["life_expectancy","country"]
df.drop(drop_list, axis=1 ,inplace=True)
df1 = df.copy()
group = []
for i in df1.columns:
    if (df1[i].dtypes == "object"):
        group.append(i)

#print(group)

lbl_encode = LabelEncoder()
for i in group :
    df1[i]=lbl_encode.fit_transform(df1[[i]])


# In[50]:


scaler = StandardScaler()
scaler.fit(df1)
scaled_df = pd.DataFrame(scaler.transform(df1), columns=df1.columns)


# In[51]:


scaled_df.head()


# In[125]:


# The number of dimensions as 3
pca = PCA(n_components=3)
pca.fit(scaled_df)
pca_data = pd.DataFrame(pca.transform(scaled_df), columns=["c1", "c2", "c3"])


# In[126]:


import plotly.graph_objs as go
x = pca_data["c1"]
y = pca_data["c2"]
z = pca_data["c3"]

fig = go.Figure(data=[go.Scatter3d(
    x=x,y=y,z=z,mode='markers',
    marker=dict(size=6,color=x,opacity=0.8))])

fig.update_layout( title={'text': "3D Plot of Size-Reduced Data",'y':0.9,
        'x':0.5,'xanchor': 'center','yanchor': 'top'},
                  margin=dict(l=200, r=220, b=0, t=0))
fig.show()


# K Means Clustering using Elbow Method

# In[55]:


from sklearn.cluster import KMeans


# - Each cluster is formed by calculation and comparing the distance of data point withon a cluster to its center
# - Within-Cluster-Sum-of-Squares(WCSS) - to fund the right number of clusters. WCSS is the sum of squares of the distances of each data point in all clusters to their respective centers, and the goal is to minimize the sum. Assume there are n observations in a dataset and we specify n number of clusters, which means k = n; so WCSS turns to 0 since data points themselves become centers and the distance will be 0, in turn this will perform a perfect cluster; but this is almost impossible as we have many clusters as the observations. Thus, we use Elbow point graph to find the optimum value for K by fitting the model in a range of values of K. We randomly initialize the K-Means algorithm for a range of K values and plot it against the WCSS for each K value.

# In[56]:


#https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
wcss = []
k = range(1,18)
for i in k:
    model = KMeans(n_clusters=i)
    model.fit(pca_data)
    wcss.append(model.inertia_)

plt.plot(k, wcss, '-o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters(K)')
plt.ylabel('WCSS')
plt.xticks(k)
plt.show()


# - optimum value for K = 3
# - increase in the number of clusters, the WCSS value decreases
# - select the value for K, the "elbow", on the basis of the rate of decrease, to indicate the model fits best at that point. In the graph, from cluster 1 to 2 to 3 in the above graph there is a huge drop in WCSS. After 3 the drop is minimal, thus we chose 3 to be the optimal value for K. Based on the Elbow Method, we can find the optimal number of clusters is 3. https://en.wikipedia.org/wiki/Elbow_method_(clustering)

# In[110]:


k_means = KMeans(n_clusters = 3, random_state = 100)
y_pred = k_means.fit_predict(pca_data)
pca_data['Cluster'] = y_pred
df['Cluster'] = y_pred


# In[118]:


sns.countplot(x=pca_data['Cluster'], palette = 'viridis')


# In[122]:


#GDP
sns.kdeplot(data=df, x='gdp', hue='Cluster', palette = 'viridis')


# In[123]:


#Income composition of resources
sns.kdeplot(data=df, x='income_composition_of_resources', hue='Cluster', palette = 'viridis')


# In[116]:


#Status
profile = ['status']

for i in profile:
    plt.figure()
    sns.countplot(x='Cluster', data=df, hue=df[i],palette = 'viridis')
    plt.show()

