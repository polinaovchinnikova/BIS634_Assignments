# Assignment-1-
```
def temp_tester(temp):
    
 def inner_function(x): 
    if x<= temp + 1 and x >= temp - 1:
        return True
    else: 
        return False
    
 return inner_function
```
```
human_tester = temp_tester(37)
chicken_tester = temp_tester(41.1)
```
```
print(chicken_tester(42)) # True 
print(human_tester(42))  #False
print(chicken_tester(43)) #False
print(human_tester(35)) #False
print(human_tester(98.6)) #False 
```

##2
```
import pandas as pd
import sqlite3
with sqlite3.connect("/Users/polina/Desktop/hw1-population.db") as db:
    data = pd.read_sql_query("SELECT * FROM population", db)
    data.head
```
>There are 4 columns and 152361 rows. Thus there 152,361 people in the pupulation with columns names 
1. name
2. age
3. weight
4. eyecolor

#Age
```
data.describe()[["age"]]
```
- Mean = 39.510528
- Standard Deviation = 24.152760
- Minimum = 0.000748
- Maximum = 99.991547


>Each bin is plotted as a bar whose height corresponds to how many data points are in that bin. Thus, we could use Sturges' Rule to calculate the optimal number of bins to use in a histigram: 

>Optima Bins = [log<sub>2</sub>(n) + 1]
```
log2 = math.log2(152361)+1  
log2
```
> Thus, showing that the optima number of bins would be 18. 
![](age_histogram.png)

>Furthermore, from the graph, it could be noted that the distribution somewhat not normal and skewed to the right. Plus, it could be noted that there are extrim drops pf age counts when it comes to 60 and then an extreme jump dowm to 80. 

#Weight 
```
data.describe()[["weight"]]
```
- Mean = 60.884134
- Standard Deviation = 18.411824
- Minimum = 3.382084
- Maximum = 100.435793

![](weight_histogram.png)

> The graph indicates a skewed to the left side histogram, as most of the values are on the right side of the graph. Thus, indicating that there are fewer people that weight then 60. Furthermore, an extreme drop coudl be noted around 70-pound weight, and some of the outliers are located near a 100 and a 0. 
```
import plotly.express as px
df = px.data.iris() 
fig = px.scatter(data, x="age", y="weight")
fig.show()
```
> The scatter plot has a positive correlation to zero correlation as the graph indicates to have both a vertical and horizon line. Thus, age and weight have somewhat of a weak relationship and are closer to not having a relationship with each other at all. 

```
# Filter to detect outlier
data[(data['age'] == 41.3) | (data['weight'] == 21.7)]
```
> Outlier: 
- count: 537
- name: Anthony Freeman
- age: 41.3
- weight: 21.7

> The outlier was found using the plotly histogram that allows you to hover on the scatter plot dots and see their exact x and y values, thus by looking at the outlier on the closer to the bottom of the graph, his x and y values were given. Then, to print his name and excat patient id, a print function was used containing the  exact values for the age and weight that were noted by hovering over the outlier. 

##3
```
import pandas as pd
url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
df = pd.read_csv(url,index_col=0,parse_dates=[0])
df.head
```
This data was taken [GitHub Pages](github.com/nytimes/covid-19-data).


##4


## License
https://github.com/nytimes/covid-19-data
