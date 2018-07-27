
# Interactive Data Exploration, Analysis, and Reporting

- Author: Team Data Science Process from Microsoft 
- Date: 2017/03
- Supported Data Sources: CSV files on the machine where the Jupyter notebook runs or data stored in SQL server
- Output: IDEAR_Report.ipynb


This is the **Interactive Data Exploration, Analysis and Reporting (IDEAR)** in _**Python**_ running on Jupyter Notebook. The data can be stored in CSV file on the machine where the Jupyter notebook runs or from a query running against a SQL server. A yaml file has to be pre-configured before running this tool to provide information about the data. 

## Step 1: Configure and Set up IDEAR

Before start utilitizing the functionalities provided by IDEAR, you need to first [configure and set up](#setup) the utilities by providing the yaml file and load necessary Python modules and libraries. 

## Step 2: Start using IDEAR
This tool provides various functionalities to help users explore the data and get insights through interactive visualization and statistical testing. 

- [Read and Summarize the data](#read and summarize)

- [Extract Descriptive Statistics of Data](#descriptive statistics)

- [Explore Individual Variables](#individual variables)

- [Explore Interactions between Variables](#multiple variables)

    - [Rank variables](#rank variables)
    
    - [Interaction between two categorical variables](#two categorical)
    
    - [Interaction between two numerical variables](#two numerical)

    - [Interaction between numerical and categorical variables](#numerical and categorical)

    - [Interaction between two numerical variables and a categorical variable](#two numerical and categorical)

- [Visualize High Dimensional Data via Projecting to Lower Dimension Principal Component Spaces](#pca)

- [Generate Data Report](#report)

After you are done with exploring the data interactively, you can choose to [show/hide the source code](#show hide codes) to make your notebook look neater. 

**Note**:

- Change the working directory and yaml file before running IDEAR in Jupyter Notebook.

- Run the cells and click *Export* button to export the code that generates the visualization/analysis result to temporary Jupyter notebooks.

- Run the last cell and click [***Generate Final Report***](#report) to create *IDEAR_Report.ipynb* in the working directory. _If you do not export codes in some sections, you may see some warnings complaining that some temporary Jupyter Notebook files are missing_. 

- Upload *IDEAR_Report.ipynb* to Jupyter Notebook server, and run it to generate report.

## <a name="setup"></a>Global Configuration and Setting Up


```python
# Set the working directory as the directory where ReportMagics.py stays
# Use \\ in your path
import os
workingDir = 'C:\\Users\\936344\\Desktop\\TDSP Framework'
os.chdir(workingDir)

from ReportMagics import *

merged_report ='IDEAR_Report.ipynb'
%reset_all
```


```python
%%add_conf_code_to_report
import os
workingDir = 'C:\\Users\\936344\\Desktop\\TDSP Framework'
os.chdir(workingDir)

conf_file = 'training_data.yaml'
Sample_Size = 10000

export_dir = 'C:\\Users\\936344\\Desktop\\TDSP Framework'
```

### Import necessary packages and set up environment parameters


```python
%%add_conf_code_to_report

import pandas as pd
import numpy as np
import os
#os.chdir(workingDir)
import collections
import matplotlib
import io
import sys
import operator

import nbformat as nbf
from IPython.core.display import HTML
from IPython.display import display
from ipywidgets import interact, interactive,fixed
from IPython.display import Javascript, display,HTML
from ipywidgets import widgets, VBox
import ipywidgets
import IPython
from IPython.display import clear_output
import scipy.stats as stats
from statsmodels.graphics.mosaicplot import mosaic
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import errno
import seaborn as sns
from string import Template
from functools import partial
from collections import OrderedDict

# Utility Classes
from ConfUtility import * 
from ReportGeneration import *
from UniVarAnalytics import *
from MultiVarAnalytics import *

%matplotlib inline

#DEBUG=0

font={'family':'normal','weight':'normal','size':8}
matplotlib.rc('font',**font)
matplotlib.rcParams['figure.figsize'] = (12.0, 5.0)
matplotlib.rc('xtick', labelsize=9) 
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('axes', labelsize=10)
matplotlib.rc('axes', titlesize=10)
sns.set_style('whitegrid')
```

### Define some functions for generating reports


```python
%%add_conf_code_to_report

if not os.path.exists(export_dir):
    os.makedirs(export_dir)
    
def gen_report(conf_md,conf_code, md, code, filename):
    ReportGeneration.write_report(conf_md, conf_code, md, code, report_name=filename)

def translate_code_commands(cell, exported_cols, composite=False):    
    new_code_store = []
    exported_cols = [each for each in exported_cols if each!='']   
    for each in exported_cols:       
        w,x,y = each.split(',')
        with open('log.txt','w') as fout:
            fout.write('Processing call for the column {}'.format(each))
        temp=cell[0]

        new_line = temp.replace('interactive','apply').replace(
            "df=fixed(df)","df").replace("filename=fixed(filename)","'"+ReportMagic.var_files+"'").replace(
            "col1=w1","'"+w+"'").replace("col2=w2","'"+x+"'").replace("col3=w3","'"+y+"'").replace(
            "col3=fixed(w3)","'"+y+"'").replace(
            "Export=w_export","False").replace("conf_dict=fixed(conf_dict)","conf_dict")       
        new_line = new_line.replace("df,","[df,")
        new_line = new_line[:len(new_line)-1]+"])"
        new_line = new_line.replace("apply(","").replace(", [", "(*[")
        new_code_store.append(new_line)        
    return new_code_store

def add_to_report(section='', task=''):
    print ('Section {}, Task {} added for report generation'.format(section ,task))

def trigger_report(widgets,export_cols_file, output_report, no_widgets=1, md_text=''):
    exported_cols = []
    with open(export_cols_file,'r') as fin:
        for each in fin:
            each = each.strip()
            if each and not each.isspace():
                exported_cols.append(each)
                
    exported_cols = list(set(exported_cols))
    conf_md, conf_code, md, code=%show_report 
    md = md_text
    cell = code
    new_code_store = translate_code_commands(cell,exported_cols)
    gen_report(conf_md,conf_code, md, new_code_store, filename=export_dir+output_report)
    
def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

def handle_change(value):
    w_export.value=False

def getWidgetValue(w):
    w_value = ''
    try:
        w_value = w.value
    except:
        pass    
    return w_value

def handle_export(widget, w1, w2, w3, export_filename='temp.ipynb',md_text=''):    
    print ('Export is successful!')
    w1_value, w2_value, w3_value = \
        getWidgetValue(w1),getWidgetValue(w2),getWidgetValue(w3)
    st = ','.join(str(each) for each in [w1_value, w2_value, w3_value])
    with open(filename,'a') as fout:
        fout.write(st+'\n')
    trigger_report(w1_value, filename, export_filename, False, md_text=md_text)  
    
            
```

## <a name="read and summarize"></a> Read and Summarize the Data

### Read data and infer column types


```python
%%add_conf_code_to_report

conf_dict = ConfUtility.parse_yaml(conf_file)

# Read in data from local file or SQL server
if 'DataSource' not in conf_dict:
    df=pd.read_csv(conf_dict['DataFilePath'][0], skipinitialspace=True)
else:
    import pyodbc
    cnxn = pyodbc.connect('driver=ODBC Driver 11 for SQL Server;server={};database={};Uid={};Pwd={}'.format(
            conf_dict['Server'], conf_dict['Database'],conf_dict['Username'],conf_dict['Password']))
    df = pd.read_sql(conf_dict['Query'],cnxn)

# Making sure that we are not reading any extra column
df = df[[each for each in df.columns if 'Unnamed' not in each]]

# Sampling Data if data size is larger than 10k
df0 = df # df0 is the unsampled data. Will be used in data exploration and analysis where sampling is not needed
         # However, keep in mind that your final report will always be based on the sampled data. 
if Sample_Size < df.shape[0]:
    df = df.sample(Sample_Size)

# change float data types
if 'FloatDataTypes' in conf_dict:   
    for col_name in conf_dict['FloatDataTypes']:
        df[col_name] = df[col_name].astype(float)      
        
# Getting the list of categorical columns if it was not there in the yaml file
if 'CategoricalColumns' not in conf_dict:
    conf_dict['CategoricalColumns'] = list(set(list(df.select_dtypes(exclude=[np.number]).columns)))

# Getting the list of numerical columns if it was not there in the yaml file
if 'NumericalColumns' not in conf_dict:
    conf_dict['NumericalColumns'] = list(df.select_dtypes(include=[np.number]).columns)    

# Exclude columns that we do not need
if 'ColumnsToExclude' in conf_dict:
    conf_dict['CategoricalColumns'] = list(set(conf_dict['CategoricalColumns'])-set(conf_dict['ColumnsToExclude']))
    conf_dict['NumericalColumns'] = list(set(conf_dict['NumericalColumns'])-set(conf_dict['ColumnsToExclude']))
    
# Ordering the categorical variables according to the number of unique categories
filtered_cat_columns = []
temp_dict = {}
for cat_var in conf_dict['CategoricalColumns']:
    temp_dict[cat_var] = len(np.unique(df[cat_var].astype(str)))
sorted_x = sorted(temp_dict.items(), key=operator.itemgetter(0), reverse=True)
conf_dict['CategoricalColumns'] = [x for (x,y) in sorted_x]

ConfUtility.dict_to_htmllist(conf_dict,['Target','CategoricalColumns','NumericalColumns'])
```




<ul><li>Numerical Columns are Survived, 
Pclass, 
Age, 
SibSp, 
Parch, 
Fare</li><li>Target variable is Survived</li><li>Categorical Columns are Ticket, 
Sex, 
Name, 
Embarked, 
Cabin</li></ul>



### Print the first n (n=5 by default) rows of the data


```python
%%add_conf_code_to_report
def custom_head(df,NoOfRows):
    return HTML(df.head(NoOfRows).style.set_table_attributes("class='table'").render())
i = interact(custom_head,df=fixed(df0), NoOfRows=ipywidgets.IntSlider(min=0, max=30, step=1, \
                                                                     value=5, description='Number of Rows'))
```


<style  type="text/css" >
</style>  
<table id="T_f930b392_90e0_11e8_bf17_b4d5bdef686f" class='table'> 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >PassengerId</th> 
        <th class="col_heading level0 col1" >Survived</th> 
        <th class="col_heading level0 col2" >Pclass</th> 
        <th class="col_heading level0 col3" >Name</th> 
        <th class="col_heading level0 col4" >Sex</th> 
        <th class="col_heading level0 col5" >Age</th> 
        <th class="col_heading level0 col6" >SibSp</th> 
        <th class="col_heading level0 col7" >Parch</th> 
        <th class="col_heading level0 col8" >Ticket</th> 
        <th class="col_heading level0 col9" >Fare</th> 
        <th class="col_heading level0 col10" >Cabin</th> 
        <th class="col_heading level0 col11" >Embarked</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_f930b392_90e0_11e8_bf17_b4d5bdef686flevel0_row0" class="row_heading level0 row0" >0</th> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col0" class="data row0 col0" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col1" class="data row0 col1" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col2" class="data row0 col2" >3</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col3" class="data row0 col3" >Braund, Mr. Owen Harris</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col4" class="data row0 col4" >male</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col5" class="data row0 col5" >22</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col6" class="data row0 col6" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col7" class="data row0 col7" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col8" class="data row0 col8" >A/5 21171</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col9" class="data row0 col9" >7.25</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col10" class="data row0 col10" >nan</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow0_col11" class="data row0 col11" >S</td> 
    </tr>    <tr> 
        <th id="T_f930b392_90e0_11e8_bf17_b4d5bdef686flevel0_row1" class="row_heading level0 row1" >1</th> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col0" class="data row1 col0" >2</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col1" class="data row1 col1" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col2" class="data row1 col2" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col3" class="data row1 col3" >Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col4" class="data row1 col4" >female</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col5" class="data row1 col5" >38</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col6" class="data row1 col6" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col7" class="data row1 col7" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col8" class="data row1 col8" >PC 17599</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col9" class="data row1 col9" >71.2833</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col10" class="data row1 col10" >C85</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow1_col11" class="data row1 col11" >C</td> 
    </tr>    <tr> 
        <th id="T_f930b392_90e0_11e8_bf17_b4d5bdef686flevel0_row2" class="row_heading level0 row2" >2</th> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col0" class="data row2 col0" >3</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col1" class="data row2 col1" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col2" class="data row2 col2" >3</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col3" class="data row2 col3" >Heikkinen, Miss. Laina</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col4" class="data row2 col4" >female</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col5" class="data row2 col5" >26</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col6" class="data row2 col6" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col7" class="data row2 col7" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col8" class="data row2 col8" >STON/O2. 3101282</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col9" class="data row2 col9" >7.925</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col10" class="data row2 col10" >nan</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow2_col11" class="data row2 col11" >S</td> 
    </tr>    <tr> 
        <th id="T_f930b392_90e0_11e8_bf17_b4d5bdef686flevel0_row3" class="row_heading level0 row3" >3</th> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col0" class="data row3 col0" >4</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col1" class="data row3 col1" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col2" class="data row3 col2" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col3" class="data row3 col3" >Futrelle, Mrs. Jacques Heath (Lily May Peel)</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col4" class="data row3 col4" >female</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col5" class="data row3 col5" >35</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col6" class="data row3 col6" >1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col7" class="data row3 col7" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col8" class="data row3 col8" >113803</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col9" class="data row3 col9" >53.1</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col10" class="data row3 col10" >C123</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow3_col11" class="data row3 col11" >S</td> 
    </tr>    <tr> 
        <th id="T_f930b392_90e0_11e8_bf17_b4d5bdef686flevel0_row4" class="row_heading level0 row4" >4</th> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col0" class="data row4 col0" >5</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col1" class="data row4 col1" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col2" class="data row4 col2" >3</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col3" class="data row4 col3" >Allen, Mr. William Henry</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col4" class="data row4 col4" >male</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col5" class="data row4 col5" >35</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col6" class="data row4 col6" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col7" class="data row4 col7" >0</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col8" class="data row4 col8" >373450</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col9" class="data row4 col9" >8.05</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col10" class="data row4 col10" >nan</td> 
        <td id="T_f930b392_90e0_11e8_bf17_b4d5bdef686frow4_col11" class="data row4 col11" >S</td> 
    </tr></tbody> 
</table> 


### Print the dimensions of the data (rows, columns)


```python
%%add_conf_code_to_report
print ('The data has {} Rows and {} columns'.format(df0.shape[0],df0.shape[1]))
```

    The data has 891 Rows and 12 columns
    

### Print the column names of the data


```python
%%add_conf_code_to_report
col_names = ','.join(each for each in list(df.columns))
print("The column names are:" + col_names)
```

    The column names are:PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    

### Print the column types


```python
%%add_conf_code_to_report
print("The types of columns are:")
df.dtypes
```

    The types of columns are:
    




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



## <a name="individual variable"></a>Extract Descriptive Statistics of Each Column


```python
%%add_conf_code_to_report
def num_missing(x):
    return len(x.index)-x.count()

def num_unique(x):
    return len(np.unique(x.astype(str)))

temp_df = df0.describe().T
missing_df = pd.DataFrame(df0.apply(num_missing, axis=0)) 
missing_df.columns = ['missing']
unq_df = pd.DataFrame(df0.apply(num_unique, axis=0))
unq_df.columns = ['unique']
types_df = pd.DataFrame(df0.dtypes)
types_df.columns = ['DataType']
```

### Print the descriptive statistics of numerical columns


```python
%%add_conf_code_to_report
summary_df = temp_df.join(missing_df).join(unq_df).join(types_df)
summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>missing</th>
      <th>unique</th>
      <th>DataType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PassengerId</th>
      <td>891.0</td>
      <td>446.000000</td>
      <td>257.353842</td>
      <td>1.00</td>
      <td>223.5000</td>
      <td>446.0000</td>
      <td>668.5</td>
      <td>891.0000</td>
      <td>0</td>
      <td>891</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>891.0</td>
      <td>0.383838</td>
      <td>0.486592</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>1.0000</td>
      <td>0</td>
      <td>2</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>891.0</td>
      <td>2.308642</td>
      <td>0.836071</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>3.0000</td>
      <td>3.0</td>
      <td>3.0000</td>
      <td>0</td>
      <td>3</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>714.0</td>
      <td>29.699118</td>
      <td>14.526497</td>
      <td>0.42</td>
      <td>20.1250</td>
      <td>28.0000</td>
      <td>38.0</td>
      <td>80.0000</td>
      <td>177</td>
      <td>89</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>891.0</td>
      <td>0.523008</td>
      <td>1.102743</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>1.0</td>
      <td>8.0000</td>
      <td>0</td>
      <td>7</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>891.0</td>
      <td>0.381594</td>
      <td>0.806057</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>6.0000</td>
      <td>0</td>
      <td>7</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>891.0</td>
      <td>32.204208</td>
      <td>49.693429</td>
      <td>0.00</td>
      <td>7.9104</td>
      <td>14.4542</td>
      <td>31.0</td>
      <td>512.3292</td>
      <td>0</td>
      <td>248</td>
      <td>float64</td>
    </tr>
  </tbody>
</table>
</div>



### Print the descriptive statistics of categorical columns


```python
%%add_conf_code_to_report
col_names = list(types_df.index) #Get all col names
num_cols = len(col_names)
index = range(num_cols)
cat_index = []
for i in index: #Find the indices of columns in Categorical columns
    if col_names[i] in conf_dict['CategoricalColumns']:
        cat_index.append(i)
summary_df_cat = missing_df.join(unq_df).join(types_df.iloc[cat_index], how='inner') #Only summarize categorical columns
summary_df_cat
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>missing</th>
      <th>unique</th>
      <th>DataType</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>891</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>2</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>681</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>148</td>
      <td>object</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>4</td>
      <td>object</td>
    </tr>
  </tbody>
</table>
</div>



## <a name="individual variables"></a>Explore Individual Variables

### Explore the target variable


```python
md_text = '## Target Variable'
filename = 'tmp/target_variables.csv'
export_filename = 'target_report2.ipynb'

if conf_dict['Target'] in conf_dict['CategoricalColumns']:
    w1_value,w2_value,w3_value = '','',''
    w1, w2, w3, w4 = None, None, None, None
    silentremove(filename)    
    w1 = widgets.Dropdown(
        options=[conf_dict['Target']],
        value=conf_dict['Target'],
        description='Target Variable:',
    )

    ReportMagic.var_files = filename
    w_export = widgets.Button(description='Export', value='Export')
    handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)      
    w1.observe(handle_change,'value')
    w_export.on_click(handle_export_partial)

    %reset_report
    %add_interaction_code_to_report i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), \
                                                    filename=fixed(filename), col1=w1, Export=w_export)
    hbox = widgets.HBox(i.children)
    display(hbox)
    hbox.on_displayed(TargetAnalytics.custom_barplot(df=df0, filename=filename, col1=w1.value, Export=w_export))
else:
    w1_value, w2_value, w3_value = '', '', ''
    w1, w2, w3, w4 = None, None, None, None
    silentremove(filename) 
    w1 = widgets.Dropdown(
            options=[conf_dict['Target']],
            value=conf_dict['Target'],
            description='Target Variable:',
        )
    w_export = widgets.Button(description='Export', value='Export')
    handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
    w1.observe(handle_change,'value')
    w_export.on_click(handle_export_partial)

    %reset_report
    %add_interaction_code_to_report ii = interactive(NumericAnalytics.custom_barplot, df=fixed(df), filename=fixed(filename),\
                                                    col1=w1, Export=w_export)
    hbox = widgets.HBox(ii.children)
    display(hbox)
    hbox.on_displayed(NumericAnalytics.custom_barplot(df=df, filename=filename, col1=w1.value, Export=w_export))
```


![png](output_27_0.png)


### Explore individual numeric variables and test for normality (on sampled data)


```python
md_text = '## Visualize Individual Numerical Variables (on Sampled Data)'
filename = ReportMagic.var_files='tmp/numeric_variables.csv'
export_filename = 'numeric_report2.ipynb'
w1_value, w2_value, w3_value = '', '', ''
w1, w2, w3, w4 = None, None, None, None
silentremove(filename) 
w1 = widgets.Dropdown(
        options=conf_dict['NumericalColumns'],
        value=conf_dict['NumericalColumns'][0],
        description='Numeric Variable:',
    )
w_export = widgets.Button(description='Export', value='Export')
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(NumericAnalytics.custom_barplot, df=fixed(df), filename=fixed(filename),\
                                                col1=w1, Export=w_export)
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(NumericAnalytics.custom_barplot(df=df, filename=filename, col1=w1.value, Export=w_export))
```


![png](output_29_0.png)


### Explore individual categorical variables (sorted by frequencies)


```python
w_export = None
md_text = '## Visualize Individual Categorical Variables'
filename = ReportMagic.var_files='tmp/categoric_variables.csv'
export_filename = 'categoric_report2.ipynb'

w1_value, w2_value, w3_value = '', '', ''
w1, w2, w3, w4 = None, None, None, None
silentremove(filename) 
w1 = widgets.Dropdown(
    options = conf_dict['CategoricalColumns'],
    value = conf_dict['CategoricalColumns'][0],
    description = 'Categorical Variable:',
)

w_export = widgets.Button(description='Export')
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe (handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(CategoricAnalytics.custom_barplot, df=fixed(df),\
                                                filename=fixed(filename), col1=w1, Export=w_export)

hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(CategoricAnalytics.custom_barplot(df=df0, filename=filename, col1=w1.value, \
                                         Export=w_export))
```


![png](output_31_0.png)


## <a name="multiple variables"></a>Explore Interactions Between Variables

### <a name="rank variables"></a>Rank variables based on linear relationships with reference variable (on sampled data)


```python
md_text = '## Rank variables based on linear relationships with reference variable (on sampled data)'
filename = ReportMagic.var_files='tmp/rank_associations.csv'
export_filename = 'rank_report2.ipynb'
silentremove(filename)
cols_list = [conf_dict['Target']] + conf_dict['NumericalColumns'] + conf_dict['CategoricalColumns'] #Make target the default reference variable
cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
w1 = widgets.Dropdown(    
    options=cols_list,
    value=cols_list[0],
    description='Ref Var:'
)
w2 = ipywidgets.Text(value="5", description='Top Num Vars:')
w3 = ipywidgets.Text(value="5", description='Top Cat Vars:')
w_export = widgets.Button(description='Export', value='Export')
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)
w1.observe (handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.rank_associations, df=fixed(df), \
                                                conf_dict=fixed(conf_dict), col1=w1, col2=w2, col3=w3, Export=w_export)
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.rank_associations(df=df, conf_dict=conf_dict, col1=w1.value, \
                                                         col2=w2.value, col3=w3.value, Export=w_export))
```

    C:\Users\936344\Documents\anaconda3\lib\site-packages\statsmodels\stats\anova.py:139: RuntimeWarning: divide by zero encountered in double_scalars
      (model.ssr / model.df_resid))
    C:\Users\936344\Documents\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in greater
      return (self.a < x) & (x < self.b)
    C:\Users\936344\Documents\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:879: RuntimeWarning: invalid value encountered in less
      return (self.a < x) & (x < self.b)
    C:\Users\936344\Documents\anaconda3\lib\site-packages\scipy\stats\_distn_infrastructure.py:1821: RuntimeWarning: invalid value encountered in less_equal
      cond2 = cond0 & (x <= self.a)
    


![png](output_34_1.png)


### <a name="two categorical"></a>Explore interactions between categorical variables


```python
md_text = '## Interaction between categorical variables'
filename = ReportMagic.var_files='tmp/cat_interactions.csv'
export_filename = 'cat_interactions_report2.ipynb'
silentremove(filename) 
w1, w2, w3, w4 = None, None, None, None

if conf_dict['Target'] in conf_dict['CategoricalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['CategoricalColumns']
    
w1 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[0],
    description='Categorical Var 1:'
)
w2 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[1],
    description='Categorical Var 2:'
)
w_export = widgets.Button(description='Export', value="Export")
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe(handle_change,'value')
w2.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.categorical_relations, df=fixed(df), \
                                         filename=fixed(filename), col1=w1, col2=w2, Export=w_export)
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.categorical_relations(df=df0, filename=filename, col1=w1.value, \
                                                             col2=w2.value, Export=w_export))
```


![png](output_36_0.png)


### <a name="two numerical"></a>Explore interactions between numerical variables (on sampled data)


```python
md_text = '## Interaction between numerical variables (on sampled data)'
filename = ReportMagic.var_files='tmp/numerical_interactions.csv'
export_filename = 'numerical_interactions_report2.ipynb'
silentremove(filename) 
w1, w2, w3, w4 = None, None, None, None

if conf_dict['Target'] in conf_dict['NumericalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['NumericalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['NumericalColumns']
w1 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[0],
    description='Numerical Var 1:'
)
w2 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[1],
    description='Numerical Var 2:'
)
w_export = widgets.Button(description='Export', value="Export")
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe(handle_change,'value')
w2.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.numerical_relations, df=fixed(df), \
                                         col1=w1, col2=w2, Export=w_export)
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.numerical_relations(df, col1=w1.value, col2=w2.value, Export=w_export))
```


![png](output_38_0.png)


### Explore correlation matrix between numerical variables


```python
md_text = '## Explore correlation matrix between numerical variables'
filename = ReportMagic.var_files='tmp/numerical_corr.csv'
export_filename = 'numerical_correlations_report2.ipynb'
silentremove(filename) 
w1, w2, w3, w4 = None, None, None, None
w1 = widgets.Dropdown(
    options=['pearson','kendall','spearman'],
    value='pearson',
    description='Correlation Method:'
)
w_export = widgets.Button(description='Export', value='Export')
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.numerical_correlation, df=fixed(df), conf_dict=fixed(conf_dict),\
                                         col1=w1, Export=w_export)

hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.numerical_correlation(df0, conf_dict=conf_dict, col1=w1.value, Export=w_export))
```


![png](output_40_0.png)


### <a name="numerical and categorical"></a>Explore interactions between numerical and categorical variables


```python
md_text = '## Explore interactions between numerical and categorical variables'
filename = ReportMagic.var_files = 'tmp/nc_int.csv'
export_filename = 'nc_report2.ipynb'
silentremove(filename) 
w1, w2, w3, w4 = None, None, None, None

if conf_dict['Target'] in conf_dict['NumericalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['NumericalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['NumericalColumns']
    
w1 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[0],
    description='Numerical Variable:'
)

if conf_dict['Target'] in conf_dict['CategoricalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['CategoricalColumns']
    
w2 = widgets.Dropdown(
    options=cols_list,
    value=cols_list[0],
    description='Categorical Variable:'
)
w_export = widgets.Button(description='Export', value=False, options=[True, False])
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)      
w1.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.nc_relation, df=fixed(df), \
                                                conf_dict=fixed(conf_dict), col1=w1, col2=w2, \
                                                col3=fixed(w3), Export=w_export)

hbox = widgets.HBox(i.children)
display( hbox )
hbox.on_displayed(InteractionAnalytics.nc_relation(df0, conf_dict, col1=w1.value, col2=w2.value, Export=w_export))
```


![png](output_42_0.png)


### <a name="two numerical and categorical"></a>Explore interactions between two numerical variables and a categorical variable (on sampled data)


```python
md_text = '## Explore interactions between two numerical variables and a categorical variable (on sampled data)'
filename = ReportMagic.var_files='tmp/nnc_int.csv'
export_filename = 'nnc_report2.ipynb'
silentremove(filename) 
w1, w2, w3, w4 = None, None, None, None

if conf_dict['Target'] in conf_dict['NumericalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['NumericalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['NumericalColumns']
    
w1 = widgets.Dropdown(
    options = cols_list,
    value = cols_list[0],
    description = 'Numerical Var 1:'
)
w2 = widgets.Dropdown(
    options = cols_list,
    value = cols_list[1],
    description = 'Numerical Var 2:'
)

if conf_dict['Target'] in conf_dict['CategoricalColumns']:
    cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
    cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
else:
    cols_list = conf_dict['CategoricalColumns']
    
w3 = widgets.Dropdown(
    options = cols_list,
    value = cols_list[0],
    description = 'Legend Cat Var:'
)
w_export = widgets.Button(description='Export', value=False, options=[True, False])
handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
w1.observe(handle_change,'value')
w_export.on_click(handle_export_partial)

%reset_report
%add_interaction_code_to_report i = interactive(InteractionAnalytics.nnc_relation, df=fixed(df),\
                                                conf_dict=fixed(conf_dict), col1=w1, col2=w2, col3=w3, Export=w_export)
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.nnc_relation(df, conf_dict, col1=w1.value,\
                                                    col2=w2.value, col3=w3.value, Export=w_export))
```


![png](output_44_0.png)


## <a name="pca"></a>Visualize numerical data by projecting to principal component spaces (on sampled data)

### Project data to 2-D principal component space (on sampled data)


```python
#  num_numeric = len(conf_dict['NumericalColumns'])
# if  num_numeric > 3:
#     md_text = '## Project Data to 2-D Principal Component Space'
#     filename = ReportMagic.var_files = 'tmp/numerical_pca.csv'
#     export_filename = 'numerical_pca_report2.ipynb'
#     silentremove(filename) 
    
#     w1, w2, w3, w4, w5 = None, None, None, None, None
#     if conf_dict['Target'] in conf_dict['CategoricalColumns']:
#         cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
#         cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
#     else:
#         cols_list = conf_dict['CategoricalColumns']
#     w1 = widgets.Dropdown(
#         options = cols_list,
#         value = cols_list[0],
#         description = 'Legend Variable:',
#         width = 10
#     )
#     w2 = widgets.Dropdown(
#         options = [str(x) for x in np.arange(1,num_numeric+1)],
#         value = '1',
#         width = 1,
#         description='PC at X-Axis:'
#     )
#     w3 = widgets.Dropdown(
#         options = [str(x) for x in np.arange(1,num_numeric+1)],
#         value = '2',
#         description = 'PC at Y-Axis:'
#     )
#     w_export = widgets.Button(description='Export', value=False, options=[True, False])
#     handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
#     w1.observe(handle_change,'value')
#     w_export.on_click(handle_export_partial)
    
#     %reset_report
#     %add_interaction_code_to_report i = interactive(InteractionAnalytics.numerical_pca, df=fixed(df),\
#                                                     conf_dict=fixed(conf_dict), col1=w1, col2=w2, col3=w3, Export=w_export)

    
#     hbox = widgets.HBox(i.children)
#     display(hbox)
#     hbox.on_displayed(InteractionAnalytics.numerical_pca(df, conf_dict=conf_dict, col1=w1.value, col2=w2.value,\
#                                                          col3=w3.value, Export=w_export))
                                                        
```

### Project data to 3-D principal component space (on sampled data)


```python
# md_text = '## Project Data to 3-D Principal Component Space (on sampled data)'
# if len(conf_dict['NumericalColumns']) > 3:
#     filename = ReportMagic.var_files='tmp/pca3d.csv'
#     export_filename = 'pca3d_report2.ipynb'
#     silentremove(filename) 
#     if conf_dict['Target'] in conf_dict['CategoricalColumns']:
#         cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
#         cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
#     else:
#         cols_list = conf_dict['CategoricalColumns']
#     w1, w2, w3, w4 = None, None, None, None
#     w1 = widgets.Dropdown(
#         options=cols_list,
#         value=cols_list[0],
#         description='Legend Variable:'
#     )
#     w2 = ipywidgets.IntSlider(min=-180, max=180, step=5, value=30, description='Angle')
#     w_export = widgets.Button(description='Export',value='Export')
#     handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, \
#                                     export_filename=export_filename, md_text=md_text)     
#     w1.observe(handle_change,'value')
#     w_export.on_click(handle_export_partial)
    
#     %reset_report
#     %add_interaction_code_to_report i = interactive(InteractionAnalytics.pca_3d, df=fixed(df), conf_dict=fixed(conf_dict),\
#                                               col1=w1, col2=w2, col3=fixed(w3),Export=w_export)

#     hbox=widgets.HBox(i.children)
#     display(hbox)
#     hbox.on_displayed(InteractionAnalytics.pca_3d(df,conf_dict,col1=w1.value,col2=w2.value,Export=w_export))
```

## <a name="report"></a>Generate the Data Report


```python
filenames = ['target_report2.ipynb', 'numeric_report2.ipynb', 'categoric_report2.ipynb', 'rank_report2.ipynb',
           'cat_interactions_report2.ipynb', 'numerical_interactions_report2.ipynb',
             'numerical_correlations_report2.ipynb', 'nc_report2.ipynb',
             'nnc_report2.ipynb', 'numerical_pca_report2.ipynb', 'pca3d_report2.ipynb'
           ]

def merge_notebooks():
    merged = None
    for fname in filenames:
        try:
            print ('Processing {}'.format(export_dir+fname))
            with io.open(export_dir+fname, 'r', encoding='utf-8') as f:
                nb = nbf.read(f, as_version=4)
            if merged is None:
                merged = nb
            else:
                merged.cells.extend(nb.cells[2:])
        except:
            print ('Warning: Unable to find the file', export_dir+'//'+fname, ', continue...')
    if not hasattr(merged.metadata, 'name'):
        merged.metadata.name = ''
    merged.metadata.name += "_merged"
    with open(merged_report, 'w') as f:
        nbf.write(merged, f)

def gen_merged_report(b):
    merge_notebooks()
    
# button=widgets.Button(description='Generate Final Report')
# button.on_click(gen_merged_report)
# display(button)
```

## <a name="show hide codes"></a>Show/Hide the Source Codes


```python
# Provide the path to the yaml file relative to the working directory
display(HTML('''<style>
    .widget-label { min-width: 20ex !important; }
    .widget-text { min-width: 60ex !important; }
</style>'''))

#Toggle Code
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();

 } else {
 $('div.input').show();

 }
 code_show = !code_show
} 
//$( document ).ready(code_toggle);//commenting code disabling by default
</script>
<form action = "javascript:code_toggle()"><input type="submit" value="Toggle Raw Code"></form>''')
```


<style>
    .widget-label { min-width: 20ex !important; }
    .widget-text { min-width: 60ex !important; }
</style>





<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();

 } else {
 $('div.input').show();

 }
 code_show = !code_show
} 
//$( document ).ready(code_toggle);//commenting code disabling by default
</script>
<form action = "javascript:code_toggle()"><input type="submit" value="Toggle Raw Code"></form>


