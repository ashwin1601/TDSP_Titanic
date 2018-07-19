
# coding: utf-8

# # Interactive Data Exploration, Analysis, and Reporting
# 
# - Author: Team Data Science Process from Microsoft 
# - Date: 2017/03
# - Supported Data Sources: CSV files on the machine where the Jupyter notebook runs or data stored in SQL server
# - Output: IDEAR_Report.ipynb
# 
# 
# This is the **Interactive Data Exploration, Analysis and Reporting (IDEAR)** in _**Python**_ running on Jupyter Notebook. The data can be stored in CSV file on the machine where the Jupyter notebook runs or from a query running against a SQL server. A yaml file has to be pre-configured before running this tool to provide information about the data. 
# 
# ## Step 1: Configure and Set up IDEAR
# 
# Before start utilitizing the functionalities provided by IDEAR, you need to first [configure and set up](#setup) the utilities by providing the yaml file and load necessary Python modules and libraries. 
# 
# ## Step 2: Start using IDEAR
# This tool provides various functionalities to help users explore the data and get insights through interactive visualization and statistical testing. 
# 
# - [Read and Summarize the data](#read and summarize)
# 
# - [Extract Descriptive Statistics of Data](#descriptive statistics)
# 
# - [Explore Individual Variables](#individual variables)
# 
# - [Explore Interactions between Variables](#multiple variables)
# 
#     - [Rank variables](#rank variables)
#     
#     - [Interaction between two categorical variables](#two categorical)
#     
#     - [Interaction between two numerical variables](#two numerical)
# 
#     - [Interaction between numerical and categorical variables](#numerical and categorical)
# 
#     - [Interaction between two numerical variables and a categorical variable](#two numerical and categorical)
# 
# - [Visualize High Dimensional Data via Projecting to Lower Dimension Principal Component Spaces](#pca)
# 
# - [Generate Data Report](#report)
# 
# After you are done with exploring the data interactively, you can choose to [show/hide the source code](#show hide codes) to make your notebook look neater. 
# 
# **Note**:
# 
# - Change the working directory and yaml file before running IDEAR in Jupyter Notebook.
# 
# - Run the cells and click *Export* button to export the code that generates the visualization/analysis result to temporary Jupyter notebooks.
# 
# - Run the last cell and click [***Generate Final Report***](#report) to create *IDEAR_Report.ipynb* in the working directory. _If you do not export codes in some sections, you may see some warnings complaining that some temporary Jupyter Notebook files are missing_. 
# 
# - Upload *IDEAR_Report.ipynb* to Jupyter Notebook server, and run it to generate report.

# ## <a name="setup"></a>Global Configuration and Setting Up

# In[28]:


# Set the working directory as the directory where ReportMagics.py stays
# Use \\ in your path
import os
workingDir = 'C:/Users\946068\Documents\Azure-TDSP-Utilities-master\Azure-TDSP-Utilities-master\DataScienceUtilities\DataReport-Utils\Python'
os.chdir(workingDir)

from ReportMagics import *

merged_report ='IDEAR_Report.ipynb'
get_ipython().run_line_magic('reset_all', '')


# In[30]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "import os\nworkingDir = 'C:\\\\Users\\946068\\Documents\\Azure-TDSP-Utilities-master\\Azure-TDSP-Utilities-master\\DataScienceUtilities\\DataReport-Utils\\Python'\nos.chdir(workingDir)\n\nconf_file = '.\\\\para-adult.yaml'\nSample_Size = 10000\n\nexport_dir = '.\\\\tmp\\\\'")


# ### Import necessary packages and set up environment parameters

# In[66]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "\nimport pandas as pd\nimport numpy as np\nimport os\n#os.chdir(workingDir)\nimport collections\nimport matplotlib\nimport io\nimport sys\nimport operator\n\nimport nbformat as nbf\nfrom IPython.core.display import HTML\nfrom IPython.display import display\nfrom ipywidgets import interact, interactive,fixed\nfrom IPython.display import Javascript, display,HTML\nfrom ipywidgets import widgets, VBox\nfrom ipywidgets import *\nimport ipywidgets as widgets\nimport IPython\nfrom IPython.display import clear_output\nimport scipy.stats as stats\nfrom statsmodels.graphics.mosaicplot import mosaic\nimport statsmodels.api as sm\nfrom statsmodels.formula.api import ols\nimport os\nimport errno\nimport seaborn as sns\nfrom string import Template\nfrom functools import partial\nfrom collections import OrderedDict\n\n# Utility Classes\nfrom ConfUtility import * \nfrom ReportGeneration import *\nfrom UniVarAnalytics import *\nfrom MultiVarAnalytics import *\n\n%matplotlib inline\n\n#DEBUG=0\n\nfont={'family':'normal','weight':'normal','size':8}\nmatplotlib.rc('font',**font)\nmatplotlib.rcParams['figure.figsize'] = (12.0, 5.0)\nmatplotlib.rc('xtick', labelsize=9) \nmatplotlib.rc('ytick', labelsize=9)\nmatplotlib.rc('axes', labelsize=10)\nmatplotlib.rc('axes', titlesize=10)\nsns.set_style('whitegrid')")


# ### Define some functions for generating reports

# In[32]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', '\nif not os.path.exists(export_dir):\n    os.makedirs(export_dir)\n    \ndef gen_report(conf_md,conf_code, md, code, filename):\n    ReportGeneration.write_report(conf_md, conf_code, md, code, report_name=filename)\n\ndef translate_code_commands(cell, exported_cols, composite=False):    \n    new_code_store = []\n    exported_cols = [each for each in exported_cols if each!=\'\']   \n    for each in exported_cols:       \n        w,x,y = each.split(\',\')\n        with open(\'log.txt\',\'w\') as fout:\n            fout.write(\'Processing call for the column {}\'.format(each))\n        temp=cell[0]\n\n        new_line = temp.replace(\'interactive\',\'apply\').replace(\n            "df=fixed(df)","df").replace("filename=fixed(filename)","\'"+ReportMagic.var_files+"\'").replace(\n            "col1=w1","\'"+w+"\'").replace("col2=w2","\'"+x+"\'").replace("col3=w3","\'"+y+"\'").replace(\n            "col3=fixed(w3)","\'"+y+"\'").replace(\n            "Export=w_export","False").replace("conf_dict=fixed(conf_dict)","conf_dict")       \n        new_line = new_line.replace("df,","[df,")\n        new_line = new_line[:len(new_line)-1]+"])"\n        new_line = new_line.replace("apply(","").replace(", [", "(*[")\n        new_code_store.append(new_line)        \n    return new_code_store\n\ndef add_to_report(section=\'\', task=\'\'):\n    print (\'Section {}, Task {} added for report generation\'.format(section ,task))\n\ndef trigger_report(widgets,export_cols_file, output_report, no_widgets=1, md_text=\'\'):\n    exported_cols = []\n    with open(export_cols_file,\'r\') as fin:\n        for each in fin:\n            each = each.strip()\n            if each and not each.isspace():\n                exported_cols.append(each)\n                \n    exported_cols = list(set(exported_cols))\n    conf_md, conf_code, md, code=%show_report \n    md = md_text\n    cell = code\n    new_code_store = translate_code_commands(cell,exported_cols)\n    gen_report(conf_md,conf_code, md, new_code_store, filename=export_dir+output_report)\n    \ndef silentremove(filename):\n    try:\n        os.remove(filename)\n    except OSError as e: # this would be "except OSError, e:" before Python 2.6\n        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory\n            raise # re-raise exception if a different error occured\n\ndef handle_change(value):\n    w_export.value=False\n\ndef getWidgetValue(w):\n    w_value = \'\'\n    try:\n        w_value = w.value\n    except:\n        pass    \n    return w_value\n\ndef handle_export(widget, w1, w2, w3, export_filename=\'temp.ipynb\',md_text=\'\'):    \n    print (\'Export is successful!\')\n    w1_value, w2_value, w3_value = \\\n        getWidgetValue(w1),getWidgetValue(w2),getWidgetValue(w3)\n    st = \',\'.join(str(each) for each in [w1_value, w2_value, w3_value])\n    with open(filename,\'a\') as fout:\n        fout.write(st+\'\\n\')\n    trigger_report(w1_value, filename, export_filename, False, md_text=md_text)  \n    \n            ')


# ## <a name="read and summarize"></a> Read and Summarize the Data

# ### Read data and infer column types

# In[33]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "\nconf_dict = ConfUtility.parse_yaml(conf_file)\n\n# Read in data from local file or SQL server\nif 'DataSource' not in conf_dict:\n    df=pd.read_csv(conf_dict['DataFilePath'][0], skipinitialspace=True)\nelse:\n    import pyodbc\n    cnxn = pyodbc.connect('driver=ODBC Driver 11 for SQL Server;server={};database={};Uid={};Pwd={}'.format(\n            conf_dict['Server'], conf_dict['Database'],conf_dict['Username'],conf_dict['Password']))\n    df = pd.read_sql(conf_dict['Query'],cnxn)\n\n# Making sure that we are not reading any extra column\ndf = df[[each for each in df.columns if 'Unnamed' not in each]]\n\n# Sampling Data if data size is larger than 10k\ndf0 = df # df0 is the unsampled data. Will be used in data exploration and analysis where sampling is not needed\n         # However, keep in mind that your final report will always be based on the sampled data. \nif Sample_Size < df.shape[0]:\n    df = df.sample(Sample_Size)\n\n# change float data types\nif 'FloatDataTypes' in conf_dict:   \n    for col_name in conf_dict['FloatDataTypes']:\n        df[col_name] = df[col_name].astype(float)      \n        \n# Getting the list of categorical columns if it was not there in the yaml file\nif 'CategoricalColumns' not in conf_dict:\n    conf_dict['CategoricalColumns'] = list(set(list(df.select_dtypes(exclude=[np.number]).columns)))\n\n# Getting the list of numerical columns if it was not there in the yaml file\nif 'NumericalColumns' not in conf_dict:\n    conf_dict['NumericalColumns'] = list(df.select_dtypes(include=[np.number]).columns)    \n\n# Exclude columns that we do not need\nif 'ColumnsToExclude' in conf_dict:\n    conf_dict['CategoricalColumns'] = list(set(conf_dict['CategoricalColumns'])-set(conf_dict['ColumnsToExclude']))\n    conf_dict['NumericalColumns'] = list(set(conf_dict['NumericalColumns'])-set(conf_dict['ColumnsToExclude']))\n\n# Ordering the categorical variables according to the number of unique categories\nfiltered_cat_columns = []\ntemp_dict = {}\nfor cat_var in conf_dict['CategoricalColumns']:\n    temp_dict[cat_var] = len(np.unique(df[cat_var]))\nsorted_x = sorted(temp_dict.items(), key=operator.itemgetter(0), reverse=True)\nconf_dict['CategoricalColumns'] = [x for (x,y) in sorted_x]\n\nConfUtility.dict_to_htmllist(conf_dict,['Target','CategoricalColumns','NumericalColumns'])")


# ### Print the first n (n=5 by default) rows of the data

# In[34]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', 'def custom_head(df,NoOfRows):\n    return HTML(df.head(NoOfRows).style.set_table_attributes("class=\'table\'").render())\ni = interact(custom_head,df=fixed(df0), NoOfRows=ipywidgets.IntSlider(min=0, max=30, step=1, \\\n                                                                     value=5, description=\'Number of Rows\'))')


# ### Print the dimensions of the data (rows, columns)

# In[35]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "print ('The data has {} Rows and {} columns'.format(df0.shape[0],df0.shape[1]))")


# ### Print the column names of the data

# In[36]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', 'col_names = \',\'.join(each for each in list(df.columns))\nprint("The column names are:" + col_names)')


# ### Print the column types

# In[37]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', 'print("The types of columns are:")\ndf.dtypes')


# ## <a name="individual variable"></a>Extract Descriptive Statistics of Each Column

# In[38]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "def num_missing(x):\n    return len(x.index)-x.count()\n\ndef num_unique(x):\n    return len(np.unique(x))\n\ntemp_df = df0.describe().T\nmissing_df = pd.DataFrame(df0.apply(num_missing, axis=0)) \nmissing_df.columns = ['missing']\nunq_df = pd.DataFrame(df0.apply(num_unique, axis=0))\nunq_df.columns = ['unique']\ntypes_df = pd.DataFrame(df0.dtypes)\ntypes_df.columns = ['DataType']")


# ### Print the descriptive statistics of numerical columns

# In[39]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', 'summary_df = temp_df.join(missing_df).join(unq_df).join(types_df)\nsummary_df')


# ### Print the descriptive statistics of categorical columns

# In[40]:


get_ipython().run_cell_magic('add_conf_code_to_report', '', "col_names = list(types_df.index) #Get all col names\nnum_cols = len(col_names)\nindex = range(num_cols)\ncat_index = []\nfor i in index: #Find the indices of columns in Categorical columns\n    if col_names[i] in conf_dict['CategoricalColumns']:\n        cat_index.append(i)\nsummary_df_cat = missing_df.join(unq_df).join(types_df.iloc[cat_index], how='inner') #Only summarize categorical columns\nsummary_df_cat")


# ## <a name="individual variables"></a>Explore Individual Variables

# ### Explore the target variable

# In[43]:


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

    get_ipython().run_line_magic('reset_report', '')
    get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
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

    get_ipython().run_line_magic('reset_report', '')
    get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(NumericAnalytics.custom_barplot, df=fixed(df), filename=fixed(filename),                                                    col1=w1, Export=w_export)')
    hbox = widgets.HBox(ii.children)
    display(hbox)
    hbox.on_displayed(NumericAnalytics.custom_barplot(df=df, filename=filename, col1=w1.value, Export=w_export))


# ### Explore individual numeric variables and test for normality (on sampled data)

# In[45]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(NumericAnalytics.custom_barplot(df=df, filename=filename, col1=w1.value, Export=w_export))


# ### Explore individual categorical variables (sorted by frequencies)

# In[47]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')

hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(CategoricAnalytics.custom_barplot(df=df0, filename=filename, col1=w1.value,                                          Export=w_export))


# ## <a name="multiple variables"></a>Explore Interactions Between Variables

# ### <a name="rank variables"></a>Rank variables based on linear relationships with reference variable (on sampled data)

# In[49]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.rank_associations(df=df, conf_dict=conf_dict, col1=w1.value,                                                          col2=w2.value, col3=w3.value, Export=w_export))


# ### <a name="two categorical"></a>Explore interactions between categorical variables

# In[51]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.categorical_relations(df=df0, filename=filename, col1=w1.value,                                                              col2=w2.value, Export=w_export))


# ### <a name="two numerical"></a>Explore interactions between numerical variables (on sampled data)

# In[52]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.numerical_relations(df, col1=w1.value, col2=w2.value, Export=w_export))


# ### Explore correlation matrix between numerical variables

# In[53]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')

hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.numerical_correlation(df0, conf_dict=conf_dict, col1=w1.value, Export=w_export))


# ### <a name="numerical and categorical"></a>Explore interactions between numerical and categorical variables

# In[55]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')

hbox = widgets.HBox(i.children)
display( hbox )
hbox.on_displayed(InteractionAnalytics.nc_relation(df0, conf_dict, col1=w1.value, col2=w2.value, Export=w_export))


# ### <a name="two numerical and categorical"></a>Explore interactions between two numerical variables and a categorical variable (on sampled data)

# In[56]:


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

get_ipython().run_line_magic('reset_report', '')
get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')
hbox = widgets.HBox(i.children)
display(hbox)
hbox.on_displayed(InteractionAnalytics.nnc_relation(df, conf_dict, col1=w1.value,                                                    col2=w2.value, col3=w3.value, Export=w_export))


# ## <a name="pca"></a>Visualize numerical data by projecting to principal component spaces (on sampled data)

# ### Project data to 2-D principal component space (on sampled data)

# In[57]:


num_numeric = len(conf_dict['NumericalColumns'])
if  num_numeric > 3:
    md_text = '## Project Data to 2-D Principal Component Space'
    filename = ReportMagic.var_files = 'tmp/numerical_pca.csv'
    export_filename = 'numerical_pca_report2.ipynb'
    silentremove(filename) 
    
    w1, w2, w3, w4, w5 = None, None, None, None, None
    if conf_dict['Target'] in conf_dict['CategoricalColumns']:
        cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
        cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
    else:
        cols_list = conf_dict['CategoricalColumns']
    w1 = widgets.Dropdown(
        options = cols_list,
        value = cols_list[0],
        description = 'Legend Variable:',
        width = 10
    )
    w2 = widgets.Dropdown(
        options = [str(x) for x in np.arange(1,num_numeric+1)],
        value = '1',
        width = 1,
        description='PC at X-Axis:'
    )
    w3 = widgets.Dropdown(
        options = [str(x) for x in np.arange(1,num_numeric+1)],
        value = '2',
        description = 'PC at Y-Axis:'
    )
    w_export = widgets.Button(description='Export', value=False, options=[True, False])
    handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3, export_filename=export_filename, md_text=md_text)       
    w1.observe(handle_change,'value')
    w_export.on_click(handle_export_partial)
    
    get_ipython().run_line_magic('reset_report', '')
    get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')

    
    hbox = widgets.HBox(i.children)
    display(hbox)
    hbox.on_displayed(InteractionAnalytics.numerical_pca(df, conf_dict=conf_dict, col1=w1.value, col2=w2.value,                                                         col3=w3.value, Export=w_export))


# ### Project data to 3-D principal component space (on sampled data)

# In[58]:


md_text = '## Project Data to 3-D Principal Component Space (on sampled data)'
if len(conf_dict['NumericalColumns']) > 3:
    filename = ReportMagic.var_files='tmp/pca3d.csv'
    export_filename = 'pca3d_report2.ipynb'
    silentremove(filename) 
    if conf_dict['Target'] in conf_dict['CategoricalColumns']:
        cols_list = [conf_dict['Target']] + conf_dict['CategoricalColumns'] #Make target the default reference variable
        cols_list = list(OrderedDict.fromkeys(cols_list)) #remove variables that might be duplicates with target
    else:
        cols_list = conf_dict['CategoricalColumns']
    w1, w2, w3, w4 = None, None, None, None
    w1 = widgets.Dropdown(
        options=cols_list,
        value=cols_list[0],
        description='Legend Variable:'
    )
    w2 = ipywidgets.IntSlider(min=-180, max=180, step=5, value=30, description='Angle')
    w_export = widgets.Button(description='Export',value='Export')
    handle_export_partial = partial(handle_export, w1=w1, w2=w2, w3=w3,                                     export_filename=export_filename, md_text=md_text)     
    w1.observe(handle_change,'value')
    w_export.on_click(handle_export_partial)
    
    get_ipython().run_line_magic('reset_report', '')
    get_ipython().run_line_magic('add_interaction_code_to_report', 'i = interactive(TargetAnalytics.custom_barplot, df=fixed(df), col1=w1)')

    hbox=widgets.HBox(i.children)
    display(hbox)
    hbox.on_displayed(InteractionAnalytics.pca_3d(df,conf_dict,col1=w1.value,col2=w2.value,Export=w_export))


# ## <a name="report"></a>Generate the Data Report

# In[69]:


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
    
button = widgets.Button(description='Generate Final Report')

button.on_click(gen_merged_report)
display(button)


# ## <a name="show hide codes"></a>Show/Hide the Source Codes

# In[61]:


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


# In[71]:


widgets.Button(description='Generate Final Report', style=ButtonStyle())

