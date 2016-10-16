from __future__ import division
#from os.path import dirname, join
from StringIO import StringIO
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import simplejson
#import seaborn as sns
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import requests
import itertools
#import pylab as pl
#import matplotlib.pyplot as plt

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput, RadioButtonGroup, Panel, Tabs, Button
from bokeh.io import curdoc, gridplot, output_file, show
#from bokeh.sampledata.movies_data import movie_path
#from bokeh.layouts import gridplot #column
from operator import itemgetter
from datetime import date
from random import randint
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import HoverTool, BoxSelectTool

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import neighbors, datasets

#import copy

# ================================================= hack code =========================================
# origin: my playground/drug_ipython.py
# first 193 lines of codes

# openFDA API key: TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU
#drugname1 = 'tylenol'
#drugname2 = 'ibuprofen'
#drugname3 = 'aspirin'
#drugname4 = 'advil'
# top n adverse effects
#print 'PATH: ' + str(join(dirname(__file__), 'drugs.txt'))
totalreports = 0
f = open('data/log.txt', 'w')
n = 8*2
api_key = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU&search'
reacts_topn = ['NAUSEA','DEPRESSION','FATIGUE','ANXIETY','DIZZINESS','VOMITING', \
          'CONSTIPATION','RASH','PAIN','INSOMNIA']
dfROR = pd.DataFrame # data frame for Proportion of Reporting Ratio of Adverse Effects
dictROR = {
   'drugname':     str,
   'n':            int,
   'eventname':    str,
   'M':            int,
   'm':            int,
   'PPR':          float
   }
          
def get_drug_adverse_event_data_(drugname):
    #global api_key
    
    api_key = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU&search'
    request_string='https://api.fda.gov/drug/event.json?api_key=' + api_key
    request_string= request_string + '=patient.drug.medicinalproduct:' + drugname
    request_string= request_string +'&count=patient.reaction.reactionmeddrapt.exact'
    
    #print request_string
    json_df = requests.get(request_string).json()
    #print 'json_df = requests.get(request_string).json()'
    #print drugname
    #print 'api_key: ' + api_key
    json_adverse_list = json_df['results']
    #copy_ = list(json_adverse_list)
    #print 'jason_df[results] ' + str(json_adverse_list)
    #copy_json_adverse_list = copy.deepcopy(json_adverse_list)

    #print 'json_adverse_list[0].count: ' + str(json_adverse_list[0]['count'])
    #json_adverse_list[0]['count'] = 100000    
    #print 'json_adverse_list[0].count: ' + str(json_adverse_list[0]['count'])    
    '''
    total100 = 0
    for i in json_adverse_list:
        #print 'i[count] ' + str(i['count'])
        total100 += i['count']
    #print 'total100 ' + str(total100)
    for i, x in enumerate(json_adverse_list):
        json_adverse_list[i]['count'] = float(json_adverse_list[i]['count']/total100)
    '''
    #print 'json_adverse_list just before return: ' + str(json_adverse_list)
    #print 'copy : ' + str(copy_)
    return json_adverse_list
    
def get_drug_adverse_event_data(drugname):
    #global api_key
    
    api_key = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU&search'
    request_string='https://api.fda.gov/drug/event.json?api_key=' + api_key
    request_string= request_string + '=patient.drug.medicinalproduct:' + drugname
    request_string= request_string +'&count=patient.reaction.reactionmeddrapt.exact'
    
    #print request_string
    json_df = requests.get(request_string).json()
    #print 'json_df = requests.get(request_string).json()'
    #print drugname
    #print 'api_key: ' + api_key
    print request_string
    json_adverse_list = json_df['results']
    #copy_ = list(json_adverse_list)
    #print 'jason_df[results] ' + str(json_adverse_list)
    #copy_json_adverse_list = copy.deepcopy(json_adverse_list)

    #print 'json_adverse_list[0].count: ' + str(json_adverse_list[0]['count'])
    #json_adverse_list[0]['count'] = 100000    
    #print 'json_adverse_list[0].count: ' + str(json_adverse_list[0]['count'])    
    
    total100 = 0
    for i in json_adverse_list:
        #print 'i[count] ' + str(i['count'])
        total100 += i['count']
    #print 'total100 ' + str(total100)
    for i, x in enumerate(json_adverse_list):
        json_adverse_list[i]['count'] = float(json_adverse_list[i]['count']/total100)
    
    #print 'json_adverse_list just before return: ' + str(json_adverse_list)
    #print 'copy : ' + str(copy_)
    return json_adverse_list
    
def get_event_freq(event_list, event):
    try:
        index=map(itemgetter('term'), event_list).index(event)
        return event_list[index].get('count')
    except ValueError:
        return 0
        
def get_ROR(listof_dnames=None):
# --------------------------------------
# The proportional reporting ratio (PRR) is a simple way to get a measure of how common an adverse event
# for a particular drug is compared to how common the event is in the overall database. 
# A PRR > 1 for a drug-event combination indicates that a greater proportion of the reports for the drug
# are for the event than the proportion of events in the rest of the database. 
# E.g., a PRR of 2 for a drug event combination indicates that the proportion of reports for the drug-event
# combination is twice the proportion of the event in the overall database of reported adverse effects.

# PRR = (m/n)/( (M-m)/(N-n) )
# Where 
# m = #reports with drug and event
# n = #reports with drug
# M = #reports with event in database
# N = #reports in database

# A similar measure is the reporting odds ratio (ROR).

# ROR = (m/d)/(M/D) 
# Where 
# m = #reports with drug and event
# d = n-m
# M = #reports with event in database
# D = N-M

    # Step 1. we first find N total # of reports in the openFDA db
    # Step 2. then M the # of reports for ea. event in the list of events (top 100) in the database    
    # Step 3. loop through list of drug names and find n # of reports with drug
    # Step 4. and m # of reports with drug and event     

# dfROR data frame:
# drug name, n, event, M, PRR    

### N: reports in database
    '''
    https://api.fda.gov/drug/event.json?search=receivedate:[19890101+TO+20160601]&count=receivedate
    '''
    global dfROR
    #global api_key
    
    api_key = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU&search'
    _string='https://api.fda.gov/drug/event.json?api_key=' + api_key
    _string=_string+'=receivedate:[19890101+TO+20160601]&count=receivedate'
    N_qry = requests.get(_string).json()
    N_qry = N_qry['results']
    N = sum(item['count'] for item in N_qry) # 6747466
    print "sum(item['count'] for item in N_qry)"    
    print 'N = ' + str(N)
    dictROR = {
       'drugname':     str,
       'n':            int,
       'eventname':    str,
       'M':            int,
       'm':            int,
       'PRR':          float
       }
    
    ### n: reports with drug
    # listof_dnames = ['aleve','caplet','tylenol','ibuprofen','advil','excedrin','aspirin','vitamin','hcl','acetaminophen']
    l_dictROR = []
    # loading M reports with event in the database
    dfadverse = pd.DataFrame
    #dfadverse = pd.read_csv(join(dirname(__file__), 'adverse.csv')) # /Users/rogerstanev/github/bokeh/examples/app/drugs/
    dfadverse = pd.read_csv('data/adverse.csv') # /Users/rogerstanev/github/bokeh/examples/app/drugs/
    dfadverse = dfadverse.set_index('term')
    
    ### populating list of dictionaries
    
    for i in range(len(listof_dnames)):
        
        _string='https://api.fda.gov/drug/event.json?api_key=' + api_key
        _string=_string+'=receivedate:[19890101+TO+20160601]+AND+patient.drug.medicinalproduct:'+listof_dnames[i]+'&count=receivedate'
        m_qry = requests.get(_string).json()
        m_qry = m_qry['results']
        n = sum(item['count'] for item in m_qry)
        
        devent_data = get_drug_adverse_event_data_(listof_dnames[i])
        
        for j in range(len(devent_data)):
            
            copy = dictROR.copy()
            copy['drugname'] = listof_dnames[i]
            copy['n'] = n
            copy['eventname'] = devent_data[j]['term']
            copy['M'] = int(dfadverse.loc[copy['eventname']])
            copy['m'] = devent_data[j]['count']
            copy['PRR'] = (copy['m']/copy['n']) / ( (copy['M']-copy['m'])/(N-copy['m']) ) # PRR = (m/n)/((M-m)/(N-n))
            l_dictROR.append(copy)

### THE complete data for PRR (ROR) plot
    df_all = pd.DataFrame(l_dictROR)
    df_all.to_csv('data/PRR.csv')
    return df_all
'''
https://api.fda.gov/drug/event.json?search=receivedate:[19890101+TO+20160601]+AND+
patient.drug.medicinalproduct:(%22ASPIRIN%22)&count=receivedate
'''


def get_dots(listof_dnames=None):
    
    
    ##############################################################################################
    global reacts_topn
    
    print 'get_dots : listof_dnames : ' + str(listof_dnames)
    drugname1 = listof_dnames[0]
    drugname2 = listof_dnames[1]
    drugname3 = listof_dnames[2]
    drugname4 = listof_dnames[3]
    
    h_ = 240
    w_ = 360
    
    factors = reacts_topn
    #print 'drugname1 ' + drugname1
    dlist1=get_drug_adverse_event_data(drugname1)
    #print 'dlist1 ' + str(dlist1)
    dlist2=get_drug_adverse_event_data(drugname2)
    dlist3=get_drug_adverse_event_data(drugname3)
    dlist4=get_drug_adverse_event_data(drugname4)
    
    d1_freq_list = [None] * n
    d2_freq_list = [None] * n
    d3_freq_list = [None] * n
    d4_freq_list = [None] * n
    
    for i in range(len(reacts_topn)):
        d1_freq_list[i]=get_event_freq(dlist1, reacts_topn[i])
        d2_freq_list[i]=get_event_freq(dlist2, reacts_topn[i])
        d3_freq_list[i]=get_event_freq(dlist3, reacts_topn[i])
        d4_freq_list[i]=get_event_freq(dlist4, reacts_topn[i])
        
    x = d1_freq_list
    #print 'd1_freq_list :' + str(x)
    
    dot1 = figure(title="Common adverse effects for %s"%(drugname1), tools="", toolbar_location=None,
        y_range=factors, x_range=[0,max(d1_freq_list)*1.1])
    
    dot1.segment(0, factors, x, factors, line_width=2, line_color="green")
    dot1.circle(x, factors, size=15, fill_color='#FB9A99', line_color="green", line_width=3, )
    dot1.plot_height=h_
    dot1.plot_width=w_
    dot1.xaxis.axis_label='probability'
    
    x = d2_freq_list
    #print 'd2_freq_list :' + str(x)
    
    dot2 = figure(title="Common adverse effects for %s"%(drugname2), tools="", toolbar_location=None,
        y_range=factors, x_range=[0,max(d1_freq_list)*1.1])
    
    dot2.segment(0, factors, x, factors, line_width=2, line_color="green")
    dot2.circle(x, factors, size=15, fill_color='#B2DF8A', line_color="green", line_width=3, )
    dot2.plot_height=h_
    dot2.plot_width=w_
    dot2.xaxis.axis_label='probability'
    
    x = d3_freq_list
    #print 'd3_freq_list :' + str(x)
    
    dot3 = figure(title="Common adverse effects for %s"%(drugname3), tools="", toolbar_location=None,
        y_range=factors, x_range=[0,max(d1_freq_list)*1.1])
    
    dot3.segment(0, factors, x, factors, line_width=2, line_color="green")
    dot3.circle(x, factors, size=15, fill_color='#A6CEE3', line_color="green", line_width=3, )
    dot3.plot_height=h_
    dot3.plot_width=w_
    dot3.xaxis.axis_label='probability'
    
    x = d4_freq_list
    #print 'd4_freq_list :' + str(x)
    
    dot4 = figure(title="Common adverse effects for %s"%(drugname4), tools="", toolbar_location=None,
        y_range=factors, x_range=[0,max(d1_freq_list)*1.1])
    
    dot4.segment(0, factors, x, factors, line_width=2, line_color="green")
    dot4.circle(x, factors, size=15, fill_color='orange', line_color="green", line_width=3, )
    dot4.plot_height=h_
    dot4.plot_width=w_
    dot4.xaxis.axis_label='probability'
    
    #output_file("dotadverse.html", title="adverse effects")
    #output_file("dotadverse.html", title="adverse effects")
    #output_notebook()
    gp = gridplot([[dot1, dot2], [dot3, dot4]])
    return gp

# ================================================ end of hack code ===================================

'''
conn = sql.connect(movie_path)
query = open(join(dirname(__file__), 'query.sql')).read()
movies = psql.read_sql(query, conn)
'''
def datetime(x):
    return np.array(x, dtype=np.datetime64)

api_key        = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU'
request_string = 'https://api.fda.gov/drug/event.json?api_key=' + api_key
master_string  = request_string
r='&search=receivedate:[20040101+TO+20160601]&count=receivedate'
r=master_string+r

# line 333
f.write('line 333')

'''
movies["color"] = np.where(movies["Oscars"] > 0, "orange", "grey")
movies["alpha"] = np.where(movies["Oscars"] > 0, 0.9, 0.25)
movies.fillna(0, inplace=True)  # just replace missing values with zero
movies["revenue"] = movies.BoxOffice.apply(lambda x: '{:,d}'.format(int(x)))
with open(join(dirname(__file__), "razzies-clean.csv")) as f:
    razzies = f.read().splitlines()
movies.loc[movies.imdbID.isin(razzies), "color"] = "purple"
movies.loc[movies.imdbID.isin(razzies), "alpha"] = 0.9
axis_map = {
    "Tomato Meter": "Meter",
    "Numeric Rating": "numericRating",
    "Number of Reviews": "Reviews",
    "Box Office (dollars)": "BoxOffice",
    "Length (minutes)": "Runtime",
    "Year": "Year",
}
'''
def make_roc_curve(
        clf,
        X_test,
        y_test,
        p_cut_list=np.arange(0., 1.01, 0.1),
        ):

        p_pos_test = clf.predict_proba(X_test)[:,0]

        i_neg = np.where(y_test==2.)[0]
        i_pos = np.where(y_test==1.)[0]
        num_neg = float(len(i_neg))
        num_pos = float(len(i_pos))

        false_pos_rate = np.empty(len(p_cut_list))
        false_pos_rate.fill(np.nan)

        true_pos_rate = np.empty(len(p_cut_list))
        true_pos_rate.fill(np.nan)
        
        for (i, p_cut) in enumerate(p_cut_list):
            
            i_neg_cut = np.where(p_pos_test[i_neg] > p_cut)[0]
            i_pos_cut = np.where(p_pos_test[i_pos] > p_cut)[0]

            num_false_pos = len(i_neg[i_neg_cut])
            num_true_pos = len(i_pos[i_pos_cut])

            false_pos_rate[i] = num_false_pos / num_neg
            true_pos_rate[i] = num_true_pos / num_pos

        return (false_pos_rate, true_pos_rate, num_pos, num_neg)




desc = Div(text=open('data/description.html').read(), width=800)
f.write('desc: ')
f.write(str(desc))

# Create Input controls
reviews = Slider(title="Minimum number of adverse reports:", value=80, start=10, end=300, step=10)
min_year = Slider(title="Starting report year", start=1990, end=2016, value=2004, step=1)
max_year = Slider(title="Age", start=0, end=120, value=35, step=1)
oscars = Slider(title="Weight (kg)", start=0, end=200, value=75, step=1)
#boxoffice = Slider(title="Number of adverse effect events:", start=0, end=800, value=0, step=1)
#checkbox_group = CheckboxGroup(
#        labels=["Male", "Female", "Both"], active=[0, 1])
radio_group = RadioButtonGroup(labels=["Male", "Female", "Either"], active=1)

indication = Select(title="Drug indication:", value="HEADACHE",
               options=open('data/indication.txt').read().split('\n'))
f.write('indication: ')
f.write(str(indication))


dname = Select(title="Drug name:", value="ALEVE", 
               options=open('data/drugs.txt').read().split('\n'))
f.write('dname: ')
f.write(str(dname))

set_indication_val = indication.value
set_dname_val      = dname.value
#print dname.enabled

director = TextInput(title="Drug name:")
cast = TextInput(title="Adverse effect name:")
#x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Tomato Meter")
#y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Number of Reviews")
# Create Column Data Source that will be used by the plot
sourced = ColumnDataSource(data=dict(x=[], y=[]))

# Add hover to this comma-separated string and see what changes
TOOLS = 'box_zoom,box_select,crosshair,resize,reset,hover'
#TOOLS = [BoxSelectTool(),HoverTool()]
#p2 = figure(width=720, height=510, toolbar_location='above', tools=TOOLS)
#print 'BEFORE calling get_dots: ' + str(dname.options[0:4])
gp_ = get_dots(dname.options[0:4])

p1 = figure(x_axis_type="datetime", width=720, height=510, toolbar_location='above')
p1.line(x="x", y="y", source=sourced, color='#FB9A99')    

tab1 = Panel(child=p1, title='R')
tab2 = Panel(child=gp_, title='4D')

f.write('line 398: ')
#f.write(str(desc))

# --------------------------------------------------------- Table work
'''
data = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        reports=[randint(0, 100) for i in range(10)],
    )
'''
d_ = get_drug_adverse_event_data(dname.value)

d_freq_list = [None] * 20 # top 20    
d_a_n = [None] * 20
for i in range(len(d_a_n)):
        d_freq_list[i]=get_event_freq(d_, d_[i]['term'])
        d_a_n[i] = d_[i]['term']
        #print d_a_n[i]
        #print d_freq_list[i]
        
#print 'd_freq_list: ' + str(d_freq_list)
data = dict(
        effects=d_a_n[0:20],
        probability=d_freq_list[0:20]
    )

source = ColumnDataSource(data)

columns = [
        TableColumn(field="effects", title="Effect"),
        TableColumn(field="probability", title="probability"),
    ]
data_table = DataTable(source=source, columns=columns, width=720, height=510)

dt = data_table.source

# ---------------------------------------------------------

p_ = Panel(child = data_table, title='20')

# -------------------------

new_df_ROR = pd.DataFrame
new_df_ROR = pd.read_csv('data/PRR.csv')
print new_df_ROR
    
source_ = ColumnDataSource(
        data=dict(
            x=new_df_ROR['m'],
            y=new_df_ROR['PRR'],
            drug=new_df_ROR['drugname'],
            event=new_df_ROR['eventname'],
            PRR=new_df_ROR['PRR']
        )
    )

hover = HoverTool(
        tooltips=[
            ("event", "@event"),            
            ("drug", "@drug"),
            ("PRR", "@PRR"),
            ("(x,y)", "($x, $y)"),
        ]
    )

#TOOLS = 'box_zoom,box_select,crosshair,resize,reset,hover'    

p2 = figure(width=720, height=510, toolbar_location='above', tools=[hover,BoxSelectTool(),'box_zoom','pan','crosshair','resize','reset'])

p2.xaxis.axis_label = 'Number of Events'
p2.yaxis.axis_label = 'PRR (Reporting Odds Ratio)'
#p2.scatter(new_df_ROR['m'],new_df_ROR['PRR'],color='#FB9A99')
#p2.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")
# Add hover to this comma-separated string and see what changes
#p2.circle(new_df_ROR['n'],new_df_ROR['PRR'], size=4, color="color", line_color=None, fill_alpha="alpha")
#p2.scatter(new_df_ROR['m'],new_df_ROR['PRR'],color='#FB9A99')

p2.circle('x', 'y', size=4, color='#FB9A99', source=source_)


# --------------------------

p_ROR = Panel(child = p2, title='ROR', width=100)


# --------------- ML portion -----------------
# See implementation in playground/experimentOct03 jupyter notebook
# Load data
ML_data = pd.read_csv('data/headache_.csv')
Model_data = pd.DataFrame(ML_data[['weight','age','sex','drug']])

# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(Model_data))

# Split test data vs. train data
size = (len(ML_data['serious'])*0.25)
x = Model_data.as_matrix()
Y = ML_data['serious'].as_matrix()

X_train = x[indices[:-size]]
Y_train = Y[indices[:-size]]
X_test  = x[indices[-size:]]
Y_test  = Y[indices[-size:]]

# Classify
neighK = KNeighborsClassifier(n_neighbors=35, algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, p=2, weights='uniform')
neighL = linear_model.LogisticRegression(C=1e5) #-- good ROC curve
neighR = RandomForestClassifier(n_estimators=30)

neighK.fit(X_train, Y_train)
neighL.fit(X_train, Y_train)
neighR.fit(X_train, Y_train)

resultsK = neighK.predict(X_test) - Y_test
resultsL = neighL.predict(X_test) - Y_test
resultsR = neighR.predict(X_test) - Y_test

false_pos_rateK, true_pos_rateK, num_pos, num_neg = make_roc_curve(neighK, X_test, Y_test,p_cut_list=np.arange(0., 1.01, 0.02))
false_pos_rateL, true_pos_rateL, num_pos, num_neg = make_roc_curve(neighL, X_test, Y_test,p_cut_list=np.arange(0., 1.01, 0.02))
false_pos_rateR, true_pos_rateR, num_pos, num_neg = make_roc_curve(neighR, X_test, Y_test,p_cut_list=np.arange(0., 1.01, 0.02))

# in case we want to generate the plot
'''
plt.figure(1, figsize=(7, 7))
plt.plot(false_pos_rateR, true_pos_rateR, '-',color='blue',label='Random Forest Classifier')
plt.plot(false_pos_rateL, true_pos_rateL, '-',color='green',label='Logistic Regression')
plt.plot(false_pos_rateK, true_pos_rateK, '-',color='red',label='K Nearest Neighbors (k=30)')
plt.plot(np.linspace(0,1,5),np.linspace(0,1,5),linestyle='--',color='grey')
plt.axes().set_aspect('equal')
plt.xlim(0,1)
plt.legend(loc='upper_left',prop={'size':12})
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
'''
mlplt = figure(toolbar_location=None)
mlplt.plot_height=305
mlplt.plot_width=305
mlplt.line(x=false_pos_rateR, y=true_pos_rateR,color='blue',legend='Random Forest Classifier')
mlplt.line(x=false_pos_rateL, y=true_pos_rateL,color='green',legend='Logistic Regression')
mlplt.line(x=false_pos_rateK, y=true_pos_rateK,color='red',legend='K Nearest Neighbors [k=30]')
mlplt.line(x=np.linspace(0,1,5),y=np.linspace(0,1,5),color='grey')
mlplt.xaxis.axis_label='False positive rate'
mlplt.yaxis.axis_label='True positive rate'
mlplt.legend.location='bottom_right'
#mlplt.legend.orientation='bottom_right'
#mlplt.title.text='predictive models ROC curves'
#mlplt.title.align='center'
mlplt.legend.label_width = 1
mlplt.legend.label_height = 1
mlplt.legend.label_text_font_size = '8pt'
mlplt.legend.legend_spacing = 1
mlplt.legend.legend_padding = 1
# Loading the plot


# --------------------------------------------

t_v="""Based on <i>18748</i> adverse effect reports submitted to the FDA involving <i>tylenol</i>, 
and given that you are <i>male</i>, <i>38</i> years old, weighing <i>75</i> kg, in the event of an adverse
reaction, the probability that such reaction is serious (e.g. resulting in hospitalization) is <b>71</b>%.
This prediction is based on machine learning logistic regression and classification algorithms using <a href="https://en.wikipedia.org/wiki/Scikit-learn">scikit</a>.
Overall predictive acuracy of the model is <i>72.5%</i>."""

ml_ = Div(text=t_v, width=580, height=100)

predict_button = Button(label='Predict')

wb = widgetbox([predict_button,ml_]) #l_ = layout([[desc],[ml_, predict_button]])

Mll = layout([[wb],[mlplt]])

p_ML = Panel(child = Mll, title='ML', width=100)

tabs = Tabs(tabs=[tab1,p_,tab2,p_ROR,p_ML],sizing_mode='fixed')

def select_reports(s=None):

    #print 'select_reports'
    global set_dname_val
    global set_indication_val
    ind_val    = indication.value
    #d_val      = dname.value
    begin_date = str(min_year.value)
    #print begin_date
    #print 's : ' + str(s)    
    #print indication.value
    #print dname.value
    
    if (ind_val != "ALL"):
    
        rr='&search=receivedate:['
        rr=rr+begin_date+'0101+TO+20160601]+AND+patient.drug.drugindication.exact:'+indication.value+'&count=receivedate'
        #print 'rr: '+rr
        # populate drug list with top 10 drugs
        ss='&search=receivedate:['
        ss=ss+begin_date+'0101+TO+20160601]+AND+patient.drug.drugindication.exact:'+indication.value+'&count=patient.drug.medicinalproduct'
        ss=master_string+ss
        #print 'ss: '+ss
        ssdf  = pd.DataFrame
        ssdf  = requests.get(ss).json()
        d_ = []
        print ss
        for i in ssdf['results']:
            d_.append(i['term'].upper())
        dname.options = d_[0:10]
        #print 'populated with :'
        #print dname.options
        #d_val = dname.options[0]
        #print 'dname.value '+dname.value
        #print 'dname.options[0] '+d_val
        
        print '************ : '
        print dname.value
        print dname.options[0]
        if (dname.value != dname.options[0]):
            #print 'they are diff'
            rr='&search=receivedate:['
            rr=rr+begin_date+'0101+TO+20160601]+AND+patient.drug.medicinalproduct.exact:'+dname.value+'&count=receivedate'    

    else:
        rr='&search=receivedate:['
        rr=rr+begin_date+'0101+TO+20160601]&count=receivedate'
    
    rr=master_string+rr
    #print rr
    selected = pd.DataFrame
    selected = requests.get(rr).json()
    print rr
    #print 'selected on: '+rr
    set_dname_val = dname.value
    f.write('select_reports: ')
    f.write(str(rr))
    f.write('select_reports: ')
    f.write(str(selected['results']))


    return selected
    #print selected['results']
    
'''
def select_movies():
    print 'select_movies()'
    genre_val = genre.value
    director_val = director.value.strip()
    cast_val = cast.value.strip()
    selected = movies[
        (movies.Reviews >= reviews.value) &
        (movies.BoxOffice >= (boxoffice.value * 1e6)) &
        (movies.Year >= min_year.value) &
        (movies.Year <= max_year.value) &
        (movies.Oscars >= oscars.value)
    ]
    if (genre_val != "All"):
        selected = selected[selected.Genre.str.contains(genre_val)==True]
    if (director_val != ""):
        selected = selected[selected.Director.str.contains(director_val)==True]
    if (cast_val != ""):
        selected = selected[selected.Cast.str.contains(cast_val)==True]
    print selected
    return selected
'''
def update_(s=None):
    global totalreports
    #global p1
    #print 'update_()'
    #print 
    #print indication.value
    #print 'calling select_reports'
    sr  = select_reports(s)
    #print sr['results']
    #print 'select_reports returned'
    cc = []
    dd = []
    print 's : ' + str(s)
    print 'dname.value : ' + str(dname.value)
    print 'dname.options[0] : ' + dname.options[0]
    for d in sr['results']:
        dd.append(d['time'][0:4]+'-'+d['time'][4:6]+'-'+d['time'][6:8])
        cc.append(d['count'])
    
    sourced.data = dict(
        x=datetime(dd),
        y=cc    
    )
    #print indication.value
    #print set_dname_val
    totalreports = len(dd)    
    if (s!=None):
        p1.title.text = "%s Reports of Adverse Effects (%s)"%(len(dd), s)
    else:
        p1.title.text = "%s Reports of Adverse Effects (%s)"%(len(dd), dname.options[0])
    
    #p1.line(datetime(dd), cc, color='#FB9A99')    
    #p1.grid.grid_line_alpha=0.3
    #p1.xaxis.axis_label = 'Date of Report'
    #p1.yaxis.axis_label = 'Number of Reports'
    print 'cc[0:10] : ' + str(cc[0:10])
    
    
    p2.xaxis.axis_label = 'Number of Events'
    p2.yaxis.axis_label = 'PRR (Reporting Odds Ratio)'
    #p1.scatter(datetime(dd), cc, color='#FB9A99')
    #p1.legend=genre.value
    #p1.legend.location = "top_left"
    #print 'end of update_()'
    #p2 = p1.copy()
    '''    
    p2.scatter(datetime(dd),cc,color='blue')
    p2.title.text = p1.title.text
    #p2.xaxis.axis_label = p1.xaxis.axis_label
    #p2.yaxis.axis_label = p1.yaxis.axis_label    
    p2.xaxis.axis_label = 'Number of Events'
    p2.yaxis.axis_label = 'PRR (Reporting Odds Ratio)'
    '''
'''
def update():
    df = select_movies()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d movies selected" % len(df)

    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        title=df["Title"],
        year=df["Year"],
        revenue=df["revenue"],
        alpha=df["alpha"]
    )
'''

def update_dname(attr, old, new):

    #print 'PREVIOUS ' + old
    print 'NEW ' + new
    update_(new)
    print 'returned update_(new)'
    #print 'Inside update_dname: dname.options[0:4] ' + str(dname.options[0:4])
    #gp_ = get_dots(dname.options[0:4])
    update_table(new)
    print 'returned update_table(new)'
    print 'calling ret_getROR'
    # this should really be under drug indication on change event
    
    # call first and only first time to generate csv file
    # ret_getROR = get_ROR(dname.options[0:10])
    up_tab()
    
    
def update_table(new):
    global data
    #global data_table  
    global d_
    #print 'inside update_table: ' + str(new)
    d_ = get_drug_adverse_event_data(new)

    d_freq_list = [None] * 20 # top 20    
    d_a_n = [None] * 20
    for i in range(len(d_a_n)):
            d_freq_list[i]=get_event_freq(d_, d_[i]['term'])
            d_a_n[i] = d_[i]['term']
            #print d_a_n[i]
            #print d_freq_list[i]
            
    #print 'd_freq_list: ' + str(d_freq_list)
    data['effects'] = d_a_n[0:20]
    data['probability'] = d_freq_list[0:20]
    
    new_data = dict()    
    new_data = dict(
        effects=d_a_n[0:20],
        probability=d_freq_list[0:20]
        )
    
    source.data = new_data
        
    
    '''
    source = ColumnDataSource(data)
    
    columns = [
            TableColumn(field="effects", title="Effect"),
            TableColumn(field="probability", title="probability"),
        ]
    data_table = DataTable(source=source, columns=columns, width=720, height=510)
    '''
    
print 'before setting controls ...'
print indication.value
print dname.value

controls = [indication, min_year, director, cast]
print 'Length of controls: ' + str(len(controls)) #, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update_())

dname.on_change('value', update_dname)
#data_table.on_change(, update_table)


def up_tab():
    
    print 'Predict button called me!'
    #gender = 
    if (radio_group.active == 0):
        gender = 'Male'
    else:
        if (radio_group.active == 1):
            gender = 'Female'
        else:
            gender = 'either gender'
    
    # making the prediction -- currently for tylenol
    print oscars.value
    print max_year.value
    print radio_group.active    
    datapoint = neighR.predict_proba([oscars.value,max_year.value,radio_group.active+1,6])[:,0][0]
    print datapoint
    s_datapoint = "{:.0%}".format(datapoint)
    new_text='Based on <i>'+str(totalreports)+'</i> adverse effect reports submitted to the FDA involving <i>'
    new_text=new_text+ dname.value + '</i>, and given that you are <i>'+gender+'</i>, <i>'+str(max_year.value)+'</i> years old, '
    new_text=new_text+'and weigh <i>'+str(oscars.value)+'</i> kg, in the event of an adverse reaction, the probability that'
    new_text=new_text+' such reaction is serious (e.g. resulting in hospitalization) is <b>'+str(s_datapoint)+'</b>. '
    new_text=new_text+'This prediction is based on machine learning random forest classifier</i> using <a href="https://en.wikipedia.org/wiki/Scikit-learn" target="_blank">scikit</a>. '
    new_text=new_text+'Overall predictive acuracy of the model is <i>72.5%</i>.'

    ml_.text = new_text
    
    
#tabs.on_click('active', up_tab)

predict_button.on_click(up_tab)

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
print 'inputs ...'
inputs = widgetbox([indication, dname, min_year, reviews, radio_group, max_year, oscars, director, cast])
l = layout([
    [desc],
    [inputs, tabs],
], sizing_mode=sizing_mode)

update_()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Drugs"