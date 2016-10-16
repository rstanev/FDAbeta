from os.path import dirname, join

import numpy as np
import pandas.io.sql as psql
import sqlite3 as sql

from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc
from bokeh.sampledata.movies_data import movie_path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
import itertools

conn = sql.connect(movie_path)
query = open(join(dirname(__file__), 'query.sql')).read()
movies = psql.read_sql(query, conn)

def datetime(x):
    return np.array(x, dtype=np.datetime64)

api_key        = 'TTwz56GNTpZnsL4il0pLIwz7F43QObZOQEssWlSU'
request_string = 'https://api.fda.gov/drug/event.json?api_key=' + api_key
master_string  = request_string
r='&search=receivedate:[20040101+TO+20160601]&count=receivedate'
r=master_string+r


#show(p1)

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

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

# Create Input controls
reviews = Slider(title="Minimum number of adverse reports:", value=80, start=10, end=300, step=10)
min_year = Slider(title="Starting report year", start=1940, end=2016, value=1990, step=1)
max_year = Slider(title="Age", start=1940, end=2014, value=2014, step=1)
oscars = Slider(title="Seriousness", start=0, end=4, value=0, step=1)
boxoffice = Slider(title="Number of adverse effect events:", start=0, end=800, value=0, step=1)
#checkbox_group = CheckboxGroup(
#        labels=["Male", "Female", "Both"], active=[0, 1])
radio_group = RadioGroup(
        labels=["Male", "Female", "Both"], active=2)

genre = Select(title="Drug indication:", value="ANXIETY",
               options=open(join(dirname(__file__), 'genres.txt')).read().split('\n'))

director = TextInput(title="Drug name:")
cast = TextInput(title="Adverse effect name:")
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Tomato Meter")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Number of Reviews")

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], title=[], year=[], revenue=[], alpha=[]))

hover = HoverTool(tooltips=[
    ("Title", "@title"),
    ("Year", "@year"),
    ("$", "@revenue")
])

p1 = figure(x_axis_type="datetime", width=700, height=600)

p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])
p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")

print 'before select_movies()'

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

def update_():
    print 'update_()'
    print genre.value
    rr='&search=receivedate:[20040101+TO+20160601]+AND+patient.drug.drugindication.exact:'+genre.value+'&count=receivedate'
    rr=master_string+rr
    #print rr
    
    All  = pd.DataFrame
    All  = requests.get(rr).json()
    #print All['results']
    
    cc = []
    dd = []
    
    for d in All['results']:
        dd.append(d['time'][0:4]+'-'+d['time'][4:6]+'-'+d['time'][6:8])
        cc.append(d['count'])
    
    p1.title.text = "%s Drug Reports of Adverse Effects (%s)"%(len(dd), genre.value) 
    p1.grid.grid_line_alpha=0.3
    p1.xaxis.axis_label = 'Date of Report'
    p1.yaxis.axis_label = 'Number of Reports'
    p1.line(datetime(dd), cc, color='#FB9A99')
    #p1.legend=genre.value
    #p1.legend.location = "top_left"
    print 'end of update_()'

def update():
    df = select_movies()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "%d movies selected" % len(df)

    px = pd.core.series.Series(dd)
    py = pd.core.series.Series(cc)

    source.data = dict(
        x=df[x_name],
        y=df[y_name],
        color=df["color"],
        title=df["Title"],
        year=df["Year"],
        revenue=df["revenue"],
        alpha=df["alpha"],
    )
print 'before setting controls'
controls = [reviews, boxoffice, genre, min_year, max_year, oscars, director, cast, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update_())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
print 'inputs'
inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [inputs, p1],
], sizing_mode=sizing_mode)

update_()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Movies"