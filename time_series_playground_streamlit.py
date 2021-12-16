#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from PIL import ImageTk, Image, ImageDraw
import PIL
#from tkinter import *
from tkinter import Canvas, Tk, Button, BOTH, YES
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import streamlit as st


# In[2]:

st.write("Hey!")

width = 365*3  # canvas width
height = 1000 # canvas height
center = height//2
white = (255, 255, 255) # canvas back

def save():
    # save image to hard drive
    filename = "user_input.jpg"
    output_image.save(filename)

lines_drawn = True
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black",width=5)
    draw.line([x1, y1, x2, y2],fill="black",width=5)

    #draw.line([100, 200, 500, 600], fill="black", width=5)

    global lines_drawn
    if not lines_drawn:
        canvas.create_line(365, 0, 365, height, fill="black", width=5)
        canvas.create_line(365*2, 0, 365*2, height, fill="black", width=5)
        lines_drawn = True

# In[3]:


master = Tk()

# create a tkinter canvas to draw on
canvas = Canvas(master, width=width, height=height, bg='white')
canvas.pack()

# create an empty PIL image and draw object to draw on
#output_image = PIL.Image.new("RGB", (width, height), white)
output_image = PIL.Image.new(mode="1", size=(width, height), color=1)
#output_image = PIL.Image.new(mode="RGB", size=(width, height), color="white")

#pixels = np.array(output_image)
#pixels[:, 360:390] = False
#output_image = PIL.Image.fromarray(pixels)

#output_image.show()

#array = np.ndarray(shape=(height, width))
#array[:, :] = True
#array[:, 100] = False

#output_image = PIL.Image.fromarray(array, mode="1")

#output_image = Image.new('RGBA', (400, 400), (0, 255, 0, 0))
draw = ImageDraw.Draw(output_image)


# draw.line((10,500, 10, 500),
#           fill=128,
#           width=3)

canvas.create_line(365, 0, 365, height, fill="black", width=5)
canvas.create_line(365*2, 0, 365*2, height, fill="black", width=5)

canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button=Button(text="save",command=save)
button.pack()


# In[4]:


master.mainloop()


# In[5]:


pixels = np.array(output_image)



# In[6]:


output_image


# In[7]:


output_image.show()


# In[8]:


start_date = "2019-01-01"
end_date = "2021-12-30"

max_value = 100
min_value = 0


# In[9]:


pixels.shape


# In[10]:


pixels_df = pd.DataFrame(pixels)


# In[11]:


pixels_white_rows_stripped = pixels_df[(pixels == False).any(axis=1)].reset_index(drop=True)
pixels_white_rows_stripped.index = pixels_white_rows_stripped.index[::-1]


# In[12]:


pixels_white_rows_stripped.shape


# In[13]:


if False:
    black_pixel_exists_in_column = (pixels_white_rows_stripped == False).any(axis=0)
    left_boundary = black_pixel_exists_in_column[black_pixel_exists_in_column == True].index[0]
    right_boundary = black_pixel_exists_in_column[black_pixel_exists_in_column == True].index[-1]
    left_boundary, right_boundary
    pixels_white_columns_stripped = pixels_white_rows_stripped.loc[:, left_boundary:right_boundary+1]
else:
    pixels_white_columns_stripped = pixels_white_rows_stripped


# In[14]:


im = PIL.Image.fromarray(pixels_white_columns_stripped.values)


# In[15]:


im.show()


# In[ ]:





# In[16]:


# if multiple values exist for an index, average them
time_series_data = pixels_white_columns_stripped.apply(lambda row: np.where(row == False, row.index, np.nan))


# In[17]:


time_series_data = time_series_data.mean(axis=0, skipna=True)


# In[18]:


time_series_data.head()


# In[19]:


date_range = pd.date_range(start=start_date, end=end_date, freq='D')


# In[20]:


len(date_range), len(time_series_data)


# In[21]:


time_series_data.index = date_range


# In[22]:


time_series_data


# In[23]:


time_series_data = time_series_data.reset_index()


# In[24]:


#time_series_data['value'].max()


# In[25]:


time_series_data = time_series_data.set_index('index')


# In[26]:


from sklearn.preprocessing import MinMaxScaler


# In[27]:


# minmax scaling to (min_value, max_value)
X_std = (time_series_data - time_series_data.min()) / (time_series_data.max() - time_series_data.min())
time_series_data_scaled = X_std * (max_value - min_value) + min_value


# In[ ]:





# In[28]:


time_series_data_scaled.max(), time_series_data_scaled.min()


# In[29]:


time_series_data_scaled = time_series_data_scaled.rename(columns={0: 'value'})


# In[30]:


time_series_data_scaled.reset_index().plot.scatter(x='index', y='value')


# In[31]:


time_series_data = time_series_data_scaled


# # Apply fbprophet

# In[32]:


if True:
    from fbprophet import Prophet


    # In[33]:


    prophet_data = time_series_data.reset_index().rename(columns={'index': 'ds', 'value': 'y'})


    # In[ ]:


    model = Prophet(changepoint_prior_scale=0.05,
                    weekly_seasonality=False,
                    yearly_seasonality=True,
                    #mcmc_samples=1000
                    )
    model.fit(prophet_data)


    # In[ ]:


    future = model.make_future_dataframe(periods=365)


    # In[ ]:


    forecast = model.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


    # In[ ]:


    fig1 = model.plot(forecast)

    st.pyplot(fig1)

    from fbprophet.plot import add_changepoints_to_plot

    a = add_changepoints_to_plot(fig1.gca(), model, forecast)

    plt.show()

    # In[ ]:


    fig2 = model.plot_components(forecast)
    plt.show()

    # In[ ]:


    from fbprophet.plot import plot_plotly, plot_components_plotly

    #plot_plotly(model, forecast)
    #plot_components_plotly(model, forecast)


    # In[ ]:





    # In[ ]:





# # Apply outlier detection

# In[ ]:


# prophet


# In[ ]:





# In[2]:


from sklearn.ensemble import IsolationForest


# In[3]:


isolation_forest_data = time_series_data.dropna()


# In[ ]:


isolation_forest = IsolationForest().fit(X=isolation_forest_data)


# In[ ]:


outliers_isolation_forest_all = pd.Series(isolation_forest.predict(isolation_forest_data), index=isolation_forest_data.index)
outliers_isolation_forest = outliers_isolation_forest_all[outliers_isolation_forest_all == -1]


quantiles = isolation_forest_data.quantile([0.1, 0.9])
outliers_isolation_forest_all = (isolation_forest_data['value'] < quantiles.loc[0.1].values[0]) | (isolation_forest_data['value'] > quantiles.loc[0.9].values[0])
outliers_isolation_forest = outliers_isolation_forest_all[outliers_isolation_forest_all == True]

# In[ ]:


ax = time_series_data_scaled.reset_index().plot.scatter(x='index', y='value')


# In[ ]:
outliers_isolation_forest = pd.DataFrame(outliers_isolation_forest)
outliers_isolation_forest['y'] = time_series_data_scaled.loc[outliers_isolation_forest.index, 'value']
outliers_isolation_forest.reset_index().plot.scatter(x='index', y='y', color='red', ax=ax)

plt.axhline(y=quantiles.loc[0.1].values[0], color='orange')
plt.axhline(y=quantiles.loc[0.9].values[0], color='orange')

#ax.plot(outliers_isolation_forest.index, outliers_isolation_forest)


plt.show()

# In[ ]:





# In[ ]:





# # Prognose

# In[ ]:


# prophet


# In[ ]:





# In[ ]:


# Gradienst boosted tree


# In[ ]:





# In[ ]:


# random forest


# In[ ]:





# In[ ]:


# lstm


# In[ ]:





# In[ ]:





# In[ ]:





# # Insights

# In[ ]:


# insights für mehrere Zeitreihen / Werte ergeben die andere


# In[ ]:




