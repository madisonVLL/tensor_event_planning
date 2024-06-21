'''
'''
from typing import *
import tensorflow as tf
import numpy as np
import requests
import justext

#we will combine these together to train the algorthim, we will test the algorithm
#with hours and minutes
time_triggers:list[str] = np.array(["Day", "Month", "Year", "AM", "PM"])
time_trig_alt: list[str] = np.array(["Days", "Months", "Years", "am", "pm"])

time_learning = tf.keras.layers.Dense(units = 1, input_shape = [1])
time_model = tf.keras.Sequential([time_learning])
time_model.compile(loss="error", optimizer=tf.keras.optimizers.Adam(0.1))
#this model can have a lot of error, just need to generally relate uppercase
#to lowercase, singulars to plurals, etc

learning_mode = time_model(time_triggers, time_trig_alt, epoch = 500, Verbose = False)
#epoch is the amount of trials to learn the model

time_model.predict(["Hour"])


date_values:list[int] = np.array([24, 24 * 30, 24 * 30 * 365, 12, 12])

sug_date_task : dict[str, list[str]] = {}

wedding_sites: list[str] = [
    "https://www.brides.com/story/brides-wedding-checklist-custom-wedding-to-do-list", 
    "https://www.reddit.com/r/weddingplanning/comments/1b3uxfv/wedding_planning_timeline_help/",
    "https://www.vaughnbarry.com/blog/wedding-planning-checklist",
    "https://www.theknot.com/content/12-month-wedding-planning-countdown",
    "https://www.minted.com/gifts/wedding-planning-checklist"
]

trial_site = "https://www.minted.com/gifts/wedding-planning-checklist"

response = requests.get(trial_site)
paragraphs = iter(justext.justext(response.content, justext.get_stoplist("English")))

for paragraph in paragraphs:
    if not paragraph.is_boilerplate:
        #checks to if there is a current word which is a key word
        #this isn't correct, we want to select the whole paragraph
        read_paragraph = list(filter(lambda key_word: key_word in paragraph.text, date_triggers))
        if len(read_paragraph) > 0:
            #checking to see what words got extracted
            print(read_paragraph)
           

'''

for site in wedding_sites:
    response = requests.get(site)
    paragraphs = iter(justext.justext(response.content, justext.get_stoplist("English")))

    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            if "Out" in paragraph.text:
                if paragraph.text[-3:] != "Out" and len(list(filter(lambda key_word: key_word in paragraph.text, date_triggers))) != 0:
                    date_key = paragraph.text.split("Out")[0][0:-2]
                else:
                    date_key = paragraph.text[0:-4]
                    
                if paragraph.text in sug_date_task.keys():
                    sug_date_task[date_key] += [next(paragraphs).text]
                else:
                    sug_date_task[date_key] = [next(paragraphs).text]
'''
print(sug_date_task.values())




