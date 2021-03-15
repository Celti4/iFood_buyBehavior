# Function to determinated the next purchase date
import numpy as np 
#giving Last day of buy we want to predict the next day based on the probabilities
def MarkovNextPurchaseDate(lastDay, dataFrame): 
    probVector = np.array(dataFrame[dataFrame.index == lastDay])[0]
    day = np.random.multinomial(1, probVector, size=1)
    dayOfWeek = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    indexDay = np.where(day == 1)[1]
    dayPredict = dayOfWeek[indexDay][0]
    return dayPredict



def randomDay(): 
    probVector = np.array([1/7,1/7,1/7,1/7,1/7,1/7,1/7])
    day = np.random.multinomial(1, probVector, size=1)
    dayOfWeek = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    indexDay = np.where(day == 1)[1]
    randomDayPredict = dayOfWeek[indexDay][0]
    return randomDayPredict
