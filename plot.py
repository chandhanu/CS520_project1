from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

dictionary = json.load(open('Newagent1.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
#print(dictionary["51"]["data"]["0"])
## LINE GRAPH ##
x = np.array(xAxis)
y = np.array(yAxis)
X_Y_Spline = make_interp_spline(x, y)
#X_ = np.linspace(x.min(),x.max() , 100)
#Y_ = X_Y_Spline(X_)
#plt.plot(X_, Y_,color='blue', marker='o')
plt.plot(np.array(xAxis),np.array(yAxis), linestyle = 'solid', color='black', marker='o')
plt.xlabel('ghost')
plt.ylabel('survivability')

dictionary = json.load(open('Newagent2.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
#print(dictionary["51"]["data"]["0"])
## LINE GRAPH ##
plt.plot(np.array(xAxis),np.array(yAxis), color='blue', marker='o')
plt.xlabel('ghost')
plt.ylabel('survivability')

dictionary = json.load(open('New3.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
#print(dictionary["51"]["data"]["0"])
## LINE GRAPH ##
plt.plot(np.array(xAxis),np.array(yAxis), color='green', marker='o')
plt.xlabel('ghost')
plt.ylabel('survivability')

#dictionary = json.load(open('Newagent4.json', 'r'))
dictionary = json.load(open('New2agent4.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
#print(dictionary["51"]["data"]["0"])
## LINE GRAPH ##
plt.plot(np.array(xAxis),np.array(yAxis), color='red', marker='o')
plt.xlabel('ghost')
plt.ylabel('survivability')


plt.show()