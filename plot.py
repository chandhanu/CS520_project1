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
x = list(map(int, xAxis))
plt.xticks(x[::5],  rotation='vertical')
plt.plot(x,yAxis, linestyle = 'solid', color='black',label="Agent 1")
plt.xlabel('ghost')
plt.ylabel('survivability')

dictionary = json.load(open('Newagent2.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
#print(dictionary["51"]["data"]["0"])
## LINE GRAPH ##
x = list(map(int, xAxis))
plt.xticks(x[::5],  rotation='vertical')
plt.plot(np.array(xAxis),np.array(yAxis), color='blue',label="Agent 2")
plt.xlabel('ghost')
plt.ylabel('survivability')


dictionary = json.load(open('New3.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
x = list(map(int, xAxis))
plt.xticks(x[::5],  rotation='vertical')
plt.plot(np.array(xAxis),np.array(yAxis), color='green',label="Agent 3")
plt.xlabel('ghost')
plt.ylabel('survivability')

#plt.show()


#dictionary = json.load(open('Newagent4.json', 'r'))
dictionary = json.load(open('New4.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
x = list(map(int, xAxis))
plt.xticks(x[::5],  rotation='vertical')
plt.plot(np.array(xAxis),np.array(yAxis), color='red',label="Agent 4")
plt.xlabel('ghost')
plt.ylabel('survivability')



dictionary = json.load(open('New5_new.json', 'r'))
xAxis = [key for key, value in dictionary["51"]["data"].items()]
yAxis = [value["success_rate"] for key, value in dictionary["51"]["data"].items()]
plt.grid(True)
x = list(map(int, xAxis))
plt.xticks(x[::5],  rotation='vertical')
plt.plot(np.array(xAxis),np.array(yAxis), color='purple', label="Agent 5")
plt.xlabel('ghost')
plt.ylabel('survivability')
plt.legend(loc="upper right")
plt.show()