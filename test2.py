import numpy as np 


right = distracted = 0
for i in range(100):
    draw = np.random.choice([["right"][0], ["Distracted"][0]], 1,p=[0.6, 0.4])
    if draw == "right":
        right+=1
    else:
        distracted+=1 
print(right, distracted)