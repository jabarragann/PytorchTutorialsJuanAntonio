import numpy as np
import matplotlib.pyplot as plt

print("plotting example")
x = np.arange(1000)
y = np.sin(2*np.pi*x*0.001)

plt.plot(x,y)
plt.show()

