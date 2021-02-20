import numpy as np 
x={2:2,4:4,1:1} 
print(sorted(x.keys(),key=x.get,reverse=True))
