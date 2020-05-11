# Best fit for Linear Regression

from statistics import mean
import numpy as np


def calculate_slope_intercept(xvalues,yvalues):
    m=(((mean(xvalues)*mean(yvalues))-mean(xvalues*yvalues))/((mean(xvalues)*mean(xvalues))-mean(xvalues*xvalues)))
    b=mean(yvalues)-m*mean(xvalues)
    return m,b

def linear_Regression():
    regression_line=[(m*x)+b for x in xvalues]
    return regression_line

def squared_error(ys_orig,ys_line):
    return sum((ys_line-ys_orig)*(ys_line-ys_orig))

def determination_coeff(ys_orig,ys_line):
    y_mean=[mean(ys_orig) for y in ys_orig]
    squared_error_regr=squared_error(ys_orig,ys_line)
    squared_error_y_mean=squared_error(ys_orig,y_mean)
    rsq=1-(squared_error_regr/squared_error_y_mean)
    return rsq

#driver function
xvalues=np.array([1,2,3,4,5],dtype=np.float64)
yvalues=np.array([5,4,6,5,6],dtype=np.float64)
print("X values",xvalues)
print("Y values",yvalues)

m,b=calculate_slope_intercept(xvalues,yvalues)
print("Slope : ",round(m,5),"Intercept : ",round(b,5))

regression_line=linear_Regression()
Rsq=determination_coeff(yvalues,regression_line)
print("R Squared Value : ",round(Rsq,3))
threshold=0.6
if Rsq<threshold:
    print("Acceptable Range")
else:
    print("Not Acceptable Range")