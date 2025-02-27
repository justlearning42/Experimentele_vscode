import classes
import General.functies as functies
import sympy as sp
from IPython.display import display

a,b,c = sp.symbols('a b c')
equation = (a-b)/c
verg = classes.vergelijking(equation)
verg1 = 2 + verg
verg2 = a ** verg
verg3 = sp.sin(verg)
























































































"""
Dit is code voor POWW, niet verwijderen pls

matrix = [[9,2,7,None],
          [6,4,3,7],
          [5,None,1,8],
          [7,6,9,4]]
min = 500
ind = 0
cykels = [[1,2,3,4],
          [1,2,4,3],
          [1,3,4,2],
          [1,4,3,2],
          [2,1,3,4],
          [2,1,4,3],
          [2,3,1,4],
          [2,3,4,1],
          [2,4,1,3],
          [2,4,3,1],
          [3,1,4,2],
          [3,2,1,4],
          [3,2,4,1],
          [3,4,1,2]
          ]
termen = []
for i in range(len(cykels)):
    term = 0
    for k in range(4):
        term += matrix[k][cykels[i][k]-1]
    termen.append(term)
    if term < min:
        ind = i
        min = term

output = ""
for i in range(len(cykels)):
    to_add = "("
    for k in range(3):
        to_add = to_add + str(cykels[i][k]) + " \hspace{0.5em} "
    to_add = to_add + str(cykels[i][3]) + ") &\longrightarrow " + str(termen[i]) + "\\\\ \n"
    output += to_add

print(output)
"""
