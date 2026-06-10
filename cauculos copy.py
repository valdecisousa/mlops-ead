# a vida de certo componente tem uma distribuição aproximandamente 
# exponencial com media de 1000 horas. 

# B, qual a probabilidade de que os componentes durem entre 900 1200 horas?
from math import exp

lam = 1/1000

p = (1 - exp(-lam*1200)) - (1 - exp(-lam*900))

print(p)
print(p*100)


