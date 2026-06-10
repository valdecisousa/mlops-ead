# a vida de certo componente tem uma distribuição aproximandamente 
# exponencial com media de 1000 horas. 


# A, determinar a porcentagem de componentes que queimarão antes de 1000 horas

from math import exp

media = 1000
x = 1000

lam = 1 / media

prob = 1 - exp(-lam * x)

print(prob * 100)



