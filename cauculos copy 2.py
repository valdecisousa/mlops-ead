# a vida de certo componente tem uma distribuição aproximandamente 
# exponencial com media de 1000 horas. 


 # C, qual o percentual de componentes que durar mais de 850 horas?

from math import exp

media = 1000
x = 1000

lam = 1 / media

prob = 1 - exp(-lam * x)

print(prob * 100)


from math import exp

lam = 1/1000
x = 850

prob = exp(-lam * x)

print(prob)
print(prob * 100)

from scipy.stats import norm

# Dados
media = 1.5       # anos
desvio = 0.3      # anos
garantia = 1      # 12 meses = 1 ano

# Cálculo do z-score
z = (garantia - media) / desvio

# Probabilidade de durar menos que 1 ano
probabilidade = norm.cdf(z)

# Resultado em porcentagem
percentual = probabilidade * 100

print(f"Z = {z:.2f}")
print(f"Percentual = {percentual:.2f}%")

from scipy.stats import expon
import math

# parâmetro lambda
lamb = 3

# tempo em minutos
t = 1

# Método 1 — usando scipy
probabilidade = expon.cdf(t, scale=1/lamb)

print("Usando scipy:")
print(f"P(T < 1) = {probabilidade:.4f}")

# Método 2 — usando a fórmula manual
prob_manual = 1 - math.exp(-lamb * t)

print("\nUsando fórmula manual:")
print(f"P(T < 1) = {prob_manual:.4f}")

# porcentagem
print(f"\nPercentual = {prob_manual*100:.2f}%")
