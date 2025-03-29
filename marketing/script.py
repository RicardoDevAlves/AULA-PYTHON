print("Rode este script com python script.py dentro do terminal, na pasta do projeto.")

# Tipos de dados
numero_inteiro = 10
numero_decimal = 3.14
texto = 'Olá, Mundo!'
booleano = True

print(numero_inteiro, numero_decimal, texto, booleano)

# em Javascript console.log('${}')
print(f"Inteiro: {numero_inteiro}, Decimal: {numero_decimal}")
print(f"Texto: {texto}")

# Uso do ; se iremos executar mais de uma instrução na mesma linha
#serve para identificar que uma instrução finalizou para que possa determinar que o que virá em seguida é uma nova instrução.
# Aqui separa as instruções
print("Olá mundo"); print("Olá mundo 2")

# Estruturas de Dados
lista = [1, 2, 3, 3, 4]
tupla = (5, 6, 7)
dicionario = {'nome': 'João', 'idade': 25}
conjunto = {1, 2, 3, 3, 4}

print(f'lista: {lista}')
print(f'tupla: {tupla}')
print(f'dicionario: {dicionario}')
print(f'conjunto: {conjunto}')

# Estruturas de Controle
for i in lista:
    print(f'Número (lista): {i}')
for i in conjunto:
    print(f'Número (conjunto): {i}')
if numero_inteiro > 5:
    print('O número é maior que 5')

#Estrutura de repetição com índice.
for indice, valor in enumerate(lista):
    print(f'Índice: {indice}, Valor: {valor}')

lista[3] = 5
for indice, valor in enumerate(lista):
    print(f'Índice: {indice}, Valor: {valor}')

# soma dos números de 1 a 10
soma = 0
for i in range(1, 11):
    print(i)
    soma += i
print(f'Soma: {soma}')

soma = 0
soma = soma + 1
soma = soma + 2
soma = soma + 3
soma = soma + 10

for i in range(1, 11):
    soma = soma + i

print(soma)

# Funções
def saudacao(nome):
    return f'Olá, {nome}!'

print(saudacao('Aluno'))

#Bibliotecas Essenciais para Machine Learning
# Seção 2: Bibliotecas Essenciais para Machine Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Numpy
array = np.array([1, 2, 3, 4, 5])
print('Array Numpy:', array)

# Pandas
df = pd.DataFrame({'Nome': ['Ana', 'Bruno', 'Carlos'], 'Idade': [23, 35, 19]})
print('DataFrame Pandas:\n', df)

# Matplotlib
plt.plot(array, array**2)
plt.xlabel('X')
plt.ylabel('X ao quadrado')
plt.title('Gráfico Exemplo')
plt.show()

# prompt: Crie exercícios de algoritmos básicos de python, sem necessidade de bibliotecas

# Exercício 1: Fatorial de um número

# Exercício 2: Verificar se um número é primo

# Exercício 3: Encontrar o maior elemento em uma lista