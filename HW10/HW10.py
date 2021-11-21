# -*- coding: utf-8 -*-
"""RNN_GA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ERUi6q99y-TQjWL4eOdWPpPj0OkVk9_
"""

pip install pygad

import pygad
import numpy as np
import scipy

X= np.array([[1,0,0,0]]).T
Y = np.array([[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1]]).T

def softmax(x): 
    return np.exp(x)/sum(np.exp(x))

def crossEntropy(X,Y):
    return -np.sum(np.log(X)*Y)

def GA_RNN(solution, solution_idx):
    Wxh = solution[0:12].reshape(3,4)
    Whh = solution[12:21].reshape(3,3)
    Why = solution[21:33].reshape(4,3)
    bh = solution[33:36].reshape(3,1)
    by = solution[36:].reshape(4,1)
    hs = np.zeros((4,1))
    xs = np.zeros((4,4))
    h_s = np.zeros((3,1))
    ys = X
    # ps = np.zeros((4,1))

    for t in range(4):
        xs = ys[:,-1].reshape(4,1)# encode in 1-of-k representation
        
        h_t = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, h_s[:,-1].reshape(3,1) )+ bh) # hidden state
        h_s = np.hstack((h_s,h_t))    
        y_t = softmax(np.dot(Why, h_t) + by) # unnormalized log probabilities for next chars
        ys = np.hstack((ys,y_t))

    h_s = h_s[:,1:]
    ys = ys[:,1:]
    return 1/np.sum(crossEntropy(ys,Y))

fitness_function = GA_RNN
num_generations = 100
num_parents_mating = 10
sol_per_pop = 20
num_genes = 40

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type ="single_point"

mutation_type = "random"
mutation_percent_genes = 20


ga_instance = pygad.GA(
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

def RNN(solution, solution_idx,X,Y):
    Wxh = solution[0:12].reshape(3,4)
    Whh = solution[12:21].reshape(3,3)
    Why = solution[21:33].reshape(4,3)
    bh = solution[33:36].reshape(3,1)
    by = solution[36:].reshape(4,1)
    hs = np.zeros((4,1))
   
    Loss =0
    xs = np.zeros((4,4))
    h_s = np.zeros((3,1))
    ys = X
    # ps = np.zeros((4,1))

    for t in range(X.shape[0]):
        xs = ys[:,-1].reshape(4,1) # encode in 1-of-k representation
        
        h_t = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, h_s[:,-1].reshape(3,1) )+ bh) # hidden state
        h_s = np.hstack((h_s,h_t))    
        y_t = softmax(np.dot(Why, h_t) + by) # unnormalized log probabilities for next chars
        ys = np.hstack((ys,y_t))

    h_s = h_s[:,1:]
    ys = ys[:,1:]
    return h_s,ys,crossEntropy(ys,Y)

h,y,Loss =RNN(solution,solution,X,Y)
print("Ket qua thu duoc y = \n {} \n Ket qua sau khi lam tron 3 chu so y= \n {}. \n Loss= {} ".format(y,np.round(y,decimals=3),Loss))

Wxh = solution[0:12].reshape(3,4)
Whh = solution[12:21].reshape(3,3)
Why = solution[21:33].reshape(4,3)
bh = solution[33:36].reshape(3,1)
by = solution[36:].reshape(4,1)

print("Wxh:\n {} ".format(np.round(Wxh,decimals=3)))
print("Whh:\n {} ".format(np.round(Whh,decimals=3)))
print("Why:\n {} ".format(np.round(Why,decimals=3)))
print("bh:\n {} ".format(np.round(bh,decimals=3)))
print("by:\n {} ".format(np.round(by,decimals=3)))

data ='Helo'
# define a mapping of chars to integers
char_to_int =  {'H':0,'e':1,'l':2,"o":3}
int_to_char = {0:'H',1:'e',2:'l',3:"o"}
# integer encode input data
integer_encoded = [char_to_int[char] for char in data]
print(integer_encoded)
# # one hot encode
onehot_encoded = list()
for value in integer_encoded:
	letter = [0 for _ in range(len(data))]
	letter[value] = 1
	onehot_encoded.append(letter)
print(np.array(onehot_encoded).T)