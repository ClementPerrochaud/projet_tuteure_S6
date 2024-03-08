from math import exp
import random


def operation(x,w,b,i):
    return sum([ w[i][j]*x[j] for j in range(len(x)) ]) + b[i]

def sigmoid(x):
    return 1/(1+exp(-x))


class NeuralNetwork:

    def __init__(self, sizes, activation_function=sigmoid, 
                 last_activation=True, coefficients=None, random_seed=0):

        self.sizes = sizes # forme du reseau de neurones
        self.len   = sum([ (sizes[i-1]+1)*sizes[i] for i in range(1,len(sizes)) ]) 
        # nombre de coefficients
        self.activ_f = activation_function # fonction d'activation
        self.last_activation = last_activation 
        # booleen exprimant si oui ou non la couche finale subit la fonction d'activation

        random.seed(random_seed)
        if coefficients == None: self.coefs = [ random.gauss() for _ in range(self.len) ]
        else:                    self.coefs = coefficients

        c = self.coefs.copy()
        self.layers = [[ # definitions des matrices w et vecteurs b
                [[c.pop(0) for _ in range(sizes[i-1])]  for _ in range(sizes[i])], # w
                [ c.pop(0)                              for _ in range(sizes[i])]  # b
            ] for i in range(1,len(sizes))]
    
    def __call__(self, x0): # list -> list !!
        x = x0.copy()
        for i,(w,b) in enumerate(self.layers): # passage de couche en couche
            if not self.last_activation and i == len(self.layers)-1: # derniere couche
                  x = [              operation(x,w,b,j)   for j in range(self.sizes[i+1]) ]
            else: x = [ self.activ_f(operation(x,w,b,j))  for j in range(self.sizes[i+1]) ]
        return x


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    plt.rc('font', size=7)

    # differentes fonctions d'activation
    from math import tanh, log
    def H(x): return 1        if x>0 else 0
    def Lo(x): return log(1+x) if x>0 else -log(1-x)

    # definition des reseaux de neurone
    NN00 = NeuralNetwork((1,2,2,1)  )
    NN01 = NeuralNetwork((1,4,4,1)  )
    NN02 = NeuralNetwork((1,5,5,5,1))
    NN10 = NeuralNetwork((1,2,2,1),   activation_function=tanh)
    NN11 = NeuralNetwork((1,4,4,1),   activation_function=tanh)
    NN12 = NeuralNetwork((1,5,5,5,1), activation_function=tanh)
    NN20 = NeuralNetwork((1,2,2,1),   activation_function=H)
    NN21 = NeuralNetwork((1,4,4,1),   activation_function=H)
    NN22 = NeuralNetwork((1,5,5,5,1), activation_function=H)
    NN30 = NeuralNetwork((1,2,2,1),   activation_function=Lo)
    NN31 = NeuralNetwork((1,4,4,1),   activation_function=Lo)
    NN32 = NeuralNetwork((1,5,5,5,1), activation_function=Lo)

    a,b,n = -8,8,1000 # bornes du graphe
    X = [a+(b-a)*i/n for i in range(n)]

    fig, ax = plt.subplots(2,2)
    fig.suptitle("Réseaux de neurones (1,...,1),\nici à coefficents aléatoirs (les mêmes pour chaque graphes)\navec différentes fonctions d'activation.")

    ax[0][0].plot(X, [NN00([x])[0] for x in X], linewidth=3, label="(1,2,2,1)"  )
    ax[0][0].plot(X, [NN01([x])[0] for x in X], linewidth=3, label="(1,4,4,1)"  )
    ax[0][0].plot(X, [NN02([x])[0] for x in X], linewidth=3, label="(1,5,5,5,1)")
    ax[0][0].set_title("ϕ sigmoïde")

    ax[0][1].plot(X, [NN10([x])[0] for x in X], linewidth=3, label="(1,2,2,1)"  )
    ax[0][1].plot(X, [NN11([x])[0] for x in X], linewidth=3, label="(1,4,4,1)"  )
    ax[0][1].plot(X, [NN12([x])[0] for x in X], linewidth=3, label="(1,5,5,5,1)")
    ax[0][1].set_title("ϕ hyperbolique")
    
    ax[1][0].plot(X, [NN20([x])[0] for x in X], linewidth=3, label="(1,2,2,1)",   zorder=2)
    ax[1][0].plot(X, [NN21([x])[0] for x in X], linewidth=3, label="(1,4,4,1)",   zorder=1)
    ax[1][0].plot(X, [NN22([x])[0] for x in X], linewidth=3, label="(1,5,5,5,1)", zorder=0)
    ax[1][0].set_title("ϕ Heaviside")

    ax[1][1].plot(X, [NN30([x])[0] for x in X], linewidth=3, label="(1,2,2,1)"  )
    ax[1][1].plot(X, [NN31([x])[0] for x in X], linewidth=3, label="(1,4,4,1)"  )
    ax[1][1].plot(X, [NN32([x])[0] for x in X], linewidth=3, label="(1,5,5,5,1)")
    ax[1][1].set_title("ϕ logarithmique")

    for i in range(2):
        for j in range(2):
            ax[i][j].grid(True)
            ax[i][j].set_xlabel("[x]")
            ax[i][j].set_ylabel("NN([x])")
            ax[i][j].legend()

    plt.show()
