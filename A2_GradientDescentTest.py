import time

def coefs_dx(coefs, i, dx):
        new_coefs = coefs.copy()
        new_coefs[i] = coefs[i]+dx
        return new_coefs

def Loss(X,Y,model,coefs):
    # en supposant que le model utilisé soit de la forme f(x,coefs)
    return sum([ (model(x,coefs) - y)**2 for x,y in zip(X,Y)])/len(X)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~


def basic(X, Y, model, coefs, Loss=Loss,
          gamma=1, dx=1E-5, iterations=1000):
    
    t0 = time.monotonic_ns()
    losses = []
    times  = []
    
    coefs = coefs.copy()
    for _ in range(iterations):
        loss0 = Loss(X,Y,model,coefs)
        grad  = [ Loss(X, Y, model, coefs_dx(coefs, i, dx)) - loss0  for  i          in range(len(coefs))]
        coefs = [ coef - gamma*grad                                  for (coef,grad) in zip(coefs,grad)  ]

        losses.append(loss0)
        times.append(time.monotonic_ns()-t0)
    
    return coefs, losses, times


def momentum(X, Y, model, coefs, Loss=Loss,
             gamma=1, beta=0.99, dx=1E-5, iterations=1000):
    
    t0 = time.monotonic_ns()
    losses = []
    times  = []

    coefs = coefs.copy()
    v = [0]*len(coefs)
    for _ in range(iterations):
        loss0 = Loss(X,Y,model,coefs)
        grad  = [ Loss(X, Y, model, coefs_dx(coefs, i, dx)) - loss0  for  i          in range(len(coefs))]
        v     = [ beta*v + (1-beta)*grad                             for (v,grad)    in zip(v,grad)      ]
        coefs = [ coef - gamma*grad                                  for (coef,grad) in zip(coefs,grad)  ]

        losses.append(loss0)
        times.append(time.monotonic_ns()-t0)
    
    return coefs, losses, times


def AdaGrad(X, Y, model, coefs, Loss=Loss,
            gamma=1, eps=1E-5, dx=1E-5, iterations=1000):
    
    t0 = time.monotonic_ns()
    losses = []
    times  = []

    coefs = coefs.copy()
    g = [0]*len(coefs)
    for _ in range(iterations):
        loss0 = Loss(X,Y,model,coefs)
        grad  = [ Loss(X, Y, model, coefs_dx(coefs, i, dx)) - loss0  for  i            in range(len(coefs))]
        g     = [ g + grad**2                                        for (g,grad)      in zip(g,grad)      ]
        coefs = [ coef - gamma/(g**(1/2)+eps)*grad                   for (g,coef,grad) in zip(g,coefs,grad)]

        losses.append(loss0)
        times.append(time.monotonic_ns()-t0)
    
    return coefs, losses, times


def RMSprop(X, Y, model, coefs, Loss=Loss,
            gamma=1, beta=0.97, eps=1E-5, dx=1E-5, iterations=1000):
    
    t0 = time.monotonic_ns()
    losses = []
    times  = []

    coefs = coefs.copy()
    v = [0]*len(coefs)
    for _ in range(iterations):
        loss0 = Loss(X,Y,model,coefs)
        grad  = [ Loss(X, Y, model, coefs_dx(coefs, i, dx)) - loss0  for  i            in range(len(coefs))]
        v     = [ beta*v + (1-beta)*grad**2                          for (v,grad)      in zip(v,grad)      ]
        coefs = [ coef - gamma/(v**(1/2)+eps)*grad                   for (v,coef,grad) in zip(v,coefs,grad)]

        losses.append(loss0)
        times.append(time.monotonic_ns()-t0)
    
    return coefs, losses, times


def Adam(X, Y, model, coefs, Loss=Loss,
         gamma=1, beta1=0.96, beta2=0.96, eps=1E-5, dx=1E-5, iterations=1000):
    
    t0 = time.monotonic_ns()
    losses = []
    times  = []

    def m_hat(m,n): return m/(1-beta1**n)
    def v_hat(v,n): return v/(1-beta2**n)

    coefs = coefs.copy()
    m = [0]*len(coefs)
    v = [0]*len(coefs)
    for n in range(1,iterations+1):
        loss0 = Loss(X,Y,model,coefs)
        grad  = [ Loss(X, Y, model, coefs_dx(coefs, i, dx)) - loss0  for  i         in range(len(coefs))]
        m     = [ beta1*m + (1-beta1)*grad                           for (m,grad)   in zip(m,grad)      ]
        v     = [ beta2*v + (1-beta2)*grad**2                        for (v,grad)   in zip(v,grad)      ]
        coefs = [ coef - gamma*m_hat(m,n)/(v_hat(v,n)**(1/2)+eps)    for (m,v,coef) in zip(m,v,coefs)   ]

        losses.append(loss0)
        times.append(time.monotonic_ns()-t0)
    
    return coefs, losses, times


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from math import log10
    import random
    random.seed(0)

    def f(x, coefs):
        return sum([c*x**n for n,c in enumerate(coefs)]) # modele : polynome 
    # Ici l'on pourait remplacer ce model par notre réseau de neurones.

    degree = 8  # NOMBRE DE PARAMETRES                <--- paramètre important !
    b,n = 1.5,15 # borne et nombre de points d'étude

    coefs_ref = [random.gauss() for _ in range(degree)]
    Xref = [b*(2*i/n-1) for i in range(n)]
    Yref = [f(x, coefs_ref) for x in Xref]

    abscissa = "log_step"  # "step", "log_step", "time" ou "log_time"
    
    N = 100000 # nombre d'ittérations 

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    X = list(range(N))
    log_X = [log10(1+x) for x in X]

    coefs = [random.gauss() for _ in range(degree)]

    print("Les calculs peuvent prendre un certain temps...")
    basic_coefs,    basic_losses,    basic_times    =    basic(Xref, Yref, f, coefs, Loss, gamma=0.1, iterations=N); print("1/5 Basic:    done")
    momentum_coefs, momentum_losses, momentum_times = momentum(Xref, Yref, f, coefs, Loss, gamma=0.1, iterations=N); print("2/5 Momentum: done")
    AdaGrad_coefs,  AdaGrad_losses,  AdaGrad_times  =  AdaGrad(Xref, Yref, f, coefs, Loss, gamma=0.1, iterations=N); print("3/5 AdaGrad:  done")
    RMSprop_coefs,  RMSprop_losses,  RMSprop_times  =  RMSprop(Xref, Yref, f, coefs, Loss, gamma=0.1, iterations=N); print("4/5 RMSprop:  done")
    Adam_coefs,     Adam_losses,     Adam_times     =     Adam(Xref, Yref, f, coefs, Loss, gamma=0.1, iterations=N); print("5/5 Adam:     done")

    basic_times    = [t*1E-9 for t in basic_times   ] # conversion en secondes
    momentum_times = [t*1E-9 for t in momentum_times]
    AdaGrad_times  = [t*1E-9 for t in AdaGrad_times ]
    RMSprop_times  = [t*1E-9 for t in RMSprop_times ]
    Adam_times     = [t*1E-9 for t in Adam_times    ]

    log_basic_losses    = [log10(x) for x in basic_losses   ]
    log_momentum_losses = [log10(x) for x in momentum_losses]
    log_AdaGrad_losses  = [log10(x) for x in AdaGrad_losses ]
    log_RMSprop_losses  = [log10(x) for x in RMSprop_losses ]
    log_Adam_losses     = [log10(x) for x in Adam_losses    ]

    eps = 1E-3
    log_basic_times    = [log10(eps+t) for t in basic_times   ]
    log_momentum_times = [log10(eps+t) for t in momentum_times]
    log_AdaGrad_times  = [log10(eps+t) for t in AdaGrad_times ]
    log_RMSprop_times  = [log10(eps+t) for t in RMSprop_times ]
    log_Adam_times     = [log10(eps+t) for t in Adam_times    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(f"Évolution de recherche de la Loss pour un polynome de degré N={degree}\npour différents algorithmes de descente de gradient.")
    if abscissa == "log_time":
        ax.plot(log_basic_times,    log_basic_losses,    label="basique" )
        ax.plot(log_momentum_times, log_momentum_losses, label="momentum")
        ax.plot(log_AdaGrad_times,  log_AdaGrad_losses,  label="AdaGrad" )
        ax.plot(log_RMSprop_times,  log_RMSprop_losses,  label="RMSprop" )
        ax.plot(log_Adam_times,     log_Adam_losses,     label="Adam"    )
        ax.set_xlabel("log(temps de calcul (s))")
    if abscissa == "time":
        ax.plot(basic_times,    log_basic_losses,    label="basique" )
        ax.plot(momentum_times, log_momentum_losses, label="momentum")
        ax.plot(AdaGrad_times,  log_AdaGrad_losses,  label="AdaGrad" )
        ax.plot(RMSprop_times,  log_RMSprop_losses,  label="RMSprop" )
        ax.plot(Adam_times,     log_Adam_losses,     label="Adam"    )
        ax.set_xlabel("temps de calcul (s)")
    elif abscissa == "log_step":
        ax.plot(log_X, log_basic_losses,    label="basique" )
        ax.plot(log_X, log_momentum_losses, label="momentum")
        ax.plot(log_X, log_AdaGrad_losses,  label="AdaGrad" )
        ax.plot(log_X, log_RMSprop_losses,  label="RMSprop" )
        ax.plot(log_X, log_Adam_losses,     label="Adam"    )
        ax.set_xlabel("log(nombre d'itération)") 
    elif abscissa == "step":
        ax.plot(X, log_basic_losses,    label="basique" )
        ax.plot(X, log_momentum_losses, label="momentum")
        ax.plot(X, log_AdaGrad_losses,  label="AdaGrad" )
        ax.plot(X, log_RMSprop_losses,  label="RMSprop" )
        ax.plot(X, log_Adam_losses,     label="Adam"    )
        ax.set_xlabel("nombre d'itération")
    ax.grid(True)
    ax.set_ylabel("log(Loss(Θi))")
    ax.legend()

    plt.show()
