# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:54:33 2024

@author: Luis Lucas García y Alfonso de Lucas Iniesta
"""
import numpy as np
import matplotlib.pyplot as plt
import time
"""
En este programa vamos a resolver ecuaciones integrales numéricamente. Primero
resolveremos ecuaciones de Volterra por el método del rectángulo y del trapecio
y a continuación resolveremos ecuaciones de Fredholm por estos mismos métodos
"""
#Definimos las funciones para Volterra
def volterraRect(k, g, a, b, N):
    
    T = b - a
    x  = np.linspace(a, b, N)
    y = np.zeros(N)
    dt = T/N
    for i in range(N):
        y[i] = g(x[i])  + dt*np.dot(k(x[i], x), y)
    return y

def volterraTrap(k, g, a, b, N):
    
    T = b - a
    x  = np.linspace(a, b, N)
    y = np.zeros(N)
    dt = T/N
    for i in range(N):
        ks = k(x[i], x)
        y[i] = (g(x[i]) + 0.5*dt*(np.dot(ks, y) + np.dot(ks[1:], y[1:])))/(1-ks[i])
    return y
#Métodos para Fredholm
def fredholmRectHom(k, a, b, N):
    
    M = np.zeros((N, N))
    x = np.linspace(a, b, N)
    dt = (b-a)/N
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 1 - dt*k(x[i], x[j])
            else:
                M[i, j] = -dt*k(x[i], x[j])
    return np.linalg.eig(M)[1]

def fredholmTrapHom(k, a, b, N):
    
    M = np.zeros((N, N))
    x = np.linspace(a, b, N)
    dt = (b-a)/N
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i, j] = 1 - dt*k(x[i], x[j])
            elif i==0 or i==N-1:
                M[i, j] = -0.5*dt*k(x[i], x[j])
            else:
                M[i, j] = -dt*k(x[i], x[j])
    M[0, 0] = 1 - 0.5*dt*k(x[0], x[0])
    M[N-1, N-1] = 1 - 0.5*dt*k(x[N-1], x[N-1])
    return np.linalg.eig(M)[1]

def main(save):
    #Un ejemplo de Volterra
    N = 100
    a = 0
    b = 10
    x = np.linspace(a, b, N)
    g = lambda x: x**3
    k = lambda t, s: s-t
    yRect = volterraRect(k, g, a, b, N)
    yAnal = lambda t: 6*t - 6*np.sin(t)
    yTrap = volterraTrap(k, g, a, b, N)
    plt.figure(figsize=(16, 9))
    plt.grid()
    plt.title("$y(t) = t^3 + \\int_0^t (s-t) y(s) \\, ds$")
    plt.plot(x, yRect, label="Rectángulo")
    plt.plot(x, yTrap, label="Trapecio")
    plt.plot(x, yAnal(x), "--", label="Analítica", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    if save: plt.savefig("Imágenes/VolterraSol.png", dpi=200)
    #Estudiamos tiempos y errores
    errRect = []
    errTrap = []
    timesRect = []
    timesTrap = []
    for i in range(2, 10000, 100):
        x = np.linspace(a, b, i)
        t0 = time.time()
        yRect = volterraRect(k, g, a, b, i)
        tf = time.time()
        timesRect.append(tf-t0)
        t0 = time.time()
        yTrap = volterraTrap(k, g, a, b, i)
        tf = time.time()
        timesTrap.append(tf-t0)
        errRect.append(np.linalg.norm(yRect-yAnal(x)))
        errTrap.append(np.linalg.norm(yTrap-yAnal(x)))
    n = np.arange(2, 10000, 100)
    plt.figure(figsize=(14, 11))
    plt.subplot(2, 1, 1)
    plt.plot(n[:20], errRect[:20], label="Rectángulo")
    plt.plot(n[:20], errTrap[:20], label="Trapecio")
    plt.title("Errores de cada método")
    plt.legend()
    plt.grid()
    plt.xlabel("N pasos")
    plt.ylabel("Error")
    plt.subplot(2, 1, 2)
    plt.plot(n, timesRect)
    plt.plot(n, timesTrap)
    plt.ylabel("t (s)")
    plt.grid()
    if save: plt.savefig("Imágenes/VolterraErrs.png", dpi=200)
    #Hacemos un problema de Fredholm también
    L = 1
    a = 0
    T0 = 1
    lamb = 1
    w = 1
    N = 200
    x = np.linspace(a, L, N)
    
    def k(x, chi):
        if x <= chi:
            return lamb*w*x*(L-chi)/(T0*L)
        else:
            return lamb*w*(L-x)*chi/(T0*L)
    
    yRect = fredholmRectHom(k, a, L, N)
    yTrap = fredholmTrapHom(k, a, L, N)
    
    plt.figure(figsize=(16, 9))
    
    for i in range(2, 5):
        
        plt.plot(x, yRect[:,i])
        plt.plot(x, yTrap[:,i])
        
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    if save: plt.savefig("Armónicos Fredholm.png", dpi=200)

if __name__ == "__main__":
    
    plt.close("all")
    main(save=True)
    plt.show()