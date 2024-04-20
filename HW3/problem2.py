import numpy as np
import random
import math
from pprint import pprint

from tqdm import tqdm

import itertools

def psi(x_1, x_2):

    if x_1 != x_2:
        return 1
    else:
        return 0


def phi(x_i):

    return math.exp(x_i)


def get_cliques(A):
  cliques = {}
  count = 0
  for i in range(len(A)):
    for j in range(len(A)):
      if A[i][j] == 1 and (i+1,j+1) not in cliques.values() and (j+1,i+1) not in cliques.values() and i != j:
        cliques[count] = (i+1,j+1)
        count += 1
  return cliques

# this is to get a valid starting point
def get_valid_coloring(A, w):

    n = len(A)
    k = len(w)

    coloring = [0] * n

    i = 1
    while i <= n:
        coloring[i-1] = random.choice(w)
        #print("i =", i, "random choice =", coloring[i-1])
        colored = False
        iterations = 0
        while(not colored and iterations <= k*100):
            colored = True
            for j in range(1,n+1):
                if A[i-1][j-1] == 1 and coloring[j-1] != 0 and not psi(coloring[i-1], coloring[j-1]):
                    colored = False
                    coloring[i-1] = random.choice(w)
                    break
            #print("Intermediate coloring = ", coloring)
            iterations += 1

        if not colored:
            i = 1
            coloring = [0] * n
        else:
            i += 1

    return coloring

def get_joint_probability_distribution(samplesinfo, k):
  pr = {}
  n = len(list(samplesinfo.keys())[0])
  totalsamples = sum(samplesinfo.values())

  combinations = list(itertools.product(range(1,k+1), repeat=n))

  for i in combinations:
    if i in samplesinfo:
      pr[i] = samplesinfo[i]/totalsamples
    else:
      pr[i] = 0

  return pr

# take only the neighbors of i
# are you calculating things twice? yes, actually mutltiple times FIXED
def get_joint_probability(A, X, Z=None):

    prd = 1
    '''for i in range((len(A))):
        for j in range(len(A)):
            if i != j and A[i][j] != 0:
                prd = prd * psi(X[i], X[j])'''

    for i in get_cliques(A).values():
        #print(i)
        prd = prd * psi(X[i[0]-1], X[i[1]-1])


    for i in range(len(X)):
        prd = prd * phi(X[i])

    if Z is None:
        return prd
    else:
        return prd/Z


def gibbs(A, w, burnin, its, initialsample=None, roundrobin=True):

    n = len(A)
    k = len(w)

    samples = []
    if initialsample is not None:
       samples.append(initialsample)
    else:
        samples.append(get_valid_coloring(A,w))

    sampleindex = 1

    print("Initial sample =", samples)

    # initial valid sample to start from is in samples[0]
    node = 0
    for t in tqdm(range(burnin+its)):
        new_sample = samples[sampleindex-1].copy()
        #print("New sample = ", new_sample)

        if roundrobin:
            # round robin selection of vertices in the graph
            node = node % n
        else:
            # randomly selecting a node
            node = random.choice(list(range(n)))

        # find the conditional probability distribution table for the node given all other vertices
        # (The Transition probability for that state)
        pr = [0] * k

        denominator = 0

        # calculate the denominator
        for color in range(1,k+1):
            colorsetting = new_sample.copy()
            #print("original colorsetting = ", colorsetting, "Node index = ", node)
            colorsetting[node] = color

            denominator = denominator + get_joint_probability(A, colorsetting)
            #print("Denominator calculated = ", denominator, "for colorsetting = ", colorsetting)
            del colorsetting

        # calculate the conditional probability distribution by calculating the numerator
        for color in range(1,k+1):
            # set 'node' to color, colors of the other vertices are unchanged. now find the probability
            # numerator is the joint probability for the current color setting
            # denominator is the marginal probability of the vertices except 'node'. so sum pr(a,b,c,..node,...,n) over 'node'
            # no need to find the partition function since its cancelled in the numerator and denominator

            # getting the t-1 sample
            colorsetting = new_sample.copy()
            colorsetting[node] = color

            numerator = get_joint_probability(A, colorsetting)
            #if numerator == 0:
             #   pr[color-1] = 0
            #else:
            pr[color-1] = numerator/denominator
            #pr[color-1] = pr[color-1]

            del colorsetting

        #print("probability table =",pr)
        randnum = random.random()

        sum = 0
        new_color = 0
        #print("Random num =", randnum)
        for i in range(k):
            sum += pr[i]
            # checking which interval randum falls in
            if randnum <= sum:
                new_color = i+1
                break


        new_sample[node] = new_color
        #print("New color =", new_color, "for node =", node+1)


        #print("New Sample =",new_sample)

        samples.append(new_sample)
        node += 1
        sampleindex += 1



    # make sure to drop the burn in samples later!!
    #samples.append(new_sample)

    #pprint(samples)

    # drop the burnin samples
    samples = samples[burnin+1:]

    samplesarray = np.array(samples)
    return samplesarray.transpose()

from VertexColoringProblem import sumprod

def log_likelihood_gradient(A, w, samples, k):

  # data part
  m = len(samples[0])
  N = k * m

  dp = list()

  for i in range(k):
    n = np.count_nonzero(samples == i+1)
    #print("i = ", i, "count = ", n, "N = " ,N)
    dp.append(n/N)
   
  dp = np.array(dp)

  #return dp WORKS!

  # model part
  # uses belief propagation
  # MODIFIED sumprod will now return a vector for the probabilites of each color for all vertices
  z, mp = sumprod(A, w, 70)

  mp = np.array(mp)

  #print("dp =", dp)
  #print("mp=", mp)

  return z, dp - mp


def colormle(A, samples):
  # inputs are matrices A (n x n) and samples (n x m)
  # samples[i,t] corresponds to observed color for vertex i in the tth sample
  # output is a vector of weights w  of length k

  n = len(A)
  m = len(samples[0])

  # this is the vector of parameters that are to be learned from the samples and data model
  # k dimensional vector

  k = samples.max()

  print(k)

  # output vector
  w = np.ones(k)

  # step count
  T = 0.01

  # do gradient ascent
  gradient = w
  i = 1

  #for i in tqdm(range(10000)):
  while np.abs(gradient > 1e-5).any():
    z, gradient = log_likelihood_gradient(A, w, samples, k)
    w_new = w + (T * gradient)
    print("gradient= ", gradient, "w =", w, "w_new=", w_new)
    w = w_new
    

  return z, w

A = np.array([[0,1,0],
              [1,0,1],
              [0,1,0]])

w = [1,2,3]

burnin = 1000

itset = [10**3, 10**2, 10**3, 10**4, 10**5]
outset = []

for its in itset:
    samples = gibbs(A,w, burnin, its,initialsample=[3,2,3],roundrobin=True)
    z, learned_weights = colormle(A, samples)
    outset.append((z, learned_weights))

print("[================= RESULT =======================]")
for r in range(len(outset)):
    print("samples =", itset[r], "learned weights are = ", outset[r][1], "Partition function = ", outset[r][0])

print("==================================================")