import numpy as np
import random
import math
from pprint import pprint

from tqdm import tqdm

import itertools
import datetime

def psi(x_1, x_2):

    if x_1 != x_2:
        return 1
    else: 
        return 0

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
    for t in tqdm(range(burnin+its+1)):
        new_sample = samples[sampleindex-1].copy()
        #print("New sample = ", new_sample)

        if roundrobin:
            # round robin selection of vertices in the graph
            node = node % n
        else:
            # randomly selecting a node
            node = random.choice(list(range(n)))
      
        # find the conditional probability distribution table for the node given all other vertices
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

        

    # make sure to drop the burn in samples later!!
    #samples.append(new_sample)

    #pprint(samples)
        
    # drop the burnin samples
    samples = samples[burnin:]

    samplesinfo = {}

    for i in tqdm(samples):
        if tuple(i) not in samplesinfo:
            samplesinfo[tuple(i)] = 1
        else:
            samplesinfo[tuple(i)] += 1

    print(samplesinfo)

    #return samplesinfo

    jpr = get_joint_probability_distribution(samplesinfo, k)


    marginal_prob = [[0 for i in range(k)] for j in range(n)]

    for node in range(n):
      for i in range(1,k+1):
        for assignment in jpr:
          if assignment[node] == i:
            marginal_prob[node][i-1] += round(jpr[assignment],3)

    return marginal_prob, samples


if __name__ == "__main__":

    ##########################################################
    #
    # Problem set 1 question 1
    #
    starttime = datetime.datetime.now()
    P1_1 = np.array([[0,1,0,0],
                  [1,0,1,0],
                  [0,1,0,1],
                  [1,0,1,0]
                  ])
    w = [1,2,3]

    burnin = 10000
    its = 1000000

    marginals = gibbs(P1_1,w,burnin,its)

    pprint(marginals)

    #########################################################3
    #
    # Problem set 1 question 2
    #

    P1_2 = np.array([[0,1,1,1],
                  [1,0,0,1],
                  [1,0,0,1],
                  [1,1,1,0]
                  ])

    w = [1,2,3,4]

    valset = [2**6, 2**10, 2**14, 2**18]

    burnin_vs_its = [[0 for i in range(len(valset))] for j in range(len(valset))]

    for i in tqdm(range(len(valset))): # for burnin
        for j in range(len(valset)): # for its
            burnin = valset[i]
            its = valset[j]
            marginal_prob = gibbs(P1_2,w,burnin,its,initialsample=[1,2,2,3])

            # get the prob for a = 4
            burnin_vs_its[i][j] = round(marginal_prob[0][3], 3)

    print("[=============== BURNIN VS ITS ================]")
    pprint(burnin_vs_its)
    print("===============================================")

    fintime = datetime.datetime.now()
    print("Exec time = ", fintime - starttime)

# Does your answer depend on the initial choice of assignment used in your Gibbs sampling algorithm?
# Yes, it depends on the initial assignment of colors. This is because the vertices in the given graph are highly correlated.
# Hence, flipping one variable at a time does not let the gibbs sampler travel to states farther from the initial state.
# Another reason is that the state space graph has lots of components that are disconnected. Hence traversal from one component to 
# another is restricted. Using Gibbs sampling for this problem does not cover all possible coloring assignments