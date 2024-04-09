import numpy as np
import random
from datetime import datetime
import math
import pprint

# Gibbs Sampling algorithm for the coloring problem
def gibbs(A, w, burnin, its):
  """
  Takes as input an n x n matrix A corresponding to the adjacency matrix of the graph G,
  a vector w representing the weights,
  burnin representing the number of burn-in samples,
  and an input that controls the number of iterations for the algorithm.

  Returns an n x k matrix of marginals whose ith, xith entry is equal to the
  probability that vertex i is colored with color xi

  """
  # Initialize the matrix of marginals
  marginal_matrix = []
  for i in range(len(A)):
    row_vector = [0] * len(w)
    marginal_matrix.append(row_vector)

  # Create an n x m k-ary matrix to hold samples (for Problem 2)
  samples_matrix = np.zeros((len(A),1))

  # Perform Gibbs sampling
  # Choose an initial assignment
  current_sample = find_valid_assignment(A, len(w))

  # Fix an ordering of the variables
  # We will just use numerical order

  for iteration in range(its):
    # Draw an index j uniformly at random (could also do round robin)
    # We will just go in order of the variables

    # Get the neighbors map for the given graph
    neighbors = get_neighbors(A)

    # Update each vertex one by one
    for vertex_j in range(len(current_sample)):
      # Create the vector to hold the conditional probability distribution
      # Each value in the vector represents x_j (a color)
      # The number for that value will be 0 if a neighbor of vertex_j has that color
      conditional_probability_distribution_vertex_j = [0] * len(w)
      for color in range(len(w)):
        # Check if the neighbors of vertex_j already have the color
        # If so, the conditional probability for that color assignment will be 0
        for neighbor in neighbors[vertex_j]:
          if current_sample[vertex_j] == color:
            conditional_probability_distribution_vertex_j[color] = 0

        # If the neighbors of vertex_j don't have that color
        conditional_probability_distribution_vertex_j[color] = np.exp(w[color])

      # Normalize the probability distribution vector
      vector_sum = sum(conditional_probability_distribution_vertex_j)
      for i in range(len(conditional_probability_distribution_vertex_j)):
        conditional_probability_distribution_vertex_j[i] /= vector_sum

      # Generate a random number in the interval [0,1)
      # Set the seed to the current time for a more random generation
      random.seed(datetime.now().timestamp())
      rand_num = random.random()

      # Use the univariate sampling scheme to sample the color z
      # First find the intervals
      intervals = compute_univariate_sampling_intervals(conditional_probability_distribution_vertex_j)
      sampled_color = 0
      for i in range(len(conditional_probability_distribution_vertex_j)):
        if rand_num >= intervals[i][0] and rand_num < intervals[i][1]:
          sampled_color = i
          break

      # Update vertex_j to be the sampled color
      current_sample[vertex_j] = sampled_color

    # Ignore the first burnin # of samples
    if iteration >= burnin:
      temp_sample = [x+1 for x in current_sample]

      # Add the sample to samples_matrix (for Problem 2)
      samples_matrix = np.insert(samples_matrix, len(samples_matrix[0]), temp_sample, axis=1)

      for vertex in range(len(current_sample)):
        marginal_matrix[vertex][current_sample[vertex]] += 1

  # Convert the marginal_matrix from counts into probabilities
  for vertex in range(len(marginal_matrix)):
    row_sum = sum(marginal_matrix[vertex])
    for color in range(len(marginal_matrix[vertex])):
      marginal_matrix[vertex][color] /= row_sum


  pprint.pprint(samples_matrix)

  return (np.array(marginal_matrix), np.array(samples_matrix)[:,1:])



# Find a valid assignment of colors for the graph problem through greedy selection
def find_valid_assignment(A, num_colors):
  # Get the degree for each vertex
  degrees = []
  for vertex in range(len(A)):
    degrees.append(sum(A[vertex]))

  # Assign the vertices all colors
  colors = []
  for vertex in range(len(A)):
    colors.append([i for i in range(num_colors)])

  # Sort the vertices based on degree
  sorted_vertices = []
  for i in range(len(degrees)):
    max_val = 0
    j = 0
    for j in range(len(degrees)):
      if j not in sorted_vertices and degrees[j] > max_val:
        max_val = degrees[j]
        ind = j
    sorted_vertices.append(ind)

  # Find the valid assignment of colors
  color_assignment = [-1] * len(A)
  for vertex in sorted_vertices:
    colors_list = colors[vertex]
    color_assignment[vertex] = colors_list[0]
    for neighbor in range(len(A[vertex])):
      if A[vertex][neighbor] == 1 and colors_list[0] in colors[neighbor]:
        colors[neighbor].remove(colors_list[0])

  return color_assignment




# Given the adjacency matrix representation of a graph,
# returns the neighbor of each vertex as a map
# with the key being the vertex and value being list of neighbors
def get_neighbors(A):
  neighbors = {}
  for vertex in range(len(A)):
    neighbor_list = []
    for node in range(len(A)):
      if A[vertex][node] == 1:
        neighbor_list.append(node)
    neighbors[vertex] = neighbor_list
  return neighbors


# Compute the intervals for the univariate sampling method
def compute_univariate_sampling_intervals(vector):
  intervals = []
  for i in range(len(vector)):
    if i == 0:
      intervals.append([0,vector[i]])
    elif i == len(vector) - 1:
      intervals.append([intervals[i-1][1],1])
    else:
      intervals.append([intervals[i-1][1], intervals[i-1][1] + vector[i]])
  return intervals


if __name__ == "__main__":

    starttime = datetime.now()
    A = np.array([[0,1,1,1],
                  [1,0,0,1],
                  [1,0,0,1],
                  [1,1,1,0]
                  ])
    

    '''A = np.array([[0,1,0,0],
                  [1,0,1,0],
                  [0,1,0,1],
                  [0,0,1,0]
                  ])'''

    w = [1,2,3,4]

    burnin = 0
    its = 50

    #print("initial sample =",get_valid_coloring(A,w))

    pprint.pprint(gibbs(A,w,burnin,its))
    fintime = datetime.now()

    print("Exec time =", fintime - starttime)