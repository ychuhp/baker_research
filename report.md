# Workflow

* PARSER -> CA BACKBONE & CA SEQUENCE 
* CA BACKBONE -> CA DISTANCE MATRIX 
* CA SEQUENCE -> HHBlits -> MSA -> MSA PAIR STATISTICS -> 1D FEATURES & DCA FEATURES

# PARSER
* Select from proteins and chains specified in pdb70_a3m.ffindex
* Only parse residues which have a CA ATOM
	* This style of selective parsing is a workflow parallel to Baker group
* Save the extracted CA BACKBONE and CA SEQUENCE

# CA DISTANCE MATRIX 
We take the CA BACKBONE 
For each chain using the CA coordinates we construct a distance matrix, M
M[ij] = distance between ith CA's xyz and jth CA's abc coordinates. distance(xyz,abc) = sqrt( (x-a)^2+(y-b)^2+(z-c)^2 )

* Matrix-based batch calculation
	* M = np.stack(L*[chain], axis=1)
	* sq_dist = ( M-M.transpose([1,0,2]) )**2
	* np.sqrt( sq_dist.sum(2) )

It took 2 days to parse and create CA matrices for 85756 chains

# MSA

Multiple Sequence Alignment 
Definition: Given a CA SEQUENCE of length L, find other related sequences to the sequence, which are aligned to the original with gaps, replacement, and deletion. The collection of sequences with length L as a result from this process is called a MSA.
This process will output an MSA alignment file

## HHblits (Soedinglab)
Classical alignment methods directly compared sequences to determine alignment
HMM (Hidden markov machine) alignment methods compare HMM to other HMM by using them as profiles.
The alignment tool used in Baker group is HHBlits
Some additional parameters used in baker are "HHblits (version 3.0.3) (28) with default parameters at 4 different e-value cutoffs: 1e−40, 1e−10, 1e−3, and 1." (Baker)

We ran hhblits with -e .001 -n 1
* -n     [1,8]   number of iterations (default=2)
* -e     [0,1]   E-value cutoff for inclusion in result alignment (def=0.001)
* -d database '/raid0/uniclust/uniclust30_2018_08/uniclust30_2018_08' 

The Expect value is a parameter that describes the number of hits one can "expect" to see by chance when searching a database of a particular size. It decreases exponentially as the Score (S) of the match increases. Essentially, the E value describes the random background noise.

It took 4 days to process and create alignments for 85756 chains in uniclust30_2018_08 with 64 separate processes

# MSA PAIR STATISTICS
The approach is from Baker group.
Several information are computed from the sequence and MSA for insertion into the neural network pipeline.
First, the MSA is converted into a one-hot encoded matrix of shape NxLx21 - (Number sequences in MSA)x(sequence Length)x(aminoacids+gap).
From this result, the 1D FEATURES and DCA FEATURES are computed.
Finally, they are combined into one feature matrix.

## 1D FEATURES
1D FEATURES include, sequence information, PSSM, and entropy.
1. The sequence information has shape Lx20
2. The PSSM has shape Lx21
3. The entropy has shape Lx1
These 3 vectors are combined into a single vector of shape Lx42
Then they are tiled L times horizontally and vertically to produce matrices of shape LxLx42 and LxLx42
Then they are combined into a single matrix of shape LxLx84

### 1. sequence information
The sequence of the matrix is one-hot encoded into a matrix of shape Lx20. L is the length of the sequence and 20 is the amino acids.
For example if there was a 3 letter amino alphabet with the sequence ABBC, it would be encoded into a shape of 4x3 as
* A - 100
* B - 010
* B - 010
* C - 001

### 2. PSSM
Position specific scoring matrix gives a probability of a specific position in L being a certain amino acid or a gap.
It has the shape Lx21 where L is the length of the aligned sequences and 21 is the amino acids + gap.

### 3. Entropy
Entropy is computed using the Shannon entropy formula where entropy = - sum( P_a log(P_a) ) where the sum iterates over P_a, the probabilty of a specific position being an amino acid a.
Entropy is calculated for each position in L, resulting in a matrix of shape Lx1.

## DCA FEATURES
DCA (direct coupling analysis) features use MSA to create inverse covariance matrix and APC 
* Inverse covariance matrix has shape (L\*21)x(L\*21) 
	* They are indexed over the ordered pair i,A. where i is the position in the sequence and A is one of 21 (amino acid+gap)
	* The expected value is computed for each i,A by counting the weighted sum of the frequencies of i,A occuring in the MSA
		* See f_i(A) and f_i,j(A,B) in [Baker pg7].
		* These two expected are used to compute the covariance matrix at each position. COV[(i,A),(j,B)] = f_i,j(A,B) - f_i(A)*f_j(B) [eq1 Baker pg7]
	* After the covariance matrix is computed, it is normalized and inversed

* APC (average product correlation score) has shape LxLx1
	* It is a summary statistic from the inverse covariance matrix [eq 3,4 Baker pg7]

These two features are combined by first reshaping the inverse covariance matrix into a matrix of shape LxLx(21\*21) and then concatenating it with the APC into a final matrix of shape LxLx(21\*21+1) =  LxL(442)
