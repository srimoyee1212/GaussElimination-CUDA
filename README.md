# GaussElimination-CUDA

## Mathematical Background
Gauss Elimination is an algorithm  for getting matrices in reduced row echelon form using elementary row operations. Gaussian Elimination has two parts. The first part (Forward Elimination) reduces a given system  to  triangular  form.  The  second  step  uses  back  substitution  to  find  the  solution  of  the  triangular  echelon  form  system

It is widely used to solve equations of the form :
- a<sub>11</sub>x<sub>1</sub> + a<sub>12</sub>x<sub>2</sub> + .... a<sub>1n</sub>x<sub>n</sub> = b<sub>1</sub>
- a<sub>21</sub>x<sub>1</sub> + a<sub>22</sub>x<sub>2</sub> + .... a<sub>2n</sub>x<sub>n</sub> = b<sub>2</sub>
.
.
.
- a<sub>n1</sub>x<sub>1</sub> + a<sub>n2</sub>x<sub>2</sub> + .... a<sub>nn</sub>x<sub>n</sub> = b<sub>n</sub>


## Problems with using Traditional Method of Gaussian Elimination

One of the major computational problems that arise is high space complexity. As the method involves division by diagonal elements and if these are 0 or very small values, the values in the rows below this diagonal become very large and is difficult to handle this in most computing systems.

## Gaussian Elimination with Partial Pivoting (GEPP)
The process :
- Forward Substitution with Partial Pivoting 
- Back Substitution

Algorithm :
- Partial pivoting: Find the kth pivot by swapping rows, to move the entry with the largest absolute value to the pivot position. This imparts computational stability to the algorithm.
- For each row below the pivot, calculate the factor f which makes the kth entry zero, and for every element in the row subtract the fth multiple of the corresponding element in the kth row.

The standard implementation of this algorithm has a worst case complexity of O(n<sup>3</sup>) and a working complexity of O(n<sup>2.25</sup>)

The parallel implementation of this algorithm for a matrix of order N is has a linear complexity.
