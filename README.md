# MLT_Project

Marium:
Just added the file computing max information gain. A few concerns and comments: 
1. k(x, x') needs to be <=1 for all x in D. Don't have anything in code checking that for now.
2. Using a Gaussian kernel (RBF) but need to double check if the implementation is accurate.
