# Tumor-Classification
Comparison of the efficacy of classification methods utilizing breast cancer tumor data.

## Classification Methods and Results
Binary Classification: malignant or benign
* Stochastic Gradient Descent Classifier.
* K Nearest Neighbors Classifier.
* Support Vector Classifier.
* Gaussian Process Classifier based on Laplace approximation.
* Decision Tree Classifier.
* Random Forrest Classifier.
* Multi-layer Perceptron classifier.
* AdaBoost Classifier (AdaBoost-SAMME).
* Gaussian Naive Bayes Classifier.
* Quadratic Discriminant Analysis.


## Dataset Information
Title: Wisconsin Diagnostic Breast Cancer (WDBC)

### Creators: 

	Dr. William H. Wolberg, General Surgery Dept., University of 	Wisconsin,  Clinical Sciences Center, Madison, WI 53792
	wolberg@eagle.surgery.wisc.edu

	W. Nick Street, Computer Sciences Dept., University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
	street@cs.wisc.edu  608-262-6619

	Olvi L. Mangasarian, Computer Sciences Dept., University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
	olvi@cs.wisc.edu 

b) Donor: Nick Street

c) Date: November 1995

### Relevant information

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image.
A few of the images can be found at: 	[http://www.cs.wisc.edu/~street/images/](http://www.cs.wisc.edu/~street/images/)

Number of instances: 569 

Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

#### Attribute information

1. ID number
2. Diagnosis (M = malignant, B = benign)
3. Ten real-valued features are computed for each cell nucleus:

   a)1. radius (mean of distances from center to points on the perimeter)  
   b) texture (standard deviation of gray-scale values)  
   c) perimeter  
   d) area  
   e) smoothness (local variation in radius lengths)  
   f) compactness (perimeter^2 / area - 1.0)  
   g) concavity (severity of concave portions of the contour)  
   h) concave points (number of concave portions of the contour)  
   i) symmetry   
   j) fractal dimension ("coastline approximation" - 1)  


*All feature values are recoded with four significant digits.*

4. Missing attribute values: none

5. Class distribution: 357 benign, 212 malignant
