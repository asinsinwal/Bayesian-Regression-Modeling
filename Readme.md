Bitcoin - Bayesian Regression Modeling

To implement the Bayesian Regression model to predict the future price variation of bitcoin.

1. Compute the price variations (Δp1, Δp2, and Δp3) for train2 using train1 as input to the Bayesian Regression equation. I have used the similarity metric in place of the Euclidean distance in Bayesian Regression.

2. Compute the linear regression parameters (w0, w1, w2, w3) by finding the best linear fit. Here I have used the 'ols' function of 'statsmodels.formula.api'. My model fits using Δp1, Δp2, and Δp3 as the covariates. Note: the bitcoin order book data was not available, so the rw4 term is neglected.

3. Use the linear regression model computed in Step 2 and Bayesian Regression estimates, to predict the price variations for the test dataset. Bayesian Regression estimates for test dataset are computed in the same way as they are computed for train2 dataset – using train1 as an input.

4. Once the price variations are predicted, compute the mean squared error (MSE) for the test dataset (the test dataset has 50 vectors => 50 predictions).
