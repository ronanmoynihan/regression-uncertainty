## Regression Uncertainty - Boston Housing Dataset
Test MSE: 12.66 (RMSE: 3.55)  

### Model

Two Hidden layers with 50 units.  

```
    model:add(nn.Dropout(p))
    model:add(nn.Linear(n_inputs, HUs)) 
    model:add(nn.ReLU())
    model:add(nn.Dropout(p))
    model:add(nn.Linear(HUs, HUs)) 
    model:add(nn.Sigmoid())
    model:add(nn.Linear(HUs, n_outputs))
```

### Test Output

% Certainty tells you how much confidence you can have in the prediction.

``` 
#   prediction     actual      +/-        var %      % certainty	
 1     21.34       24.50       2.80       0.11       0.89	
 2     16.91       16.10       1.66       0.08       0.92	
 3     13.25       15.40       2.67       0.17       0.83	
 4     15.10        5.60       9.59       0.61       0.39	
 5     25.30       23.20       5.93       0.22       0.78	
 6     12.47       10.50       0.64       0.02       0.98	
 7     17.31       19.10       3.26       0.17       0.83	
 8     21.53       24.00       2.22       0.09       0.91	
 9     42.48       42.30      11.50       0.26       0.74	
10     16.65       14.30       0.61       0.01       0.99	
11     17.89       17.80       4.64       0.24       0.76	
12     11.38        8.80       1.47       0.10       0.90	
13     31.33       24.00       6.10       0.18       0.82	
14     30.08       36.00      10.00       0.32       0.68	
15     24.72       21.20       2.81       0.10       0.90	
16     23.07       23.80       2.07       0.07       0.93	
17     17.95       20.40       0.70       0.02       0.98	
18     20.00       21.80       0.88       0.03       0.97	
19     27.25       26.40       3.66       0.12       0.88	
20     24.14       18.80       3.70       0.14       0.86
```