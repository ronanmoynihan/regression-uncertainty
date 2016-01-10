## Regression Uncertainty
Torch version of the Javascript [demo](https://github.com/yaringal/DropoutUncertaintyDemos) for Representing Model Uncertainty in Deep Learning as explained in this blog [post](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) 

### Dependencies

- [torch7](http://torch.ch/docs/getting-started.html)   
- [gnuplot](https://github.com/torch/gnuplot)  

### Running

The code takes in two arguments as follows,  

-std     -> how many levels of standard deviations to plot [1-4]. Each level is half a standard deviation.  
-ticks   -> number of times to run 1 full cycle


```lua
 th -i main.lua -std 2 -ticks 50
`` 

### Plots

#### 2 Iterations
![](https://raw.githubusercontent.com/ronanmoynihan/regression-uncertainty/master/plots/ticks2.png?token=ABo7hwfOZp6_dt52H8HgXqEf18_1fwduks5Wlpo7wA%3D%3D "Optional Title")

#### 50 Iterations
![](https://raw.githubusercontent.com/ronanmoynihan/regression-uncertainty/master/plots/ticks50.png?token=ABo7h1wRLLBl-RUjz_c0HDyqNRIwEEuJks5WlpqEwA%3D%3D "Optional Title")