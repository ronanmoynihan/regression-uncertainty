require 'torch'
require 'nn'
require 'gnuplot'

ticks = 10
epochs = 50
iterations_complete = 0

torch.manualSeed(4)

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------
-- x = torch.Tensor(20,1):normal()
-- y = torch.Tensor(20,1):normal()
x = torch.Tensor({ {-3.974552475847304},{3.890193263068795},{3.3466140856035054},{-4.120391132310033},
{4.980689950753003},{-4.15529357502237},{-4.585367743857205},{0.13327481225132942},
{-3.1235018325969577},{-2.5773165305145085},{-4.321673638187349},{4.571251834277064},
{4.427828260231763},{0.7843026076443493},{-1.445614816620946},{2.104754731990397},
{2.7231292380020022},{4.150048489682376},{0.414216797798872},{4.164597182534635}})

-- labels
y = torch.Tensor({ {-2.940873316797358},{-2.6477206762556778},{-0.6813309473819722},{-3.4192138588925243},
{-4.802494260893207},{-3.526989980920669},{-4.548426386701057},{0.017709639779618713},
{0.05650363034260941},{1.3783599567624734},{-3.995978735928225},{-4.525798392146082},
{-4.24976344280986},{0.5539777801246921},{1.4343029112592043},{1.8117712111627475},
{1.106562448038169},{-3.510980676831623},{0.16671111275180542},{-3.5552193366671077}})

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_inputs = 1
local HUs = 20
local n_outputs = 1
sum_y = {}
sum_y_sq = {}
for i=0, 1000 do
	sum_y[i] = 0
	sum_y_sq[i] = 0
end

model = nn.Sequential()   
local criterion = nn.MSECriterion()        

model:add(nn.Dropout(0.05))
model:add(nn.Linear(n_inputs, HUs)) 
model:add(nn.ReLU())
model:add(nn.Dropout(0.05))
model:add(nn.Sigmoid())
model:add(nn.Linear(HUs, n_outputs))

------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------

function update_reg()
	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = 0.01
	trainer.maxIteration = 1
	trainer.momentum = 0.0
	trainer.learningRateDecay = 0.00001

	local avloss = 0
	local iteration = 0

	local w, dl_dx = model:getParameters()

	for i = 1,epochs do
	 	iteration = iteration + 1

	    for i = 1, x:size(1) do
	 	-- 	 model:zeroGradParameters()
			-- local output = model:forward(x[i])
			-- local loss_x = criterion:forward(output,y[i])
			-- local dl_dy = criterion:backward(output, y[i])
			-- model:backward(x[i], dl_dy)
			-- avloss = avloss + loss_x
			-- print(loss_x)

			dataset = {}
			function dataset:size() return 1 end 
			dataset[1] = {x[i], y[i]}
			trainer:train(dataset)
		end	
		avloss = avloss / (x:size(1)*iteration)
		-- print(avloss)
	end	

	y_hat = model:forward(x)

	-- gnuplot.plot(x:reshape(20),y_hat:reshape(20),'+')
	-- gnuplot.figure()
end	


------------------------------------------------------------------------------
-- DRAW REGRESSION
------------------------------------------------------------------------------
function draw_reg()
	c = 0
	density = 5.0
	ss = 30
	WIDTH = 700
	HEIGHT = 500
	sum_y_count = 0

	------------------------------------------------------------------------------
	-- 1. Plot last forward pass.
	------------------------------------------------------------------------------

	final_decision_points = torch.Tensor(141,2)

	for i=0.0,WIDTH, density do 
		_x = (i-WIDTH/2)/ss
		input = torch.Tensor({_x})
	    _y = model:forward(input);
	    sum_y[c] = sum_y[c] + _y[1]
	    sum_y_sq[c] = sum_y_sq[c] + (_y[1]*_y[1])
		
		final_decision_points[c+1][1] = i

		-- JS Version adds a minus to -y[1]. Not sure why?
		final_decision_points[c+1][2] = _y[1]*ss+HEIGHT/2

		c = c + 1
		sum_y_count = c
		
	end
	iterations_complete = iterations_complete + 1

	x_axis = final_decision_points[{{},{1}}]:reshape(final_decision_points:size(1))
	y_axis = final_decision_points[{{},{2}}]:reshape(final_decision_points:size(1))

	------------------------------------------------------------------------------
	-- 2. Draw the mean plus minus 2 standard deviations
	------------------------------------------------------------------------------
	mean_plus_minus2_std = torch.Tensor(141,2)
	c = 0;
	for i=0.0,WIDTH, density do 
		
		mean = sum_y[c] / iterations_complete
		
		mean_plus_minus2_std[c+1][1] = i

		-- JS Version adds a minus to -y[1]. Not sure why?	
		mean_plus_minus2_std[c+1][2] = mean*ss+HEIGHT/2
		c = c + 1
	end	
	x_axis = mean_plus_minus2_std[{{},{1}}]:reshape(final_decision_points:size(1))
	y_axis = mean_plus_minus2_std[{{},{2}}]:reshape(final_decision_points:size(1))

	------------------------------------------------------------------------------
	-- 3. Draw the uncertainty
	------------------------------------------------------------------------------

	uncertainty1_plus = torch.Tensor(141,2)
	uncertainty1_minus = torch.Tensor(141,2)

    l2 = 0.005
    tau_inv = (2 * x:size(1) * 0.00001) / (1 - 0.05) / l2
    for u = 1, 1 do

    	c = 0;
    	for i=0.0, WIDTH, density do
    		mean = sum_y[c] / iterations_complete
    		y_sq_avg = sum_y_sq[c] / iterations_complete
    		std = torch.sqrt(y_sq_avg - mean * mean) + tau_inv 
    		mean = mean + 2*std * u/4

    		uncertainty1_plus[c+1][1] = i

			-- JS Version adds a minus to -y[1]. Not sure why?
			uncertainty1_plus[c+1][2] = mean*ss+HEIGHT/2
    		c = c+1
    	end
    	c = c - 1;
    	for i=WIDTH,0.0, -density do

    		mean = sum_y[c] / iterations_complete
    		y_sq_avg = sum_y_sq[c] / iterations_complete
    		std = math.sqrt(y_sq_avg - mean * mean) + tau_inv
    		mean = mean - 2*std * u/4
   
        	uncertainty1_minus[c+1][1] = i

			-- JS Version adds a minus to -y[1]. Not sure why?		
			uncertainty1_minus[c+1][2] = mean*ss+HEIGHT/2
    		c = c - 1
    	end

    end 

end

function plot()

	lastFP_x_axis = final_decision_points[{{},{1}}]:reshape(final_decision_points:size(1))
	lastFP_y_axis = final_decision_points[{{},{2}}]:reshape(final_decision_points:size(1))

	x_axis = mean_plus_minus2_std[{{},{1}}]:reshape(final_decision_points:size(1))
	y_axis = mean_plus_minus2_std[{{},{2}}]:reshape(final_decision_points:size(1))

	u_x = uncertainty1_plus[{{},{1}}]:reshape(uncertainty1_plus:size(1))
	yp = uncertainty1_plus[{{},{2}}]:reshape(uncertainty1_plus:size(1))
	ym = uncertainty1_minus[{{},{2}}]:reshape(uncertainty1_minus:size(1))
	yy = torch.cat(u_x,ym,2)
	yy = torch.cat(yy,yp,2)
	gnuplot.plot({'uncertainty',yy,' filledcurves'},
				 {'Average',x_axis,y_axis,'-'},
				 {'Last forward Pass',lastFP_x_axis, lastFP_y_axis,'-'},
				 {u_x,yp,'lines ls 1'},
				 {u_x,ym,'lines ls 1'})

end

function NPGtick(r) 
	update_reg()
    draw_reg()
    plot()
end

for r=1,ticks do
	NPGtick(r)
end

-- gnuplot.figure()
-- gnuplot.plot(x:reshape(20),y:reshape(20),'+')
-- gnuplot.figure()