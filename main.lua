require 'torch'
require 'nn'
require 'gnuplot'

batch_size = 1
epochs = 10

torch.manualSeed(4)

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------
-- x = torch.Tensor(20,1):normal()
-- y = torch.Tensor(20,1):normal()


-- x = torch.Tensor({ {-1.2298692343756557},{-1.4790558232925832},{-3.7045500241219997},{0.5631082132458687},
-- {-2.8645302983932197},{-0.9899101150222123},{-3.936412369366735},{1.6861401032656431},
-- {-4.39313477370888},{-1.7984784278087318},{-2.7519936161115766},{-4.03499003034085},
-- {3.952302092220634},{3.0520377843640745},{3.4072951041162014},{0.6105249538086355},
-- {4.702801615931094},{-2.7155804028734565},{-4.719105246476829},{-2.2375843627378345}})

x = torch.Tensor({ {-3.974552475847304},{3.890193263068795},{3.3466140856035054},{-4.120391132310033},
{4.980689950753003},{-4.15529357502237},{-4.585367743857205},{0.13327481225132942},
{-3.1235018325969577},{-2.5773165305145085},{-4.321673638187349},{4.571251834277064},
{4.427828260231763},{0.7843026076443493},{-1.445614816620946},{2.104754731990397},
{2.7231292380020022},{4.150048489682376},{0.414216797798872},{4.164597182534635}})

-- labels
-- y = torch.Tensor({ {1.1590842176742084},{1.4728360838646786},{-1.9770795193024098},{0.30059678062623},
-- {0.7835384691668746},{0.8275417479417482},{-2.8095645056200893},{1.6749361717052422},
-- {-4.171148600419123},{1.7520636702980203},{1.0452554197803037},{-3.1440868625162217},
-- {-2.8645342896311523},{0.2729596435210097},{-0.8947118533945095},{0.3500125264712734},
-- {-4.702585482568453},{1.122193967714708},{-4.718998811633974},{1.758321099480447}})

y = torch.Tensor({ {-2.940873316797358},{-2.6477206762556778},{-0.6813309473819722},{-3.4192138588925243},
{-4.802494260893207},{-3.526989980920669},{-4.548426386701057},{0.017709639779618713},
{0.05650363034260941},{1.3783599567624734},{-3.995978735928225},{-4.525798392146082},
{-4.24976344280986},{0.5539777801246921},{1.4343029112592043},{1.8117712111627475},
{1.106562448038169},{-3.510980676831623},{0.16671111275180542},{-3.5552193366671077}})

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

-- TODO: Check that model matches js demo.
local n_inputs = 1
local HUs = 20
local n_outputs = 1

model = nn.Sequential()   
local criterion = nn.MSECriterion()        

model:add(nn.Dropout(0.05))
model:add(nn.Linear(n_inputs, HUs)) 
model:add(nn.ReLU())
model:add(nn.Dropout(0.05))
-- model:add(nn.Linear(20, 20))
model:add(nn.Sigmoid())
model:add(nn.Linear(HUs, n_outputs))

------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------

-- learning_rate:0.01, momentum:0.0, batch_size:10, l2_decay:0.00001
local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 1
trainer.momentum = 0.0
trainer.learningRateDecay = 0.00001


local iterations = epochs * math.ceil(x:size(1) / batch_size) -- integer number of minibatches to process
local avloss = 0
local iteration = 0

local w, dl_dx = model:getParameters()

for i = 1,iterations do
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
print(y_hat)
gnuplot.plot(x:reshape(20),y_hat:reshape(20))
gnuplot.figure()

------------------------------------------------------------------------------
-- DRAW REGRESSION
------------------------------------------------------------------------------
c = 0
density = 5.0
ss = 30
WIDTH = 700
HEIGHT = 500

sum_y = {}
sum_y_sq = {}
for i=0, 1000 do
   sum_y[i] = 0
   sum_y_sq[i] = 0
end

final_decision_points = torch.Tensor(141,2)

for i=0.0,WIDTH, density do 
	_x = (i-WIDTH/2)/ss
	input = torch.Tensor({_x})
    _y = model:forward(input);
    sum_y[c] = sum_y[c] + _y[1]
    sum_y_sq[c] = sum_y_sq[c] + (_y[1]*_y[1])

    print(_y[1])
--	print(-y*ss+HEIGHT/2)	
	final_decision_points[c+1][1] = i

	-- JS Version adds a minus to -y[1]. Not sure why?
	final_decision_points[c+1][2] = _y[1]*ss+HEIGHT/2

	c = c + 1
end

x_axis = final_decision_points[{{},{1}}]:reshape(final_decision_points:size(1))
y_axis = final_decision_points[{{},{2}}]:reshape(final_decision_points:size(1))
gnuplot.plot(x_axis, y_axis)
