require 'torch'
require 'nn'
require 'gnuplot'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Trains a Regression model that can be biased to under or over predict.')
cmd:text('Example:')
cmd:text('$> th main.lua -std 1 -ticks 10')
cmd:text('Options:')
-- cmd:option('-std', 1, 'how many levels of standard deviations to plot [1-4].')
-- cmd:option('-ticks', 10, 'number of times to run 1 full cycle.')
cmd:text()
local opt = cmd:parse(arg or {})

ticks = opt.ticks
iterations_complete = 0

density = 5.0
ss = 30
WIDTH = 700
HEIGHT = 500


  optimState = {
    learningRate = 1e-1,
    momentum = 0.4,
    learningRateDecay = 1e-4
  }
  opt.batch_size = 50
  opt.epochs = 30


-- initialise arrays
sum_y = {}
sum_y_sq = {}
for i=0, WIDTH / density do
	sum_y[i] = 0
	sum_y_sq[i] = 0
end

------------------------------------------------------------------------------
-- Data
------------------------------------------------------------------------------

data = torch.load('data/boston.t7')

local mins, maxes = data.xr:min(1), data.xr:max(1)
maxes:add(-mins)
maxes:maskedFill(maxes:eq(0), 1)
  
data.xr:add(-mins:expandAs(data.xr)):cdiv(maxes:expandAs(data.xr))

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

local n_inputs = 13
local HUs = 9
local n_outputs = 1

model = nn.Sequential()   
local criterion = nn.MSECriterion()        

model:add(nn.Dropout(0.15))
model:add(nn.Linear(n_inputs, HUs)) 
model:add(nn.ReLU())
model:add(nn.Dropout(0.15))
model:add(nn.Linear(HUs, HUs)) 
model:add(nn.Sigmoid())
model:add(nn.Linear(HUs, n_outputs))

------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------

	local trainer = nn.StochasticGradient(model, criterion)
	trainer.learningRate = optimState.learningRate
	-- trainer.maxIteration = 1
	trainer.momentum = optimState.momentum
	trainer.learningRateDecay = optimState.learningRateDecay

	local n_train_data = data.xr:size(1) -- number of training data

	local avloss = 0
	local iteration = 0

	local iterations = opt.epochs * math.ceil(data.xr:size(1) / opt.batch_size)

	local counter = 0

	for i = 1,iterations do

	  -- get start/end indices for our minibatch (in this code we'll call a minibatch a "batch")
	  --           ------- 
	  --          |  ...  |
	  --        ^ ---------<- start index = i * batchsize + 1
	  --  batch | |       |
	  --   size | | batch |       
	  --        v |   i   |<- end index (inclusive) = start index + batchsize
	  --          ---------                         = (i + 1) * batchsize + 1
	  --          |  ...  |                 (except possibly for the last minibatch, we can't 
	  --          --------                   let that one go past the end of the data, so we take a min())
	  local start_index = counter * opt.batch_size + 1
	  local end_index = math.min(n_train_data, (counter + 1) * opt.batch_size + 1)
	  if end_index == n_train_data then
	    counter = 0
	  else
	    counter = counter + 1
	  end

	  local batch_inputs = data.xr[{{start_index, end_index}, {}}]
	  -- print(batch_inputs)
	  local batch_targets = data.yr[{{start_index, end_index}}]


		dataset = {}
		function dataset:size() return batch_inputs:size(1) end 
		for i=1,dataset:size() do 
				-- batch_targets[i] = torch.Tensor(batch_targets[i])
				-- print(torch.Tensor({batch_targets[i]}))
			
			  dataset[i] = {batch_inputs[i], torch.Tensor({batch_targets[i]})}
		end
		trainer:train(dataset)

	end	


