require 'torch'
require 'optim'
require 'math'
require 'nn'
local data_loader = require 'data'
local train = require 'train'
local test = require 'test_model'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Model uncertainty - Boston Housing Dataset.')
cmd:text('Example:')
cmd:text('$> th main.lua -')
cmd:text('Options:')
-- cmd:option('-under', false, 'train model to under predict')
-- cmd:option('-over', false, 'train model to over predict')
cmd:text()
local opt = cmd:parse(arg or {})
local criterion_opt = {}

-- The Boston.csv file has been converted to torch.
local data_file = 'data/Boston.th'

torch.manualSeed(4)

p = 0.1

opt.model_name = 'model'
opt.optimization = 'sgd'
opt.print_training_loss = 1000
opt.test_model_iteration = 15000 -- how often to print the training & test loss.

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine

-- |   Hyperparameter   | Best Seen | Previous |
-- ---------------------------------------------	
-- |         x1         | 7.14e-03  | 5.15e-01 |	
-- |         x2         | 8.87e-01  | 4.35e-02 |	
-- |         x3         | 7.76e-04  | 5.10e-02 |	
-- |         x4         | 5.11e-02  | 2.08e-01 |	
-- |         x5         | 7.22e-01  | 6.71e-01 |	

optimState = {
  learningRate = 7.14e-03 ,
  --momentum = 8.87e-01,
  learningRateDecay = 7.76e-04

}
opt.batch_size = 50
opt.epochs = 6000
optimMethod = optim.sgd


local data_train_percentage = 70 
data = data_loader.load_data(data_file, data_train_percentage)
print(string.format('\n Training data rows: %d , features: %d', data.train_data:size(1),data.train_data:size(2)) )
print(string.format('\n Test data rows: %d , features: %d \n', data.test_data:size(1),data.test_data:size(2) ))

-- Use regular MSE as default criterion.
local criterion = nn.MSECriterion()

-- Train.
local model, training_losses, test_losses = train(opt,optimMethod,optimState, data, criterion, p)

-- Test.
test(data,model) 
-- test(data,model) 
-- test(data,model) 

-- probs = []
-- for _ in xrange(T):
--     probs += [model.output_probs(input_x)]
-- predictive_mean = numpy.mean(prob, axis=0)
-- predictive_variance = numpy.var(prob, axis=0)
-- tau = l**2 * (1 - model.p) / (2 * N * model.weight_decay)
-- predictive_variance += tau**-1

l = 0.005
weight_decay = 0.001

outputs = torch.zeros(data.test_data:size(1),10)
-- outputs = {}
y_hat = 0
for n=1,data.test_data:size(1) do
	for i=1,10 do
		--outputs[n] = outputs[n] + 0 + model:forward(data.test_data[n])[1]
		outputs[n][i] = model:forward(data.test_data[n])[1]
	end
end

-- 
predictive_mean = outputs:mean(2):expand(data.test_data:size(1), 10)
m = outputs - outputs:mean(2):expand(data.test_data:size(1), 10)

-- outputs = torch.Tensor{1,2,3}

-- predictive_mean = outputs:mean()
-- predictive_variance = torch.mean(torch.pow(torch.abs(outputs - outputs:mean()),2))
-- tau = torch.pow(l,2) * (1 - p) / (2 * data.train_data:size(1) * weight_decay)
-- print(predictive_variance)
-- predictive_variance = predictive_variance + tau
-- print(predictive_variance)