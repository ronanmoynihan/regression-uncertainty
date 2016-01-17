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

opt.model_name = 'model'
opt.optimization = 'sgd'
opt.print_training_loss = 1000
opt.test_model_iteration = 20000 -- how often to print the training & test loss.

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine

optimState = {
  learningRate = 1e-1,
  momentum = 0.4,
  learningRateDecay = 1e-4
}
opt.batch_size = 500
opt.epochs = 20000
optimMethod = optim.sgd


local data_train_percentage = 70 
data = data_loader.load_data(data_file, data_train_percentage)
print(string.format('\n Training data rows: %d , features: %d', data.train_data:size(1),data.train_data:size(2)) )
print(string.format('\n Test data rows: %d , features: %d \n', data.test_data:size(1),data.test_data:size(2) ))

-- Use regular MSE as default criterion.
local criterion = nn.MSECriterion()

-- Train.
local model, training_losses, test_losses = train(opt,optimMethod,optimState, data, criterion)

-- Test.
test(data,model) 
test(data,model) 
test(data,model) 