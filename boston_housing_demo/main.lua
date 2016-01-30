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
cmd:text('$> th -i main.lua')
cmd:text('Options:')

cmd:text()
local opt = cmd:parse(arg or {})
local criterion_opt = {}

-- The Boston.csv file has been converted to torch.
local data_file = 'data/Boston.th'

torch.manualSeed(6)

opt.p = 0.1
opt.learningRate = 7.14e-03
opt.learningRateDecay = 7.76e-04

opt.model_name = 'model'
opt.optimization = 'sgd'
opt.print_training_loss = 1000
opt.test_model_iteration = 15000 -- how often to print the training & test loss.

-- NOTE: the code below changes the optimization algorithm used, and its settings
local optimState       -- stores a lua table with the optimization algorithm's settings, and state during iterations
local optimMethod      -- stores a function corresponding to the optimization routine	

optimState = {
  learningRate = opt.learningRate,
  --momentum = 8.87e-01,
  learningRateDecay = opt.learningRateDecay

}
opt.batch_size = 100
opt.epochs = 6000
optimMethod = optim.sgd


local data_train_percentage = 90 
data = data_loader.load_data(data_file, data_train_percentage)
print(string.format('\n Training data rows: %d , features: %d', data.train_data:size(1),data.train_data:size(2)) )
print(string.format('\n Test data rows: %d , features: %d \n', data.test_data:size(1),data.test_data:size(2) ))


local criterion = nn.MSECriterion()
local model = create_model(ninputs,opt.p)

-- reset weights
local method = 'uncertainty'
local model = require('weight-init')(model, method)

-- Train.
local model, training_losses, test_losses = train(model,opt,optimMethod,optimState, data, criterion)

-- this is anew hyperparamater for uncertaincty.
-- First, define a prior length-scale ll. This captures our belief over the function frequency. 
-- A short length-scale ll corresponds to high frequency data, 
-- and a long length-scale corresponds to low frequency data. 
l = 1.45

outputs = torch.zeros(data.test_data:size(1),10)

y_hat = 0
for n=1,data.test_data:size(1) do
	for i=1,10 do
		outputs[n][i] = model:forward(data.test_data[n])[1]
	end
end

predictive_mean = outputs:mean(2)
diff = outputs - outputs:mean(2):expand(data.test_data:size(1), 10)

predictive_variance = torch.mean(torch.pow(torch.abs(diff),2),2)
tau = torch.pow(l,2) * (1 - opt.p) / (2 * data.train_data:size(1) * opt.learningRateDecay)
predictive_variance_tau = predictive_variance + torch.pow(tau,-1)


final_test_outputs = model:forward(data.test_data)
pred_prob = torch.cdiv((final_test_outputs - predictive_variance), final_test_outputs)
var_perc = torch.cdiv(predictive_variance,final_test_outputs)

print('\n#   prediction     actual      +/-        var %      % certainty')
for i = 1,20 do
	print(string.format("%2d    %6.2f      %6.2f     %6.2f     %6.2f     %6.2f", i,  final_test_outputs[i][1],data.test_targets[i][1],predictive_variance_tau[i][1],var_perc[i][1],pred_prob[i][1]))
end
