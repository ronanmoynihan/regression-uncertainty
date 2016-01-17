require 'torch' 
require 'nn'
require 'optim'
require 'math'
local create_model = require 'create_model'

local function train(opt,optimMethod,optimState,data,criterion)

	------------------------------------------------------------------------
	-- create model and loss/grad evaluation function
	--
	local ninputs = data.train_data:size(2)
	local n_train_data = data.train_data:size(1) -- number of training data
	local model = create_model(ninputs)

	local x, dl_dx = model:getParameters()

	print(model)

	local counter = 0

	local feval = function(x_new)
	   -- set x to x_new, if differnt
	   -- (in this simple example, x_new will typically always point to x,
	   -- so the copy is really useless)
	   if x ~= x_new then
	      x:copy(x_new)
	   end

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

	  local batch_inputs = data.train_data[{{start_index, end_index}, {}}]
	  local batch_targets = data.train_targets[{{start_index, end_index}}]

	   -- reset gradients (gradients are always accumulated, to accomodate 
	   -- batch methods)
	   dl_dx:zero()

	   -- evaluate the loss function and its derivative wrt x, for that sample
	   local loss_x = criterion:forward(model:forward(batch_inputs), batch_targets)
	   model:backward(batch_inputs, criterion:backward(model.output, batch_targets))

	   -- return loss(x) and dloss/dx
	   return loss_x, dl_dx
	end


	local losses = {}          -- training losses for each iteration/minibatch
	local test_losses = {}
	local epochs = opt.epochs  -- number of full passes over all the training data
	local iterations = epochs * math.ceil(n_train_data / opt.batch_size) -- integer number of minibatches to process


	   -- this variable is used to estimate the average loss
	   local current_loss = 0
	   local test_loss = 0

	   -- an epoch is a full loop over our training data
	   for i = 1,iterations do

	      -- optim contains several optimization algorithms. 
	      -- All of these algorithms assume the same parameters:
	      --   + a closure that computes the loss, and its gradient wrt to x, 
	      --     given a point x
	      --   + a point x
	      --   + some parameters, which are algorithm-specific
	      local _,minibatch_loss = optimMethod(feval,x,optimState)
	      -- Functions in optim all return two things:
	      --   + the new x, found by the optimization method (here SGD)
	      --   + the value of the loss functions at all points that were used by
	      --     the algorithm. SGD only estimates the function once, so
	      --     that list just contains one value.

	      current_loss = current_loss + minibatch_loss[1]
	      losses[#losses + 1] = minibatch_loss[1] -- append the new loss

	      if i % opt.print_training_loss == 0 then
			print('training loss = ' .. minibatch_loss[1])
		  end	

		  if i % opt.test_model_iteration == 0 then
		   		local test_outputs = model:forward(data.test_data)
		    	local test_loss = criterion:forward(test_outputs, data.test_targets)
		    	test_losses[#test_losses + 1] = test_loss -- append the new loss
		    	print('--------------------------------')
			    print('test_loss = ' .. test_loss)
			    print('--------------------------------')
		  end

	   end  

    return model, losses, test_losses
end

return train