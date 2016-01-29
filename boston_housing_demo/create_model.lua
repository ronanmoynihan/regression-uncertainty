require 'nn'

function create_model(input_dim,p)

  	------------------------------------------------------------------------------
   	-- MODEL
    ------------------------------------------------------------------------------

    local n_inputs = 13
    local HUs = 9
    local n_outputs = 1

    model = nn.Sequential()   
    local criterion = nn.MSECriterion()        

    model:add(nn.Dropout(p))
    model:add(nn.Linear(n_inputs, 50)) 
    model:add(nn.ReLU())
    model:add(nn.Dropout(p))
    model:add(nn.Linear(50, 50)) 
    model:add(nn.Sigmoid())
    model:add(nn.Linear(50, n_outputs))

    return model
end

return create_model