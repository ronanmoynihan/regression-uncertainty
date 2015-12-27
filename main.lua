require 'torch'
require 'nn'

batch_size = 1
epochs = 1

torch.manualSeed(2)

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------
-- x = torch.Tensor(20,1):normal()
-- y = torch.Tensor(20,1):normal()


x = torch.Tensor({ {-1.2298692343756557},{-1.4790558232925832},{-3.7045500241219997},{0.5631082132458687},
{-2.8645302983932197},{-0.9899101150222123},{-3.936412369366735},{1.6861401032656431},
{-4.39313477370888},{-1.7984784278087318},{-2.7519936161115766},{-4.03499003034085},
{3.952302092220634},{3.0520377843640745},{3.4072951041162014},{0.6105249538086355},
{4.702801615931094},{-2.7155804028734565},{-4.719105246476829},{-2.2375843627378345}})

-- labels
y = torch.Tensor({ {1.1590842176742084},{1.4728360838646786},{-1.9770795193024098},{0.30059678062623},
{0.7835384691668746},{0.8275417479417482},{-2.8095645056200893},{1.6749361717052422},
{-4.171148600419123},{1.7520636702980203},{1.0452554197803037},{-3.1440868625162217},
{-2.8645342896311523},{0.2729596435210097},{-0.8947118533945095},{0.3500125264712734},
{-4.702585482568453},{1.122193967714708},{-4.718998811633974},{1.758321099480447}})

------------------------------------------------------------------------------
-- MODEL
------------------------------------------------------------------------------

-- TODO: Check that model matches js demo.
local n_inputs = 1
local n_outputs = 1

local model = nn.Sequential()   
local criterion = nn.MSECriterion()        

model:add(nn.Dropout())
model:add(nn.Linear(n_inputs, 20)) 
model:add(nn.ReLU())
model:add(nn.Dropout())
model:add(nn.Linear(20, n_inputs))
model:add(nn.Sigmoid())
model:add(nn.Linear(n_inputs, n_outputs))

------------------------------------------------------------------------------
-- TRAIN
------------------------------------------------------------------------------
local iterations = epochs * math.ceil(x:size(1) / batch_size) -- integer number of minibatches to process
local avloss = 0
local iteration = 0
for i = 1,iterations do
 	iteration = iteration + 1
    for i = 1, x:size(1) do
		local output = model:forward(x[i])
		local loss_x = criterion:forward(output,y[i])
		local dl_dy = criterion:backward(output, y[i])
		local dl_dx = model:backward(x[i], dl_dy)
		avloss = avloss + loss_x
	end	
	avloss = avloss / (x:size(1)*iteration)
	print(avloss)
end	

y = model:forward(torch.Tensor({-11.666666666666666}))
print(y) -- -0.64

y1 = model:forward(torch.Tensor({-8}))
print(y1) -- -0.084

