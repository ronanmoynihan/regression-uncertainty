require 'nn'
require 'gnuplot'


local function test_model(data,model)

	local y_hat = model:forward(data.test_data)
	local diff = y_hat - data.test_targets

	gnuplot.hist(diff,30)
	gnuplot.figure()

	print('\n#   prediction     actual      diff')
	for i = 1,20 do
	    print(string.format("%2d    %6.2f      %6.2f     %6.2f", i,  y_hat[i][1],data.test_targets[i][1],diff[i][1]))
	end
end

return test_model