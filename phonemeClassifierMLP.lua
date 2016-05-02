require 'io'
require 'nn'


test = io.open('test_train/test.txt', 'r')
train = io.open('test_train/train.txt', 'r')
header = test:read()
header = train:read()

map = {iy=1, aa=2, qq=3, ch=4, ei=5, ai=6, ii=7, zh=8, p=9, ng=10, breath=11, sh=12, sil=13, th=14, iq=15, uh=16, q=17, dh=18, oi=19, ow=20, eir=21, jh=22, a=23, oo=24, b=25, e=26, d=27, g=28, f=29, i=30, h=31, k=32, m=33, l=34, o=35, n=36, uu=37, s=38, r=39, u=40, t=41, w=42, v=43, y=44, ou=45, z=46}
map2 = {'iy', 'aa', 'qq', 'ch', 'ei', 'ai', 'ii', 'zh', 'p', 'ng', 'breath', 'sh', 'sil', 'th', 'iq', 'uh', 'q', 'dh', 'oi', 'ow', 'eir', 'jh', 'a', 'oo', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'm', 'l', 'o', 'n', 'uu', 's', 'r', 'u', 't', 'w', 'v', 'y', 'ou', 'z'}

testData = {}
trainData = {}
training = {}

local i = 0  
for line in test:lines('*l') do  
	i = i + 1
	local l = line:split(', ')
	label = l[3]
	input = {}
	for x=4,42 do
		input[x-3] = l[x]
	end
	output = torch.Tensor(46):zero()
	lab = map[label]
	output[lab] = 1
	testData[i] = {torch.Tensor(input), output, label, l[1]}
end

local i = 0  
for line in train:lines('*l') do  
	i = i + 1
	local l = line:split(', ')
	label = l[3]
	input = {}
	for x=4,42 do
		input[x-3] = l[x]
	end
	output = torch.Tensor(46):zero()
	lab = map[label]
	output[lab] = 1
	trainData[i] = {torch.Tensor(input), output, label, l[1]}
	training[i] = {torch.Tensor(input), output}
end


function training:size() return #training end


function testEval(data, outfile)
	outF = io.open(outfile, 'w')
	for d=1,#data do
		output = eval(data[d][1], mlp)
		corr = data[d][3]
		outF:write(output .. ' ' .. corr .. ' ' .. data[d][4] .. '\n')
	end
end

function eval(input, nn)
	x, out = torch.max(nn:forward(input),1)
	output = map2[out[1]]
	return output
end

mlp = nn.Sequential()
mlp:add(nn.Linear(39, 100))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(100,46))
mlp:add(nn.SoftMax())

trainer = nn.StochasticGradient(mlp, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(training)



testEval(testData, 'MLPout.txt')

testEval(trainData, 'MLPTrainout.txt')
