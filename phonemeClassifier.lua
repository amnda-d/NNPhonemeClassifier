require 'io'
require 'nn'


test = io.open('test.txt', 'r')
train = io.open('train.txt', 'r')
header = test:read()
header = train:read()

testData = {}
trainData = {}

local i = 0  
for line in test:lines('*l') do  
	i = i + 1
	local l = line:split(', ')
	label = l[3]
	input = {}
	for x=4,42 do
		input[x-3] = l[x]
	end
	testData[i] = {torch.Tensor(input), label, l[1]}
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
	trainData[i] = {torch.Tensor(input), label, l[1]}
end


function loadTraining(file, outputSize)
	training = {}
	f = io.open(file, 'r')
	local i = 0
	for line in f:lines('*l') do
		i = i+1
		local l = line:split(', ')
		output = torch.Tensor(outputSize):zero()
		output[l[1]] = 1
		input = {}
		for x=2,40 do
			input[x-1] = l[x]
		end
		training[i] = {torch.Tensor(input), output}
	end
	return training, #training
end



function testEval(data, outfile)
	out = io.open(outfile, 'w')
	for d=1,#data do
		output = eval(data[d][1])
		corr = data[d][2]
		out:write(output .. ' ' .. corr .. ' ' .. data[d][3] .. '\n')
	end
end


-- vowel/consonant/silence classifier
print('\ntraining vcs\n')

vcs = nn.Sequential()
vcs:add(nn.Linear(39, 70))
vcs:add(nn.Sigmoid())
vcs:add(nn.Linear(70,30))
vcs:add(nn.Sigmoid())
vcs:add(nn.Linear(30, 3))
vcs:add(nn.SoftMax())

vcsTrain, size = loadTraining('vcsTrain.txt', 3)
function vcsTrain:size() return size end

trainer = nn.StochasticGradient(vcs, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(vcsTrain)


--- VOWEL -> front/central/back
print('\ntraining fcb\n')
fcb = nn.Sequential()
fcb:add(nn.Linear(39, 70))
fcb:add(nn.Sigmoid())
fcb:add(nn.Linear(70,30))
fcb:add(nn.Sigmoid())
fcb:add(nn.Linear(30, 3))
fcb:add(nn.SoftMax())

fcbTrain, size = loadTraining('fcbTrain.txt', 3)
function fcbTrain:size() return size end

trainer = nn.StochasticGradient(fcb, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(fcbTrain)


-- VOWEL -> FRONT -> close/mid/open
print('\ntraining fcmo\n')
fcmo = nn.Sequential()
fcmo:add(nn.Linear(39, 70))
fcmo:add(nn.Sigmoid())
fcmo:add(nn.Linear(70,30))
fcmo:add(nn.Sigmoid())
fcmo:add(nn.Linear(30, 3))
fcmo:add(nn.SoftMax())

fcmoTrain, size = loadTraining('fcmoTrain.txt', 3)
function fcmoTrain:size() return size end

trainer = nn.StochasticGradient(fcmo, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(fcmoTrain)


-- VOWEL -> FRONT -> CLOSE -> ii/iy
print('\ntraining iiiy\n')
iiiy = nn.Sequential()
iiiy:add(nn.Linear(39, 70))
iiiy:add(nn.Sigmoid())
iiiy:add(nn.Linear(70,30))
iiiy:add(nn.Sigmoid())
iiiy:add(nn.Linear(30, 2))
iiiy:add(nn.SoftMax())

iiiyTrain, size = loadTraining('iiiyTrain.txt', 2)
function iiiyTrain:size() return size end

trainer = nn.StochasticGradient(iiiy, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(iiiyTrain)


-- VOWEL -> FRONT -> MID -> ei/eir
print('\ntraining eieir\n')
eieir = nn.Sequential()
eieir:add(nn.Linear(39, 70))
eieir:add(nn.Sigmoid())
eieir:add(nn.Linear(70,30))
eieir:add(nn.Sigmoid())
eieir:add(nn.Linear(30, 2))
eieir:add(nn.SoftMax())

eieirTrain, size = loadTraining('eieirTrain.txt', 2)
function eieirTrain:size() return size end

trainer = nn.StochasticGradient(eieir, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(eieirTrain)


-- VOWEL -> FRONT -> OPEN -> e/a
print('\ntraining ea\n')
ea = nn.Sequential()
ea:add(nn.Linear(39, 70))
ea:add(nn.Sigmoid())
ea:add(nn.Linear(70,30))
ea:add(nn.Sigmoid())
ea:add(nn.Linear(30, 2))
ea:add(nn.SoftMax())

eaTrain, size = loadTraining('eaTrain.txt', 2)
function eaTrain:size() return size end

trainer = nn.StochasticGradient(ea, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(eaTrain)


-- VOWEL -> CENTRAL -> close/mid/open
print('\ntraining ccmo\n')
ccmo = nn.Sequential()
ccmo:add(nn.Linear(39, 70))
ccmo:add(nn.Sigmoid())
ccmo:add(nn.Linear(70,30))
ccmo:add(nn.Sigmoid())
ccmo:add(nn.Linear(30, 3))
ccmo:add(nn.SoftMax())

ccmoTrain, size = loadTraining('ccmoTrain.txt', 3)
function ccmoTrain:size() return size end

trainer = nn.StochasticGradient(ccmo, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(ccmoTrain)


-- VOWEL -> CENTRAL -> CLOSE -> i/u
print('\ntraining iu\n')
iu = nn.Sequential()
iu:add(nn.Linear(39, 70))
iu:add(nn.Sigmoid())
iu:add(nn.Linear(70,30))
iu:add(nn.Sigmoid())
iu:add(nn.Linear(30, 2))
iu:add(nn.SoftMax())

iuTrain, size = loadTraining('iuTrain.txt', 2)
function iuTrain:size() return size end

trainer = nn.StochasticGradient(iu, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(iuTrain)


-- VOWEL -> CENTRAL -> MID -> @/@@/i@
print('\ntraining qqqi\n')
qqqi = nn.Sequential()
qqqi:add(nn.Linear(39, 70))
qqqi:add(nn.Sigmoid())
qqqi:add(nn.Linear(70,30))
qqqi:add(nn.Sigmoid())
qqqi:add(nn.Linear(30, 3))
qqqi:add(nn.SoftMax())

qqqiTrain, size = loadTraining('qqqiTrain.txt', 3)
function qqqiTrain:size() return size end

trainer = nn.StochasticGradient(qqqi, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(qqqiTrain)


-- VOWEL -> CENTRAL -> OPEN -> o/aa/ai
print('\ntraining oaa\n')
oaa = nn.Sequential()
oaa:add(nn.Linear(39, 70))
oaa:add(nn.Sigmoid())
oaa:add(nn.Linear(70,30))
oaa:add(nn.Sigmoid())
oaa:add(nn.Linear(30, 3))
oaa:add(nn.SoftMax())

oaaTrain, size = loadTraining('oaaTrain.txt', 3)
function oaaTrain:size() return size end

trainer = nn.StochasticGradient(oaa, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(oaaTrain)


-- VOWEL -> BACK -> close/mid/open
print('\ntraining bcmo\n')
bcmo = nn.Sequential()
bcmo:add(nn.Linear(39, 70))
bcmo:add(nn.Sigmoid())
bcmo:add(nn.Linear(70,30))
bcmo:add(nn.Sigmoid())
bcmo:add(nn.Linear(30, 3))
bcmo:add(nn.SoftMax())

bcmoTrain, size = loadTraining('bcmoTrain.txt', 3)
function bcmoTrain:size() return size end

trainer = nn.StochasticGradient(bcmo, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(bcmoTrain)


-- VOWEL -> BACK -> MID -> rounded/unrounded
print('\ntraining rnd\n')
rnd = nn.Sequential()
rnd:add(nn.Linear(39, 70))
rnd:add(nn.Sigmoid())
rnd:add(nn.Linear(70,30))
rnd:add(nn.Sigmoid())
rnd:add(nn.Linear(30, 2))
rnd:add(nn.SoftMax())

rndTrain, size = loadTraining('rndTrain.txt', 2)
function rndTrain:size() return size end

trainer = nn.StochasticGradient(rnd, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(rndTrain)


-- VOWEL -> BACK -> MID -> ROUNDED -> oi/oo/ou
print('\ntraining oioo\n')
oioo = nn.Sequential()
oioo:add(nn.Linear(39, 70))
oioo:add(nn.Sigmoid())
oioo:add(nn.Linear(70,30))
oioo:add(nn.Sigmoid())
oioo:add(nn.Linear(30, 3))
oioo:add(nn.SoftMax())

oiooTrain, size = loadTraining('oiooTrain.txt', 3)
function oiooTrain:size() return size end

trainer = nn.StochasticGradient(oioo, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(oiooTrain)

-- CONSONANT -> voiced/voiceless
print('\ntraining stfr\n')
stfr = nn.Sequential()
stfr:add(nn.Linear(39, 70))
stfr:add(nn.Sigmoid())
stfr:add(nn.Linear(70,30))
stfr:add(nn.Sigmoid())
stfr:add(nn.Linear(30, 3))
stfr:add(nn.SoftMax())

stfrTrain, size = loadTraining('stfrTrain.txt', 3)
function stfrTrain:size() return size end

trainer = nn.StochasticGradient(stfr, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(stfrTrain)


-- CONSONANT -> STOP -> voiced/voiceless/nasal/liquid
print('\ntraining stop\n')
stop = nn.Sequential()
stop:add(nn.Linear(39, 70))
stop:add(nn.Sigmoid())
stop:add(nn.Linear(70,30))
stop:add(nn.Sigmoid())
stop:add(nn.Linear(30, 4))
stop:add(nn.SoftMax())

stopTrain, size = loadTraining('stopTrain.txt', 4)
function stopTrain:size() return size end

trainer = nn.StochasticGradient(stop, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(stopTrain)


-- CONSONANT -> VOICED -> STOP -> NASAL -> m/n/ng
print('\ntraining mnng\n')
mnng = nn.Sequential()
mnng:add(nn.Linear(39, 70))
mnng:add(nn.Sigmoid())
mnng:add(nn.Linear(70,30))
mnng:add(nn.Sigmoid())
mnng:add(nn.Linear(30, 3))
mnng:add(nn.SoftMax())

mnngTrain, size = loadTraining('mnngTrain.txt', 3)
function mnngTrain:size() return size end

trainer = nn.StochasticGradient(mnng, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(mnngTrain)


-- CONSONANT -> VOICED -> STOP -> NONNASAL -> b/d/g
print('\ntraining bdg\n')
bdg = nn.Sequential()
bdg:add(nn.Linear(39, 70))
bdg:add(nn.Sigmoid())
bdg:add(nn.Linear(70,30))
bdg:add(nn.Sigmoid())
bdg:add(nn.Linear(30, 3))
bdg:add(nn.SoftMax())

bdgTrain, size = loadTraining('bdgTrain.txt', 3)
function bdgTrain:size() return size end

trainer = nn.StochasticGradient(bdg, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(bdgTrain)


-- CONSONANT -> FRICATIVE -> voiced/voiceless
print('\ntraining fric\n')
fric = nn.Sequential()
fric:add(nn.Linear(39, 70))
fric:add(nn.Sigmoid())
fric:add(nn.Linear(70,30))
fric:add(nn.Sigmoid())
fric:add(nn.Linear(30, 2))
fric:add(nn.SoftMax())

fricTrain, size = loadTraining('fricTrain.txt', 2)
function fricTrain:size() return size end

trainer = nn.StochasticGradient(fric, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(fricTrain)


-- CONSONANT -> FRICATIVE -> VOICED -> v/z/dh/zh
print('\ntraining vzd\n')
vzd = nn.Sequential()
vzd:add(nn.Linear(39, 70))
vzd:add(nn.Sigmoid())
vzd:add(nn.Linear(70,30))
vzd:add(nn.Sigmoid())
vzd:add(nn.Linear(30, 4))
vzd:add(nn.SoftMax())

vzdTrain, size = loadTraining('vzdTrain.txt', 4)
function vzdTrain:size() return size end

trainer = nn.StochasticGradient(vzd, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(vzdTrain)


-- CONSONANT -> VOICED -> CONTINUANT -> LIQUID/GLIDE -> l/r/w/y
print('\ntraining lrwy\n')
lrwy = nn.Sequential()
lrwy:add(nn.Linear(39, 70))
lrwy:add(nn.Sigmoid())
lrwy:add(nn.Linear(70,30))
lrwy:add(nn.Sigmoid())
lrwy:add(nn.Linear(30, 4))
lrwy:add(nn.SoftMax())

lrwyTrain, size = loadTraining('lrwyTrain.txt', 4)
function lrwyTrain:size() return size end

trainer = nn.StochasticGradient(lrwy, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(lrwyTrain)



-- CONSONANT -> VOICELESS -> STOP -> p/k/t
print('\ntraining pkt\n')
pkt = nn.Sequential()
pkt:add(nn.Linear(39, 70))
pkt:add(nn.Sigmoid())
pkt:add(nn.Linear(70,30))
pkt:add(nn.Sigmoid())
pkt:add(nn.Linear(30, 3))
pkt:add(nn.SoftMax())

pktTrain, size = loadTraining('pktTrain.txt', 3)
function pktTrain:size() return size end

trainer = nn.StochasticGradient(pkt, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(pktTrain)



-- CONSONANT -> FRICATIVE -> VOICELESS -> f/sh/th/s/h
print('\ntraining fshsh\n')
fshsh = nn.Sequential()
fshsh:add(nn.Linear(39, 70))
fshsh:add(nn.Sigmoid())
fshsh:add(nn.Linear(70,30))
fshsh:add(nn.Sigmoid())
fshsh:add(nn.Linear(30, 5))
fshsh:add(nn.SoftMax())

fshshTrain, size = loadTraining('fshshTrain.txt', 5)
function fshshTrain:size() return size end

trainer = nn.StochasticGradient(fshsh, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(fshshTrain)


-- CONSONANT -> AFFRICATE -> jh/ch
print('\ntraining fshsh\n')
jhch = nn.Sequential()
jhch:add(nn.Linear(39, 70))
jhch:add(nn.Sigmoid())
jhch:add(nn.Linear(70,30))
jhch:add(nn.Sigmoid())
jhch:add(nn.Linear(30, 2))
jhch:add(nn.SoftMax())

jhchTrain, size = loadTraining('jhchTrain.txt', 2)
function jhchTrain:size() return size end

trainer = nn.StochasticGradient(jhch, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(jhchTrain)


-- SILENCE -> breath/sil
print('\ntraining sil\n')
sil = nn.Sequential()
sil:add(nn.Linear(39, 70))
sil:add(nn.Sigmoid())
sil:add(nn.Linear(70,30))
sil:add(nn.Sigmoid())
sil:add(nn.Linear(30, 2))
sil:add(nn.SoftMax())

silTrain, size = loadTraining('silTrain.txt', 2)
function silTrain:size() return size end

trainer = nn.StochasticGradient(sil, nn.MSECriterion())
trainer.learningRate = 0.05
trainer.maxIteration = 300
trainer.learningRateDecay = .05
trainer:train(silTrain)

function eval(input)
	x, vcsOut = torch.max(vcs:forward(input), 1)

	if vcsOut[1] == 1 then
		x, fcbOut = torch.max(fcb:forward(input),1)

		if fcbOut[1] == 1 then
			x, fcmoOut = torch.max(fcmo:forward(input),1)

			if fcmoOut[1] == 1 then
				x, iiiyOut = torch.max(iiiy:forward(input),1)

				if iiiyOut[1] == 1 then
					return 'ii'
				else
					return 'iy'
				end	

			elseif fcmoOut[1] == 2 then
				x, eieirOut = torch.max(eieir:forward(input),1)

				if eieirOut[1] == 1 then
					return 'ei'
				else
					return 'eir'
				end

			else
				x, eaOut = torch.max(ea:forward(input),1)

				if eaOut[1] == 1 then
					return 'e'
				else
					return 'a'
				end
			end

		elseif fcbOut[1] == 2 then
			x, ccmoOut = torch.max(ccmo:forward(input),1)

			if ccmoOut[1] == 1 then
				x, iuOut = torch.max(iu:forward(input),1)

				if iuOut[1] == 1 then
					return 'i'
				else
					return 'u'
				end

			elseif ccmoOut[1] == 2 then
				x, qqqiOut = torch.max(qqqi:forward(input),1)

				if qqqiOut[1] == 1 then
					return 'q'
				elseif qqqiOut[1] == 2 then
					return 'qq'
				else
					return 'qi'
				end

			else
				x, oaaOut = torch.max(oaa:forward(input),1)

				if oaaOut[1] == 1 then
					return 'o'
				elseif oaaOut[1] == 2 then
					return 'aa'
				else
					return 'ai'
				end
			end

		else
			x, bcmoOut = torch.max(bcmo:forward(input),1)

			if bcmoOut[1] == 1 then
				return 'uu'
			elseif bcmoOut[1] == 2 then
				x, rndOut = torch.max(rnd:forward(input),1)

				if rndOut[1] == 1 then
					x, oiooOut = torch.max(oioo:forward(input),1)

					if oiooOut[1] == 1 then
						return 'oi'
					elseif oiooOut[1] == 2 then
						return 'oo'
					else
						return 'ou'
					end
				else
					return 'uh'
				end
			else
				return 'ow'
			end
		end

	elseif vcsOut[1] == 2 then
		x, stfrOut = torch.max(stfr:forward(input),1)

		if stfrOut[1] == 1 then
			x, stopOut = torch.max(stop:forward(input),1)

			if stopOut[1] == 1 then
				x, bdgOut = torch.max(bdg:forward(input),1)

				if bdgOut[1] == 1 then
					return 'b'
				elseif bdgOut == 2 then
					return 'd'
				else
					return 'g'
				end

			elseif stopOut[1] == 2 then
				x, pktOut = torch.max(pkt:forward(input), 1)

				if pktOut[1] == 1 then
					return 'p'
				elseif pktOut[1] == 2 then
					return 'k'
				else
					return 't'
				end

			elseif stopOut[1] == 3 then
				x, mnngOut = torch.max(mnng:forward(input),1)

				if mnngOut[1] == 1 then
					return 'm'
				elseif mnngOut[1] == 2 then
					return 'n'
				else
					return 'ng'
				end

			else
				x, lrwyOut = torch.max(lrwy:forward(input),1)

				if lrwyOut[1] == 1 then
					return 'l'
				elseif lrwyOut[1] == 2 then
					return 'r'
				elseif lrwyOut[1] == 3 then
					return 'w'
				else
					return 'y'
				end

			end

		elseif stfrOut[1] == 2 then
			x, fricOut = torch.max(fric:forward(input),1)

			if fricOut[1] == 1 then
				x, vzdOut = torch.max(vzd:forward(input),1)

				if vzdOut[1] == 1 then
					return 'v'
				elseif vzdOut[1] == 2 then
					return 'z'
				elseif vzdOut[1] == 3 then
					return 'dh'
				else
					return 'zh'
				end

			elseif fricOut[1] == 2 then
				x, fshshOut = torch.max(fshsh:forward(input),1)

				if fshshOut[1] == 1 then
					return 'f'
				elseif fshshOut[1] == 2 then
					return 'sh'
				elseif fshshOut[1] == 3 then
					return 'th'
				elseif fshshOut[1] == 4 then
					return 's'
				else
					return 'h'
				end
			end

		else
			x, jhchOut = torch.max(jhch:forward(input),1)

			if jhchOut[1] == 1 then
				return 'jh'
			else
				return 'ch'
			end
		end

	else
		x, silOut = torch.max(sil:forward(input),1)

		if silOut[1] == 1 then
			return 'breath'
		else
			return 'sil'
		end
	end
end

testEval(testData, 'out.txt')
testEval(trainData, 'trainOut.txt')
