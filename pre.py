from features import mfcc
from features import logfbank
import soundfile as sf
import os
import numpy as np
from collections import defaultdict

def getLabelBins(labelFile):
	bins = [0]
	labels = []
	for line in labelFile:
		labs = line.split()
		labels = labels + [labs[2]]
		bins = bins + [float(labs[1])]
	return labels, bins

def labelTime(time, labels, bins):
	b = np.digitize(time, bins)
	return labels[b-1]


linesDict = defaultdict(list)

files = os.listdir('test')
outfile = open('test.txt', 'w')
outfile.write('file, startTime, label\n')

for f in files:
	if f[-3:] == 'wav':
		labFile = open('test/' + f[:-3] + 'lab', 'r')
		l, b = getLabelBins(labFile)
		wav, rate = sf.read('test/' + f)
		mfccData = mfcc(wav, rate)
		time = 0.01
		mfccs = np.concatenate((mfccData[0], mfccData[1]))
		for t in mfccData[2:]:
			label = labelTime(time, l, b)
			if label == '@':
				label = 'q'
			elif label == '@@':
				label = 'qq'
			elif label == 'i@':
				label = 'iq'
			mfccs = np.concatenate((mfccs, t))
			mfccString = ', '.join([str(x) for x in mfccs])
			outString = f + ', ' + str(time) + ', ' + label + ', ' + mfccString + '\n'
			outfile.write(outString)
			time += 0.01
			mfccs = mfccs[13:]


files = os.listdir('train')
outfile = open('train.txt', 'w')
outfile.write('file, startTime, label\n')

for f in files:
	if f[-3:] == 'wav':
		labFile = open('train/' + f[:-3] + 'lab', 'r')
		l, b = getLabelBins(labFile)
		wav, rate = sf.read('train/' + f)
		mfccData = mfcc(wav, rate)
		time = 0.01
		mfccs = np.concatenate((mfccData[0], mfccData[1]))
		for t in mfccData[2:]:
			label = labelTime(time, l, b)
			if label == '@':
				label = 'q'
			elif label == '@@':
				label = 'qq'
			elif label == 'i@':
				label = 'iq'
			mfccs = np.concatenate((mfccs, t))
			mfccString = ', '.join([str(x) for x in mfccs])
			outString = f + ', ' + str(time) + ', ' + label + ', ' + mfccString + '\n'
			outfile.write(outString)
			time += 0.01
			linesDict[label] += [[label, mfccString, outString[:-1]]]
			mfccs = mfccs[13:]




# create training files

# 1=vowel, 2=consonant, 3=sil
vcs = open('vcsTrain.txt', 'w')

# vowel -> 1=front, 2=central, 3=back
fcb = open('fcbTrain.txt', 'w')

# vowel -> front, 1=close, 2=mid, 3=open
fcmo = open('fcmoTrain.txt', 'w')

# vowel -> front -> close -> 1=ii, 2=iy
iiiy = open('iiiyTrain.txt', 'w')

# vowel -> front -> mid -> 1=ei, 2=eir
eieir = open('eieirTrain.txt', 'w')

# vowel -> front -> open -> 1=e, 2=a
ea = open('eaTrain.txt', 'w')

# vowel -> central -> 1=close, 2=mid, 3=open
ccmo = open('ccmoTrain.txt', 'w')

# vowel -> central -> close -> 1=i, 2=u
iu = open('iuTrain.txt', 'w')

# vowel -> central -> mid -> 1=q, 2=qq, 3=iq
qqqi = open('qqqiTrain.txt', 'w')

# vowel -> central -> open -> 1=o, 2=aa, 3=ai
oaa = open('oaaTrain.txt', 'w')

# vowel -> back -> 1=close, 2=mid, 3=open
bcmo = open('bcmoTrain.txt', 'w')

# vowel -> back -> mid -> 1=rounded, 2=unrounded
rnd = open('rndTrain.txt', 'w')

# vowel -> back -> mid -> rounded -> 1=oi, 2=oo, 3=ou
oioo = open('oiooTrain.txt', 'w')

# consonant -> voiced -> stop -> nasal -> 1=m, 2=n, 3=ng
mnng = open('mnngTrain.txt', 'w')

# consonant -> voiced -> stop -> nonnasal -> 1=b, 2=d, 3=g
bdg = open('bdgTrain.txt', 'w')

# consonant -> voiced -> continuant -> fricative -> 1=v, 2=z, 3=dh, 4=zh
vzd = open('vzdTrain.txt', 'w')

# consonant -> voiced -> continuant -> liquid -> 1=l, 2=r, 3=w, 4=y
lrwy = open('lrwyTrain.txt', 'w')

# consonant -> voiceless -> stop -> 1=p, 2=t, 3=k
pkt = open('pktTrain.txt', 'w')

# consonant -> fricative -> 1=voiced, 2=voiceless
fric = open('fricTrain.txt', 'w')

# consonant -> fricative -> voiced -> 1=v, 2=z 3=dh 4=zh
vzd = open('vzdTrain.txt', 'w')

# consonant -> fricative -> voiceless -> 1=f 2=sh 3=th 4=s 5=h
fshsh = open('fshshTrain.txt', 'w')

# consonant -> affricate -> 1=jh 2=ch
jhch = open('jhchTrain.txt', 'w')

# silence -> 1=breath, 2=sil
sil = open('silTrain.txt', 'w')

# consonant -> 1=stop 2=fric 3=affricate
stfr = open('stfrTrain.txt', 'w')

# consonant -> stop -> 1=voiced 2=voiceless 3=nasal 4=liquid
stop = open('stopTrain.txt', 'w')

for l in linesDict['iy']:
	vcs.write('1' + ', ' + l[1] + '\n')
	fcb.write('1' + ', ' +l[1] + '\n')
	fcmo.write('1' +', ' + l[1] + '\n')
	iiiy.write('2' + ', ' +l[1] + '\n')

for l in linesDict['aa']:
	vcs.write('1' + ', ' +l[1] + '\n')
	fcb.write('2' + ', ' +l[1] + '\n')
	ccmo.write('3' +', ' + l[1] + '\n')
	oaa.write('2' + ', ' +l[1] + '\n')

for l in linesDict['qq']:
	vcs.write('1' + ', ' +l[1] + '\n')
	fcb.write('2' + ', ' +l[1] + '\n')
	ccmo.write('2' +', ' + l[1] + '\n')
	qqqi.write('2' +', ' + l[1] + '\n')

for l in linesDict['ch']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('3'+ ', ' +l[1] + '\n')
	jhch.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['ei']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('1'+ ', ' +l[1] + '\n')
	fcmo.write('2'+ ', ' +l[1] + '\n')
	eieir.write('1'+', ' + l[1] + '\n')

for l in linesDict['ai']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('3'+ ', ' +l[1] + '\n')
	oaa.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['ii']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('1'+ ', ' +l[1] + '\n')
	fcmo.write('1'+ ', ' +l[1] + '\n')
	iiiy.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['zh']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('1'+ ', ' +l[1] + '\n')
	vzd.write('4'+ ', ' +l[1] + '\n')

for l in linesDict['p']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('2'+ ', ' +l[1] + '\n')
	pkt.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['ng']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('3'+ ', ' +l[1] + '\n')
	mnng.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['breath']:
	vcs.write('3'+ ', ' +l[1] + '\n')
	sil.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['sh']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('2'+ ', ' +l[1] + '\n')
	fshsh.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['sil']:
	vcs.write('3'+ ', ' +l[1] + '\n')
	sil.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['th']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('2'+ ', ' +l[1] + '\n')
	fshsh.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['iq']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('2'+ ', ' +l[1] + '\n')
	qqqi.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['uh']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('2'+ ', ' +l[1] + '\n')
	rnd.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['q']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('2'+ ', ' +l[1] + '\n')
	qqqi.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['dh']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('1'+ ', ' +l[1] + '\n')
	vzd.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['oi']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('2'+ ', ' +l[1] + '\n')
	rnd.write('1'+ ', ' +l[1] + '\n')
	oioo.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['ow']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['eir']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('1'+ ', ' +l[1] + '\n')
	fcmo.write('2'+ ', ' +l[1] + '\n')
	eieir.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['jh']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('3'+ ', ' +l[1] + '\n')
	jhch.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['a']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('1'+ ', ' +l[1] + '\n')
	fcmo.write('3'+ ', ' +l[1] + '\n')
	ea.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['oo']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('2'+ ', ' +l[1] + '\n')
	rnd.write('1'+ ', ' +l[1] + '\n')
	oioo.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['b']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('1'+ ', ' +l[1] + '\n')
	bdg.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['e']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	fcb.write('1'+ ', ' +l[1] + '\n')
	fcmo.write('3'+ ', ' +l[1] + '\n')
	ea.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['d']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('1'+ ', ' +l[1] + '\n')
	bdg.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['g']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('1'+ ', ' +l[1] + '\n')
	bdg.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['f']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('2'+ ', ' +l[1] + '\n')
	fshsh.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['i']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('1'+ ', ' +l[1] + '\n')
	iu.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['h']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('2'+ ', ' +l[1] + '\n')
	fshsh.write('5'+ ', ' +l[1] + '\n')

for l in linesDict['k']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('2'+ ', ' +l[1] + '\n')
	pkt.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['m']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('3'+ ', ' +l[1] + '\n')
	mnng.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['l']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('4'+ ', ' +l[1] + '\n')
	lrwy.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['o']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('3'+ ', ' +l[1] + '\n')
	oaa.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['n']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('4'+ ', ' +l[1] + '\n')
	mnng.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['uu']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['s']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('2'+ ', ' +l[1] + '\n')
	fshsh.write('4'+ ', ' +l[1] + '\n')

for l in linesDict['r']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('4'+ ', ' +l[1] + '\n')
	lrwy.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['u']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('2'+ ', ' +l[1] + '\n')
	ccmo.write('1'+ ', ' +l[1] + '\n')
	iu.write('2'+ ', ' +l[1] + '\n')

for l in linesDict['t']:
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('2'+ ', ' +l[1] + '\n')
	pkt.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['w']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('4'+ ', ' +l[1] + '\n')
	lrwy.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['v']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('1'+ ', ' +l[1] + '\n')
	vzd.write('1'+ ', ' +l[1] + '\n')

for l in linesDict['y']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('1'+ ', ' +l[1] + '\n')
	stop.write('4'+ ', ' +l[1] + '\n')
	lrwy.write('4'+ ', ' +l[1] + '\n')

for l in linesDict['ou']:
	vcs.write('1'+ ', ' +l[1] + '\n')
	fcb.write('3'+ ', ' +l[1] + '\n')
	bcmo.write('2'+ ', ' +l[1] + '\n')
	rnd.write('1'+ ', ' +l[1] + '\n')
	oioo.write('3'+ ', ' +l[1] + '\n')

for l in linesDict['z']:
	vcs.write('2'+ ', ' +l[1] + '\n')
	stfr.write('2'+ ', ' +l[1] + '\n')
	fric.write('1'+ ', ' +l[1] + '\n')
	vzd.write('2'+ ', ' +l[1] + '\n')