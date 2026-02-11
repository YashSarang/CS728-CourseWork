## task_2.py

### Output

docs=64172 vocab=25014
[X built] shape=(25014, 64172) nnz=13436408 time=19.17s

[SVD] k=50
[E] shape=(25014, 50) time=2.97s
united -> [('world', 0.825598955154419), ('international', 0.8116531372070312), ('national', 0.8067892789840698), ('states', 0.8034349679946899), ('published', 0.7983080744743347)]
city -> [('town', 0.8218404650688171), ('forest', 0.7556678652763367), ('coast', 0.7536479234695435), ('council', 0.7489017248153687), ('west', 0.7456218600273132)]
president -> [('secretary', 0.9653260707855225), ('presidency', 0.9456819891929626), ('inauguration', 0.9444350004196167), ('presidential', 0.9415919184684753), ('justice', 0.9303842186927795)]

[SVD] k=100
[E] shape=(25014, 100) time=9.33s
united -> [('states', 0.8060932159423828), ('international', 0.7776795625686646), ('nations', 0.7010786533355713), ('threat', 0.689346432685852), ('germany', 0.6840161085128784)]
city -> [('mayor', 0.8678319454193115), ('pioneers', 0.8133324384689331), ('twilight', 0.8112553954124451), ('sombre', 0.8040875792503357), ('shelters', 0.8002539873123169)]
president -> [('presidency', 0.9398995041847229), ('presidential', 0.9162214994430542), ('vice', 0.9108531475067139), ('secretary', 0.9053446054458618), ('adviser', 0.8834694027900696)]

[SVD] k=200
[E] shape=(25014, 200) time=24.90s
united -> [('states', 0.8710097074508667), ('nations', 0.6586356163024902), ('international', 0.6232835054397583), ('sovereignty', 0.5616664886474609), ('cooperation', 0.5609067678451538)]
city -> [('mayor', 0.8505051732063293), ('urban', 0.7216485738754272), ('5040', 0.6968865394592285), ('downtown', 0.644932746887207), ('cities', 0.6417769193649292)]
president -> [('presidency', 0.9144443273544312), ('vice', 0.891502857208252), ('bush', 0.769262433052063), ('adviser', 0.7647198438644409), ('secretary', 0.7609491944313049)]

[SVD] k=300
[E] shape=(25014, 300) time=35.36s
united -> [('states', 0.8706765174865723), ('terrain', 0.5266706943511963), ('nations', 0.5257905721664429), ('overran', 0.47169092297554016), ('ham', 0.4575392007827759)]
city -> [('mayor', 0.8002616167068481), ('urban', 0.6657459139823914), ('5040', 0.6296809315681458), ('cities', 0.5823534727096558), ('downtown', 0.5818690061569214)]
president -> [('presidency', 0.9003404378890991), ('vice', 0.8576797842979431), ('bush', 0.7317122220993042), ('pence', 0.7215834856033325), ('administration', 0.6825240850448608)]

## Task_3

NER labels: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

CRF Feature Set (inventory):

- bias
- w
- w.lower
- isupper
- istitle
- islower
- isdigit
- has_digit
- has_hyphen
- shape
- pref1
- pref2
- pref3
- pref4
- suf1
- suf2
- suf3
- suf4
- BOS
- EOS
- -1:w.lower
- -1:istitle
- -1:isupper
- -1:shape
- +1:w.lower
- +1:istitle
- +1:isupper
- +1:shape
  [dev] c1=0.1 c2=0.1 acc=0.9790 macroF1=0.8930
  [dev] c1=0.05 c2=0.1 acc=0.9790 macroF1=0.8926
  [dev] c1=0.1 c2=0.05 acc=0.9787 macroF1=0.8925
  [dev] c1=0.0 c2=0.1 acc=0.9793 macroF1=0.8930
  [dev] c1=0.1 c2=0.0 acc=0.9779 macroF1=0.8893
  BEST CONFIG: {'c1': 0.1, 'c2': 0.1} dev_macroF1: 0.8930033479013182

[TEST] acc=0.9614 macroF1=0.8212

Top state features (by |weight|):
+5.3377 B-ORG -1:w.lower:v
+5.3132 I-LOC -1:w.lower:colo
+5.3110 I-LOC -1:w.lower:wisc
+5.2667 O has_digit
+4.7820 O BOS
+4.6938 B-PER +1:shape:dx
+4.6414 B-ORG BOS
+4.5994 B-PER BOS
+4.4004 B-ORG +1:w.lower:v
-4.2961 O +1:w.lower:arose
+4.2929 B-ORG +1:w.lower:arose
-4.0384 O -1:w.lower:lloyd
+4.0360 B-PER -1:w.lower:b
+3.9627 B-ORG +1:w.lower:yr
+3.8089 B-LOC BOS
+3.7987 B-LOC +1:shape:d-d-d
+3.7531 O islower
+3.6649 O suf3:day
+3.5622 B-LOC -1:w.lower:at
+3.5305 B-ORG suf4:hire
+3.4841 B-MISC suf3:ian
+3.4827 O shape:x
+3.4618 B-LOC -1:w.lower:near
+3.4053 I-ORG -1:w.lower:auth
-3.3166 I-LOC +1:shape:d
+3.2373 B-MISC BOS
+3.2313 I-ORG -1:w.lower:assoc
+3.2243 I-ORG -1:w.lower:fac
+3.1703 I-ORG -1:shape:x
+3.1636 I-ORG -1:w.lower:cdu
+3.1598 B-PER +1:shape:d-d-d-d
+3.1163 I-ORG -1:w.lower:moody
+3.0879 B-MISC shape:X$
+3.0795 B-ORG w:interior
+3.0642 I-ORG -1:w.lower:lloyd
-3.0405 O -1:w.lower:moody
+3.0244 O bias
+3.0111 I-LOC -1:w.lower:lord
+3.0066 B-PER -1:w.lower:minister
+2.9707 I-ORG -1:w.lower:mladost
