??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
dcn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*#
shared_namedcn/dense_2/kernel
z
&dcn/dense_2/kernel/Read/ReadVariableOpReadVariableOpdcn/dense_2/kernel*
_output_shapes
:	?*
dtype0
x
dcn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedcn/dense_2/bias
q
$dcn/dense_2/bias/Read/ReadVariableOpReadVariableOpdcn/dense_2/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *%
shared_nameembedding/embeddings
~
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes
:	? *
dtype0
?
embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameembedding_2/embeddings
?
*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes

: *
dtype0
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:< *
dtype0
}
dcn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`?*!
shared_namedcn/dense/kernel
v
$dcn/dense/kernel/Read/ReadVariableOpReadVariableOpdcn/dense/kernel*
_output_shapes
:	`?*
dtype0
u
dcn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedcn/dense/bias
n
"dcn/dense/bias/Read/ReadVariableOpReadVariableOpdcn/dense/bias*
_output_shapes	
:?*
dtype0
?
dcn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*#
shared_namedcn/dense_1/kernel
{
&dcn/dense_1/kernel/Read/ReadVariableOpReadVariableOpdcn/dense_1/kernel* 
_output_shapes
:
??*
dtype0
y
dcn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namedcn/dense_1/bias
r
$dcn/dense_1/bias/Read/ReadVariableOpReadVariableOpdcn/dense_1/bias*
_output_shapes	
:?*
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name115*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name137*
value_dtype0	
m
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name159*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dcn/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/dcn/dense_2/kernel/m
?
-Adam/dcn/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dcn/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dcn/dense_2/bias/m

+Adam/dcn/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *,
shared_nameAdam/embedding/embeddings/m
?
/Adam/embedding/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/m*
_output_shapes
:	? *
dtype0
?
Adam/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameAdam/embedding_2/embeddings/m
?
1Adam/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/m*
_output_shapes

: *
dtype0
?
Adam/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *.
shared_nameAdam/embedding_1/embeddings/m
?
1Adam/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/m*
_output_shapes

:< *
dtype0
?
Adam/dcn/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`?*(
shared_nameAdam/dcn/dense/kernel/m
?
+Adam/dcn/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense/kernel/m*
_output_shapes
:	`?*
dtype0
?
Adam/dcn/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dcn/dense/bias/m
|
)Adam/dcn/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dcn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/dcn/dense_1/kernel/m
?
-Adam/dcn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_1/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dcn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dcn/dense_1/bias/m
?
+Adam/dcn/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_1/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dcn/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?**
shared_nameAdam/dcn/dense_2/kernel/v
?
-Adam/dcn/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dcn/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/dcn/dense_2/bias/v

+Adam/dcn/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *,
shared_nameAdam/embedding/embeddings/v
?
/Adam/embedding/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding/embeddings/v*
_output_shapes
:	? *
dtype0
?
Adam/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameAdam/embedding_2/embeddings/v
?
1Adam/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_2/embeddings/v*
_output_shapes

: *
dtype0
?
Adam/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:< *.
shared_nameAdam/embedding_1/embeddings/v
?
1Adam/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/embedding_1/embeddings/v*
_output_shapes

:< *
dtype0
?
Adam/dcn/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`?*(
shared_nameAdam/dcn/dense/kernel/v
?
+Adam/dcn/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense/kernel/v*
_output_shapes
:	`?*
dtype0
?
Adam/dcn/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dcn/dense/bias/v
|
)Adam/dcn/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dcn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameAdam/dcn/dense_1/kernel/v
?
-Adam/dcn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_1/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dcn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameAdam/dcn/dense_1/bias/v
?
+Adam/dcn/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dcn/dense_1/bias/v*
_output_shapes	
:?*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
??
Const_3Const*
_output_shapes	
:?*
dtype0*??
value??B???B 1199 - GHC HMO NON BARB Access PPO HSA $1,400B Access PPO HSA $2,500B Core HMO $1,000B Core HMO $350B Core HMO HSA $1,500B Critical IllnessB DX Lab/X-Ray RiderB
 HSA $2500B	 PSFA HRAB	 PSFA PPOB
 Plan $100B Plan $1000B Plan $1500B
 Plan $200B Plan $2000B	 Plan $50B
 Plan $500B Plan $500 - ILB
 Plan $750B UPFFA HDHP $2000B UPFFA HDHP $2000 (Fairfield)B UPFFA POS PlanB$10 Copay PlanB	$1000 HMOB	$1000 MaxB$1000 Max VolB	$1000 PPOB$1000 PPO - Buy Up PlanB$100K Basic Life AD&DB$10K CH Basic Life #198B$125K Basic LifeB$12K Basic Life AD&DB$15 Copay PlanB$150K Basic LifeB$15K Basic Life AD&DB	$2000 MaxB$2000 Max VolB$2000 PPO - Mid PlanB$200K Basic LifeB$20K Basic Life AD&DB$225K Basic LifeB$24K Basic Life AD&DB	$2500 HMOB$2500 HMO H.S.AB	$2500 PPOB$2500 PPO H.S.AB$25K Basic LifeB$25K Basic Life AD&DB	$3000 PPOB$3500 HDHP - Base PlanB$36K Basic Life AD&DB$40K Basic Life AD&DB$4500 HSA HMOB$4500 HSA PPO + HRAB$48K Basic Life AD&DB	$5000 HMOB$5000 HMO + HRAB	$5000 PPOB$5000 PPO + HRAB$50K Basic LifeB$50K Basic Life AD&DB$50K Basic Life AD&D #198B$50K SP Basic Life #198B(FSA) MLP (Lim. Purp.)B	(FSA) MRAB1-3 Sessions EAPB1-5 Sessions EAPB1-8 Sessions EAPB1199 - GH - Access PPOB15K Basic Life & ADDB15K Basic Life & ADD S2&3B15k Basic Life & ADDB15k LifeB1x Salary Life $100K MaxB1x Salary Life $150K MaxB1x Salary Life $50K MaxB20k Flat Life AD&DB25K Basic Life & ADDB25k LifeB2x Salary Basic Life NO ADDB2x Salary Basic Life w/ ADDB2x Salary Life $200K MaxB3 Visit EAPB3 Visits ModelB40k LTC RiderB50k LifeB6 Visit EAPB6 Visits ModelBA250B
A250 PRIMEB
A250 PrimeB
ACA DentalBACA MedicalBACA PEPMBACA ReportingB
ACA VisionBAD & DBAD&DB	AD&D $25KB	AD&D $50KBAD&D Non Ins Fees Plan 1BAD&D Non Ins Fees Plan 2BAHN 250 – MultiCare WellCityB$AHN 250 – U of W Medicine WellCityBAK CLASSIC 1500B	AK DentalBAK Dental Buy UpBAK PPO 2500BAWB 1BAWB 2BAWB 4BAWB Exam PlusBAWB Exam Plus 2019BAWB Hardware 1BAWB Hardware 1 2019BAWB Hardware 2BAWB Hardware 2 2019BAWC HEALTHFIRST 250 PLANBAWC HealthFirst 250BAWC HealthFirst 250 WellCityBAWC HealthFirst 500BAWC HealthFirst 500 WellCityB
Access PPOBAccess PPO  WellCityBAccess PPO $3,000BAccess PPO $350BAccess PPO $750BAccess PPO 1000BAccess PPO 1500 HSABAccess PPO 200BAccess PPO 2000BAccess PPO 3000BAccess PPO 500BAccess PPO 5000BAccess PPO Basic - QualstarBAccess PPO Bronze HSABAccess PPO Silver HSAB Access PPO VisitsPlus Gold HD LXBAccess PPO VisitsPlus Gold LXB!Access PPO VisitsPlus Platinum LXB"Access PPO VisitsPlus Silver LD LXBAccess PPO VisitsPlus Silver LXB#Access PPO VisitsPlus Silver LX - EB$Access PPO VisitsPlus Silver LX - EOBAccident Insurance HighBAccident Insurance LowBAccident Plan 1BAccident Plan 2BAccident Plan 3BAccident Plan 4BAccident- High PlanBAccident- Low PlanBActives Base PlanBActives Buy-up PlanB	Admin FeeBAdmin Fee 2021BAero Basic Life & ADDBAero Dependent Life InsuranceBAero Dependent Supplemental ADDBAero LTDBAero STDBAero Spouse Supplemental LifeBAero Supplemental EE ADDBAero Supplemental LifeBAero Supplemental Spouse ADDBAetna DentalBAetna MedicalBAflacBAlt Care Rider 25/1500BAlternative Care Option 1 $15BAlternative Care Option 2 $25BAnchorage AKBAnchorage HSA Plan $2000BAnchorage Plan $2000BAnthem Blue Cross HMO 5093BAnthem Select HMO 5063BAnthem Trad HMO 5091BAnthem Trad HMO 5663BB500B
B500 PRIMEB	B500 PlusB
B500 PrimeBBCBS Copay 750BBCBS Core 500BBCBS HDHP 1500BBCBS HDHP 2500BBCBS HDHP 5000BBL - $10,000 No AD&DBBL - $10,000 W/AD&DBBL - $100,000 No AD&DBBL - $100,000 W/AD&DBBL - $110,000 No AD&DBBL - $12,000 No AD&DBBL - $12,000 W/AD&DBBL - $12,500 No AD&DBBL - $15,000 No AD&DBBL - $15,000 W/AD&DBBL - $150,000 W/AD&DBBL - $17,000 No AD&DBBL - $20,000 No AD&DBBL - $20,000 W/AD&DBBL - $200,000 W/AD&DBBL - $24,000 No AD&DBBL - $24,000 W/AD&DBBL - $25,000 No AD&DBBL - $25,000 W/AD&DBBL - $250,000 W/AD&DBBL - $26,000 W/AD&DBBL - $30,000 W/AD&DBBL - $30,000/No AD&DBBL - $300,000 No AD&DBBL - $300,000 W/ AD&DBBL - $35,000 W/AD&DBBL - $36,000 W/AD&DBBL - $40,000 No AD&DBBL - $40,000 W/AD&DBBL - $47,000 W/AD&DBBL - $48,000 No AD&DBBL - $5,000 No AD&DBBL - $5,000 W/AD&DBBL - $50,000 No AD&DBBL - $50,000 W/AD&DBBL - $60,000 No AD&DBBL - $60,000 W/AD&DBBL - $75,000 W/AD&DBBL - $8,000 W/AD&DBBL - $80,000 W/AD&DBBLADDBBSI FlatBBase DentalBBase HMO + HRAB	Base PlanBBase VisionBBase Vision - CustomBBasicB
Basic AD&DBBasic AD&D - HourlyBBasic AD&D - SalaryBBasic Dental 750BBasic Dental PlanBBasic Dependent LifeB
Basic LifeBBasic Life $100KBBasic Life $200KB+Basic Life $300K (Custom Plan for Florence)BBasic Life $50KBBasic Life & AD&DBBasic Life & AD&D - Class 1B!Basic Life & AD&D - Classes 2 & 3BBasic Life & ADDBBasic Life & Ad&dBBasic Life - HourlyBBasic Life - SalaryBBasic Life - UndeadBBasic Life / AD&DBBasic Life/AD&DBBasic Life/AD&D 200K 09B
Basic PlanBBasic VisionBBerkeley SpecialBBlue Shield Access+ 5251BBlue View VisionBBridgeHealth Add-OnBBronze 8150 PREFBBronze 8550 PREFBBronze Care on Demand 5500 PREFBBronze Essential 7500 PREFB
Bronze HMOBBronze HSA 5000 PREFBBronze HSA 5500 PREFBBronze PlanBBuy Up - Access PPO 1000BBuy Up - Access PPO 2500BBuy-Up PlanBBuyUp PPO + HRABC750BCDHPBCDHP Admin Fee - 2019BCDHP Admin Fee - 2021BCDHP Admin Fee -2020BCDHP Admin Fee -2020-2021BCDHP Admin Fee 2020BCDHP Admin Fee 2021BCDHP FeeBCHOICE 2BCL99BCOM AK Classic 1500BCT - Durable 1500+BCT - Durable 3500+BCT - Durable 500+BCT - Durable 6000+BCT - Durable 8000+BCT - H.S.A. 2000+BCT - H.S.A. 2000PBCT - H.S.A. 3000+BCT - H.S.A. 3000PBCT - H.S.A. 5000+BCT - H.S.A. 5000PBCT - POSBCT - Sustainable 1000+BCT - Sustainable 1000PBCT - Sustainable 1500+BCT - Sustainable 200+BCT - Sustainable 2000+BCT - Sustainable 2000PBCT - Sustainable 250+BCT - Sustainable 2500+BCT - Sustainable 250PBCT - Sustainable 3000+BCT - Sustainable 3000PBCT - Sustainable 500+BCT - Sustainable 5000+BCT - Sustainable 5000PBCT - Sustainable 500PBCT - Sustainable 750+BCadillac PlanBCall a Doctor PlusBCancer CoverageBChild AD & DB
Child ADnDBChild Basic Life/AD&DB
Child LifeBChild Life and AD&D $10kBChild Life and AD&D $2kBChild Life and AD&D $4kBChild Life and AD&D $6kBChild Life and AD&D $8kBChild Only Ortho RiderBChild Vol AD&DBChild Vol Life/AD&DBChild Voluntary LifeBChild Voluntary Life/AD&DBChild(ren) Vol LifeBChiro / Acu ExpandedBChiropractic ExpandedB)Chocie $50/40%/50%/$8550/$2800sd/$70/$250BChoiceB(Choice $15/20%/40%/$4000/$500sd/$30/$250B)Choice $30/25%/50%/$8200/$1500sd/$50/$250BChoice 1BChoice 2BChoice 3BChoice 4BChoice ABChoice A wSafety CoPayBChoice BBChoice B w/Pro TecBChoice B wSafety CoPayBChoice CBChoice C w/Pro TecBChoice C wSafety CoPayB(Choice$50/50%/50%/$8550/$7000sd/$85/$250BCigna MedicalBCity ContributionBClassic $1000 w/ RiderBClassic $2000BClassic $2000 w/ RiderBClassic $3000BClassic $3000 w/ RiderBClassic $4000 30/50BClassic 1000BClassic 1500BClassic 250 GANIR 2021BClassic 500BClassic 5000 OOPMBComPsych EAPBComprehensiveB*Connect $50/40%/50%/$8550/$2800sd/$70/$250BContributions $100/MonthBContributions $150/MonthBContributions $200/MonthBContributions $250/MonthBContributions $300/MonthBContributions $400/MonthBContributions $450/MonthBContributions $50/MonthBContributions $750/MonthBCore Bronze HSAB	Core GoldBCore HMO - QualstarBCore HMO 2000BCore HMO 250BCore HMO 2500BCore HMO 750BCore SilverBCore Silver HSABCore VisitsPlus Gold HD LXBCore VisitsPlus Gold LXBCore VisitsPlus Platinum LXBCore VisitsPlus Silver LXBCore VisitsPlus Silver LX - EOB
Create PPOBCritical IllnessBCritical Illness $15KBCritical Illness $30KBCritical Illness - $10kBCritical Illness - $20kBCritical Illness - $30kBCritical Illness HighBCritical Illness LowBCritical Illness Plan 1BCritical Illness Plan 2BCritical Illness Plan 3BCritical Illness Plan 4BD1000 PRIMEB
D1000 PlusBD1000 PrimeBD1500 PRIMEB
D1500 PlusBD1500 PrimeBD2000 PRIMEB
D2000 PlusBD2000 PrimeBD2500 PRIMEB
D2500 PlusBD2500 PrimeB
D750 PRIMEB	D750 PlusB
D750 PrimeBDCAPBDCAP - 2019BDCAP - 2021BDEP Voluntary LifeB	DHMO HighBDHMO High OptionBDeductible - LEOFF 2BDeductible ReimbursementBDelta DentalB$Delta Dental - (Plan F/Ortho 5) (FI)BDelta Dental - BARGAINBDelta Dental - CoreBDelta Dental - EnhancedBDelta Dental - Leoff 1BDelta Dental - Leoff 2BDelta Dental - MTBDelta Dental - NONBARGAINBDelta Dental - PlusBDelta Dental AdminBDelta Dental VoluntaryBDentalBDental $1000BDental $1500BDental $1500 with ORTHOBDental $2000BDental $2000 with ORTHOBDental - Delta DentalBDental - High OptionBDental - Low OptionBDental - RMT JoreBDental -Willamette DentalBDental 1000BDental 1000 + OrthoBDental 1500BDental 1500 + OrthoBDental 1500w/OrthoBDental 2000BDental 2000 + OrthoBDental 2000w/OrthoBDental 2500BDental 2500 + OrthoB
Dental 500BDental Buy UpBDental High - ERBDental High - ER + OrthoBDental High - VOLBDental High - VOL + OrthoB
Dental LowBDental Low - ERBDental Low - VOLBDental Med - ERBDental Med - ER + OrthoBDental Med - VOLBDental Med - VOL + OrthoBDental Non Ins Fees Plan 1BDental Non Ins Fees Plan 2BDental Non Ins Fees Plan 3B
Dental PPOBDental PPO SelectBDental PlanBDental Plan 1BDental Plan 2BDental Plan 3BDental Plan 4BDental Plan IBDental Plan I w/OrthoBDental Plan IVBDental Plan IV w/OrthoBDental Plan VolBDental PlusBDental PremierBDental SelectBDep LifeBDep Life - 10KBDep Vol Life/AD&DBDep Voluntary AD&DBDep Voluntary TermBDependent BLADDBDependent Basic LifeBDependent Care - DCABDependent LifeBDependent Life 1BDependent Life 2BDependent Life 3BDependent Life 4BDependent Life InsuranceBDependent Voluntary AD&DBDependent Voluntary TermBDiscountB
E3000 PlusBE3000 PrimeBEAPBEAP 1-6 VisitBEAP 3 VisitBEAP Basic - 3 (Core)BEAP Enhanced - 5 (Buy-Up)BEAP PlanBEE Supplemental LifeBEE Vol Life - Non SmokerBEE Vol Life - SmokerBEE Vol Life and AD&DBER PAID HIGH- 5P435BER PAID LOW- 5P432BER PAID MED- 5P433BER Paid LTD 50 - 90DYBERHCBERHC EmployeeBEasyOptionsBElect PPO VisitsPlus Silver LXBElite 1010-1BEmployee Vol AD&DBEmployee Vol Life/AD&DBEmployee Voluntary AD&DBEmployee Voluntary LifeBEmployee Voluntary TermBEngageBEngage EmployeeBEnhancedBEnhanced + CVCBEnhanced EAPBEnhanced PPOBEnhanced PPO w/ OrthoBEnhanced PlanBEnhanced Plan C1BEnhanced Plan C2BEnhanced Plan D3BEnhanced Plan D4BEnhanced Plan E3BEnhanced+CVCBEverMed DPCB	Exam PlusBExam Plus (Div 8) >10 - 2BExam Plus -   Div 3BExpressions Plan 1BExpressions Plan 2B
F5000 PlusBF5000 PrimeBFSA MedicalBFSA Medical - 2019BFSA Medical - 2020BFSA Medical - 2020-2021BFSA Medical - 2021BFSA Medical -2020BFSA Medical -2021BFSA Medical 2020BFSA Medical 2021BFamily Ortho RiderBFamily Ortho Rider $2000BFamily Vol AD&DBFirst Health PPO NetworkBFully Insured OptionB
G6000 PlusBG6000 PrimeBGF Choice 1BGF Choice 2BGF Choice 3ABGF Choice 3BBGF Choice 4BGF Secure 1000BGF Solutions 1500BGF Solutions 500BGH - Montana PlanBGS50-50-5000BGSA15-250-2-4000BGSA20-500-2-5000BGSA25-1000-2-6000BGSCC3T10-500-2-5000DXBGSCC3T15-1000-2-6000DXBGSCC3T15-1000-3-6000ESBGSCC3T20-2000-3-6500ESBGSCC3T35-3000-3-7350ESBGSCC3T35-5000-3-7350ESBGSCC3T50-5000-5-7350ESBGSE25-1000-2-6000BGSE30-2000-2-6600BGSE35-3000-2-6600BGSE35-3000-2-7350BGSE35-5000-2-7350BGSE50-5000-5-7350BGSFE35-3000-2-6600BGSFE35-5000-2-7350BGSHDE655010050BGTL $10KBGTL $20KBGold 1000 PREFBGold 1000 PREF VBGold 1500 PREFBGold 2000 PREFBGold 2000 PREF VBGold 2500 PREFBGold 2500 PREF VBGold 500 PREFBGold 500 PREF VBGold Access PPOBGold HSA 1500 PREFBGold PPO MedicalBGold PPO Medical UtahB	Gold PlanBGroup AccidentBGroup Life & ADDBH.S.A $1250BH.S.A. $1500+BH.S.A. $3000BH.S.A. 1500 +BH.S.A. 1500 PrimeBH.S.A. 2500 +BH.S.A. 2500 PrimeBH.S.A. 3500 +BH.S.A. 3500 PrimeBH.S.A. 5000 +BH.S.A. 5000 PrimeBHDHPBHDHP WellCityBHDHP/HSA PlanB	HHSA 2500B	HHSA 4000BHMO $200BHMO $500BHMO - H S A PlanBHMO - Jade PlanBHMO - Onyx PlanBHMO - Pearl PlanBHMO - Topaz PlanBHMO - Zircon PlanBHMO 200BHMO 2000BHMO 3000BHMO 5000BHMO 750BHPE 5000B
HRA - 2019B	HRA -2020B	HSA $1500B
HSA $1500+B
HSA $1500PB	HSA $2800B
HSA $3000+B
HSA $3000PB	HSA $3500B	HSA $4500B
HSA $4500+B
HSA $4500PBHSA 100 $6900BHSA 100 - 3500BHSA 100 - 6900BHSA 1500B
HSA 1500 +BHSA 1500 PlusBHSA 1500 PrimeBHSA 1700BHSA 20%BHSA 2000BHSA 2000 UTAHBHSA 2000 with MotionBHSA 2500B
HSA 2500 +BHSA 2500 PlusBHSA 2500 PrimeBHSA 2500PlusBHSA 2800BHSA 3.0 Preferred 5000BHSA 3.0 Preferred-1500/3000BHSA 3.0 Preferred-3500/7000BHSA 3.0 Preferred-OptimumBHSA 30%BHSA 30%/50%/$6750/$2500BHSA 30%/50%/$6750/$3500BHSA 3000BHSA 3200BHSA 3450BHSA 3500B
HSA 3500 +BHSA 3500 PlusBHSA 3500 PrimeB	HSA 3500PBHSA 4000BHSA 4500BHSA 50%/50%/$6750/$5000BHSA 5000B
HSA 5000 +BHSA 5000 with MotionBHSA 5500 PlusBHSA 5500 PrimeBHSA 5500PrimeBHSA 6000BHSA 80 $2800BHSA 80 $3500BHSA 80 - 1500BHSA 80 - 3000BHSA 80 - 4500BHSA ABHSA BBHSA Bank AccountBHSA CYC CompanionBHSA CYC Companion - 2021BHSA HMOBHSA Medical PlanBHSA Medical Plan UtahBHSA Over 55 - 2021BHSA Under 55BHSA Under 55 - 2021BHSA1350 PrimeBHSA1400 PRIMEBHSA1700 PRIMEBHSA1700 PlusBHSA1700 PrimeBHSA2500 PRIMEBHSA2500 PlusBHSA2500 PrimeBHSA3500 PlusBHSA3500 PrimeBHSA5500T PlusBHSA5500T PrimeB
Hardware 1B
Hardware 2BHealth Net SmartCare 5283BHearingBHeritage Plan P2BHeritage Plan P3BHeritage Plan P4BHeritage Plan P5BHeritage Plan P6BHeritage Plan P7BHeritage Plan P7ABHeritage Plan P7BBHeritage Plan P8-HSABHeritage Plan P9-HSABHeritage Plus Plan C-1500B	High PlanBHospital Indemnity - HighBHospital Indemnity - LowBHospital Indemnity HighBHospital Indemnity LowBHospital Indemnity Plan 1BHospital Indemnity Plan 2BHospital Indemnity Plan 3BHospital Indemnity Plan 4BIdentity Guard - PremierBIdentity Guard - TotalBIdentity Guard - UltimateBIdentity Guard - ValueBIncentive Plan 1BInnova $250BIsland County Base PlanBIsland County Buy-Up D3B
J8150 PlusBJ8150 PrimeBKP HSA Plan 5B	KP Plan 2B	KP Plan 3B	KP Plan 4BKP Plan B- Basic Life/AD&DBKP Plan D- Basic Life/AD&DBKaiser & Sr Adv 53910BKaiser & Sr Adv 53912BKaiser & Sr Adv 5399B
Kaiser 200BKaiser 200 WellCityB
Kaiser 500BKaiser Base PlanBKaiser COMBO Sr/HMO 5397BKaiser DentalBKaiser DirectBKaiser HMO Basic 5331BKaiser HMO Basic 5332BKaiser Sr Adv +1 w/Dent 5425BKaiser Sr Adv Reg 1 - 5364BKaiser Sr Adv Reg 1 5365BKaiser Sr Adv Reg 2 - 5374BKaiser Sr Adv w/Dent 5424BKasier Buy Up PlanBLS - H.S.A 5000+BLS - H.S.A 5000PBLS - H.S.A. 2000+BLS - H.S.A. 3000+BLS - PLAN A+BLS - PLAN B+BLS - PLAN BPBLS - PLAN C+BLS - PLAN CPBLS - PLAN D+BLS - PLAN E+BLS - PLAN F+BLS - PLAN G+BLS - PLAN H+BLS - PLAN I+BLS - PLAN IPBLS - PLAN J+BLS - PLAN K+BLS - PLAN L+BLS - PLAN LPBLTDBLTD - HourlyBLTD - SalaryBLTD BaseBLTD Base (Stand Alone)BLTD Base - Employer PaidB
LTD Buy UPB
LTD Buy UpB
LTD Buy-UpBLTD Buy-Up (Stand Alone)BLTD Buy-Up with MedicalBLTD CoreB#LTD Low Risk Option 1:  60%; 90-dayB$LTD Low Risk Option 2:  60%; 180-dayB#LTD Low Risk Option 3:  67%; 90-dayBLTD Option 1:  60%; 90-dayBLTD Option 2:  60%; 180-dayBLTD Option 3:  67%; 90-dayBLTD Option 4:  67%; 180-dayB
LTD Plan 1BLTD Plan 1 180 DayBLTD Plan 1 90 DayB
LTD Plan 2BLTD Plan 2 180 DayBLTD Plan 2 90 DayB
LTD Plan 3BLTD Plan 3 180 DayBLTD Plan 3 90 DayB
LTD Plan 4BLTD Plan 4 90 DayB
LTD Plan 5B
LTD Plan 6BLTD99BLeave ManagementBLifeB	Life $15KB	Life $30KB	Life $50KBLife & ADD SalaryBLife / AD&DBLife / AD&D $10KBLife / AD&D $15KBLife / AD&D $25KBLife / AD&D $50KBLife AD&D $10KBLife AD&D $10K Buy-UpBLife AD&D $20K Buy-UpBLife AD&D $30KBLife AD&D $40K Buy-UpBLife AD&D Additional $15KBLife AD&D Plan 1BLife AD&D Plan 2BLife AD&D Plan 3BLife AD&D Plan 4BLife Balance CardBLife Option 2B	Life PlanBLife and AD&DBLife/AD&D - Flat $10KBLife/AD&D - Flat $5KBLife/AD&D - Salary-BasedBLife/AD&D Plan 1 - 300KBLife/AD&D Plan 2 - 300KBLife/AD&D Plan 2.5 - 400KBLife/AD&D Plan 3 - 500KBLife/AD&D Plan 5 - 50KBLife/AD&D Plan 6 - 25KBLife/ADD: $10kBLife/ADD: $15kBLifeBalanceBLincoln DentalBLong Term DisabilityBLong Term Disability - Class 1BMEC Admin FeeBMEC Admin Fee 1BMEC Broker Fee 1BMEC COBRA Fee 1BMEC Medical PlanBMEC OperatingBMEC PCORI DRCBMEC Ternian Fee 1B
MED Plan 1B	MP1500RX1BMaterials Only Plan BB
Med Plan 1B
Med Plan 2B
Med Plan 3B
Med Plan 4BMedicalBMedical - BronzeBMedical - GoldBMedical - PPO1STCHOICEBMedical - PPOCNXUSBMedical - PPOPHCSBMedical - SilverBMedical Non Ins Fees Plan 1BMedical Non Ins Fees Plan 2BMedical Non Ins Fees Plan 3BMedical Non Ins Fees Plan 4BMedical Plan 1BMedical Plan 2BMedical Savings PlanBMedical Savings Plan 100BMedical Savings Plan 125BMedical Savings Plan 140BMedical Savings Plan 150BMedical Savings Plan 175BMedical Savings Plan 200BMedical Savings Plan 225BMedical Savings Plan 250BMedical Savings Plan 300BMedical Savings Plan 50BMedical Savings Plan 75BMedical WaiverBMesher Dent w/OrthoBMet Legal - HighBMet Legal - LowBMetLifeBMetLife AccidentBMetLife Hospital IndemnityBMetLife Short Term DisabilityB"MetLife Spouse Life and AD&D $250kB%MetLife Voluntary Life and AD&D $500kBMomentum 1000 20BMomentum 1000 20 RAUBMomentum 2000 20BMomentum 2000 20 #337BMomentum 2000 20 #338BMomentum 2000 30BMomentum 3000 20BMomentum 3000 30BMomentum 3000 30 RAUBMomentum 3500 30 AC25BMomentum 500 20BMomentum 5000 20BMomentum 5000 30BMomentum 5000 50BMomentum 5000 50 RAUBMomentum 8550BMomentum HSA 4000BMomentum HSA 5250BMomentum HSA 6850BMomentum HSA 7000BNavigate 1750BNavigate 3500BNon-Ortho Incentive 1BNon-Pooled GroupsBNon-Trust 1-3 Sessions EAPBNon-Trust 1-5 Sessions EAPBNon-Trust 1-8 Sessions EAPBNon-Trust Product AdminBOLD $100/MonthB(OP ADV $15/30%/50%/$4000/$500cd/$25/$250B)OP ADV $20/30%/40%/$7500/$1500sd/$60/$250B)OP ADV $40/35%/50%/$8550/$3500sd/$60/$250B)OP ADV $40/35%/50%/$8550/$4500sd/$40/$250B)OP ADV $40/40%/50%/$8550/$2500sd/$60/$250BOption IBOption I w. Ortho IIBOption I w. Ortho IIIB	Option IIBOption II w. Ortho IBOption II w. Ortho IIIB
Option IIIBOption III w. Otho IIIB	Option IVBOption VB	Option VIBOptions 200BOptions 500B	Options ABOrthoBOrtho $1000BOrtho $2000BOrtho Plan 1 - ChildBOrtho Plan 2 - FamilyBOut-of-Area PlanBPEAK 0/2000BPEAK 100/2500BPEAK 1500/5000BPEAK 500/4000BPERS Choice 3242BPERSCare & MedSupp COMBO 5729BPERSCare 3241BPERSCare Med Supp +1 Reg 1-5695BPERSCare Med Supp Out 3394BPERSCare Med Supp Reg 1 - 5694BPERSCare Med Supp Reg 3 - 5711BPERSCare Out of State 3292BPERSChoice - 5482BPERSChoice Basic Reg 1 - 5481BPERSChoice MSupp +1 Reg1 - 5515BPERSChoice Med Supp Reg1 - 5514BPERSChoice Supp +1 Out 3345BPERSChoice Supp Out 3344BPERSChoice/Med Sup COMBO 5548B	PHSA 2500B	PHSA 4500BPLAN AB'PLAN A w/ Ortho Plan I (Dep Only Ortho)B(PLAN A w/ Ortho Plan II (Dep Only Ortho)BPLAN A w/ Ortho Plan IIIB(PLAN A w/ Ortho Plan IV (Dep Only Ortho)BPLAN A w/ Ortho Plan VBPLAN BB(PLAN B w/ Ortho Plan IV (Dep Only Ortho)BPLAN B w/ Ortho Plan VBPLAN CBPLAN DBPLAN D w/ Ortho Plan IIIBPLAN EB'PLAN E w/ Ortho Plan I (Dep Only Ortho)B(PLAN E w/ Ortho Plan II (Dep Only Ortho)BPLAN E w/ Ortho Plan IIIB(PLAN E w/ Ortho Plan IV (Dep Only Ortho)BPLAN E w/ Ortho Plan VBPLAN FB'PLAN F w/ Ortho Plan I (Dep Only Ortho)B(PLAN F w/ Ortho Plan II (Dep Only Ortho)BPLAN F w/ Ortho Plan IIIB(PLAN F w/ Ortho Plan IV (Dep Only Ortho)BPLAN F w/ Ortho Plan VBPLAN GB(PLAN G w/ Ortho Plan IV (Dep Only Ortho)BPLAN G w/ Ortho Plan VBPLAN JB(PLAN J w/ Ortho Plan II (Dep Only Ortho)BPLAN J w/ Ortho Plan IIIBPLAN J w/ Ortho Plan VBPPOBPPO - 100 8000 PlusBPPO - 100 8000 PrimeBPPO - Diamond PlanBPPO - Emerald PlanBPPO - Opal PlanBPPO - Quartz PlanBPPO - Ruby (NSP) PlanBPPO - Ruby PlanBPPO - Sapphire PlanBPPO 0BPPO 100BPPO 100 - 5000BPPO 1000BPPO 1000 PRIMEBPPO 1000 PlusBPPO 1500BPPO 1500 PlusBPPO 1500 PrimeBPPO 200BPPO 2000BPPO 2000 PlusBPPO 25-2000A2-O w/ 1500 OrthoBPPO 250BPPO 250  Peak CareBPPO 250 CLASSICBPPO 250 VALUEBPPO 2500BPPO 2700BPPO 300BPPO 3000BPPO 3000 PlusBPPO 3500 PrimeBPPO 4000BPPO 50 - 1000 PrimeBPPO 50 - 500 PlusBPPO 50-1000 ValueBPPO 50-1000AVBPPO 50-1500 A2 OptimumBPPO 50-1500 ValueBPPO 50-1500A2- OBPPO 50-1500A2-OBPPO 50-1500AVBPPO 500BPPO 500 PlusBPPO 500 UTAHBPPO 5000BPPO 6000BPPO 6350BPPO 6850BPPO 70  -  3000 E RX2BPPO 70 $3000BPPO 70 $4000BPPO 70 $5000BPPO 70 $6000BPPO 70 - $3000BPPO 70 - $4000BPPO 70 - $5000BPPO 70 - $6000BPPO 70 - 1000 PrimeBPPO 70 - 1500 PlusBPPO 70 - 2000BPPO 70 - 2000 CC RX1BPPO 70 - 2000 PlusBPPO 70 - 2000 PrimeBPPO 70 - 2500 PlusBPPO 70 - 2500 PrimeBPPO 70 - 3000BPPO 70 - 3000 CC RX1BPPO 70 - 3000 PlusBPPO 70 - 3000 PrimeBPPO 70 - 4000 PlusBPPO 70 - 4000 PrimeBPPO 70 - 4000PrimeBPPO 70 - 5000 CC RX1BPPO 70 - 5000 CC RX2BPPO 70 - 5000 PlusBPPO 70 - 5000 PrimeBPPO 70 - 6000 E RX2BPPO 70 - 6000 PlusBPPO 70 - 6000 PrimeBPPO 750BPPO 750 PlusBPPO 80 $1000BPPO 80 $1500BPPO 80 $2000BPPO 80 $2500BPPO 80 $3000BPPO 80 $4000BPPO 80 $5000BPPO 80 $6000BPPO 80 - $1500BPPO 80 - $2000BPPO 80 - $2500BPPO 80 - $3000BPPO 80 - $4000BPPO 80 - $5000BPPO 80 - $6000BPPO 80 - 1000BPPO 80 - 1000 PlusBPPO 80 - 1000 PrimeBPPO 80 - 1000 RX1BPPO 80 - 1000PBPPO 80 - 1000PlusBPPO 80 - 1500BPPO 80 - 1500 CC RX1BPPO 80 - 1500 PlusBPPO 80 - 1500 PrimeBPPO 80 - 1500 RX1BPPO 80 - 2000 E RX1BPPO 80 - 2000 PlusBPPO 80 - 2000 PrimeBPPO 80 - 2000 RX1BPPO 80 - 2000PrimeBPPO 80 - 2500BPPO 80 - 2500 PlusBPPO 80 - 2500 PrimeBPPO 80 - 2500PlusBPPO 80 - 3000BPPO 80 - 3000 E RX1BPPO 80 - 3000 E RX2BPPO 80 - 3000 PlusBPPO 80 - 3000 PrimeBPPO 80 - 3000 RX1BPPO 80 - 3000PrimeBPPO 80 - 4000BPPO 80 - 4000 PlusBPPO 80 - 4000 PrimeBPPO 80 - 500BPPO 80 - 500 PlusBPPO 80 - 500 PrimeBPPO 80 - 5000BPPO 80 - 5000 E RX1BPPO 80 - 5000 E RX2BPPO 80 - 5000 PlusBPPO 80 - 5000 RX1BPPO 80 - 5000 RX2BPPO 80 - 6000BPPO 80 - 750BPPO 80 - 750 PlusBPPO 80 - 750 PrimeBPPO 80 - 750 RX1BPPO 80 - 750PlusBPPO 90 - 1000BPPO 90 - 500BPPO BBPPO EmployeeBPPO FAMBPPO PlanBPPO Plan 100BPPO Plan 1500BPPO Plan 50BPPO Plan 500BPPO Plan A1BPPO Plan B4BPPO Standard PPO80BPPO w/ OrthoBPPO-ABPeak Care 0BPeak Care 2500BPeak Care 500BPediatric DentalBPersonal AccidentBPersonal Accident $100KBPersonal Accident $200KBPersonal Accident $250KBPersonal Accident $25KBPersonal Accident $50KBPhysician AccessBPhysician Access - HDHPBPlan 1BPlan 1 - CaldwellBPlan 1 WillametteBPlan 100B	Plan 100VBPlan 150B
Plan 150-0BPlan 150-0VBPlan 150-10BPlan 150-10VB	Plan 1500B	Plan 150VBPlan 2BPlan 3BPlan 3 w/ ProTecBPlan 4BPlan 5BPlan 5 - VolBPlan 6BPlan 6 - VolBPlan 7BPlan 7 -VolBPlan 8 -VolBPlan 9BPlan ABPlan A + OrthoBPlan A - $10KBPlan A - EconomyBPlan A - GHCBPlan A - GHC POSBPlan A - GHC PPOBPlan A - KPBPlan A - PBCBPlan A - PremiumBPlan A -Basic Life / AD&DBPlan A 1000BPlan A 1500BPlan A 2000BPlan A 2500BPlan A 3500B
Plan A 500BPlan A 5000B
Plan AlphaBPlan BBPlan B + OrthoBPlan B - GHCBPlan B - GHC HMO BARGAINBPlan B - GHC HMO NON BARBPlan B - GHC POS BARGAINBPlan B - GHC POS NON BARBPlan B - GHC PPO BARGAINBPlan B - KPBPlan B - KP BARGAINEDBPlan B - KP NON BARBPlan B - PBCBPlan B -Basic Life / AD&DBPlan B 1000BPlan B 1500BPlan B 2000BPlan B 2500BPlan B 3500B
Plan B 500BPlan B 5000BPlan B Series - 500BPlan CBPlan C + OrthoBPlan C - $25KBPlan C - GHCBPlan C - GHC HMO NON BARBPlan C - GHC PPO NON BARBPlan C - KPBPlan C - PBCBPlan C -Basic Life / AD&DBPlan C 2500BPlan C 3000BPlan C 4000BPlan C 5500BPlan C 7000BPlan C MM Series - 1000BPlan DBPlan D - GHCBPlan D - PBCBPlan D -Basic Life / AD&DBPlan D 1500BPlan D 2500BPlan D 3500BPlan D 5000BPlan D w/ Ortho 2B
Plan DeltaBPlan EBPlan E + OrthoBPlan E w/ Ortho 1BPlan E w/ Ortho 2BPlan FBPlan GBPlan G w/ Ortho 1BPlan G w/ Ortho 2B
Plan GammaBPlan R - GHC HMO BARGAINBPlan R - GHC POS BARGAINBPlan R - GHC PPO BARGAINBPlan R - KP BARGAINEDB	Plan R1CCBPlan R2B	Plan R2CCB	Plan R4CCBPlan R5B	Plan R6CCBPlan V1BPlan V2BPlan V3 BuyUpBPlan V3 CoreBPlan V4BPlatform EAPBPlatinum 250 PREFBPlatinum 250 PREF VBPlatinum 500 PREFBPlatinum 500 PREF VB	Plus 20-1B	PreferredBPreferred 1000BPreferred 1025-2BPreferred 1025-3BPreferred 2000BPreferred 2500BPreferred 3000BPreferred 5000BPreferred EmployeeBPreferred PlanBPremera - Dickinson FF 2020BPremera - OPC 2020BPremera - PNW VEG 2020BPremera - RMT/Jore 2020BPremierBPremier + OrthoBPremier 1000BPremier 1500BPremier 2000BPremier 250BPremier 500BPremier Plan 1BPremier Plan 2BPremier Plan 3BPremier Plan 3 w/ORTHOBPremier Plan 4BPremier Plan IV OrthoBPremium CreditBPrescription Drug Plan 1BPrescription Drug Plan 2BPrincipal DentalB	RBS - HSABRBS - no HSABRBS Economy LEOFF 2BRBS Prem AdminBRBS Premium LEOFF 1BRBS Premium LEOFF 2BRFA HRA PlanBRFA PPO PlanBRXBRegence MedicalBRegence Medical + HRABRetireeBRetiree Basic Life - $5KBRisk RTBRiskRTB
RiskRT MECBRiskRT WaiverBRx 1BRx 2B
SECURE 750BSOLUTIONS 1000BSOLUTIONS 1500BSOLUTIONS 500BSP Vol Life - Non SmokerBSP Vol Life - SmokerBSTDBSTD $125/26wkBSTD 180BSTD 90B
STD Buy UpBSTD Buy Up PlanBSTD CoreBSTD Core PlanBSTD Non Ins Fees Plan 1B
STD Plan 1BSTD Plan 1 13 wkBSTD Plan 1 26 wkB
STD Plan 2BSTD Plan 2 13 wkBSTD Plan 2 26 wkBSTD Plan 3 13 wkBSTD Plan 3 26 wkBSTD Plan 4 13 wkBSTDF 3151 - 13 Week BenefitBSTERLING 1000 +BSTERLING 1500 +BSTERLING 1500 PrimeBSTERLING 2000 +BSTERLING 2000 PrimeBSTERLING 2500 +BSTERLING 2500 PrimeBSTERLING 3000 +BSTERLING 4000 +BSTERLING 500 +BSTERLING 5000 +BSTERLING 5000 PrimeBSTERLING 750 +BSalary Based AD&DBSalary Based LifeBSalary Based Life + AD&DBSalary Based Life - FlatBSalary Based no ADDBSalary Based w/ ADDBSalary Based w/ADD  (max $475)BSamaritan VisionBSamaritan Vision SpecialBSavings PlusBSecure 1000BSecure 1500BSecure 2000B
Secure 500BSelect Plus 1000BSelect Plus 2500BSelect Voluntary LifeBService FeesBShort Term DisabilityBSilver 3000 PREFBSilver 5500 PREFBSilver 5500 PREF VBSilver Essential 2500 PREFBSilver Essential 4000 PREFBSilver HSA 2000 PREFBSilver HSA 3500 PREFBSilver HSA 4250 PREFBSilver HSA Embedded 3000 PREFBSilver HSA HMOBSilver PPO MedicalBSilver PPO Medical UtahBSilver PlanBSolutions 1000BSolutions 1500BSolutions 500BSound Transit DentalBSpouse AD&DBSpouse ADnDBSpouse Basic Life/AD&DBSpouse LifeBSpouse Life InsuranceBSpouse Supplemental LifeBSpouse Vol AD&DBSpouse Vol LifeBSpouse Vol Life and AD&DBSpouse Vol Life/AD&DBSpouse Voluntary AD&DBSpouse Voluntary LifeBSpouse Voluntary Life/AD&DBSpouse Voluntary TermBStaff Plan - QualstarBStand - VSP Plan $10/$0BStand- VSP Plan $10/$25BStandardBStandard + OrthoBStandard - LEOFF 1BStandard - LEOFF 2BStandard DentalBStandard Dental UTAHBStandard PPOBSterling 1000 +BSterling 1000 PrimeBSterling 1500 +BSterling 1500 PrimeBSterling 2000 +BSterling 2000 PrimeBSterling 250 +BSterling 250 PrimeBSterling 2500 +BSterling 2500 PrimeBSterling 3000 +BSterling 3000 PrimeBSterling 4000 +BSterling 4000 PrimeBSterling 500 +BSterling 500 PrimeBSterling 5000 +BSterling 5000 PrimeBSterling 750 +BSterling 750 PrimeBSupplememental Child LifeBSupplememental Child Life & ADDBSupplememental EE Life & ADDBSupplememental SPO Life & ADDBSupplemental AD&DBSupplemental ADnDBSupplemental Benefit PlanBSupplemental Benefit Plan 2021BSupplemental Child LifeBSupplemental Child Life & ADDBSupplemental Dep LifeBSupplemental EE ADDBSupplemental EE LifeBSupplemental EE Life & ADDBSupplemental LifeBSupplemental Life & AD&DBSupplemental Life - UndeadBSupplemental SPO Life & AD&DBSupplemental Spouse ADDBSupplemental Spouse LifeBSupplemental Spouse Life & ADDBTDA Prepaid DentalBTITANIUM 200 +BTSG Actuary FeeBTXEO0116BTXEQ0029 EHDN5kBTech 80 $1000+BTech 80 $1000PBTech 80 $1500+BTech 80 $1500+ w/VisionBTech 80 $1500PBTech 80 $2000+BTech 80 $2000PBTech 80 $250 with VisionBTech 80 $250+BTech 80 $2500+BTech 80 $2500PBTech 80 $250PBTech 80 $3000+BTech 80 $3000PBTech 80 $350+BTech 80 $4000+BTech 80 $4000PBTech 80 $500BTech 80 $500+BTech 80 $500PBTech 80 $750+BTech 80 $750PBTech 90 $200+BTech 90 $350+BTech 90 $500+BTech 90 $500PBTech 90 $750+BTech 90 $750PBTech PremierBTech Premier+BTelemedicineBTerm Life 1BTerm Life 2BTitanium 200 +BTitanium 200 PrimeBTitanium 350 +BTitanium 350 PrimeBTitanium 500 +BTitanium 500 PrimeBTrust Admin FeeB	Trust FeeBTrust Fee - South PierceBUHC Med Adv PPO +1 - 5795BUHC Med Adv PPO +1 - 5855BUHC Med Adv PPO 5854BUHC Med Adv PPO Reg1 - 5794BUHC Med Adv PPO Reg2 - 5864BUMR Copay 750BUMR Core 500BUMR HDHP 1500BUMR HDHP 2500BUMR HDHP 5000BVEBABVEBA - AK waived coverageBVEBA - AK with medicalBVERA Whole HealthBVIMLY HearingBVOL LOW- 5P437BVP 5000BVSP - $0 Ded/2nd Pair RiderB
VSP BudgetB
VSP ChoiceBVSP Choice PlanBVSP ExtendedBVSP StandardB
VSP VisionBVSP Vision - Leoff 1BVSP Vision - Leoff 2BVSP Vision AdminBVSP VoluntaryBVTL DEP Life &  ADDBVTL Dependent LifeB
Value 1000B
Value 2500B
Value 3500B
Value 5000BVimly - MedBVimly - No MedBVimly AdminBVimly Admin FeeBVimly WaiverBVirtual Plus 2000BVirtual Plus 3000BVirtual Plus 5000BVirtual Plus SilverBVisionBVision $0 CopayBVision $0 Copay w/2nd PrBVision $10 CopayBVision $10 Copay w/2nd PrBVision $10/$15 CopayBVision $25 CopayBVision $25 Copay w/2nd PrBVision Non Ins Fees Plan 1BVision Non Ins Fees Plan 2BVision PerfectBVision PlanBVision Plan 1BVision Plan 2BVision Plan 3BVision Plan 4BVision PlusBVision PremiumBVol AD&DBVol AD&D - EmployeeBVol AD&D - SpouseBVol Accident - EE+SpouseBVol Accident - FamilyBVol Accident - IndividualBVol Accident - Parent+CHBVol Dep LifeB!Vol Dependent Child Life and AD&DBVol HMOBVol HighBVol LifeBVol Life OnlyBVol Spouse Life and AD&DBVoluntary AD&D - EmployeeBVoluntary AD&D - FamilyBVoluntary DentalBVoluntary Dep LifeBVoluntary Dep Life & ADDBVoluntary Dependent LifeBVoluntary EE LifeBVoluntary Employee LifeBVoluntary Group LegalBVoluntary LTDBVoluntary LifeBVoluntary Life & ADDBVoluntary Life and AD&DBVoluntary Life-NSBVoluntary Life-SBVoluntary Life/AD&DBVoluntary Plan 1BVoluntary Plan 2BVoluntary SP Life & ADDBVoluntary Spouse LifeBVoluntary Term LifeBVoluntary Vision PlanBWA CLASSIC 2000BWA PPO 1500BWCIF 1250 (Heritage)BWCIF 1250 (Prime)BWCIF 1500 HSA (Heritage)BWCIF 1500 HSA (Prime)BWCIF 200BWCIF 200 (Heritage)BWCIF 200 (Prime)BWCIF 2000 (Heritage)BWCIF 2000 (Prime)BWCIF 3000 (Heritage)BWCIF 3000 (Prime)BWCIF 3000 HSA (Heritage)BWCIF 500BWCIF 500 (Heritage)BWCIF 500 (Prime)BWCIF 5000 (Heritage)BWCIF 5000 (Prime)BWCIF 750BWCIF 750 (Heritage)BWCIF 750 (Prime)BWCIF High DeductibleB	WD DentalBWD HMO PlanBWD High PlanBWD Option 2BWD Staff EnhancedBWD Standard - BARGAINEDBWD Voluntary HighBWaitingBWaived CoverageBWaiver of CoverageB
Wellness 1BWellness IncentiveBWellness discountBWellnet OON Buy UpBWillamette DentalBWillamette Dental AdminBWillamette Dental PlansBYour Choice HRA
?k
Const_4Const*
_output_shapes	
:?*
dtype0	*?j
value?jB?j	?"?j                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      
?
Const_5Const*
_output_shapes
:;*
dtype0*?
value?B?;B B0BAKBALBARBAZBBCBCABCOBCTBDCBDEBFLBGABGUBHIBIABIDBILBINBKSBKYBLABMABMDBMEBMIBMNBMOBMSBMTBNCBNDBNEBNHBNJBNMBNVBNYBOHBOKBORBPABPRBRIBSCBSDBSEBTNBTXBUTBVABVIBVTBWBWABWIBWVBWY
?
Const_6Const*
_output_shapes
:;*
dtype0	*?
value?B?	;"?                                                        	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       
W
Const_7Const*
_output_shapes
:*
dtype0*
valueBB BFBM
h
Const_8Const*
_output_shapes
:*
dtype0	*-
value$B"	"                     
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_3Const_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_2552
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_5Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_2560
?
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_2Const_7Const_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *"
fR
__inference_<lambda>_2568
^
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^StatefulPartitionedCall_2
?@
Const_9Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
_all_features
_embeddings
_deep_layers
_logit_layer
task
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
 
+
PlanName
eemState
	eemGender

0
1
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
_ranking_metrics
_prediction_metrics
_label_metrics
_loss_metrics
trainable_variables
regularization_losses
	variables
	keras_api
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratem?m?%m?&m?'m?(m?)m?*m?+m?v?v?%v?&v?'v?(v?)v?*v?+v?
 
?
%0
&1
'2
(3
)4
*5
+6
7
8
 
?
%0
&1
'2
(3
)4
*5
+6
7
8
?
,layer_metrics

-layers
trainable_variables
	regularization_losses

	variables
.layer_regularization_losses
/metrics
0non_trainable_variables
 
?
1layer-0
2layer_with_weights-0
2layer-1
3trainable_variables
4regularization_losses
5	variables
6	keras_api
?
7layer-0
8layer_with_weights-0
8layer-1
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?
=layer-0
>layer_with_weights-0
>layer-1
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

(kernel
)bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
h

*kernel
+bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
VT
VARIABLE_VALUEdcn/dense_2/kernel._logit_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdcn/dense_2/bias,_logit_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Klayer_metrics

Llayers
trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
Nmetrics
Onon_trainable_variables

P0
 
 
 
 
 
 
?
Qlayer_metrics

Rlayers
trainable_variables
regularization_losses
	variables
Slayer_regularization_losses
Tmetrics
Unon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEembedding/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEembedding_2/embeddings0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEembedding_1/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdcn/dense/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdcn/dense/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdcn/dense_1/kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdcn/dense_1/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 

P0
 
!
Vlookup_table
W	keras_api
b
%
embeddings
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api

%0
 

%0
?
\layer_metrics

]layers
3trainable_variables
4regularization_losses
5	variables
^layer_regularization_losses
_metrics
`non_trainable_variables
!
alookup_table
b	keras_api
b
'
embeddings
ctrainable_variables
dregularization_losses
e	variables
f	keras_api

'0
 

'0
?
glayer_metrics

hlayers
9trainable_variables
:regularization_losses
;	variables
ilayer_regularization_losses
jmetrics
knon_trainable_variables
!
llookup_table
m	keras_api
b
&
embeddings
ntrainable_variables
oregularization_losses
p	variables
q	keras_api

&0
 

&0
?
rlayer_metrics

slayers
?trainable_variables
@regularization_losses
A	variables
tlayer_regularization_losses
umetrics
vnon_trainable_variables

(0
)1
 

(0
)1
?
wlayer_metrics

xlayers
Ctrainable_variables
Dregularization_losses
E	variables
ylayer_regularization_losses
zmetrics
{non_trainable_variables

*0
+1
 

*0
+1
?
|layer_metrics

}layers
Gtrainable_variables
Hregularization_losses
I	variables
~layer_regularization_losses
metrics
?non_trainable_variables
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api


PRMSE
 
 

P0
 

?_initializer
 

%0
 

%0
?
?layer_metrics
?layers
Xtrainable_variables
Yregularization_losses
Z	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 

10
21
 
 
 

?_initializer
 

'0
 

'0
?
?layer_metrics
?layers
ctrainable_variables
dregularization_losses
e	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 

70
81
 
 
 

?_initializer
 

&0
 

&0
?
?layer_metrics
?layers
ntrainable_variables
oregularization_losses
p	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
 

=0
>1
 
 
 
 
 
 
 
 
 
 
 
 
 
SQ
VARIABLE_VALUEtotal8task/_ranking_metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount8task/_ranking_metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
yw
VARIABLE_VALUEAdam/dcn/dense_2/kernel/mJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dcn/dense_2/bias/mH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/embedding/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/embedding_2/embeddings/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/embedding_1/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dcn/dense/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dcn/dense/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dcn/dense_1/kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dcn/dense_1/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dcn/dense_2/kernel/vJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dcn/dense_2/bias/vH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/embedding/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/embedding_2/embeddings/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/embedding_1/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dcn/dense/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dcn/dense/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dcn/dense_1/kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dcn/dense_1/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
s
serving_default_PlanNamePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
t
serving_default_eemGenderPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
s
serving_default_eemStatePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_3StatefulPartitionedCallserving_default_PlanNameserving_default_eemGenderserving_default_eemState
hash_tableConstembedding/embeddingshash_table_1Const_1embedding_1/embeddingshash_table_2Const_2embedding_2/embeddingsdcn/dense/kerneldcn/dense/biasdcn/dense_1/kerneldcn/dense_1/biasdcn/dense_2/kerneldcn/dense_2/bias*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2055
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename&dcn/dense_2/kernel/Read/ReadVariableOp$dcn/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp*embedding_2/embeddings/Read/ReadVariableOp*embedding_1/embeddings/Read/ReadVariableOp$dcn/dense/kernel/Read/ReadVariableOp"dcn/dense/bias/Read/ReadVariableOp&dcn/dense_1/kernel/Read/ReadVariableOp$dcn/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Adam/dcn/dense_2/kernel/m/Read/ReadVariableOp+Adam/dcn/dense_2/bias/m/Read/ReadVariableOp/Adam/embedding/embeddings/m/Read/ReadVariableOp1Adam/embedding_2/embeddings/m/Read/ReadVariableOp1Adam/embedding_1/embeddings/m/Read/ReadVariableOp+Adam/dcn/dense/kernel/m/Read/ReadVariableOp)Adam/dcn/dense/bias/m/Read/ReadVariableOp-Adam/dcn/dense_1/kernel/m/Read/ReadVariableOp+Adam/dcn/dense_1/bias/m/Read/ReadVariableOp-Adam/dcn/dense_2/kernel/v/Read/ReadVariableOp+Adam/dcn/dense_2/bias/v/Read/ReadVariableOp/Adam/embedding/embeddings/v/Read/ReadVariableOp1Adam/embedding_2/embeddings/v/Read/ReadVariableOp1Adam/embedding_1/embeddings/v/Read/ReadVariableOp+Adam/dcn/dense/kernel/v/Read/ReadVariableOp)Adam/dcn/dense/bias/v/Read/ReadVariableOp-Adam/dcn/dense_1/kernel/v/Read/ReadVariableOp+Adam/dcn/dense_1/bias/v/Read/ReadVariableOpConst_9*/
Tin(
&2$	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_2710
?
StatefulPartitionedCall_5StatefulPartitionedCallsaver_filenamedcn/dense_2/kerneldcn/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding/embeddingsembedding_2/embeddingsembedding_1/embeddingsdcn/dense/kerneldcn/dense/biasdcn/dense_1/kerneldcn/dense_1/biastotalcountAdam/dcn/dense_2/kernel/mAdam/dcn/dense_2/bias/mAdam/embedding/embeddings/mAdam/embedding_2/embeddings/mAdam/embedding_1/embeddings/mAdam/dcn/dense/kernel/mAdam/dcn/dense/bias/mAdam/dcn/dense_1/kernel/mAdam/dcn/dense_1/bias/mAdam/dcn/dense_2/kernel/vAdam/dcn/dense_2/bias/vAdam/embedding/embeddings/vAdam/embedding_2/embeddings/vAdam/embedding_1/embeddings/vAdam/dcn/dense/kernel/vAdam/dcn/dense/bias/vAdam/dcn/dense_1/kernel/vAdam/dcn/dense_1/bias/v*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_2822??

?'
?
=__inference_dcn_layer_call_and_return_conditional_losses_2010
planname
	eemgender
eemstate
sequential_1971
sequential_1973	"
sequential_1975:	? 
sequential_1_1978
sequential_1_1980	#
sequential_1_1982:< 
sequential_2_1985
sequential_2_1987	#
sequential_2_1989: 

dense_1994:	`?

dense_1996:	? 
dense_1_1999:
??
dense_1_2001:	?
dense_2_2004:	?
dense_2_2006:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallplannamesequential_1971sequential_1973sequential_1975*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13612$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCalleemstatesequential_1_1978sequential_1_1980sequential_1_1982*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15772&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall	eemgendersequential_2_1985sequential_2_1987sequential_2_1989*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14692&
$sequential_2/StatefulPartitionedCallt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0
dense_1994
dense_1996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_16642
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1999dense_1_2001*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_16812!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_2004dense_2_2006*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_16972!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_2269

inputs
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
__inference_<lambda>_25686
2key_value_init158_lookuptableimportv2_table_handle.
*key_value_init158_lookuptableimportv2_keys0
,key_value_init158_lookuptableimportv2_values	
identity??%key_value_init158/LookupTableImportV2?
%key_value_init158/LookupTableImportV2LookupTableImportV22key_value_init158_lookuptableimportv2_table_handle*key_value_init158_lookuptableimportv2_keys,key_value_init158_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init158/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init158/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init158/LookupTableImportV2%key_value_init158/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1608
string_lookup_1_input>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	"
embedding_1_1604:< 
identity??#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlestring_lookup_1_input;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_1604*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15312%
#embedding_1/StatefulPartitionedCall?
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_1545
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:< 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
}
(__inference_embedding_layer_call_fn_2449

inputs	
unknown:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1511
string_lookup_2_input>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	"
embedding_2_1507: 
identity??#embedding_2/StatefulPartitionedCall?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handlestring_lookup_2_input;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0embedding_2_1507*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14232%
#embedding_2/StatefulPartitionedCall?
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_2/StatefulPartitionedCall.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_2_input:

_output_shapes
: 
?	
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_2474

inputs	'
embedding_lookup_2468:< 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2468inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/2468*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/2468*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_<lambda>_25526
2key_value_init114_lookuptableimportv2_table_handle.
*key_value_init114_lookuptableimportv2_keys0
,key_value_init114_lookuptableimportv2_values	
identity??%key_value_init114/LookupTableImportV2?
%key_value_init114/LookupTableImportV2LookupTableImportV22key_value_init114_lookuptableimportv2_table_handle*key_value_init114_lookuptableimportv2_keys,key_value_init114_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init114/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init114/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init114/LookupTableImportV2%key_value_init114/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_2258

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?Z
?
__inference__wrapped_model_1295
planname
	eemgender
eemstateK
Gdcn_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleL
Hdcn_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value	A
.dcn_sequential_embedding_embedding_lookup_1249:	? O
Kdcn_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleP
Ldcn_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	D
2dcn_sequential_1_embedding_1_embedding_lookup_1258:< O
Kdcn_sequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleP
Ldcn_sequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	D
2dcn_sequential_2_embedding_2_embedding_lookup_1267: ;
(dcn_dense_matmul_readvariableop_resource:	`?8
)dcn_dense_biasadd_readvariableop_resource:	?>
*dcn_dense_1_matmul_readvariableop_resource:
??:
+dcn_dense_1_biasadd_readvariableop_resource:	?=
*dcn_dense_2_matmul_readvariableop_resource:	?9
+dcn_dense_2_biasadd_readvariableop_resource:
identity?? dcn/dense/BiasAdd/ReadVariableOp?dcn/dense/MatMul/ReadVariableOp?"dcn/dense_1/BiasAdd/ReadVariableOp?!dcn/dense_1/MatMul/ReadVariableOp?"dcn/dense_2/BiasAdd/ReadVariableOp?!dcn/dense_2/MatMul/ReadVariableOp?)dcn/sequential/embedding/embedding_lookup?:dcn/sequential/string_lookup/None_Lookup/LookupTableFindV2?-dcn/sequential_1/embedding_1/embedding_lookup?>dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?-dcn/sequential_2/embedding_2/embedding_lookup?>dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
:dcn/sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Gdcn_sequential_string_lookup_none_lookup_lookuptablefindv2_table_handleplannameHdcn_sequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2<
:dcn/sequential/string_lookup/None_Lookup/LookupTableFindV2?
%dcn/sequential/string_lookup/IdentityIdentityCdcn/sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2'
%dcn/sequential/string_lookup/Identity?
)dcn/sequential/embedding/embedding_lookupResourceGather.dcn_sequential_embedding_embedding_lookup_1249.dcn/sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@dcn/sequential/embedding/embedding_lookup/1249*'
_output_shapes
:????????? *
dtype02+
)dcn/sequential/embedding/embedding_lookup?
2dcn/sequential/embedding/embedding_lookup/IdentityIdentity2dcn/sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@dcn/sequential/embedding/embedding_lookup/1249*'
_output_shapes
:????????? 24
2dcn/sequential/embedding/embedding_lookup/Identity?
4dcn/sequential/embedding/embedding_lookup/Identity_1Identity;dcn/sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 26
4dcn/sequential/embedding/embedding_lookup/Identity_1?
>dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Kdcn_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleeemstateLdcn_sequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2@
>dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
)dcn/sequential_1/string_lookup_1/IdentityIdentityGdcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2+
)dcn/sequential_1/string_lookup_1/Identity?
-dcn/sequential_1/embedding_1/embedding_lookupResourceGather2dcn_sequential_1_embedding_1_embedding_lookup_12582dcn/sequential_1/string_lookup_1/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*E
_class;
97loc:@dcn/sequential_1/embedding_1/embedding_lookup/1258*'
_output_shapes
:????????? *
dtype02/
-dcn/sequential_1/embedding_1/embedding_lookup?
6dcn/sequential_1/embedding_1/embedding_lookup/IdentityIdentity6dcn/sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@dcn/sequential_1/embedding_1/embedding_lookup/1258*'
_output_shapes
:????????? 28
6dcn/sequential_1/embedding_1/embedding_lookup/Identity?
8dcn/sequential_1/embedding_1/embedding_lookup/Identity_1Identity?dcn/sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2:
8dcn/sequential_1/embedding_1/embedding_lookup/Identity_1?
>dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Kdcn_sequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handle	eemgenderLdcn_sequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2@
>dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
)dcn/sequential_2/string_lookup_2/IdentityIdentityGdcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2+
)dcn/sequential_2/string_lookup_2/Identity?
-dcn/sequential_2/embedding_2/embedding_lookupResourceGather2dcn_sequential_2_embedding_2_embedding_lookup_12672dcn/sequential_2/string_lookup_2/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*E
_class;
97loc:@dcn/sequential_2/embedding_2/embedding_lookup/1267*'
_output_shapes
:????????? *
dtype02/
-dcn/sequential_2/embedding_2/embedding_lookup?
6dcn/sequential_2/embedding_2/embedding_lookup/IdentityIdentity6dcn/sequential_2/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*E
_class;
97loc:@dcn/sequential_2/embedding_2/embedding_lookup/1267*'
_output_shapes
:????????? 28
6dcn/sequential_2/embedding_2/embedding_lookup/Identity?
8dcn/sequential_2/embedding_2/embedding_lookup/Identity_1Identity?dcn/sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2:
8dcn/sequential_2/embedding_2/embedding_lookup/Identity_1|
dcn/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
dcn/concatenate/concat/axis?
dcn/concatenate/concatConcatV2=dcn/sequential/embedding/embedding_lookup/Identity_1:output:0Adcn/sequential_1/embedding_1/embedding_lookup/Identity_1:output:0Adcn/sequential_2/embedding_2/embedding_lookup/Identity_1:output:0$dcn/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
dcn/concatenate/concat?
dcn/dense/MatMul/ReadVariableOpReadVariableOp(dcn_dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02!
dcn/dense/MatMul/ReadVariableOp?
dcn/dense/MatMulMatMuldcn/concatenate/concat:output:0'dcn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dcn/dense/MatMul?
 dcn/dense/BiasAdd/ReadVariableOpReadVariableOp)dcn_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dcn/dense/BiasAdd/ReadVariableOp?
dcn/dense/BiasAddBiasAdddcn/dense/MatMul:product:0(dcn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dcn/dense/BiasAddw
dcn/dense/ReluReludcn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dcn/dense/Relu?
!dcn/dense_1/MatMul/ReadVariableOpReadVariableOp*dcn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02#
!dcn/dense_1/MatMul/ReadVariableOp?
dcn/dense_1/MatMulMatMuldcn/dense/Relu:activations:0)dcn/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dcn/dense_1/MatMul?
"dcn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+dcn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"dcn/dense_1/BiasAdd/ReadVariableOp?
dcn/dense_1/BiasAddBiasAdddcn/dense_1/MatMul:product:0*dcn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dcn/dense_1/BiasAdd}
dcn/dense_1/ReluReludcn/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dcn/dense_1/Relu?
!dcn/dense_2/MatMul/ReadVariableOpReadVariableOp*dcn_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!dcn/dense_2/MatMul/ReadVariableOp?
dcn/dense_2/MatMulMatMuldcn/dense_1/Relu:activations:0)dcn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dcn/dense_2/MatMul?
"dcn/dense_2/BiasAdd/ReadVariableOpReadVariableOp+dcn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"dcn/dense_2/BiasAdd/ReadVariableOp?
dcn/dense_2/BiasAddBiasAdddcn/dense_2/MatMul:product:0*dcn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dcn/dense_2/BiasAddw
IdentityIdentitydcn/dense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dcn/dense/BiasAdd/ReadVariableOp ^dcn/dense/MatMul/ReadVariableOp#^dcn/dense_1/BiasAdd/ReadVariableOp"^dcn/dense_1/MatMul/ReadVariableOp#^dcn/dense_2/BiasAdd/ReadVariableOp"^dcn/dense_2/MatMul/ReadVariableOp*^dcn/sequential/embedding/embedding_lookup;^dcn/sequential/string_lookup/None_Lookup/LookupTableFindV2.^dcn/sequential_1/embedding_1/embedding_lookup?^dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2.^dcn/sequential_2/embedding_2/embedding_lookup?^dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2D
 dcn/dense/BiasAdd/ReadVariableOp dcn/dense/BiasAdd/ReadVariableOp2B
dcn/dense/MatMul/ReadVariableOpdcn/dense/MatMul/ReadVariableOp2H
"dcn/dense_1/BiasAdd/ReadVariableOp"dcn/dense_1/BiasAdd/ReadVariableOp2F
!dcn/dense_1/MatMul/ReadVariableOp!dcn/dense_1/MatMul/ReadVariableOp2H
"dcn/dense_2/BiasAdd/ReadVariableOp"dcn/dense_2/BiasAdd/ReadVariableOp2F
!dcn/dense_2/MatMul/ReadVariableOp!dcn/dense_2/MatMul/ReadVariableOp2V
)dcn/sequential/embedding/embedding_lookup)dcn/sequential/embedding/embedding_lookup2x
:dcn/sequential/string_lookup/None_Lookup/LookupTableFindV2:dcn/sequential/string_lookup/None_Lookup/LookupTableFindV22^
-dcn/sequential_1/embedding_1/embedding_lookup-dcn/sequential_1/embedding_1/embedding_lookup2?
>dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2>dcn/sequential_1/string_lookup_1/None_Lookup/LookupTableFindV22^
-dcn/sequential_2/embedding_2/embedding_lookup-dcn/sequential_2/embedding_2/embedding_lookup2?
>dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2>dcn/sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
9
__inference__creator_2513
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name137*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?	
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_1423

inputs	'
embedding_lookup_1417: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1417inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/1417*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1417*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_2458

inputs	(
embedding_lookup_2452:	? 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2452inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/2452*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/2452*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_2280

inputs
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
9
__inference__creator_2531
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name159*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_1681

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_1329
string_lookup_input
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13202
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
"__inference_dcn_layer_call_fn_1737
planname
	eemgender
eemstate
unknown
	unknown_0	
	unknown_1:	? 
	unknown_2
	unknown_3	
	unknown_4:< 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:	`?
	unknown_9:	?

unknown_10:
??

unknown_11:	?

unknown_12:	?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplanname	eemgendereemstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_dcn_layer_call_and_return_conditional_losses_17042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_2389

inputs>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	3
!embedding_2_embedding_lookup_2383: 
identity??embedding_2/embedding_lookup?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_2383!string_lookup_2/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*4
_class*
(&loc:@embedding_2/embedding_lookup/2383*'
_output_shapes
:????????? *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/2383*'
_output_shapes
:????????? 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_2/embedding_lookup/Identity_1?
IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding_2/embedding_lookup.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_2328

inputs
unknown
	unknown_0	
	unknown_1:< 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
+
__inference__destroyer_2508
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1361

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_1357:	? 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1357*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13152#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
?__inference_dense_layer_call_and_return_conditional_losses_1664

inputs1
matmul_readvariableop_resource:	`?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1619
string_lookup_1_input>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	"
embedding_1_1615:< 
identity??#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handlestring_lookup_1_input;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_1615*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15312%
#embedding_1/StatefulPartitionedCall?
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
~
*__inference_embedding_1_layer_call_fn_2465

inputs	
unknown:< 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15312
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__traced_restore_2822
file_prefix6
#assignvariableop_dcn_dense_2_kernel:	?1
#assignvariableop_1_dcn_dense_2_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: :
'assignvariableop_7_embedding_embeddings:	? ;
)assignvariableop_8_embedding_2_embeddings: ;
)assignvariableop_9_embedding_1_embeddings:< 7
$assignvariableop_10_dcn_dense_kernel:	`?1
"assignvariableop_11_dcn_dense_bias:	?:
&assignvariableop_12_dcn_dense_1_kernel:
??3
$assignvariableop_13_dcn_dense_1_bias:	?#
assignvariableop_14_total: #
assignvariableop_15_count: @
-assignvariableop_16_adam_dcn_dense_2_kernel_m:	?9
+assignvariableop_17_adam_dcn_dense_2_bias_m:B
/assignvariableop_18_adam_embedding_embeddings_m:	? C
1assignvariableop_19_adam_embedding_2_embeddings_m: C
1assignvariableop_20_adam_embedding_1_embeddings_m:< >
+assignvariableop_21_adam_dcn_dense_kernel_m:	`?8
)assignvariableop_22_adam_dcn_dense_bias_m:	?A
-assignvariableop_23_adam_dcn_dense_1_kernel_m:
??:
+assignvariableop_24_adam_dcn_dense_1_bias_m:	?@
-assignvariableop_25_adam_dcn_dense_2_kernel_v:	?9
+assignvariableop_26_adam_dcn_dense_2_bias_v:B
/assignvariableop_27_adam_embedding_embeddings_v:	? C
1assignvariableop_28_adam_embedding_2_embeddings_v: C
1assignvariableop_29_adam_embedding_1_embeddings_v:< >
+assignvariableop_30_adam_dcn_dense_kernel_v:	`?8
)assignvariableop_31_adam_dcn_dense_bias_v:	?A
-assignvariableop_32_adam_dcn_dense_1_kernel_v:
??:
+assignvariableop_33_adam_dcn_dense_1_bias_v:	?
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B._logit_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,_logit_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB8task/_ranking_metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB8task/_ranking_metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_dcn_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dcn_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp'assignvariableop_7_embedding_embeddingsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_embedding_2_embeddingsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_embedding_1_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dcn_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dcn_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_dcn_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dcn_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_adam_dcn_dense_2_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_dcn_dense_2_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp/assignvariableop_18_adam_embedding_embeddings_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_adam_embedding_2_embeddings_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp1assignvariableop_20_adam_embedding_1_embeddings_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dcn_dense_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dcn_dense_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_dcn_dense_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_dcn_dense_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_dcn_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_dcn_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_adam_embedding_embeddings_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_embedding_2_embeddings_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_embedding_1_embeddings_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp+assignvariableop_30_adam_dcn_dense_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dcn_dense_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp-assignvariableop_32_adam_dcn_dense_1_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dcn_dense_1_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34f
Identity_35IdentityIdentity_34:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_35?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
&__inference_dense_2_layer_call_fn_2248

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_16972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1403
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_1399:	? 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1399*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13152#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?'
?
=__inference_dcn_layer_call_and_return_conditional_losses_1966
planname
	eemgender
eemstate
sequential_1927
sequential_1929	"
sequential_1931:	? 
sequential_1_1934
sequential_1_1936	#
sequential_1_1938:< 
sequential_2_1941
sequential_2_1943	#
sequential_2_1945: 

dense_1950:	`?

dense_1952:	? 
dense_1_1955:
??
dense_1_1957:	?
dense_2_1960:	?
dense_2_1962:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallplannamesequential_1927sequential_1929sequential_1931*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13202$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCalleemstatesequential_1_1934sequential_1_1936sequential_1_1938*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15362&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall	eemgendersequential_2_1941sequential_2_1943sequential_2_1945*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14282&
$sequential_2/StatefulPartitionedCallt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0
dense_1950
dense_1952*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_16642
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1955dense_1_1957*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_16812!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_1960dense_2_1962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_16972!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
__inference__initializer_25216
2key_value_init136_lookuptableimportv2_table_handle.
*key_value_init136_lookuptableimportv2_keys0
,key_value_init136_lookuptableimportv2_values	
identity??%key_value_init136/LookupTableImportV2?
%key_value_init136/LookupTableImportV2LookupTableImportV22key_value_init136_lookuptableimportv2_table_handle*key_value_init136_lookuptableimportv2_keys,key_value_init136_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init136/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init136/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :;:;2N
%key_value_init136/LookupTableImportV2%key_value_init136/LookupTableImportV2: 

_output_shapes
:;: 

_output_shapes
:;
?
~
*__inference_embedding_2_layer_call_fn_2481

inputs	
unknown: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14232
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
+
__inference__destroyer_2526
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
A__inference_dense_2_layer_call_and_return_conditional_losses_1697

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference_dcn_layer_call_fn_1922
planname
	eemgender
eemstate
unknown
	unknown_0	
	unknown_1:	? 
	unknown_2
	unknown_3	
	unknown_4:< 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:	`?
	unknown_9:	?

unknown_10:
??

unknown_11:	?

unknown_12:	?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplanname	eemgendereemstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_dcn_layer_call_and_return_conditional_losses_18522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
+__inference_sequential_2_layer_call_fn_1437
string_lookup_2_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_2_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_2_input:

_output_shapes
: 
?V
?
=__inference_dcn_layer_call_and_return_conditional_losses_2184
features_planname
features_eemgender
features_eemstateG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	=
*sequential_embedding_embedding_lookup_2138:	? K
Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleL
Hsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	@
.sequential_1_embedding_1_embedding_lookup_2147:< K
Gsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleL
Hsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	@
.sequential_2_embedding_2_embedding_lookup_2156: 7
$dense_matmul_readvariableop_resource:	`?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?)sequential_2/embedding_2/embedding_lookup?:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_plannameDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_2138*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2138*'
_output_shapes
:????????? *
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2138*'
_output_shapes
:????????? 20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 22
0sequential/embedding/embedding_lookup/Identity_1?
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handlefeatures_eemstateHsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2<
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
%sequential_1/string_lookup_1/IdentityIdentityCsequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2'
%sequential_1/string_lookup_1/Identity?
)sequential_1/embedding_1/embedding_lookupResourceGather.sequential_1_embedding_1_embedding_lookup_2147.sequential_1/string_lookup_1/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/2147*'
_output_shapes
:????????? *
dtype02+
)sequential_1/embedding_1/embedding_lookup?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/2147*'
_output_shapes
:????????? 24
2sequential_1/embedding_1/embedding_lookup/Identity?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_1/embedding_1/embedding_lookup/Identity_1?
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handlefeatures_eemgenderHsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2<
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
%sequential_2/string_lookup_2/IdentityIdentityCsequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2'
%sequential_2/string_lookup_2/Identity?
)sequential_2/embedding_2/embedding_lookupResourceGather.sequential_2_embedding_2_embedding_lookup_2156.sequential_2/string_lookup_2/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@sequential_2/embedding_2/embedding_lookup/2156*'
_output_shapes
:????????? *
dtype02+
)sequential_2/embedding_2/embedding_lookup?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@sequential_2/embedding_2/embedding_lookup/2156*'
_output_shapes
:????????? 24
2sequential_2/embedding_2/embedding_lookup/Identity?
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_2/embedding_2/embedding_lookup/Identity_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0=sequential_2/embedding_2/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdds
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookup;^sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2*^sequential_2/embedding_2/embedding_lookup;^sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2x
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV22V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2x
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:V R
#
_output_shapes
:?????????
+
_user_specified_namefeatures/PlanName:WS
#
_output_shapes
:?????????
,
_user_specified_namefeatures/eemGender:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1392
string_lookup_input<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_1388:	? 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handlestring_lookup_input9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1388*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13152#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?L
?
__inference__traced_save_2710
file_prefix1
-savev2_dcn_dense_2_kernel_read_readvariableop/
+savev2_dcn_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop3
/savev2_embedding_embeddings_read_readvariableop5
1savev2_embedding_2_embeddings_read_readvariableop5
1savev2_embedding_1_embeddings_read_readvariableop/
+savev2_dcn_dense_kernel_read_readvariableop-
)savev2_dcn_dense_bias_read_readvariableop1
-savev2_dcn_dense_1_kernel_read_readvariableop/
+savev2_dcn_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_adam_dcn_dense_2_kernel_m_read_readvariableop6
2savev2_adam_dcn_dense_2_bias_m_read_readvariableop:
6savev2_adam_embedding_embeddings_m_read_readvariableop<
8savev2_adam_embedding_2_embeddings_m_read_readvariableop<
8savev2_adam_embedding_1_embeddings_m_read_readvariableop6
2savev2_adam_dcn_dense_kernel_m_read_readvariableop4
0savev2_adam_dcn_dense_bias_m_read_readvariableop8
4savev2_adam_dcn_dense_1_kernel_m_read_readvariableop6
2savev2_adam_dcn_dense_1_bias_m_read_readvariableop8
4savev2_adam_dcn_dense_2_kernel_v_read_readvariableop6
2savev2_adam_dcn_dense_2_bias_v_read_readvariableop:
6savev2_adam_embedding_embeddings_v_read_readvariableop<
8savev2_adam_embedding_2_embeddings_v_read_readvariableop<
8savev2_adam_embedding_1_embeddings_v_read_readvariableop6
2savev2_adam_dcn_dense_kernel_v_read_readvariableop4
0savev2_adam_dcn_dense_bias_v_read_readvariableop8
4savev2_adam_dcn_dense_1_kernel_v_read_readvariableop6
2savev2_adam_dcn_dense_1_bias_v_read_readvariableop
savev2_const_9

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B._logit_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,_logit_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB8task/_ranking_metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB8task/_ranking_metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBJ_logit_layer/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBH_logit_layer/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dcn_dense_2_kernel_read_readvariableop+savev2_dcn_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop/savev2_embedding_embeddings_read_readvariableop1savev2_embedding_2_embeddings_read_readvariableop1savev2_embedding_1_embeddings_read_readvariableop+savev2_dcn_dense_kernel_read_readvariableop)savev2_dcn_dense_bias_read_readvariableop-savev2_dcn_dense_1_kernel_read_readvariableop+savev2_dcn_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_adam_dcn_dense_2_kernel_m_read_readvariableop2savev2_adam_dcn_dense_2_bias_m_read_readvariableop6savev2_adam_embedding_embeddings_m_read_readvariableop8savev2_adam_embedding_2_embeddings_m_read_readvariableop8savev2_adam_embedding_1_embeddings_m_read_readvariableop2savev2_adam_dcn_dense_kernel_m_read_readvariableop0savev2_adam_dcn_dense_bias_m_read_readvariableop4savev2_adam_dcn_dense_1_kernel_m_read_readvariableop2savev2_adam_dcn_dense_1_bias_m_read_readvariableop4savev2_adam_dcn_dense_2_kernel_v_read_readvariableop2savev2_adam_dcn_dense_2_bias_v_read_readvariableop6savev2_adam_embedding_embeddings_v_read_readvariableop8savev2_adam_embedding_2_embeddings_v_read_readvariableop8savev2_adam_embedding_1_embeddings_v_read_readvariableop2savev2_adam_dcn_dense_kernel_v_read_readvariableop0savev2_adam_dcn_dense_bias_v_read_readvariableop4savev2_adam_dcn_dense_1_kernel_v_read_readvariableop2savev2_adam_dcn_dense_1_bias_v_read_readvariableopsavev2_const_9"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:: : : : : :	? : :< :	`?:?:
??:?: : :	?::	? : :< :	`?:?:
??:?:	?::	? : :< :	`?:?:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	? :$	 

_output_shapes

: :$
 

_output_shapes

:< :%!

_output_shapes
:	`?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	? :$ 

_output_shapes

: :$ 

_output_shapes

:< :%!

_output_shapes
:	`?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	? :$ 

_output_shapes

: :$ 

_output_shapes

:< :%!

_output_shapes
:	`?:! 

_output_shapes	
:?:&!"
 
_output_shapes
:
??:!"

_output_shapes	
:?:#

_output_shapes
: 
?
?
+__inference_sequential_2_layer_call_fn_1489
string_lookup_2_input
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_2_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_2_input:

_output_shapes
: 
?
?
"__inference_dcn_layer_call_fn_2092
features_planname
features_eemgender
features_eemstate
unknown
	unknown_0	
	unknown_1:	? 
	unknown_2
	unknown_3	
	unknown_4:< 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:	`?
	unknown_9:	?

unknown_10:
??

unknown_11:	?

unknown_12:	?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_plannamefeatures_eemgenderfeatures_eemstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_dcn_layer_call_and_return_conditional_losses_17042
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
#
_output_shapes
:?????????
+
_user_specified_namefeatures/PlanName:WS
#
_output_shapes
:?????????
,
_user_specified_namefeatures/eemGender:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
A__inference_dense_1_layer_call_and_return_conditional_losses_2442

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_1381
string_lookup_input
unknown
	unknown_0	
	unknown_1:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13612
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:?????????
-
_user_specified_namestring_lookup_input:

_output_shapes
: 
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1428

inputs>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	"
embedding_2_1424: 
identity??#embedding_2/StatefulPartitionedCall?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0embedding_2_1424*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14232%
#embedding_2/StatefulPartitionedCall?
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_2/StatefulPartitionedCall.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
"__inference_signature_wrapper_2055
planname
	eemgender
eemstate
unknown
	unknown_0	
	unknown_1:	? 
	unknown_2
	unknown_3	
	unknown_4:< 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:	`?
	unknown_9:	?

unknown_10:
??

unknown_11:	?

unknown_12:	?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallplanname	eemgendereemstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_12952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
PlanName:NJ
#
_output_shapes
:?????????
#
_user_specified_name	eemGender:MI
#
_output_shapes
:?????????
"
_user_specified_name
eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
+
__inference__destroyer_2544
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
+__inference_sequential_2_layer_call_fn_2365

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_2402

inputs>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	3
!embedding_2_embedding_lookup_2396: 
identity??embedding_2/embedding_lookup?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
embedding_2/embedding_lookupResourceGather!embedding_2_embedding_lookup_2396!string_lookup_2/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*4
_class*
(&loc:@embedding_2/embedding_lookup/2396*'
_output_shapes
:????????? *
dtype02
embedding_2/embedding_lookup?
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_2/embedding_lookup/2396*'
_output_shapes
:????????? 2'
%embedding_2/embedding_lookup/Identity?
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_2/embedding_lookup/Identity_1?
IdentityIdentity0embedding_2/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding_2/embedding_lookup.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?	
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_2490

inputs	'
embedding_lookup_2484: 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_2484inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/2484*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/2484*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_2306

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	2
embedding_embedding_lookup_2300:	? 
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_2300string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/2300*'
_output_shapes
:????????? *
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/2300*'
_output_shapes
:????????? 2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2'
%embedding/embedding_lookup/Identity_1?
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
"__inference_dcn_layer_call_fn_2129
features_planname
features_eemgender
features_eemstate
unknown
	unknown_0	
	unknown_1:	? 
	unknown_2
	unknown_3	
	unknown_4:< 
	unknown_5
	unknown_6	
	unknown_7: 
	unknown_8:	`?
	unknown_9:	?

unknown_10:
??

unknown_11:	?

unknown_12:	?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_plannamefeatures_eemgenderfeatures_eemstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2			*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *F
fAR?
=__inference_dcn_layer_call_and_return_conditional_losses_18522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
#
_output_shapes
:?????????
+
_user_specified_namefeatures/PlanName:WS
#
_output_shapes
:?????????
,
_user_specified_namefeatures/eemGender:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1577

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	"
embedding_1_1573:< 
identity??#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_1573*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15312%
#embedding_1/StatefulPartitionedCall?
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_1597
string_lookup_1_input
unknown
	unknown_0	
	unknown_1:< 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallstring_lookup_1_inputunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_1_input:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1320

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	!
embedding_1316:	? 
identity??!embedding/StatefulPartitionedCall?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstring_lookup/Identity:output:0embedding_1316*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_13152#
!embedding/StatefulPartitionedCall?
IdentityIdentity*embedding/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp"^embedding/StatefulPartitionedCall,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1536

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	"
embedding_1_1532:< 
identity??#embedding_1/StatefulPartitionedCall?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_1/Identity:output:0embedding_1_1532*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_15312%
#embedding_1/StatefulPartitionedCall?
IdentityIdentity,embedding_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_1/StatefulPartitionedCall.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?'
?
=__inference_dcn_layer_call_and_return_conditional_losses_1704
features

features_1

features_2
sequential_1630
sequential_1632	"
sequential_1634:	? 
sequential_1_1637
sequential_1_1639	#
sequential_1_1641:< 
sequential_2_1644
sequential_2_1646	#
sequential_2_1648: 

dense_1665:	`?

dense_1667:	? 
dense_1_1682:
??
dense_1_1684:	?
dense_2_1698:	?
dense_2_1700:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallfeaturessequential_1630sequential_1632sequential_1634*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13202$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall
features_2sequential_1_1637sequential_1_1639sequential_1_1641*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15362&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall
features_1sequential_2_1644sequential_2_1646sequential_2_1648*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14282&
$sequential_2/StatefulPartitionedCallt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0
dense_1665
dense_1667*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_16642
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1682dense_1_1684*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_16812!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_1698dense_2_1700*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_16972!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?
?
__inference__initializer_25036
2key_value_init114_lookuptableimportv2_table_handle.
*key_value_init114_lookuptableimportv2_keys0
,key_value_init114_lookuptableimportv2_values	
identity??%key_value_init114/LookupTableImportV2?
%key_value_init114/LookupTableImportV2LookupTableImportV22key_value_init114_lookuptableimportv2_table_handle*key_value_init114_lookuptableimportv2_keys,key_value_init114_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init114/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init114/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?:?2N
%key_value_init114/LookupTableImportV2%key_value_init114/LookupTableImportV2:!

_output_shapes	
:?:!

_output_shapes	
:?
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_2341

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	3
!embedding_1_embedding_lookup_2335:< 
identity??embedding_1/embedding_lookup?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_2335!string_lookup_1/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/2335*'
_output_shapes
:????????? *
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/2335*'
_output_shapes
:????????? 2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_1/embedding_lookup/Identity_1?
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding_1/embedding_lookup.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
__inference_<lambda>_25606
2key_value_init136_lookuptableimportv2_table_handle.
*key_value_init136_lookuptableimportv2_keys0
,key_value_init136_lookuptableimportv2_values	
identity??%key_value_init136/LookupTableImportV2?
%key_value_init136/LookupTableImportV2LookupTableImportV22key_value_init136_lookuptableimportv2_table_handle*key_value_init136_lookuptableimportv2_keys,key_value_init136_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init136/LookupTableImportV2S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init136/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :;:;2N
%key_value_init136/LookupTableImportV2%key_value_init136/LookupTableImportV2: 

_output_shapes
:;: 

_output_shapes
:;
?'
?
=__inference_dcn_layer_call_and_return_conditional_losses_1852
features

features_1

features_2
sequential_1813
sequential_1815	"
sequential_1817:	? 
sequential_1_1820
sequential_1_1822	#
sequential_1_1824:< 
sequential_2_1827
sequential_2_1829	#
sequential_2_1831: 

dense_1836:	`?

dense_1838:	? 
dense_1_1841:
??
dense_1_1843:	?
dense_2_1846:	?
dense_2_1848:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?$sequential_1/StatefulPartitionedCall?$sequential_2/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCallfeaturessequential_1813sequential_1815sequential_1817*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_13612$
"sequential/StatefulPartitionedCall?
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall
features_2sequential_1_1820sequential_1_1822sequential_1_1824*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15772&
$sequential_1/StatefulPartitionedCall?
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall
features_1sequential_2_1827sequential_2_1829sequential_2_1831*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14692&
$sequential_2/StatefulPartitionedCallt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2+sequential/StatefulPartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0-sequential_2/StatefulPartitionedCall:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0
dense_1836
dense_1838*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_16642
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_1841dense_1_1843*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_16812!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_1846dense_2_1848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_16972!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?	
?
C__inference_embedding_layer_call_and_return_conditional_losses_1315

inputs	(
embedding_lookup_1309:	? 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1309inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/1309*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1309*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__initializer_25396
2key_value_init158_lookuptableimportv2_table_handle.
*key_value_init158_lookuptableimportv2_keys0
,key_value_init158_lookuptableimportv2_values	
identity??%key_value_init158/LookupTableImportV2?
%key_value_init158/LookupTableImportV2LookupTableImportV22key_value_init158_lookuptableimportv2_table_handle*key_value_init158_lookuptableimportv2_keys,key_value_init158_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 2'
%key_value_init158/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstX
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: 2

Identityv
NoOpNoOp&^key_value_init158/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2N
%key_value_init158/LookupTableImportV2%key_value_init158/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
?__inference_dense_layer_call_and_return_conditional_losses_2422

inputs1
matmul_readvariableop_resource:	`?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
$__inference_dense_layer_call_fn_2411

inputs
unknown:	`?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_16642
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????`
 
_user_specified_nameinputs
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1469

inputs>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	"
embedding_2_1465: 
identity??#embedding_2/StatefulPartitionedCall?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0embedding_2_1465*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14232%
#embedding_2/StatefulPartitionedCall?
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_2/StatefulPartitionedCall.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1500
string_lookup_2_input>
:string_lookup_2_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_2_none_lookup_lookuptablefindv2_default_value	"
embedding_2_1496: 
identity??#embedding_2/StatefulPartitionedCall?-string_lookup_2/None_Lookup/LookupTableFindV2?
-string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_2_none_lookup_lookuptablefindv2_table_handlestring_lookup_2_input;string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_2/None_Lookup/LookupTableFindV2?
string_lookup_2/IdentityIdentity6string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_2/Identity?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCall!string_lookup_2/Identity:output:0embedding_2_1496*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_14232%
#embedding_2/StatefulPartitionedCall?
IdentityIdentity,embedding_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp$^embedding_2/StatefulPartitionedCall.^string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2^
-string_lookup_2/None_Lookup/LookupTableFindV2-string_lookup_2/None_Lookup/LookupTableFindV2:Z V
#
_output_shapes
:?????????
/
_user_specified_namestring_lookup_2_input:

_output_shapes
: 
?
9
__inference__creator_2495
identity??
hash_tabley

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name115*
value_dtype0	2

hash_tablec
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: 2

Identity[
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
F__inference_sequential_1_layer_call_and_return_conditional_losses_2354

inputs>
:string_lookup_1_none_lookup_lookuptablefindv2_table_handle?
;string_lookup_1_none_lookup_lookuptablefindv2_default_value	3
!embedding_1_embedding_lookup_2348:< 
identity??embedding_1/embedding_lookup?-string_lookup_1/None_Lookup/LookupTableFindV2?
-string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2:string_lookup_1_none_lookup_lookuptablefindv2_table_handleinputs;string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2/
-string_lookup_1/None_Lookup/LookupTableFindV2?
string_lookup_1/IdentityIdentity6string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup_1/Identity?
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_2348!string_lookup_1/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/2348*'
_output_shapes
:????????? *
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/2348*'
_output_shapes
:????????? 2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2)
'embedding_1/embedding_lookup/Identity_1?
IdentityIdentity0embedding_1/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding_1/embedding_lookup.^string_lookup_1/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2^
-string_lookup_1/None_Lookup/LookupTableFindV2-string_lookup_1/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_2293

inputs<
8string_lookup_none_lookup_lookuptablefindv2_table_handle=
9string_lookup_none_lookup_lookuptablefindv2_default_value	2
embedding_embedding_lookup_2287:	? 
identity??embedding/embedding_lookup?+string_lookup/None_Lookup/LookupTableFindV2?
+string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV28string_lookup_none_lookup_lookuptablefindv2_table_handleinputs9string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2-
+string_lookup/None_Lookup/LookupTableFindV2?
string_lookup/IdentityIdentity4string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2
string_lookup/Identity?
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_2287string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/2287*'
_output_shapes
:????????? *
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/2287*'
_output_shapes
:????????? 2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2'
%embedding/embedding_lookup/Identity_1?
IdentityIdentity.embedding/embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identity?
NoOpNoOp^embedding/embedding_lookup,^string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 28
embedding/embedding_lookupembedding/embedding_lookup2Z
+string_lookup/None_Lookup/LookupTableFindV2+string_lookup/None_Lookup/LookupTableFindV2:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
+__inference_sequential_1_layer_call_fn_2317

inputs
unknown
	unknown_0	
	unknown_1:< 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_1_layer_call_and_return_conditional_losses_15362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?
?
&__inference_dense_1_layer_call_fn_2431

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_16812
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_sequential_2_layer_call_fn_2376

inputs
unknown
	unknown_0	
	unknown_1: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_sequential_2_layer_call_and_return_conditional_losses_14692
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: 
?V
?
=__inference_dcn_layer_call_and_return_conditional_losses_2239
features_planname
features_eemgender
features_eemstateG
Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handleH
Dsequential_string_lookup_none_lookup_lookuptablefindv2_default_value	=
*sequential_embedding_embedding_lookup_2193:	? K
Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handleL
Hsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value	@
.sequential_1_embedding_1_embedding_lookup_2202:< K
Gsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handleL
Hsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value	@
.sequential_2_embedding_2_embedding_lookup_2211: 7
$dense_matmul_readvariableop_resource:	`?4
%dense_biasadd_readvariableop_resource:	?:
&dense_1_matmul_readvariableop_resource:
??6
'dense_1_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?%sequential/embedding/embedding_lookup?6sequential/string_lookup/None_Lookup/LookupTableFindV2?)sequential_1/embedding_1/embedding_lookup?:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?)sequential_2/embedding_2/embedding_lookup?:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
6sequential/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Csequential_string_lookup_none_lookup_lookuptablefindv2_table_handlefeatures_plannameDsequential_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????28
6sequential/string_lookup/None_Lookup/LookupTableFindV2?
!sequential/string_lookup/IdentityIdentity?sequential/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2#
!sequential/string_lookup/Identity?
%sequential/embedding/embedding_lookupResourceGather*sequential_embedding_embedding_lookup_2193*sequential/string_lookup/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2193*'
_output_shapes
:????????? *
dtype02'
%sequential/embedding/embedding_lookup?
.sequential/embedding/embedding_lookup/IdentityIdentity.sequential/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*=
_class3
1/loc:@sequential/embedding/embedding_lookup/2193*'
_output_shapes
:????????? 20
.sequential/embedding/embedding_lookup/Identity?
0sequential/embedding/embedding_lookup/Identity_1Identity7sequential/embedding/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 22
0sequential/embedding/embedding_lookup/Identity_1?
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_table_handlefeatures_eemstateHsequential_1_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2<
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2?
%sequential_1/string_lookup_1/IdentityIdentityCsequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2'
%sequential_1/string_lookup_1/Identity?
)sequential_1/embedding_1/embedding_lookupResourceGather.sequential_1_embedding_1_embedding_lookup_2202.sequential_1/string_lookup_1/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/2202*'
_output_shapes
:????????? *
dtype02+
)sequential_1/embedding_1/embedding_lookup?
2sequential_1/embedding_1/embedding_lookup/IdentityIdentity2sequential_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@sequential_1/embedding_1/embedding_lookup/2202*'
_output_shapes
:????????? 24
2sequential_1/embedding_1/embedding_lookup/Identity?
4sequential_1/embedding_1/embedding_lookup/Identity_1Identity;sequential_1/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_1/embedding_1/embedding_lookup/Identity_1?
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2LookupTableFindV2Gsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_table_handlefeatures_eemgenderHsequential_2_string_lookup_2_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:?????????2<
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2?
%sequential_2/string_lookup_2/IdentityIdentityCsequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????2'
%sequential_2/string_lookup_2/Identity?
)sequential_2/embedding_2/embedding_lookupResourceGather.sequential_2_embedding_2_embedding_lookup_2211.sequential_2/string_lookup_2/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*A
_class7
53loc:@sequential_2/embedding_2/embedding_lookup/2211*'
_output_shapes
:????????? *
dtype02+
)sequential_2/embedding_2/embedding_lookup?
2sequential_2/embedding_2/embedding_lookup/IdentityIdentity2sequential_2/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*A
_class7
53loc:@sequential_2/embedding_2/embedding_lookup/2211*'
_output_shapes
:????????? 24
2sequential_2/embedding_2/embedding_lookup/Identity?
4sequential_2/embedding_2/embedding_lookup/Identity_1Identity;sequential_2/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 26
4sequential_2/embedding_2/embedding_lookup/Identity_1t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV29sequential/embedding/embedding_lookup/Identity_1:output:0=sequential_1/embedding_1/embedding_lookup/Identity_1:output:0=sequential_2/embedding_2/embedding_lookup/Identity_1:output:0 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????`2
concatenate/concat?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	`?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdds
IdentityIdentitydense_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp&^sequential/embedding/embedding_lookup7^sequential/string_lookup/None_Lookup/LookupTableFindV2*^sequential_1/embedding_1/embedding_lookup;^sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2*^sequential_2/embedding_2/embedding_lookup;^sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:?????????:?????????:?????????: : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2N
%sequential/embedding/embedding_lookup%sequential/embedding/embedding_lookup2p
6sequential/string_lookup/None_Lookup/LookupTableFindV26sequential/string_lookup/None_Lookup/LookupTableFindV22V
)sequential_1/embedding_1/embedding_lookup)sequential_1/embedding_1/embedding_lookup2x
:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV2:sequential_1/string_lookup_1/None_Lookup/LookupTableFindV22V
)sequential_2/embedding_2/embedding_lookup)sequential_2/embedding_2/embedding_lookup2x
:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:sequential_2/string_lookup_2/None_Lookup/LookupTableFindV2:V R
#
_output_shapes
:?????????
+
_user_specified_namefeatures/PlanName:WS
#
_output_shapes
:?????????
,
_user_specified_namefeatures/eemGender:VR
#
_output_shapes
:?????????
+
_user_specified_namefeatures/eemState:

_output_shapes
: :

_output_shapes
: :


_output_shapes
: 
?	
?
E__inference_embedding_1_layer_call_and_return_conditional_losses_1531

inputs	'
embedding_lookup_1525:< 
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_1525inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*(
_class
loc:@embedding_lookup/1525*'
_output_shapes
:????????? *
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@embedding_lookup/1525*'
_output_shapes
:????????? 2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:????????? 2
embedding_lookup/Identity_1
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*'
_output_shapes
:????????? 2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_4:0StatefulPartitionedCall_58"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
9
PlanName-
serving_default_PlanName:0?????????
;
	eemGender.
serving_default_eemGender:0?????????
9
eemState-
serving_default_eemState:0?????????>
output_12
StatefulPartitionedCall_3:0?????????tensorflow/serving/predict:??
?
_all_features
_embeddings
_deep_layers
_logit_layer
task
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_model
 "
trackable_list_wrapper
K
PlanName
eemState
	eemGender"
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
_ranking_metrics
_prediction_metrics
_label_metrics
_loss_metrics
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 iter

!beta_1

"beta_2
	#decay
$learning_ratem?m?%m?&m?'m?(m?)m?*m?+m?v?v?%v?&v?'v?(v?)v?*v?+v?"
	optimizer
 "
trackable_dict_wrapper
_
%0
&1
'2
(3
)4
*5
+6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
%0
&1
'2
(3
)4
*5
+6
7
8"
trackable_list_wrapper
?
,layer_metrics

-layers
trainable_variables
	regularization_losses

	variables
.layer_regularization_losses
/metrics
0non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
1layer-0
2layer_with_weights-0
2layer-1
3trainable_variables
4regularization_losses
5	variables
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
7layer-0
8layer_with_weights-0
8layer-1
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
=layer-0
>layer_with_weights-0
>layer-1
?trainable_variables
@regularization_losses
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

(kernel
)bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
%:#	?2dcn/dense_2/kernel
:2dcn/dense_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Klayer_metrics

Llayers
trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
Nmetrics
Onon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qlayer_metrics

Rlayers
trainable_variables
regularization_losses
	variables
Slayer_regularization_losses
Tmetrics
Unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%	? 2embedding/embeddings
(:& 2embedding_2/embeddings
(:&< 2embedding_1/embeddings
#:!	`?2dcn/dense/kernel
:?2dcn/dense/bias
&:$
??2dcn/dense_1/kernel
:?2dcn/dense_1/bias
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
:
Vlookup_table
W	keras_api"
_tf_keras_layer
?
%
embeddings
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
?
\layer_metrics

]layers
3trainable_variables
4regularization_losses
5	variables
^layer_regularization_losses
_metrics
`non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
alookup_table
b	keras_api"
_tf_keras_layer
?
'
embeddings
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
?
glayer_metrics

hlayers
9trainable_variables
:regularization_losses
;	variables
ilayer_regularization_losses
jmetrics
knon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:
llookup_table
m	keras_api"
_tf_keras_layer
?
&
embeddings
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
?
rlayer_metrics

slayers
?trainable_variables
@regularization_losses
A	variables
tlayer_regularization_losses
umetrics
vnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
wlayer_metrics

xlayers
Ctrainable_variables
Dregularization_losses
E	variables
ylayer_regularization_losses
zmetrics
{non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
|layer_metrics

}layers
Gtrainable_variables
Hregularization_losses
I	variables
~layer_regularization_losses
metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
*
PRMSE"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
P0"
trackable_list_wrapper
 "
trackable_list_wrapper
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
'
%0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
%0"
trackable_list_wrapper
?
?layer_metrics
?layers
Xtrainable_variables
Yregularization_losses
Z	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
?
?layer_metrics
?layers
ctrainable_variables
dregularization_losses
e	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
V
?_initializer
?_create_resource
?_initialize
?_destroy_resourceR 
"
_generic_user_object
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
?
?layer_metrics
?layers
ntrainable_variables
oregularization_losses
p	variables
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
*:(	?2Adam/dcn/dense_2/kernel/m
#:!2Adam/dcn/dense_2/bias/m
,:*	? 2Adam/embedding/embeddings/m
-:+ 2Adam/embedding_2/embeddings/m
-:+< 2Adam/embedding_1/embeddings/m
(:&	`?2Adam/dcn/dense/kernel/m
": ?2Adam/dcn/dense/bias/m
+:)
??2Adam/dcn/dense_1/kernel/m
$:"?2Adam/dcn/dense_1/bias/m
*:(	?2Adam/dcn/dense_2/kernel/v
#:!2Adam/dcn/dense_2/bias/v
,:*	? 2Adam/embedding/embeddings/v
-:+ 2Adam/embedding_2/embeddings/v
-:+< 2Adam/embedding_1/embeddings/v
(:&	`?2Adam/dcn/dense/kernel/v
": ?2Adam/dcn/dense/bias/v
+:)
??2Adam/dcn/dense_1/kernel/v
$:"?2Adam/dcn/dense_1/bias/v
?2?
"__inference_dcn_layer_call_fn_1737
"__inference_dcn_layer_call_fn_2092
"__inference_dcn_layer_call_fn_2129
"__inference_dcn_layer_call_fn_1922?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
=__inference_dcn_layer_call_and_return_conditional_losses_2184
=__inference_dcn_layer_call_and_return_conditional_losses_2239
=__inference_dcn_layer_call_and_return_conditional_losses_1966
=__inference_dcn_layer_call_and_return_conditional_losses_2010?
???
FullArgSpec+
args#? 
jself

jfeatures

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference__wrapped_model_1295PlanName	eemGendereemState"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_2_layer_call_fn_2248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_2_layer_call_and_return_conditional_losses_2258?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec\
argsT?Q
jself
jlabels
jpredictions
jsample_weight

jtraining
jcompute_metrics
varargs
 
varkw
 
defaults?

 
p 
p

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_2055PlanName	eemGendereemState"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_1329
)__inference_sequential_layer_call_fn_2269
)__inference_sequential_layer_call_fn_2280
)__inference_sequential_layer_call_fn_1381?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_2293
D__inference_sequential_layer_call_and_return_conditional_losses_2306
D__inference_sequential_layer_call_and_return_conditional_losses_1392
D__inference_sequential_layer_call_and_return_conditional_losses_1403?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_1_layer_call_fn_1545
+__inference_sequential_1_layer_call_fn_2317
+__inference_sequential_1_layer_call_fn_2328
+__inference_sequential_1_layer_call_fn_1597?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_1_layer_call_and_return_conditional_losses_2341
F__inference_sequential_1_layer_call_and_return_conditional_losses_2354
F__inference_sequential_1_layer_call_and_return_conditional_losses_1608
F__inference_sequential_1_layer_call_and_return_conditional_losses_1619?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_2_layer_call_fn_1437
+__inference_sequential_2_layer_call_fn_2365
+__inference_sequential_2_layer_call_fn_2376
+__inference_sequential_2_layer_call_fn_1489?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_sequential_2_layer_call_and_return_conditional_losses_2389
F__inference_sequential_2_layer_call_and_return_conditional_losses_2402
F__inference_sequential_2_layer_call_and_return_conditional_losses_1500
F__inference_sequential_2_layer_call_and_return_conditional_losses_1511?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_2411?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_2422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_2431?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_2442?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_embedding_layer_call_fn_2449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_embedding_layer_call_and_return_conditional_losses_2458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_1_layer_call_fn_2465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_1_layer_call_and_return_conditional_losses_2474?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_2_layer_call_fn_2481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_2_layer_call_and_return_conditional_losses_2490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_2495?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2503?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2508?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2513?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2521?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2526?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2531?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2539?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2544?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_85
__inference__creator_2495?

? 
? "? 5
__inference__creator_2513?

? 
? "? 5
__inference__creator_2531?

? 
? "? 7
__inference__destroyer_2508?

? 
? "? 7
__inference__destroyer_2526?

? 
? "? 7
__inference__destroyer_2544?

? 
? "? @
__inference__initializer_2503V???

? 
? "? @
__inference__initializer_2521a???

? 
? "? @
__inference__initializer_2539l???

? 
? "? ?
__inference__wrapped_model_1295?V?%a?'l?&()*+???
???
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????
? "3?0
.
output_1"?
output_1??????????
=__inference_dcn_layer_call_and_return_conditional_losses_1966?V?%a?'l?&()*+???
???
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????
p 
? "%?"
?
0?????????
? ?
=__inference_dcn_layer_call_and_return_conditional_losses_2010?V?%a?'l?&()*+???
???
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????
p
? "%?"
?
0?????????
? ?
=__inference_dcn_layer_call_and_return_conditional_losses_2184?V?%a?'l?&()*+???
???
???
3
PlanName'?$
features/PlanName?????????
5
	eemGender(?%
features/eemGender?????????
3
eemState'?$
features/eemState?????????
p 
? "%?"
?
0?????????
? ?
=__inference_dcn_layer_call_and_return_conditional_losses_2239?V?%a?'l?&()*+???
???
???
3
PlanName'?$
features/PlanName?????????
5
	eemGender(?%
features/eemGender?????????
3
eemState'?$
features/eemState?????????
p
? "%?"
?
0?????????
? ?
"__inference_dcn_layer_call_fn_1737?V?%a?'l?&()*+???
???
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????
p 
? "???????????
"__inference_dcn_layer_call_fn_1922?V?%a?'l?&()*+???
???
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????
p
? "???????????
"__inference_dcn_layer_call_fn_2092?V?%a?'l?&()*+???
???
???
3
PlanName'?$
features/PlanName?????????
5
	eemGender(?%
features/eemGender?????????
3
eemState'?$
features/eemState?????????
p 
? "???????????
"__inference_dcn_layer_call_fn_2129?V?%a?'l?&()*+???
???
???
3
PlanName'?$
features/PlanName?????????
5
	eemGender(?%
features/eemGender?????????
3
eemState'?$
features/eemState?????????
p
? "???????????
A__inference_dense_1_layer_call_and_return_conditional_losses_2442^*+0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? {
&__inference_dense_1_layer_call_fn_2431Q*+0?-
&?#
!?
inputs??????????
? "????????????
A__inference_dense_2_layer_call_and_return_conditional_losses_2258]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_dense_2_layer_call_fn_2248P0?-
&?#
!?
inputs??????????
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_2422]()/?,
%?"
 ?
inputs?????????`
? "&?#
?
0??????????
? x
$__inference_dense_layer_call_fn_2411P()/?,
%?"
 ?
inputs?????????`
? "????????????
E__inference_embedding_1_layer_call_and_return_conditional_losses_2474W'+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? x
*__inference_embedding_1_layer_call_fn_2465J'+?(
!?
?
inputs?????????	
? "?????????? ?
E__inference_embedding_2_layer_call_and_return_conditional_losses_2490W&+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? x
*__inference_embedding_2_layer_call_fn_2481J&+?(
!?
?
inputs?????????	
? "?????????? ?
C__inference_embedding_layer_call_and_return_conditional_losses_2458W%+?(
!?
?
inputs?????????	
? "%?"
?
0????????? 
? v
(__inference_embedding_layer_call_fn_2449J%+?(
!?
?
inputs?????????	
? "?????????? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1608qa?'B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_1619qa?'B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_2341ba?'3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_1_layer_call_and_return_conditional_losses_2354ba?'3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
+__inference_sequential_1_layer_call_fn_1545da?'B??
8?5
+?(
string_lookup_1_input?????????
p 

 
? "?????????? ?
+__inference_sequential_1_layer_call_fn_1597da?'B??
8?5
+?(
string_lookup_1_input?????????
p

 
? "?????????? ?
+__inference_sequential_1_layer_call_fn_2317Ua?'3?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
+__inference_sequential_1_layer_call_fn_2328Ua?'3?0
)?&
?
inputs?????????
p

 
? "?????????? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1500ql?&B??
8?5
+?(
string_lookup_2_input?????????
p 

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_1511ql?&B??
8?5
+?(
string_lookup_2_input?????????
p

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_2389bl?&3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
F__inference_sequential_2_layer_call_and_return_conditional_losses_2402bl?&3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
+__inference_sequential_2_layer_call_fn_1437dl?&B??
8?5
+?(
string_lookup_2_input?????????
p 

 
? "?????????? ?
+__inference_sequential_2_layer_call_fn_1489dl?&B??
8?5
+?(
string_lookup_2_input?????????
p

 
? "?????????? ?
+__inference_sequential_2_layer_call_fn_2365Ul?&3?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
+__inference_sequential_2_layer_call_fn_2376Ul?&3?0
)?&
?
inputs?????????
p

 
? "?????????? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1392oV?%@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1403oV?%@?=
6?3
)?&
string_lookup_input?????????
p

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2293bV?%3?0
)?&
?
inputs?????????
p 

 
? "%?"
?
0????????? 
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2306bV?%3?0
)?&
?
inputs?????????
p

 
? "%?"
?
0????????? 
? ?
)__inference_sequential_layer_call_fn_1329bV?%@?=
6?3
)?&
string_lookup_input?????????
p 

 
? "?????????? ?
)__inference_sequential_layer_call_fn_1381bV?%@?=
6?3
)?&
string_lookup_input?????????
p

 
? "?????????? ?
)__inference_sequential_layer_call_fn_2269UV?%3?0
)?&
?
inputs?????????
p 

 
? "?????????? ?
)__inference_sequential_layer_call_fn_2280UV?%3?0
)?&
?
inputs?????????
p

 
? "?????????? ?
"__inference_signature_wrapper_2055?V?%a?'l?&()*+???
? 
???
*
PlanName?
PlanName?????????
,
	eemGender?
	eemGender?????????
*
eemState?
eemState?????????"3?0
.
output_1"?
output_1?????????