       гK"	  Аue╓Abrain.Event:2░H▀░q     О╞iЕ	ЛГue╓A"гу
[
xPlaceholder*
dtype0*
shape: */
_output_shapes
:           
S
yPlaceholder*
dtype0*
shape: *'
_output_shapes
:         

p
train_cnn/Reshape/shapeConst*
dtype0*%
valueB"               *
_output_shapes
:
А
train_cnn/ReshapeReshapextrain_cnn/Reshape/shape*/
_output_shapes
:           *
T0*
Tshape0
С
ConvNet/conv1/WVariable*
dtype0*
shape:@*
	container *
shared_name *&
_output_shapes
:@
м
/ConvNet/conv1/W/Initializer/random_normal/shapeConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*%
valueB"         @   *
_output_shapes
:
Ч
.ConvNet/conv1/W/Initializer/random_normal/meanConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *    *
_output_shapes
: 
Щ
0ConvNet/conv1/W/Initializer/random_normal/stddevConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *oГ:*
_output_shapes
: 
В
>ConvNet/conv1/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/ConvNet/conv1/W/Initializer/random_normal/shape*&
_output_shapes
:@*
dtype0*
seed2*

seed**
T0*"
_class
loc:@ConvNet/conv1/W
√
-ConvNet/conv1/W/Initializer/random_normal/mulMul>ConvNet/conv1/W/Initializer/random_normal/RandomStandardNormal0ConvNet/conv1/W/Initializer/random_normal/stddev*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
ф
)ConvNet/conv1/W/Initializer/random_normalAdd-ConvNet/conv1/W/Initializer/random_normal/mul.ConvNet/conv1/W/Initializer/random_normal/mean*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
┌
ConvNet/conv1/W/AssignAssignConvNet/conv1/W)ConvNet/conv1/W/Initializer/random_normal*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
Ж
ConvNet/conv1/W/readIdentityConvNet/conv1/W*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
y
ConvNet/conv1/bVariable*
dtype0*
shape:@*
	container *
shared_name *
_output_shapes
:@
Т
!ConvNet/conv1/b/Initializer/ConstConst*
dtype0*"
_class
loc:@ConvNet/conv1/b*
valueB@*    *
_output_shapes
:@
╞
ConvNet/conv1/b/AssignAssignConvNet/conv1/b!ConvNet/conv1/b/Initializer/Const*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
z
ConvNet/conv1/b/readIdentityConvNet/conv1/b*"
_class
loc:@ConvNet/conv1/b*
T0*
_output_shapes
:@
т
train_cnn/ConvNet/conv1/Conv2DConv2Dtrain_cnn/ReshapeConvNet/conv1/W/read*/
_output_shapes
:           @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Т
train_cnn/ConvNet/conv1/addAddtrain_cnn/ConvNet/conv1/Conv2DConvNet/conv1/b/read*
T0*/
_output_shapes
:           @
{
train_cnn/ConvNet/conv1/ReluRelutrain_cnn/ConvNet/conv1/add*
T0*/
_output_shapes
:           @
╘
train_cnn/ConvNet/conv1/MaxPoolMaxPooltrain_cnn/ConvNet/conv1/Relu*/
_output_shapes
:         @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
С
ConvNet/conv2/WVariable*
dtype0*
shape:@@*
	container *
shared_name *&
_output_shapes
:@@
м
/ConvNet/conv2/W/Initializer/random_normal/shapeConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*%
valueB"      @   @   *
_output_shapes
:
Ч
.ConvNet/conv2/W/Initializer/random_normal/meanConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*
valueB
 *    *
_output_shapes
: 
Щ
0ConvNet/conv2/W/Initializer/random_normal/stddevConst*
dtype0*"
_class
loc:@ConvNet/conv2/W*
valueB
 *oГ:*
_output_shapes
: 
В
>ConvNet/conv2/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal/ConvNet/conv2/W/Initializer/random_normal/shape*&
_output_shapes
:@@*
dtype0*
seed2*

seed**
T0*"
_class
loc:@ConvNet/conv2/W
√
-ConvNet/conv2/W/Initializer/random_normal/mulMul>ConvNet/conv2/W/Initializer/random_normal/RandomStandardNormal0ConvNet/conv2/W/Initializer/random_normal/stddev*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
ф
)ConvNet/conv2/W/Initializer/random_normalAdd-ConvNet/conv2/W/Initializer/random_normal/mul.ConvNet/conv2/W/Initializer/random_normal/mean*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
┌
ConvNet/conv2/W/AssignAssignConvNet/conv2/W)ConvNet/conv2/W/Initializer/random_normal*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
Ж
ConvNet/conv2/W/readIdentityConvNet/conv2/W*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
y
ConvNet/conv2/bVariable*
dtype0*
shape:@*
	container *
shared_name *
_output_shapes
:@
Т
!ConvNet/conv2/b/Initializer/ConstConst*
dtype0*"
_class
loc:@ConvNet/conv2/b*
valueB@*═╠╠=*
_output_shapes
:@
╞
ConvNet/conv2/b/AssignAssignConvNet/conv2/b!ConvNet/conv2/b/Initializer/Const*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
z
ConvNet/conv2/b/readIdentityConvNet/conv2/b*"
_class
loc:@ConvNet/conv2/b*
T0*
_output_shapes
:@
Ё
train_cnn/ConvNet/conv2/Conv2DConv2Dtrain_cnn/ConvNet/conv1/MaxPoolConvNet/conv2/W/read*/
_output_shapes
:         @*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Т
train_cnn/ConvNet/conv2/addAddtrain_cnn/ConvNet/conv2/Conv2DConvNet/conv2/b/read*
T0*/
_output_shapes
:         @
{
train_cnn/ConvNet/conv2/ReluRelutrain_cnn/ConvNet/conv2/add*
T0*/
_output_shapes
:         @
╘
train_cnn/ConvNet/conv2/MaxPoolMaxPooltrain_cnn/ConvNet/conv2/Relu*/
_output_shapes
:         @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
p
train_cnn/ConvNet/Reshape/shapeConst*
dtype0*
valueB"       *
_output_shapes
:
з
train_cnn/ConvNet/ReshapeReshapetrain_cnn/ConvNet/conv2/MaxPooltrain_cnn/ConvNet/Reshape/shape*(
_output_shapes
:         А *
T0*
Tshape0
Г
ConvNet/fc1/WVariable*
dtype0*
shape:
А А*
	container *
shared_name * 
_output_shapes
:
А А
а
-ConvNet/fc1/W/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB"   А  *
_output_shapes
:
У
,ConvNet/fc1/W/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB
 *    *
_output_shapes
: 
Х
.ConvNet/fc1/W/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@ConvNet/fc1/W*
valueB
 *oГ:*
_output_shapes
: 
Ў
<ConvNet/fc1/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-ConvNet/fc1/W/Initializer/random_normal/shape* 
_output_shapes
:
А А*
dtype0*
seed2,*

seed**
T0* 
_class
loc:@ConvNet/fc1/W
э
+ConvNet/fc1/W/Initializer/random_normal/mulMul<ConvNet/fc1/W/Initializer/random_normal/RandomStandardNormal.ConvNet/fc1/W/Initializer/random_normal/stddev* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
А А
╓
'ConvNet/fc1/W/Initializer/random_normalAdd+ConvNet/fc1/W/Initializer/random_normal/mul,ConvNet/fc1/W/Initializer/random_normal/mean* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
А А
╠
ConvNet/fc1/W/AssignAssignConvNet/fc1/W'ConvNet/fc1/W/Initializer/random_normal*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
z
ConvNet/fc1/W/readIdentityConvNet/fc1/W* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
А А
y
ConvNet/fc1/bVariable*
dtype0*
shape:А*
	container *
shared_name *
_output_shapes	
:А
Р
ConvNet/fc1/b/Initializer/ConstConst*
dtype0* 
_class
loc:@ConvNet/fc1/b*
valueBА*    *
_output_shapes	
:А
┐
ConvNet/fc1/b/AssignAssignConvNet/fc1/bConvNet/fc1/b/Initializer/Const*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
u
ConvNet/fc1/b/readIdentityConvNet/fc1/b* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:А
о
train_cnn/ConvNet/fc1/MatMulMatMultrain_cnn/ConvNet/ReshapeConvNet/fc1/W/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         А
Е
train_cnn/ConvNet/fc1/addAddtrain_cnn/ConvNet/fc1/MatMulConvNet/fc1/b/read*
T0*(
_output_shapes
:         А
p
train_cnn/ConvNet/fc1/ReluRelutrain_cnn/ConvNet/fc1/add*
T0*(
_output_shapes
:         А
Г
ConvNet/fc2/WVariable*
dtype0*
shape:
А└*
	container *
shared_name * 
_output_shapes
:
А└
а
-ConvNet/fc2/W/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB"А  └   *
_output_shapes
:
У
,ConvNet/fc2/W/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB
 *    *
_output_shapes
: 
Х
.ConvNet/fc2/W/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@ConvNet/fc2/W*
valueB
 *oГ:*
_output_shapes
: 
Ў
<ConvNet/fc2/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-ConvNet/fc2/W/Initializer/random_normal/shape* 
_output_shapes
:
А└*
dtype0*
seed2<*

seed**
T0* 
_class
loc:@ConvNet/fc2/W
э
+ConvNet/fc2/W/Initializer/random_normal/mulMul<ConvNet/fc2/W/Initializer/random_normal/RandomStandardNormal.ConvNet/fc2/W/Initializer/random_normal/stddev* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
А└
╓
'ConvNet/fc2/W/Initializer/random_normalAdd+ConvNet/fc2/W/Initializer/random_normal/mul,ConvNet/fc2/W/Initializer/random_normal/mean* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
А└
╠
ConvNet/fc2/W/AssignAssignConvNet/fc2/W'ConvNet/fc2/W/Initializer/random_normal*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
z
ConvNet/fc2/W/readIdentityConvNet/fc2/W* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
А└
y
ConvNet/fc2/bVariable*
dtype0*
shape:└*
	container *
shared_name *
_output_shapes	
:└
Р
ConvNet/fc2/b/Initializer/ConstConst*
dtype0* 
_class
loc:@ConvNet/fc2/b*
valueB└*    *
_output_shapes	
:└
┐
ConvNet/fc2/b/AssignAssignConvNet/fc2/bConvNet/fc2/b/Initializer/Const*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
u
ConvNet/fc2/b/readIdentityConvNet/fc2/b* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:└
п
train_cnn/ConvNet/fc2/MatMulMatMultrain_cnn/ConvNet/fc1/ReluConvNet/fc2/W/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:         └
Е
train_cnn/ConvNet/fc2/addAddtrain_cnn/ConvNet/fc2/MatMulConvNet/fc2/b/read*
T0*(
_output_shapes
:         └
p
train_cnn/ConvNet/fc2/ReluRelutrain_cnn/ConvNet/fc2/add*
T0*(
_output_shapes
:         └
Б
ConvNet/fc3/WVariable*
dtype0*
shape:	└
*
	container *
shared_name *
_output_shapes
:	└

а
-ConvNet/fc3/W/Initializer/random_normal/shapeConst*
dtype0* 
_class
loc:@ConvNet/fc3/W*
valueB"└   
   *
_output_shapes
:
У
,ConvNet/fc3/W/Initializer/random_normal/meanConst*
dtype0* 
_class
loc:@ConvNet/fc3/W*
valueB
 *    *
_output_shapes
: 
Х
.ConvNet/fc3/W/Initializer/random_normal/stddevConst*
dtype0* 
_class
loc:@ConvNet/fc3/W*
valueB
 *oГ:*
_output_shapes
: 
ї
<ConvNet/fc3/W/Initializer/random_normal/RandomStandardNormalRandomStandardNormal-ConvNet/fc3/W/Initializer/random_normal/shape*
_output_shapes
:	└
*
dtype0*
seed2L*

seed**
T0* 
_class
loc:@ConvNet/fc3/W
ь
+ConvNet/fc3/W/Initializer/random_normal/mulMul<ConvNet/fc3/W/Initializer/random_normal/RandomStandardNormal.ConvNet/fc3/W/Initializer/random_normal/stddev* 
_class
loc:@ConvNet/fc3/W*
T0*
_output_shapes
:	└

╒
'ConvNet/fc3/W/Initializer/random_normalAdd+ConvNet/fc3/W/Initializer/random_normal/mul,ConvNet/fc3/W/Initializer/random_normal/mean* 
_class
loc:@ConvNet/fc3/W*
T0*
_output_shapes
:	└

╦
ConvNet/fc3/W/AssignAssignConvNet/fc3/W'ConvNet/fc3/W/Initializer/random_normal*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

y
ConvNet/fc3/W/readIdentityConvNet/fc3/W* 
_class
loc:@ConvNet/fc3/W*
T0*
_output_shapes
:	└

w
ConvNet/fc3/bVariable*
dtype0*
shape:
*
	container *
shared_name *
_output_shapes
:

О
ConvNet/fc3/b/Initializer/ConstConst*
dtype0* 
_class
loc:@ConvNet/fc3/b*
valueB
*    *
_output_shapes
:

╛
ConvNet/fc3/b/AssignAssignConvNet/fc3/bConvNet/fc3/b/Initializer/Const*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

t
ConvNet/fc3/b/readIdentityConvNet/fc3/b* 
_class
loc:@ConvNet/fc3/b*
T0*
_output_shapes
:

о
train_cnn/ConvNet/fc3/MatMulMatMultrain_cnn/ConvNet/fc2/ReluConvNet/fc3/W/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:         

Д
train_cnn/ConvNet/fc3/addAddtrain_cnn/ConvNet/fc3/MatMulConvNet/fc3/b/read*
T0*'
_output_shapes
:         

q
*train_cnn/ConvNet/fc3/HistogramSummary/tagConst*
dtype0*
valueB Blogits*
_output_shapes
: 
в
&train_cnn/ConvNet/fc3/HistogramSummaryHistogramSummary*train_cnn/ConvNet/fc3/HistogramSummary/tagtrain_cnn/ConvNet/fc3/add*
T0*
_output_shapes
: 
m
!train_cnn/cross-entropy-loss/CastCasty*

DstT0*

SrcT0*'
_output_shapes
:         

c
!train_cnn/cross-entropy-loss/RankConst*
dtype0*
value	B :*
_output_shapes
: 
{
"train_cnn/cross-entropy-loss/ShapeShapetrain_cnn/ConvNet/fc3/add*
out_type0*
T0*
_output_shapes
:
e
#train_cnn/cross-entropy-loss/Rank_1Const*
dtype0*
value	B :*
_output_shapes
: 
}
$train_cnn/cross-entropy-loss/Shape_1Shapetrain_cnn/ConvNet/fc3/add*
out_type0*
T0*
_output_shapes
:
d
"train_cnn/cross-entropy-loss/Sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
С
 train_cnn/cross-entropy-loss/SubSub#train_cnn/cross-entropy-loss/Rank_1"train_cnn/cross-entropy-loss/Sub/y*
T0*
_output_shapes
: 
М
(train_cnn/cross-entropy-loss/Slice/beginPack train_cnn/cross-entropy-loss/Sub*
N*
T0*
_output_shapes
:*

axis 
q
'train_cnn/cross-entropy-loss/Slice/sizeConst*
dtype0*
valueB:*
_output_shapes
:
╓
"train_cnn/cross-entropy-loss/SliceSlice$train_cnn/cross-entropy-loss/Shape_1(train_cnn/cross-entropy-loss/Slice/begin'train_cnn/cross-entropy-loss/Slice/size*
Index0*
T0*
_output_shapes
:
p
.train_cnn/cross-entropy-loss/concat/concat_dimConst*
dtype0*
value	B : *
_output_shapes
: 

,train_cnn/cross-entropy-loss/concat/values_0Const*
dtype0*
valueB:
         *
_output_shapes
:
▌
#train_cnn/cross-entropy-loss/concatConcat.train_cnn/cross-entropy-loss/concat/concat_dim,train_cnn/cross-entropy-loss/concat/values_0"train_cnn/cross-entropy-loss/Slice*
_output_shapes
:*
T0*
N
╕
$train_cnn/cross-entropy-loss/ReshapeReshapetrain_cnn/ConvNet/fc3/add#train_cnn/cross-entropy-loss/concat*0
_output_shapes
:                  *
T0*
Tshape0
e
#train_cnn/cross-entropy-loss/Rank_2Const*
dtype0*
value	B :*
_output_shapes
: 
Е
$train_cnn/cross-entropy-loss/Shape_2Shape!train_cnn/cross-entropy-loss/Cast*
out_type0*
T0*
_output_shapes
:
f
$train_cnn/cross-entropy-loss/Sub_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
Х
"train_cnn/cross-entropy-loss/Sub_1Sub#train_cnn/cross-entropy-loss/Rank_2$train_cnn/cross-entropy-loss/Sub_1/y*
T0*
_output_shapes
: 
Р
*train_cnn/cross-entropy-loss/Slice_1/beginPack"train_cnn/cross-entropy-loss/Sub_1*
N*
T0*
_output_shapes
:*

axis 
s
)train_cnn/cross-entropy-loss/Slice_1/sizeConst*
dtype0*
valueB:*
_output_shapes
:
▄
$train_cnn/cross-entropy-loss/Slice_1Slice$train_cnn/cross-entropy-loss/Shape_2*train_cnn/cross-entropy-loss/Slice_1/begin)train_cnn/cross-entropy-loss/Slice_1/size*
Index0*
T0*
_output_shapes
:
r
0train_cnn/cross-entropy-loss/concat_1/concat_dimConst*
dtype0*
value	B : *
_output_shapes
: 
Б
.train_cnn/cross-entropy-loss/concat_1/values_0Const*
dtype0*
valueB:
         *
_output_shapes
:
х
%train_cnn/cross-entropy-loss/concat_1Concat0train_cnn/cross-entropy-loss/concat_1/concat_dim.train_cnn/cross-entropy-loss/concat_1/values_0$train_cnn/cross-entropy-loss/Slice_1*
_output_shapes
:*
T0*
N
─
&train_cnn/cross-entropy-loss/Reshape_1Reshape!train_cnn/cross-entropy-loss/Cast%train_cnn/cross-entropy-loss/concat_1*0
_output_shapes
:                  *
T0*
Tshape0
т
)train_cnn/cross-entropy-loss/crossentropySoftmaxCrossEntropyWithLogits$train_cnn/cross-entropy-loss/Reshape&train_cnn/cross-entropy-loss/Reshape_1*
T0*?
_output_shapes-
+:         :                  
f
$train_cnn/cross-entropy-loss/Sub_2/yConst*
dtype0*
value	B :*
_output_shapes
: 
У
"train_cnn/cross-entropy-loss/Sub_2Sub!train_cnn/cross-entropy-loss/Rank$train_cnn/cross-entropy-loss/Sub_2/y*
T0*
_output_shapes
: 
t
*train_cnn/cross-entropy-loss/Slice_2/beginConst*
dtype0*
valueB: *
_output_shapes
:
П
)train_cnn/cross-entropy-loss/Slice_2/sizePack"train_cnn/cross-entropy-loss/Sub_2*
N*
T0*
_output_shapes
:*

axis 
у
$train_cnn/cross-entropy-loss/Slice_2Slice"train_cnn/cross-entropy-loss/Shape*train_cnn/cross-entropy-loss/Slice_2/begin)train_cnn/cross-entropy-loss/Slice_2/size*
Index0*
T0*#
_output_shapes
:         
╛
&train_cnn/cross-entropy-loss/Reshape_2Reshape)train_cnn/cross-entropy-loss/crossentropy$train_cnn/cross-entropy-loss/Slice_2*#
_output_shapes
:         *
T0*
Tshape0
l
"train_cnn/cross-entropy-loss/ConstConst*
dtype0*
valueB: *
_output_shapes
:
│
!train_cnn/cross-entropy-loss/lossMean&train_cnn/cross-entropy-loss/Reshape_2"train_cnn/cross-entropy-loss/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
В
/train_cnn/cross-entropy-loss/ScalarSummary/tagsConst*
dtype0*#
valueB Bcross-entropy loss*
_output_shapes
: 
░
*train_cnn/cross-entropy-loss/ScalarSummaryScalarSummary/train_cnn/cross-entropy-loss/ScalarSummary/tags!train_cnn/cross-entropy-loss/loss*
T0*
_output_shapes
: 
e
#train_cnn/accuracy/ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
Е
train_cnn/accuracy/ArgMaxArgMaxy#train_cnn/accuracy/ArgMax/dimension*#
_output_shapes
:         *
T0*

Tidx0
g
%train_cnn/accuracy/ArgMax_1/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
б
train_cnn/accuracy/ArgMax_1ArgMaxtrain_cnn/ConvNet/fc3/add%train_cnn/accuracy/ArgMax_1/dimension*#
_output_shapes
:         *
T0*

Tidx0
З
train_cnn/accuracy/EqualEqualtrain_cnn/accuracy/ArgMaxtrain_cnn/accuracy/ArgMax_1*
T0	*#
_output_shapes
:         
v
train_cnn/accuracy/CastCasttrain_cnn/accuracy/Equal*

DstT0*

SrcT0
*#
_output_shapes
:         
b
train_cnn/accuracy/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Р
train_cnn/accuracy/MeanMeantrain_cnn/accuracy/Casttrain_cnn/accuracy/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
n
%train_cnn/accuracy/ScalarSummary/tagsConst*
dtype0*
valueB Baccuracy*
_output_shapes
: 
Т
 train_cnn/accuracy/ScalarSummaryScalarSummary%train_cnn/accuracy/ScalarSummary/tagstrain_cnn/accuracy/Mean*
T0*
_output_shapes
: 
╩
#train_cnn/MergeSummary/MergeSummaryMergeSummary&train_cnn/ConvNet/fc3/HistogramSummary*train_cnn/cross-entropy-loss/ScalarSummary train_cnn/accuracy/ScalarSummary*
_output_shapes
: *
N
\
train_cnn/gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
^
train_cnn/gradients/ConstConst*
dtype0*
valueB
 *  А?*
_output_shapes
: 
w
train_cnn/gradients/FillFilltrain_cnn/gradients/Shapetrain_cnn/gradients/Const*
T0*
_output_shapes
: 
Т
Htrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
ф
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ReshapeReshapetrain_cnn/gradients/FillHtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
ж
@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ShapeShape&train_cnn/cross-entropy-loss/Reshape_2*
out_type0*
T0*
_output_shapes
:
Н
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/TileTileBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Reshape@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
и
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_1Shape&train_cnn/cross-entropy-loss/Reshape_2*
out_type0*
T0*
_output_shapes
:
Е
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_2Const*
dtype0*
valueB *
_output_shapes
: 
К
@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
Л
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ProdProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_1@train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
М
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
П
Atrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Prod_1ProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Shape_2Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Ж
Dtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
ў
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/MaximumMaximumAtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Prod_1Dtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum/y*
T0*
_output_shapes
: 
Ё
Ctrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/floordivDiv?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/ProdBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Maximum*
T0*
_output_shapes
: 
╝
?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/CastCastCtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/floordiv*

DstT0*

SrcT0*
_output_shapes
: 
∙
Btrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/truedivDiv?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Tile?train_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/Cast*
T0*#
_output_shapes
:         
о
Etrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ShapeShape)train_cnn/cross-entropy-loss/crossentropy*
out_type0*
T0*
_output_shapes
:
Щ
Gtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ReshapeReshapeBtrain_cnn/gradients/train_cnn/cross-entropy-loss/loss_grad/truedivEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
У
train_cnn/gradients/zeros_like	ZerosLike+train_cnn/cross-entropy-loss/crossentropy:1*
T0*0
_output_shapes
:                  
Ь
Qtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims/dimConst*
dtype0*
valueB :
         *
_output_shapes
: 
╡
Mtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims
ExpandDimsGtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_2_grad/ReshapeQtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:         
Д
Ftrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/mulMulMtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/ExpandDims+train_cnn/cross-entropy-loss/crossentropy:1*
T0*0
_output_shapes
:                  
Ь
Ctrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ShapeShapetrain_cnn/ConvNet/fc3/add*
out_type0*
T0*
_output_shapes
:
Э
Etrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeReshapeFtrain_cnn/gradients/train_cnn/cross-entropy-loss/crossentropy_grad/mulCtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/Shape*'
_output_shapes
:         
*
T0*
Tshape0
Ф
8train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/ShapeShapetrain_cnn/ConvNet/fc3/MatMul*
out_type0*
T0*
_output_shapes
:
Д
:train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Shape_1Const*
dtype0*
valueB:
*
_output_shapes
:
Ф
Htrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/BroadcastGradientArgsBroadcastGradientArgs8train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Shape:train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
О
6train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/SumSumEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeHtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ў
:train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/ReshapeReshape6train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Sum8train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Shape*'
_output_shapes
:         
*
T0*
Tshape0
Т
8train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Sum_1SumEtrain_cnn/gradients/train_cnn/cross-entropy-loss/Reshape_grad/ReshapeJtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ё
<train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape_1Reshape8train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Sum_1:train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Shape_1*
_output_shapes
:
*
T0*
Tshape0
╟
Ctrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/group_depsNoOp;^train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape=^train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape_1
┌
Ktrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/control_dependencyIdentity:train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/ReshapeD^train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/group_deps*M
_classC
A?loc:@train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape*
T0*'
_output_shapes
:         

╙
Mtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/control_dependency_1Identity<train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape_1D^train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/Reshape_1*
T0*
_output_shapes
:

А
<train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMulMatMulKtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/control_dependencyConvNet/fc3/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         └
Б
>train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/fc2/ReluKtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	└

╬
Ftrain_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul?^train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul_1
х
Ntrain_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMulG^train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul*
T0*(
_output_shapes
:         └
т
Ptrain_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul_1G^train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	└

ч
<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradReluGradNtrain_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/control_dependencytrain_cnn/ConvNet/fc2/Relu*
T0*(
_output_shapes
:         └
Ф
8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ShapeShapetrain_cnn/ConvNet/fc2/MatMul*
out_type0*
T0*
_output_shapes
:
Е
:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1Const*
dtype0*
valueB:└*
_output_shapes
:
Ф
Htrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgsBroadcastGradientArgs8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Е
6train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/SumSum<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradHtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
°
:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ReshapeReshape6train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape*(
_output_shapes
:         └*
T0*
Tshape0
Й
8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum_1Sum<train_cnn/gradients/train_cnn/ConvNet/fc2/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ё
<train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1Reshape8train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Sum_1:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Shape_1*
_output_shapes	
:└*
T0*
Tshape0
╟
Ctrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_depsNoOp;^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape=^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1
█
Ktrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependencyIdentity:train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/ReshapeD^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_deps*M
_classC
A?loc:@train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape*
T0*(
_output_shapes
:         └
╘
Mtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency_1Identity<train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1D^train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/Reshape_1*
T0*
_output_shapes	
:└
А
<train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMulMatMulKtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependencyConvNet/fc2/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         А
В
>train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/fc1/ReluKtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
А└
╬
Ftrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul?^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1
х
Ntrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMulG^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul*
T0*(
_output_shapes
:         А
у
Ptrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1G^train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
А└
ч
<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradReluGradNtrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependencytrain_cnn/ConvNet/fc1/Relu*
T0*(
_output_shapes
:         А
Ф
8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ShapeShapetrain_cnn/ConvNet/fc1/MatMul*
out_type0*
T0*
_output_shapes
:
Е
:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1Const*
dtype0*
valueB:А*
_output_shapes
:
Ф
Htrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgsBroadcastGradientArgs8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Е
6train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/SumSum<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradHtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
°
:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ReshapeReshape6train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape*(
_output_shapes
:         А*
T0*
Tshape0
Й
8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum_1Sum<train_cnn/gradients/train_cnn/ConvNet/fc1/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
ё
<train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1Reshape8train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Sum_1:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Shape_1*
_output_shapes	
:А*
T0*
Tshape0
╟
Ctrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_depsNoOp;^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape=^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1
█
Ktrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependencyIdentity:train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/ReshapeD^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_deps*M
_classC
A?loc:@train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape*
T0*(
_output_shapes
:         А
╘
Mtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency_1Identity<train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1D^train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/Reshape_1*
T0*
_output_shapes	
:А
А
<train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMulMatMulKtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependencyConvNet/fc1/W/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:         А 
Б
>train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1MatMultrain_cnn/ConvNet/ReshapeKtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
А А
╬
Ftrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul?^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1
х
Ntrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMulG^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul*
T0*(
_output_shapes
:         А 
у
Ptrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1G^train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
А А
Ч
8train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/ShapeShapetrain_cnn/ConvNet/conv2/MaxPool*
out_type0*
T0*
_output_shapes
:
Ч
:train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/ReshapeReshapeNtrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency8train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/Shape*/
_output_shapes
:         @*
T0*
Tshape0
┌
Dtrain_cnn/gradients/train_cnn/ConvNet/conv2/MaxPool_grad/MaxPoolGradMaxPoolGradtrain_cnn/ConvNet/conv2/Relutrain_cnn/ConvNet/conv2/MaxPool:train_cnn/gradients/train_cnn/ConvNet/Reshape_grad/Reshape*/
_output_shapes
:         @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
ш
>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradReluGradDtrain_cnn/gradients/train_cnn/ConvNet/conv2/MaxPool_grad/MaxPoolGradtrain_cnn/ConvNet/conv2/Relu*
T0*/
_output_shapes
:         @
Ш
:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ShapeShapetrain_cnn/ConvNet/conv2/Conv2D*
out_type0*
T0*
_output_shapes
:
Ж
<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1Const*
dtype0*
valueB:@*
_output_shapes
:
Ъ
Jtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgsBroadcastGradientArgs:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Л
8train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/SumSum>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Е
<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ReshapeReshape8train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape*/
_output_shapes
:         @*
T0*
Tshape0
П
:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum_1Sum>train_cnn/gradients/train_cnn/ConvNet/conv2/Relu_grad/ReluGradLtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ў
>train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1Reshape:train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Sum_1<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
═
Etrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape?^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1
ъ
Mtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/ReshapeF^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape*
T0*/
_output_shapes
:         @
█
Otrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1F^train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/Reshape_1*
T0*
_output_shapes
:@
Ь
=train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/ShapeShapetrain_cnn/ConvNet/conv1/MaxPool*
out_type0*
T0*
_output_shapes
:
▓
Ktrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/ShapeConvNet/conv2/W/readMtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ш
?train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"      @   @   *
_output_shapes
:
Э
Ltrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltertrain_cnn/ConvNet/conv1/MaxPool?train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Shape_1Mtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency*&
_output_shapes
:@@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
Htrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_depsNoOpL^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputM^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilter
О
Ptrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependencyIdentityKtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInputI^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_deps*^
_classT
RPloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:         @
Й
Rtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency_1IdentityLtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilterI^train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/group_deps*_
_classU
SQloc:@train_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@@
Ё
Dtrain_cnn/gradients/train_cnn/ConvNet/conv1/MaxPool_grad/MaxPoolGradMaxPoolGradtrain_cnn/ConvNet/conv1/Relutrain_cnn/ConvNet/conv1/MaxPoolPtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency*/
_output_shapes
:           @*
data_formatNHWC*
paddingSAME*
strides
*
ksize
*
T0
ш
>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradReluGradDtrain_cnn/gradients/train_cnn/ConvNet/conv1/MaxPool_grad/MaxPoolGradtrain_cnn/ConvNet/conv1/Relu*
T0*/
_output_shapes
:           @
Ш
:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ShapeShapetrain_cnn/ConvNet/conv1/Conv2D*
out_type0*
T0*
_output_shapes
:
Ж
<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1Const*
dtype0*
valueB:@*
_output_shapes
:
Ъ
Jtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Л
8train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/SumSum>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradJtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Е
<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ReshapeReshape8train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape*/
_output_shapes
:           @*
T0*
Tshape0
П
:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum_1Sum>train_cnn/gradients/train_cnn/ConvNet/conv1/Relu_grad/ReluGradLtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
Ў
>train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1Reshape:train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Sum_1<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Shape_1*
_output_shapes
:@*
T0*
Tshape0
═
Etrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_depsNoOp=^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape?^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1
ъ
Mtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependencyIdentity<train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/ReshapeF^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_deps*O
_classE
CAloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape*
T0*/
_output_shapes
:           @
█
Otrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency_1Identity>train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1F^train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/group_deps*Q
_classG
ECloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/Reshape_1*
T0*
_output_shapes
:@
О
=train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/ShapeShapetrain_cnn/Reshape*
out_type0*
T0*
_output_shapes
:
▓
Ktrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput=train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/ShapeConvNet/conv1/W/readMtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency*J
_output_shapes8
6:4                                    *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
Ш
?train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Shape_1Const*
dtype0*%
valueB"         @   *
_output_shapes
:
П
Ltrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltertrain_cnn/Reshape?train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Shape_1Mtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency*&
_output_shapes
:@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0
э
Htrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_depsNoOpL^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputM^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilter
О
Ptrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependencyIdentityKtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInputI^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_deps*^
_classT
RPloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropInput*
T0*/
_output_shapes
:           
Й
Rtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependency_1IdentityLtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilterI^train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/group_deps*_
_classU
SQloc:@train_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/Conv2DBackpropFilter*
T0*&
_output_shapes
:@
М
#train_cnn/beta1_power/initial_valueConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *fff?*
_output_shapes
: 
Ы
train_cnn/beta1_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@ConvNet/conv1/W*
shared_name 
╨
train_cnn/beta1_power/AssignAssigntrain_cnn/beta1_power#train_cnn/beta1_power/initial_value*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
В
train_cnn/beta1_power/readIdentitytrain_cnn/beta1_power*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
М
#train_cnn/beta2_power/initial_valueConst*
dtype0*"
_class
loc:@ConvNet/conv1/W*
valueB
 *w╛?*
_output_shapes
: 
Ы
train_cnn/beta2_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@ConvNet/conv1/W*
shared_name 
╨
train_cnn/beta2_power/AssignAssigntrain_cnn/beta2_power#train_cnn/beta2_power/initial_value*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
В
train_cnn/beta2_power/readIdentitytrain_cnn/beta2_power*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
t
train_cnn/zerosConst*
dtype0*%
valueB@*    *&
_output_shapes
:@
─
train_cnn/ConvNet/conv1/W/AdamVariable*
	container *&
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/W*
shared_name 
▐
%train_cnn/ConvNet/conv1/W/Adam/AssignAssigntrain_cnn/ConvNet/conv1/W/Adamtrain_cnn/zeros*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
д
#train_cnn/ConvNet/conv1/W/Adam/readIdentitytrain_cnn/ConvNet/conv1/W/Adam*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
v
train_cnn/zeros_1Const*
dtype0*%
valueB@*    *&
_output_shapes
:@
╞
 train_cnn/ConvNet/conv1/W/Adam_1Variable*
	container *&
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/W*
shared_name 
ф
'train_cnn/ConvNet/conv1/W/Adam_1/AssignAssign train_cnn/ConvNet/conv1/W/Adam_1train_cnn/zeros_1*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
и
%train_cnn/ConvNet/conv1/W/Adam_1/readIdentity train_cnn/ConvNet/conv1/W/Adam_1*"
_class
loc:@ConvNet/conv1/W*
T0*&
_output_shapes
:@
^
train_cnn/zeros_2Const*
dtype0*
valueB@*    *
_output_shapes
:@
м
train_cnn/ConvNet/conv1/b/AdamVariable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/b*
shared_name 
╘
%train_cnn/ConvNet/conv1/b/Adam/AssignAssigntrain_cnn/ConvNet/conv1/b/Adamtrain_cnn/zeros_2*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
Ш
#train_cnn/ConvNet/conv1/b/Adam/readIdentitytrain_cnn/ConvNet/conv1/b/Adam*"
_class
loc:@ConvNet/conv1/b*
T0*
_output_shapes
:@
^
train_cnn/zeros_3Const*
dtype0*
valueB@*    *
_output_shapes
:@
о
 train_cnn/ConvNet/conv1/b/Adam_1Variable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv1/b*
shared_name 
╪
'train_cnn/ConvNet/conv1/b/Adam_1/AssignAssign train_cnn/ConvNet/conv1/b/Adam_1train_cnn/zeros_3*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
Ь
%train_cnn/ConvNet/conv1/b/Adam_1/readIdentity train_cnn/ConvNet/conv1/b/Adam_1*"
_class
loc:@ConvNet/conv1/b*
T0*
_output_shapes
:@
v
train_cnn/zeros_4Const*
dtype0*%
valueB@@*    *&
_output_shapes
:@@
─
train_cnn/ConvNet/conv2/W/AdamVariable*
	container *&
_output_shapes
:@@*
dtype0*
shape:@@*"
_class
loc:@ConvNet/conv2/W*
shared_name 
р
%train_cnn/ConvNet/conv2/W/Adam/AssignAssigntrain_cnn/ConvNet/conv2/W/Adamtrain_cnn/zeros_4*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
д
#train_cnn/ConvNet/conv2/W/Adam/readIdentitytrain_cnn/ConvNet/conv2/W/Adam*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
v
train_cnn/zeros_5Const*
dtype0*%
valueB@@*    *&
_output_shapes
:@@
╞
 train_cnn/ConvNet/conv2/W/Adam_1Variable*
	container *&
_output_shapes
:@@*
dtype0*
shape:@@*"
_class
loc:@ConvNet/conv2/W*
shared_name 
ф
'train_cnn/ConvNet/conv2/W/Adam_1/AssignAssign train_cnn/ConvNet/conv2/W/Adam_1train_cnn/zeros_5*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
и
%train_cnn/ConvNet/conv2/W/Adam_1/readIdentity train_cnn/ConvNet/conv2/W/Adam_1*"
_class
loc:@ConvNet/conv2/W*
T0*&
_output_shapes
:@@
^
train_cnn/zeros_6Const*
dtype0*
valueB@*    *
_output_shapes
:@
м
train_cnn/ConvNet/conv2/b/AdamVariable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv2/b*
shared_name 
╘
%train_cnn/ConvNet/conv2/b/Adam/AssignAssigntrain_cnn/ConvNet/conv2/b/Adamtrain_cnn/zeros_6*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
Ш
#train_cnn/ConvNet/conv2/b/Adam/readIdentitytrain_cnn/ConvNet/conv2/b/Adam*"
_class
loc:@ConvNet/conv2/b*
T0*
_output_shapes
:@
^
train_cnn/zeros_7Const*
dtype0*
valueB@*    *
_output_shapes
:@
о
 train_cnn/ConvNet/conv2/b/Adam_1Variable*
	container *
_output_shapes
:@*
dtype0*
shape:@*"
_class
loc:@ConvNet/conv2/b*
shared_name 
╪
'train_cnn/ConvNet/conv2/b/Adam_1/AssignAssign train_cnn/ConvNet/conv2/b/Adam_1train_cnn/zeros_7*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
Ь
%train_cnn/ConvNet/conv2/b/Adam_1/readIdentity train_cnn/ConvNet/conv2/b/Adam_1*"
_class
loc:@ConvNet/conv2/b*
T0*
_output_shapes
:@
j
train_cnn/zeros_8Const*
dtype0*
valueB
А А*    * 
_output_shapes
:
А А
┤
train_cnn/ConvNet/fc1/W/AdamVariable*
	container * 
_output_shapes
:
А А*
dtype0*
shape:
А А* 
_class
loc:@ConvNet/fc1/W*
shared_name 
╘
#train_cnn/ConvNet/fc1/W/Adam/AssignAssigntrain_cnn/ConvNet/fc1/W/Adamtrain_cnn/zeros_8*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
Ш
!train_cnn/ConvNet/fc1/W/Adam/readIdentitytrain_cnn/ConvNet/fc1/W/Adam* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
А А
j
train_cnn/zeros_9Const*
dtype0*
valueB
А А*    * 
_output_shapes
:
А А
╢
train_cnn/ConvNet/fc1/W/Adam_1Variable*
	container * 
_output_shapes
:
А А*
dtype0*
shape:
А А* 
_class
loc:@ConvNet/fc1/W*
shared_name 
╪
%train_cnn/ConvNet/fc1/W/Adam_1/AssignAssigntrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/zeros_9*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
Ь
#train_cnn/ConvNet/fc1/W/Adam_1/readIdentitytrain_cnn/ConvNet/fc1/W/Adam_1* 
_class
loc:@ConvNet/fc1/W*
T0* 
_output_shapes
:
А А
a
train_cnn/zeros_10Const*
dtype0*
valueBА*    *
_output_shapes	
:А
к
train_cnn/ConvNet/fc1/b/AdamVariable*
	container *
_output_shapes	
:А*
dtype0*
shape:А* 
_class
loc:@ConvNet/fc1/b*
shared_name 
╨
#train_cnn/ConvNet/fc1/b/Adam/AssignAssigntrain_cnn/ConvNet/fc1/b/Adamtrain_cnn/zeros_10*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
У
!train_cnn/ConvNet/fc1/b/Adam/readIdentitytrain_cnn/ConvNet/fc1/b/Adam* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:А
a
train_cnn/zeros_11Const*
dtype0*
valueBА*    *
_output_shapes	
:А
м
train_cnn/ConvNet/fc1/b/Adam_1Variable*
	container *
_output_shapes	
:А*
dtype0*
shape:А* 
_class
loc:@ConvNet/fc1/b*
shared_name 
╘
%train_cnn/ConvNet/fc1/b/Adam_1/AssignAssigntrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/zeros_11*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
Ч
#train_cnn/ConvNet/fc1/b/Adam_1/readIdentitytrain_cnn/ConvNet/fc1/b/Adam_1* 
_class
loc:@ConvNet/fc1/b*
T0*
_output_shapes	
:А
k
train_cnn/zeros_12Const*
dtype0*
valueB
А└*    * 
_output_shapes
:
А└
┤
train_cnn/ConvNet/fc2/W/AdamVariable*
	container * 
_output_shapes
:
А└*
dtype0*
shape:
А└* 
_class
loc:@ConvNet/fc2/W*
shared_name 
╒
#train_cnn/ConvNet/fc2/W/Adam/AssignAssigntrain_cnn/ConvNet/fc2/W/Adamtrain_cnn/zeros_12*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
Ш
!train_cnn/ConvNet/fc2/W/Adam/readIdentitytrain_cnn/ConvNet/fc2/W/Adam* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
А└
k
train_cnn/zeros_13Const*
dtype0*
valueB
А└*    * 
_output_shapes
:
А└
╢
train_cnn/ConvNet/fc2/W/Adam_1Variable*
	container * 
_output_shapes
:
А└*
dtype0*
shape:
А└* 
_class
loc:@ConvNet/fc2/W*
shared_name 
┘
%train_cnn/ConvNet/fc2/W/Adam_1/AssignAssigntrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/zeros_13*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
Ь
#train_cnn/ConvNet/fc2/W/Adam_1/readIdentitytrain_cnn/ConvNet/fc2/W/Adam_1* 
_class
loc:@ConvNet/fc2/W*
T0* 
_output_shapes
:
А└
a
train_cnn/zeros_14Const*
dtype0*
valueB└*    *
_output_shapes	
:└
к
train_cnn/ConvNet/fc2/b/AdamVariable*
	container *
_output_shapes	
:└*
dtype0*
shape:└* 
_class
loc:@ConvNet/fc2/b*
shared_name 
╨
#train_cnn/ConvNet/fc2/b/Adam/AssignAssigntrain_cnn/ConvNet/fc2/b/Adamtrain_cnn/zeros_14*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
У
!train_cnn/ConvNet/fc2/b/Adam/readIdentitytrain_cnn/ConvNet/fc2/b/Adam* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:└
a
train_cnn/zeros_15Const*
dtype0*
valueB└*    *
_output_shapes	
:└
м
train_cnn/ConvNet/fc2/b/Adam_1Variable*
	container *
_output_shapes	
:└*
dtype0*
shape:└* 
_class
loc:@ConvNet/fc2/b*
shared_name 
╘
%train_cnn/ConvNet/fc2/b/Adam_1/AssignAssigntrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/zeros_15*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
Ч
#train_cnn/ConvNet/fc2/b/Adam_1/readIdentitytrain_cnn/ConvNet/fc2/b/Adam_1* 
_class
loc:@ConvNet/fc2/b*
T0*
_output_shapes	
:└
i
train_cnn/zeros_16Const*
dtype0*
valueB	└
*    *
_output_shapes
:	└

▓
train_cnn/ConvNet/fc3/W/AdamVariable*
	container *
_output_shapes
:	└
*
dtype0*
shape:	└
* 
_class
loc:@ConvNet/fc3/W*
shared_name 
╘
#train_cnn/ConvNet/fc3/W/Adam/AssignAssigntrain_cnn/ConvNet/fc3/W/Adamtrain_cnn/zeros_16*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

Ч
!train_cnn/ConvNet/fc3/W/Adam/readIdentitytrain_cnn/ConvNet/fc3/W/Adam* 
_class
loc:@ConvNet/fc3/W*
T0*
_output_shapes
:	└

i
train_cnn/zeros_17Const*
dtype0*
valueB	└
*    *
_output_shapes
:	└

┤
train_cnn/ConvNet/fc3/W/Adam_1Variable*
	container *
_output_shapes
:	└
*
dtype0*
shape:	└
* 
_class
loc:@ConvNet/fc3/W*
shared_name 
╪
%train_cnn/ConvNet/fc3/W/Adam_1/AssignAssigntrain_cnn/ConvNet/fc3/W/Adam_1train_cnn/zeros_17*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

Ы
#train_cnn/ConvNet/fc3/W/Adam_1/readIdentitytrain_cnn/ConvNet/fc3/W/Adam_1* 
_class
loc:@ConvNet/fc3/W*
T0*
_output_shapes
:	└

_
train_cnn/zeros_18Const*
dtype0*
valueB
*    *
_output_shapes
:

и
train_cnn/ConvNet/fc3/b/AdamVariable*
	container *
_output_shapes
:
*
dtype0*
shape:
* 
_class
loc:@ConvNet/fc3/b*
shared_name 
╧
#train_cnn/ConvNet/fc3/b/Adam/AssignAssigntrain_cnn/ConvNet/fc3/b/Adamtrain_cnn/zeros_18*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

Т
!train_cnn/ConvNet/fc3/b/Adam/readIdentitytrain_cnn/ConvNet/fc3/b/Adam* 
_class
loc:@ConvNet/fc3/b*
T0*
_output_shapes
:

_
train_cnn/zeros_19Const*
dtype0*
valueB
*    *
_output_shapes
:

к
train_cnn/ConvNet/fc3/b/Adam_1Variable*
	container *
_output_shapes
:
*
dtype0*
shape:
* 
_class
loc:@ConvNet/fc3/b*
shared_name 
╙
%train_cnn/ConvNet/fc3/b/Adam_1/AssignAssigntrain_cnn/ConvNet/fc3/b/Adam_1train_cnn/zeros_19*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

Ц
#train_cnn/ConvNet/fc3/b/Adam_1/readIdentitytrain_cnn/ConvNet/fc3/b/Adam_1* 
_class
loc:@ConvNet/fc3/b*
T0*
_output_shapes
:

a
train_cnn/Adam/learning_rateConst*
dtype0*
valueB
 *╖╤8*
_output_shapes
: 
Y
train_cnn/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
Y
train_cnn/Adam/beta2Const*
dtype0*
valueB
 *w╛?*
_output_shapes
: 
[
train_cnn/Adam/epsilonConst*
dtype0*
valueB
 *w╠+2*
_output_shapes
: 
х
/train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam	ApplyAdamConvNet/conv1/Wtrain_cnn/ConvNet/conv1/W/Adam train_cnn/ConvNet/conv1/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonRtrain_cnn/gradients/train_cnn/ConvNet/conv1/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*&
_output_shapes
:@
╓
/train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam	ApplyAdamConvNet/conv1/btrain_cnn/ConvNet/conv1/b/Adam train_cnn/ConvNet/conv1/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonOtrain_cnn/gradients/train_cnn/ConvNet/conv1/add_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv1/b*
use_locking( *
T0*
_output_shapes
:@
х
/train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam	ApplyAdamConvNet/conv2/Wtrain_cnn/ConvNet/conv2/W/Adam train_cnn/ConvNet/conv2/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonRtrain_cnn/gradients/train_cnn/ConvNet/conv2/Conv2D_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv2/W*
use_locking( *
T0*&
_output_shapes
:@@
╓
/train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam	ApplyAdamConvNet/conv2/btrain_cnn/ConvNet/conv2/b/Adam train_cnn/ConvNet/conv2/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonOtrain_cnn/gradients/train_cnn/ConvNet/conv2/add_grad/tuple/control_dependency_1*"
_class
loc:@ConvNet/conv2/b*
use_locking( *
T0*
_output_shapes
:@
╙
-train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam	ApplyAdamConvNet/fc1/Wtrain_cnn/ConvNet/fc1/W/Adamtrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/fc1/MatMul_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc1/W*
use_locking( *
T0* 
_output_shapes
:
А А
╦
-train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam	ApplyAdamConvNet/fc1/btrain_cnn/ConvNet/fc1/b/Adamtrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonMtrain_cnn/gradients/train_cnn/ConvNet/fc1/add_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc1/b*
use_locking( *
T0*
_output_shapes	
:А
╙
-train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam	ApplyAdamConvNet/fc2/Wtrain_cnn/ConvNet/fc2/W/Adamtrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/fc2/MatMul_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc2/W*
use_locking( *
T0* 
_output_shapes
:
А└
╦
-train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam	ApplyAdamConvNet/fc2/btrain_cnn/ConvNet/fc2/b/Adamtrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonMtrain_cnn/gradients/train_cnn/ConvNet/fc2/add_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc2/b*
use_locking( *
T0*
_output_shapes	
:└
╥
-train_cnn/Adam/update_ConvNet/fc3/W/ApplyAdam	ApplyAdamConvNet/fc3/Wtrain_cnn/ConvNet/fc3/W/Adamtrain_cnn/ConvNet/fc3/W/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonPtrain_cnn/gradients/train_cnn/ConvNet/fc3/MatMul_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc3/W*
use_locking( *
T0*
_output_shapes
:	└

╩
-train_cnn/Adam/update_ConvNet/fc3/b/ApplyAdam	ApplyAdamConvNet/fc3/btrain_cnn/ConvNet/fc3/b/Adamtrain_cnn/ConvNet/fc3/b/Adam_1train_cnn/beta1_power/readtrain_cnn/beta2_power/readtrain_cnn/Adam/learning_ratetrain_cnn/Adam/beta1train_cnn/Adam/beta2train_cnn/Adam/epsilonMtrain_cnn/gradients/train_cnn/ConvNet/fc3/add_grad/tuple/control_dependency_1* 
_class
loc:@ConvNet/fc3/b*
use_locking( *
T0*
_output_shapes
:

°
train_cnn/Adam/mulMultrain_cnn/beta1_power/readtrain_cnn/Adam/beta10^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/b/ApplyAdam*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
╕
train_cnn/Adam/AssignAssigntrain_cnn/beta1_powertrain_cnn/Adam/mul*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*
_output_shapes
: 
·
train_cnn/Adam/mul_1Multrain_cnn/beta2_power/readtrain_cnn/Adam/beta20^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/b/ApplyAdam*"
_class
loc:@ConvNet/conv1/W*
T0*
_output_shapes
: 
╝
train_cnn/Adam/Assign_1Assigntrain_cnn/beta2_powertrain_cnn/Adam/mul_1*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking( *
T0*
_output_shapes
: 
░
train_cnn/AdamNoOp0^train_cnn/Adam/update_ConvNet/conv1/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv1/b/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/W/ApplyAdam0^train_cnn/Adam/update_ConvNet/conv2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc1/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc2/b/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/W/ApplyAdam.^train_cnn/Adam/update_ConvNet/fc3/b/ApplyAdam^train_cnn/Adam/Assign^train_cnn/Adam/Assign_1
P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 
и
save/save/tensor_namesConst*
dtype0*▌
value╙B╨ BConvNet/conv1/WBConvNet/conv1/bBConvNet/conv2/WBConvNet/conv2/bBConvNet/fc1/WBConvNet/fc1/bBConvNet/fc2/WBConvNet/fc2/bBConvNet/fc3/WBConvNet/fc3/bBtrain_cnn/ConvNet/conv1/W/AdamB train_cnn/ConvNet/conv1/W/Adam_1Btrain_cnn/ConvNet/conv1/b/AdamB train_cnn/ConvNet/conv1/b/Adam_1Btrain_cnn/ConvNet/conv2/W/AdamB train_cnn/ConvNet/conv2/W/Adam_1Btrain_cnn/ConvNet/conv2/b/AdamB train_cnn/ConvNet/conv2/b/Adam_1Btrain_cnn/ConvNet/fc1/W/AdamBtrain_cnn/ConvNet/fc1/W/Adam_1Btrain_cnn/ConvNet/fc1/b/AdamBtrain_cnn/ConvNet/fc1/b/Adam_1Btrain_cnn/ConvNet/fc2/W/AdamBtrain_cnn/ConvNet/fc2/W/Adam_1Btrain_cnn/ConvNet/fc2/b/AdamBtrain_cnn/ConvNet/fc2/b/Adam_1Btrain_cnn/ConvNet/fc3/W/AdamBtrain_cnn/ConvNet/fc3/W/Adam_1Btrain_cnn/ConvNet/fc3/b/AdamBtrain_cnn/ConvNet/fc3/b/Adam_1Btrain_cnn/beta1_powerBtrain_cnn/beta2_power*
_output_shapes
: 
в
save/save/shapes_and_slicesConst*
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
: 
╦
	save/save
SaveSlices
save/Constsave/save/tensor_namessave/save/shapes_and_slicesConvNet/conv1/WConvNet/conv1/bConvNet/conv2/WConvNet/conv2/bConvNet/fc1/WConvNet/fc1/bConvNet/fc2/WConvNet/fc2/bConvNet/fc3/WConvNet/fc3/btrain_cnn/ConvNet/conv1/W/Adam train_cnn/ConvNet/conv1/W/Adam_1train_cnn/ConvNet/conv1/b/Adam train_cnn/ConvNet/conv1/b/Adam_1train_cnn/ConvNet/conv2/W/Adam train_cnn/ConvNet/conv2/W/Adam_1train_cnn/ConvNet/conv2/b/Adam train_cnn/ConvNet/conv2/b/Adam_1train_cnn/ConvNet/fc1/W/Adamtrain_cnn/ConvNet/fc1/W/Adam_1train_cnn/ConvNet/fc1/b/Adamtrain_cnn/ConvNet/fc1/b/Adam_1train_cnn/ConvNet/fc2/W/Adamtrain_cnn/ConvNet/fc2/W/Adam_1train_cnn/ConvNet/fc2/b/Adamtrain_cnn/ConvNet/fc2/b/Adam_1train_cnn/ConvNet/fc3/W/Adamtrain_cnn/ConvNet/fc3/W/Adam_1train_cnn/ConvNet/fc3/b/Adamtrain_cnn/ConvNet/fc3/b/Adam_1train_cnn/beta1_powertrain_cnn/beta2_power*)
T$
"2 
{
save/control_dependencyIdentity
save/Const
^save/save*
_class
loc:@save/Const*
T0*
_output_shapes
: 
n
save/restore_slice/tensor_nameConst*
dtype0* 
valueB BConvNet/conv1/W*
_output_shapes
: 
c
"save/restore_slice/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╢
save/restore_sliceRestoreSlice
save/Constsave/restore_slice/tensor_name"save/restore_slice/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╕
save/AssignAssignConvNet/conv1/Wsave/restore_slice*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
p
 save/restore_slice_1/tensor_nameConst*
dtype0* 
valueB BConvNet/conv1/b*
_output_shapes
: 
e
$save/restore_slice_1/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_1RestoreSlice
save/Const save/restore_slice_1/tensor_name$save/restore_slice_1/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
░
save/Assign_1AssignConvNet/conv1/bsave/restore_slice_1*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
p
 save/restore_slice_2/tensor_nameConst*
dtype0* 
valueB BConvNet/conv2/W*
_output_shapes
: 
e
$save/restore_slice_2/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_2RestoreSlice
save/Const save/restore_slice_2/tensor_name$save/restore_slice_2/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╝
save/Assign_2AssignConvNet/conv2/Wsave/restore_slice_2*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
p
 save/restore_slice_3/tensor_nameConst*
dtype0* 
valueB BConvNet/conv2/b*
_output_shapes
: 
e
$save/restore_slice_3/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_3RestoreSlice
save/Const save/restore_slice_3/tensor_name$save/restore_slice_3/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
░
save/Assign_3AssignConvNet/conv2/bsave/restore_slice_3*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
n
 save/restore_slice_4/tensor_nameConst*
dtype0*
valueB BConvNet/fc1/W*
_output_shapes
: 
e
$save/restore_slice_4/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_4RestoreSlice
save/Const save/restore_slice_4/tensor_name$save/restore_slice_4/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
▓
save/Assign_4AssignConvNet/fc1/Wsave/restore_slice_4*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
n
 save/restore_slice_5/tensor_nameConst*
dtype0*
valueB BConvNet/fc1/b*
_output_shapes
: 
e
$save/restore_slice_5/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_5RestoreSlice
save/Const save/restore_slice_5/tensor_name$save/restore_slice_5/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
н
save/Assign_5AssignConvNet/fc1/bsave/restore_slice_5*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
n
 save/restore_slice_6/tensor_nameConst*
dtype0*
valueB BConvNet/fc2/W*
_output_shapes
: 
e
$save/restore_slice_6/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_6RestoreSlice
save/Const save/restore_slice_6/tensor_name$save/restore_slice_6/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
▓
save/Assign_6AssignConvNet/fc2/Wsave/restore_slice_6*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
n
 save/restore_slice_7/tensor_nameConst*
dtype0*
valueB BConvNet/fc2/b*
_output_shapes
: 
e
$save/restore_slice_7/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_7RestoreSlice
save/Const save/restore_slice_7/tensor_name$save/restore_slice_7/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
н
save/Assign_7AssignConvNet/fc2/bsave/restore_slice_7*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
n
 save/restore_slice_8/tensor_nameConst*
dtype0*
valueB BConvNet/fc3/W*
_output_shapes
: 
e
$save/restore_slice_8/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_8RestoreSlice
save/Const save/restore_slice_8/tensor_name$save/restore_slice_8/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
▒
save/Assign_8AssignConvNet/fc3/Wsave/restore_slice_8*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

n
 save/restore_slice_9/tensor_nameConst*
dtype0*
valueB BConvNet/fc3/b*
_output_shapes
: 
e
$save/restore_slice_9/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
╝
save/restore_slice_9RestoreSlice
save/Const save/restore_slice_9/tensor_name$save/restore_slice_9/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
м
save/Assign_9AssignConvNet/fc3/bsave/restore_slice_9*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

А
!save/restore_slice_10/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/conv1/W/Adam*
_output_shapes
: 
f
%save/restore_slice_10/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_10RestoreSlice
save/Const!save/restore_slice_10/tensor_name%save/restore_slice_10/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
═
save/Assign_10Assigntrain_cnn/ConvNet/conv1/W/Adamsave/restore_slice_10*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
В
!save/restore_slice_11/tensor_nameConst*
dtype0*1
value(B& B train_cnn/ConvNet/conv1/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_11/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_11RestoreSlice
save/Const!save/restore_slice_11/tensor_name%save/restore_slice_11/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╧
save/Assign_11Assign train_cnn/ConvNet/conv1/W/Adam_1save/restore_slice_11*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*&
_output_shapes
:@
А
!save/restore_slice_12/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/conv1/b/Adam*
_output_shapes
: 
f
%save/restore_slice_12/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_12RestoreSlice
save/Const!save/restore_slice_12/tensor_name%save/restore_slice_12/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┴
save/Assign_12Assigntrain_cnn/ConvNet/conv1/b/Adamsave/restore_slice_12*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
В
!save/restore_slice_13/tensor_nameConst*
dtype0*1
value(B& B train_cnn/ConvNet/conv1/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_13/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_13RestoreSlice
save/Const!save/restore_slice_13/tensor_name%save/restore_slice_13/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
├
save/Assign_13Assign train_cnn/ConvNet/conv1/b/Adam_1save/restore_slice_13*
validate_shape(*"
_class
loc:@ConvNet/conv1/b*
use_locking(*
T0*
_output_shapes
:@
А
!save/restore_slice_14/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/conv2/W/Adam*
_output_shapes
: 
f
%save/restore_slice_14/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_14RestoreSlice
save/Const!save/restore_slice_14/tensor_name%save/restore_slice_14/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
═
save/Assign_14Assigntrain_cnn/ConvNet/conv2/W/Adamsave/restore_slice_14*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
В
!save/restore_slice_15/tensor_nameConst*
dtype0*1
value(B& B train_cnn/ConvNet/conv2/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_15/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_15RestoreSlice
save/Const!save/restore_slice_15/tensor_name%save/restore_slice_15/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╧
save/Assign_15Assign train_cnn/ConvNet/conv2/W/Adam_1save/restore_slice_15*
validate_shape(*"
_class
loc:@ConvNet/conv2/W*
use_locking(*
T0*&
_output_shapes
:@@
А
!save/restore_slice_16/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/conv2/b/Adam*
_output_shapes
: 
f
%save/restore_slice_16/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_16RestoreSlice
save/Const!save/restore_slice_16/tensor_name%save/restore_slice_16/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┴
save/Assign_16Assigntrain_cnn/ConvNet/conv2/b/Adamsave/restore_slice_16*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
В
!save/restore_slice_17/tensor_nameConst*
dtype0*1
value(B& B train_cnn/ConvNet/conv2/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_17/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_17RestoreSlice
save/Const!save/restore_slice_17/tensor_name%save/restore_slice_17/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
├
save/Assign_17Assign train_cnn/ConvNet/conv2/b/Adam_1save/restore_slice_17*
validate_shape(*"
_class
loc:@ConvNet/conv2/b*
use_locking(*
T0*
_output_shapes
:@
~
!save/restore_slice_18/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc1/W/Adam*
_output_shapes
: 
f
%save/restore_slice_18/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_18RestoreSlice
save/Const!save/restore_slice_18/tensor_name%save/restore_slice_18/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
├
save/Assign_18Assigntrain_cnn/ConvNet/fc1/W/Adamsave/restore_slice_18*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
А
!save/restore_slice_19/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc1/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_19/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_19RestoreSlice
save/Const!save/restore_slice_19/tensor_name%save/restore_slice_19/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┼
save/Assign_19Assigntrain_cnn/ConvNet/fc1/W/Adam_1save/restore_slice_19*
validate_shape(* 
_class
loc:@ConvNet/fc1/W*
use_locking(*
T0* 
_output_shapes
:
А А
~
!save/restore_slice_20/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc1/b/Adam*
_output_shapes
: 
f
%save/restore_slice_20/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_20RestoreSlice
save/Const!save/restore_slice_20/tensor_name%save/restore_slice_20/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╛
save/Assign_20Assigntrain_cnn/ConvNet/fc1/b/Adamsave/restore_slice_20*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
А
!save/restore_slice_21/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc1/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_21/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_21RestoreSlice
save/Const!save/restore_slice_21/tensor_name%save/restore_slice_21/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
└
save/Assign_21Assigntrain_cnn/ConvNet/fc1/b/Adam_1save/restore_slice_21*
validate_shape(* 
_class
loc:@ConvNet/fc1/b*
use_locking(*
T0*
_output_shapes	
:А
~
!save/restore_slice_22/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc2/W/Adam*
_output_shapes
: 
f
%save/restore_slice_22/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_22RestoreSlice
save/Const!save/restore_slice_22/tensor_name%save/restore_slice_22/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
├
save/Assign_22Assigntrain_cnn/ConvNet/fc2/W/Adamsave/restore_slice_22*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
А
!save/restore_slice_23/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc2/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_23/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_23RestoreSlice
save/Const!save/restore_slice_23/tensor_name%save/restore_slice_23/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┼
save/Assign_23Assigntrain_cnn/ConvNet/fc2/W/Adam_1save/restore_slice_23*
validate_shape(* 
_class
loc:@ConvNet/fc2/W*
use_locking(*
T0* 
_output_shapes
:
А└
~
!save/restore_slice_24/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc2/b/Adam*
_output_shapes
: 
f
%save/restore_slice_24/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_24RestoreSlice
save/Const!save/restore_slice_24/tensor_name%save/restore_slice_24/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╛
save/Assign_24Assigntrain_cnn/ConvNet/fc2/b/Adamsave/restore_slice_24*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
А
!save/restore_slice_25/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc2/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_25/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_25RestoreSlice
save/Const!save/restore_slice_25/tensor_name%save/restore_slice_25/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
└
save/Assign_25Assigntrain_cnn/ConvNet/fc2/b/Adam_1save/restore_slice_25*
validate_shape(* 
_class
loc:@ConvNet/fc2/b*
use_locking(*
T0*
_output_shapes	
:└
~
!save/restore_slice_26/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc3/W/Adam*
_output_shapes
: 
f
%save/restore_slice_26/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_26RestoreSlice
save/Const!save/restore_slice_26/tensor_name%save/restore_slice_26/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┬
save/Assign_26Assigntrain_cnn/ConvNet/fc3/W/Adamsave/restore_slice_26*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

А
!save/restore_slice_27/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc3/W/Adam_1*
_output_shapes
: 
f
%save/restore_slice_27/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_27RestoreSlice
save/Const!save/restore_slice_27/tensor_name%save/restore_slice_27/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
─
save/Assign_27Assigntrain_cnn/ConvNet/fc3/W/Adam_1save/restore_slice_27*
validate_shape(* 
_class
loc:@ConvNet/fc3/W*
use_locking(*
T0*
_output_shapes
:	└

~
!save/restore_slice_28/tensor_nameConst*
dtype0*-
value$B" Btrain_cnn/ConvNet/fc3/b/Adam*
_output_shapes
: 
f
%save/restore_slice_28/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_28RestoreSlice
save/Const!save/restore_slice_28/tensor_name%save/restore_slice_28/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
╜
save/Assign_28Assigntrain_cnn/ConvNet/fc3/b/Adamsave/restore_slice_28*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

А
!save/restore_slice_29/tensor_nameConst*
dtype0*/
value&B$ Btrain_cnn/ConvNet/fc3/b/Adam_1*
_output_shapes
: 
f
%save/restore_slice_29/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_29RestoreSlice
save/Const!save/restore_slice_29/tensor_name%save/restore_slice_29/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┐
save/Assign_29Assigntrain_cnn/ConvNet/fc3/b/Adam_1save/restore_slice_29*
validate_shape(* 
_class
loc:@ConvNet/fc3/b*
use_locking(*
T0*
_output_shapes
:

w
!save/restore_slice_30/tensor_nameConst*
dtype0*&
valueB Btrain_cnn/beta1_power*
_output_shapes
: 
f
%save/restore_slice_30/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_30RestoreSlice
save/Const!save/restore_slice_30/tensor_name%save/restore_slice_30/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┤
save/Assign_30Assigntrain_cnn/beta1_powersave/restore_slice_30*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
w
!save/restore_slice_31/tensor_nameConst*
dtype0*&
valueB Btrain_cnn/beta2_power*
_output_shapes
: 
f
%save/restore_slice_31/shape_and_sliceConst*
dtype0*
valueB B *
_output_shapes
: 
┐
save/restore_slice_31RestoreSlice
save/Const!save/restore_slice_31/tensor_name%save/restore_slice_31/shape_and_slice*
preferred_shard         *
dt0*
_output_shapes
:
┤
save/Assign_31Assigntrain_cnn/beta2_powersave/restore_slice_31*
validate_shape(*"
_class
loc:@ConvNet/conv1/W*
use_locking(*
T0*
_output_shapes
: 
м
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31"ЭХN